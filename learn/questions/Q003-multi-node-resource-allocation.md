---
date: 2026-04-04
question_id: Q003
topics: ["multi-node", "ray", "distributed-training", "resource-allocation"]
related_files:
  - slime/ray/placement_group.py
  - slime/ray/actor_group.py
  - slime/backends/megatron_utils/actor.py
---

# Question

如果是多机器，那么又会如何分配资源？

# Answer

## 1. 多机器架构概述

Slime 使用 **Ray 集群** 来管理多机器资源。Ray 会自动跨节点调度资源，Slime 在此基础上进行 GPU 分配和分布式训练协调。

### 1.1 核心参数

```python
# 训练节点配置
--actor-num-nodes 2          # Actor 使用 2 个节点
--actor-num-gpus-per-node 8  # 每节点 8 个 GPU
--critic-num-nodes 2         # Critic 使用 2 个节点 (PPO)
--critic-num-gpus-per-node 8 # 每节点 8 个 GPU

# 推理节点配置  
--rollout-num-gpus 8         # Rollout 总共使用 8 个 GPU（跨节点）
--num-gpus-per-node 8        # 每节点 GPU 数（用于计算 rollout 分布）
```

---

## 2. Ray 集群启动

### 2.1 Head 节点启动

**文件**: `examples/retool/retool_qwen3_4b_rl.sh:128-129`

```bash
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 在 head 节点启动 Ray
ray start --head \
    --node-ip-address ${MASTER_ADDR} \
    --num-gpus 8 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265
```

### 2.2 Worker 节点连接

在其他机器上执行：

```bash
ray start --address=<HEAD_IP>:6379 --num-gpus 8
```

Ray 会自动发现集群中的所有 GPU 资源。

---

## 3. Placement Group 跨节点分配

### 3.1 创建跨节点的 Placement Group

**文件**: `slime/ray/placement_group.py:41-76`

```python
def _create_placement_group(num_gpus):
    """创建跨节点的 Placement Group"""
    # 每个 GPU 一个 bundle
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    
    # 使用 PACK 策略：紧凑放置
    pg = placement_group(bundles, strategy="PACK")
    
    # 等待 placement group 在所有节点上就绪
    ray.get(pg.ready())
```

**关键点**:
- Ray 会自动将 bundles 分布到不同节点
- `PACK` 策略优先填满一个节点，再分配到下一个节点
- 例如 16 个 bundle（8+8 GPU），会先填满节点1的 8 个，再填节点2的 8 个

### 3.2 多机器 GPU 发现与排序

**文件**: `slime/ray/placement_group.py:14-76`

```python
@ray.remote(num_gpus=1)
class InfoActor:
    """在每个 bundle 上运行，获取所在节点的 IP 和 GPU ID"""
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]

def _create_placement_group(num_gpus):
    # ... 创建 PG ...
    
    # 为每个 bundle 创建 InfoActor 来获取位置信息
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    
    # 收集所有 (index, ip, gpu_id)
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    
    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    
    # 按 (IP, GPU_ID) 排序，保证跨节点一致性
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    
    # 生成映射：逻辑索引 -> 物理 bundle 索引 -> 物理 GPU
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]
```

**排序规则**:
```python
def sort_key(x):
    index, node_identifier, gpu_id = x
    # 解析 IP 地址为数字列表 [192, 168, 1, 10]
    node_ip_parts = list(map(int, ip_address.split(".")))
    return (node_ip_parts, gpu_id)
```

**示例**: 2 节点，每节点 4 GPU
```
节点1 (IP: 192.168.1.10): GPU [0, 1, 2, 3]
节点2 (IP: 192.168.1.11): GPU [0, 1, 2, 3]

排序前 (bundle_index, ip, gpu_id):
[(0, "192.168.1.10", 0), (1, "192.168.1.10", 1), 
 (2, "192.168.1.10", 2), (3, "192.168.1.10", 3),
 (4, "192.168.1.11", 0), (5, "192.168.1.11", 1),
 (6, "192.168.1.11", 2), (7, "192.168.1.11", 3)]

排序后（已经是有序的）:
逻辑索引 0 -> bundle 0 -> 节点1 GPU0
逻辑索引 1 -> bundle 1 -> 节点1 GPU1
逻辑索引 2 -> bundle 2 -> 节点1 GPU2
逻辑索引 3 -> bundle 3 -> 节点1 GPU3
逻辑索引 4 -> bundle 4 -> 节点2 GPU0
逻辑索引 5 -> bundle 5 -> 节点2 GPU1
...
```

---

## 4. 多机器资源分段

### 4.1 GPU 数量计算（多节点）

**文件**: `slime/ray/placement_group.py:79-119`

```python
def create_placement_groups(args):
    # 场景：2 节点训练，colocate 模式
    # actor_num_nodes=2, actor_num_gpus_per_node=8
    # 总共 16 个 GPU 跨 2 个节点
    
    if args.colocate:
        # colocate: 训练和推理共用 GPU
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node  # 2*8=16
        rollout_offset = 0  # 共用，offset=0
        
        if args.use_critic:
            # PPO: 加上 critic 的 GPU
            num_gpus += args.critic_num_nodes * args.critic_num_gpus_per_node
            critic_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
```

### 4.2 返回的多节点资源映射

```python
return {
    "actor": (pg, actor_indices, actor_gpu_ids),
    "critic": (pg, critic_indices, critic_gpu_ids),
    "rollout": (pg, rollout_indices, rollout_gpu_ids),
}

# 示例：16 GPU，colocate 模式
# actor_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# rollout_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] （共用）

# 示例：24 GPU，分离模式（16 actor + 8 rollout）
# actor_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# rollout_indices: [16, 17, 18, 19, 20, 21, 22, 23]
```

---

## 5. RayTrainGroup 跨节点初始化

### 5.1 创建分布式训练 Actor

**文件**: `slime/ray/actor_group.py:79-99`

```python
def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
    pg, reordered_bundle_indices, reordered_gpu_ids = pg
    world_size = self._num_nodes * self._num_gpus_per_node  # 总 GPU 数
    
    for rank in range(world_size):
        # 使用 PlacementGroupSchedulingStrategy 指定在哪个 bundle 上运行
        actor = TrainRayActor.options(
            num_cpus=num_gpus_per_actor,
            num_gpus=num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=reordered_bundle_indices[rank],
            ),
        ).remote(world_size, rank, master_addr, master_port)
```

**跨节点调度**:
- `reordered_bundle_indices[rank]` 指定了该 actor 应该运行在哪个 bundle 上
- 由于 bundle 已经分布在不同节点，actor 自然被调度到对应节点

### 5.2 分布式通信初始化

**文件**: `slime/ray/actor_group.py:86-99`

```python
for rank in range(world_size):
    actor = TrainRayActor.options(...).remote(world_size, rank, master_addr, master_port)
    
    if rank == 0:
        # Rank 0 获取 master 地址和端口，分发给其他节点
        master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
```

**文件**: `slime/backends/megatron_utils/actor.py:48-80`

```python
def init(self, args, role, with_ref=False, with_opd_teacher=False):
    # 初始化 Megatron 分布式环境
    init(args)
    
    # 获取并行配置
    self.train_parallel_config = {
        "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
    }
```

---

## 6. Rollout 跨节点部署

### 6.1 SGLang Engine 多节点配置

**文件**: `slime/ray/rollout.py:86-90`

```python
@property
def nodes_per_engine(self):
    """计算每个引擎需要多少节点"""
    # 例如：engine 需要 8 GPU，每节点 4 GPU，则需要 2 个节点
    return max(1, self.num_gpus_per_engine // self.args.num_gpus_per_node)

@property
def engines(self):
    """只返回每个 engine 在节点 0 上的 actor（多节点推理）"""
    return self.all_engines[:: self.nodes_per_engine]
```

### 6.2 多节点 Engine 启动

**文件**: `slime/ray/rollout.py:70-176`

```python
def start_engines(self, port_cursors: dict[int, int] | None = None):
    num_gpu_per_engine = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)
    
    for i in range(len(self.all_engines)):
        # 计算该 engine 在 placement group 中的 GPU 偏移
        gpu_index = self.gpu_offset + i * num_gpu_per_engine
        base_gpu_id = int(reordered_gpu_ids[gpu_index])
        
        # 创建 SGLang Engine Actor
        rollout_engine = RolloutRayActor.options(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            ),
        ).remote(...)
```

---

## 7. 多机器场景示例

### 7.1 场景：4 节点训练（Colocate）

```
集群：4 节点 x 8 GPU = 32 GPU
配置：
  --actor-num-nodes 4
  --actor-num-gpus-per-node 8
  --colocate

Placement Group Bundles:
[节点1] [节点1] [节点1] [节点1] [节点1] [节点1] [节点1] [节点1]
  B0      B1      B2      B3      B4      B5      B6      B7
  
[节点2] [节点2] [节点2] [节点2] [节点2] [节点2] [节点2] [节点2]
  B8      B9      B10     B11     B12     B13     B14     B15
  
[节点3] [节点3] [节点3] [节点3] [节点3] [节点3] [节点3] [节点3]
  B16     B17     B18     B19     B20     B21     B22     B23
  
[节点4] [节点4] [节点4] [节点4] [节点4] [节点4] [节点4] [节点4]
  B24     B25     B26     B27     B28     B29     B30     B31

资源分配（colocate，共用）：
- Actor:   B0-B31 (32 actors, 每 GPU 一个 actor)
- Rollout: B0-B31 (sglang engines，TP/DP 并行)
```

### 7.2 场景：2 节点训练 + 1 节点推理

```
集群：3 节点 x 8 GPU = 24 GPU
配置：
  --actor-num-nodes 2
  --actor-num-gpus-per-node 8
  --rollout-num-gpus 8

Placement Group Bundles:
[节点1] B0-B7   (Actor)
[节点2] B8-B15  (Actor)
[节点3] B16-B23 (Rollout)

资源分配（分离模式）：
- Actor:   B0-B15  (16 actors)
- Rollout: B16-B23 (sglang engines)
```

### 7.3 场景：PPO + 多节点 Critic

```
集群：4 节点 x 8 GPU = 32 GPU
配置：
  --actor-num-nodes 2
  --actor-num-gpus-per-node 8
  --critic-num-nodes 1
  --critic-num-gpus-per-node 8
  --rollout-num-gpus 8

Placement Group Bundles:
[节点1] B0-B7   (Actor)
[节点2] B8-B15  (Actor)
[节点3] B16-B23 (Critic)
[节点4] B24-B31 (Rollout)

资源分配：
- Actor:   B0-B15  (16 actors)
- Critic:  B16-B23 (8 critics)
- Rollout: B24-B31 (sglang engines)
```

---

## 8. 多机器通信

### 8.1 训练内部通信（Megatron）

```python
# 在 MegatronTrainRayActor 内部
# TP (Tensor Parallel): 同一节点内 GPU 间通信
# DP (Data Parallel): 跨节点梯度同步
# PP (Pipeline Parallel): 跨阶段数据传递

dp_size = mpu.get_data_parallel_world_size()
tp_size = mpu.get_tensor_model_parallel_world_size()
pp_size = mpu.get_pipeline_model_parallel_world_size()
```

### 8.2 跨组件通信（Ray）

```python
# Actor 与 RolloutManager 通信（跨节点）
rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

# 权重更新（跨节点）
actor_model.update_weights()  # 内部通过 Ray 调用
```

---

## Key Points

1. **Ray 负责跨节点调度**: 使用 Placement Group 跨多个节点分配资源
2. **自动发现**: Ray 会自动发现集群中所有节点的 GPU 资源
3. **IP+GPU 排序**: 多节点环境下按 IP 和 GPU ID 排序，保证分配一致性
4. **分布式通信**: 
   - 训练内部用 NCCL（Megatron TP/DP/PP）
   - 组件间用 Ray 远程调用
5. **灵活的节点配置**: 可以独立配置 actor/critic/rollout 的节点数

## Follow-up Questions

- [ ] Ray 集群的故障恢复机制？
- [ ] 多节点下的网络拓扑优化（NVLink/IB）？
- [ ] 多节点权重更新的带宽优化？
- [ ] 跨机房的分布式训练支持？
