---
date: 2026-04-04
question_id: Q002
topics: ["ray", "placement-group", "resource-allocation", "gpu-scheduling"]
related_files:
  - slime/ray/placement_group.py
  - slime/ray/actor_group.py
  - slime/ray/rollout.py
---

# Question

create_placement_groups 是在做什么？分配是按照什么规则分配的？

# Answer

## 1. create_placement_groups 的作用

`create_placement_groups` 是 Slime 中**GPU 资源分配的核心函数**，它创建 Ray 的 Placement Group 来管理训练（Actor/Critic）和推理（Rollout）所需的 GPU 资源。

### 1.1 核心职责

1. **计算 GPU 需求**：根据运行模式计算总共需要多少 GPU
2. **创建 Placement Group**：使用 Ray 的 placement_group API 分配资源
3. **GPU 排序与映射**：获取 GPU ID 并排序，建立逻辑索引到物理 GPU 的映射
4. **资源分段**：将 GPU 划分为 Actor、Critic（可选）、Rollout 三部分

---

## 2. 分配规则详解

### 2.1 GPU 数量计算逻辑

**文件**: `slime/ray/placement_group.py:79-119`

```python
def create_placement_groups(args):
    num_gpus = 0
    
    # 模式 1: 仅调试训练（不启动 SGLang）
    if args.debug_train_only:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0  # 无 rollout
        if args.use_critic:
            num_gpus += args.critic_num_nodes * args.critic_num_gpus_per_node
    
    # 模式 2: 仅调试 rollout（不训练）
    elif args.debug_rollout_only:
        num_gpus = args.rollout_num_gpus
        rollout_offset = 0
    
    # 模式 3: Colocate 模式（训练+推理共用 GPU）
    elif args.colocate:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        rollout_offset = 0  # 共用，所以 offset=0
        if args.use_critic:
            num_gpus += args.critic_num_nodes * args.critic_num_gpus_per_node
    
    # 模式 4: 分离模式（训练 GPU + 推理 GPU 分开）
    else:
        num_gpus = (args.actor_num_nodes * args.actor_num_gpus_per_node + 
                   args.rollout_num_gpus)
        rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
        if args.use_critic:
            num_gpus += args.critic_num_nodes * args.critic_num_gpus_per_node
            rollout_offset += args.critic_num_nodes * args.critic_num_gpus_per_node
```

### 2.2 Placement Group Bundle 创建

**文件**: `slime/ray/placement_group.py:41-76`

```python
def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    # 每个 GPU 创建一个 bundle
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    
    # 使用 PACK 策略（紧凑放置）
    pg = placement_group(bundles, strategy="PACK")
    
    # 等待 placement group 就绪
    ray.get(pg.ready())
```

**Bundle 策略说明**:
- 每个 bundle 请求 1 GPU + 1 CPU
- `strategy="PACK"`：紧凑放置，优先填满一个节点再分配到下一个
- 这样可以让同一节点的 GPU 尽可能连续

### 2.3 GPU 排序规则

**文件**: `slime/ray/placement_group.py:14-38`

```python
@ray.remote(num_gpus=1)
class InfoActor:
    """用于获取 IP 和 GPU ID 的辅助 Actor"""
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]

def sort_key(x):
    """排序规则：先按 IP 排序，再按 GPU ID 排序"""
    index, node_identifier, gpu_id = x
    
    # 尝试将 node_identifier 解析为 IP 地址
    try:
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # 如果不是 IP，尝试解析 hostname
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # 失败则使用 ASCII 值
            node_ip_parts = [ord(c) for c in node_identifier]
    
    return (node_ip_parts, gpu_id)
```

**排序流程**:
1. 为每个 bundle 创建一个 `InfoActor`
2. 每个 actor 获取所在节点的 IP 和其分配的 GPU ID
3. 收集所有 (index, ip, gpu_id) 信息
4. **按 (IP, GPU_ID) 排序**：先按 IP 地址排序，同一节点内按 GPU ID 排序
5. 生成重新排序后的 bundle 索引映射

**为什么要排序**？
- 保证多机环境下资源分配的一致性
- 让相同节点的 GPU 在逻辑上连续
- 便于后续的 DP/TP/PP 并行划分

### 2.4 资源分段映射

**文件**: `slime/ray/placement_group.py:109-119`

```python
# 根据运行模式划分 GPU 归属
rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]

if args.use_critic:
    critic_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[critic_offset:]
    critic_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[critic_offset:]

return {
    "actor": (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
    "critic": (pg, critic_pg_reordered_bundle_indices, critic_pg_reordered_gpu_ids) if args.use_critic else None,
    "rollout": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
}
```

---

## 3. 不同运行模式的分配示例

### 3.1 Colocate 模式（--colocate）

**场景**: 4 GPU，Actor 训练 + Rollout 推理共用

```
物理 GPU:        [GPU0] [GPU1] [GPU2] [GPU3]
                 ↓      ↓      ↓      ↓
Placement Group: [B0]   [B1]   [B2]   [B3]  (bundles)
                 ↓      ↓      ↓      ↓
逻辑分配:         Actor  Actor  Actor  Actor  
                 +SGL   +SGL   +SGL   +SGL   (推理引擎也在这4个GPU上)
```

代码中：
- `num_gpus = 4`
- `rollout_offset = 0`（共用，不偏移）
- actor 和 rollout 使用相同的 indices 和 gpu_ids

### 3.2 分离模式（非 colocate）

**场景**: 8 GPU，4个给训练，4个给推理

```
物理 GPU:        [GPU0] [GPU1] [GPU2] [GPU3] [GPU4] [GPU5] [GPU6] [GPU7]
                 ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓
Placement Group: [B0]   [B1]   [B2]   [B3]   [B4]   [B5]   [B6]   [B7]
                 ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓
逻辑分配:         Actor  Actor  Actor  Actor  Rollout Rollout Rollout Rollout
```

代码中：
- `num_gpus = 8`
- `rollout_offset = 4`（前4个给 actor，后4个给 rollout）
- actor: indices [0,1,2,3], rollout: indices [4,5,6,7]

### 3.3 带 Critic 的 PPO 模式

**场景**: 8 GPU，2个 Actor，2个 Critic，4个 Rollout

```
物理 GPU:        [GPU0] [GPU1] [GPU2] [GPU3] [GPU4] [GPU5] [GPU6] [GPU7]
                 ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓
Placement Group: [B0]   [B1]   [B2]   [B3]   [B4]   [B5]   [B6]   [B7]
                 ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓
逻辑分配:         Actor  Actor  Critic Critic Rollout Rollout Rollout Rollout
```

代码中：
- `critic_offset = 2`
- `rollout_offset = 4` (2 actor + 2 critic)

---

## 4. 如何使用这些资源

### 4.1 RayTrainGroup 使用 Placement Group

**文件**: `slime/ray/actor_group.py:83-99`

```python
def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
    pg, reordered_bundle_indices, reordered_gpu_ids = pg
    
    for rank in range(world_size):
        actor = TrainRayActor.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=reordered_bundle_indices[rank],
            ),
        ).remote(...)
```

**关键点**:
- `placement_group_bundle_index=reordered_bundle_indices[rank]`
- 使用排序后的索引，确保 actor 获得正确的物理 GPU

### 4.2 RolloutManager 使用 Placement Group

**文件**: `slime/ray/rollout.py:1044-1061`

```python
def _make_group(group_cfg, router_ip, router_port, overrides_extra=None):
    group = ServerGroup(
        args=args,
        pg=pg,  # 传入 placement group
        all_engines=[None] * num_engines,
        num_gpus_per_engine=gpus_per_engine,
        gpu_offset=gpu_offset,  # 在 PG 中的起始偏移
        ...
    )
```

---

## 5. 分配规则总结

| 规则 | 说明 |
|------|------|
| **计算规则** | 根据运行模式（debug_train/debug_rollout/colocate/分离）计算总 GPU 数 |
| **Bundle 规则** | 每个 GPU 一个 bundle，每个 bundle 包含 1 GPU + 1 CPU |
| **放置策略** | `PACK` 策略，紧凑放置 |
| **排序规则** | 按 (IP, GPU_ID) 排序，保证跨节点一致性 |
| **分段规则** | Actor → Critic（可选）→ Rollout 顺序划分 |
| **索引映射** | 建立逻辑 rank → bundle_index → 物理 GPU 的映射 |

## Key Points

1. **Placement Group 是 Ray 的资源抽象**，保证资源原子性分配
2. **排序是为了稳定性**，确保多机环境下每次运行资源分配一致
3. **Colocate 模式共用 GPU**，通过 offload/onload 切换训练/推理
4. **分离模式 GPU 独立**，训练和推理互不干扰
5. **返回的三元组** `(pg, indices, gpu_ids)` 被后续组件用于调度 actors

## Follow-up Questions

- [ ] Ray 的 Placement Group 深入原理？
- [ ] 多机环境下的资源发现和分配机制？
- [ ] 如何在 placement group 中实现故障恢复？
- [ ] GPU 拓扑感知调度（考虑 NVLink）？
