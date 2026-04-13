---
date: 2026-04-12
question_id: Q013
topics: ["training", "ray", "megatron", "actor", "critic"]
related_files:
  - slime/ray/actor_group.py
---

# Question

讲解一下 `RayTrainGroup` 这个类是在干什么？

# Answer

## 一句话总结

`RayTrainGroup` 是 **Megatron 训练模型的 Ray Actor 组管理器**，负责创建和管理一组 Ray Actor 来运行 Megatron 分布式训练。

## 核心作用

```
┌─────────────────────────────────────────────────────────────┐
│                   RayTrainGroup                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Megatron    │  │ Megatron    │  │ Megatron    │  ...    │
│  │ Actor 0     │  │ Actor 1     │  │ Actor 2     │         │
│  │ (Rank 0)    │  │ (Rank 1)    │  │ (Rank 2)    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│       Master Addr/Port 协调                                 │
└─────────────────────────────────────────────────────────────┘
         ↓ 封装统一接口
   async_init() / async_train() / save_model() / update_weights()
```

## 初始化流程（`__init__`）

**文件**: `slime/ray/actor_group.py:29-44`

```python
def __init__(self, args, num_nodes, num_gpus_per_node, pg, num_gpus_per_actor=0.4, role="actor"):
    self.args = args
    self._num_nodes = num_nodes          # 如 2 节点
    self._num_gpus_per_node = num_gpus_per_node  # 如每节点 4 GPU
    self.role = role                     # "actor" 或 "critic"
    
    # 分配 GPU 并创建 Ray Actors
    self._allocate_gpus_for_actor(pg, num_gpus_per_actor)
```

**关键参数**：
- `num_gpus_per_actor=0.4`：类似 SGLang，Ray 只调度，Megatron 实际管理 GPU
- `pg`：Placement Group，确保 Actor 绑定到正确的 GPU bundle

## GPU 分配（`_allocate_gpus_for_actor`）

**文件**: `slime/ray/actor_group.py:46-99`

```python
def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
    world_size = self._num_nodes * self._num_gpus_per_node  # 总 GPU 数
    
    # 环境变量设置
    env_vars = {
        "NCCL_CUMEM_ENABLE": "0",
        **self.args.train_env_vars,
    }
    
    # 如果使用 offload_train，设置 torch_memory_saver
    if self.args.offload_train and self.args.train_backend == "megatron":
        env_vars["LD_PRELOAD"] = torch_memory_saver_hook
        env_vars["TMS_INIT_ENABLE"] = "1"
    
    # 创建 Ray Actor 类
    from slime.backends.megatron_utils.actor import MegatronTrainRayActor
    TrainRayActor = ray.remote(num_gpus=1, runtime_env={"env_vars": env_vars})(MegatronTrainRayActor)
    
    # 创建 world_size 个 Actor
    self._actor_handlers = []
    master_addr, master_port = None, None
    for rank in range(world_size):
        actor = TrainRayActor.options(
            num_cpus=num_gpus_per_actor,
            num_gpus=num_gpus_per_actor,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=reordered_bundle_indices[rank],
            ),
        ).remote(world_size, rank, master_addr, master_port)
        
        # Rank 0 提供 Master Addr/Port 给其他 rank
        if rank == 0:
            master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
        
        self._actor_handlers.append(actor)
```

**关键点**：
- 所有 Actor 共享同一个 Placement Group
- Rank 0 作为 coordinator，提供 master_addr/master_port
- 其他 rank 使用 Rank 0 的地址进行进程间通信

## 主要方法

| 方法 | 作用 | 返回 |
|------|------|------|
| `async_init()` | 初始化模型、优化器、检查点 | `list[ObjectRef]` |
| `async_train()` | 执行一次 rollout 的训练 | `list[ObjectRef]` |
| `save_model()` | 保存模型检查点 | 同步等待完成 |
| `update_weights()` | 广播权重到所有 rank（并同步到 SGLang） | 同步等待完成 |
| `onload()` | 从 CPU 恢复 GPU 内存 | 同步等待完成 |
| `offload()` | 将 GPU 内存 offload 到 CPU | 同步等待完成 |
| `clear_memory()` | 清理内存 | 同步等待完成 |
| `connect()` | 连接 Actor 和 Critic（用于 PPO） | 同步等待完成 |
| `set_rollout_manager()` | 关联 RolloutManager | 同步等待完成 |

## 方法实现示例

**文件**: `slime/ray/actor_group.py:111-141`

```python
def async_train(self, rollout_id, rollout_data_ref):
    """Do one rollout training"""
    return [
        actor.train.remote(rollout_id, rollout_data_ref) 
        for actor in self._actor_handlers
    ]

def update_weights(self):
    """Broadcast weights from rank 0 to all other ranks."""
    return ray.get([
        actor.update_weights.remote() 
        for actor in self._actor_handlers
    ])

def connect(self, critic_group):
    """连接 Actor 和 Critic（PPO 需要）"""
    return ray.get([
        actor.connect_actor_critic.remote(critic)
        for actor, critic in zip(
            self._actor_handlers, 
            critic_group._actor_handlers, 
            strict=False
        )
    ])
```

## Actor vs Critic 的区别

两者都是 `RayTrainGroup`，只是 `role` 不同：

```python
# Actor
actor_model = RayTrainGroup(..., role="actor")

# Critic  
critic_model = RayTrainGroup(..., role="critic")

# 连接两者（PPO 需要 Critic 给 Actor 提供 value）
actor_model.connect(critic_model)
```

## 与 RolloutManager 的关系

```
┌─────────────────┐     set_rollout_manager()     ┌─────────────────┐
│  RayTrainGroup  │  ─────────────────────────────► │  RolloutManager │
│  (Megatron)     │                                 │  (SGLang)       │
│                 │  ◄───────────────────────────── │                 │
└─────────────────┘      update_weights()          └─────────────────┘
        │
        │ async_train()
        ▼
┌─────────────────┐
│  GPU Training   │
└─────────────────┘
```

## Key Points

- `RayTrainGroup` 是 Megatron 训练的一层 Ray 封装
- 管理一组分布式训练的 Ray Actor（world_size 个）
- Rank 0 作为 coordinator 提供 master_addr/master_port
- 提供统一的初始化、训练、保存、权重同步等接口
- Actor 和 Critic 都是 RayTrainGroup 实例，只是 role 不同

## Code References

- `slime/ray/actor_group.py:10` - 类定义
- `slime/ray/actor_group.py:46` - `_allocate_gpus_for_actor` 方法
- `slime/ray/actor_group.py:101` - `async_init` 方法

## Follow-up Questions

- [ ] `MegatronTrainRayActor` 内部是如何实现 Megatron 初始化的？
- [ ] `update_weights` 是如何将权重同步到 SGLang 的？
- [ ] `torch_memory_saver` 在 colocate 模式中的工作原理？
