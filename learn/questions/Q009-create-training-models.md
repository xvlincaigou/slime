---
date: 2026-04-09
question_id: Q009
topics: ["training", "actor", "critic", "megatron", "rl"]
related_files:
  - slime/ray/placement_group.py
  - slime/ray/actor_group.py
---

# Question

`actor_model, critic_model = create_training_models(args, pgs, rollout_manager)` 是在干什么？

# Answer

这是创建 **Actor（策略模型）** 和 **Critic（价值模型）** 训练实例的函数，负责初始化 Megatron 训练的模型。

## 核心作用

**文件**: `slime/ray/placement_group.py:133-180`

```python
def create_training_models(args, pgs, rollout_manager):
    # 1. 创建 Actor 模型（必须）
    actor_model = allocate_train_group(
        args=args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
    )
    
    # 2. 创建 Critic 模型（可选，由 use_critic 控制）
    if args.use_critic:
        critic_model = allocate_train_group(
            args=args,
            num_nodes=args.critic_num_nodes,
            num_gpus_per_node=args.critic_num_gpus_per_node,
            pg=pgs["critic"],
            role="critic",
        )
        critic_init_handle = critic_model.async_init(args, role="critic", with_ref=False)
    else:
        critic_model = None

    # 3. 初始化 Actor
    start_rollout_ids = ray.get(
        actor_model.async_init(
            args,
            role="actor",
            with_ref=args.kl_coef != 0 or args.use_kl_loss,  # 是否需要参考模型
            with_opd_teacher=args.use_opd and args.opd_type == "megatron",
        )
    )

    # 4. Actor 和 Critic 连接（用于 PPO 等算法）
    if args.use_critic:
        critic_start_rollout_ids = ray.get(critic_init_handle)
        if not args.critic_train_only:
            actor_model.connect(critic_model)

    # 5. 关联 RolloutManager
    actor_model.set_rollout_manager(rollout_manager)
    if args.use_critic:
        critic_model.set_rollout_manager(rollout_manager)

    return actor_model, critic_model
```

## 内部实现：`allocate_train_group`

**文件**: `slime/ray/placement_group.py:122-130`

```python
def allocate_train_group(args, num_nodes, num_gpus_per_node, pg, role="actor"):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.4,  # 类似 SGLang 的 0.2，Ray 只调度，实际 GPU 由 Megatron 管理
        role=role,
    )
```

## 关键流程

```
create_training_models()
    ├── allocate_train_group(role="actor")     → 创建 Actor 的 RayTrainGroup
    ├── allocate_train_group(role="critic")    → 创建 Critic 的 RayTrainGroup（可选）
    ├── actor_model.async_init()               → 初始化 Actor（Megatron 并行）
    ├── critic_model.async_init()              → 初始化 Critic（Megatron 并行）
    ├── actor_model.connect(critic_model)      → Actor-Critic 连接
    └── set_rollout_manager()                  → 关联推理引擎
```

## Actor vs Critic

| 组件 | 作用 | 是否必需 |
|------|------|----------|
| **Actor** | 策略模型，生成文本，接收训练更新 | ✅ 必需 |
| **Critic** | 价值模型，估计状态价值，用于 PPO 等算法 | ❌ 可选 |

## 为什么需要 `with_ref` 参数？

```python
with_ref=args.kl_coef != 0 or args.use_kl_loss
```

当使用 KL 散度约束（如 GRPO、PPO with KL）时，需要维护一个 **参考模型（Reference Model）** 的副本，用于计算新旧策略的差异。

## `num_gpus_per_actor=0.4` 的含义

与 SGLang 的 `num_gpus=0.2` 类似，这是一个**调度技巧**：
- Ray 用这个值做调度决策
- 实际 GPU 使用由 Megatron 内部管理
- 确保 Ray 知道这是 GPU 任务，但不限制实际 GPU 占用

## Key Points

- `create_training_models` 是训练流程的核心初始化函数
- Actor 必须创建，Critic 由 `use_critic` 参数控制
- 使用 `async_init` 异步初始化，通过 `ray.get` 等待完成
- Actor 和 Critic 通过 `connect` 方法建立连接
- 两个模型都通过 `set_rollout_manager` 关联到推理引擎

## Code References

- `slime/ray/placement_group.py:133` - `create_training_models` 函数
- `slime/ray/placement_group.py:122` - `allocate_train_group` 函数
- `slime/ray/actor_group.py` - `RayTrainGroup` 类实现

## Follow-up Questions

- [ ] `RayTrainGroup` 内部是如何管理 Megatron 训练的？
- [ ] `async_init` 具体做了什么初始化工作？
- [ ] `actor_model.connect(critic_model)` 建立了什么连接？
- [ ] 参考模型（Reference Model）是如何实现的？
