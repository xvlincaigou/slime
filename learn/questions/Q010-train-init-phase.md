---
date: 2026-04-09
question_id: Q010
topics: ["training", "initialization", "weight-sync", "megatron", "sglang"]
related_files:
  - train.py
  - slime/ray/rollout.py
---

# Question

```python
if args.offload_rollout:
    ray.get(rollout_manager.onload_weights.remote())

# always update weight first so that sglang has the loaded weights from training.
if not args.critic_train_only:
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

if args.offload_rollout:
    ray.get(rollout_manager.onload_kv.remote())

# special case for eval-only
if args.num_rollout == 0 and args.eval_interval is not None:
    ray.get(rollout_manager.eval.remote(rollout_id=0))
```

这段代码是在干什么？

# Answer

这是 **训练开始前的初始化/同步阶段**，主要做三件事：**恢复 SGLang 状态**、**同步训练权重**、**处理纯评估模式**。

## 代码拆解

**文件**: `train.py:26-41`

```python
# 1. 加载权重到 GPU（如果是 colocate 模式且之前 offload 了）
if args.offload_rollout:
    ray.get(rollout_manager.onload_weights.remote())

# 2. 将 Actor 的权重同步到 SGLang（关键！）
if not args.critic_train_only:
    actor_model.update_weights()  # Megatron → SGLang 权重同步
    
    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

# 3. 加载 KV cache 和 CUDA graphs
if args.offload_rollout:
    ray.get(rollout_manager.onload_kv.remote())

# 4. 纯评估模式特殊处理
if args.num_rollout == 0 and args.eval_interval is not None:
    ray.get(rollout_manager.eval.remote(rollout_id=0))
```

## 详细解释

### 1. `onload_weights()` - 恢复权重到 GPU

当使用 `--offload-rollout`（colocate 模式）时：
- SGLang 引擎的权重之前被 offload 到了 CPU
- 在同步权重前，需要先把权重恢复到 GPU

### 2. `actor_model.update_weights()` - 关键同步 ⭐

**这是 Megatron 和 SGLang 之间的桥梁**：

```
Megatron Actor (训练后)
    ↓
actor_model.update_weights()
    ↓
通过 NCCL/RDMA 发送权重
    ↓
SGLang 引擎接收并更新
```

**为什么需要？**
- Megatron 和 SGLang 是两个独立的进程
- 训练在 Megatron 中进行，推理在 SGLang 中进行
- 每次 rollout 前必须确保 SGLang 使用最新的模型权重

### 3. `check_weights(action="compare")` - 调试检查

验证 Megatron 和 SGLang 的权重是否一致，用于调试权重同步问题。

### 4. `onload_kv()` - 恢复 KV cache

权重同步完成后，恢复 KV cache 和 CUDA graphs 到 GPU。

### 5. Eval-only 特殊处理

如果 `--num-rollout 0` 且设置了 `--eval-interval`，直接运行评估（不进入训练循环）。

## 流程图

```
训练开始前初始化:
│
├─► [colocate模式?] ──Yes──► onload_weights() ──► 权重加载到GPU
│                              │
│                              ▼
│                         update_weights() ──► Megatron→SGLang同步
│                              │
│                              ▼
│                         [检查权重?] ──Yes──► check_weights(compare)
│                              │
│                              ▼
│                         onload_kv() ──► KV cache加载到GPU
│
└─► [eval-only?] ──Yes──► eval(rollout_id=0) ──► 直接评估并退出
```

## 对比：训练循环中的相同逻辑

注意第 94-99 行也有类似逻辑：

```python
# 训练循环中，每次 rollout 后
if args.offload_rollout:
    ray.get(rollout_manager.onload_weights.remote())
if not args.critic_train_only:
    actor_model.update_weights()  # ← 同样同步权重
if args.offload_rollout:
    ray.get(rollout_manager.onload_kv.remote())
```

**区别**：
- 第 26-37 行：**初始化阶段**（第一次同步）
- 第 94-99 行：**训练循环中**（每次 rollout 后同步）

## Key Points

- 这是训练前的**权重同步准备阶段**
- `update_weights()` 是 Megatron 和 SGLang 之间的关键桥梁
- colocate 模式下需要分步恢复：权重 → 同步 → KV cache
- eval-only 模式会跳过训练循环直接评估

## Code References

- `train.py:26-41` - 初始化阶段
- `train.py:94-99` - 训练循环中的同步
- `slime/ray/rollout.py:520-526` - `onload_weights` / `onload_kv` 实现

## Follow-up Questions

- [ ] `update_weights()` 内部是如何实现权重传输的？
- [ ] NCCL/RDMA 在权重同步中的作用？
- [ ] 为什么 KV cache 要在权重同步后加载？
