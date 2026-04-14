---
date: 2026-04-14
question_id: Q019
topics: ["training", "async", "sync", "rollout", "pipeline"]
related_files:
  - train.py
  - train_async.py
---

# Question

train_async 和同步训练 train.py 有什么不一样的地方？

# Answer

## 一句话总结

`train_async.py` 实现了 **rollout 生成**和**模型训练**的**流水线重叠**，而 `train.py` 是顺序执行（等 rollout 完成再训练）。异步版本不支持 colocate，但吞吐量更高。

## 核心区别：执行流程

### 同步训练 (`train.py`)

```python
for rollout_id in range(num_rollout):
    # 1. 生成 rollout（阻塞等待完成）
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
    
    # 2. 训练（阻塞等待完成）
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
    
    # 3. 更新权重到 SGLang
    actor_model.update_weights()
```

**特点**：
- 生成 → 等待 → 训练 → 等待 → 更新权重
- 每个阶段串行执行

### 异步训练 (`train_async.py`)

```python
# 预启动第一个 rollout
rollout_data_next_future = rollout_manager.generate.remote(start_rollout_id)

for rollout_id in range(num_rollout):
    # 1. 获取当前 rollout（已经提前开始生成了）
    rollout_data_curr_ref = ray.get(rollout_data_next_future)
    
    # 2. 立即启动下一个 rollout（和当前训练并行）
    if rollout_id + 1 < num_rollout:
        rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)
    
    # 3. 训练当前 rollout
    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
```

**特点**：
- 训练当前 rollout 的同时，已经在生成下一个 rollout
- 实现了计算重叠

## 时间线对比

```
同步训练 timeline:
Rollout 0 | Train 0 | Update | Rollout 1 | Train 1 | Update |
[=======] [=======] [======] [=======]   [=======] [======]

异步训练 timeline:
Rollout 0 |          
          | Train 0 | Rollout 1 |
                    | Train 1 | Rollout 2 |
                              | Train 2 | ...
[=======] [=======] [=======] [=======] [=======]
          ↑ 重叠执行，节省总时间 ↑
```

## 关键差异点

| 特性 | train.py (同步) | train_async.py (异步) |
|------|----------------|----------------------|
| **Colocate 支持** | ✅ 支持 | ❌ 不支持 |
| **Rollout-Train 重叠** | ❌ 否 | ✅ 是 |
| **权重更新频率** | 每轮更新 | 按 `update_weights_interval` |
| **内存管理** | 复杂（offload） | 简化 |
| **吞吐量** | 较低 | 较高 |

## 代码细节对比

### 同步版本特有的：内存卸载

```python
# train.py
if args.offload_rollout:
    ray.get(rollout_manager.offload.remote())

# 训练
if args.use_critic:
    critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
    ...

# 内存管理
def offload_train(rollout_id):
    if args.offload_train:
        if args.use_critic:
            critic_model.offload()
            ...

offload_train(rollout_id)

# 重新加载
if args.offload_rollout:
    ray.get(rollout_manager.onload_weights.remote())
    ray.get(rollout_manager.onload_kv.remote())
```

### 异步版本特有的：预取机制

```python
# train_async.py
# 预启动第一个 rollout
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

for rollout_id in range(args.start_rollout_id, args.num_rollout):
    # 同步获取当前 rollout
    if rollout_data_next_future is not None:
        rollout_data_curr_ref = ray.get(rollout_data_next_future)
    
    # 立即启动下一个（关键！）
    if rollout_id + 1 < args.num_rollout:
        rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)
    
    # 训练当前 rollout（同时下一个 rollout 正在生成）
    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
```

### 异步版本特有的：按 interval 更新权重

```python
# train_async.py
if (rollout_id + 1) % args.update_weights_interval == 0:
    # 同步等待当前的生成完成，防止更新权重时还有生成在进行
    rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
    rollout_data_next_future = None
    
    if not args.critic_train_only:
        actor_model.update_weights()
```

**为什么需要这个？**
- 异步模式下，权重更新前必须确保没有正在进行的 rollout
- 否则可能出现：正在生成 response，权重突然更新了

## 为什么异步版本不支持 Colocate？

**Colocate 模式**：训练 GPU 和推理 GPU 是同一批，需要：
1. 训练时：`offload_train`（推理权重换出）
2. 推理时：`onload_weights`（推理权重加载）

**异步模式要求**：
- 同时做两件事：训练（用训练 GPU）+ rollout（用推理 GPU）

**矛盾**：
- 如果 colocate，同一批 GPU 无法同时执行训练和 rollout
- 所以 `train_async.py` 明确禁止：`assert not args.colocate`

## 选择建议

| 场景 | 推荐方案 |
|------|----------|
| GPU 资源紧张，需要训练/推理共用 | `train.py` + colocate |
| GPU 资源充足，追求最大吞吐 | `train_async.py` |
| 需要频繁评估、保存 checkpoint | `train.py`（流程更可控） |
| 快速原型验证 | `train.py`（更简单直接） |

## Key Points

1. **核心差异**：异步版本通过预取机制实现 rollout 和训练的重叠执行
2. **Colocate 限制**：异步版本不支持 colocate（GPU 无法同时做两件事）
3. **权重更新**：异步版本按 interval 更新，同步版本每轮都更新
4. **适用场景**：资源紧张用同步+colocate，追求速度用异步

## Code References

- `train.py:73-102` - 同步训练主循环
- `train_async.py:36-76` - 异步训练主循环
- `train_async.py:10` - colocate 不支持断言

## Follow-up Questions

- [ ] `update_weights_interval` 对训练效果有什么影响？
- [ ] 异步模式下如何处理 critic-only steps？
- [ ] 有没有 fully async 的实现（examples/full_async）？
