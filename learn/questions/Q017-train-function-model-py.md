---
date: 2026-04-13
question_id: Q017
topics: ["training", "megatron", "pipeline-parallelism", "ddp", "optimizer"]
related_files:
  - slime/backends/megatron_utils/model.py
---

# Question

讲解一下 `slime/backends/megatron_utils/model.py` 中的 `train()` 函数。

# Answer

## 一句话总结

`train` 函数是 Megatron 训练流程的**最外层循环**，负责配置 Pipeline Parallelism（梯度/参数同步、DDP hooks），循环执行每个 training step，并记录训练指标。

## 逐行详解

### 第 1 部分：初始化与模式设置

```python
args = get_args()  # 获取 Megatron 全局参数

# 重置 data_iterator，确保从第一个 microbatch 开始遍历
for iterator in data_iterator:
    iterator.reset()

# 切换到训练模式（启用 dropout、batch_norm 更新等）
for model_module in model:
    model_module.train()
```

### 第 2 部分：配置 Megatron Pipeline 并行

```python
config = get_model_config(model[0])  # 获取模型配置
config.grad_scale_func = optimizer.scale_loss  # 混合精度梯度缩放
config.timers = None  # 关闭计时器
```

#### 梯度同步重叠（overlap_grad_reduce）

```python
if isinstance(model[0], DDP) and args.overlap_grad_reduce:
    # 配置 no_sync：在 backward 期间延迟梯度同步，与计算重叠
    config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
    
    # 配置 grad_sync_func：手动触发梯度同步（用于对齐所有 rank）
    if args.align_grad_reduce:
        config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
```

**作用**：Pipeline Parallelism 中，梯度 all-reduce 可以和 backward 计算重叠，提升效率。

#### 参数同步重叠（overlap_param_gather）

```python
if args.overlap_param_gather and args.align_param_gather:
    config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
```

**作用**：Forward 前异步预取下一层参数，减少等待时间。

```python
config.finalize_model_grads_func = finalize_model_grads  # 梯度后处理
```

### 第 3 部分：特殊配置

#### 重置优化器状态（可选）

```python
if args.reset_optimizer_states:
    for chained_optimizer in optimizer.chained_optimizers:
        for group in chained_optimizer.optimizer.param_groups:
            if "step" in group: group["step"] = 0
        for state in chained_optimizer.optimizer.state.values():
            if "step" in state: state["step"] = 0
            if "exp_avg" in state: state["exp_avg"].zero_()
            if "exp_avg_sq" in state: state["exp_avg_sq"].zero_()
```

#### 手动垃圾回收（可选）

```python
if args.manual_gc:
    gc.disable()  # 关闭自动 GC
    gc.collect()  # 立即执行一次 GC
```

#### 禁用 Forward Pre-hook（初始阶段）

```python
if should_disable_forward_pre_hook(args):
    disable_forward_pre_hook(model, param_sync=False)
    param_sync_func = config.param_sync_func  # 暂存
    config.param_sync_func = None  # 临时禁用
    pre_hook_enabled = False
```

**原因**：第一次 forward 前，参数可能未正确加载，禁用 hook 避免错误传播。

### 第 4 部分：训练循环

```python
num_steps_per_rollout = len(num_microbatches)

for step_id in range(num_steps_per_rollout):
    # 执行单个训练 step
    loss_dict, grad_norm = train_one_step(
        args, rollout_id, step_id, data_iterator, model,
        optimizer, opt_param_scheduler, num_microbatches[step_id]
    )
    
    # 第一个 step 成功后，启用 forward pre-hook
    if step_id == 0 and should_disable_forward_pre_hook(args):
        enable_forward_pre_hook(model)
        config.param_sync_func = param_sync_func
        pre_hook_enabled = True
```

### 第 5 部分：MTP（Multi-Token Prediction）损失

```python
if args.enable_mtp_training:
    mtp_loss_scale = 1 / num_microbatches[step_id]
    tracker = MTPLossLoggingHelper.tracker
    if "values" in tracker:
        values = tracker["values"]
        if tracker.get("reduce_group") is not None:
            torch.distributed.all_reduce(values, group=tracker.get("reduce_group"))
        mtp_losses = (tracker["values"] * mtp_loss_scale).item()
```

### 第 6 部分：日志记录

```python
# 只在特定 rank 记录（PP last stage, TP rank 0, DP rank 0）
if (mpu.get_data_parallel_rank(...) == 0 and 
    mpu.get_tensor_model_parallel_rank() == 0 and
    mpu.get_pipeline_model_parallel_rank() == last_stage):
    
    accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
    role = getattr(model[0], "role", "actor")
    
    log_dict = {f"train/{role}/{key}": val for key, val in loss_dict.items()}
    log_dict[f"train/{role}/grad_norm"] = grad_norm
    
    # 记录各参数组的学习率
    for param_group_id, param_group in enumerate(optimizer.param_groups):
        log_dict[f"train/{role}/lr-pg_{param_group_id}"] = opt_param_scheduler.get_lr(param_group)
    
    log_dict["train/step"] = accumulated_step_id
    logging_utils.log(args, log_dict, step_key="train/step")
```

### 第 7 部分：清理

```python
if pre_hook_enabled:
    disable_forward_pre_hook(model)
```

## 关键流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                         train()                                 │
├─────────────────────────────────────────────────────────────────┤
│ 1. 初始化                                                       │
│    ├── 重置 data_iterator                                        │
│    └── 切换到 train() 模式                                       │
├─────────────────────────────────────────────────────────────────┤
│ 2. 配置 Pipeline 并行                                            │
│    ├── grad_scale_func（梯度缩放）                                │
│    ├── no_sync_func（延迟梯度同步）                               │
│    ├── grad_sync_func（手动触发同步）                             │
│    └── param_sync_func（参数预取）                                │
├─────────────────────────────────────────────────────────────────┤
│ 3. 特殊配置（可选）                                               │
│    ├── 重置优化器状态                                             │
│    ├── 手动垃圾回收                                               │
│    └── 临时禁用 forward pre-hook                                 │
├─────────────────────────────────────────────────────────────────┤
│ 4. 训练循环（foreach step）                                      │
│    ├── train_one_step()  ← 实际训练                              │
│    ├── step 0 后启用 forward pre-hook                            │
│    ├── 记录 MTP 损失（如启用）                                    │
│    └── 记录训练日志（loss, grad_norm, lr）                        │
├─────────────────────────────────────────────────────────────────┤
│ 5. 清理                                                         │
│    └── 禁用 forward pre-hook                                     │
└─────────────────────────────────────────────────────────────────┘
```

## `train()` vs `train_one_step()` 区别

| 函数 | 职责 | 调用次数 |
|------|------|----------|
| `train()` | **外层循环**，配置环境，管理状态 | 每个 rollout 1 次 |
| `train_one_step()` | **单个 step**，执行 forward/backward/optimizer | 每个 step 1 次 |

## Pipeline 并行配置详解

| 配置项 | 作用 | 优化效果 |
|--------|------|----------|
| `no_sync_func` | 延迟梯度同步 | backward 和 all-reduce 重叠 |
| `grad_sync_func` | 手动触发梯度同步 | 精确控制同步时机 |
| `param_sync_func` | 异步预取参数 | forward 和参数传输重叠 |
| `finalize_model_grads_func` | 梯度后处理 | bucket 合并、norm clipping |

## Key Points

1. **Pipeline 并行配置**：`train()` 负责设置 Megatron Pipeline 并行的各种回调函数（梯度同步、参数预取）。

2. **Forward Pre-hook 管理**：初始阶段禁用，第一个 step 成功后启用，避免初始化错误传播。

3. **日志聚合**：只在 Pipeline last stage + TP rank 0 + DP rank 0 记录，避免重复日志。

4. **MTP 支持**：可选的 Multi-Token Prediction 损失记录和同步。

## Code References

- `slime/backends/megatron_utils/model.py:489` - `train()` 函数定义
- `slime/backends/megatron_utils/model.py:299` - `train_one_step()` 函数
- `slime/backends/megatron_utils/model.py:152` - `forward_only()` 函数

## Follow-up Questions

- [ ] `train_one_step` 内部的具体 forward/backward 流程是怎样的？
- [ ] `forward_backward_func` 是如何实现 Pipeline 并行的？
- [ ] DDP 的 `no_sync` 和 `start_grad_sync` 具体是如何工作的？
