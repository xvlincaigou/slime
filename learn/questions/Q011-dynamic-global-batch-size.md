---
date: 2026-04-12
question_id: Q011
topics: ["training", "batch-size", "optimization", "rollout"]
related_files:
  - slime/ray/rollout.py
---

# Question

我没有理解 RolloutManager 里面的 `_compute_dynamic_global_batch_size` 在干什么。这个和 DAPO 里面的动态批处理有关系吗？

# Answer

## `_compute_dynamic_global_batch_size` 详解

**文件**: `slime/ray/rollout.py:610-635`

```python
def _compute_dynamic_global_batch_size(self, num_samples: int) -> int:
    """Calculate dynamic global_batch_size to ensure only one training step.

    Strategy: global_batch_size = num_samples rounded down to a multiple of dp_size
    This ensures num_steps_per_rollout = num_samples // global_batch_size = 1
    """
    dp_size = self.train_parallel_config["dp_size"]
    original_gbs = self.args.global_batch_size

    # Round down to a multiple of dp_size to ensure only one training step
    dynamic_gbs = (num_samples // dp_size) * dp_size

    if dynamic_gbs == 0:
        dynamic_gbs = dp_size
        logger.warning(f"num_samples={num_samples} < dp_size={dp_size}")

    wasted = num_samples - dynamic_gbs  # 被丢弃的样本数

    return dynamic_gbs
```

## 核心目的：**确保每次 Rollout 只进行一次训练步骤**

### 背景问题

正常训练中：
```
num_steps_per_rollout = num_samples / global_batch_size
```

假设：
- rollout 生成 1000 个样本
- global_batch_size = 256
- num_steps = 1000 / 256 = 3.9 → 3 或 4 步

**问题**：一次 rollout 需要多步训练，导致：
1. 训练效率低（多次前向/反向传播）
2. 代码逻辑复杂（需要处理多步梯度累积）

### 解决方案

动态调整 global_batch_size，使得：
```
num_steps = num_samples / dynamic_gbs = 1
```

**计算方式**：
```python
dynamic_gbs = (num_samples // dp_size) * dp_size
```

**示例**：
```
num_samples = 1000, dp_size = 8

dynamic_gbs = (1000 // 8) * 8 = 124 * 8 = 992

num_steps = 1000 / 992 ≈ 1.008 → 1 步
wasted = 1000 - 992 = 8 个样本被丢弃
```

## 与 DAPO 的关系

### DAPO 的动态批处理

DAPO (Dynamic Action Preference Optimization) 是 ByteDance 2025 年提出的 RL 算法，其**动态批处理**指的是：

| 特性 | DAPO 动态批处理 | Slime 的 `_compute_dynamic_global_batch_size` |
|------|----------------|----------------------------------------------|
| **核心思想** | 根据样本难度/质量动态调整 batch 组成 | 根据 rollout 样本数动态调整 GBS |
| **目的** | 提高训练效率，优先学习难样本 | 确保每次 rollout 只训练一步 |
| **调整对象** | Batch 内的样本选择策略 | Global batch size 数值 |
| **丢弃样本** | 通常不丢弃 | 可能丢弃少量样本（`wasted`） |

### 结论

**不是同一个东西**：
- Slime 的函数是**工程优化**：简化训练循环，确保 `num_steps=1`
- DAPO 的动态批处理是**算法优化**：根据样本特性调整 batch 组成

两者可以**结合使用**：
```python
# Slime 确保 num_steps=1
dynamic_gbs = _compute_dynamic_global_batch_size(num_samples)

# DAPO 在这个 batch 内做动态采样
# （如果 DAPO 实现在数据加载层面）
```

## 调用时机

**文件**: `slime/ray/rollout.py:591-598`

```python
if self.args.use_dynamic_global_batch_size:
    logger.info(f"Collected {len(data)} samples from rollout")
    self._dynamic_global_batch_size = self._compute_dynamic_global_batch_size(len(data))
    global_batch_size = self._dynamic_global_batch_size
```

在 `_get_rollout_data()` 中，当 `use_dynamic_global_batch_size=True` 时调用。

## 最终数据分割

动态 GBS 会传递给训练侧：

```python
# _split_train_data_by_dp 中
if hasattr(self, "_dynamic_global_batch_size"):
    rollout_data["dynamic_global_batch_size"] = self._dynamic_global_batch_size
```

训练侧使用这个值替代固定的 `global_batch_size`。

## Key Points

- 目的是**确保每次 rollout 只进行一次训练步骤**（`num_steps=1`）
- 通过向下取整到 dp_size 的倍数来计算动态 GBS
- 可能会丢弃少量样本（`wasted`）
- 与 DAPO 的动态批处理**不是同一个概念**，可以结合使用

## Code References

- `slime/ray/rollout.py:610` - `_compute_dynamic_global_batch_size` 函数
- `slime/ray/rollout.py:591` - 调用时机

## Follow-up Questions

- [ ] 动态 GBS 如何影响训练收敛性？
- [ ] 被丢弃的样本是否可以用于下一次 rollout？
- [ ] DAPO 算法在 Slime 中是如何实现的？
