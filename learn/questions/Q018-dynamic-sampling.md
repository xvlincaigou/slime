---
date: 2026-04-14
question_id: Q018
topics: ["rollout", "dynamic-sampling", "filter", "reward"]
related_files:
  - slime/rollout/sglang_rollout.py
  - slime/rollout/filter_hub/dynamic_sampling_filters.py
  - slime/rollout/filter_hub/base_types.py
---

# Question

以 `--over-sampling-batch-size ${OVER_SAMPLING_BS} --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std` 为例子，说明 Slime 里面的 dynamic sampling 是如何起作用的。

# Answer

## 一句话总结

Dynamic Sampling 通过**过采样（over-sampling）+ 动态过滤（dynamic filter）**的组合，自动筛选出有学习价值的样本（奖励方差非零），提升训练效率。

## 背景问题

在 RL 训练中，某些 prompt 生成的 responses 可能存在：
- 奖励完全相同（比如全是 0，或全是 1）
- 没有学习信号（无法区分好坏）
- 浪费计算资源

**Dynamic Sampling** 就是用来过滤掉这些"无价值"样本的机制。

## 参数配置示例

```bash
--over-sampling-batch-size 256 \      # 每次从 buffer 取 256 个样本
--rollout-batch-size 128 \            # 但只保留 128 个有效样本
--dynamic-sampling-filter-path \
  slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

## 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                    Dynamic Sampling Flow                    │
├─────────────────────────────────────────────────────────────┤
│ 1. 从 Buffer 取 over_sampling_batch_size (256) 个样本        │
│                    ↓                                         │
│ 2. 提交到 SGLang 生成 responses                               │
│                    ↓                                         │
│ 3. 对每个 group (n_samples_per_prompt 个样本)                 │
│    ├── 计算奖励                                               │
│    ├── 调用 dynamic_filter(args, group)                       │
│    │       └── check_reward_nonzero_std                       │
│    │           ├── 计算 group 内奖励的标准差                   │
│    │           └── std > 1e-6 ? keep : drop                   │
│    └── 如果 keep=False，丢弃，remaining_batch_size -= 1        │
│                    ↓                                         │
│ 4. 直到收集够 rollout_batch_size (128) 个有效样本             │
└─────────────────────────────────────────────────────────────┘
```

## 代码详解

### 1. 过滤器实现

**文件**: `slime/rollout/filter_hub/dynamic_sampling_filters.py`

```python
def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    # samples 是一个 group（同一个 prompt 的多个 responses）
    rewards = [sample.get_reward_value(args) for sample in samples]
    
    # 计算奖励的标准差
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    
    return DynamicFilterOutput(
        keep=keep,  # 是否保留这个 group
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",  # 丢弃原因
    )
```

**核心逻辑**：
- 如果奖励标准差接近 0（所有 response 奖励相同），丢弃
- 原因示例：`zero_std_0.0`（全是0）或 `zero_std_1.0`（全是1）

### 2. 过滤调用流程

**文件**: `slime/rollout/sglang_rollout.py:420-425`

```python
all_data.append(group)
dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
if not dynamic_filter_output.keep:
    metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
    state.remaining_batch_size -= 1
    continue  # 跳过这个 group，不加入 data

# 只有 keep=True 的 group 才会被加入训练数据
if len(data) < target_data_size:
    data.append(group)
```

### 3. 输出类型定义

**文件**: `slime/rollout/filter_hub/base_types.py`

```python
@dataclass
class DynamicFilterOutput:
    keep: bool          # 是否保留
    reason: str | None  # 丢弃原因（用于 metrics）

def call_dynamic_filter(fn, *args, **kwargs):
    if fn is None:
        return DynamicFilterOutput(keep=True)  # 没有 filter 时全保留
    
    output = fn(*args, **kwargs)
    
    # 兼容旧版 bool 返回值
    if not isinstance(output, DynamicFilterOutput):
        output = DynamicFilterOutput(keep=output)
    
    return output
```

### 4. Metrics 收集

**文件**: `slime/rollout/filter_hub/base_types.py:24-37`

```python
class MetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def collect(self):
        return {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
```

**输出示例**：
```python
{
    "rollout/dynamic_filter/drop_zero_std_0.0": 15,  # 15 个 group 因奖励全0被丢弃
    "rollout/dynamic_filter/drop_zero_std_1.0": 20,  # 20 个 group 因奖励全1被丢弃
}
```

## 为什么需要 Over-sampling？

假设：
- `rollout_batch_size = 128`（目标）
- `over_sampling_batch_size = 256`（实际采样）
- 50% 的 group 被过滤掉

**没有 over-sampling**：
- 采样 128 个，过滤掉 64 个，只剩 64 个有效样本
- 训练 batch 不足，效率降低

**有 over-sampling**：
- 采样 256 个，过滤掉 128 个，还剩 128 个有效样本
- 正好满足 `rollout_batch_size`

## 自定义过滤器

你可以实现自己的过滤器：

```python
from slime.rollout.filter_hub.base_types import DynamicFilterOutput

def my_custom_filter(args, samples: list[Sample], **kwargs):
    # 自定义过滤逻辑
    rewards = [sample.get_reward_value(args) for sample in samples]
    
    # 例如：只保留有正奖励的 group
    has_positive = any(r > 0 for r in rewards)
    
    return DynamicFilterOutput(
        keep=has_positive,
        reason="no_positive_reward" if not has_positive else None,
    )
```

然后使用：
```bash
--dynamic-sampling-filter-path my_module.my_custom_filter
```

## Key Points

1. **Over-sampling**：多采一些样本，预留过滤的余地
2. **Dynamic Filter**：根据奖励方差判断样本是否有学习价值
3. **Keep/Drop**：`keep=True` 保留，`keep=False` 丢弃并记录原因
4. **Metrics**：统计各种丢弃原因，用于监控和调试

## Code References

- `slime/rollout/sglang_rollout.py:389` - dynamic_filter 加载
- `slime/rollout/sglang_rollout.py:420` - dynamic_filter 调用
- `slime/rollout/filter_hub/dynamic_sampling_filters.py` - 示例过滤器
- `slime/rollout/filter_hub/base_types.py` - DynamicFilterOutput 定义

## Follow-up Questions

- [ ] 还有哪些内置的 dynamic filter？
- [ ] buffer-filter 和 rollout-sample-filter 有什么区别？
- [ ] 如何评估 dynamic filter 对训练效果的影响？
