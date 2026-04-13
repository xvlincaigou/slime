---
date: 2026-04-12
question_id: Q012
topics: ["training", "data-parallel", "grpo", "distributed-training"]
related_files:
  - slime/ray/rollout.py
  - slime/rollout/data_source.py
---

# Question

在 `_split_train_data_by_dp` 当中，我们会把 GRPO 的一个 sample 的多个 response 分到不同的 DP 上面去吗？

# Answer

## 答案：**是的，会分到不同的 DP 上**

**文件**: `slime/ray/rollout.py:751-802`

```python
def _split_train_data_by_dp(self, data, dp_size):
    """Split the train data by data parallel size."""
    ...
    if self.args.balance_data:
        partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
    else:
        # 默认：简单轮询分割
        partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]
```

## 分割逻辑

假设 GRPO 配置：
- `n_samples_per_prompt = 4`（一个 prompt 生成 4 个 response）
- `dp_size = 4`（4 个数据并行 rank）

**Data Source 生成的样本**（连续排列）：
```
索引:  0     1     2     3     4     5     6     7    ...
      [G0]  [G0]  [G0]  [G0]  [G1]  [G1]  [G1]  [G1]  ...
      R1    R2    R3    R4    R1    R2    R3    R4
```
（G0=Group 0, G1=Group 1, R1-R4=Response 1-4）

**分割结果**：
| DP Rank | 分到的样本索引 | 来自哪些 Group |
|---------|---------------|---------------|
| DP 0 | 0, 4, 8, ... | G0, G1, G2, ... 的 Response 1 |
| DP 1 | 1, 5, 9, ... | G0, G1, G2, ... 的 Response 2 |
| DP 2 | 2, 6, 10, ... | G0, G1, G2, ... 的 Response 3 |
| DP 3 | 3, 7, 11, ... | G0, G1, G2, ... 的 Response 4 |

## 后果：GRPO 需要在 DP 间通信

因为同一个 group 的样本被分散到不同 DP rank，GRPO 的 advantage 计算需要 **all-gather** 操作：

```
DP 0: [reward_G0_R1, reward_G1_R1, reward_G2_R1, ...]
DP 1: [reward_G0_R2, reward_G1_R2, reward_G2_R2, ...]
DP 2: [reward_G0_R3, reward_G1_R3, reward_G2_R3, ...]
DP 3: [reward_G0_R4, reward_G1_R4, reward_G2_R4, ...]
       ↓
All-Gather
       ↓
每个 DP 都有完整的 [reward_G0_R1..R4, reward_G1_R1..R4, ...]
才能计算 mean/std
```

## 为什么不按 Group 分割？

**设计权衡**：
1. **简单性**：轮询分割代码简单，无需关心 group 边界
2. **负载均衡**：序列长度通常比 group 内差异更大，按长度平衡更重要
3. **通信成本可接受**：DP all-gather 是轻量级操作（只传 reward，不传整个样本）

## 相关代码

**Group 创建**（`slime/rollout/data_source.py:108-116`）：
```python
for prompt_sample in prompt_samples:
    group = []
    for _ in range(self.args.n_samples_per_prompt):
        sample = copy.deepcopy(prompt_sample)
        sample.group_index = self.sample_group_index  # 同一个 group 共享 index
        sample.index = self.sample_index
        self.sample_index += 1
        group.append(sample)
    self.sample_group_index += 1
```

**GRPO 计算**（需要 all-gather）：
```python
# 在 Megatron 训练侧，advantage 计算前需要从所有 DP rank 收集同 group 的 reward
rewards = all_gather_from_all_dp_ranks(rewards)  # 伪代码
mean = rewards.mean(dim=-1, keepdim=True)
advantage = rewards - mean
```

## Key Points

- `_split_train_data_by_dp` 使用简单轮询分割，**不保持 GRPO group 的完整性**
- 同一个 prompt 的多个 response 会被分散到不同 DP rank
- GRPO 的 advantage 计算需要做跨 DP 的 all-gather 来收集同 group 的 reward
- 这是设计权衡的结果：简单性 > 避免通信

## Code References

- `slime/ray/rollout.py:751` - `_split_train_data_by_dp` 函数
- `slime/rollout/data_source.py:108` - Group 创建逻辑

## Follow-up Questions

- [ ] GRPO 的 advantage 计算具体在哪里实现 all-gather？
- [ ] `balance_data=True` 时的 `get_seqlen_balanced_partitions` 是否会考虑 group 边界？
- [ ] 跨 DP 通信对训练性能的影响有多大？
