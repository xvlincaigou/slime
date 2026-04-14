---
date: 2026-04-14
question_id: Q020
topics: ["training", "data-packing", "loss", "dynamic-batch-size"]
related_files:
  - slime/backends/megatron_utils/cp_utils.py
  - slime/backends/megatron_utils/loss.py
---

# Question

怎么理解文档里面的这句话：

> ⚠️ slime 总是会通过 data packing 的方法训练模型，并且严格保证 per sample loss 或 per token loss，也就是开启 dynamic batch size 不会对 loss 计算有影响，推荐开启。

# Answer

## 一句话总结

Slime 通过 **sample 级别的独立 loss 计算**（先把 packed 数据切分回单个 sample，各自计算 loss 后再聚合），使得无论 batch 如何划分（dynamic batch size），总 loss 只与样本本身有关，与组合方式无关。

## 1. Data Packing（数据打包）

**问题背景**:
- 每个 prompt 生成的 response 长度不同
- 如果不处理，短序列需要 padding 到最长，浪费计算
- Data packing 把多个 sample 拼接成一个长序列训练

**示例**:
```
Sample 1: [A, B, C]           (长度 3)
Sample 2: [D, E, F, G]        (长度 4)
Sample 3: [H, I]              (长度 2)

Packing 后: [A, B, C, D, E, F, G, H, I]
            ↑ Sample 1 ↑ Sample 2  ↑ Sample 3
            
通过 cu_seqlens 记录边界: [0, 3, 7, 9]
```

## 2. Per Sample Loss vs Per Token Loss

### Per Sample Loss（默认）

```python
# 每个 sample 单独计算平均 loss，然后求和
loss = sum([
    (sample_loss * loss_mask).sum() / sample_token_count  # 该 sample 的平均 loss
    for sample in samples
])
```

**特点**:
- 长短样本权重相同（都贡献 1 份平均 loss）
- 不会因为某个样本很长就主导总 loss

### Per Token Loss

```python
# 所有 token 的 loss 直接求和（不按 sample 区分）
loss = sum([
    (sample_loss * loss_mask).sum()  # 只求和，不除以 token 数
    for sample in samples
])
```

**特点**:
- 长样本贡献更大（token 多）
- 适合某些特定场景

## 3. 为什么 Dynamic Batch Size 不影响 Loss？

**关键代码** (`slime/backends/megatron_utils/cp_utils.py:53-120`):

```python
def get_sum_of_sample_mean(...):
    """计算正确的 sample mean，支持 Context Parallel"""
    
    def sum_of_sample_mean(x: torch.Tensor) -> torch.Tensor:
        return sum([
            (x_i * loss_mask_i).sum() / torch.clamp_min(loss_mask_i.sum(), 1)
            for x_i, loss_mask_i in zip(
                x.split(response_lengths, dim=0),  # ← 按样本分割
                loss_masks,
                strict=False
            )
        ])
    
    return sum_of_sample_mean
```

**核心逻辑**:
1. **按样本分割**: `x.split(response_lengths)` 把 packed tensor 切回每个 sample
2. **独立计算**: 每个 sample 的 loss 单独计算（除以该 sample 的有效 token 数）
3. **求和聚合**: 所有 sample 的 loss 加起来

**数学证明**:

假设有 4 个样本，两种 microbatch 划分方式：

```
方式 1 (2 microbatches):
  Microbatch 0: [Sample A, Sample B] → Loss_MB0 = Loss_A + Loss_B
  Microbatch 1: [Sample C, Sample D] → Loss_MB1 = Loss_C + Loss_D
  总 Loss = Loss_MB0 + Loss_MB1 = Loss_A + Loss_B + Loss_C + Loss_D

方式 2 (1 microbatch):
  Microbatch 0: [Sample A, B, C, D] → 总 Loss = Loss_A + Loss_B + Loss_C + Loss_D
```

无论样本如何分组，最终总 loss 都是各样本 loss 之和，与分组方式无关！

## 4. 代码中的实际使用

**Loss 计算** (`slime/backends/megatron_utils/loss.py:744-767`):

```python
# 创建 reducer 函数（sample 级别的聚合器）
sum_of_sample_mean = get_sum_of_sample_mean(
    total_lengths,
    response_lengths,
    loss_masks,
    args.calculate_per_token_loss,  # False = per sample, True = per token
    args.qkv_format,
    max_seq_lens,
)

# 使用 reducer 计算各项 loss
pg_loss = pg_loss_reducer(pg_loss)          # policy gradient loss
pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
ppo_kl = sum_of_sample_mean(ppo_kl)         # KL 散度
entropy_loss = sum_of_sample_mean(entropy)  # 熵奖励
```

## 5. Context Parallel 的支持

CP（Context Parallel）会把一个 sample 切分到多个 GPU 上，`get_sum_of_sample_mean` 也处理了这种情况：

```python
if cp_size > 1:
    # 对每个 sample，根据 CP 的切分方式重组 loss_mask
    for i, (total_length, response_length, loss_mask) in enumerate(...):
        _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(...)
        # 提取当前 CP rank 负责的 chunks
        loss_mask_0 = loss_mask[tokens_offset[0][0] - prompt_length : ...]
        loss_mask_1 = loss_mask[tokens_offset[1][0] - prompt_length : ...]
        chunked_loss_masks.append(torch.cat([loss_mask_0, loss_mask_1]))
```

即使在 CP 环境下，每个 sample 的 loss 计算仍然是独立的。

## 6. 总结

| 特性 | Per Sample Loss | Per Token Loss |
|------|----------------|---------------|
| 计算方式 | 每个 sample 平均后求和 | 所有 token 直接求和 |
| 样本权重 | 长短样本权重相同 | 长样本权重更大 |
| Dynamic Batch Size | 不受影响 ✅ | 不受影响 ✅ |
| 推荐场景 | 通用 RL 训练（默认） | 特定需求 |

**文档推荐开启 dynamic batch size 的原因**:
- Packing 机制保证了无论 batch 如何划分，loss 计算都是正确的
- 可以享受 dynamic batch size 的显存优化（根据序列长度动态调整，避免 OOM）
- 不影响训练效果和收敛性

## Code References

- `slime/backends/megatron_utils/cp_utils.py:53` - `get_sum_of_sample_mean` 函数定义
- `slime/backends/megatron_utils/loss.py:617` - `policy_loss_function` 中使用
- `slime/backends/megatron_utils/data.py:25` - `get_batch` 函数（packing 实现）

## Follow-up Questions

- [ ] `calculate_per_token_loss` 参数在什么场景下会开启？
- [ ] Packing 对梯度累积有什么影响？
- [ ] 如何在日志中区分 per sample 和 per token 的 metric？
