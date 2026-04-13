---
date: 2026-04-13
question_id: Q016
topics: ["training", "data-loading", "megatron", "micro-batch", "dynamic-batch-size"]
related_files:
  - slime/backends/megatron_utils/data.py
  - slime/backends/megatron_utils/data.py
---

# Question

`backends/megatron_utils/data.py` 的 `get_data_iterator` 在干什么，详细讲一下。

# Answer

## 一句话总结

`get_data_iterator` 是训练数据加载的核心函数，负责将 rollout 数据分割成 micro-batch，支持**固定 batch size**和**动态 batch size**两种模式，并通过**序列长度均衡**算法平衡 GPU 负载。

## 逐行详解

### 第 1 部分：获取分布式配置

```python
dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
dp_group = mpu.get_data_parallel_group()
vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
if vpp_size is None:
    vpp_size = 1
if vpp_size > 1:
    from megatron.core.utils import get_model_config
    config = get_model_config(model[0])
    microbatch_group_size_per_vp_stage = config.microbatch_group_size_per_vp_stage
cp_size = mpu.get_context_parallel_world_size()
```

**每行作用**：
- `dp_size`: 获取 Data Parallel 大小（不含 Context Parallel）
- `dp_group`: 获取 NCCL 通信组，用于后续 AllReduce
- `vpp_size`: 获取 Virtual Pipeline Parallelism 大小（默认 1）
- `microbatch_group_size_per_vp_stage`: VPP 要求的 microbatch 组大小，用于对齐
- `cp_size`: Context Parallel 大小，影响 token 计算

### 第 2 部分：计算基本参数

```python
num_local_samples = len(rollout_data["total_lengths"])
global_batch_size = rollout_data.get("dynamic_global_batch_size", args.global_batch_size)
num_local_gbs = global_batch_size // dp_size
num_steps_per_rollout = num_local_samples // num_local_gbs
```

**计算逻辑**：
- `global_batch_size` 可能是动态的（来自 `_compute_dynamic_global_batch_size`）
- `num_local_gbs`: 每个 DP rank 要处理的样本数
- `num_steps_per_rollout`: 每个 rollout 需要多少训练 step 才能处理完

**示例**：GBS=64, DP=4 → 每个 rank 处理 16 个样本，如果共有 64 个样本，则需要 4 个 step。

### 第 3 部分：辅助函数

```python
def _generate_data_iterator(rollout_data, micro_batch_size, micro_batch_indices=None):
    data_iterator = []
    for _ in range(vpp_size):
        data_iterator.append(DataIterator(rollout_data, micro_batch_size, micro_batch_indices))
    return data_iterator
```

为每个 Virtual Pipeline Stage 创建一个 `DataIterator`。VPP 将模型层进一步细分，每个 stage 需要独立的数据迭代器。

### 第 4 部分：固定 Micro-batch Size 模式

```python
if not args.use_dynamic_batch_size:
    num_microbatches = [num_local_gbs // args.micro_batch_size for _ in range(num_steps_per_rollout)]
    data_iterator = _generate_data_iterator(rollout_data, args.micro_batch_size)
```

**逻辑**：
- 每个 step 使用相同的 microbatch 数
- 所有 microbatch 大小固定为 `micro_batch_size`
- **缺点**：序列长度差异大时，某些 microbatch 可能 OOM

### 第 5 部分：动态 Batch Size 模式（核心）

#### 5.1 计算每个 step 需要的 microbatch 数

```python
assert args.max_tokens_per_gpu is not None
samples = rollout_data["total_lengths"]

num_microbatches = []
for i in range(num_steps_per_rollout):
    start, end = i * num_local_gbs, (i + 1) * num_local_gbs
    num_microbatches.append(
        get_minimum_num_micro_batch_size(samples[start:end], args.max_tokens_per_gpu * cp_size)
    )
```

**`get_minimum_num_micro_batch_size` 作用**：
- 输入：样本序列长度列表、每个 GPU 最大 token 数
- 输出：最少需要多少个 microbatch 才能不 OOM
- 算法：根据最长序列计算，确保每个 microbatch 的 token 数不超过限制

**示例**：16 个样本，有的长有的短，限制 4096 tokens/GPU，可能需要分成 8 个 microbatch。

#### 5.2 跨 DP 同步（AllReduce MAX）

```python
num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=dp_group)
```

**为什么需要同步？**
- 不同 DP rank 的样本长度分布不同
- 但梯度 AllReduce 要求所有 rank 使用相同的 schedule
- 取 MAX 确保最慢的 rank 也能完成

#### 5.3 VPP 对齐调整

```python
if vpp_size > 1:
    num_microbatches = torch.clamp(
        num_microbatches // microbatch_group_size_per_vp_stage * microbatch_group_size_per_vp_stage,
        min=1,
    )
```

VPP 要求 microbatch 数能被 `microbatch_group_size_per_vp_stage` 整除，否则流水线会卡住。

#### 5.4 序列长度均衡（Sequence Length Balancing）

```python
num_microbatches = num_microbatches.tolist()

micro_batch_indices = []
for i, num_mbs in enumerate(num_microbatches):
    start, end = i * num_local_gbs, (i + 1) * num_local_gbs
    samples = rollout_data["total_lengths"][start:end]
    
    # 关键：使用序列长度均衡算法
    partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)
    
    # 调整索引到全局范围
    for j in range(num_mbs):
        for k in range(len(partitions[j])):
            partitions[j][k] += start
    micro_batch_indices.extend(partitions)

assert len(set(sum(micro_batch_indices, []))) == num_local_samples

data_iterator = _generate_data_iterator(rollout_data, None, micro_batch_indices)
```

**`get_seqlen_balanced_partitions` 作用**：
- 类似装箱问题（Bin Packing）
- 将长短序列均匀分配到不同 microbatch
- 使得每个 microbatch 的总 token 数接近
- 避免某些 GPU 因序列过长而过载

**验证**：最后的 assert 确保所有样本都被覆盖且不重复。

## 关键流程图

```
┌──────────────────────────────────────────────────────────────┐
│                    get_data_iterator                         │
├──────────────────────────────────────────────────────────────┤
│ 1. 获取并行配置 (DP/VPP/CP)                                   │
├──────────────────────────────────────────────────────────────┤
│ 2. 计算基本参数                                               │
│    num_local_gbs = GBS // DP                                 │
│    num_steps_per_rollout = samples // num_local_gbs          │
├──────────────────────────────────────────────────────────────┤
│ 3. 固定模式？                                                 │
│    YES: num_mbs = [num_local_gbs // MBS] * steps             │
│    NO:  动态计算（继续第4步）                                  │
├──────────────────────────────────────────────────────────────┤
│ 4. 动态模式：为每个 step 计算 microbatch 数                    │
│    for each step:                                            │
│      num_mbs = get_minimum_num_micro_batch_size(...)         │
├──────────────────────────────────────────────────────────────┤
│ 5. 跨 DP 同步（AllReduce MAX）                               │
│    确保所有 DP rank 使用相同 schedule                          │
├──────────────────────────────────────────────────────────────┤
│ 6. VPP 对齐（如启用）                                         │
├──────────────────────────────────────────────────────────────┤
│ 7. 序列长度均衡                                               │
│    partitions = get_seqlen_balanced_partitions(...)          │
├──────────────────────────────────────────────────────────────┤
│ 8. 返回 DataIterator 列表和 num_microbatches                 │
└──────────────────────────────────────────────────────────────┘
```

## Key Points

1. **两种模式**：固定 micro-batch size 简单但可能 OOM；动态模式根据序列长度自适应

2. **AllReduce MAX 同步**：不同 DP rank 的样本分布不同，但必须使用相同 schedule 才能梯度同步

3. **序列长度均衡**：通过装箱算法平衡不同 microbatch 的 token 数，最大化 GPU 利用率

4. **VPP 对齐**：Virtual Pipeline Parallelism 要求 microbatch 数满足特定整除条件

## Code References

- `slime/backends/megatron_utils/data.py:290` - `get_data_iterator` 函数定义
- `slime/backends/megatron_utils/data.py:226` - `DataIterator` 类定义
- `slime/utils/seqlen_balancing.py` - `get_seqlen_balanced_partitions` 实现
- `slime/utils/data.py` - `get_minimum_num_micro_batch_size` 实现

## Follow-up Questions

- [ ] `get_seqlen_balanced_partitions` 的具体装箱算法是什么？
- [ ] `get_minimum_num_micro_batch_size` 如何计算最小 microbatch 数？
- [ ] DataIterator 的 `get_next` 方法如何在 VPP 场景下工作？
