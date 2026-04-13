---
date: 2026-04-13
question_id: Q015
topics: ["training", "megatron", "data-processing", "routing-replay"]
related_files:
  - slime/backends/megatron_utils/actor.py
  - slime/backends/megatron_utils/data.py
  - slime/utils/data.py
---

# Question

详细讲解 `MegatronTrainRayActor` 里面的 `_get_rollout_data`、`process_rollout_data`、`train_actor`、`get_data_iterator`、`fill_routing_replay` 这些函数每行都在干什么。

# Answer

## 1. `_get_rollout_data` - 数据预处理和 GPU 传输

**文件**: `slime/backends/megatron_utils/actor.py:187-260`

```python
def _get_rollout_data(self, rollout_data_ref: Box) -> RolloutBatch:
    # 从 Ray 对象存储获取数据（在 CPU 上）
    # PP 的第一阶段和最后阶段都会收到数据
    rollout_data = process_rollout_data(
        self.args,
        rollout_data_ref,
        mpu.get_data_parallel_rank(with_context_parallel=False),      # 当前 DP rank
        mpu.get_data_parallel_world_size(with_context_parallel=False), # DP 总大小
    )
    
    # 将 tokens 提前移到 GPU（避免训练时重复传输）
    rollout_data["tokens"] = [
        torch.tensor(t, dtype=torch.long, device=torch.cuda.current_device())
        for t in rollout_data["tokens"]
    ]
    
    # 将 loss_masks 移到 GPU
    rollout_data["loss_masks"] = [
        torch.tensor(t, dtype=torch.int, device=torch.cuda.current_device())
        for t in rollout_data["loss_masks"]
    ]
    
    # 多模态输入处理（如果有）
    if "multimodal_train_inputs" in rollout_data:
        rollout_data["multimodal_train_inputs"] = [
            {
                key: (
                    torch.from_numpy(v.copy()).to(device=torch.cuda.current_device())
                    if isinstance(v, np.ndarray)
                    else v.to(device=torch.cuda.current_device())
                )
                for key, v in mm_dict.items()
            }
            if mm_dict is not None else None
            for mm_dict in rollout_data["multimodal_train_inputs"]
        ]

    # bshd 格式（batch-sequence-head-dim）的特殊处理
    """
    这是 bshd 格式（Batch-Sequence-Head-Dim）
  的特殊处理，用于将变长序列对齐到固定长度。逐行解释：

  逐行解释

  if self.args.qkv_format == "bshd":
      # bshd = Batch-Sequence-Head-Dim，是 Megatron 的一种数据布局格式
      # 与 "thd"（Token-Head-Dim，变长）相对，bshd 要求固定序列长度

      # 1. 获取这批数据中的最大序列长度
      max_seq_len = max(rollout_data["total_lengths"])
      # 例: [100, 150, 120, 200] -> max_seq_len = 200

      # 2. 计算 padding 的步长（对齐粒度）
      # TP (Tensor Parallel) size * 乘数，假设 TP=4, multiplier=128
      # pad_size = 4 * 128 = 512
      pad_size = mpu.get_tensor_model_parallel_world_size() *
  self.args.data_pad_size_multiplier

      # 3. 将 max_seq_len 向上取整到 pad_size 的倍数
      # 数学公式: ceil(max_seq_len / pad_size) * pad_size
      # 例: (200 + 512 - 1) // 512 * 512 = 711 // 512 * 512 = 1 * 512 = 512
      max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size

      # 4. 为每个样本设置相同的最大序列长度
      # 所有样本都要 padding 到这个长度才能组成 batch
      # 例: 4 个样本都设置 max_seq_len=512
      rollout_data["max_seq_lens"] = [max_seq_len] * len(rollout_data["tokens"])

  为什么要这样做？

  1. bshd 格式的要求

  ┌──────┬──────────────────────────┬─────────────────────────────┐
  │ 格式 │           特点           │          适用场景           │
  ├──────┼──────────────────────────┼─────────────────────────────┤
  │ thd  │ 变长序列，token 维度打包 │ 高效，支持 Context Parallel │
  ├──────┼──────────────────────────┼─────────────────────────────┤
  │ bshd │ 固定长度，矩阵运算友好   │ 某些 kernel 优化，简单实现  │
  └──────┴──────────────────────────┴─────────────────────────────┘

  bshd 要求所有样本在 batch 维度对齐：
  # bshd 格式: [batch_size, seq_len, num_heads, head_dim]
  # 所有样本必须有相同的 seq_len

  # 例: 4 个样本，都 padding 到 512
  tokens.shape = [4, 512, hidden_dim]

  2. Padding 到 TP size 的倍数

  假设:
  - TP (Tensor Parallel) size = 4
  - data_pad_size_multiplier = 128
  - pad_size = 512

  原始序列长度: [100, 150, 120, 200]
  最大长度: 200
  对齐后长度: 512  (200 -> 512，向上取整到 512 的倍数)

  为什么对齐到 TP size 的倍数?
  ┌─────────────────────────────────────┐
  │  序列被分成 TP=4 份，每份给1个 GPU   │
  │  如果长度不是4的倍数，无法均分        │
  │  512 / 4 = 128，每个 GPU 处理 128 个 │
  │  token，没有浪费                     │
  └─────────────────────────────────────┘

  3. 内存碎片优化

  不对齐的情况:
  - 序列1: 长度 100 -> padding 到 512，浪费 412
  - 但内存分配器可能分配 100, 150, 120, 200 等不同大小的块
  - 导致内存碎片

  对齐后:
  - 所有 batch 都按 512 的倍数分配
  - 内存分配规整，减少碎片，缓存命中率高

  对比 thd 格式

  # thd 格式（变长，更灵活）
  tokens = [[t1, t2, ...], [t1, t2, ...], ...]  # 每个样本真实长度
  # 使用 cu_seqlens 标记边界，支持不同长度

  # bshd 格式（固定长度，简单）
  tokens = [
      [t1, t2, ..., pad, pad, pad],  # 长度固定为 512
      [t1, t2, ..., pad, pad, pad],
      ...
  ]

  实际例子

  # 4 个样本，真实长度 [100, 150, 120, 200]
  rollout_data["total_lengths"] = [100, 150, 120, 200]

  # 经过这段代码处理
  max_seq_len = 200
  pad_size = 512  # TP=4 * multiplier=128
  max_seq_len = 512  # 向上取整

  rollout_data["max_seq_lens"] = [512, 512, 512, 512]

  # 后续 get_batch 会 padding:
  # sample 0: 100 tokens + 412 pad tokens = 512
  # sample 1: 150 tokens + 362 pad tokens = 512
  # sample 2: 120 tokens + 392 pad tokens = 512
  # sample 3: 200 tokens + 312 pad tokens = 512

  ---
  一句话总结：这段代码将变长序列对齐到固定的、TP size 倍数的最大长度，以满足
  bshd 格式要求并优化内存分配。
    """
    if self.args.qkv_format == "bshd":
        # 获取最大序列长度
        max_seq_len = max(rollout_data["total_lengths"])
        
        # 对齐到 pad_size 的倍数（减少内存碎片）
        pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
        max_seq_len = (max_seq_len + pad_size - 1) // pad_size * pad_size
        
        rollout_data["max_seq_lens"] = [max_seq_len] * len(rollout_data["tokens"])

    # 处理 rollout 时的 log_probs（用于 PPO off-policy 修正）
    for key in ["rollout_log_probs", "teacher_log_probs"]:
        if key not in rollout_data:
            continue
        rollout_data[key] = [
            torch.tensor(
                slice_log_prob_with_cp(  # Context Parallel 切片
                    log_prob,
                    total_length,
                    response_length,
                    self.args.qkv_format,
                    rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                ),
                device=torch.cuda.current_device(),
                dtype=torch.float32,
            )
            for i, (log_prob, total_length, response_length) in enumerate(...)
        ]
    """
    slice_log_prob_with_cp 详解

  文件: slime/backends/megatron_utils/cp_utils.py:219

  这个函数用于处理 Context Parallel (CP) 场景下的 log probabilities 切片。

  背景：Context Parallel 的数据分布

  在 CP 模式下，序列被交错分割成两个 chunk 分布到不同 GPU：

  序列: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  (共10个token)
  CP size = 4

  分割方式（环形/交错）:
  Chunk 0: GPU 0: [0, 1], GPU 1: [2, 3], GPU 2: [4, 5], GPU 3: [6, 7]
  Chunk 1: GPU 0: [9, 8] (逆序), GPU 1: [7, 6], ...

  实际分布:
  GPU 0: [0, 1] + [8, 9]  (来自序列的两端)
  GPU 1: [2, 3] + [6, 7]
  ...

  函数作用

  slice_log_prob_with_cp(
      log_prob,        # rollout 时计算的 log_prob 列表
      total_length,    # prompt + response 总长度
      response_length, # response 长度
      qkv_format,      # "thd" 或 "bshd"
      max_token_len,   # bshd 格式下的最大长度
  )

  为什么要切片？

  1. Rollout 时：在 SGLang 中生成，log_prob 是连续的 [response_length]
  2. 训练时：在 Megatron 中使用 CP，log_prob 需要对应到当前 GPU 持有的 token
  3. 问题：CP 的 token 分布是交错的（两个不连续的 chunk），需要提取对应的
  log_prob

  代码逐行解释

  def slice_log_prob_with_cp(log_prob, total_length, response_length, 
  qkv_format, max_token_len):
      assert len(log_prob) == response_length  # 验证长度一致

      cp_size = mpu.get_context_parallel_world_size()

      if cp_size == 1:
          # 没有 CP，直接返回原始 log_prob
          return log_prob

      # 计算 prompt 长度
      prompt_length = total_length - response_length

      # 获取当前 rank 应该处理的 token 偏移量
      _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(
          total_length, response_length, qkv_format, max_token_len
      )

      # 从 log_prob 中提取两个 chunk 对应的部分
      # 注意：log_prob 只包含 response 部分，需要减去 prompt_length 来对齐索引
      chunk_1 = log_prob[logits_offset[0][0] - (prompt_length - 1) :
  logits_offset[0][1] - (prompt_length - 1)]
      chunk_2 = log_prob[logits_offset[1][0] - (prompt_length - 1) :
  logits_offset[1][1] - (prompt_length - 1)]

      # 合并两个 chunk
      if isinstance(log_prob, list):
          return chunk_1 + chunk_2
      else:
          return torch.cat([chunk_1, chunk_2], dim=0)

  具体例子

  假设:
  - total_length = 100 (prompt=80, response=20)
  - CP size = 4
  - 当前 rank = 0

  get_logits_and_tokens_offset_with_cp 返回:
  - logits_offset = [(80, 85), (95, 100)]  # 当前 GPU 应该计算的 logits 范围

  原始 log_prob (长度为 20，对应 response token 81-100):
  log_prob = [lp_81, lp_82, ..., lp_100]

  切片计算:
  - chunk_1 = log_prob[80 - 79 : 85 - 79] = log_prob[1:6]   → [lp_81..lp_85]
  - chunk_2 = log_prob[95 - 79 : 100 - 79] = log_prob[16:21] → [lp_96..lp_100]

  结果: [lp_81, ..., lp_85, lp_96, ..., lp_100]

  总结

  ┌──────┬───────────────────────────────────────────────────────┐
  │ 场景 │                       处理方式                        │
  ├──────┼───────────────────────────────────────────────────────┤
  │ CP=1 │ 直接返回原 log_prob                                   │
  ├──────┼───────────────────────────────────────────────────────┤
  │ CP>1 │ 根据当前 rank 的 token 分布，提取对应的 log_prob 切片 │
  └──────┴───────────────────────────────────────────────────────┘

  目的：让 Megatron 训练时的 log_prob 与 CP 分布对齐，用于计算 PPO/GRPO 的
  off-policy 修正（ratio = exp(current_log_prob - rollout_log_prob))。

✻ Cogitated for 1m 31s

❯ 为什么CP是按照这种怪方式去分的？为什么还需要正过来反过来                      

⏺ 这是 Ring Attention (环形注意力) 的通信优化策略。这种"怪异"的分割方式是为了让
  KV Cache 的通信最高效。

  标准分割 vs Ring Attention 分割

  1. 如果简单连续分割

  序列: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  (12 tokens)
  CP=4，简单分割:

  GPU 0: [0, 1, 2]      GPU 1: [3, 4, 5]
  GPU 2: [6, 7, 8]      GPU 3: [9, 10, 11]

  问题: 计算 attention 时，GPU 0 的 token 2 需要看所有后面的 token (3-11)
        需要向 GPU 1, 2, 3 都发送请求，通信次数多

  2. Ring Attention 的交错分割

  同样的序列，Ring Attention 分割:

  Chunk 0 (正向):  [0,1]  [2,3]  [4,5]  [6,7]
                   GPU0   GPU1   GPU2   GPU3

  Chunk 1 (反向):  [11,10] [9,8] [7,6] [5,4]  ← 注意是逆序!
                    GPU0   GPU1  GPU2  GPU3

  最终分布:
  GPU 0: [0, 1] + [10, 11]  ← 头尾两端
  GPU 1: [2, 3] + [8, 9]
  GPU 2: [4, 5] + [6, 7]

  为什么要这样设计？

  原因 1: 负载均衡

  每个 GPU 处理的 token 数相同，且计算量相近：
  - 简单分割：GPU 0 计算 [0,1,2] 的 attention，需要看 12 个 token
  - Ring 分割：每个 GPU 处理的是分散在整个序列中的 token，计算量更均衡

  原因 2: 环形通信高效

  Ring Attention 的计算流程:

  Step 1: 每个 GPU 先用自己的 local KV
          GPU 0: 计算 [0,1] 的 attention，使用 KV([0,1])

  Step 2: 环形传递 KV (GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0)
          GPU 0 收到 GPU 3 的 KV([10,11])
          GPU 0 计算 [0,1] 的 attention，累加 KV([10,11])

  Step 3: 继续传递
          GPU 0 收到 GPU 2 的 KV([8,9])
          GPU 0 计算 [0,1] 的 attention，累加 KV([8,9])

  ...

  关键点:
  - 每个 GPU 只需要维护一个 KV block
  - 通信是单向的环形，没有冲突
  - 每个 step 所有 GPU 同时通信，充分利用带宽

  原因 3: 为什么 Chunk 1 要反过来？

  如果 Chunk 1 也正着分:
  GPU 0: [0,1] + [8,9]  → token 1 和 token 8 距离很远
                         需要很多通信步骤才能看到彼此

  如果 Chunk 1 反着分:
  GPU 0: [0,1] + [10,11] → token 1 和 token 10/11 是相邻的!
                           (在序列的两端)

  可视化:
  序列:  [0] [1] ... [8] [9] [10] [11]
         │   │       │   │   │    │
         └───┴───────┴───┴───┴────┘
          GPU 0 持有了序列的"两端"

  为什么这样好？
  - Attention 计算是双向的
  - Token 0,1 需要看到 token 10,11
  - Token 10,11 也需要看到 token 0,1
  - 放在同一个 GPU 上可以减少通信!

  直观理解

  想象一个长桌子开会：

  简单分割 = 大家坐在桌子两边，左边4人右边4人
  - 左边的人要听右边人说话，必须大喊（跨 GPU 通信）
  - 距离越远喊得越费劲（通信次数多）

  Ring 分割 = 大家围成一圈，每人拿两端的座位
  - 左边的人拿第1和第8个座位
  - 这样相邻的人就在旁边（同一个 GPU 上）
  - 只需要和左右邻居说话（环形通信）

  代码中的体现

  # cp_utils.py 中的分割逻辑
  chunk_0 = (cp_rank * chunk_size, (cp_rank + 1) * chunk_size)  # 正向
  chunk_1 = ((2 * cp_size - cp_rank - 1) * chunk_size,
             (2 * cp_size - cp_rank) * chunk_size)  # 反向（逆序）

  这种设计让 Ring Attention 的通信次数从 $O(N^2)$ 降低到
  $O(N)$，同时保持负载均衡。
    """
    # MoE 路由专家信息（用于 routing replay）
    if "rollout_routed_experts" in rollout_data:
        rollout_data["rollout_routed_experts"] = [
            torch.from_numpy(r) for r in rollout_data["rollout_routed_experts"]
        ]
    return rollout_data
```

## 2. `process_rollout_data` - 从 Ray 获取数据并分区

**文件**: `slime/utils/data.py:299-313`

```python
def process_rollout_data(args, rollout_data_ref, dp_rank, dp_size):
    # rollout_data_ref 是每个 DP rank 的数据引用列表
    assert len(rollout_data_ref) == dp_size
    
    # 获取当前 DP rank 对应的数据（从 Ray 对象存储拉到本地 CPU）
    rollout_data = ray.get(rollout_data_ref[dp_rank].inner)

    # 获取分区信息（_split_train_data_by_dp 时生成的）
    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # 保存整个 rollout batch 的序列长度（用于性能统计）
    Timer().seq_lens = total_lengths
    
    # 根据 partition 筛选出当前 DP rank 处理的样本的序列长度
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
```

**关键流程**:
```
RolloutManager._split_train_data_by_dp
    ├── ray.put(data) for each DP rank  →  存入 Ray 对象存储
    └── 返回引用列表 rollout_data_refs

MegatronTrainRayActor._get_rollout_data
    └── process_rollout_data
        └── ray.get(rollout_data_ref[dp_rank])  →  从对象存储拉取
```

## 3. `get_data_iterator` - 创建数据迭代器

**文件**: `slime/backends/megatron_utils/data.py:290-380`

```python
def get_data_iterator(args, model, rollout_data):
    # 获取并行配置
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    dp_group = mpu.get_data_parallel_group()
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
    cp_size = mpu.get_context_parallel_world_size()

    # 本地样本数
    num_local_samples = len(rollout_data["total_lengths"])
    
    # 获取 global batch size（可能是动态的）
    global_batch_size = rollout_data.get("dynamic_global_batch_size", args.global_batch_size)
    
    # 每个 DP rank 处理的本地 batch size
    num_local_gbs = global_batch_size // dp_size
    
    # 计算每个 rollout 需要多少训练步骤
    num_steps_per_rollout = num_local_samples // num_local_gbs

    # 创建数据迭代器的辅助函数
    def _generate_data_iterator(rollout_data, micro_batch_size, micro_batch_indices=None):
        data_iterator = []
        for _ in range(vpp_size):  # 每个 VPP stage 一个迭代器
            data_iterator.append(DataIterator(rollout_data, micro_batch_size, micro_batch_indices))
        return data_iterator

    # 情况1：固定 batch size
    if not args.use_dynamic_batch_size:
        # 每个 step 的 micro batch 数 = 本地 GBS / micro_batch_size
        num_microbatches = [num_local_gbs // args.micro_batch_size for _ in range(num_steps_per_rollout)]
        data_iterator = _generate_data_iterator(rollout_data, args.micro_batch_size)
    
    # 情况2：动态 batch size（根据 max_tokens_per_gpu）
    else:
        samples = rollout_data["total_lengths"]
        num_microbatches = []
        
        # 为每个 step 计算需要的 micro batch 数
        for i in range(num_steps_per_rollout):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            step_samples = samples[start:end]
            # 根据 max_tokens_per_gpu 计算最少需要多少 micro batches
            num_microbatches.append(
                get_minimum_num_micro_batch_size(step_samples, args.max_tokens_per_gpu * cp_size)
            )

        # All-reduce 取最大值（确保所有 DP rank 一致）
        num_microbatches = torch.tensor(num_microbatches, ...)
        dist.all_reduce(num_microbatches, op=dist.ReduceOp.MAX, group=dp_group)

        # VPP 要求 micro batch 数能被 vpp_size 整除
        if vpp_size > 1:
            num_microbatches = torch.clamp(
                num_microbatches // microbatch_group_size_per_vp_stage * microbatch_group_size_per_vp_stage,
                min=1,
            )

        # 平衡每个 micro batch 的序列长度
        micro_batch_indices = []
        for i, num_mbs in enumerate(num_microbatches):
            start, end = i * num_local_gbs, (i + 1) * num_local_gbs
            samples = rollout_data["total_lengths"][start:end]
            # 使用序列长度平衡算法分区
            partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)
            # 调整索引为全局索引
            for j in range(num_mbs):
                for k in range(len(partitions[j])):
                    partitions[j][k] += start
            micro_batch_indices.extend(partitions)

        data_iterator = _generate_data_iterator(rollout_data, None, micro_batch_indices)

    return data_iterator, num_microbatches
```

## 4. `train_actor` - Actor 训练主流程

**文件**: `slime/backends/megatron_utils/actor.py:406-514`

```python
def train_actor(self, rollout_id: int, rollout_data: RolloutBatch) -> None:
    # 1. 创建数据迭代器
    data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)

    # 2. 如果使用 rollout routing replay，填充路由信息
    if self.args.use_rollout_routing_replay:
        self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

    with inverse_timer("train_wait"), timer("train"):
        # 3. 计算 advantage 和 return（如果需要）
        if self.args.compute_advantages_and_returns:
            
            # 3.1 切换到 ref 模型，计算参考 log_prob（用于 KL 散度）
            if "ref" in self.weights_backuper.backup_tags:
                if self.args.use_routing_replay:
                    os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                self._switch_model("ref")
                rollout_data.update(self.compute_log_prob(data_iterator, num_microbatches, "ref_"))

            # 3.2 切换到 teacher 模型（用于蒸馏）
            if "teacher" in self.weights_backuper.backup_tags:
                if self.args.use_routing_replay:
                    os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                self._switch_model("teacher")
                rollout_data.update(self.compute_log_prob(data_iterator, num_microbatches, "teacher_"))

            # 3.3 计算当前/旧策略的 log_prob
            self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
            if not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics:
                if self.args.use_routing_replay:
                    stage = "replay_forward" if self.args.use_rollout_routing_replay else "record"
                    os.environ["ROUTING_REPLAY_STAGE"] = stage
                rollout_data.update(self.compute_log_prob(data_iterator, num_microbatches, ""))
                if self.args.use_rollout_routing_replay:
                    RoutingReplay.clear_all_forward()

            # 3.4 同步 Critic 数据（如果有 Critic）
            if self.args.use_critic:
                sync_actor_critic_data(self.args, rollout_data, self._actor_critic_groups)
            
            # 切回 actor 模型
            if self._active_model_tag != "actor":
                self._switch_model("actor")

            # 3.5 计算 advantage 和 return
            compute_advantages_and_returns(self.args, rollout_data)

        # 4. 数据后处理（如果有自定义处理函数）
        if self.rollout_data_postprocess is not None:
            self.rollout_data_postprocess(self.args, rollout_id, rollout_data)

        # 5. 记录训练数据
        log_rollout_data(rollout_id, self.args, rollout_data)

        # 6. 执行训练
        if self.args.use_routing_replay:
            os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
        with timer("actor_train"):
            train(rollout_id, self.model, self.optimizer, self.opt_param_scheduler, 
                  data_iterator, num_microbatches)

        self.prof.step(rollout_id=rollout_id)

    # 7. 保存调试数据
    train_dump_utils.save_debug_train_data(self.args, rollout_id, rollout_data)

    # 8. 清理 routing replay
    if self.args.use_routing_replay:
        RoutingReplay.clear_all()

    # 9. 备份更新后的 actor 权重
    self.weights_backuper.backup("actor")

    # 10. 定期更新 ref 模型（如果需要）
    if (self.args.ref_update_interval is not None and
        (rollout_id + 1) % self.args.ref_update_interval == 0 and
        "ref" in self.weights_backuper.backup_tags):
        self.weights_backuper.backup("ref")

    log_perf_data(rollout_id, self.args)
```

## 5. `fill_routing_replay` - MoE 路由回放

**文件**: `slime/backends/megatron_utils/actor.py:268-344`

```python
def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
    # 验证需要 rollout_routed_experts
    if "rollout_routed_experts" not in rollout_data:
        raise ValueError("rollout_routed_experts is required when use_rollout_routing_replay is set")

    # 获取 TP 信息
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    # 辅助函数：padding experts
    def pad_func(experts, pad):
        _, num_layers, topk = experts.shape
        # 创建 padding 值（循环使用专家 ID）
        pad = (torch.arange(pad * num_layers * topk, device=experts.device, dtype=experts.dtype)
               .reshape((pad, num_layers, topk)) % self.args.num_experts)
        return torch.cat([experts, pad], dim=0)

    # 遍历所有 micro batches
    for _ in range(sum(num_microbatches)):
        # 获取下一个 batch
        batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
        rollout_routed_experts = batch["rollout_routed_experts"]
        tokens = batch["tokens"]
        
        # 验证长度匹配（experts 比 tokens 少 1，因为最后一个 token 没有预测）
        for a, b in zip(rollout_routed_experts, tokens):
            assert a.shape[0] == b.shape[0] - 1

        # Padding 到序列末尾（最后一个 token 不计算 loss）
        rollout_routed_experts = [pad_func(r, 1) for r in rollout_routed_experts]
        
        # Context Parallel 切片
        rollout_routed_experts = [slice_with_cp(r, pad_func) for r in rollout_routed_experts]
        
        # 合并并 padding 到 pad_size 倍数
        rollout_routed_experts = torch.cat(rollout_routed_experts, dim=0)
        pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
        pad = (pad_size - rollout_routed_experts.size(0) % pad_size) % pad_size
        if pad != 0:
            rollout_routed_experts = pad_func(rollout_routed_experts, pad)

        # Sequence Parallel 切片
        if self.args.sequence_parallel:
            seqlen = rollout_routed_experts.size(0)
            assert seqlen % tp_size == 0
            start = seqlen // tp_size * tp_rank
            end = seqlen // tp_size * (tp_rank + 1)
            rollout_routed_experts = rollout_routed_experts[start:end]

        # 按层记录路由信息
        routing_replay_offset = 0
        for vp_stage, model in enumerate(self.model):
            config = model.module.config
            num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
            offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
            
            for layer_id in range(offset, offset + num_layers_to_build):
                # 跳过非 MoE 层（dense layer）
                if isinstance(config.moe_layer_freq, int):
                    if layer_id % config.moe_layer_freq != 0:
                        continue
                elif isinstance(config.moe_layer_freq, list):
                    if config.moe_layer_freq[layer_id] == 0:
                        continue
                
                # 记录该层的路由专家选择
                layer_routed_experts = rollout_routed_experts[:, layer_id]
                RoutingReplay.all_routing_replays[routing_replay_offset].record(layer_routed_experts)
                routing_replay_offset += 1

    # 清理已使用的数据
    del rollout_data["rollout_routed_experts"]
    
    # 重置迭代器
    for iterator in data_iterator:
        iterator.reset()
```

## 整体流程图

```
train_actor
    ├── get_data_iterator          ← 创建数据迭代器
    │   ├── 计算 num_steps_per_rollout
    │   ├── 固定/动态 micro_batch 策略
    │   └── 返回 DataIterator 列表
    │
    ├── fill_routing_replay        ← MoE 路由回放（可选）
    │   └── 遍历 micro batches，记录专家路由
    │
    ├── compute_log_prob (ref)     ← 参考模型前向
    ├── compute_log_prob (teacher) ← 教师模型前向（可选）
    ├── compute_log_prob (actor)   ← 当前/旧策略前向
    ├── sync_actor_critic_data     ← 同步 Critic 数据
    ├── compute_advantages_and_returns  ← 计算 advantage
    │
    ├── train()                    ← Megatron 训练
    │   └── 反向传播更新权重
    │
    ├── weights_backuper.backup("actor")  ← 备份新权重
    └── log_perf_data              ← 记录性能数据
```

## Key Points

1. **`_get_rollout_data`**: 从 Ray 对象存储拉取数据，提前移到 GPU，处理多模态和 CP
2. **`process_rollout_data`**: 简单的数据获取和分区筛选
3. **`get_data_iterator`**: 支持固定和动态 batch size，考虑 VPP 和序列长度平衡
4. **`train_actor`**: 完整训练流程，包括 ref/teacher/actor 多模型切换、advantage 计算、训练
5. **`fill_routing_replay`**: MoE 专用，记录 rollout 时的路由选择用于训练时回放

## Code References

- `slime/backends/megatron_utils/actor.py:187` - `_get_rollout_data`
- `slime/utils/data.py:299` - `process_rollout_data`
- `slime/backends/megatron_utils/data.py:290` - `get_data_iterator`
- `slime/backends/megatron_utils/actor.py:406` - `train_actor`
- `slime/backends/megatron_utils/actor.py:268` - `fill_routing_replay`

## Follow-up Questions

- [ ] `DataIterator` 的具体实现是怎样的？
- [ ] `compute_advantages_and_returns` 的具体算法？
- [ ] `RoutingReplay` 的工作原理和用途？
- [ ] `slice_with_cp` 如何处理 Context Parallel？
