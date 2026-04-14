---
date: 2026-04-14
question_id: Q022
topics: ["off-policy", "tis", "importance-sampling", "training", "mismatch"]
related_files:
  - slime/backends/megatron_utils/loss.py
  - slime/utils/arguments.py
  - examples/train_infer_mismatch_helper/README.md
  - examples/train_infer_mismatch_helper/mis.py
---

# Question

Fully async 模式会加重 off-policy 程度吧？怎么处理的？

# Answer

## 一句话总结

Slime 通过 **TIS (Truncated Importance Sampling)** 和 **多种 Rollout Correction 算法** 来处理 off-policy 问题，同时提供 `--update-weights-interval` 参数控制权重更新频率以平衡效率与稳定性。

## Off-Policy 问题来源

在 fully async 模式下，问题更加严重：

```
Timeline:
T0: Worker 开始生成 Rollout A (使用 Policy θ₀)
T1: 训练 Policy 更新为 θ₁
T2: 训练 Policy 更新为 θ₂
T3: Rollout A 完成，进入队列
T4: 训练进程使用 θ₃ 训练 Rollout A  ← 严重 off-policy！
```

**Off-policy 程度**：`policy lag = 当前训练步数 - rollout 生成时的策略版本`

## Slime 的处理方式

### 1. 控制权重更新频率

`train_async.py:68-73`

```python
if (rollout_id + 1) % args.update_weights_interval == 0:
    # 同步等待当前的生成完成，防止更新权重时还有生成在进行
    rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
    rollout_data_next_future = None
    if not args.critic_train_only:
        actor_model.update_weights()
```

**参数**：`--update-weights-interval`
- `1`: 每轮都更新（同步程度高，off-policy 程度低）
- `>1`: 每隔 N 轮更新（吞吐量高，但 off-policy 更严重）

### 2. TIS (Truncated Importance Sampling)

**核心思想**：用重要性采样权重修正 off-policy 带来的偏差

**算法公式** (`loss.py:563-582`):

```python
def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    
    # 计算重要性采样比率: π_θ / π_rollout
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    
    # 截断到合理范围 [tis_clip_low, tis_clip]
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    
    # 将 IS 权重乘到 policy gradient loss 上
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics
```

**数学表达**:

$$
w_t = \text{clip}\left(\exp(\log \pi_\theta - \log \pi_{\text{rollout}}), C_{\text{low}}, C_{\text{high}}\right)
$$

```
标准 PPO:    L = -E[min(ratio * A, clip(ratio) * A)]
TIS 修正:   L = -E[min(ratio * A, clip(ratio) * A) * w_t]
                         ↑ 加入 IS 权重
```

### 3. 三种 Rollout Correction 算法

`examples/train_infer_mismatch_helper/README.md:30-106`

| 算法 | `use_rollout_logprobs` | `use_rollout_correction` | 说明 |
|------|------------------------|-------------------------|------|
| **标准 PPO** | False | False | 2个策略 (π_θ, π_old)，每次训练重新计算 log_probs |
| **Bypass PPO** | True | False | 2个策略 (π_θ, π_sglang)，直接使用 rollout log_probs 作为 old policy，跳过重计算 |
| **Decoupled PPO** | False | True | 3个策略 (π_θ, π_old, π_sglang)，使用重要性采样修正 |

**Decoupled PPO 公式**:

$$
L_{\text{PPO-decoupled}} = -\mathbb{E}\left[ \frac{\pi_{\text{old}}}{\pi_{\text{sglang}}} \min\left( \frac{\pi_\theta}{\pi_{\text{old}}} A, \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}\right) A \right) \right]
$$

**优势**：
- 实现 batch size 不变性
- 支持高效利用 stale data
- 精确的 off-policy 监控

### 4. ICE (Importance Sampling with Catastrophic Exclusion)

`loss.py:587-612`

```python
def icepop_function(...):
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    # 如果 ratio 超出范围，直接置零（拒绝采样风格）
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), 
        ice_ratio, 
        torch.zeros_like(ice_ratio)  # 超出范围就拒绝
    )
    pg_loss = pg_loss * ice_weight
```

### 5. 多级别 IS 计算

`README.md:128-150`

| 级别 | 计算方式 | 特点 |
|------|---------|------|
| **Token** | $w_i = \exp(\log \pi_{\text{train}} - \log \pi_{\text{rollout}})$ | 有偏但计算简单，适合大多数场景 |
| **Sequence** | $w_{\text{seq}} = \exp(\sum_i (\log \pi_{\text{train}} - \log \pi_{\text{rollout}}))$ | 无偏但高方差，适合序列级优化 |
| **Geometric** | $w_{\text{seq}} = \exp(\frac{1}{n} \sum_i (\log \pi_{\text{train}} - \log \pi_{\text{rollout}}))$ | 有偏但低方差，平衡方案 |

### 6. 自动监控指标

`README.md:175-194`

| 指标 | 描述 |
|------|------|
| `mismatch_kl` | Forward KL divergence |
| `mismatch_training_ppl` | 训练策略的 perplexity |
| `mismatch_rollout_ppl` | Rollout 策略的 perplexity |
| `tis` | Importance sampling ratio |
| `tis_clipfrac` | 被截断的 IS 权重比例 |

## 配置使用

### 启用 TIS

```bash
python train_async.py \
    --use-tis \
    --tis-clip 2.0 \
    --tis-clip-low 0.0 \
    --tis-mode truncate \
    --tis-level token
```

### 启用 Decoupled PPO (推荐用于 fully async)

```bash
python train_async.py \
    --use-tis \
    --use-rollout-correction \
    --tis-mode truncate \
    --tis-level geometric \
    --update-weights-interval 5
```

### 仅监控 mismatch（不修正）

```bash
python train_async.py \
    --get-mismatch-metrics \
    --custom-tis-function-path examples/train_infer_mismatch_helper/mis.py:compute_mis_weights_with_cp
```

## Fully Async 特有的挑战

**挑战 1: 无法控制 rollout 生成时机**

```python
# 在 standard async 中，可以控制：
rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)
# 生成和训练是成对出现的

# 在 fully async 中：
worker = get_global_worker(args, data_buffer)  # 持续在后台运行
completed = worker.get_completed_groups()  # 不知道是什么时候生成的
```

**应对**：
- 使用更保守的 TIS clip 范围
- 增加 `update_weights_interval` 减少权重更新频率（反而减轻 off-policy）
- 监控 `tis_abs` 指标，如果过大则降低 async 程度

**挑战 2: 队列积压**

```python
# 如果训练比生成慢，队列会积压大量 old rollout
while len(data) < target_data_size:
    completed = worker.get_completed_groups()  # 可能拿到很旧的 rollout
```

**应对**：
- 设置队列大小上限 `output_queue = queue.Queue(maxsize=1000)`
- 使用 `--use-dynamic-batch-size` 自适应调节

## 最佳实践

| 场景 | 推荐配置 |
|------|----------|
| 追求极致吞吐 | fully_async + TIS + update_weights_interval=5~10 |
| 平衡效率与稳定 | semi_async (train_async.py) + TIS + update_weights_interval=1 |
| 严格 on-policy | sync (train.py) + standard PPO |

## Key Points

1. **Off-policy 不可避免**：fully async 天生会加重 off-policy 程度，因为 rollout 和训练完全解耦
2. **TIS 是核心手段**：通过重要性采样权重修正 off-policy 偏差，配合 clip 防止方差爆炸
3. **多种算法可选**：从简单的 bypass PPO 到复杂的 decoupled PPO，根据需求选择
4. **监控是关键**：通过 `mismatch_kl`、`tis` 等指标监控 off-policy 程度，及时调整
5. **权衡取舍**：fully async 的高吞吐 vs 增加的 off-policy 风险，需要根据任务权衡

## Code References

- `slime/backends/megatron_utils/loss.py:563` - `vanilla_tis_function`
- `slime/backends/megatron_utils/loss.py:587` - `icepop_function`
- `slime/backends/megatron_utils/loss.py:711` - TIS 在 `policy_loss_function` 中的应用
- `slime/utils/arguments.py:902` - TIS 相关参数定义
- `train_async.py:68` - `update_weights_interval` 控制
- `examples/train_infer_mismatch_helper/README.md` - 完整算法文档

## Follow-up Questions

- [ ] 如何根据 `tis_abs` 指标动态调整 async 程度？
- [ ] Rejection Sampling (RS) 和 TIS 的适用场景有何不同？
- [ ] Veto Mechanism 在什么情况下会触发？
