# BrowseComp-Plus

This example implements two GRPO variants for BrowseComp-Plus style web-retrieval training in `slime/examples`.

The rollout follows the current Qwen3 tool-calling pattern used by recent examples in this repo:

- prompt stored as conversation messages
- tools passed through dataset `tool_key`
- assistant emits native function calls
- tool results are appended as `role="tool"` messages
- policy loss only applies to assistant-generated tokens

- `run_qwen3_4b_thinking_2507_grpo.sh`: outcome-only GRPO.
- `run_qwen3_4b_thinking_2507_hit_weighted.sh`: same reward, but scales policy-gradient loss on search turns by whether that turn retrieved a gold document.

Both variants use:

- `Qwen3-4B-Thinking-2507`
- BM25-only retrieval over the provided local corpus
- max `20` search tool calls
- `tp=1`, `dp=4` during rollout/training launch
- `--max-tokens-per-gpu 8192`
- `--context-parallel-size 2`

## Data Preparation

Prepare a `train.parquet` / `eval.parquet` split from the BrowseComp-Plus training shards:

```bash
cd /root/slime
python examples/browsecomp_plus/prepare_dataset.py \
  --dataset-root /root/browsecomp-plus \
  --output-dir /root/browsecomp-plus-processed
```

The split is deterministic: all training shards are merged, sorted by a stable hash of `query_id`, then split `90/10`.
Each row contains:

- `prompt`: conversation messages
- `reward_model`: gold answer and gold doc ids
- `metadata`: auxiliary fields
- `tools`: Qwen3-compatible search tool schema

## Required Packages

Install these extra Python packages in the training environment:

```bash
pip install rank-bm25 pandas pyarrow
```

## Reward

The exact-match answer reward is binary:

- `acc_reward = 1` if the final `<answer>` exactly matches the gold answer after normalization
- otherwise `0`

The final reward is scheduled by training rollout step:

- for training `rollout_id` `0..4`: `0.8 * acc_reward + 0.2 * length_bonus`
- for later training rollouts: `0.9 * acc_reward + 0.1 * length_bonus`
- for eval: `0.9 * acc_reward + 0.1 * length_bonus`

where `length_bonus = max(0, 1 - tool_calls / 20)`.

## Turn-Hit Weighted Variant

The second variant keeps the same final reward, but multiplies token-level policy loss on each generated search turn by:

- `1.0` if that turn's BM25 results hit any gold document
- `0.5` otherwise

Tool-result observation tokens remain masked out from policy loss.
