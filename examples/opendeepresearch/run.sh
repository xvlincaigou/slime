#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3

# raise file descriptor limit for high concurrency (search + sglang + ray)
ulimit -n 65536

# ensure CUDA is available
export CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda-12.9"}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B-Instruct-2507.sh"

CKPT_ARGS=(
   --hf-checkpoint /home/t2vg-a100-G4-13/xietian/xl/models/deepresearch-sft/
   --ref-load /home/t2vg-a100-G4-13/xietian/xl/models/deepresearch-sft_torch_dist/
   --load /home/t2vg-a100-G4-13/xietian/xl/models/deepresearch-sft/rl/
   --save /home/t2vg-a100-G4-13/xietian/xl/models/deepresearch-sft/rl/
   --save-interval 15
)

ROLLOUT_ARGS=(
   --prompt-data /home/t2vg-a100-G4-13/xietian/xl/data/opendeepresearch/train.jsonl
   --input-key query
   --label-key reference
   --rollout-shuffle
   --num-rollout 300
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1

   --eval-interval 15
   --eval-prompt-data opendeepresearch_test /home/t2vg-a100-G4-13/xietian/xl/data/drb/prompt_data/query.jsonl
   --eval-input-key prompt
   --eval-label-key topic
   --n-samples-per-eval-prompt 1

   --global-batch-size 16
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # --use-kl-loss
   # --kl-loss-coef 0.001
   # --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # whether enabling TIS
   --use-tis

   # aggregate loss per token instead of per sequence
   --calculate-per-token-loss
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project bisheruns
   --wandb-group rl_sftv7_opendeepresearch_race_tokenloss
   --wandb-key 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
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   --sglang-page-size 64 
   --sglang-enable-hierarchical-cache 
   --sglang-hicache-size 110
   --sglang-hicache-io-backend kernel 
   --sglang-hicache-write-policy write_through
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_search_race.generate
   --custom-rm-path generate_with_search_race.reward_func
   --custom-rollout-log-function-path generate_with_search_race.log_rollout
   --custom-eval-rollout-log-function-path generate_with_search_race.log_eval_rollout
)

# ===== Ray: 8 GPUs only =====
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats

# ===== Judge env vars for RACE reward =====
# Set these before running, or export in your shell:
#   export JUDGE_API_KEY=sk-or-v1-xxx
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.deepseek.com/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-sk-c375b78f6f0d4b029da133ed125e77e1}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-flash}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/home/t2vg-a100-G4-13/xietian/xl/drppl/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"CUDA_HOME\": \"/usr/local/cuda-12.9\",
    \"PATH\": \"/usr/local/cuda-12.9/bin:${PATH}\",
    \"LD_LIBRARY_PATH\": \"/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH}\",
    \"JUDGE_BASE_URL\": \"${JUDGE_BASE_URL}\",
    \"JUDGE_API_KEY\": \"${JUDGE_API_KEY}\",
    \"JUDGE_MODEL\": \"${JUDGE_MODEL}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
