#!/bin/bash

pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
sleep 2

set -ex

export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen3-4B-Instruct-2507.sh"

export BROWSECOMP_PLUS_ROOT="${BROWSECOMP_PLUS_ROOT:-/root/browsecomp-plus}"
export BROWSECOMP_PLUS_DATA_DIR="${BROWSECOMP_PLUS_DATA_DIR:-/root/browsecomp-plus-processed}"
export BROWSECOMP_PLUS_CORPUS_ROOT="${BROWSECOMP_PLUS_CORPUS_ROOT:-${BROWSECOMP_PLUS_ROOT}}"
export BROWSECOMP_PLUS_MAX_TOOL_CALLS=20
export BROWSECOMP_PLUS_TOPK="${BROWSECOMP_PLUS_TOPK:-5}"

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B-Thinking-2507/
   --ref-load /root/Qwen3-4B-Thinking-2507_torch_dist/
   --save /root/Qwen3-4B-Thinking-2507_browsecomp_plus_grpo/
)

ROLLOUT_ARGS=(
   --prompt-data "${BROWSECOMP_PLUS_DATA_DIR}/train.parquet"
   --input-key prompt
   --label-key reward_model
   --metadata-key metadata
   --tool-key tools
   --rollout-shuffle
   --num-rollout 25
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 2048
   --rollout-temperature 0.7
   --eval-interval 5
   --eval-prompt-data browsecomp_plus_eval "${BROWSECOMP_PLUS_DATA_DIR}/eval.parquet"
   --eval-input-key prompt
   --eval-label-key reward_model
   --eval-tool-key tools
   --n-samples-per-eval-prompt 1
   --global-batch-size 256
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.browsecomp_plus.generate_with_browsecomp_plus.generate
   --custom-rm-path examples.browsecomp_plus.generate_with_browsecomp_plus.reward_func
)

ray start --head --node-ip-address "${MASTER_ADDR:-127.0.0.1}" --num-gpus 4 --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"BROWSECOMP_PLUS_ROOT\": \"${BROWSECOMP_PLUS_ROOT}\",
    \"BROWSECOMP_PLUS_DATA_DIR\": \"${BROWSECOMP_PLUS_DATA_DIR}\",
    \"BROWSECOMP_PLUS_CORPUS_ROOT\": \"${BROWSECOMP_PLUS_CORPUS_ROOT}\",
    \"BROWSECOMP_PLUS_MAX_TOOL_CALLS\": \"${BROWSECOMP_PLUS_MAX_TOOL_CALLS}\",
    \"BROWSECOMP_PLUS_TOPK\": \"${BROWSECOMP_PLUS_TOPK}\"
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
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
