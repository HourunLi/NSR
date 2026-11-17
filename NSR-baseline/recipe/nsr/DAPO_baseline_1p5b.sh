#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
WORKING_DIR="$(pwd)"

RAY_DATA_HOME="${RAY_DATA_HOME:-${NSR_ROOT:-/home/lihourun/hsc/NSR-work}}"
MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/models/Qwen2.5-Math-1.5B}"
TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/dapo-math-17k.parquet}"
TEST_FILE="${TEST_FILE:-${RAY_DATA_HOME}/data/aime-2024.parquet}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
NNODES="${NNODES:-1}"
# 共用同一份小模型配置，NSR=true 时启用 NSR。
NSR="${NSR:-false}"
STEPS="${STEPS:-300}"
# 迭代默认值，可调整；输入与回答上限之和不能超过原生上下文。
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
EXP_NAME="${EXP_NAME:-small-nsr${NSR}_$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-${RAY_DATA_HOME}/paper-runs/${EXP_NAME}}"
VAL_ONLY="${VAL_ONLY:-false}"
# 当前权重同步接口使用 vLLM V0。
export VLLM_USE_V1=0
# 单机使用独立 Ray；不设置 rollout.seed，避免同题多次采样重复。
export RAY_ADDRESS=local

mkdir -p "${RUN_DIR}/validation" "${RUN_DIR}/checkpoints"
echo "Working directory: ${WORKING_DIR}"
echo "Run directory: ${RUN_DIR}"

python3 - "${MODEL_PATH}/config.json" "${TEST_FILE}" "${MAX_PROMPT_LENGTH}" "${MAX_RESPONSE_LENGTH}" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

model_config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    model_config.get("max_position_embeddings") != 4096
    or float(model_config.get("rope_theta", 0)) != 10000.0
    or model_config.get("rope_scaling") is not None
):
    raise SystemExit("MODEL_PATH must point to the original Qwen2.5-Math-1.5B 4096-token config")
prompt_length, response_length = map(int, sys.argv[3:5])
if prompt_length < 1 or response_length <= 512 or prompt_length + response_length > model_config["max_position_embeddings"]:
    raise SystemExit("Need prompt > 0, response > overlong buffer (512), and prompt + response <= model context")
if pq.ParquetFile(sys.argv[2]).metadata.num_rows != 960:
    raise SystemExit("TEST_FILE must contain 30 AIME questions repeated 32 times (960 rows)")
PY

python3 -m recipe.dapo.main_dapo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.gen_batch_size=96 \
    data.train_batch_size=32 \
    data.filter_overlong_prompts=false \
    ++data.cache_dir="${RAY_DATA_HOME}/cache/verl/rlhf" \
    data.custom_cls.path=pkg://recipe.nsr.answer_format_dataset \
    data.custom_cls.name=AnswerFormatDataset \
    ++data.seed=1 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.NSR="${NSR}" \
    actor_rollout_ref.actor.nsr_delta=0.1 \
    actor_rollout_ref.actor.use_random_adv=false \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=6144 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.checkpoint.save_contents='["model","optimizer","extra","hf_model"]' \
    actor_rollout_ref.actor.checkpoint.load_contents='["model","optimizer","extra"]' \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=6144 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=6144 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.filter_groups.enable=true \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_num_gen_batches=10 \
    reward_model.reward_manager=dapo \
    custom_reward_function.path="${WORKING_DIR}/recipe/nsr/answer_reward.py" \
    custom_reward_function.name=compute_score \
    reward_model.overlong_buffer.enable=true \
    reward_model.overlong_buffer.len=512 \
    reward_model.overlong_buffer.penalty_factor=1.0 \
    trainer.logger='["console"]' \
    trainer.project_name=DAPO-paper-aligned \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    ray_init.num_cpus=$((N_GPUS_PER_NODE + 2)) \
    trainer.val_before_train=true \
    trainer.val_only="${VAL_ONLY}" \
    trainer.test_freq=5 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.total_epochs=3 \
    trainer.total_training_steps="${STEPS}" \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${RUN_DIR}/checkpoints" \
    trainer.rollout_data_dir=null \
    trainer.validation_data_dir="${RUN_DIR}/validation" \
    hydra.run.dir="${RUN_DIR}/hydra"
