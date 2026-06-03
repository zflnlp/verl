#!/usr/bin/env bash
# GRPO | Qwen2.5-2B | WebShop Environment | FSDP Training
#
# This script trains a Qwen2.5-2B model on the WebShop shopping task
# using Group Relative Policy Optimization (GRPO).
#
# Prerequisites:
# 1. Install WebShop environment:
#    pip install webshop
#    # Start WebShop server in a separate terminal:
#    python -m webshop.run_server
#
# 2. Preprocess WebShop data:
#    python examples/webshop_grpo/data_preprocess.py --local_save_dir ~/data/webshop
#
# 3. Start training:
#    bash examples/webshop_grpo/run_qwen2_5_2b_webshop_grpo.sh
#
# Note: For Qwen3.5-2B, change MODEL_PATH to the appropriate HuggingFace path.
#       As of now, you might use: Qwen/Qwen2.5-1.5B-Instruct or similar small models.

set -xeuo pipefail

########################### user-adjustable ###########################
# Model configuration
# Options:
#   - Qwen/Qwen2.5-1.5B-Instruct (recommended for small model)
#   - Qwen/Qwen2.5-3B-Instruct (if more compute available)
#   - Qwen/Qwen3-0.6B (if available)
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}

# Hardware configuration
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}  # Adjust based on your GPU setup

# Training hyperparameters (optimized for small model + WebShop)
train_batch_size=${TRAIN_BATCH_SIZE:-64}       # Smaller batch for multi-turn
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}   # Longer prompts for task description
max_response_length=${MAX_RESPONSE_LENGTH:-4096}  # Longer for multi-turn interaction
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

# Learning rate (smaller for fine-tuning small models)
actor_lr=${ACTOR_LR:-5e-7}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0.01}  # Small entropy for exploration

# Rollout configuration
rollout_n=${ROLLOUT_N:-4}  # Number of rollouts per prompt (group size)
rollout_tp=${ROLLOUT_TP:-1}  # Tensor parallelism (1 for small models)
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}

# WebShop specific
max_steps_per_task=${MAX_STEPS_PER_TASK:-10}
webshop_server=${WEBSHOP_SERVER:-http://localhost:3000}

# Training schedule
total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:-10}
test_freq=${TEST_FREQ:-5}

# Experiment tracking
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_webshop}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_2b_webshop_grpo_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

########################### derived defaults ###########################
# Auto-detect device (GPU or NPU)
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}

case "${DEVICE}" in
    gpu)
        actor_param_offload=False
        actor_optimizer_offload=False
        n_trainer_devices=${NGPUS_PER_NODE}
        ;;
    npu)
        NGPUS_PER_NODE=8
        n_trainer_devices=${NGPUS_PER_NODE}
        actor_param_offload=True
        actor_optimizer_offload=True
        ;;
esac

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$HOME/data/webshop/train.parquet']"
    data.val_files="['$HOME/data/webshop/test.parquet']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    # Use Flash Attention for efficiency
    actor_rollout_ref.model.attn_implementation=flash_attention_2
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload}
    # PPO epochs for GRPO
    actor_rollout_ref.actor.ppo_epochs=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    # Multi-turn configuration for WebShop
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.function_tool_path=examples/webshop_grpo/webshop_tools.py
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=15
    actor_rollout_ref.rollout.multi_turn.max_user_turns=1
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024
    actor_rollout_ref.rollout.multi_turn.format=hermes
    # Agent loop configuration
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.agent.num_workers=4
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

REWARD=(
    reward.custom_reward_function.path=examples/webshop_grpo/reward_function.py
    reward.custom_reward_function.name=compute_webshop_reward
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${n_trainer_devices}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    # Validation
    trainer.val_before_train=True
    trainer.val_only=False
)

########################### launch ###########################
echo "=========================================="
echo "WebShop GRPO Training Configuration"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Device: ${DEVICE}"
echo "GPUs: ${NGPUS_PER_NODE}"
echo "Batch size: ${train_batch_size}"
echo "Rollout N: ${rollout_n}"
echo "Learning rate: ${actor_lr}"
echo "Max steps per task: ${max_steps_per_task}"
echo "WebShop server: ${webshop_server}"
echo "=========================================="

# Pre-flight checks
echo "Running pre-flight checks..."

# Check if data exists
if [ ! -f "$HOME/data/webshop/train.parquet" ]; then
    echo "ERROR: WebShop training data not found!"
    echo "Please run: python examples/webshop_grpo/data_preprocess.py --local_save_dir ~/data/webshop"
    exit 1
fi

# Check if WebShop server is accessible (optional, comment out if not needed)
# curl -s "${webshop_server}" > /dev/null 2>&1 || {
#     echo "WARNING: WebShop server at ${webshop_server} may not be running"
#     echo "Please start it with: python -m webshop.run_server"
# }

echo "Pre-flight checks passed!"
echo "Starting training..."

# Launch training
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
