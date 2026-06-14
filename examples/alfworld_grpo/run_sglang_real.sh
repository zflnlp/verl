#!/usr/bin/env bash
# GRPO | ALFWorld Real Training with SGLang Multi-Turn | v0.4.1 Compatible
#
# This script trains a model using the REAL ALFWorld environment with SGLang for multi-turn rollout.
# Requires: pip install alfworld[full] && alfworld-download && pip install "sglang[all]==0.4.6.post5"
#
# Usage:
#   bash examples/alfworld_grpo/run_sglang_real.sh
#
# For mock testing, use:
#   bash examples/alfworld_grpo/run_sglang_multiturn.sh

set -xeuo pipefail

########################### user-adjustable ###########################
# Model configuration
MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen3-1.7B}

# Hardware configuration
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

# Training hyperparameters
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-10240}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-6144}

# Learning rate
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-1.0}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

# Rollout configuration
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}

# Training schedule
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:-2}

# Data configuration
DATA_DIR=${DATA_DIR:-/workspace/data/alfworld_real}

# ALFWorld configuration
ALFWORLD_DATA_DIR=${ALFWORLD_DATA_DIR:-/workspace/data/alf_data}

# Experiment tracking
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_alfworld_sglang_real}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-alfworld_sglang_real_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

# Get project directory
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/alfworld_grpo/config"

echo "=========================================="
echo "ALFWorld GRPO Real Training (SGLang Multi-Turn)"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "GPUs: ${NGPUS_PER_NODE}"
echo "Batch size: ${TRAIN_BATCH_SIZE}"
echo "Rollout N: ${ROLLOUT_N}"
echo "Data dir: ${DATA_DIR}"
echo "ALFWorld data: ${ALFWORLD_DATA_DIR}"
echo "NOTE: Using REAL ALFWorld environment with SGLang"
echo "=========================================="

# Generate real training config with use_mock: false
TMPCONF=$(mktemp -d)
cat > "${TMPCONF}/alfworld_interaction_config.yaml" <<EOF
interaction:
  - class_name: "verl.interactions.alfworld_interaction.AlfworldInteraction"
    config:
      use_mock: false
      max_steps: 30
      game_files_dir: "${ALFWORLD_DATA_DIR}"
      task_types:
        - pick_and_place
        - pick_clean_then_place
        - pick_heat_then_place
        - pick_cool_then_place
        - look_at_obj_in_light
        - pick_two_obj
      train_eval: eval_out_of_distribution
      num_games: -1
EOF

cat > "${TMPCONF}/alfworld_tool_config.yaml" <<'EOF'
tools:
  - class_name: "verl.tools.alfworld_tool.AlfworldTool"
    config:
      type: native
      use_mock: false
    tool_schema:
      type: "function"
      function:
        name: "action"
        description: "Execute an action in the ALFWorld household environment."
        parameters:
          type: "object"
          properties:
            command:
              type: "string"
              description: "The action command to execute (e.g., 'go to countertop 1', 'take apple from countertop 1', 'put apple in fridge 1', 'open fridge 1', 'clean apple 1', 'heat apple 1', 'cool apple 1')"
          required: ["command"]
EOF

# Launch training
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='alfworld_sglang_multiturn' \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE} \
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TMPCONF}/alfworld_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="${TMPCONF}/alfworld_interaction_config.yaml" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    "$@"

# Cleanup temp config
rm -rf "${TMPCONF}"
