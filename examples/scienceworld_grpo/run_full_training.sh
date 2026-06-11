#!/usr/bin/env bash
# Full GRPO training for ScienceWorld matching TCOD paper settings
#
# This script trains a model using the REAL ScienceWorld environment
# with parameters aligned to the TCOD paper:
# - All 30 ScienceWorld task types
# - 250 training steps
# - Batch size 64
# - Max prompt length 10240
# - Max response length 512
# - Max steps 30
#
# Usage:
#   conda activate verl
#   bash examples/scienceworld_grpo/run_full_training.sh
#
# For evaluation after training:
#   python examples/scienceworld_grpo/eval_zero_shot.py \
#       --model_path checkpoints/verl_grpo_scienceworld_all/<experiment>/global_step_250/actor/huggingface \
#       --task_name boil --num_variations 30 --max_steps 30

set -xeuo pipefail

########################### user-adjustable ###########################
# Model configuration
MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen3-1.7B}

# Hardware configuration
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}

# Training hyperparameters (aligned with TCOD paper)
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-10240}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}

# Learning rate (same as TCOD paper)
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

# Rollout configuration
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}

# Training schedule (250 steps as in TCOD paper)
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_STEPS=${TOTAL_STEPS:-250}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-5}

# Data configuration (all 30 tasks)
DATA_DIR=${DATA_DIR:-/workspace/data/scienceworld_all}

# Experiment tracking
PROJECT_NAME=${PROJECT_NAME:-verl_grpo_scienceworld_all}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-scienceworld_all_grpo_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ###########################

# Get project directory
PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/scienceworld_grpo/config"

echo "=========================================="
echo "ScienceWorld GRPO Full Training"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "GPUs: ${NGPUS_PER_NODE}"
echo "Batch size: ${TRAIN_BATCH_SIZE}"
echo "Rollout N: ${ROLLOUT_N}"
echo "Data dir: ${DATA_DIR}"
echo "Total steps: ${TOTAL_STEPS}"
echo "Max prompt length: ${MAX_PROMPT_LENGTH}"
echo "Max response length: ${MAX_RESPONSE_LENGTH}"
echo "NOTE: Using REAL ScienceWorld environment (all 30 tasks)"
echo "=========================================="

# Check if data exists
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "Error: Training data not found at ${DATA_DIR}/train.parquet"
    echo "Please run data generation first:"
    echo "  bash examples/scienceworld_grpo/generate_all_data.sh"
    exit 1
fi

# Generate real training config with use_mock: false
TMPCONF=$(mktemp -d)
cat > "${TMPCONF}/scienceworld_interaction_config.yaml" <<'EOF'
interaction:
  - class_name: "verl.interactions.scienceworld_interaction.ScienceWorldInteraction"
    config:
      use_mock: false
      max_steps: 30
      simplifications_preset: easy
      env_step_limit: 100
EOF

cat > "${TMPCONF}/scienceworld_tool_config.yaml" <<'EOF'
tools:
  - class_name: "verl.tools.scienceworld_tool.ScienceWorldTool"
    config:
      type: native
      use_mock: false
    tool_schema:
      type: "function"
      function:
        name: "action"
        description: "Execute an action in the ScienceWorld laboratory environment."
        parameters:
          type: "object"
          properties:
            command:
              type: "string"
              description: "The action command to execute (e.g., 'look around', 'examine beaker', 'take flask from table', 'pour water into pot')"
          required: ["command"]
EOF

# Calculate total training samples needed
# TCOD paper: 250 steps * 64 batch_size = 16,000 samples
# With rollout_n=4, we need 250 * 64 / 4 = 4,000 unique prompts
# But we can reuse samples across epochs

# Launch training
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='scienceworld_grpo' \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/val.parquet \
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
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TMPCONF}/scienceworld_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="${TMPCONF}/scienceworld_interaction_config.yaml" \
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

echo ""
echo "Training complete!"
echo "Checkpoint saved to: checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/"
echo ""
echo "To evaluate the trained model:"
echo "  python examples/scienceworld_grpo/eval_zero_shot.py \\"
echo "      --model_path checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/global_step_${TOTAL_STEPS}/actor/huggingface \\"
echo "      --task_name boil --num_variations 30 --max_steps 30"
