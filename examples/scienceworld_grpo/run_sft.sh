#!/usr/bin/env bash
# SFT cold start for ScienceWorld using llama-factory
#
# This script fine-tunes Qwen3-1.7B on ScienceWorld gold trajectories
# before RL training. The SFT model learns:
# 1. The prompt format (thought + action tags)
# 2. Basic task completion strategies
# 3. Environment interaction patterns
#
# Usage:
#   cd /path/to/verl
#   bash examples/scienceworld_grpo/run_sft.sh
#
# Prerequisites:
#   1. Clone llama-factory: git clone https://github.com/hiyouga/LLaMA-Factory.git llama-factory
#   2. Install llama-factory: cd llama-factory && pip install -e .
#   3. Prepare SFT data: python examples/scienceworld_grpo/prepare_sft_data.py

set -xeuo pipefail

########################### user-adjustable ###########################
# Model configuration
MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen3-1.7B}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/models/Qwen3-1.7B-SFT}

# SFT hyperparameters
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-8}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-4}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}

# Data configuration
DATA_DIR=${DATA_DIR:-/workspace/data/scienceworld_sft}
########################### end user-adjustable ###########################

# Get project directory
PROJECT_DIR="$(pwd)"
LLAMA_FACTORY_DIR="$PROJECT_DIR/llama-factory"

echo "=========================================="
echo "ScienceWorld SFT Cold Start"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Max seq length: ${MAX_SEQ_LENGTH}"
echo "=========================================="

# Check if llama-factory exists
if [ ! -d "$LLAMA_FACTORY_DIR" ]; then
    echo "Error: llama-factory not found at $LLAMA_FACTORY_DIR"
    echo "Please clone it first:"
    echo "  git clone https://github.com/hiyouga/LLaMA-Factory.git llama-factory"
    exit 1
fi

# Check if SFT data exists
if [ ! -f "$DATA_DIR/train.json" ]; then
    echo "Error: SFT data not found at $DATA_DIR/train.json"
    echo "Please run data preparation first:"
    echo "  python examples/scienceworld_grpo/prepare_sft_data.py"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create dataset_info.json in llama-factory/data if not exists
if [ ! -f "$LLAMA_FACTORY_DIR/data/dataset_info.json" ]; then
    echo "Warning: dataset_info.json not found in llama-factory/data/"
    echo "Copying from SFT data directory..."
    cp "$DATA_DIR/dataset_info.json" "$LLAMA_FACTORY_DIR/data/"
fi

# Copy dataset_info.json to llama-factory/data if not exists
if [ ! -f "$LLAMA_FACTORY_DIR/data/dataset_info.json" ]; then
    echo "Copying dataset_info.json to llama-factory/data..."
    cp "$DATA_DIR/dataset_info.json" "$LLAMA_FACTORY_DIR/data/"
fi

# Create symlinks for SFT data (no copy needed)
echo "Creating symlinks for SFT data..."
ln -sf "$DATA_DIR/train.json" "$LLAMA_FACTORY_DIR/data/scienceworld_train.json"
ln -sf "$DATA_DIR/dev.json" "$LLAMA_FACTORY_DIR/data/scienceworld_dev.json" 2>/dev/null || true

# Create training config (full fine-tuning)
cat > "$LLAMA_FACTORY_DIR/examples/train_full/scienceworld_sft.yaml" << EOF
### model
model_name_or_path: ${MODEL_PATH}

### method
stage: sft
do_train: true
finetuning_type: full

### dataset
dataset: scienceworld_train
template: qwen3
cutoff_len: ${MAX_SEQ_LENGTH}
max_samples: 10000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: ${OUTPUT_DIR}
logging_steps: 10
save_steps: 500
save_total_limit: 3

### train
per_device_train_batch_size: ${BATCH_SIZE}
gradient_accumulation_steps: ${GRADIENT_ACCUMULATION}
learning_rate: ${LEARNING_RATE}
num_train_epochs: ${EPOCHS}
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 8
eval_strategy: steps
eval_steps: 500
EOF

echo ""
echo "Training config created at: $LLAMA_FACTORY_DIR/examples/train_full/scienceworld_sft.yaml"
echo ""
echo "Starting SFT training..."
echo "=========================================="

# Run SFT training
# Use single GPU by default, or specified GPUs
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES

cd "$LLAMA_FACTORY_DIR"
llamafactory-cli train examples/train_full/scienceworld_sft.yaml

# Note: llama-factory is pinned to v0.9.2 for compatibility with CUDA 12.4
# To update: cd llama-factory && git checkout v0.9.2
