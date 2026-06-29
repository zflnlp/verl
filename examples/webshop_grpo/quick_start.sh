#!/usr/bin/env bash
# WebShop GRPO Training - Quick Start Script
#
# This script automates the entire WebShop GRPO training pipeline:
# 1. Check prerequisites
# 2. Generate training data
# 3. Start WebShop server (optional)
# 4. Run training
# 5. Evaluate model
#
# Usage:
#   bash examples/webshop_grpo/quick_start.sh [--skip-server] [--eval-only]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
SKIP_SERVER=false
EVAL_ONLY=false
for arg in "$@"; do
    case $arg in
        --skip-server)
            SKIP_SERVER=true
            shift
            ;;
        --eval-only)
            EVAL_ONLY=true
            shift
            ;;
        *)
            ;;
    esac
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}WebShop GRPO Training - Quick Start${NC}"
echo -e "${GREEN}========================================${NC}"

# Step 1: Check prerequisites
echo -e "\n${YELLOW}[Step 1/5] Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 found${NC}"

# Check if verl is installed
if ! python3 -c "import verl" &> /dev/null; then
    echo -e "${RED}ERROR: verl not installed${NC}"
    echo "Please install verl: pip install -e /path/to/verl"
    exit 1
fi
echo -e "${GREEN}✓ verl installed${NC}"

# Check WebShop
if ! python3 -c "import webshop" &> /dev/null; then
    echo -e "${YELLOW}WARNING: WebShop not installed${NC}"
    echo "Installing WebShop..."
    pip install webshop || {
        echo -e "${RED}ERROR: Failed to install WebShop${NC}"
        echo "Please install manually: pip install webshop"
        exit 1
    }
fi
echo -e "${GREEN}✓ WebShop available${NC}"

# Step 2: Generate training data
echo -e "\n${YELLOW}[Step 2/5] Generating training data...${NC}"

DATA_DIR="$HOME/data/webshop"
if [ -f "$DATA_DIR/train.parquet" ]; then
    echo -e "${GREEN}✓ Training data already exists${NC}"
else
    echo "Generating WebShop tasks..."
    python3 examples/webshop_grpo/data_preprocess.py \
        --local_save_dir "$DATA_DIR" \
        --num_tasks 1000 \
        --train_ratio 0.8 \
        --seed 42
    echo -e "${GREEN}✓ Training data generated${NC}"
fi

# Step 3: Start WebShop server (if not skipped)
if [ "$SKIP_SERVER" = false ] && [ "$EVAL_ONLY" = false ]; then
    echo -e "\n${YELLOW}[Step 3/5] Starting WebShop server...${NC}"

    # Check if server is already running
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ WebShop server already running${NC}"
    else
        echo "Starting WebShop server in background..."
        python3 -m webshop.run_server &
        SERVER_PID=$!
        echo "Server PID: $SERVER_PID"

        # Wait for server to start
        echo "Waiting for server to start..."
        for i in {1..30}; do
            if curl -s http://localhost:3000 > /dev/null 2>&1; then
                echo -e "${GREEN}✓ WebShop server started${NC}"
                break
            fi
            sleep 1
        done

        # Check if server started successfully
        if ! curl -s http://localhost:3000 > /dev/null 2>&1; then
            echo -e "${RED}ERROR: WebShop server failed to start${NC}"
            echo "Please start manually: python -m webshop.run_server"
            exit 1
        fi
    fi
else
    echo -e "\n${YELLOW}[Step 3/5] Skipping WebShop server${NC}"
fi

# Step 4: Run training
if [ "$EVAL_ONLY" = false ]; then
    echo -e "\n${YELLOW}[Step 4/5] Starting GRPO training...${NC}"

    # Default training configuration
    export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
    export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
    export ROLLOUT_N=${ROLLOUT_N:-4}
    export ACTOR_LR=${ACTOR_LR:-5e-7}
    export TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}

    echo "Training configuration:"
    echo "  Model: $MODEL_PATH"
    echo "  Batch size: $TRAIN_BATCH_SIZE"
    echo "  Rollout N: $ROLLOUT_N"
    echo "  Learning rate: $ACTOR_LR"
    echo "  Epochs: $TOTAL_EPOCHS"

    # Run training
    bash examples/webshop_grpo/run_qwen2_5_2b_webshop_grpo.sh

    echo -e "${GREEN}✓ Training completed${NC}"
else
    echo -e "\n${YELLOW}[Step 4/5] Skipping training (eval-only mode)${NC}"
fi

# Step 5: Evaluate model
echo -e "\n${YELLOW}[Step 5/5] Evaluating model...${NC}"

# Find the latest checkpoint
CHECKPOINT_DIR="checkpoints/webshop_grpo"
if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST_CKPT=$(ls -td "$CHECKPOINT_DIR"/global_step_* 2>/dev/null | head -1)
    if [ -n "$LATEST_CKPT" ]; then
        MODEL_PATH="$LATEST_CKPT/actor"
        echo "Evaluating checkpoint: $MODEL_PATH"

        python3 examples/webshop_grpo/evaluate.py \
            --model_path "$MODEL_PATH" \
            --num_episodes 50 \
            --output_dir results/webshop_eval

        echo -e "${GREEN}✓ Evaluation completed${NC}"

        # Show results
        if [ -f "results/webshop_eval/eval_summary.txt" ]; then
            echo -e "\n${GREEN}Evaluation Results:${NC}"
            cat results/webshop_eval/eval_summary.txt
        fi
    else
        echo -e "${YELLOW}WARNING: No checkpoints found${NC}"
    fi
else
    echo -e "${YELLOW}WARNING: Checkpoint directory not found${NC}"
    echo "Skipping evaluation"
fi

# Cleanup
if [ -n "${SERVER_PID:-}" ]; then
    echo -e "\n${YELLOW}Stopping WebShop server...${NC}"
    kill $SERVER_PID 2>/dev/null || true
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Quick Start Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo "1. View training logs: wandb sync wandb/"
echo "2. Evaluate best model: python examples/webshop_grpo/evaluate.py --model_path <best_checkpoint>"
echo "3. Adjust hyperparameters in run_qwen2_5_2b_webshop_grpo.sh"
echo ""
echo "For more information, see: examples/webshop_grpo/README.md"
