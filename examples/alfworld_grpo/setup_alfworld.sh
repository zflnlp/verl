#!/usr/bin/env bash
# Setup script for ALFWorld environment
#
# This script installs ALFWorld and downloads game files.
# Run this once before using run_real.sh.
#
# Usage:
#   bash examples/alfworld_grpo/setup_alfworld.sh

set -xeuo pipefail

ALFWORLD_DATA_DIR=${ALFWORLD_DATA_DIR:-/workspace/alfworld_data}

echo "=========================================="
echo "ALFWorld Environment Setup"
echo "=========================================="
echo "Data directory: ${ALFWORLD_DATA_DIR}"
echo "=========================================="

# Step 1: Install alfworld package
echo "[Step 1/4] Installing alfworld package..."
pip install alfworld[full] 2>&1 | tail -5

# Step 2: Set ALFWORLD_DATA env var (used by alfworld-download and the config)
export ALFWORLD_DATA="${ALFWORLD_DATA_DIR}"
echo "export ALFWORLD_DATA=${ALFWORLD_DATA_DIR}" >> ~/.bashrc

# Step 3: Download game files
echo "[Step 3/4] Downloading ALFWorld game files..."
mkdir -p "${ALFWORLD_DATA_DIR}"
alfworld-download 2>&1 | tail -5

# Step 4: Verify installation
echo "[Step 4/4] Verifying installation..."
python3 -c "
import os
import alfworld
import alfworld.agents.environment
data_dir = '${ALFWORLD_DATA_DIR}'
print(f'ALFWorld installed OK')
print(f'Data directory: {data_dir}')
assert os.path.exists(data_dir), f'Data directory not found: {data_dir}'
# Check for game files
import glob
games = glob.glob(os.path.join(data_dir, 'json_2.1.1', '**', 'game.tw-pddl'), recursive=True)
print(f'Found {len(games)} game files')
assert len(games) > 0, 'No game files found! Run alfworld-download first.'
print('Verification: OK')
"

echo ""
echo "=========================================="
echo "ALFWorld setup complete!"
echo ""
echo "To run real training:"
echo "  bash examples/alfworld_grpo/run_real.sh"
echo ""
echo "To generate real training data:"
echo "  python examples/alfworld_grpo/data_preprocess.py \\"
echo "      --local_save_dir /workspace/data/alfworld_real \\"
echo "      --use_real_env --task_type pick_and_place --num_games 10"
echo "=========================================="
