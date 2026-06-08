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

# Step 2: Download game files
echo "[Step 2/4] Downloading ALFWorld game files..."
mkdir -p "${ALFWORLD_DATA_DIR}"
alfworld-download --data "${ALFWORLD_DATA_DIR}" 2>&1 | tail -5

# Step 3: Set environment variable
echo "[Step 3/4] Setting ALFROOT environment variable..."
export ALFROOT="${ALFWORLD_DATA_DIR}"
echo "export ALFROOT=${ALFWORLD_DATA_DIR}" >> ~/.bashrc

# Step 4: Verify installation
echo "[Step 4/4] Verifying installation..."
python3 -c "
import alfworld
import alfworld.agents.environment
print(f'ALFWorld version: {alfworld.__version__ if hasattr(alfworld, \"__version__\") else \"installed\"}')
print(f'Data directory: ${ALFWORLD_DATA_DIR}')
import os
assert os.path.exists('${ALFWORLD_DATA_DIR}'), 'Data directory not found!'
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
