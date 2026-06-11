#!/usr/bin/env bash
# Generate training data for all 30 ScienceWorld tasks
#
# Usage:
#   conda activate scienceworld
#   bash examples/scienceworld_grpo/generate_all_data.sh
#
# This script generates training data matching the TCOD paper setup:
# - All 30 ScienceWorld task types
# - Built-in train/dev/test splits
# - Output: /workspace/data/scienceworld_all/

set -euo pipefail

DATA_DIR=${DATA_DIR:-/workspace/data/scienceworld_all}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "ScienceWorld Data Generation (All 30 Tasks)"
echo "=========================================="
echo "Output directory: ${DATA_DIR}"
echo "=========================================="

# Check if scienceworld is installed
if ! python3 -c "import scienceworld" 2>/dev/null; then
    echo "Error: scienceworld package not found."
    echo "Please install it: pip install scienceworld"
    echo "Or activate the correct conda environment: conda activate scienceworld"
    exit 1
fi

# Generate data for all 30 tasks
python3 "$PROJECT_DIR/examples/scienceworld_grpo/data_preprocess.py" \
    --local_save_dir "$DATA_DIR" \
    --use_real_env

echo ""
echo "Data generation complete!"
echo "Files saved to: $DATA_DIR"
echo "  - train.parquet"
echo "  - val.parquet"
echo "  - test.parquet"
echo "  - task_metadata.json"
