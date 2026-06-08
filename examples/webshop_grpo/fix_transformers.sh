#!/bin/bash
# Fix transformers version to support Qwen3
#
# Qwen3 requires transformers>=4.51.0
#
# Usage:
#   conda activate webshop
#   bash examples/webshop_grpo/fix_transformers.sh

set -e

echo "=========================================="
echo "Fixing transformers version for Qwen3"
echo "=========================================="

# Check current version
echo ""
echo "Current transformers version:"
pip show transformers | grep Version

# Upgrade transformers
echo ""
echo "Upgrading transformers to support Qwen3..."
pip install transformers>=4.51.0 --upgrade

# Verify
echo ""
echo "New transformers version:"
pip show transformers | grep Version

# Test Qwen3 import
echo ""
echo "Testing Qwen3 support..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('AutoTokenizer import OK')
print('AutoModelForCausalLM import OK')

# Test loading Qwen3 config
import json
config_path = '/workspace/models/Qwen3-1.7B/config.json'
with open(config_path) as f:
    config = json.load(f)
print(f'Model type: {config.get(\"model_type\", \"unknown\")}')
print(f'Architectures: {config.get(\"architectures\", \"unknown\")}')
"

echo ""
echo "=========================================="
echo "Fix complete!"
echo "=========================================="
echo ""
echo "Now run evaluation:"
echo "  python examples/webshop_grpo/evaluate_zero_shot.py \\"
echo "      --model_path /workspace/models/Qwen3-1.7B \\"
echo "      --num_episodes 50 \\"
echo "      --output_dir results/webshop_zero_shot"
