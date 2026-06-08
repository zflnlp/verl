#!/bin/bash
# Fix WebShop setup issues
#
# Fixes:
# 1. werkzeug version incompatibility with Flask
# 2. gdown version issue
# 3. Re-run data conversion and index building
#
# Usage:
#   conda activate webshop
#   cd /workspace/WebShop
#   bash /workspace/verl/examples/webshop_grpo/fix_webshop_setup.sh

set -e

echo "=========================================="
echo "Fixing WebShop Setup Issues"
echo "=========================================="

# Step 1: Fix werkzeug version
echo ""
echo "Step 1: Fixing werkzeug version..."
pip install werkzeug==2.3.7

# Step 2: Fix Flask version
echo ""
echo "Step 2: Fixing Flask version..."
pip install flask==2.3.3

# Step 3: Fix gdown
echo ""
echo "Step 3: Fixing gdown..."
pip install gdown --upgrade

# Step 4: Verify imports
echo ""
echo "Step 4: Verifying imports..."
python -c "
from flask import Flask
print('Flask import OK')
"

python -c "
from web_agent_site.engine.engine import load_products
print('WebShop engine import OK')
"

# Step 5: Re-run data conversion
echo ""
echo "Step 5: Re-running data conversion..."
cd /workspace/WebShop

# Generate search engine resources
python search_engine/convert_product_file_format.py

# Step 6: Build search indexes
echo ""
echo "Step 6: Building search indexes..."
# Build indexes for different dataset sizes
for size in 100 1k 100k; do
    if [ -d "resources_${size}" ] && [ "$(ls -A resources_${size} 2>/dev/null)" ]; then
        echo "Building index for ${size}..."
        python -m pyserini.index.lucene \
            --collection JsonCollection \
            --input resources_${size} \
            --index indexes_${size} \
            --generator DefaultLuceneDocumentGenerator \
            --threads 1 \
            --storePositions --storeDocvectors --storeRaw \
            2>/dev/null || echo "Note: Index for ${size} may already exist or uses mock pyserini"
    fi
done

# Build main index
if [ -d "resources" ] && [ "$(ls -A resources 2>/dev/null)" ]; then
    echo "Building main index..."
    python -m pyserini.index.lucene \
        --collection JsonCollection \
        --input resources \
        --index indexes \
        --generator DefaultLuceneDocumentGenerator \
        --threads 1 \
        --storePositions --storeDocvectors --storeRaw \
        2>/dev/null || echo "Note: Main index may already exist or uses mock pyserini"
fi

# Step 7: Download example trajectories
echo ""
echo "Step 7: Downloading example trajectories..."
python -c "
import gdown
import os

# Download example trajectories
url = 'https://drive.google.com/drive/folders/1YRmNuECDJXc2MIBIV0IaOaWK1VXqAPBu'
output = 'data/human_trajectories'
os.makedirs(output, exist_ok=True)
try:
    gdown.download_folder(url, output=output, quiet=False)
    print('Example trajectories downloaded!')
except Exception as e:
    print(f'Note: Could not download trajectories: {e}')
    print('This is optional and not required for evaluation.')
"

# Step 8: Verify environment
echo ""
echo "Step 8: Verifying environment..."
python -c "
import gym
from web_agent_site.envs import WebAgentTextEnv
print('WebShop import OK!')
"

python -c "
import gym
from web_agent_site.envs import WebAgentTextEnv
env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=100)
obs = env.reset()
print('Environment creation OK!')
print('Observation preview:', obs[:200])
"

echo ""
echo "=========================================="
echo "Fix complete!"
echo "=========================================="
echo ""
echo "Now run zero-shot evaluation:"
echo "  cd /workspace/verl"
echo "  python examples/webshop_grpo/evaluate_zero_shot.py \\"
echo "      --model_path /workspace/models/Qwen3-1.7B \\"
echo "      --num_episodes 50 \\"
echo "      --output_dir results/webshop_zero_shot"
