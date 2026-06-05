#!/bin/bash
# Install WebShop dependencies
#
# This script installs all required dependencies for WebShop environment.
#
# Usage:
#   conda activate webshop
#   bash examples/webshop_grpo/install_webshop_deps.sh

set -e

echo "=========================================="
echo "Installing WebShop Dependencies"
echo "=========================================="

# Install gym and core dependencies
echo ""
echo "Step 1: Installing gym and core dependencies..."
pip install gym==0.23.1
pip install flask flask-cors requests beautifulsoup4 lxml regex numpy pandas tqdm

# Install WebShop package
echo ""
echo "Step 2: Installing WebShop package..."
cd /workspace/WebShop
pip install -e .

# Install additional dependencies that WebShop might need
echo ""
echo "Step 3: Installing additional dependencies..."
pip install pyserini==0.17.0 2>/dev/null || echo "Note: pyserini already mocked"
pip install jsonlines
pip install datasets
pip install tokenizers

# Setup mock pyserini (in case pyserini was installed)
echo ""
echo "Step 4: Setting up mock pyserini..."
cd /workspace/verl
bash examples/webshop_grpo/setup_mock_pyserini.sh

# Verify installation
echo ""
echo "Step 5: Verifying installation..."
python -c "
import gym
print(f'gym version: {gym.__version__}')
"

python -c "
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
echo "WebShop dependencies installed!"
echo "=========================================="
echo ""
echo "Run zero-shot evaluation:"
echo "  cd /workspace/verl"
echo "  python examples/webshop_grpo/evaluate_zero_shot.py \\"
echo "      --model_path /workspace/models/Qwen3-1.7B \\"
echo "      --num_episodes 50 \\"
echo "      --output_dir results/webshop_zero_shot"
