#!/bin/bash
# Complete WebShop environment setup
#
# This script handles the entire setup process:
# 1. Create conda environment with Python 3.10
# 2. Install PyTorch with CUDA 12.4
# 3. Install WebShop dependencies
# 4. Setup mock pyserini
# 5. Test the environment
#
# Usage:
#   bash examples/webshop_grpo/setup_webshop_complete.sh

set -e

echo "=========================================="
echo "Complete WebShop Environment Setup"
echo "=========================================="

# Step 1: Create conda environment
echo ""
echo "Step 1: Creating conda environment with Python 3.10..."
conda env remove -n webshop -y 2>/dev/null || true
conda create -n webshop python=3.10 -y

# Activate environment
eval "$(conda shell.bash hook)"
conda activate webshop

echo "Python version: $(python --version 2>&1)"

# Step 2: Install PyTorch
echo ""
echo "Step 2: Installing PyTorch with CUDA 12.4..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Step 3: Install WebShop dependencies
echo ""
echo "Step 3: Installing WebShop dependencies..."
cd /workspace/WebShop

# Install core WebShop package
pip install -e .

# Install additional dependencies
pip install flask flask-cors requests beautifulsoup4 lxml regex numpy pandas tqdm gym

# Step 4: Setup mock pyserini
echo ""
echo "Step 4: Setting up mock pyserini..."
cd /workspace/verl
bash examples/webshop_grpo/setup_mock_pyserini.sh

# Step 5: Test environment
echo ""
echo "Step 5: Testing WebShop environment..."
python -c "
import gym
from web_agent_site.envs import WebAgentTextEnv
print('WebShop import OK!')
"

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
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
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To use the environment:"
echo "  conda activate webshop"
echo "  cd /workspace/verl"
echo ""
echo "Run zero-shot evaluation:"
echo "  python examples/webshop_grpo/evaluate_zero_shot.py \\"
echo "      --model_path /workspace/models/Qwen3-1.7B \\"
echo "      --num_episodes 50 \\"
echo "      --output_dir results/webshop_zero_shot"
