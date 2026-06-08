#!/bin/bash
# Complete WebShop environment setup with Python 3.10
#
# This script:
# 1. Removes old webshop environment (Python 3.8)
# 2. Creates new webshop environment with Python 3.10
# 3. Installs PyTorch with CUDA 12.4
# 4. Installs WebShop dependencies
# 5. Installs transformers (supports Qwen3)
# 6. Sets up mock pyserini
# 7. Tests the environment
#
# Usage:
#   bash examples/webshop_grpo/setup_webshop_py310.sh

set -e

echo "=========================================="
echo "WebShop Environment Setup (Python 3.10)"
echo "=========================================="

# Step 1: Remove old environment
echo ""
echo "Step 1: Removing old webshop environment..."
conda env remove -n webshop -y 2>/dev/null || true
echo "Old environment removed."

# Step 2: Create new environment with Python 3.10
echo ""
echo "Step 2: Creating new conda environment with Python 3.10..."
conda create -n webshop python=3.10 -y

# Activate environment
eval "$(conda shell.bash hook)"
conda activate webshop

echo "Python version: $(python --version 2>&1)"

# Step 3: Install PyTorch with CUDA 12.4
echo ""
echo "Step 3: Installing PyTorch with CUDA 12.4..."
echo "This may take 5-10 minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Step 4: Install core dependencies
echo ""
echo "Step 4: Installing core dependencies..."
pip install gym==0.23.1
pip install flask flask-cors requests beautifulsoup4 lxml regex numpy pandas tqdm
pip install jsonlines datasets tokenizers

# Step 5: Install transformers (supports Qwen3)
echo ""
echo "Step 5: Installing transformers (supports Qwen3)..."
pip install transformers>=4.51.0
pip install accelerate

# Step 6: Install WebShop package
echo ""
echo "Step 6: Installing WebShop package..."
cd /workspace/WebShop
pip install -e .

# Step 7: Setup mock pyserini
echo ""
echo "Step 7: Setting up mock pyserini..."
cd /workspace/verl
bash examples/webshop_grpo/setup_mock_pyserini.sh

# Step 8: Verify installation
echo ""
echo "Step 8: Verifying installation..."
echo ""

echo "1. Checking Python..."
python --version

echo ""
echo "2. Checking PyTorch..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "3. Checking transformers..."
python -c "
import transformers
print(f'  transformers: {transformers.__version__}')
from transformers import AutoTokenizer
print('  AutoTokenizer import OK')
"

echo ""
echo "4. Checking gym..."
python -c "
import gym
print(f'  gym: {gym.__version__}')
"

echo ""
echo "5. Checking WebShop..."
python -c "
from web_agent_site.envs import WebAgentTextEnv
print('  WebShop import OK!')
"

echo ""
echo "6. Testing WebShop environment..."
python -c "
import gym
from web_agent_site.envs import WebAgentTextEnv
env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=100)
obs = env.reset()
print('  Environment creation OK!')
print('  Observation preview:', obs[:100])
"

echo ""
echo "7. Testing Qwen3 tokenizer..."
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/workspace/models/Qwen3-1.7B', trust_remote_code=True)
print(f'  Qwen3 tokenizer loaded: {type(tokenizer).__name__}')
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
