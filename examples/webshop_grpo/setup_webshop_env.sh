#!/bin/bash
# Setup WebShop environment with Python 3.10
#
# This script creates a new conda environment with Python 3.10
# to match the system PyTorch (compiled for Python 3.10 + CUDA 12.4).
#
# Usage:
#   bash examples/webshop_grpo/setup_webshop_env.sh

set -e

echo "=========================================="
echo "Setting up WebShop environment (Python 3.10)"
echo "=========================================="

# Remove old webshop environment if it exists
echo "Removing old webshop environment..."
conda env remove -n webshop -y 2>/dev/null || true

# Create new environment with Python 3.10
echo ""
echo "Creating new conda environment with Python 3.10..."
conda create -n webshop python=3.10 -y

# Activate environment
echo ""
echo "Activating webshop environment..."
eval "$(conda shell.bash hook)"
conda activate webshop

# Verify Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "Python version: $PYTHON_VERSION"

# Verify system PyTorch is accessible
echo ""
echo "Verifying PyTorch..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
"

# Install WebShop dependencies
echo ""
echo "Installing WebShop dependencies..."
cd /workspace/WebShop

# Install core dependencies
pip install -e .

# Install additional dependencies
pip install flask flask-cors requests beautifulsoup4 lxml regex numpy pandas tqdm

echo ""
echo "=========================================="
echo "WebShop environment setup complete!"
echo "=========================================="
echo ""
echo "To use the environment:"
echo "  conda activate webshop"
echo "  cd /workspace/verl"
echo ""
echo "Test WebShop:"
echo "  python -c \"import gym; from web_agent_site.envs import WebAgentTextEnv; print('OK')\""
