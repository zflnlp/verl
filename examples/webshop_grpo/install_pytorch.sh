#!/bin/bash
# Install PyTorch in WebShop conda environment
#
# This script installs PyTorch with CUDA 12.4 support for the webshop conda environment.
#
# Usage:
#   conda activate webshop
#   bash examples/webshop_grpo/install_pytorch.sh

set -e

echo "=========================================="
echo "Installing PyTorch for WebShop environment"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python --version 2>&1)
echo "Python version: $PYTHON_VERSION"

# Install PyTorch with CUDA 12.4
echo ""
echo "Installing PyTorch with CUDA 12.4 support..."
echo "This may take 5-10 minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify installation
echo ""
echo "Verifying PyTorch installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'Current device: {torch.cuda.current_device()}')
"

echo ""
echo "=========================================="
echo "PyTorch installation complete!"
echo "=========================================="
echo ""
echo "Now test WebShop environment:"
echo "  python -c \"import gym; from web_agent_site.envs import WebAgentTextEnv; print('OK')\""
