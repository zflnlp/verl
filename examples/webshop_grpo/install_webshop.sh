#!/bin/bash
# Install WebShop dependencies and setup Python path
#
# WebShop doesn't have setup.py, so we need to:
# 1. Install dependencies from requirements.txt
# 2. Add WebShop to Python path
#
# Usage:
#   conda activate webshop
#   bash /workspace/verl/examples/webshop_grpo/install_webshop.sh

set -e

echo "=========================================="
echo "Installing WebShop Dependencies"
echo "=========================================="

cd /workspace/WebShop

# Step 1: Install dependencies
echo ""
echo "Step 1: Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Step 2: Fix werkzeug version
echo ""
echo "Step 2: Fixing werkzeug version..."
pip install werkzeug==2.3.7

# Step 3: Add WebShop to Python path
echo ""
echo "Step 3: Adding WebShop to Python path..."
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
echo "Site-packages: $SITE_PACKAGES"

# Create .pth file to add WebShop to Python path
echo "/workspace/WebShop" > $SITE_PACKAGES/webshop.pth
echo "Created $SITE_PACKAGES/webshop.pth"

# Step 4: Verify installation
echo ""
echo "Step 4: Verifying installation..."
python -c "
import sys
print('Python path includes WebShop:', '/workspace/WebShop' in sys.path)
from web_agent_site.envs import WebAgentTextEnv
print('WebShop import OK!')
"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Now run the fix script to rebuild indexes:"
echo "  bash /workspace/verl/examples/webshop_grpo/fix_webshop_setup.sh"
