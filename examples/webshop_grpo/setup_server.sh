#!/bin/bash
# Complete server setup for WebShop GRPO training
#
# This script sets up the WebShop environment on the server.
# Run this on the server (hgx18) after SSHing in.
#
# Usage:
#   bash examples/webshop_grpo/setup_server.sh

set -e

echo "=========================================="
echo "WebShop GRPO Server Setup"
echo "=========================================="

# Step 1: Setup mock pyserini
echo ""
echo "Step 1: Setting up mock pyserini..."
bash examples/webshop_grpo/setup_mock_pyserini.sh

# Step 2: Test WebShop environment
echo ""
echo "Step 2: Testing WebShop environment..."
cd /workspace/WebShop

python -c "
import gym
from web_agent_site.envs import WebAgentTextEnv
print('WebShop import OK!')
"

# Step 3: Test environment creation
echo ""
echo "Step 3: Testing environment creation..."
python -c "
import gym
from web_agent_site.envs import WebAgentTextEnv
env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=100)
obs = env.reset()
print('Environment creation OK!')
print('Observation preview:', obs[:200])
"

# Step 4: Return to verl directory
cd /workspace/verl

echo ""
echo "=========================================="
echo "Server setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run zero-shot evaluation:"
echo "   python examples/webshop_grpo/evaluate_zero_shot.py \\"
echo "       --model_path /workspace/models/Qwen3-1.7B \\"
echo "       --num_episodes 50 \\"
echo "       --output_dir results/webshop_zero_shot"
echo ""
echo "2. Prepare real training data:"
echo "   python examples/webshop_grpo/data_preprocess.py \\"
echo "       --local_save_dir /workspace/data/webshop_real \\"
echo "       --num_tasks 1000"
echo ""
echo "3. Run real training (requires WebShop server):"
echo "   export WEBSHOP_SERVER=http://localhost:3000"
echo "   bash examples/webshop_grpo/run_real.sh"
