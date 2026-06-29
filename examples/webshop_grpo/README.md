# WebShop GRPO Training

<<<<<<< HEAD
This directory contains scripts for training a shopping agent using GRPO (Group Relative Policy Optimization) on the WebShop environment.

## Overview

WebShop is a simulated e-commerce environment where an agent needs to:
1. Search for products based on user instructions
2. Browse product listings
3. Select and purchase the appropriate product

## Quick Start (Mock Mode)

Mock mode allows you to test the training pipeline without a real WebShop server.

### 1. Generate Training Data
=======
This directory contains scripts for training language models on the WebShop environment using Group Relative Policy Optimization (GRPO).

## Overview

WebShop is an RL environment where agents interact with a simulated e-commerce website to find and purchase products based on natural language instructions. This setup enables training shopping assistant agents using GRPO.

## Quick Start

### Option A: Quick Test (No WebShop Server Required)

If you just want to test the training pipeline without setting up WebShop:

```bash
# 1. Generate mock training data
python examples/webshop_grpo/data_preprocess.py --local_save_dir ~/data/webshop --num_tasks 100

# 2. Run training with mock tools
bash examples/webshop_grpo/run_qwen2_5_2b_webshop_grpo_mock.sh
```

This uses `mock_webshop_tools.py` which simulates WebShop behavior locally.

### Option B: Full Training with Real WebShop

### 1. Install Dependencies

```bash
# Install WebShop environment
pip install webshop

# Install verl dependencies
cd /path/to/verl
pip install -e .
```

### 2. Start WebShop Server

In a separate terminal:
```bash
python -m webshop.run_server
```

The server will start at `http://localhost:3000` by default.

### 3. Preprocess Data

Generate training data for WebShop tasks:
>>>>>>> main

```bash
python examples/webshop_grpo/data_preprocess.py \
    --local_save_dir ~/data/webshop \
<<<<<<< HEAD
    --num_tasks 100 \
    --seed 42
```

### 2. Run Mock Training

```bash
bash examples/webshop_grpo/run_mock.sh
```

This will:
- Use a mock WebShop environment
- Train for 1 epoch on 100 tasks
- Use Qwen2.5-0.5B-Instruct by default

## Real WebShop Training

For training with a real WebShop server:

### 1. Set Up WebShop Server

Follow the [WebShop installation guide](https://webshop-pnlp.readthedocs.io/) to set up the server.

### 2. Prepare Real Data

```bash
python examples/webshop_grpo/data_preprocess.py \
    --local_save_dir ~/data/webshop_real \
    --num_tasks 1000 \
    --seed 42
```

### 3. Run Real Training

```bash
# Set WebShop server URL
export WEBSHOP_SERVER=http://localhost:3000

# Run training
bash examples/webshop_grpo/run_real.sh
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | Qwen/Qwen2.5-0.5B-Instruct | Path to the model |
| `WEBSHOP_SERVER` | http://localhost:3000 | WebShop server URL (real mode) |
| `TRAIN_BATCH_SIZE` | 32 (mock) / 64 (real) | Training batch size |
| `ROLLOUT_N` | 4 (mock) / 8 (real) | Number of rollouts per prompt |
| `TOTAL_EPOCHS` | 1 (mock) / 3 (real) | Number of training epochs |
| `DATA_DIR` | ~/data/webshop | Data directory |

### Hyperparameters

| Parameter | Mock | Real | Description |
|-----------|------|------|-------------|
| `MAX_PROMPT_LENGTH` | 1024 | 1024 | Max prompt tokens |
| `MAX_RESPONSE_LENGTH` | 2048 | 3072 | Max response tokens (longer for real) |
| `ACTOR_LR` | 1e-6 | 1e-6 | Learning rate |
| `KL_LOSS_COEF` | 0.001 | 0.001 | KL divergence coefficient |
| `ROLLOUT_GPU_MEM_UTIL` | 0.6 | 0.6 | GPU memory utilization |

## Architecture

### Interaction Class

`verl/interactions/webshop_interaction.py` manages multi-turn interactions:
- `start_interaction`: Initialize a shopping task
- `generate_response`: Process agent actions and return observations
- `calculate_score`: Calculate reward based on purchase quality
- `finalize_interaction`: Clean up resources

### Tool Class

`verl/tools/webshop_tool.py` provides WebShop functions:
- `search`: Search for products
- `click`: View product details
- `buy`: Purchase items

### Reward Function

The reward is calculated based on:
1. Whether a purchase was made (0.5 base reward)
2. How well the purchased item matches the task description
3. Efficiency (fewer steps = higher reward)
=======
    --num_tasks 1000 \
    --train_ratio 0.8 \
    --seed 42
```

This creates:
- `~/data/webshop/train.parquet` - Training data
- `~/data/webshop/test.parquet` - Test data
- `~/data/webshop/task_metadata.json` - Task metadata

### 4. Run Training

```bash
# Default configuration (Qwen2.5-1.5B-Instruct)
bash examples/webshop_grpo/run_qwen2_5_2b_webshop_grpo.sh

# Custom configuration
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
NGPUS_PER_NODE=8 \
TRAIN_BATCH_SIZE=128 \
ROLLOUT_N=8 \
bash examples/webshop_grpo/run_qwen2_5_2b_webshop_grpo.sh
```

### 5. Evaluate Trained Model

```bash
python examples/webshop_grpo/evaluate.py \
    --model_path checkpoints/webshop_grpo/global_step_100/actor \
    --num_episodes 100 \
    --output_dir results/webshop_eval
```
>>>>>>> main

## File Structure

```
examples/webshop_grpo/
<<<<<<< HEAD
├── config/
│   ├── webshop_interaction_config.yaml   # Interaction config (mock)
│   └── webshop_tool_config.yaml          # Tool config
├── data_preprocess.py                    # Data generation script
├── run_mock.sh                          # Mock training script
├── run_real.sh                          # Real training script
└── README.md                            # This file
=======
├── README.md                              # This file
├── webshop_tool.py                       # BaseTool implementation for WebShop
├── webshop_tools.py                      # FunctionTool wrappers for real WebShop
├── mock_webshop_tools.py                 # Mock FunctionTool (no server needed)
├── reward_function.py                    # Reward computation functions
├── data_preprocess.py                    # Data preprocessing script
├── run_qwen2_5_2b_webshop_grpo.sh        # Training with real WebShop
├── run_qwen2_5_2b_webshop_grpo_mock.sh   # Training with mock tools
├── evaluate.py                           # Evaluation script
├── test_tools.py                         # Tool testing script
├── quick_start.sh                        # Automated setup script
└── tool_config.yaml                      # Tool configuration
```

## Configuration

### Model Options

For small models (recommended for WebShop):
- `Qwen/Qwen2.5-1.5B-Instruct` - Good balance of capability and speed
- `Qwen/Qwen2.5-3B-Instruct` - Better performance, needs more compute
- `Qwen/Qwen3-0.6B` - Smallest option (if available)

For larger models (if compute allows):
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen3-8B`

### Training Hyperparameters

Key parameters in the training script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_BATCH_SIZE` | 64 | Prompts per global step |
| `ROLLOUT_N` | 4 | Rollouts per prompt (group size) |
| `ACTOR_LR` | 5e-7 | Learning rate |
| `MAX_PROMPT_LENGTH` | 2048 | Maximum prompt tokens |
| `MAX_RESPONSE_LENGTH` | 4096 | Maximum response tokens |
| `MAX_STEPS_PER_TASK` | 10 | Max environment steps per task |

### GRPO-Specific Settings

```bash
# Key GRPO settings
algorithm.adv_estimator=grpo           # Use GRPO advantage estimator
algorithm.use_kl_in_reward=False       # Use KL loss instead
actor_rollout_ref.actor.use_kl_loss=True
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
```

### Multi-Turn Configuration

WebShop requires multi-turn interaction:

```bash
# Enable multi-turn
actor_rollout_ref.rollout.multi_turn.enable=True
actor_rollout_ref.rollout.multi_turn.function_tool_path=examples/webshop_grpo/webshop_tools.py
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=15
actor_rollout_ref.rollout.multi_turn.max_user_turns=1
actor_rollout_ref.rollout.multi_turn.format=hermes
```

## Tools

The training uses the following tools:

### FunctionTool API (webshop_tools.py)

- `webshop_search(query, task_id)` - Search for products
- `webshop_click(element, task_id)` - Click on products/buttons
- `webshop_buy(task_id)` - Purchase current product
- `webshop_get_status(task_id)` - Get session status
- `webshop_get_history(task_id)` - Get action history

### BaseTool API (webshop_tool.py)

Alternative implementation using the full BaseTool lifecycle:
- `create()` - Initialize WebShop session
- `execute()` - Execute actions
- `calc_reward()` - Calculate final reward
- `release()` - Clean up session

## Reward Function

The reward function (`reward_function.py`) evaluates:

1. **Purchase Success** (0.0 or 1.0) - Did the agent buy something?
2. **Goal Matching** (0.0-0.3) - Does the product match the requirements?
3. **Attribute Matching** (0.0-0.2) - Are specific attributes correct?
4. **Efficiency Bonus** (0.0-0.1) - Fewer steps = higher reward

Total reward range: 0.0 to 1.0

## Monitoring

### WandB Logging

Training progress is logged to WandB by default:

```bash
# View training logs
wandb login
# Logs will be available at: https://wandb.ai/<your-entity>/verl_grpo_webshop
```

### Console Output

Training also prints metrics to console:
- Episode rewards
- Success rates
- Step counts
- Loss values

## Advanced Usage

### Custom WebShop Configuration

To use a custom WebShop server:

```bash
WEBSHOP_SERVER=http://your-server:3000 \
bash examples/webshop_grpo/run_qwen2_5_2b_webshop_grpo.sh
```

### LoRA Training

For memory-efficient training with LoRA:

```bash
# Add LoRA configuration
actor_rollout_ref.model.lora.r=16
actor_rollout_ref.model.lora.alpha=32
actor_rollout_ref.model.lora.target_modules=all
```

### Multi-Node Training

For distributed training across multiple nodes:

```bash
NNODES=2 \
NGPUS_PER_NODE=8 \
MASTER_ADDR=node0 \
MASTER_PORT=29500 \
bash examples/webshop_grpo/run_qwen2_5_2b_webshop_grpo.sh
>>>>>>> main
```

## Troubleshooting

<<<<<<< HEAD
### Training hangs

If training hangs, check:
1. Is the WebShop server running? (real mode only)
2. Is the data format correct?
3. Are there any error messages in the logs?

### Low rewards

If rewards are low:
1. Increase `ROLLOUT_N` for more exploration
2. Increase `TOTAL_EPOCHS` for more training
3. Check if the reward function is appropriate

### OOM (Out of Memory)

If you run out of memory:
1. Decrease `TRAIN_BATCH_SIZE`
2. Decrease `MICRO_BATCH_SIZE`
3. Decrease `ROLLOUT_GPU_MEM_UTIL`
4. Enable `actor_rollout_ref.actor.fsdp_config.param_offload=True`

## References

- [WebShop Paper](https://arxiv.org/abs/2207.01206)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [verl Documentation](https://verl.readthedocs.io/)
=======
### WebShop Server Not Starting

```bash
# Check if port is in use
lsof -i :3000

# Kill existing process
kill -9 <PID>

# Restart server
python -m webshop.run_server
```

### Out of Memory

Reduce batch size and sequence length:

```bash
TRAIN_BATCH_SIZE=32 \
MAX_PROMPT_LENGTH=1024 \
MAX_RESPONSE_LENGTH=2048 \
PPO_MAX_TOKEN_LEN_PER_GPU=8192 \
bash examples/webshop_grpo/run_qwen2_5_2b_webshop_grpo.sh
```

### Low Success Rate

1. Increase training data: `--num_tasks 2000`
2. Increase rollout count: `ROLLOUT_N=8`
3. Adjust learning rate: `ACTOR_LR=1e-6`
4. Increase training epochs: `TOTAL_EPOCHS=20`

## References

- [WebShop Paper](https://webshop-pnlpnlpnlp.github.io/)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [verl Documentation](https://verl.readthedocs.io/)

## Citation

If you use this code, please cite:

```bibtex
@article{yao2023webshop,
  title={WebShop: Towards Scalable Real-World Web Shopping},
  author={Yao, Shunyu and others},
  journal={arXiv preprint arXiv:2207.01206},
  year={2023}
}
```
>>>>>>> main
