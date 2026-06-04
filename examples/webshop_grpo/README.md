# WebShop GRPO Training

This directory contains scripts for training a shopping agent using GRPO (Group Relative Policy Optimization) on the WebShop environment.

## Overview

WebShop is a simulated e-commerce environment where an agent needs to:
1. Search for products based on user instructions
2. Browse product listings
3. Select and purchase the appropriate product

## Quick Start (Mock Mode)

Mock mode allows you to test the training pipeline without a real WebShop server.

### 1. Generate Training Data

```bash
python examples/webshop_grpo/data_preprocess.py \
    --local_save_dir ~/data/webshop \
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

## File Structure

```
examples/webshop_grpo/
├── config/
│   ├── webshop_interaction_config.yaml   # Interaction config (mock)
│   └── webshop_tool_config.yaml          # Tool config
├── data_preprocess.py                    # Data generation script
├── run_mock.sh                          # Mock training script
├── run_real.sh                          # Real training script
└── README.md                            # This file
```

## Troubleshooting

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
