# ALFWorld GRPO Training

This example demonstrates GRPO (Group Relative Policy Optimization) training on the [ALFWorld](https://alfworld.github.io/) benchmark using the verl framework.

## Overview

ALFWorld is a text-game environment that aligns with the ALFRED benchmark for embodied instruction following. Agents must complete household tasks by navigating rooms, manipulating objects, and using household appliances. The environment has 6 task types requiring different sequences of actions.

## Quick Start

### 1. Install ALFWorld (for real training)

```bash
bash examples/alfworld_grpo/setup_alfworld.sh
```

### 2. Generate Training Data

```bash
# Mock data (no ALFWorld installation needed)
python examples/alfworld_grpo/data_preprocess.py \
    --local_save_dir /workspace/data/alfworld \
    --num_tasks 100 --seed 42

# Real data (requires ALFWorld installation)
python examples/alfworld_grpo/data_preprocess.py \
    --local_save_dir /workspace/data/alfworld_real \
    --use_real_env --task_type pick_and_place --num_games 10
```

### 3. Run Mock Training (No ALFWorld Required)

```bash
bash examples/alfworld_grpo/run_mock.sh
```

This uses a mock environment for testing the training pipeline without requiring ALFWorld.

### 4. Run Real Training

```bash
bash examples/alfworld_grpo/run_real.sh
```

## Task Types

ALFWorld includes 6 task types:

| Task Type | Description | Required Actions |
|-----------|-------------|-----------------|
| `pick_and_place` | Pick up object, place in receptacle | take → put |
| `pick_clean_then_place` | Pick → clean → place | take → clean → put |
| `pick_heat_then_place` | Pick → heat → place | take → heat → put |
| `pick_cool_then_place` | Pick → cool → place | take → cool → put |
| `look_at_obj_in_light` | Examine object under light | take → use → examine |
| `pick_two_obj` | Pick up two instances of same object | take → put (×2) |

## Environment Actions

| Action | Description | Example |
|--------|-------------|---------|
| `look` | Describe current room | `look` |
| `inventory` | Check carried objects | `inventory` |
| `go to` | Navigate to location | `go to countertop 1` |
| `take` | Pick up object | `take apple from countertop 1` |
| `put` | Place object | `put apple in fridge 1` |
| `open` | Open container | `open fridge 1` |
| `close` | Close container | `close fridge 1` |
| `toggle` | Toggle object state | `toggle lamp 1` |
| `use` | Use an object | `use desklamp 1` |
| `heat` | Heat in microwave | `heat apple 1` |
| `clean` | Clean in sink | `clean apple 1` |
| `cool` | Cool in fridge | `cool apple 1` |
| `examine` | Examine object closely | `examine apple 1` |

## Model Configuration

Default model: `/workspace/models/Qwen3-1.7B`

To use a different model:

```bash
MODEL_PATH=/path/to/your/model bash examples/alfworld_grpo/run_mock.sh
```

## Hyperparameters

Key hyperparameters (can be overridden via environment variables):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAIN_BATCH_SIZE` | 32 | Training batch size |
| `MICRO_BATCH_SIZE` | 8 | Micro batch size per GPU |
| `MAX_PROMPT_LENGTH` | 2048 | Maximum prompt length |
| `MAX_RESPONSE_LENGTH` | 4096 | Maximum response length |
| `ROLLOUT_N` | 4 | Number of rollouts per prompt |
| `ACTOR_LR` | 1e-6 | Actor learning rate |
| `KL_LOSS_COEF` | 0.001 | KL loss coefficient |
| `TOTAL_EPOCHS` | 1 | Number of training epochs |

## Architecture

The implementation consists of:

- **`AlfworldInteraction`** (`verl/interactions/alfworld_interaction.py`): Manages multi-turn interactions with the ALFWorld environment
- **`AlfworldTool`** (`verl/tools/alfworld_tool.py`): Provides the action tool for environment interaction
- **`reward_function.py`**: Computes rewards based on task completion (binary from env) and action-sequence heuristics (fallback)

## File Structure

```
examples/alfworld_grpo/
├── README.md
├── data_preprocess.py      # Data generation script
├── reward_function.py      # Reward computation
├── run_mock.sh             # Mock training script
├── run_real.sh             # Real training script
├── eval_zero_shot.py       # Zero-shot evaluation
├── setup_alfworld.sh       # ALFWorld installation script
└── config/
    ├── alfworld_grpo.yaml              # Main training config
    ├── alfworld_interaction_config.yaml # Interaction config
    └── alfworld_tool_config.yaml       # Tool config
```

## Zero-Shot Evaluation

Evaluate a model without training:

```bash
python examples/alfworld_grpo/eval_zero_shot.py \
    --model_path /workspace/models/Qwen3-1.7B \
    --task_type pick_and_place \
    --num_games 10 \
    --max_steps 30
```

## Expected Behavior

In mock mode:
- The agent receives simulated household scene observations
- Actions return canned responses
- Rewards are computed based on action diversity and task-type matching
- Training completes without external dependencies

In real mode:
- The agent interacts with the actual ALFWorld TextWorld environment
- Observations come from the ALFWorld game engine
- Rewards are binary (1.0 if task won, 0.0 otherwise)
- Requires the `alfworld` Python package and downloaded game files

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure verl is installed: `pip install -e .`
2. **CUDA Out of Memory**: Reduce `MICRO_BATCH_SIZE` or `ROLLOUT_N`
3. **Data Not Found**: Run `data_preprocess.py` first with the correct `--local_save_dir`
4. **ALFWorld Not Found**: Run `bash examples/alfworld_grpo/setup_alfworld.sh`

### Debug Mode

To enable verbose logging:

```bash
VERL_LOGGING_LEVEL=DEBUG bash examples/alfworld_grpo/run_mock.sh
```
