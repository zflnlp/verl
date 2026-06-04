# ScienceWorld GRPO Training

This example demonstrates GRPO (Group Relative Policy Optimization) training on the [ScienceWorld](https://scienceworld.github.io/) benchmark using the verl framework.

## Overview

ScienceWorld is a text-based science simulation environment where agents perform experiments and manipulate objects to complete science tasks. The agent must:

- Navigate a virtual laboratory
- Interact with scientific equipment
- Follow scientific procedures
- Complete experiments to achieve task goals

## Quick Start

### 1. Generate Training Data

```bash
python examples/scienceworld_grpo/data_preprocess.py \
    --local_save_dir ~/data/scienceworld \
    --num_tasks 100 \
    --seed 42
```

### 2. Run Mock Training (No Server Required)

```bash
bash examples/scienceworld_grpo/run_mock.sh
```

This uses a mock environment for testing the training pipeline without requiring a real ScienceWorld server.

### 3. Run Real Training (Requires ScienceWorld Server)

First, start the ScienceWorld server:

```bash
# Install ScienceWorld
pip install scienceworld

# Start the server
python -m scienceworld.server --port 8080
```

Then run training:

```bash
bash examples/scienceworld_grpo/run_real.sh
```

## Environment Actions

The ScienceWorld environment supports the following actions:

| Action | Description | Example |
|--------|-------------|---------|
| `look around` | Describe current room | `look around` |
| `examine` | Examine an object closely | `examine beaker` |
| `open` | Open a container | `open cabinet` |
| `close` | Close a container | `close door` |
| `take` | Pick up an object | `take flask from table` |
| `put` | Place an object | `put chemical in beaker` |
| `use` | Use an object | `use thermometer on water` |
| `toggle` | Turn on/off | `toggle stove` |
| `pour` | Pour a liquid | `pour water into pot` |
| `mix` | Mix contents | `mix solution` |
| `go to` | Move to location | `go to workbench` |
| `wait` | Wait | `wait` |
| `task` | Show current task | `task` |

## Task Categories

ScienceWorld includes tasks across multiple science domains:

- **Chemistry**: Mixing chemicals, identifying acids/bases, crystallization
- **Biology**: Growing plants, seed germination, cell observation
- **Physics**: Boiling water, circuit building, magnetic fields
- **Earth Science**: Rock identification, water cycle, weather measurement

## Model Configuration

Default model: `/workspace/models/Qwen3-1.7B`

To use a different model:

```bash
MODEL_PATH=/path/to/your/model bash examples/scienceworld_grpo/run_mock.sh
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

- **`ScienceWorldInteraction`** (`verl/interactions/scienceworld_interaction.py`): Manages multi-turn interactions with the ScienceWorld environment
- **`ScienceWorldTool`** (`verl/tools/scienceworld_tool.py`): Provides the action tool for environment interaction
- **`reward_function.py`**: Computes rewards based on task completion and action diversity

## File Structure

```
examples/scienceworld_grpo/
├── README.md
├── data_preprocess.py      # Data generation script
├── reward_function.py      # Reward computation
├── run_mock.sh             # Mock training script
├── run_real.sh             # Real training script (TODO)
└── config/
    ├── scienceworld_grpo.yaml              # Main training config
    ├── scienceworld_interaction_config.yaml # Interaction config
    └── scienceworld_tool_config.yaml       # Tool config
```

## Expected Behavior

In mock mode:
- The agent receives simulated laboratory observations
- Actions return canned responses
- Rewards are computed based on action diversity
- Training completes without external dependencies

In real mode:
- The agent interacts with the actual ScienceWorld environment
- Observations come from the ScienceWorld server
- Rewards are based on actual task completion
- Requires the ScienceWorld Python package and server

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure verl is installed: `pip install -e .`
2. **CUDA Out of Memory**: Reduce `MICRO_BATCH_SIZE` or `ROLLOUT_N`
3. **Data Not Found**: Run `data_preprocess.py` first with the correct `--local_save_dir`

### Debug Mode

To enable verbose logging:

```bash
VERL_LOGGING_LEVEL=DEBUG bash examples/scienceworld_grpo/run_mock.sh
```
