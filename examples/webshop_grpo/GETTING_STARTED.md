# WebShop GRPO Training - Getting Started Guide

## Overview

This guide will help you set up and run GRPO training for Qwen2.5 models on the WebShop shopping environment.

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU(s) with at least 16GB VRAM
- verl framework installed

## Installation

```bash
# 1. Clone verl repository (if not already done)
git clone https://github.com/verl-project/verl.git
cd verl

# 2. Install verl
pip install -e .

# 3. Install WebShop (optional, for real environment training)
pip install webshop

# 4. Install additional dependencies
pip install wandb  # For logging (optional)
```

## Step-by-Step Training

### Step 1: Generate Training Data

```bash
# Generate 1000 WebShop tasks
python examples/webshop_grpo/data_preprocess.py \
    --local_save_dir ~/data/webshop \
    --num_tasks 1000 \
    --train_ratio 0.8 \
    --seed 42

# This creates:
# - ~/data/webshop/train.parquet
# - ~/data/webshop/test.parquet
# - ~/data/webshop/task_metadata.json
```

### Step 2: Choose Training Mode

#### Option A: Mock Mode (Recommended for Testing)

No WebShop server required. Uses simulated environment.

```bash
# Run training with mock tools
bash examples/webshop_grpo/run_qwen2_5_2b_webshop_grpo_mock.sh
```

#### Option B: Real WebShop Mode

Requires WebShop server running.

```bash
# Terminal 1: Start WebShop server
python -m webshop.run_server

# Terminal 2: Run training
bash examples/webshop_grpo/run_qwen2_5_2b_webshop_grpo.sh
```

### Step 3: Monitor Training

Training logs are saved to:
- Console output
- WandB (if configured)
- Local log files

```bash
# View WandB logs (if using WandB)
wandb sync wandb/

# Or view local logs
tail -f logs/training.log
```

### Step 4: Evaluate Trained Model

```bash
# Find your checkpoint
ls checkpoints/webshop_grpo/

# Evaluate the model
python examples/webshop_grpo/evaluate.py \
    --model_path checkpoints/webshop_grpo/global_step_100/actor \
    --num_episodes 100 \
    --output_dir results/webshop_eval

# View results
cat results/webshop_eval/eval_summary.txt
```

## Configuration Examples

### Small Model (Fast Training)

```bash
MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct \
TRAIN_BATCH_SIZE=32 \
ROLLOUT_N=4 \
NGPUS_PER_NODE=2 \
bash examples/webshop_grpo/run_qwen2_5_2b_webshop_grpo_mock.sh
```

### Medium Model (Better Performance)

```bash
MODEL_PATH=Qwen/Qwen2.5-3B-Instruct \
TRAIN_BATCH_SIZE=64 \
ROLLOUT_N=8 \
NGPUS_PER_NODE=4 \
ACTOR_LR=1e-6 \
bash examples/webshop_grpo/run_qwen2_5_2b_webshop_grpo.sh
```

### Large Model (Best Performance, More Compute)

```bash
MODEL_PATH=Qwen/Qwen2.5-7B-Instruct \
TRAIN_BATCH_SIZE=128 \
ROLLOUT_N=8 \
NGPUS_PER_NODE=8 \
NNODES=2 \
ACTOR_LR=5e-7 \
bash examples/webshop_grpo/run_qwen2_5_2b_webshop_grpo.sh
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size and sequence length

```bash
TRAIN_BATCH_SIZE=16 \
MAX_PROMPT_LENGTH=1024 \
MAX_RESPONSE_LENGTH=2048 \
PPO_MAX_TOKEN_LEN_PER_GPU=8192 \
bash examples/webshop_grpo/run_qwen2_5_2b_webshop_grpo_mock.sh
```

### Issue: WebShop Server Won't Start

**Solution**: Check if port is in use

```bash
# Check port 3000
lsof -i :3000

# Kill any existing process
kill -9 <PID>

# Try starting again
python -m webshop.run_server --port 3001
```

### Issue: Low Training Reward

**Solutions**:
1. Increase training data: `--num_tasks 2000`
2. Increase rollout count: `ROLLOUT_N=8`
3. Adjust learning rate: `ACTOR_LR=1e-6`
4. Train longer: `TOTAL_EPOCHS=20`
5. Increase entropy for exploration: `ENTROPY_COEFF=0.02`

### Issue: Import Errors

**Solution**: Ensure verl is installed

```bash
cd /path/to/verl
pip install -e .
```

## File Reference

| File | Purpose |
|------|---------|
| `webshop_tool.py` | BaseTool implementation for real WebShop |
| `webshop_tools.py` | FunctionTool wrappers for real WebShop |
| `mock_webshop_tools.py` | Mock tools (no server needed) |
| `reward_function.py` | Reward computation |
| `data_preprocess.py` | Data generation |
| `run_qwen2_5_2b_webshop_grpo.sh` | Real WebShop training |
| `run_qwen2_5_2b_webshop_grpo_mock.sh` | Mock training |
| `evaluate.py` | Model evaluation |
| `test_tools.py` | Tool testing |
| `quick_start.sh` | Automated setup |

## Next Steps

After training:

1. **Evaluate**: Run evaluation to measure success rate
2. **Iterate**: Adjust hyperparameters based on results
3. **Scale**: Try larger models or more training data
4. **Deploy**: Use the trained model for real shopping tasks

## Support

For issues with:
- **verl framework**: See [verl documentation](https://verl.readthedocs.io/)
- **WebShop**: See [WebShop repository](https://webshop-pnlpnlpnlp.github.io/)
- **This training setup**: Check the README.md in this directory

## Citation

If you use this setup, please cite:

```bibtex
@article{shao2024deepseekmath,
  title={DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author={Shao, Zhihong and others},
  journal={arXiv preprint arXiv:2402.03300},
  year={2024}
}
```
