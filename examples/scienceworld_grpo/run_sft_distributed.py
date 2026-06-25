#!/usr/bin/env python3
"""
Distributed SFT training for ScienceWorld using llama-factory.

Usage:
    CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 examples/scienceworld_grpo/run_sft_distributed.py
"""

import os
import subprocess
import sys


def main():
    # Get environment variables
    model_path = os.environ.get("MODEL_PATH", "/workspace/models/Qwen3-1.7B")
    output_dir = os.environ.get("OUTPUT_DIR", "/workspace/models/Qwen3-1.7B-SFT")
    data_dir = os.environ.get("DATA_DIR", "/workspace/data/scienceworld_sft")
    epochs = os.environ.get("EPOCHS", "3")
    batch_size = os.environ.get("BATCH_SIZE", "4")
    learning_rate = os.environ.get("LEARNING_RATE", "0.00002")
    max_seq_length = os.environ.get("MAX_SEQ_LENGTH", "8192")

    # Create config file
    config_content = f"""### model
model_name_or_path: {model_path}

### method
stage: sft
do_train: true
finetuning_type: full

### dataset
dataset_dir: {data_dir}
dataset: scienceworld_train
template: qwen3
cutoff_len: {max_seq_length}
max_samples: 10000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: {output_dir}
logging_steps: 10
save_steps: 500
save_total_limit: 3

### train
per_device_train_batch_size: {batch_size}
gradient_accumulation_steps: 4
learning_rate: {learning_rate}
num_train_epochs: {epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
ddp_timeout: 180000000
gradient_checkpointing: true

### FSDP configuration
fsdp: "full_shard auto_wrap"
fsdp_config:
  fsdp_offload_params: false
  fsdp_backward_prefetch: "backward_pre"
  fsdp_forward_prefetch: false
  fsdp_use_orig_params: true
  fsdp_cpu_ram_efficient_loading: true
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: "Qwen2DecoderLayer"

### eval
val_size: 0.1
per_device_eval_batch_size: 8
eval_strategy: steps
eval_steps: 500
"""

    # Write config to llama-factory directory
    config_path = os.path.join("llama-factory", "examples", "train_full", "scienceworld_sft.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        f.write(config_content)

    print(f"Config written to: {config_path}")
    print(f"Starting distributed training...")

    # Run llamafactory-cli train
    cmd = ["llamafactory-cli", "train", config_path]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
