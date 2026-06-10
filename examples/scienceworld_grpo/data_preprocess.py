# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Data preprocessing script for ScienceWorld GRPO training.

Uses ScienceWorld's built-in train/dev/test splits.

Usage:
    python examples/scienceworld_grpo/data_preprocess.py \
        --local_save_dir /workspace/data/scienceworld_real \
        --task_name boil --use_real_env
"""

import argparse
import json
import os
import random

import pandas as pd


def generate_real_dataset_with_splits(task_name: str) -> dict:
    """Generate dataset from real ScienceWorld with built-in train/dev/test splits.

    Uses ScienceWorld's get_variations_train/dev/test methods.

    Returns:
        Dictionary with 'train', 'dev', 'test' keys, each containing a list of task dicts.
    """
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv()

    # Must load a task first to initialize variations
    env.load(task_name, 0)

    splits = {}
    for split_name, get_fn in [
        ("train", env.get_variations_train),
        ("dev", env.get_variations_dev),
        ("test", env.get_variations_test),
    ]:
        variations = get_fn()
        tasks = []
        for var_idx in variations:
            try:
                env.load(task_name, var_idx)
                goal = env.taskdescription()
            except Exception as e:
                print(f"Warning: Could not load {task_name} var {var_idx}: {e}")
                goal = f"Complete the {task_name} task"

            tasks.append({
                "task_id": f"{task_name}_var{var_idx}",
                "task_name": task_name,
                "variation": var_idx,
                "goal": goal,
            })
        splits[split_name] = tasks
        print(f"  {split_name}: {len(tasks)} variations")

    del env
    return splits


def format_for_verl(tasks: list) -> pd.DataFrame:
    """Format tasks for verl training."""
    data = []

    for task in tasks:
        prompt = f"""Your ScienceWorld task is: {task['goal']}.
Prior to this step, you have already taken 0 step(s). Below are the most recent 0 observations and the corresponding actions you took: (no history)
You are now at step 1 and your current observation is: You are in a well-equipped science laboratory. There are workbenches with various equipment, chemical supplies, and scientific instruments. A sink is available for water.
Your valid actions of the current situation are: [look around, examine <object>, open <object>, close <object>, take <object> from <location>, put <object> in/on <location>, use <object> [on <object>], toggle <object>, pour <object> into <object>, mix <object>, go to <location>, look at <object>, wait, task].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <thought> tags.
Once you've finished your reasoning, you should choose a valid action for the current step and present it within <action> </action> tags."""

        messages = [
            {"role": "user", "content": prompt},
        ]

        data.append({
            "data_source": "scienceworld",
            "prompt": messages,
            "ability": "science",
            "reward_model": {
                "ground_truth": task,
            },
            "extra_info": {
                "task_name": task["task_name"],
                "variation": task.get("variation", 0),
            },
        })

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="Generate ScienceWorld training data")
    parser.add_argument("--local_save_dir", type=str, default="~/data/scienceworld",
                        help="Directory to save the generated data")
    parser.add_argument("--task_name", type=str, default="boil",
                        help="ScienceWorld task name (e.g. boil, melt, find-living-thing)")
    parser.add_argument("--use_real_env", action="store_true",
                        help="Use real ScienceWorld API with built-in train/dev/test splits")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    if not args.use_real_env:
        print("Error: This script requires --use_real_env flag for proper train/dev/test splits.")
        print("Usage: python data_preprocess.py --local_save_dir /workspace/data/scienceworld_real --task_name boil --use_real_env")
        return

    print(f"Generating data from real ScienceWorld task: {args.task_name}")
    print("Using ScienceWorld's built-in train/dev/test splits...")

    splits = generate_real_dataset_with_splits(args.task_name)

    # Format for verl
    train_df = format_for_verl(splits["train"])
    val_df = format_for_verl(splits["dev"])
    test_df = format_for_verl(splits["test"])

    # Save
    train_path = os.path.join(save_dir, "train.parquet")
    val_path = os.path.join(save_dir, "val.parquet")
    test_path = os.path.join(save_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    metadata_path = os.path.join(save_dir, "task_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "task_name": args.task_name,
            "split_method": "scienceworld_builtin",
            "train_size": len(splits["train"]),
            "val_size": len(splits["dev"]),
            "test_size": len(splits["test"]),
            "train_variations": [t["variation"] for t in splits["train"]],
            "val_variations": [t["variation"] for t in splits["dev"]],
            "test_variations": [t["variation"] for t in splits["test"]],
        }, f, indent=2)

    print(f"\nGenerated:")
    print(f"  Train: {len(splits['train'])} tasks")
    print(f"  Dev:   {len(splits['dev'])} tasks")
    print(f"  Test:  {len(splits['test'])} tasks")
    print(f"\nSaved to: {save_dir}")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")
    print(f"  - {metadata_path}")


if __name__ == "__main__":
    main()
