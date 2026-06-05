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

Supports two modes:
1. Mock mode (--use_mock): Generates synthetic data for pipeline testing.
2. Real mode (--task_name boil --num_variations 10): Uses ScienceWorld API
   to generate data from real task variations.

Usage (mock):
    python examples/scienceworld_grpo/data_preprocess.py \
        --local_save_dir ~/data/scienceworld --num_tasks 100

Usage (real):
    python examples/scienceworld_grpo/data_preprocess.py \
        --local_save_dir ~/data/scienceworld_real \
        --task_name boil --num_variations 10 --use_real_env
"""

import argparse
import json
import os
import random

import pandas as pd


# ScienceWorld tasks with their variations (from README)
SCIENCEWORLD_TASKS = {
    "boil": {"variations": 30, "goal": "Boil a liquid using appropriate laboratory equipment"},
    "melt": {"variations": 30, "goal": "Melt a solid substance by applying heat"},
    "freeze": {"variations": 30, "goal": "Freeze a liquid by lowering its temperature"},
    "change-the-state-of-matter-of": {"variations": 30, "goal": "Change the state of matter of a substance"},
    "use-thermometer": {"variations": 540, "goal": "Use a thermometer to measure temperature"},
    "power-component": {"variations": 20, "goal": "Power an electrical component using a battery"},
    "test-conductivity": {"variations": 900, "goal": "Test whether materials are electrical conductors"},
    "find-living-thing": {"variations": 300, "goal": "Find and identify a living thing in the environment"},
    "find-non-living-thing": {"variations": 300, "goal": "Find and identify a non-living thing"},
    "find-plant": {"variations": 300, "goal": "Find and identify a plant"},
    "find-animal": {"variations": 300, "goal": "Find and identify an animal"},
    "grow-plant": {"variations": 126, "goal": "Grow a plant from a seed and observe its growth"},
    "grow-fruit": {"variations": 126, "goal": "Grow a plant that produces fruit"},
    "chemistry-mix": {"variations": 32, "goal": "Mix chemicals to create a chemical reaction"},
    "identify-life-stages-1": {"variations": 14, "goal": "Identify the life stages of an organism"},
    "inclined-plane-determine-angle": {"variations": 168, "goal": "Determine the angle of an inclined plane"},
}


def generate_mock_dataset(num_tasks: int, seed: int) -> list:
    """Generate a mock dataset of ScienceWorld tasks."""
    random.seed(seed)
    tasks = []
    task_names = list(SCIENCEWORLD_TASKS.keys())

    for i in range(num_tasks):
        task_name = random.choice(task_names)
        task_info = SCIENCEWORLD_TASKS[task_name]
        variation = random.randint(0, min(task_info["variations"] - 1, 29))
        tasks.append({
            "task_id": f"task_{seed + i}",
            "task_name": task_name,
            "variation": variation,
            "goal": task_info["goal"],
        })
    return tasks


def generate_real_dataset(task_name: str, num_variations: int, seed: int) -> list:
    """Generate dataset from real ScienceWorld task variations.

    Uses the ScienceWorld API to get real task descriptions.
    Falls back to predefined descriptions if API unavailable.
    """
    random.seed(seed)

    if task_name not in SCIENCEWORLD_TASKS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(SCIENCEWORLD_TASKS.keys())}")

    task_info = SCIENCEWORLD_TASKS[task_name]
    max_variations = task_info["variations"]
    num_variations = min(num_variations, max_variations)

    # Try to get real task descriptions from ScienceWorld API
    real_goals = {}
    try:
        from scienceworld import ScienceWorldEnv
        env = ScienceWorldEnv()
        for var_idx in range(num_variations):
            try:
                env.load(task_name, var_idx)
                real_goals[var_idx] = env.taskDescription()
            except Exception as e:
                print(f"Warning: Could not load variation {var_idx}: {e}")
        del env
    except ImportError:
        print("Warning: scienceworld package not installed, using default goals")
    except Exception as e:
        print(f"Warning: Could not initialize ScienceWorld: {e}")

    tasks = []
    for var_idx in range(num_variations):
        goal = real_goals.get(var_idx, task_info["goal"])
        tasks.append({
            "task_id": f"{task_name}_var{var_idx}",
            "task_name": task_name,
            "variation": var_idx,
            "goal": goal,
        })

    return tasks


def format_for_verl(tasks: list) -> pd.DataFrame:
    """Format tasks for verl training."""
    data = []

    for task in tasks:
        prompt = f"""You are an expert scientist working in a laboratory environment.
Your task is: {task['goal']}.
Task name: {task['task_name']}

Prior to this step, you have already taken 0 step(s).
Below are the most recent 0 observations and the corresponding actions you took:
(no history)

You are now at step 1 and your current observation is:
You are in a well-equipped science laboratory. There are workbenches with various equipment, chemical supplies, and scientific instruments. A sink is available for water.

Your admissible actions of the current situation are:
- look around: Describe the current room and visible objects
- examine <object>: Examine an object closely
- open <object>: Open a container or door
- close <object>: Close a container or door
- take <object> from <location>: Pick up an object
- put <object> in/on <location>: Place an object somewhere
- use <object> [on <object>]: Use an object (optionally on another)
- toggle <object>: Turn something on or off
- pour <object> into <object>: Pour a liquid
- mix <object>: Mix contents of a container
- go to <location>: Move to a different location
- look at <object>: Look at something specific
- wait: Wait for something to happen
- task: Describe the current task

Now it's your turn to take one action for the current step. You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the science task. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

        messages = [
            {"role": "system", "content": "You are an expert scientist working in a laboratory environment."},
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
    parser.add_argument("--num_variations", type=int, default=10,
                        help="Number of task variations to use (real mode)")
    parser.add_argument("--num_tasks", type=int, default=100,
                        help="Number of tasks to generate (mock mode)")
    parser.add_argument("--use_real_env", action="store_true",
                        help="Use real ScienceWorld API to get task descriptions")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of training data")

    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Generate tasks
    if args.use_real_env:
        print(f"Generating data from real ScienceWorld task: {args.task_name}")
        print(f"Using {args.num_variations} variations")
        tasks = generate_real_dataset(args.task_name, args.num_variations, args.seed)
    else:
        print(f"Generating {args.num_tasks} mock ScienceWorld tasks...")
        tasks = generate_mock_dataset(args.num_tasks, args.seed)

    # Split into train and test
    random.seed(args.seed)
    random.shuffle(tasks)
    split_idx = int(len(tasks) * args.train_ratio)
    train_tasks = tasks[:split_idx]
    test_tasks = tasks[split_idx:]

    # Format for verl
    train_df = format_for_verl(train_tasks)
    test_df = format_for_verl(test_tasks)

    # Save
    train_path = os.path.join(save_dir, "train.parquet")
    test_path = os.path.join(save_dir, "test.parquet")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    metadata_path = os.path.join(save_dir, "task_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "mode": "real" if args.use_real_env else "mock",
            "task_name": args.task_name if args.use_real_env else "mixed",
            "num_variations": args.num_variations if args.use_real_env else None,
            "num_tasks": args.num_tasks if not args.use_real_env else None,
            "seed": args.seed,
            "train_size": len(train_tasks),
            "test_size": len(test_tasks),
        }, f, indent=2)

    print(f"Generated {len(train_tasks)} training tasks and {len(test_tasks)} test tasks")
    print(f"Saved to: {save_dir}")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"  - {metadata_path}")


if __name__ == "__main__":
    main()
