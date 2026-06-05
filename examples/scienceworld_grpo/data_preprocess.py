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

This script generates training and test data for ScienceWorld tasks.
Each data point contains a task description and expected completion criteria.

Usage:
    python examples/scienceworld_grpo/data_preprocess.py \
        --local_save_dir ~/data/scienceworld \
        --num_tasks 100 \
        --seed 42
"""

import argparse
import json
import os
import random

import pandas as pd


# ScienceWorld task categories
TASK_CATEGORIES = {
    "chemistry": [
        {"name": "chemistry-mix", "goal": "Mix two chemicals to create a chemical reaction", "difficulty": "easy"},
        {"name": "identify-acid-base", "goal": "Identify whether a substance is an acid or a base using pH paper", "difficulty": "medium"},
        {"name": "crystallization", "goal": "Create crystals by evaporating a supersaturated solution", "difficulty": "hard"},
    ],
    "biology": [
        {"name": "growing-plants", "goal": "Grow a plant from a seed and measure its growth", "difficulty": "easy"},
        {"name": "germination-test", "goal": "Test which conditions affect seed germination", "difficulty": "medium"},
        {"name": "cell-observation", "goal": "Observe plant cells under a microscope", "difficulty": "hard"},
    ],
    "physics": [
        {"name": "boiling-water", "goal": "Boil water and measure its temperature changes", "difficulty": "easy"},
        {"name": "circuit-building", "goal": "Build a simple circuit to light a bulb", "difficulty": "medium"},
        {"name": "magnetic-fields", "goal": "Explore magnetic field patterns with iron filings", "difficulty": "hard"},
    ],
    "earth-science": [
        {"name": "rock-identification", "goal": "Identify different types of rocks based on their properties", "difficulty": "easy"},
        {"name": "water-cycle", "goal": "Demonstrate the water cycle using a terrarium", "difficulty": "medium"},
        {"name": "weather-measurement", "goal": "Measure and record weather data using scientific instruments", "difficulty": "hard"},
    ],
}

# Difficulty to numeric mapping
DIFFICULTY_SCORE = {"easy": 0.3, "medium": 0.6, "hard": 0.9}


def generate_task(category: str, task_info: dict, seed: int) -> dict:
    """Generate a single ScienceWorld task.

    Args:
        category: The task category (chemistry, biology, etc.).
        task_info: Task information dictionary.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing task information.
    """
    random.seed(seed)

    return {
        "task_id": f"task_{seed}",
        "task_name": task_info["name"],
        "category": category,
        "goal": task_info["goal"],
        "difficulty": task_info["difficulty"],
        "difficulty_score": DIFFICULTY_SCORE[task_info["difficulty"]],
        "max_steps": random.randint(10, 25),
    }


def generate_dataset(num_tasks: int, seed: int) -> list:
    """Generate a dataset of ScienceWorld tasks.

    Args:
        num_tasks: Number of tasks to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of task dictionaries.
    """
    random.seed(seed)
    tasks = []

    # Flatten task list
    all_tasks = []
    for category, task_list in TASK_CATEGORIES.items():
        for task_info in task_list:
            all_tasks.append((category, task_info))

    # Sample tasks (with replacement if needed)
    for i in range(num_tasks):
        category, task_info = random.choice(all_tasks)
        task = generate_task(category, task_info, seed + i)
        tasks.append(task)

    return tasks


def format_for_verl(tasks: list) -> pd.DataFrame:
    """Format tasks for verl training.

    Args:
        tasks: List of task dictionaries.

    Returns:
        DataFrame formatted for verl.
    """
    data = []

    for task in tasks:
        # Create the prompt (user message)
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

        # Format as chat messages
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
                "category": task["category"],
                "difficulty": task["difficulty"],
            },
        })

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="Generate ScienceWorld training data")
    parser.add_argument("--local_save_dir", type=str, default="~/data/scienceworld",
                        help="Directory to save the generated data")
    parser.add_argument("--num_tasks", type=int, default=100,
                        help="Number of tasks to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of training data")

    args = parser.parse_args()

    # Expand home directory
    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Generate tasks
    print(f"Generating {args.num_tasks} ScienceWorld tasks...")
    tasks = generate_dataset(args.num_tasks, args.seed)

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

    # Also save task metadata for reference
    metadata_path = os.path.join(save_dir, "task_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "num_tasks": args.num_tasks,
            "seed": args.seed,
            "train_size": len(train_tasks),
            "test_size": len(test_tasks),
            "categories": list(TASK_CATEGORIES.keys()),
            "tasks": [t["task_name"] for t in train_tasks[:5]],  # Sample
        }, f, indent=2)

    print(f"Generated {len(train_tasks)} training tasks and {len(test_tasks)} test tasks")
    print(f"Saved to: {save_dir}")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"  - {metadata_path}")


if __name__ == "__main__":
    main()
