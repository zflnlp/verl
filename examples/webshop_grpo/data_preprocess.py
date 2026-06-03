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
Preprocess WebShop dataset for verl GRPO training.

This script converts WebShop tasks into the parquet format required
by verl's RL training pipeline.

Usage:
    python examples/webshop_grpo/data_preprocess.py \
        --local_save_dir ~/data/webshop \
        --num_tasks 1000 \
        --split train
"""

import argparse
import json
import os
import random
from typing import Any, Optional

import pandas as pd


def generate_webshop_tasks(num_tasks: int = 1000, seed: int = 42) -> list[dict[str, Any]]:
    """Generate WebShop task configurations.

    This function creates task configurations for WebShop training.
    In production, you would load these from the actual WebShop dataset.

    Args:
        num_tasks: Number of tasks to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of task dictionaries.
    """
    random.seed(seed)

    # Example product categories and attributes
    categories = [
        "clothing",
        "shoes",
        "electronics",
        "home",
        "beauty",
        "sports",
        "books",
        "toys",
    ]

    clothing_items = [
        {"name": "t-shirt", "color": ["red", "blue", "black", "white", "green"], "size": ["S", "M", "L", "XL"]},
        {"name": "dress", "color": ["red", "blue", "black", "pink", "yellow"], "size": ["S", "M", "L"]},
        {"name": "jacket", "color": ["black", "brown", "navy", "gray"], "size": ["M", "L", "XL"]},
        {"name": "pants", "color": ["black", "blue", "khaki", "gray"], "size": ["S", "M", "L", "XL"]},
    ]

    shoes_items = [
        {"name": "sneakers", "color": ["white", "black", "red", "blue"], "size": ["7", "8", "9", "10", "11"]},
        {"name": "boots", "color": ["black", "brown"], "size": ["7", "8", "9", "10"]},
        {"name": "sandals", "color": ["brown", "black", "white"], "size": ["6", "7", "8", "9"]},
    ]

    electronics_items = [
        {"name": "headphones", "color": ["black", "white", "red"], "feature": ["wireless", "noise-cancelling"]},
        {"name": "phone case", "color": ["black", "clear", "blue", "red"], "compatibility": ["iphone", "samsung"]},
        {"name": "charger", "color": ["white", "black"], "type": ["usb-c", "lightning", "micro-usb"]},
    ]

    tasks = []
    for i in range(num_tasks):
        # Select random category and item
        category = random.choice(categories)

        if category == "clothing":
            item = random.choice(clothing_items)
            color = random.choice(item["color"])
            size = random.choice(item["size"])
            goal = f"I want a {color} {item['name']} in size {size}"
            attributes = {"color": color, "size": size, "type": item["name"]}

        elif category == "shoes":
            item = random.choice(shoes_items)
            color = random.choice(item["color"])
            size = random.choice(item["size"])
            goal = f"I need {color} {item['name']} in size {size}"
            attributes = {"color": color, "size": size, "type": item["name"]}

        elif category == "electronics":
            item = random.choice(electronics_items)
            color = random.choice(item["color"])
            feature = random.choice(item.get("feature", ["standard"]))
            goal = f"Find me a {color} {item['name']} that is {feature}"
            attributes = {"color": color, "feature": feature, "type": item["name"]}

        else:
            # Generic task for other categories
            goal = f"Find a product in the {category} category"
            attributes = {"category": category}

        # Create task with price constraint (sometimes)
        if random.random() < 0.3:
            max_price = random.choice([20, 30, 50, 100])
            goal += f" under ${max_price}"
            attributes["max_price"] = max_price

        task = {
            "task_id": f"webshop_task_{i:04d}",
            "category": category,
            "goal": goal,
            "attributes": attributes,
            "difficulty": random.choice(["easy", "medium", "hard"]),
        }
        tasks.append(task)

    return tasks


def create_prompt(task: dict[str, Any]) -> list[dict[str, str]]:
    """Create the prompt messages for a WebShop task.

    Args:
        task: Task configuration dictionary.

    Returns:
        List of message dictionaries in chat format.
    """
    system_prompt = (
        "You are a shopping assistant agent. Your task is to help users find and purchase "
        "products from an online store.\n\n"
        "You can perform the following actions:\n"
        "- search[query]: Search for products using a query\n"
        "- click[button]: Click on a product or button (e.g., click[1] to select first product)\n"
        "- buy: Purchase the currently viewed product\n\n"
        "Guidelines:\n"
        "1. Start by searching for products matching the user's request\n"
        "2. Review the search results and click on promising items\n"
        "3. Check product details to ensure they match requirements\n"
        "4. Purchase the item that best matches the user's needs\n"
        "5. Be efficient - try to complete the task in few steps\n\n"
        "Always explain your reasoning before taking an action."
    )

    user_prompt = (
        f"Help me find and buy a product with the following requirements:\n\n"
        f"Task: {task['goal']}\n\n"
        f"Please search for suitable products, review options, and purchase the best match. "
        f"Use the available tools to complete this shopping task."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def preprocess_webshop_dataset(
    tasks: list[dict[str, Any]],
    split: str = "train",
) -> pd.DataFrame:
    """Preprocess WebShop tasks into verl-compatible format.

    Args:
        tasks: List of task dictionaries.
        split: Dataset split name ('train' or 'test').

    Returns:
        DataFrame in verl's required format.
    """
    data_records = []

    for idx, task in enumerate(tasks):
        prompt = create_prompt(task)

        record = {
            "data_source": "webshop",
            "agent_name": "tool_agent",  # Use tool agent loop
            "prompt": prompt,
            "ability": "shopping",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "task_id": task["task_id"],
                    "goal": task["goal"],
                    "category": task["category"],
                    "attributes": task["attributes"],
                },
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "task_id": task["task_id"],
                "difficulty": task["difficulty"],
                "need_tools_kwargs": True,
                "tools_kwargs": {
                    "webshop": {
                        "create_kwargs": {
                            "task_id": task["task_id"],
                            "env_config": {
                                "server": "http://localhost:3000",  # WebShop server
                            },
                            "max_steps": 10,
                        },
                    },
                },
            },
        }
        data_records.append(record)

    return pd.DataFrame(data_records)


def main():
    parser = argparse.ArgumentParser(description="Preprocess WebShop dataset for verl GRPO training")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/webshop",
        help="Directory to save preprocessed data",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=1000,
        help="Number of tasks to generate",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Which split to generate",
    )

    args = parser.parse_args()

    # Expand user directory
    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Generate tasks
    print(f"Generating {args.num_tasks} WebShop tasks...")
    all_tasks = generate_webshop_tasks(num_tasks=args.num_tasks, seed=args.seed)

    # Split into train/test
    random.seed(args.seed)
    random.shuffle(all_tasks)
    split_idx = int(len(all_tasks) * args.train_ratio)
    train_tasks = all_tasks[:split_idx]
    test_tasks = all_tasks[split_idx:]

    # Process and save
    if args.split in ["train", "both"]:
        print(f"Processing {len(train_tasks)} training tasks...")
        train_df = preprocess_webshop_dataset(train_tasks, split="train")
        train_path = os.path.join(save_dir, "train.parquet")
        train_df.to_parquet(train_path)
        print(f"Saved training data to {train_path}")

    if args.split in ["test", "both"]:
        print(f"Processing {len(test_tasks)} test tasks...")
        test_df = preprocess_webshop_dataset(test_tasks, split="test")
        test_path = os.path.join(save_dir, "test.parquet")
        test_df.to_parquet(test_path)
        print(f"Saved test data to {test_path}")

    # Save task metadata for reference
    metadata_path = os.path.join(save_dir, "task_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "num_tasks": args.num_tasks,
                "train_size": len(train_tasks),
                "test_size": len(test_tasks),
                "seed": args.seed,
                "categories": list(set(t["category"] for t in all_tasks)),
            },
            f,
            indent=2,
        )
    print(f"Saved metadata to {metadata_path}")
    print("Done!")


if __name__ == "__main__":
    main()
