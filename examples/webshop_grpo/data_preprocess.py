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
<<<<<<< HEAD
Data preprocessing script for WebShop GRPO training.

This script generates training and test data for WebShop tasks.
Each data point contains a task description and expected attributes.
=======
Preprocess WebShop dataset for verl GRPO training.

This script converts WebShop tasks into the parquet format required
by verl's RL training pipeline.
>>>>>>> main

Usage:
    python examples/webshop_grpo/data_preprocess.py \
        --local_save_dir ~/data/webshop \
<<<<<<< HEAD
        --num_tasks 100 \
        --seed 42
=======
        --num_tasks 1000 \
        --split train
>>>>>>> main
"""

import argparse
import json
import os
import random
<<<<<<< HEAD
=======
from typing import Any, Optional
>>>>>>> main

import pandas as pd


<<<<<<< HEAD
# Mock product categories and attributes
CATEGORIES = ["clothing", "shoes", "electronics", "home", "books"]

CLOTHING_ITEMS = [
    {"name": "Red T-Shirt", "color": "red", "size": ["S", "M", "L", "XL"], "price_range": (15, 30)},
    {"name": "Blue Jeans", "color": "blue", "size": ["28", "30", "32", "34"], "price_range": (40, 80)},
    {"name": "Black Hoodie", "color": "black", "size": ["M", "L", "XL"], "price_range": (30, 60)},
    {"name": "White Shirt", "color": "white", "size": ["S", "M", "L"], "price_range": (25, 50)},
    {"name": "Summer Dress", "color": "floral", "size": ["S", "M", "L"], "price_range": (35, 70)},
]

SHOE_ITEMS = [
    {"name": "Running Shoes", "color": "black", "size": ["8", "9", "10", "11"], "price_range": (60, 120)},
    {"name": "Casual Sneakers", "color": "white", "size": ["7", "8", "9", "10"], "price_range": (40, 80)},
    {"name": "Formal Shoes", "color": "brown", "size": ["8", "9", "10"], "price_range": (80, 150)},
]

ELECTRONICS_ITEMS = [
    {"name": "Wireless Headphones", "features": ["bluetooth", "noise-cancelling"], "price_range": (30, 100)},
    {"name": "Smart Watch", "features": ["fitness-tracking", "notifications"], "price_range": (50, 200)},
    {"name": "Portable Charger", "features": ["10000mAh", "fast-charging"], "price_range": (20, 50)},
]


def generate_task(category: str, seed: int) -> dict:
    """Generate a single WebShop task.

    Args:
        category: The product category.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing task information.
    """
    random.seed(seed)

    if category == "clothing":
        item = random.choice(CLOTHING_ITEMS)
        size = random.choice(item["size"])
        price = random.randint(item["price_range"][0], item["price_range"][1])
        goal = f"a {item['color']} {item['name']} in size {size} under ${price}"
        attributes = {
            "color": item["color"],
            "size": size,
            "max_price": price,
            "category": "clothing",
        }
    elif category == "shoes":
        item = random.choice(SHOE_ITEMS)
        size = random.choice(item["size"])
        price = random.randint(item["price_range"][0], item["price_range"][1])
        goal = f"{item['color']} {item['name']} in size {size} under ${price}"
        attributes = {
            "color": item["color"],
            "size": size,
            "max_price": price,
            "category": "shoes",
        }
    elif category == "electronics":
        item = random.choice(ELECTRONICS_ITEMS)
        price = random.randint(item["price_range"][0], item["price_range"][1])
        features = random.sample(item["features"], min(2, len(item["features"])))
        goal = f"{item['name']} with {', '.join(features)} under ${price}"
        attributes = {
            "features": features,
            "max_price": price,
            "category": "electronics",
        }
    else:
        # Default
        goal = "a product under $50"
        attributes = {"max_price": 50, "category": "other"}

    return {
        "task_id": f"task_{seed}",
        "goal": goal,
        "category": category,
        "attributes": attributes,
    }


def generate_dataset(num_tasks: int, seed: int) -> list:
    """Generate a dataset of WebShop tasks.
=======
def generate_webshop_tasks(num_tasks: int = 1000, seed: int = 42) -> list[dict[str, Any]]:
    """Generate WebShop task configurations.

    This function creates task configurations for WebShop training.
    In production, you would load these from the actual WebShop dataset.
>>>>>>> main

    Args:
        num_tasks: Number of tasks to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of task dictionaries.
    """
    random.seed(seed)
<<<<<<< HEAD
    tasks = []

    for i in range(num_tasks):
        category = random.choice(CATEGORIES)
        task = generate_task(category, seed + i)
=======

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
>>>>>>> main
        tasks.append(task)

    return tasks


<<<<<<< HEAD
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
        prompt = f"""You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task['goal']}.

Prior to this step, you have already taken 0 step(s).
Below are the most recent 0 observations and the corresponding actions you took:
(no history)

You are now at step 1 and your current observation is:
Welcome to WebShop! You can search for products and browse listings.

Your admissible actions of the current situation are:
- search[<query>]: Search for products using a text query
- click[<button name>]: Click on interactive elements (e.g., product links, filter buttons, pagination)

Now it's your turn to take one action for the current step. You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

        # Format as chat messages
        messages = [
            {"role": "system", "content": "You are an expert autonomous agent operating in the WebShop e-commerce environment."},
            {"role": "user", "content": prompt},
        ]

        data.append({
            "prompt": messages,
            "reward_model": {
                "ground_truth": task,
            },
        })

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="Generate WebShop training data")
    parser.add_argument("--local_save_dir", type=str, default="~/data/webshop",
                        help="Directory to save the generated data")
    parser.add_argument("--num_tasks", type=int, default=100,
                        help="Number of tasks to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of training data")

    args = parser.parse_args()

    # Expand home directory
=======
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
>>>>>>> main
    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Generate tasks
    print(f"Generating {args.num_tasks} WebShop tasks...")
<<<<<<< HEAD
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
            "categories": CATEGORIES,
        }, f, indent=2)

    print(f"Generated {len(train_tasks)} training tasks and {len(test_tasks)} test tasks")
    print(f"Saved to: {save_dir}")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"  - {metadata_path}")
=======
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
>>>>>>> main


if __name__ == "__main__":
    main()
