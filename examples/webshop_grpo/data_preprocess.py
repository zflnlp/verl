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
Data preprocessing script for WebShop GRPO training.

This script generates training and test data for WebShop tasks.
Each data point contains a task description and expected attributes.

Usage:
    python examples/webshop_grpo/data_preprocess.py \
        --local_save_dir ~/data/webshop \
        --num_tasks 100 \
        --seed 42
"""

import argparse
import json
import os
import random

import pandas as pd


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

    Args:
        num_tasks: Number of tasks to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of task dictionaries.
    """
    random.seed(seed)
    tasks = []

    for i in range(num_tasks):
        category = random.choice(CATEGORIES)
        task = generate_task(category, seed + i)
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
        prompt = f"""You are a shopping assistant. Your task is to help the user find and purchase a product.

Task: Find {task['goal']}

You can use the following actions:
- search[query]: Search for products
- click[item_id]: View product details
- click[buy]: Purchase the current product

Please complete this task by searching for and purchasing the appropriate product."""

        # Format as chat messages
        messages = [
            {"role": "system", "content": "You are a helpful shopping assistant."},
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
    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Generate tasks
    print(f"Generating {args.num_tasks} WebShop tasks...")
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


if __name__ == "__main__":
    main()
