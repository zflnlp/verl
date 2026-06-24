#!/usr/bin/env python3
"""
Prepare ScienceWorld gold trajectories for SFT with llama-factory.

This script:
1. Extracts gold trajectories from ScienceWorld benchmark
2. Converts them to llama-factory SFT format
3. Splits into train/dev/test sets

Usage:
    python examples/scienceworld_grpo/prepare_sft_data.py \
        --goldpaths_dir benchmarks/ScienceWorld/goldpaths \
        --output_dir llama-factory/data/scienceworld_sft

Output format (llama-factory sharegpt):
    [
        {
            "conversations": [
                {"from": "system", "value": "You are a science experiment agent..."},
                {"from": "human", "value": "Your ScienceWorld task is: ..."},
                {"from": "gpt", "value": "<think>...\n</think>\n<action>...</action>"},
                {"from": "human", "value": "Observation: ..."},
                {"from": "gpt", "value": "<think>...\n</think>\n<action>...</action>"},
                ...
            ]
        }
    ]
"""

import argparse
import json
import os
import zipfile


def extract_goldpaths(goldpaths_dir: str) -> dict:
    """Extract gold trajectories from zip file."""
    zip_path = os.path.join(goldpaths_dir, "goldpaths-all.zip")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Gold paths zip not found: {zip_path}")

    print(f"Extracting gold paths from: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        json_filename = zip_ref.namelist()[0]
        print(f"  Extracting: {json_filename}")

        with zip_ref.open(json_filename) as f:
            data = json.load(f)

    print(f"  Loaded {len(data)} tasks")
    return data


def convert_to_llama_factory(data: dict, system_prompt: str = None) -> list:
    """Convert gold trajectories to llama-factory format.

    IMPORTANT: The prompt format must match the RL training format exactly
    (from scienceworld_interaction.py) for consistency.
    """
    if system_prompt is None:
        system_prompt = (
            "You are a science experiment agent. You must always respond with exactly one "
            "<think>...</think> block followed by exactly one <action>...</action> block. "
            "The action must be chosen from the provided valid actions list."
        )

    conversations = []

    for task_idx, task_data in data.items():
        task_name = task_data.get('taskName', f'task-{task_idx}')

        for seq in task_data.get('goldActionSequences', []):
            variation = seq.get('variationIdx', 0)
            task_description = seq.get('taskDescription', '')
            fold = seq.get('fold', 'train')
            path = seq.get('path', [])

            if not path:
                continue

            # Build conversation
            conv = {
                "conversations": [
                    {"from": "system", "value": system_prompt}
                ],
                "task_name": task_name,
                "variation": variation,
                "fold": fold,
            }

            # Add initial user message (task description)
            # IMPORTANT: This format must match scienceworld_interaction.py exactly
            initial_observation = path[0]['observation'] if path else 'You are in a well-equipped science laboratory.'
            initial_prompt = f"""Your ScienceWorld task is: {task_description}
Prior to this step, you have already taken 0 step(s).
Below are the most recent 0 observations and the corresponding actions you took:
(no history)
You are now at step 1 and your current observation is:
{initial_observation}
Your valid actions of the current situation are: [look around, examine <object>, open <object>, close <object>, take <object> from <location>, put <object> in/on <location>, use <object> [on <object>], toggle <object>, pour <object> into <object>, mix <object>, go to <location>, look at <object>, wait, task].
Now it's your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose a valid action for the current step and present it within <action> </action> tags."""

            conv["conversations"].append({"from": "human", "value": initial_prompt})

            # Add assistant responses and subsequent user messages
            for i, step in enumerate(path):
                action = step.get('action', '')
                observation = step.get('observation', '')

                # Assistant response with thought and action
                assistant_response = f"<thought>I need to perform the action: {action}</thought>\n<action>{action}</action>"
                conv["conversations"].append({"from": "gpt", "value": assistant_response})

                # If this is not the last step, add user message with observation
                # IMPORTANT: This format must match scienceworld_interaction.py exactly
                if i < len(path) - 1:
                    # Build action history (last 3 steps)
                    history_length = min(3, i + 1)
                    history_lines = []
                    for j in range(max(0, i - history_length + 1), i + 1):
                        history_step = path[j]
                        history_action = history_step.get('action', '')
                        history_obs = history_step.get('observation', '')
                        history_lines.append(f"Step {j + 1}: Action: {history_action}")
                        history_lines.append(f"Observation: {history_obs[:300]}")

                    action_history = "\n".join(history_lines) if history_lines else "(no history)"

                    user_message = f"""Your ScienceWorld task is: {task_description}
Prior to this step, you have already taken {i + 1} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took:
{action_history}
You are now at step {i + 2} and your current observation is:
{observation}
Your valid actions of the current situation are: [look around, examine <object>, open <object>, close <object>, take <object> from <location>, put <object> in/on <location>, use <object> [on <object>], toggle <object>, pour <object> into <object>, mix <object>, go to <location>, look at <object>, wait, task].
Now it's your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose a valid action for the current step and present it within <action> </action> tags."""

                    conv["conversations"].append({"from": "human", "value": user_message})

            conversations.append(conv)

    return conversations


def split_by_fold(conversations: list) -> dict:
    """Split conversations by fold (train/dev/test)."""
    splits = {"train": [], "dev": [], "test": []}

    for conv in conversations:
        fold = conv.get("fold", "train")
        if fold in splits:
            splits[fold].append(conv)
        else:
            splits["train"].append(conv)

    return splits


def save_llama_factory_format(conversations: list, output_path: str):
    """Save conversations in llama-factory format."""
    clean_conversations = []
    for conv in conversations:
        clean_conv = {
            "conversations": conv["conversations"]
        }
        clean_conversations.append(clean_conv)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_conversations, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(clean_conversations)} conversations to: {output_path}")


def save_dataset_info(splits: dict, output_dir: str):
    """Save dataset_info.json for llama-factory."""
    dataset_info = {}

    for split_name, conversations in splits.items():
        if not conversations:
            continue

        dataset_name = f"scienceworld_{split_name}"
        dataset_info[dataset_name] = {
            "file_name": f"{split_name}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
                "system_tag": "system"
            }
        }

    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print(f"  Saved dataset_info.json to: {info_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare ScienceWorld gold trajectories for SFT")
    parser.add_argument("--goldpaths_dir", type=str,
                        default="benchmarks/ScienceWorld/goldpaths",
                        help="Directory containing goldpaths-all.zip")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/data/scienceworld_sft",
                        help="Output directory for SFT data")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="Custom system prompt")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract gold paths
    print("=" * 60)
    print("ScienceWorld Gold Trajectories → SFT Data")
    print("=" * 60)

    data = extract_goldpaths(args.goldpaths_dir)

    # Convert to llama-factory format
    print("\nConverting to llama-factory format...")
    conversations = convert_to_llama_factory(data, args.system_prompt)
    print(f"  Total conversations: {len(conversations)}")

    # Split by fold
    print("\nSplitting by fold...")
    splits = split_by_fold(conversations)
    for split_name, split_data in splits.items():
        print(f"  {split_name}: {len(split_data)} conversations")

    # Save each split
    print("\nSaving SFT data...")
    for split_name, split_data in splits.items():
        if split_data:
            output_path = os.path.join(args.output_dir, f"{split_name}.json")
            save_llama_factory_format(split_data, output_path)

    # Save dataset_info.json
    print("\nSaving dataset info...")
    save_dataset_info(splits, args.output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Files created:")
    for split_name in splits:
        if splits[split_name]:
            print(f"  - {split_name}.json ({len(splits[split_name])} conversations)")
    print(f"  - dataset_info.json")
    print("\nTo use with llama-factory:")
    print(f"  1. Copy {args.output_dir}/dataset_info.json to llama-factory/data/")
    print(f"  2. Copy {args.output_dir}/*.json to llama-factory/data/")
    print(f"  3. Add 'scienceworld_train' to your training config")


if __name__ == "__main__":
    main()
