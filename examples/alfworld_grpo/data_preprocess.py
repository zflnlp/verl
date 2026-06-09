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
Data preprocessing script for ALFWorld GRPO training.

Supports two modes:
1. Mock mode (--num_tasks 100): Generates synthetic data for pipeline testing.
2. Real mode (--use_real_env --task_type pick_and_place --num_games 10):
   Uses ALFWorld API to generate data from real game files.

Usage (mock):
    python examples/alfworld_grpo/data_preprocess.py \
        --local_save_dir /workspace/data/alfworld --num_tasks 100

Usage (real):
    python examples/alfworld_grpo/data_preprocess.py \
        --local_save_dir /workspace/data/alfworld_real \
        --task_type pick_and_place --num_games 10 --use_real_env
"""

import argparse
import json
import os
import random

import pandas as pd


# ALFWorld task types with example goals
# Keys are short names; internal_name is used for ALFWorld directory/ID lookup
TASK_TYPE_MAP = {
    "pick_and_place":        {"internal_name": "pick_and_place_simple",            "id": 1},
    "look_at_obj_in_light":  {"internal_name": "look_at_obj_in_light",             "id": 2},
    "pick_clean_then_place": {"internal_name": "pick_clean_then_place_in_recep",   "id": 3},
    "pick_heat_then_place":  {"internal_name": "pick_heat_then_place_in_recep",    "id": 4},
    "pick_cool_then_place":  {"internal_name": "pick_cool_then_place_in_recep",    "id": 5},
    "pick_two_obj":          {"internal_name": "pick_two_obj_and_place",           "id": 6},
}

ALFWORLD_TASK_TYPES = {
    "pick_and_place": {
        "goals": [
            "put a apple in countertop.",
            "put a mug in cabinet.",
            "put a plate in diningtable.",
            "put a cup in shelf.",
            "put a bowl in cabinet.",
            "put a fork in drawer.",
            "put a knife in drawer.",
            "put a spoon in drawer.",
            "put a glass in cabinet.",
            "put a bottle in fridge.",
        ],
        "description": "Pick up an object and place it in a receptacle.",
    },
    "pick_clean_then_place": {
        "goals": [
            "put a clean apple in countertop.",
            "put a clean mug in cabinet.",
            "put a clean plate in diningtable.",
            "put a clean cup in shelf.",
            "put a clean bowl in cabinet.",
            "put a clean fork in drawer.",
            "put a clean knife in drawer.",
            "put a clean spoon in drawer.",
            "put a clean glass in cabinet.",
            "put a clean lettuce in countertop.",
        ],
        "description": "Pick up an object, clean it, then place it in a receptacle.",
    },
    "pick_heat_then_place": {
        "goals": [
            "put a hot apple in countertop.",
            "put a hot mug in cabinet.",
            "put a hot plate in diningtable.",
            "put a hot cup in shelf.",
            "put a hot bowl in cabinet.",
            "put a hot potato in diningtable.",
            "put a hot egg in plate.",
            "put a hot bread in diningtable.",
        ],
        "description": "Pick up an object, heat it in the microwave, then place it.",
    },
    "pick_cool_then_place": {
        "goals": [
            "put a cool apple in countertop.",
            "put a cool mug in cabinet.",
            "put a cool plate in diningtable.",
            "put a cool cup in shelf.",
            "put a cool bowl in cabinet.",
            "put a cool bottle in fridge.",
            "put a cool bread in diningtable.",
        ],
        "description": "Pick up an object, cool it in the fridge, then place it.",
    },
    "look_at_obj_in_light": {
        "goals": [
            "look at alarmclock with desklamp.",
            "look at book with desklamp.",
            "look at cellphone with desklamp.",
            "look at creditcard with desklamp.",
            "look at keychain with desklamp.",
            "look at pen with desklamp.",
            "look at pencil with desklamp.",
            "look at remotecontrol with desklamp.",
        ],
        "description": "Examine an object under a light source.",
    },
    "pick_two_obj": {
        "goals": [
            "find two apple and put them in countertop.",
            "find two mug and put them in cabinet.",
            "find two plate and put them in diningtable.",
            "find two cup and put them in shelf.",
            "find two bowl and put them in cabinet.",
            "find two fork and put them in drawer.",
            "find two knife and put them in drawer.",
        ],
        "description": "Pick up two instances of the same object type.",
    },
}


def generate_mock_dataset(num_tasks: int, seed: int) -> list:
    """Generate a mock dataset of ALFWorld tasks."""
    random.seed(seed)
    tasks = []
    task_types = list(ALFWORLD_TASK_TYPES.keys())

    for i in range(num_tasks):
        task_type = random.choice(task_types)
        task_info = ALFWORLD_TASK_TYPES[task_type]
        goal = random.choice(task_info["goals"])
        tasks.append({
            "task_id": f"task_{seed + i}",
            "task_type": task_type,
            "goal": goal,
            "game_file": "",
        })
    return tasks


def generate_real_dataset(task_type: str, num_games: int, seed: int, game_files_dir: str) -> list:
    """Generate dataset from real ALFWorld game files.

    Uses the ALFWorld API to load game files and extract task descriptions.
    Falls back to predefined goals if API unavailable.
    """
    random.seed(seed)

    if task_type not in ALFWORLD_TASK_TYPES:
        raise ValueError(f"Unknown task type: {task_type}. Available: {list(ALFWORLD_TASK_TYPES.keys())}")

    task_info = ALFWORLD_TASK_TYPES[task_type]
    internal_name = TASK_TYPE_MAP[task_type]["internal_name"]

    # Discover game files from ALFWorld data directory
    # Structure: $ALFWORLD_DATA/json_2.1.1/{train|valid_seen|valid_unseen}/{internal_name}/.../game.tw-pddl
    game_files = []
    if game_files_dir:
        import glob
        for split in ["train", "valid_seen", "valid_unseen"]:
            pattern = os.path.join(game_files_dir, "json_2.1.1", split, internal_name, "**", "game.tw-pddl")
            found = sorted(glob.glob(pattern, recursive=True))
            game_files.extend(found)
        # Deduplicate
        game_files = sorted(set(game_files))

    if not game_files:
        print(f"Warning: No game files found for {task_type} ({internal_name}), using predefined goals")
        num_games = min(num_games, len(task_info["goals"]))

    # Try to get real task descriptions from ALFWorld API
    real_goals = {}
    if game_files:
        try:
            from alfworld.agents.environment import get_environment

            task_type_id = TASK_TYPE_MAP[task_type]["id"]
            alfworld_config = _build_alfworld_config_simple(game_files_dir, [task_type_id])

            for i, gf in enumerate(game_files[:num_games]):
                try:
                    env_type = alfworld_config["env"]["type"]
                    alfred_env = get_environment(env_type)(alfworld_config, train_eval="eval_out_of_distribution")
                    alfred_env.game_files = [gf]
                    alfred_env.num_games = 1
                    env = alfred_env.init_env(batch_size=1)
                    obs, info = env.reset()
                    obs_text = obs[0] if isinstance(obs, list) else obs
                    if "Your task is:" in obs_text:
                        real_goals[i] = obs_text.split("Your task is:")[-1].strip().split("\n")[0]
                    del env
                except Exception as e:
                    print(f"Warning: Could not load game file {gf}: {e}")
        except ImportError:
            print("Warning: alfworld package not installed, using default goals")
        except Exception as e:
            print(f"Warning: Could not initialize ALFWorld: {e}")

    tasks = []
    for i in range(num_games):
        if game_files and i < len(game_files):
            goal = real_goals.get(i, task_info["goals"][i % len(task_info["goals"])])
            game_file = game_files[i]
        else:
            goal = real_goals.get(i, task_info["goals"][i % len(task_info["goals"])])
            game_file = ""

        tasks.append({
            "task_id": f"{task_type}_game{i}",
            "task_type": task_type,
            "goal": goal,
            "game_file": game_file,
        })

    return tasks


def _build_alfworld_config_simple(game_files_dir: str, task_type_ids: list) -> dict:
    """Build a minimal ALFWorld config dict for game file loading."""
    data_root = game_files_dir.rstrip("/")
    return {
        "dataset": {
            "data_path": f"{data_root}/json_2.1.1/train",
            "eval_id_data_path": f"{data_root}/json_2.1.1/valid_seen",
            "eval_ood_data_path": f"{data_root}/json_2.1.1/valid_unseen",
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {
            "domain": f"{data_root}/logic/alfred.pddl",
            "grammar": f"{data_root}/logic/alfred.twl2",
        },
        "env": {
            "type": "AlfredTWEnv",
            "domain_randomization": False,
            "task_types": task_type_ids,
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
        },
        "general": {
            "random_seed": 42,
            "use_cuda": True,
            "task": "alfred",
            "training_method": "dagger",
        },
        "dagger": {
            "training": {
                "max_nb_steps_per_episode": 50,
            },
        },
    }


def format_for_verl(tasks: list) -> pd.DataFrame:
    """Format tasks for verl training."""
    data = []

    for task in tasks:
        task_type = task["task_type"]
        goal = task["goal"]

        # Build the initial prompt matching the interaction's _format_observation output
        prompt = f"""You are a household robot agent performing tasks in a simulated home environment.
Your task is: {goal}
Task type: {task_type}

Prior to this step, you have already taken 0 step(s).
Below are the most recent 0 observations and the corresponding actions you took:
(no history)

You are now at step 1 and your current observation is:
You are in the middle of a room. Looking quickly around you, you see a cabinet 6, a cabinet 5,
a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3,
a countertop 2, a countertop 1, a diningtable 1, a drawer 3, a drawer 2, a drawer 1,
a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1,
a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1,
and a toaster 1.

Your admissible actions of the current situation are:
- look
- inventory
- go to cabinet 1
- go to countertop 1
- go to fridge 1
- go to sinkbasin 1
- go to diningtable 1
- go to stoveburner 1
- go to microwave 1
- go to garbagecan 1
- take apple from countertop 1
- take cup from cabinet 1
- take knife from drawer 1
- take mug from shelf 1
- take plate from diningtable 1
- take sponge from sinkbasin 1
- examine cabinet 1
- examine countertop 1
- examine fridge 1

Now it's your turn to take one action for the current step. You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the household task. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

        messages = [
            {"role": "system", "content": "You are a household robot agent performing tasks in a simulated home environment."},
            {"role": "user", "content": prompt},
        ]

        data.append({
            "data_source": "alfworld",
            "prompt": messages,
            "ability": "household",
            "reward_model": {
                "ground_truth": task,
            },
            "extra_info": {
                "task_type": task_type,
                "game_file": task.get("game_file", ""),
            },
        })

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description="Generate ALFWorld training data")
    parser.add_argument("--local_save_dir", type=str, default="/workspace/data/alfworld",
                        help="Directory to save the generated data")
    parser.add_argument("--task_type", type=str, default="pick_and_place",
                        help="ALFWorld task type (e.g. pick_and_place, pick_clean_then_place)")
    parser.add_argument("--num_games", type=int, default=10,
                        help="Number of game files to use (real mode)")
    parser.add_argument("--num_tasks", type=int, default=100,
                        help="Number of tasks to generate (mock mode)")
    parser.add_argument("--use_real_env", action="store_true",
                        help="Use real ALFWorld API to get task descriptions")
    parser.add_argument("--game_files_dir", type=str, default="/workspace/alfworld_data",
                        help="Path to ALFWorld game files directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Ratio of training data")

    args = parser.parse_args()

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Generate tasks
    if args.use_real_env:
        print(f"Generating data from real ALFWorld task type: {args.task_type}")
        print(f"Using {args.num_games} game files")
        tasks = generate_real_dataset(args.task_type, args.num_games, args.seed, args.game_files_dir)
    else:
        print(f"Generating {args.num_tasks} mock ALFWorld tasks...")
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
            "task_type": args.task_type if args.use_real_env else "mixed",
            "num_games": args.num_games if args.use_real_env else None,
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
