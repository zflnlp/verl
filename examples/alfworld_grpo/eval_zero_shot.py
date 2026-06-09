#!/usr/bin/env python3
"""
Zero-shot evaluation of an LLM on ALFWorld tasks.

Usage:
    python examples/alfworld_grpo/eval_zero_shot.py \
        --model_path /workspace/models/Qwen3-1.7B \
        --task_type pick_and_place \
        --num_games 10 \
        --max_steps 30

Requirements:
    pip install alfworld[full] transformers torch accelerate
    alfworld-download
"""

import argparse
import glob
import json
import os
import re
import time
from datetime import datetime


# Short name → (internal name, integer ID)
TASK_TYPE_MAP = {
    "pick_and_place":        ("pick_and_place_simple", 1),
    "look_at_obj_in_light":  ("look_at_obj_in_light", 2),
    "pick_clean_then_place": ("pick_clean_then_place_in_recep", 3),
    "pick_heat_then_place":  ("pick_heat_then_place_in_recep", 4),
    "pick_cool_then_place":  ("pick_cool_then_place_in_recep", 5),
    "pick_two_obj":          ("pick_two_obj_and_place", 6),
}


def extract_action(text: str) -> str:
    """Extract action from <action> tags or fallback to last line."""
    match = re.search(r'<action>\s*(.*?)\s*</action>', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try to find a known action pattern
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("<") and not line.startswith("#"):
            return line
    return text.strip()


def format_prompt(task_description: str, step_count: int, history: list,
                  observation: str, admissible_actions: list) -> str:
    """Format the prompt for the LLM."""
    # Build action history (last 3 steps)
    history_length = min(3, len(history))
    history_lines = []
    for i, step in enumerate(history[-history_length:]):
        step_action = step.get("action", "")
        step_obs = step.get("observation", "")
        history_lines.append(f"Step {step_count - history_length + i + 1}: Action: {step_action}")
        history_lines.append(f"Observation: {step_obs[:300]}")

    action_history = "\n".join(history_lines) if history_lines else "(no history)"

    # Format admissible actions
    if admissible_actions:
        available_actions = "\n".join(f"- {a}" for a in admissible_actions[:30])
    else:
        available_actions = "- look\n- inventory"

    return f"""You are a household robot agent performing tasks in a simulated home environment.
Your task is: {task_description}

Prior to this step, you have already taken {step_count - 1} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took:
{action_history}

You are now at step {step_count} and your current observation is:
{observation}

Your admissible actions of the current situation are:
{available_actions}

Now it's your turn to take one action for the current step. You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the household task. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""


def build_alfworld_config(alfworld_data_dir: str, task_type_ids: list) -> dict:
    """Build an ALFWorld config dict for AlfredTWEnv."""
    data_root = alfworld_data_dir.rstrip("/")
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


def make_single_game_env(config: dict, game_file: str, train_eval: str):
    """Create an ALFWorld env initialized with a single specific game file."""
    from alfworld.agents.environment import get_environment

    env_type = config["env"]["type"]
    alfred_env = get_environment(env_type)(config, train_eval=train_eval)
    # Override game_files to contain only the target game
    alfred_env.game_files = [game_file]
    alfred_env.num_games = 1
    env = alfred_env.init_env(batch_size=1)
    return env


def run_episode(env, model, tokenizer, max_steps: int) -> dict:
    """Run a single episode and return results."""
    obs, info = env.reset()

    observation = obs[0] if isinstance(obs, list) else obs
    admissible_actions = info["admissible_commands"][0] \
        if isinstance(info.get("admissible_commands"), list) else []

    # Extract goal from observation
    task_description = observation
    if "Your task is:" in observation:
        task_description = observation.split("Your task is:")[-1].strip().split("\n")[0]

    history = []
    won = False

    for step in range(1, max_steps + 1):
        # Format prompt
        prompt = format_prompt(task_description, step, history, observation, admissible_actions)

        # Generate response
        messages = [
            {"role": "system", "content": "You are a household robot agent performing tasks in a simulated home environment."},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with __import__("torch").no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        # Extract action
        action = extract_action(response)

        # Step environment: (obs, scores, dones, infos)
        obs, scores, dones, infos = env.step([action])
        observation = obs[0] if isinstance(obs, list) else obs
        admissible_actions = infos["admissible_commands"][0] \
            if isinstance(infos.get("admissible_commands"), list) else []

        won = infos["won"][0] if isinstance(infos.get("won"), list) else infos.get("won", False)
        is_done = dones[0] if isinstance(dones, list) else dones

        history.append({
            "step": step,
            "action": action,
            "observation": observation,
            "won": won,
        })

        if is_done:
            break

    return {
        "num_steps": len(history),
        "won": won,
        "reward": 1.0 if won else 0.0,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation on ALFWorld")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--task_type", type=str, default="pick_and_place",
                        help="ALFWorld task type (short name: pick_and_place, pick_clean_then_place, etc.)")
    parser.add_argument("--num_games", type=int, default=10, help="Number of games to evaluate")
    parser.add_argument("--max_steps", type=int, default=30, help="Max steps per episode")
    parser.add_argument("--alfworld_data_dir", type=str, default=os.environ.get("ALFWORLD_DATA", "/workspace/data/alf_data"),
                        help="Path to ALFWorld data directory (ALFWORLD_DATA)")
    parser.add_argument("--train_eval", type=str, default="eval_out_of_distribution",
                        choices=["train", "eval_in_distribution", "eval_out_of_distribution"],
                        help="Which data split to use")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (default: auto)")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")

    args = parser.parse_args()

    # Resolve task type
    if args.task_type not in TASK_TYPE_MAP:
        print(f"Unknown task type: {args.task_type}")
        print(f"Available: {list(TASK_TYPE_MAP.keys())}")
        return

    internal_name, task_type_id = TASK_TYPE_MAP[args.task_type]

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        args.output_dir = f"eval_results/alfworld_{args.task_type}_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build ALFWorld config
    print(f"Initializing ALFWorld (task type: {args.task_type} / {internal_name})")
    alfworld_config = build_alfworld_config(args.alfworld_data_dir, [task_type_id])

    # Discover game files for this task type
    # ALFWorld directory structure is FLAT: each game is a dir named like
    # pick_and_place_simple-SaltShaker-None-Drawer-10 directly under the split dir
    data_split = {
        "train": "train",
        "eval_in_distribution": "valid_seen",
        "eval_out_of_distribution": "valid_unseen",
    }[args.train_eval]

    split_dir = os.path.join(args.alfworld_data_dir, "json_2.1.1", data_split)
    all_game_dirs = sorted(glob.glob(os.path.join(split_dir, internal_name + "-*")))
    game_files = []
    for d in all_game_dirs:
        gf = os.path.join(d, "game.tw-pddl")
        if os.path.exists(gf):
            game_files.append(gf)

    if not game_files:
        print(f"No game files found at: {split_dir}/{internal_name}-*")
        print("Make sure ALFWORLD_DATA is set correctly and game files are downloaded.")
        return

    num_games = min(args.num_games, len(game_files))
    game_files = game_files[:num_games]
    print(f"Found {len(game_files)} game files, evaluating {num_games}")

    # Run evaluation
    results = []
    total_wins = 0

    for i, game_file in enumerate(game_files):
        # Extract a readable name from the path: .../pick_and_place_simple-X-None-Y-ID/game.tw-pddl
        game_name = os.path.basename(os.path.dirname(game_file))
        print(f"\n[{i+1}/{num_games}] Running: {game_name}")

        start_time = time.time()

        try:
            # Create a fresh env for each game file
            env = make_single_game_env(alfworld_config, game_file, args.train_eval)
            result = run_episode(env, model, tokenizer, args.max_steps)
            del env
        except Exception as e:
            print(f"  Error: {e}")
            result = {"num_steps": 0, "won": False, "reward": 0.0, "history": [], "error": str(e)}

        elapsed = time.time() - start_time

        result["game_index"] = i
        result["game_file"] = game_file
        result["elapsed_seconds"] = elapsed
        results.append(result)
        total_wins += 1 if result["won"] else 0

        status = "WON" if result["won"] else "LOST"
        print(f"  Status: {status}, Steps: {result['num_steps']}, Time: {elapsed:.1f}s")

    # Summary
    win_rate = total_wins / num_games if num_games > 0 else 0.0
    print(f"\n{'='*50}")
    print(f"Task type: {args.task_type} ({internal_name})")
    print(f"Games played: {num_games}")
    print(f"Wins: {total_wins}")
    print(f"Win rate: {win_rate:.3f}")
    print(f"{'='*50}")

    # Save results
    summary = {
        "model_path": args.model_path,
        "task_type": args.task_type,
        "task_type_internal": internal_name,
        "num_games": num_games,
        "max_steps": args.max_steps,
        "total_wins": total_wins,
        "win_rate": win_rate,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
