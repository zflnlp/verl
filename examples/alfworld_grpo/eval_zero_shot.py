#!/usr/bin/env python3
"""
Zero-shot evaluation of an LLM on ALFWorld tasks.

Usage (single task):
    python examples/alfworld_grpo/eval_zero_shot.py \
        --model_path /workspace/models/Qwen3-1.7B \
        --task_type pick_and_place --num_games 10 --max_steps 30

Usage (all 6 task types):
    python examples/alfworld_grpo/eval_zero_shot.py \
        --model_path /workspace/models/Qwen3-1.7B \
        --all_task_types --num_games 10 --max_steps 30
"""

import argparse
import glob
import json
import os
import re
import time
from datetime import datetime

TASK_TYPE_MAP = {
    "pick_and_place":        ("pick_and_place_simple", 1),
    "look_at_obj_in_light":  ("look_at_obj_in_light", 2),
    "pick_clean_then_place": ("pick_clean_then_place_in_recep", 3),
    "pick_heat_then_place":  ("pick_heat_then_place_in_recep", 4),
    "pick_cool_then_place":  ("pick_cool_then_place_in_recep", 5),
    "pick_two_obj":          ("pick_two_obj_and_place", 6),
}


def extract_action(text: str) -> str:
    match = re.search(r'<action>\s*(.*?)\s*</action>', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("<") and not line.startswith("#"):
            return line
    return text.strip()


def format_prompt(task_description, step_count, history, observation, admissible_actions):
    history_length = min(3, len(history))
    history_lines = []
    for i, step in enumerate(history[-history_length:]):
        history_lines.append(f"Step {step_count - history_length + i + 1}: Action: {step.get('action', '')}")
        history_lines.append(f"Observation: {step.get('observation', '')[:300]}")
    action_history = "\n".join(history_lines) if history_lines else "(no history)"
    avail = ", ".join(admissible_actions[:30]) if admissible_actions else "look, inventory"
    return f"""You are an expert agent operating in the ALFRED Embodied Environment. Your task is: {task_description}
Prior to this step, you have already taken {step_count - 1} step(s). Below are the most recent {history_length} observations and the corresponding actions you took:
{action_history}
You are now at step {step_count} and your current observation is: {observation}
Your admissible actions of the current situation are: [{avail}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <thought> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""


def build_alfworld_config(alfworld_data_dir, task_type_ids):
    data_root = alfworld_data_dir.rstrip("/")
    return {
        "dataset": {
            "data_path": f"{data_root}/json_2.1.1/train",
            "eval_id_data_path": f"{data_root}/json_2.1.1/valid_seen",
            "eval_ood_data_path": f"{data_root}/json_2.1.1/valid_unseen",
            "num_train_games": -1, "num_eval_games": -1,
        },
        "logic": {
            "domain": f"{data_root}/logic/alfred.pddl",
            "grammar": f"{data_root}/logic/alfred.twl2",
        },
        "env": {
            "type": "AlfredTWEnv", "domain_randomization": False,
            "task_types": task_type_ids, "expert_timeout_steps": 150,
            "expert_type": "handcoded", "goal_desc_human_anns_prob": 0.0,
        },
        "general": {"random_seed": 42, "use_cuda": True, "task": "alfred", "training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": 50}},
    }


def make_single_game_env(config, game_file, train_eval):
    from alfworld.agents.environment import get_environment
    alfred_env = get_environment(config["env"]["type"])(config, train_eval=train_eval)
    alfred_env.game_files = [game_file]
    alfred_env.num_games = 1
    return alfred_env.init_env(batch_size=1)


def run_episode(env, model, tokenizer, max_steps):
    obs, info = env.reset()
    observation = obs[0] if isinstance(obs, list) else obs
    admissible_actions = info["admissible_commands"][0] if isinstance(info.get("admissible_commands"), list) else []
    task_description = observation
    if "Your task is:" in observation:
        task_description = observation.split("Your task is:")[-1].strip().split("\n")[0]
    history = []
    won = False
    for step in range(1, max_steps + 1):
        prompt = format_prompt(task_description, step, history, observation, admissible_actions)
        messages = [
            {"role": "system", "content": "You are an expert agent operating in the ALFRED Embodied Environment."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with __import__("torch").no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=1.0, top_p=1.0,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        action = extract_action(response)
        obs, scores, dones, infos = env.step([action])
        observation = obs[0] if isinstance(obs, list) else obs
        admissible_actions = infos["admissible_commands"][0] if isinstance(infos.get("admissible_commands"), list) else []
        won = infos["won"][0] if isinstance(infos.get("won"), list) else infos.get("won", False)
        is_done = dones[0] if isinstance(dones, (list, tuple)) else dones
        history.append({"step": step, "action": action, "observation": observation, "won": won})
        if is_done:
            break
    return {"num_steps": len(history), "won": won, "reward": 1.0 if won else 0.0, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation on ALFWorld")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--task_type", type=str, default="pick_and_place")
    parser.add_argument("--all_task_types", action="store_true", help="Evaluate on all 6 task types")
    parser.add_argument("--num_games", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--alfworld_data_dir", type=str, default=os.environ.get("ALFWORLD_DATA", "/workspace/data/alf_data"))
    parser.add_argument("--train_eval", type=str, default="eval_out_of_distribution",
                        choices=["train", "eval_in_distribution", "eval_out_of_distribution"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tp_size", type=int, default=1)
    args = parser.parse_args()

    if args.all_task_types:
        task_types_to_eval = list(TASK_TYPE_MAP.keys())
    else:
        if args.task_type not in TASK_TYPE_MAP:
            print(f"Unknown task type: {args.task_type}. Available: {list(TASK_TYPE_MAP.keys())}")
            return
        task_types_to_eval = [args.task_type]

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        args.output_dir = f"eval_results/alfworld_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception:
        print("Warning: fast tokenizer failed, falling back to slow tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
                                                  device_map="auto", trust_remote_code=True)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    data_split = {"train": "train", "eval_in_distribution": "valid_seen",
                  "eval_out_of_distribution": "valid_unseen"}[args.train_eval]
    split_dir = os.path.join(args.alfworld_data_dir, "json_2.1.1", data_split)

    all_task_results = {}

    for task_type in task_types_to_eval:
        internal_name, task_type_id = TASK_TYPE_MAP[task_type]
        print(f"\n{'='*60}")
        print(f"Evaluating: {task_type} ({internal_name})")
        print(f"{'='*60}")

        alfworld_config = build_alfworld_config(args.alfworld_data_dir, [task_type_id])
        game_files = sorted(glob.glob(
            os.path.join(split_dir, internal_name + "-*", "**", "game.tw-pddl"), recursive=True))
        if not game_files:
            print(f"No game files found for {task_type}, skipping")
            continue

        num_games = min(args.num_games, len(game_files))
        game_files = game_files[:num_games]
        print(f"Found {len(game_files)} game files, evaluating {num_games}")

        results, total_wins = [], 0
        for i, game_file in enumerate(game_files):
            game_name = os.path.basename(os.path.dirname(game_file))
            print(f"  [{i+1}/{num_games}] {game_name}", end="", flush=True)
            start_time = time.time()
            try:
                env = make_single_game_env(alfworld_config, game_file, args.train_eval)
                result = run_episode(env, model, tokenizer, args.max_steps)
                del env
            except Exception as e:
                print(f" Error: {e}")
                result = {"num_steps": 0, "won": False, "reward": 0.0, "history": [], "error": str(e)}
            elapsed = time.time() - start_time
            result["game_index"], result["game_file"], result["elapsed_seconds"] = i, game_file, elapsed
            results.append(result)
            total_wins += 1 if result["won"] else 0
            print(f" -> {'WON' if result['won'] else 'LOST'} ({result['num_steps']} steps, {elapsed:.1f}s)")

        win_rate = total_wins / num_games if num_games > 0 else 0.0
        avg_all = sum(r['num_steps'] for r in results) / len(results) if results else 0
        won_r = [r for r in results if r.get('won')]
        lost_r = [r for r in results if not r.get('won')]
        avg_won = sum(r['num_steps'] for r in won_r) / len(won_r) if won_r else 0
        avg_lost = sum(r['num_steps'] for r in lost_r) / len(lost_r) if lost_r else 0
        print(f"  Win rate: {win_rate:.3f} ({total_wins}/{num_games})")
        if won_r: print(f"  Avg steps (won): {avg_won:.1f}")
        if lost_r: print(f"  Avg steps (lost): {avg_lost:.1f}")

        all_task_results[task_type] = {
            "task_type_internal": internal_name, "num_games": num_games,
            "total_wins": total_wins, "win_rate": win_rate,
            "avg_steps_all": avg_all, "avg_steps_won": avg_won, "avg_steps_lost": avg_lost,
            "results": results,
        }
        task_dir = os.path.join(args.output_dir, task_type)
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, "results.json"), "w") as f:
            json.dump(all_task_results[task_type], f, indent=2, default=str)

    if all_task_results:
        total_games = sum(tr['num_games'] for tr in all_task_results.values())
        total_wins = sum(tr['total_wins'] for tr in all_task_results.values())
        overall_wr = total_wins / total_games if total_games > 0 else 0.0
        print(f"\n{'='*60}")
        print(f"OVERALL SUMMARY | Model: {args.model_path} | Split: {data_split}")
        print(f"{'='*60}")
        for tt, tr in all_task_results.items():
            print(f"  {tt:30s}  Win rate: {tr['win_rate']:.3f}  ({tr['total_wins']}/{tr['num_games']})")
        print(f"  {'OVERALL':30s}  Win rate: {overall_wr:.3f}  ({total_wins}/{total_games})")
        print(f"{'='*60}")
        summary = {
            "model_path": args.model_path, "train_eval": args.train_eval, "data_split": data_split,
            "max_steps": args.max_steps, "overall_win_rate": overall_wr,
            "total_games": total_games, "total_wins": total_wins,
            "timestamp": datetime.now().isoformat(),
            "per_task": {tt: {k: v for k, v in tr.items() if k != 'results'} for tt, tr in all_task_results.items()},
        }
        with open(os.path.join(args.output_dir, "overall_summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
