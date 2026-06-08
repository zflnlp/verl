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
    pip install alfworld[full] transformers torch
    alfworld-download
"""

import argparse
import json
import os
import re
import time
from datetime import datetime


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


def run_episode(env, model, tokenizer, max_steps: int, game_file: str = None) -> dict:
    """Run a single episode and return results."""
    if game_file:
        obs, info = env.reset(game_file=game_file)
    else:
        obs, info = env.reset()

    observation = obs[0] if isinstance(obs, list) else obs
    admissible_actions = info.get("admissible_commands", [[]])[0] \
        if isinstance(info.get("admissible_commands"), list) else info.get("admissible_commands", [])

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

        # Step environment
        obs, reward, done, info = env.step([action])
        observation = obs[0] if isinstance(obs, list) else obs
        admissible_actions = info.get("admissible_commands", [[]])[0] \
            if isinstance(info.get("admissible_commands"), list) else info.get("admissible_commands", [])

        won = info.get("won", [False])[0] if isinstance(info.get("won"), list) else info.get("won", False)
        is_done = done[0] if isinstance(done, list) else done

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
                        help="ALFWorld task type to evaluate")
    parser.add_argument("--num_games", type=int, default=10, help="Number of games to evaluate")
    parser.add_argument("--max_steps", type=int, default=30, help="Max steps per episode")
    parser.add_argument("--alfworld_data_dir", type=str, default="/workspace/alfworld_data",
                        help="Path to ALFWorld data directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (default: auto)")

    args = parser.parse_args()

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

    # Initialize ALFWorld
    print(f"Initializing ALFWorld (task type: {args.task_type})")
    import alfworld
    import alfworld.agents.environment

    env_class = alfworld.agents.environment.AlfredTWEnv
    env = env_class(args.alfworld_data_dir, train_eval="eval_out_of_distribution")
    env = env.init_env(batch_size=1)

    # Discover game files
    import glob
    pattern = os.path.join(args.alfworld_data_dir, "json_2.1.1", args.task_type, "*.json")
    game_files = sorted(glob.glob(pattern))
    if not game_files:
        pattern = os.path.join(args.alfworld_data_dir, args.task_type, "*.json")
        game_files = sorted(glob.glob(pattern))

    num_games = min(args.num_games, len(game_files)) if game_files else args.num_games
    print(f"Found {len(game_files)} game files, evaluating {num_games}")

    # Run evaluation
    results = []
    total_wins = 0

    for i in range(num_games):
        game_file = game_files[i] if game_files and i < len(game_files) else None
        print(f"\n[{i+1}/{num_games}] Running game {i}...")

        start_time = time.time()
        result = run_episode(env, model, tokenizer, args.max_steps, game_file)
        elapsed = time.time() - start_time

        result["game_index"] = i
        result["game_file"] = game_file or ""
        result["elapsed_seconds"] = elapsed
        results.append(result)
        total_wins += 1 if result["won"] else 0

        status = "WON" if result["won"] else "LOST"
        print(f"  Status: {status}, Steps: {result['num_steps']}, Time: {elapsed:.1f}s")

    # Summary
    win_rate = total_wins / num_games if num_games > 0 else 0.0
    print(f"\n{'='*50}")
    print(f"Task type: {args.task_type}")
    print(f"Games played: {num_games}")
    print(f"Wins: {total_wins}")
    print(f"Win rate: {win_rate:.3f}")
    print(f"{'='*50}")

    # Save results
    summary = {
        "model_path": args.model_path,
        "task_type": args.task_type,
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

    # Cleanup
    del env


if __name__ == "__main__":
    main()
