#!/usr/bin/env python3
"""
Zero-shot evaluation of an LLM on ScienceWorld tasks.

Usage:
    python examples/scienceworld_grpo/eval_zero_shot.py \
        --model_path /workspace/models/Qwen3-1.7B \
        --task_name boil \
        --num_variations 10 \
        --max_steps 25 \
        --simplifications_preset easy

Requirements:
    conda activate scienceworld
    pip install vllm scienceworld
"""

import argparse
import json
import os
import re
import time
from datetime import datetime

from scienceworld import ScienceWorldEnv


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
                  observation: str, possible_actions: list) -> str:
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

    # Format available actions
    if possible_actions:
        available_actions = "\n".join(f"- {a}" for a in possible_actions[:30])
    else:
        available_actions = "- look around\n- task"

    return f"""You are an expert scientist working in a laboratory environment.
Your task is: {task_description}

Prior to this step, you have already taken {step_count - 1} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took:
{action_history}

You are now at step {step_count} and your current observation is:
{observation}

Your admissible actions of the current situation are:
{available_actions}

Now it's your turn to take one action for the current step. You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the science task. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""


def run_episode(env, model, tokenizer, task_name: str, variation: int,
                max_steps: int, simplifications: str = "", task_description: str = None) -> dict:
    """Run a single episode and return results."""
    env.load(task_name, variation, simplificationStr=simplifications)
    task_desc = task_description or env.taskDescription()

    history = []
    total_reward = 0.0

    for step in range(1, max_steps + 1):
        # Get current observation and possible actions
        current_obs = env.getObservation()
        possible_actions = env.getPossibleActions()

        # Format prompt
        prompt = format_prompt(task_desc, step, history, current_obs, possible_actions)

        # Generate response
        messages = [
            {"role": "system", "content": "You are an expert scientist working in a laboratory environment."},
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
        obs, score, is_done, info = env.step(action)
        total_reward = score / 100.0 if score > 1.0 else score

        history.append({
            "step": step,
            "action": action,
            "observation": obs,
            "score": score,
        })

        if is_done:
            break

    final_score = env.getScore()
    final_reward = final_score / 100.0 if final_score > 1.0 else final_score

    return {
        "task_name": task_name,
        "variation": variation,
        "num_steps": len(history),
        "final_score": final_score,
        "final_reward": final_reward,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot evaluation on ScienceWorld")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--task_name", type=str, default="boil", help="ScienceWorld task name")
    parser.add_argument("--num_variations", type=int, default=10, help="Number of variations to test")
    parser.add_argument("--max_steps", type=int, default=25, help="Max steps per episode")
    parser.add_argument("--simplifications_preset", type=str, default="easy",
                        help="Simplification preset (easy/none)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (default: auto)")
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size for vLLM")

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        args.output_dir = f"eval_results/scienceworld_{args.task_name}_{timestamp}"
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

    # Initialize ScienceWorld
    print(f"Initializing ScienceWorld (task: {args.task_name})")
    simplifications = args.simplifications_preset if args.simplifications_preset != "none" else ""
    env = ScienceWorldEnv()

    # Get available variations
    available_variations = env.getVariations(args.task_name)
    num_variations = min(args.num_variations, len(available_variations))
    print(f"Available variations: {len(available_variations)}, testing: {num_variations}")

    # Run evaluation
    results = []
    total_score = 0.0

    for i in range(num_variations):
        variation = available_variations[i] if i < len(available_variations) else i
        print(f"\n[{i+1}/{num_variations}] Running variation {variation}...")

        start_time = time.time()
        result = run_episode(env, model, tokenizer, args.task_name, variation, args.max_steps, simplifications=simplifications)
        elapsed = time.time() - start_time

        result["elapsed_seconds"] = elapsed
        results.append(result)
        total_score += result["final_reward"]

        print(f"  Score: {result['final_score']:.1f} (reward: {result['final_reward']:.3f}), "
              f"Steps: {result['num_steps']}, Time: {elapsed:.1f}s")

    # Summary
    avg_score = total_score / num_variations if num_variations > 0 else 0.0
    print(f"\n{'='*50}")
    print(f"Task: {args.task_name}")
    print(f"Variations tested: {num_variations}")
    print(f"Average reward: {avg_score:.3f}")
    print(f"Average score (0-100): {avg_score * 100:.1f}")
    print(f"{'='*50}")

    # Save results
    summary = {
        "model_path": args.model_path,
        "task_name": args.task_name,
        "num_variations": num_variations,
        "max_steps": args.max_steps,
        "simplifications_preset": args.simplifications_preset,
        "average_reward": avg_score,
        "average_score": avg_score * 100,
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
