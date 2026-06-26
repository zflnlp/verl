#!/usr/bin/env python3
"""
Evaluate model on all 30 ScienceWorld tasks (test split).

This script evaluates the model on all ScienceWorld tasks using the official
test split, producing results suitable for paper reporting.

Metrics reported:
- Success Rate (SR): Percentage of tasks completed successfully
- Average Score: Average score across all tasks (0-100)
- Average Action Rounds: Average number of actions per task

Usage:
    python examples/scienceworld_grpo/eval_all_tasks.py \
        --model_path /workspace/models/Qwen3-1.7B-SFT \
        --max_steps 30 \
        --simplifications_preset easy

Requirements:
    conda activate scienceworld
    pip install transformers scienceworld
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scienceworld import ScienceWorldEnv


# All 30 ScienceWorld task names
ALL_TASK_NAMES = [
    "boil", "melt", "freeze", "change-the-state-of-matter-of",
    "use-thermometer", "measure-melting-point-known-substance",
    "measure-melting-point-unknown-substance", "power-component",
    "power-component-renewable-vs-nonrenewable-energy", "test-conductivity",
    "test-conductivity-of-unknown-substances", "find-living-thing",
    "find-non-living-thing", "find-plant", "find-animal", "grow-plant",
    "grow-fruit", "chemistry-mix", "chemistry-mix-paint-secondary-color",
    "chemistry-mix-paint-tertiary-color", "lifespan-longest-lived",
    "lifespan-shortest-lived", "lifespan-longest-lived-then-shortest-lived",
    "identify-life-stages-1", "identify-life-stages-2",
    "inclined-plane-determine-angle", "inclined-plane-friction-named-surfaces",
    "inclined-plane-friction-unnamed-surfaces", "mendelian-genetics-known-plant",
    "mendelian-genetics-unknown-plant",
]


def extract_action(text: str) -> str:
    """Extract action from <action> tags with robust fallback."""
    match = re.search(r'<action>\s*(.*?)\s*</action>', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback patterns
    action_patterns = [
        r'(?:go to|go)\s+\w+',
        r'(?:take|get|pick up)\s+.+?(?:\s+from\s+.+)?',
        r'(?:open|close)\s+\w+',
        r'(?:use|toggle|activate|turn on|turn off)\s+\w+',
        r'(?:pour|put|place)\s+.+?(?:\s+(?:in|into|on)\s+.+)?',
        r'(?:examine|look at|look)\s*\w*',
        r'(?:mix|stir)\s+\w+',
        r'(?:wait|task|inventory|look around)',
    ]

    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip().lower()
        for pattern in action_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return line

    return "look around"


def format_prompt(task_description, step_count, history, observation, possible_actions):
    """Format the prompt for the LLM.

    IMPORTANT: This format must match SFT/GRPO training format exactly
    (from scienceworld_interaction.py and prepare_sft_data.py).
    """
    history_length = min(3, len(history))
    history_lines = []
    for i, step in enumerate(history[-history_length:]):
        step_action = step.get("action", "")
        step_obs = step.get("observation", "")
        history_lines.append(f"Step {step_count - history_length + i + 1}: Action: {step_action}")
        history_lines.append(f"Observation: {step_obs[:300]}")

    action_history = "\n".join(history_lines) if history_lines else "(no history)"

    if possible_actions:
        available_actions = ", ".join(possible_actions[:30])
    else:
        available_actions = "look around, examine <object>, task"

    # IMPORTANT: This format must match scienceworld_interaction.py exactly
    return f"""Your ScienceWorld task is: {task_description}
Prior to this step, you have already taken {step_count - 1} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took:
{action_history}
You are now at step {step_count} and your current observation is:
{observation}
Your valid actions of the current situation are: [{available_actions}].
Now it's your turn to take an action. You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose a valid action for the current step and present it within <action> </action> tags."""


def run_episode(env, model, tokenizer, task_name, variation, max_steps, simplifications=""):
    """Run a single episode and return results."""
    env.load(task_name, variation, simplificationStr=simplifications)
    task_desc = env.taskdescription()

    history = []
    last_score = 0.0

    for step in range(1, max_steps + 1):
        current_obs = env.look()
        possible_actions = env.get_possible_actions()

        prompt = format_prompt(task_desc, step, history, current_obs, possible_actions)

        messages = [
            {"role": "system", "content": "You are a helpful assistant that completes science tasks. You MUST always respond with exactly one <thought>...</thought> block followed by exactly one <action>...</action> block."},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.4,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        action = extract_action(response)

        obs, score, is_done, info = env.step(action)
        last_score = info.get("score", score)

        history.append({
            "step": step,
            "action": action,
            "observation": obs,
            "score": last_score,
        })

        if is_done:
            break

    return {
        "task_name": task_name,
        "variation": variation,
        "num_steps": len(history),
        "final_score": last_score,
        "success": last_score >= 100.0,
    }


def evaluate_task(env, model, tokenizer, task_name, max_steps, simplifications=""):
    """Evaluate all test variations of a task."""
    env.load(task_name, 0)
    test_variations = env.get_variations_test()

    results = []
    for var_idx in test_variations:
        try:
            result = run_episode(env, model, tokenizer, task_name, var_idx, max_steps, simplifications)
            results.append(result)
        except Exception as e:
            print(f"  Error on {task_name} var {var_idx}: {e}")
            results.append({
                "task_name": task_name,
                "variation": var_idx,
                "num_steps": 0,
                "final_score": 0.0,
                "success": False,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate on all ScienceWorld tasks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--simplifications_preset", type=str, default="easy", help="Simplification preset")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--tasks", nargs="+", default=None, help="Specific tasks to evaluate (default: all)")

    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        args.output_dir = f"eval_results/scienceworld_all_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    task_names = args.tasks if args.tasks else ALL_TASK_NAMES

    print(f"Loading model: {args.model_path}")
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

    print(f"Initializing ScienceWorld")
    simplifications = args.simplifications_preset if args.simplifications_preset != "none" else ""
    env = ScienceWorldEnv()

    all_results = {}
    task_metrics = {}

    print(f"\nEvaluating {len(task_names)} tasks...")
    print("=" * 60)

    for task_name in task_names:
        print(f"\n[{task_name}]")
        start_time = time.time()

        results = evaluate_task(env, model, tokenizer, task_name, args.max_steps, simplifications)
        all_results[task_name] = results

        # Calculate metrics
        scores = [r["final_score"] for r in results]
        successes = [r["success"] for r in results]
        steps = [r["num_steps"] for r in results]

        avg_score = sum(scores) / len(scores) if scores else 0.0
        success_rate = sum(successes) / len(successes) * 100 if successes else 0.0
        avg_steps = sum(steps) / len(steps) if steps else 0.0

        task_metrics[task_name] = {
            "num_variations": len(results),
            "avg_score": avg_score,
            "success_rate": success_rate,
            "avg_steps": avg_steps,
        }

        elapsed = time.time() - start_time
        print(f"  Variations: {len(results)}")
        print(f"  Avg Score: {avg_score:.1f}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Avg Steps: {avg_steps:.1f}")
        print(f"  Time: {elapsed:.1f}s")

    # Calculate overall metrics
    all_scores = []
    all_successes = []
    all_steps = []
    for task_name, results in all_results.items():
        for r in results:
            all_scores.append(r["final_score"])
            all_successes.append(r["success"])
            all_steps.append(r["num_steps"])

    overall = {
        "total_variations": len(all_scores),
        "overall_avg_score": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "overall_success_rate": sum(all_successes) / len(all_successes) * 100 if all_successes else 0.0,
        "overall_avg_steps": sum(all_steps) / len(all_steps) if all_steps else 0.0,
    }

    # Print summary
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Tasks evaluated: {len(task_names)}")
    print(f"Total variations: {overall['total_variations']}")
    print(f"Overall Average Score: {overall['overall_avg_score']:.1f}")
    print(f"Overall Success Rate: {overall['overall_success_rate']:.1f}%")
    print(f"Overall Average Steps: {overall['overall_avg_steps']:.1f}")
    print("=" * 60)

    # Save results
    summary = {
        "model_path": args.model_path,
        "max_steps": args.max_steps,
        "simplifications_preset": args.simplifications_preset,
        "tasks_evaluated": len(task_names),
        "overall": overall,
        "task_metrics": task_metrics,
        "detailed_results": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save task metrics as CSV
    csv_path = os.path.join(args.output_dir, "task_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("task_name,num_variations,avg_score,success_rate,avg_steps\n")
        for task_name, metrics in task_metrics.items():
            f.write(f"{task_name},{metrics['num_variations']},{metrics['avg_score']:.1f},{metrics['success_rate']:.1f},{metrics['avg_steps']:.1f}\n")

    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - results.json (detailed)")
    print(f"  - task_metrics.csv (summary)")

    # Cleanup
    del env


if __name__ == "__main__":
    main()
