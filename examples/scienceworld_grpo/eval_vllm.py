#!/usr/bin/env python3
"""
Evaluate model on ScienceWorld tasks using vLLM inference (matching training).

This script uses vLLM for inference, the same engine used during GRPO training,
to ensure evaluation results match training behavior.

Usage:
    python examples/scienceworld_grpo/eval_vllm.py \
        --model_path /tmp/grpo_step30_hf \
        --output_dir eval_results/grpo_step30_vllm
"""

import argparse
import json
import os
import re
from datetime import datetime

from vllm import LLM, SamplingParams
from scienceworld import ScienceWorldEnv


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
    match = re.search(r"<action>\s*(.*?)\s*</action>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return "look around"


def run_episode(llm, sampling_params, env, task_name, variation, max_steps=50):
    env.load(task_name, variation, simplificationStr="easy")
    task_desc = env.taskdescription()

    # Multi-turn conversation matching AgentLoop training format
    actions_history = []
    final_score = 0.0

    for step in range(1, max_steps + 1):
        obs = env.look()
        actions = env.get_possible_actions()

        # Build action history (last 3 steps)
        history_length = min(3, len(actions_history))
        history_lines = []
        for i, a in enumerate(actions_history[-history_length:]):
            history_lines.append(f"Step {step - history_length + i + 1}: Action: {a['action']}")
            history_lines.append(f"Observation: {a['obs'][:300]}")
        action_history = "\n".join(history_lines) if history_lines else "(no history)"

        # Format observation — matches AgentLoop exactly
        observation_text = (
            f"Your ScienceWorld task is: {task_desc}\n"
            f"Prior to this step, you have already taken {step - 1} step(s).\n"
            f"Below are the most recent {history_length} observations and the corresponding actions you took:\n"
            f"{action_history}\n"
            f"You are now at step {step} and your current observation is:\n"
            f"{obs}\n"
            f"Your valid actions of the current situation are: [{', '.join(actions[:30])}].\n"
            f"Now it's your turn to take an action. You should first reason step-by-step about the current situation. "
            f"This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, "
            f"you should choose a valid action for the current step and present it within <action> </action> tags."
        )

        # Multi-turn conversation: accumulate messages like AgentLoop
        if step == 1:
            messages = [{"role": "user", "content": observation_text}]
        else:
            # Add previous assistant response as assistant message
            messages.append({"role": "assistant", "content": last_response_text})
            # Add new observation as user message
            messages.append({"role": "user", "content": observation_text})

        response = llm.chat(messages, sampling_params)
        text = response[0].outputs[0].text
        last_response_text = text
        action = extract_action(text)

        obs2, reward, is_done, info = env.step(action)
        final_score = info.get("score", 0)  # info["score"] is total, reward is delta
        actions_history.append({"action": action, "obs": obs2})

        if is_done:
            break

    return final_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.3)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"eval_results/vllm_{datetime.now().strftime('%Y%m%d_%H%M')}"
    os.makedirs(args.output_dir, exist_ok=True)

    task_names = args.tasks if args.tasks else ALL_TASK_NAMES

    print(f"Loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=512,
        top_p=0.95,
    )

    print(f"Evaluating {len(task_names)} tasks...")
    env = ScienceWorldEnv()

    task_metrics = {}
    all_scores = []
    all_successes = []

    for task_name in task_names:
        env.load(task_name, 0)
        test_vars = env.get_variations_test()

        scores = []
        successes = []

        for var in test_vars:
            try:
                score = run_episode(llm, sampling_params, env, task_name, var, args.max_steps)
                print(f"  [{task_name} var {var}] score={score}")
                scores.append(score)
                successes.append(score >= 100)
            except Exception as e:
                print(f"  Error {task_name} var {var}: {e}")
                scores.append(0)
                successes.append(False)

        avg = sum(scores) / len(scores) if scores else 0
        sr = sum(successes) / len(successes) * 100
        task_metrics[task_name] = {"avg_score": avg, "success_rate": sr, "variations": len(scores)}
        all_scores.extend(scores)
        all_successes.extend(successes)

        print(f"[{task_name}] Avg: {avg:.1f}, SR: {sr:.1f}% ({len(scores)} vars)")

    overall_avg = sum(all_scores) / len(all_scores)
    overall_sr = sum(all_successes) / len(all_successes) * 100

    print(f"\n{'='*60}")
    print(f"OVERALL: Avg Score={overall_avg:.1f}, Success Rate={overall_sr:.1f}%")
    print(f"{'='*60}")

    with open(f"{args.output_dir}/results.json", "w") as f:
        json.dump({
            "model_path": args.model_path, "max_steps": args.max_steps,
            "overall_avg_score": overall_avg, "overall_success_rate": overall_sr,
            "task_metrics": task_metrics, "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"Saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
