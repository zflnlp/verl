#!/usr/bin/env python3
"""
Debug evaluation script to check step-by-step execution.

Usage:
    python examples/scienceworld_grpo/debug_eval.py \
        --model_path /workspace/models/Qwen3-1.7B-SFT \
        --task_name boil \
        --variation 0 \
        --max_steps 5
"""

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scienceworld import ScienceWorldEnv


def extract_action(text: str) -> str:
    """Extract action from <action> tags."""
    match = re.search(r"<action>\s*(.*?)\s*</action>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return "look around"


def main():
    parser = argparse.ArgumentParser(description="Debug evaluation")
    parser.add_argument("--model_path", type=str, default="/workspace/models/Qwen3-1.7B-SFT")
    parser.add_argument("--task_name", type=str, default="boil")
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading task: {args.task_name}, variation: {args.variation}")
    env = ScienceWorldEnv()
    env.load(args.task_name, args.variation, simplificationStr="easy")

    history = []

    for step in range(1, args.max_steps + 1):
        current_obs = env.look()
        possible_actions = env.get_possible_actions()

        # Build history
        history_length = min(3, len(history))
        history_lines = []
        for i, h in enumerate(history[-history_length:]):
            history_lines.append(f"Step {step - history_length + i + 1}: Action: {h['action']}")
            history_lines.append(f"Observation: {h['obs'][:300]}")
        action_history = "\n".join(history_lines) if history_lines else "(no history)"

        # Format prompt (same as eval_all_tasks.py)
        prompt = (
            f"Your ScienceWorld task is: {env.taskdescription()}\n"
            f"Prior to this step, you have already taken {step - 1} step(s).\n"
            f"Below are the most recent {history_length} observations and the corresponding actions you took:\n"
            f"{action_history}\n"
            f"You are now at step {step} and your current observation is:\n"
            f"{current_obs}\n"
            f"Your valid actions of the current situation are: [{', '.join(possible_actions[:30])}].\n"
            f"Now it's your turn to take an action. You should first reason step-by-step about the current situation. "
            f"This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, "
            f"you should choose a valid action for the current step and present it within <action> </action> tags."
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that completes science tasks. "
                "You MUST always respond with exactly one <thought>...</thought> block "
                "followed by exactly one <action>...</action> block.",
            },
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.4, top_p=1.0)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        action = extract_action(response)

        obs, score, is_done, info = env.step(action)

        print(f"Step {step}:")
        print(f"  Action: {action}")
        print(f"  Score: {info.get('score', score)}")
        print(f"  is_done: {is_done}")
        print()

        history.append({"action": action, "obs": obs})

        if is_done:
            print("Task completed!")
            break

    final_score = info.get("score", score) if history else 0.0
    print(f"Final score: {final_score}")
    print(f"Success: {final_score >= 100.0}")


if __name__ == "__main__":
    main()
