#!/usr/bin/env python3
"""Debug tokenizer output for SFT model evaluation."""

import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scienceworld import ScienceWorldEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/workspace/models/Qwen3-1.7B-SFT-v2")
    parser.add_argument("--task_name", type=str, default="boil")
    parser.add_argument("--variation", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    env = ScienceWorldEnv()
    env.load(args.task_name, args.variation, simplificationStr="easy")

    history = []
    for step in range(1, args.max_steps + 1):
        current_obs = env.look()
        possible_actions = env.get_possible_actions()

        history_lines = []
        for i, h in enumerate(history[-3:]):
            history_lines.append(f"Step {step - 3 + i + 1}: Action: {h['action']}")
            history_lines.append(f"Observation: {h['obs'][:300]}")
        action_history = "\n".join(history_lines) if history_lines else "(no history)"

        prompt = (
            f"Your ScienceWorld task is: {env.taskdescription()}\n"
            f"Prior to this step, you have already taken {step - 1} step(s).\n"
            f"Below are the most recent {min(3, len(history))} observations and the corresponding actions you took:\n"
            f"{action_history}\n"
            f"You are now at step {step} and your current observation is:\n"
            f"{current_obs}\n"
            f"Your valid actions of the current situation are: [{', '.join(possible_actions[:30])}].\n"
            f"Now it's your turn to take an action. You should first reason step-by-step about the current situation. "
            f"This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, "
            f"you should choose a valid action for the current step and present it within <action> </action> tags."
        )

        messages = [
            {"role": "system", "content": "You are a science experiment agent."},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if step == 1:
            print("=" * 60)
            print("TOKENIZER OUTPUT (first 800 chars):")
            print("=" * 60)
            print(text[:800])
            print("=" * 60)
            print()

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.4)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        action_match = re.search(r"<action>\s*(.*?)\s*</action>", response, re.IGNORECASE | re.DOTALL)
        action = action_match.group(1).strip() if action_match else "look around"

        obs, score, is_done, info = env.step(action)

        print(f"Step {step}: action={action}, score={info.get('score', score)}")
        print(f"  Response: {response[:200]}")
        print()

        history.append({"action": action, "obs": obs})
        if is_done:
            break


if __name__ == "__main__":
    main()
