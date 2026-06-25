#!/usr/bin/env python3
"""
Debug script to check SFT model output.

Usage:
    python examples/scienceworld_grpo/debug_sft_model.py \
        --model_path /workspace/models/Qwen3-1.7B-SFT
"""

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Debug SFT model output")
    parser.add_argument("--model_path", type=str, default="/workspace/models/Qwen3-1.7B-SFT")
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

    # Use exact same format as eval_all_tasks.py
    prompt = (
        "Your ScienceWorld task is: Task Description: Your task is to boil water. "
        "First, focus on the water. Then, take actions that will cause it to change its state of matter.\n"
        "Prior to this step, you have already taken 0 step(s).\n"
        "Below are the most recent 0 observations and the corresponding actions you took:\n"
        "(no history)\n"
        "You are now at step 1 and your current observation is:\n"
        "You are in a well-equipped science laboratory. There are workbenches with various equipment, "
        "chemical supplies, and scientific instruments. A sink is available for water.\n"
        "Your valid actions of the current situation are: [look around, examine <object>, open <object>, "
        "close <object>, take <object> from <location>, put <object> in/on <location>, "
        "use <object> [on <object>], toggle <object>, pour <object> into <object>, mix <object>, "
        "go to <location>, look at <object>, wait, task].\n"
        "Now it's your turn to take an action. You should first reason step-by-step about the current situation. "
        "This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, "
        "you should choose a valid action for the current step and present it within <action> </action> tags."
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

    print(f"\nInput length: {inputs['input_ids'].shape[1]} tokens")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("MODEL OUTPUT:")
    print("=" * 60)
    print(response)
    print("=" * 60)

    # Check for <action> tag
    action_match = re.search(r"<action>\s*(.*?)\s*</action>", response, re.IGNORECASE | re.DOTALL)
    if action_match:
        print(f"\n✅ EXTRACTED ACTION: {action_match.group(1).strip()}")
    else:
        print("\n❌ NO <action> TAG FOUND!")
        print("Checking for common patterns...")
        if "<thought>" in response.lower():
            print("  ✅ Found '<thought>' tag")
        else:
            print("  ❌ No '<thought>' tag")
        if "look around" in response.lower():
            print("  ✅ Found 'look around' in response")
        if "take" in response.lower():
            print("  ✅ Found 'take' in response")


if __name__ == "__main__":
    main()
