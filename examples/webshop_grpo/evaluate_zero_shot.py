#!/usr/bin/env python3
"""
Zero-shot evaluation of Qwen3-1.7B on WebShop.

This script evaluates a language model on WebShop tasks without any training.
The model is given a task description and must generate actions to complete it.

Usage:
    python examples/webshop_grpo/evaluate_zero_shot.py \
        --model_path /workspace/models/Qwen3-1.7B \
        --num_episodes 50 \
        --max_steps 15
"""

import argparse
import json
import os
import re
import sys
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_model(model_path: str, device: str = "auto"):
    """Load the model and tokenizer."""
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer


def generate_action(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    """Generate an action given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def parse_action(response: str) -> str:
    """Parse the action from the model's response."""
    # Try to extract action from <action> tags
    action_match = re.search(r'<action>(.*?)</action>', response, re.IGNORECASE | re.DOTALL)
    if action_match:
        return action_match.group(1).strip()

    # Try to find action patterns
    patterns = [
        r'search\[([^\]]+)\]',
        r'click\[([^\]]+)\]',
        r'buy',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(0)

    # Default: return the last line
    lines = response.strip().split('\n')
    return lines[-1].strip() if lines else response


def create_prompt(task_description: str, step_count: int, history: List[Dict], current_observation: str) -> str:
    """Create a prompt for the model."""
    # Build action history
    history_lines = []
    for i, h in enumerate(history[-3:]):  # Last 3 steps
        history_lines.append(f"Step {step_count - len(history[-3:]) + i + 1}:")
        history_lines.append(f"  Action: {h['action']}")
        history_lines.append(f"  Result: {h['observation'][:100]}...")

    history_text = "\n".join(history_lines) if history_lines else "(no history)"

    prompt = f"""You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}

Prior to this step, you have already taken {step_count} step(s).
Below are the most recent {min(3, len(history))} observations and the corresponding actions you took:
{history_text}

You are now at step {step_count + 1} and your current observation is:
{current_observation}

Your admissible actions of the current situation are:
- search[<query>]: Search for products using a text query
- click[<button name>]: Click on interactive elements (e.g., product links, filter buttons, pagination)
- click[buy]: Purchase the current item

Now it's your turn to take one action for the current step. You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

    return prompt


def evaluate_episode(model, tokenizer, env, task_description: str, max_steps: int = 15) -> Tuple[float, List[Dict]]:
    """Evaluate a single episode."""
    obs = env.reset()
    history = []
    total_reward = 0.0

    for step in range(max_steps):
        # Create prompt
        prompt = create_prompt(task_description, step, history, obs)

        # Generate action
        response = generate_action(model, tokenizer, prompt)
        action = parse_action(response)

        # Execute action
        try:
            if "search" in action.lower():
                # Extract search query
                match = re.search(r'search\[([^\]]+)\]', action, re.IGNORECASE)
                query = match.group(1) if match else action
                obs, reward, done, info = env.step(f"search[{query}]")
            elif "click" in action.lower():
                # Extract click target
                match = re.search(r'click\[([^\]]+)\]', action, re.IGNORECASE)
                target = match.group(1) if match else action
                obs, reward, done, info = env.step(f"click[{target}]")
            elif "buy" in action.lower():
                obs, reward, done, info = env.step("buy")
            else:
                # Try to execute as-is
                obs, reward, done, info = env.step(action)
        except Exception as e:
            print(f"Error executing action: {e}")
            reward = 0.0
            done = False

        # Update history
        history.append({
            "step": step + 1,
            "action": action,
            "observation": obs[:200],
            "reward": reward,
        })

        total_reward = reward

        if done:
            break

    return total_reward, history


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on WebShop (zero-shot)")
    parser.add_argument("--model_path", type=str, default="/workspace/models/Qwen3-1.7B",
                        help="Path to the model")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes to evaluate")
    parser.add_argument("--max_steps", type=int, default=15,
                        help="Maximum steps per episode")
    parser.add_argument("--num_products", type=int, default=1000,
                        help="Number of products in WebShop")
    parser.add_argument("--output_dir", type=str, default="results/webshop_zero_shot",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cpu)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model(args.model_path, args.device)

    # Import WebShop environment
    sys.path.insert(0, "/workspace/WebShop")
    import gym
    from web_agent_site.envs import WebAgentTextEnv

    # Create environment
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=args.num_products)

    # Evaluate
    rewards = []
    all_histories = []

    print(f"\nEvaluating {args.num_episodes} episodes...")
    for episode in tqdm(range(args.num_episodes)):
        # Get task description from environment
        obs = env.reset()
        task_description = env.get_instruction() if hasattr(env, 'get_instruction') else "Find and purchase a product"

        # Evaluate episode
        reward, history = evaluate_episode(model, tokenizer, env, task_description, args.max_steps)
        rewards.append(reward)
        all_histories.append(history)

        if (episode + 1) % 10 == 0:
            avg_reward = sum(rewards) / len(rewards)
            print(f"Episode {episode + 1}/{args.num_episodes}: Avg Reward = {avg_reward:.3f}")

    # Calculate statistics
    avg_reward = sum(rewards) / len(rewards)
    max_reward = max(rewards)
    min_reward = min(rewards)
    success_rate = sum(1 for r in rewards if r > 0.5) / len(rewards)

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Max reward: {max_reward:.3f}")
    print(f"Min reward: {min_reward:.3f}")
    print(f"Success rate (reward > 0.5): {success_rate:.1%}")
    print("=" * 50)

    # Save results
    results = {
        "model_path": args.model_path,
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "num_products": args.num_products,
        "avg_reward": avg_reward,
        "max_reward": max_reward,
        "min_reward": min_reward,
        "success_rate": success_rate,
        "rewards": rewards,
        "histories": all_histories,
    }

    results_path = os.path.join(args.output_dir, "zero_shot_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
