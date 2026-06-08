#!/usr/bin/env python3
"""
Zero-shot evaluation of Qwen3-1.7B on WebShop.

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


def generate_action(model, tokenizer, prompt: str, max_new_tokens: int = 300) -> str:
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
        action = action_match.group(1).strip()
        # Clean up the action
        action = action.split('\n')[0].strip()
        return action

    # Try to find action patterns
    patterns = [
        (r'click\[buy\]', 'click[buy]'),
        (r'buy', 'click[buy]'),
        (r'search\[([^\]]+)\]', None),  # Will be handled specially
        (r'click\[([^\]]+)\]', None),   # Will be handled specially
    ]

    for pattern, replacement in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            if replacement:
                return replacement
            return match.group(0)

    # Default: return empty string (will be handled as invalid action)
    return ""


def create_prompt(task_description: str, step_count: int, history: List[Dict], current_observation: str) -> str:
    """Create a prompt for the model."""
    # Build action history
    history_lines = []
    for i, h in enumerate(history[-3:]):  # Last 3 steps
        history_lines.append(f"Step {step_count - len(history[-3:]) + i + 1}:")
        history_lines.append(f"  Action: {h['action']}")
        history_lines.append(f"  Result: {h['observation'][:150]}")

    history_text = "\n".join(history_lines) if history_lines else "(no history)"

    # Determine available actions based on current state
    if step_count == 0:
        available_actions = """Your available actions are:
- search[<query>]: Search for products. Use SHORT keywords (2-5 words), NOT the full instruction.
  Example: search[red dress size M]
  Example: search[wireless headphones bluetooth]"""
    else:
        available_actions = """Your available actions are:
- search[<query>]: Search for products. Use SHORT keywords (2-5 words).
- click[<button>]: Click a product link or button. Use the exact text shown.
- click[buy]: Purchase the currently viewed product (only when you see product details)."""

    prompt = f"""You are a shopping agent in an online store.

Task: {task_description}

{available_actions}

Previous actions:
{history_text}

Current page:
{current_observation[:800]}

What action should you take next? Choose ONE action from the available actions above.

Action:"""

    return prompt


def evaluate_episode(model, tokenizer, env, task_description: str, max_steps: int = 15) -> Tuple[float, List[Dict]]:
    """Evaluate a single episode."""
    obs = env.reset()
    
    # Extract task description from observation if needed
    if isinstance(obs, tuple):
        obs = obs[0] if isinstance(obs[0], str) else str(obs[0])
    
    history = []
    total_reward = 0.0

    for step in range(max_steps):
        # Create prompt
        prompt = create_prompt(task_description, step, history, obs)

        # Generate action
        response = generate_action(model, tokenizer, prompt)
        action = parse_action(response)

        # Skip empty actions
        if not action:
            action = "search[product]"

        # Execute action
        try:
            obs, reward, done, info = env.step(action)
            if isinstance(obs, tuple):
                obs = obs[0] if isinstance(obs[0], str) else str(obs[0])
        except Exception as e:
            print(f"Error executing action '{action}': {e}")
            obs = "Error: Invalid action"
            reward = 0.0
            done = False

        # Update history
        history.append({
            "step": step + 1,
            "action": action,
            "observation": obs[:300],
            "reward": reward,
        })

        total_reward = reward

        if done:
            break

    return total_reward, history


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on WebShop (zero-shot)")
    parser.add_argument("--model_path", type=str, default="/workspace/models/Qwen3-1.7B")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=15)
    parser.add_argument("--num_products", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="results/webshop_zero_shot")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

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
        if isinstance(obs, tuple):
            task_description = obs[1] if len(obs) > 1 and obs[1] else "Find and purchase a product"
            obs = obs[0] if isinstance(obs[0], str) else str(obs[0])
        else:
            task_description = "Find and purchase a product"

        # Extract task from observation if available
        if "Instruction:" in obs:
            match = re.search(r'Instruction:\s*(.*?)\[SEP\]', obs)
            if match:
                task_description = match.group(1).strip()

        # Evaluate episode
        reward, history = evaluate_episode(model, tokenizer, env, task_description, args.max_steps)
        rewards.append(reward)
        all_histories.append(history)

        if (episode + 1) % 10 == 0:
            avg_reward = sum(rewards) / len(rewards)
            nonzero = sum(1 for r in rewards if r > 0)
            print(f"Episode {episode + 1}/{args.num_episodes}: Avg Reward = {avg_reward:.3f}, Non-zero = {nonzero}")

    # Calculate statistics
    avg_reward = sum(rewards) / len(rewards)
    max_reward = max(rewards)
    min_reward = min(rewards)
    success_rate = sum(1 for r in rewards if r > 0.5) / len(rewards)
    nonzero_rate = sum(1 for r in rewards if r > 0) / len(rewards)

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Max reward: {max_reward:.3f}")
    print(f"Min reward: {min_reward:.3f}")
    print(f"Non-zero reward rate: {nonzero_rate:.1%}")
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
        "nonzero_rate": nonzero_rate,
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
