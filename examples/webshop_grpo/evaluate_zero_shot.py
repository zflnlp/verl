#!/usr/bin/env python3
"""
Zero-shot evaluation on WebShop.

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


def generate_action(model, tokenizer, prompt: str, max_new_tokens: int = 500) -> str:
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
    """Parse the action from the model's response.

    Expects format: <action>search[query]</action> or <action>click[button]</action>
    """
    # Try to extract action from <action> tags
    action_match = re.search(r'<action>(.*?)</action>', response, re.IGNORECASE | re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
        # Clean up: take only the first line
        action = action.split('\n')[0].strip()
        # Validate action format
        if re.match(r'^(search|click)\[', action, re.IGNORECASE):
            return action
        # If it's just "buy", convert to click[buy]
        if action.lower() == 'buy':
            return 'click[buy]'

    # Fallback: try to find action patterns directly
    patterns = [
        r'(search\[[^\]]+\])',
        r'(click\[[^\]]+\])',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1)

    # Last resort: check for "buy" keyword
    if 'buy' in response.lower():
        return 'click[buy]'

    return ""


def extract_instruction(observation: str) -> str:
    """Extract the task instruction from WebShop observation.

    WebShop observations typically start with:
        Instruction:  <instruction text>  [SEP]
    The instruction may span multiple lines.
    """
    # Try "Instruction:" followed by "[SEP]"
    match = re.search(r'Instruction:\s*(.*?)\[SEP\]', observation, re.DOTALL)
    if match:
        instruction = match.group(1).strip()
        # Collapse whitespace
        instruction = re.sub(r'\s+', ' ', instruction)
        if instruction:
            return instruction

    # Try extracting from the first line if it looks like an instruction
    first_line = observation.split('\n')[0].strip()
    if first_line and len(first_line) > 10 and 'Instruction' not in first_line:
        # Might be the instruction directly
        pass

    return ""


def extract_clickable_elements(observation: str) -> List[str]:
    """Extract actual clickable elements from WebShop observation.

    WebShop shows clickable elements in square brackets like:
        [B07XYZ123], [Next >], [Buy Now], [Back to Search], [Prev <]
    Also handles ASIN-style product IDs.
    """
    # Find all bracketed elements: [text]
    elements = re.findall(r'\[([^\]]+)\]', observation)
    # Filter out very long elements (likely not buttons) and empty ones
    elements = [e.strip() for e in elements if e.strip() and len(e.strip()) < 50]
    return elements


def _get_available_actions(observation: str) -> str:
    """Extract available actions from the current observation.

    Paper defines two action types:
    - search[<query>]: Search for products using a text query (only when search bar present)
    - click[<button name>]: Click on interactive elements (product links, filter buttons, pagination)

    We extract actual clickable elements from the observation to show the model
    what it can actually click on.
    """
    actions = []

    # search is available when search bar is present (usually on search results page)
    if 'search' in observation.lower() or 'Search' in observation:
        actions.append('search[<query>]')

    # Extract actual clickable elements from the observation
    elements = extract_clickable_elements(observation)

    # Categorize elements
    product_ids = []
    buttons = []
    for elem in elements:
        elem_lower = elem.lower()
        # Skip instruction-like text in brackets
        if len(elem) > 30:
            continue
        # Product IDs typically start with B followed by digits, or are ASINs
        if re.match(r'^B\d+$', elem) or re.match(r'^[A-Z0-9]{10}$', elem):
            product_ids.append(elem)
        # Navigation and action buttons
        elif elem_lower in ['next >', 'next', 'prev <', 'prev', 'back to search',
                            'buy now', 'buy', 'add to cart']:
            buttons.append(elem)
        # Filter options (ratings, price ranges, etc.)
        elif any(kw in elem_lower for kw in ['star', 'rating', 'price', '$']):
            buttons.append(elem)
        # Other short clickable elements
        elif len(elem) < 20:
            buttons.append(elem)

    # Add product click actions
    for pid in product_ids[:5]:  # Limit to first 5
        actions.append(f'click[{pid}]')

    # Add button click actions
    for btn in buttons[:10]:  # Limit to first 10
        actions.append(f'click[{btn}]')

    # If no specific actions detected, provide defaults
    if len(actions) <= 1:
        actions.append('click[<button>]')

    return "\n".join(f"  {a}" for a in actions)


def create_prompt(task_description: str, step_count: int, history: List[Dict], current_observation: str) -> str:
    """Create a prompt following the paper's template."""
    # Build action history
    history_length = min(3, len(history))
    history_lines = []
    for i, h in enumerate(history[-history_length:]):
        step_num = step_count - history_length + i + 1
        history_lines.append(f"Step {step_num}: Action: {h['action']}")
        obs_preview = h['observation'][:200] + "..." if len(h['observation']) > 200 else h['observation']
        history_lines.append(f"Observation: {obs_preview}")

    action_history = "\n".join(history_lines) if history_lines else "(no history)"

    # Determine available actions based on current page
    available_actions = _get_available_actions(current_observation)

    prompt = f"""You are an expert autonomous agent operating in the WebShop e-commerce environment. Your task is to: {task_description}.

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took:
{action_history}

You are now at step {step_count + 1} and your current observation is:
{current_observation}

Your admissible actions of the current situation are:
[
{available_actions}
]

Now it's your turn to take one action for the current step. You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

    return prompt


def evaluate_episode(model, tokenizer, env, initial_obs: str, task_description: str,
                     max_steps: int = 15, verbose: bool = False) -> Tuple[float, List[Dict]]:
    """Evaluate a single episode.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        env: The WebShop environment.
        initial_obs: The initial observation from env.reset() (already called by caller).
        task_description: Extracted task instruction.
        max_steps: Maximum steps per episode.
        verbose: If True, print model outputs for debugging.
    """
    obs_text = initial_obs
    history = []
    total_reward = 0.0

    for step in range(max_steps):
        # Create prompt following paper's template
        prompt = create_prompt(task_description, step, history, obs_text)

        # Generate response with thought and action
        response = generate_action(model, tokenizer, prompt)

        # Parse action from response
        action = parse_action(response)

        if verbose:
            print(f"\n--- Step {step + 1} ---")
            print(f"Model response (first 300 chars): {response[:300]}")
            print(f"Parsed action: '{action}'")

        # Skip empty actions with a default
        if not action:
            action = "search[product]"
            if verbose:
                print(f"  -> Empty action, fallback to: {action}")

        # Execute action
        try:
            obs, reward, done, info = env.step(action)
            if isinstance(obs, tuple):
                obs_text = obs[0] if isinstance(obs[0], str) else str(obs[0])
            else:
                obs_text = str(obs)

            if verbose:
                print(f"  -> Reward: {reward}, Done: {done}")
                print(f"  -> Obs (first 200 chars): {obs_text[:200]}")

        except Exception as e:
            print(f"Error executing action '{action}': {e}")
            obs_text = "Error: Invalid action. Please try a different action."
            reward = 0.0
            done = False

        # Update history
        history.append({
            "step": step + 1,
            "action": action,
            "observation": obs_text[:500],
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
    parser.add_argument("--verbose", action="store_true", help="Print model outputs for debugging")
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
        # Reset environment ONCE per episode
        obs = env.reset()
        if isinstance(obs, tuple):
            obs_text = obs[0] if isinstance(obs[0], str) else str(obs[0])
        else:
            obs_text = str(obs)

        # Extract task instruction from the observation
        task_description = extract_instruction(obs_text)

        if not task_description:
            task_description = "Find and purchase a product that matches the given requirements"
            if args.verbose:
                print(f"\n[WARNING] Could not extract instruction from observation:")
                print(f"  First 300 chars: {obs_text[:300]}")

        if args.verbose:
            print(f"\n=== Episode {episode + 1} ===")
            print(f"Task: {task_description[:100]}...")
            print(f"Initial obs (first 200 chars): {obs_text[:200]}")

        # Evaluate episode — pass the initial observation, don't reset again
        reward, history = evaluate_episode(
            model, tokenizer, env, obs_text, task_description,
            args.max_steps, verbose=args.verbose
        )
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
