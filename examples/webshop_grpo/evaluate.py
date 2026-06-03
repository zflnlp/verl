# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Evaluation script for WebShop GRPO trained models.

This script evaluates a trained model on WebShop tasks and reports
metrics like success rate, average reward, and efficiency.

Usage:
    # Evaluate a checkpoint
    python examples/webshop_grpo/evaluate.py \
        --model_path checkpoints/webshop_grpo/global_step_100/actor \
        --num_episodes 100 \
        --output_dir results/webshop_eval

    # Evaluate with WebShop server
    python examples/webshop_grpo/evaluate.py \
        --model_path Qwen/Qwen2.5-1.5B-Instruct \
        --webshop_server http://localhost:3000 \
        --num_episodes 50
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """Evaluation metrics for WebShop tasks."""
    num_episodes: int = 0
    successful_purchases: int = 0
    total_reward: float = 0.0
    total_steps: int = 0
    episode_rewards: list[float] = None
    episode_steps: list[int] = None
    success_rate: float = 0.0
    avg_reward: float = 0.0
    avg_steps: float = 0.0

    def __post_init__(self):
        if self.episode_rewards is None:
            self.episode_rewards = []
        if self.episode_steps is None:
            self.episode_steps = []

    def update(self, reward: float, steps: int, success: bool):
        """Update metrics with a new episode result."""
        self.num_episodes += 1
        self.total_reward += reward
        self.total_steps += steps
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        if success:
            self.successful_purchases += 1

    def compute_summary(self):
        """Compute summary statistics."""
        if self.num_episodes > 0:
            self.success_rate = self.successful_purchases / self.num_episodes
            self.avg_reward = self.total_reward / self.num_episodes
            self.avg_steps = self.total_steps / self.num_episodes


class WebShopEvaluator:
    """Evaluator for WebShop tasks."""

    def __init__(
        self,
        model_path: str,
        webshop_server: str = "http://localhost:3000",
        max_steps: int = 10,
        device: str = "auto",
    ):
        self.webshop_server = webshop_server
        self.max_steps = max_steps
        self.device = self._get_device(device)

        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

        # WebShop environment
        self.env = None

    def _get_device(self, device: str) -> str:
        """Determine the device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch, "npu") and torch.npu.is_available():
                return "npu"
            else:
                return "cpu"
        return device

    def _init_env(self):
        """Initialize WebShop environment."""
        if self.env is None:
            try:
                from webshop.web_agent_text_env import WebAgentTextEnv
                self.env = WebAgentTextEnv(
                    observation_mode="text",
                    server=self.webshop_server,
                )
            except ImportError:
                raise RuntimeError(
                    "WebShop not installed. Install with: pip install webshop"
                )

    def _generate_response(self, messages: list[dict[str, str]], max_new_tokens: int = 512) -> str:
        """Generate a response from the model.

        Args:
            messages: Chat messages in OpenAI format.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated response text.
        """
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return response

    def _parse_action(self, response: str) -> Optional[tuple[str, str]]:
        """Parse action from model response.

        Args:
            response: Model's response text.

        Returns:
            Tuple of (action_type, action_arg) or None if no action found.
        """
        response_lower = response.lower()

        # Check for search action
        if "search[" in response_lower:
            start = response_lower.index("search[") + 7
            end = response_lower.index("]", start)
            query = response[start:end].strip()
            return ("search", query)

        # Check for click action
        if "click[" in response_lower:
            start = response_lower.index("click[") + 6
            end = response_lower.index("]", start)
            element = response[start:end].strip()
            return ("click", element)

        # Check for buy action
        if "buy" in response_lower and ("purchase" in response_lower or "buy now" in response_lower):
            return ("buy", "")

        return None

    def run_episode(self, task: dict[str, Any]) -> tuple[float, int, bool]:
        """Run a single evaluation episode.

        Args:
            task: Task configuration dictionary.

        Returns:
            Tuple of (reward, steps, success).
        """
        self._init_env()

        # Reset environment
        obs, info = self.env.reset()
        goal = task.get("goal", "")

        # System prompt
        system_prompt = (
            "You are a shopping assistant. Help find and purchase products.\n"
            "Available actions:\n"
            "- search[query]: Search for products\n"
            "- click[element]: Click on a product or button\n"
            "- buy: Purchase the current product\n"
            "Always explain your reasoning before acting."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Find and buy: {goal}\n\nCurrent page:\n{obs}"},
        ]

        steps = 0
        done = False
        reward = 0.0

        while not done and steps < self.max_steps:
            # Generate response
            response = self._generate_response(messages)

            # Parse action
            action = self._parse_action(response)

            if action is None:
                # Model didn't provide a valid action
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": "Please take an action using the available tools: search[query], click[element], or buy."
                })
                steps += 1
                continue

            # Execute action
            action_type, action_arg = action
            if action_type == "search":
                env_action = f"search[{action_arg}]"
            elif action_type == "click":
                env_action = f"click[{action_arg}]"
            elif action_type == "buy":
                env_action = "buy"
            else:
                continue

            obs, reward, done, info = self.env.step(env_action)
            steps += 1

            # Update messages
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Action result:\n{obs}\n\nContinue shopping or confirm purchase."
            })

        success = reward > 0 and info.get("purchase_successful", False)

        return reward, steps, success

    def evaluate(
        self,
        num_episodes: int = 100,
        tasks: Optional[list[dict[str, Any]]] = None,
    ) -> EvalMetrics:
        """Run evaluation on multiple episodes.

        Args:
            num_episodes: Number of episodes to evaluate.
            tasks: Optional list of tasks. If None, generates random tasks.

        Returns:
            Evaluation metrics.
        """
        if tasks is None:
            # Generate random tasks
            from data_preprocess import generate_webshop_tasks
            tasks = generate_webshop_tasks(num_episodes)

        metrics = EvalMetrics()

        logger.info(f"Evaluating on {min(num_episodes, len(tasks))} episodes")

        for i, task in enumerate(tqdm(tasks[:num_episodes], desc="Evaluating")):
            try:
                reward, steps, success = self.run_episode(task)
                metrics.update(reward, steps, success)

                if (i + 1) % 10 == 0:
                    logger.info(
                        f"Episode {i+1}: reward={reward:.3f}, steps={steps}, "
                        f"success={success}, running_success_rate={metrics.successful_purchases/metrics.num_episodes:.3f}"
                    )
            except Exception as e:
                logger.error(f"Episode {i+1} failed: {e}")
                continue

        metrics.compute_summary()
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate WebShop GRPO model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model",
    )
    parser.add_argument(
        "--webshop_server",
        type=str,
        default="http://localhost:3000",
        help="WebShop server URL",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/webshop_eval",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, npu, cpu)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize evaluator
    evaluator = WebShopEvaluator(
        model_path=args.model_path,
        webshop_server=args.webshop_server,
        max_steps=args.max_steps,
        device=args.device,
    )

    # Run evaluation
    logger.info("Starting evaluation...")
    start_time = time.time()
    metrics = evaluator.evaluate(num_episodes=args.num_episodes)
    eval_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 50)
    print("WebShop Evaluation Results")
    print("=" * 50)
    print(f"Model: {args.model_path}")
    print(f"Episodes: {metrics.num_episodes}")
    print(f"Success Rate: {metrics.success_rate:.3f}")
    print(f"Average Reward: {metrics.avg_reward:.3f}")
    print(f"Average Steps: {metrics.avg_steps:.1f}")
    print(f"Total Time: {eval_time:.1f}s")
    print(f"Time per Episode: {eval_time/metrics.num_episodes:.2f}s")
    print("=" * 50)

    # Save results
    results = {
        "model_path": args.model_path,
        "webshop_server": args.webshop_server,
        "num_episodes": metrics.num_episodes,
        "max_steps": args.max_steps,
        "success_rate": metrics.success_rate,
        "avg_reward": metrics.avg_reward,
        "avg_steps": metrics.avg_steps,
        "total_time": eval_time,
        "episode_rewards": metrics.episode_rewards,
        "episode_steps": metrics.episode_steps,
    }

    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")

    # Also save a summary
    summary_path = os.path.join(args.output_dir, "eval_summary.txt")
    with open(summary_path, "w") as f:
        f.write("WebShop Evaluation Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Episodes: {metrics.num_episodes}\n")
        f.write(f"Success Rate: {metrics.success_rate:.3f}\n")
        f.write(f"Average Reward: {metrics.avg_reward:.3f}\n")
        f.write(f"Average Steps: {metrics.avg_steps:.1f}\n")
        f.write(f"Total Time: {eval_time:.1f}s\n")

    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
