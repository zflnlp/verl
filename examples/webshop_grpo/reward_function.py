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
Reward function for WebShop GRPO training.
"""

import re
from typing import Any, Optional


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs
) -> float:
    """Compute reward for WebShop task completion.

    Args:
        solution_str: The agent's final response or action sequence.
        ground_truth: Dictionary containing task information.
        extra_info: Additional information.

    Returns:
        Reward score between 0.0 and 1.0.
    """
    # Parse ground_truth if it's a string
    if isinstance(ground_truth, str):
        import json
        try:
            ground_truth = json.loads(ground_truth)
        except:
            ground_truth = {}

    # Extract tool rewards from extra_info if available
    tool_rewards = []
    if extra_info and "tool_rewards" in extra_info:
        tool_rewards = extra_info["tool_rewards"]

    # If we have tool rewards, use them
    if tool_rewards:
        env_reward = tool_rewards[-1] if tool_rewards else 0.0
        return float(env_reward)

    # Fallback: Parse the solution to check for successful purchase
    return _parse_solution_reward(solution_str, ground_truth)


def _parse_solution_reward(solution_str: str, ground_truth: dict) -> float:
    """Parse the solution string to estimate reward.

    Args:
        solution_str: The agent's response text.
        ground_truth: Expected task attributes.

    Returns:
        Estimated reward score.
    """
    reward = 0.0
    solution_lower = solution_str.lower()

    # Check if the agent indicates a purchase was made
    purchase_indicators = [
        "purchased",
        "bought",
        "order placed",
        "successfully bought",
        "item purchased",
        "bought the item",
        "completed the purchase",
        "purchase successful",
    ]

    for indicator in purchase_indicators:
        if indicator in solution_lower:
            reward = 0.5  # Base reward for purchase
            break

    # Check if the agent mentions matching the target
    if ground_truth:
        goal = ground_truth.get("goal", "").lower()
        if goal and any(word in solution_lower for word in goal.split()[:3]):
            reward += 0.3  # Bonus for matching goal

    # Check for attribute matching
    attributes = ground_truth.get("attributes", {})
    if attributes:
        matched_attrs = sum(
            1
            for attr, value in attributes.items()
            if str(value).lower() in solution_lower
        )
        if matched_attrs > 0:
            reward += 0.2 * (matched_attrs / len(attributes))

    return min(1.0, reward)
