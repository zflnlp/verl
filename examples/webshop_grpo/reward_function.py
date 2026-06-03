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
Reward functions for WebShop GRPO training.

This module provides reward computation for WebShop tasks, evaluating
whether the agent successfully completed the shopping task and how
well the purchased item matches the instruction.
"""

import re
from typing import Any, Optional


def compute_webshop_reward(
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    """Compute reward for WebShop task completion.

    This function evaluates the agent's performance on WebShop by:
    1. Checking if a purchase was made (from tool rewards)
    2. Evaluating the match score (from WebShop environment)
    3. Considering the number of steps taken (efficiency)

    Args:
        solution_str: The agent's final response or action sequence.
        ground_truth: Dictionary containing:
            - task_id: The WebShop task ID
            - goal: The target product description
            - category: Product category
            - attributes: Required product attributes
        extra_info: Additional information including tool rewards.

    Returns:
        Reward score between 0.0 and 1.0.
    """
    # Extract tool rewards from extra_info if available
    tool_rewards = []
    if extra_info and "tool_rewards" in extra_info:
        tool_rewards = extra_info["tool_rewards"]

    # If we have tool rewards (from WebShop environment), use them
    if tool_rewards:
        # The last tool reward typically contains the final environment reward
        env_reward = tool_rewards[-1] if tool_rewards else 0.0
        return float(env_reward)

    # Fallback: Parse the solution to check for successful purchase
    # This handles cases where tool rewards might not be available
    return _parse_solution_reward(solution_str, ground_truth)


def _parse_solution_reward(solution_str: str, ground_truth: dict[str, Any]) -> float:
    """Parse the solution string to estimate reward.

    This is a fallback method when tool rewards are not available.
    It checks for indicators of successful task completion.

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


def compute_webshop_reward_from_trajectory(
    trajectory: list[dict[str, Any]],
    ground_truth: dict[str, Any],
) -> float:
    """Compute reward from a complete trajectory.

    This function analyzes the entire agent trajectory to compute
    a more accurate reward score.

    Args:
        trajectory: List of (action, observation, reward) tuples.
        ground_truth: Expected task attributes.

    Returns:
        Final reward score between 0.0 and 1.0.
    """
    if not trajectory:
        return 0.0

    # Get the final step's reward
    final_step = trajectory[-1]
    final_reward = final_step.get("reward", 0.0)

    # If we have a valid environment reward, use it
    if final_reward > 0:
        return final_reward

    # Otherwise, estimate from the trajectory
    return _estimate_reward_from_trajectory(trajectory, ground_truth)


def _estimate_reward_from_trajectory(
    trajectory: list[dict[str, Any]],
    ground_truth: dict[str, Any],
) -> float:
    """Estimate reward from trajectory when environment reward is unavailable.

    Args:
        trajectory: Agent's action-observation history.
        ground_truth: Expected task attributes.

    Returns:
        Estimated reward score.
    """
    reward = 0.0

    # Check if a buy action was taken
    actions = [step.get("action", "") for step in trajectory]
    has_buy = any("buy" in action.lower() for action in actions)

    if has_buy:
        reward += 0.4

    # Check for search actions (shows intent to find the right product)
    search_actions = [a for a in actions if "search" in a.lower()]
    if search_actions:
        reward += 0.1

    # Check for product inspection (clicking on products)
    click_actions = [a for a in actions if "click" in a.lower()]
    if click_actions:
        reward += 0.1

    # Bonus for efficiency (fewer steps is better)
    num_steps = len(trajectory)
    if num_steps <= 5:
        reward += 0.2
    elif num_steps <= 10:
        reward += 0.1

    # Check goal matching in observations
    if ground_truth:
        goal = ground_truth.get("goal", "").lower()
        observations = " ".join(step.get("observation", "") for step in trajectory).lower()
        if goal and any(word in observations for word in goal.split()[:3]):
            reward += 0.2

    return min(1.0, reward)
