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
Reward function for ScienceWorld GRPO training.
"""

import re
from typing import Any, Optional


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs
) -> float:
    """Compute reward for ScienceWorld task completion.

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

    # Fallback: Parse the solution to estimate reward
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

    # Extract actions from <action> tags
    actions = re.findall(r'<action>\s*(.*?)\s*</action>', solution_str, re.IGNORECASE | re.DOTALL)
    action_text = " ".join(actions).lower() if actions else solution_lower

    # Reward for action diversity (different types of actions taken)
    action_types = set()
    action_keywords = ["look", "examine", "open", "take", "put", "use", "toggle", "pour", "mix", "go to"]
    for keyword in action_keywords:
        if keyword in action_text:
            action_types.add(keyword)

    # Base reward for taking diverse actions
    if len(action_types) >= 5:
        reward += 0.4
    elif len(action_types) >= 3:
        reward += 0.2
    elif len(action_types) >= 1:
        reward += 0.1

    # Reward for completing task-related actions
    task_name = ground_truth.get("task_name", "").lower()
    goal = ground_truth.get("goal", "").lower()

    # Check for task-specific successful actions
    task_success_indicators = {
        "boiling-water": ["boil", "temperature", "100"],
        "growing-plants": ["water", "grow", "soil", "seed"],
        "chemistry-mix": ["mix", "react", "combine", "chemical"],
        "circuit-building": ["connect", "wire", "bulb", "light"],
        "rock-identification": ["identify", "examine", "classify"],
    }

    for task_key, indicators in task_success_indicators.items():
        if task_key in task_name or task_key in goal:
            matched = sum(1 for ind in indicators if ind in action_text)
            if matched >= 2:
                reward += 0.3
                break
            elif matched >= 1:
                reward += 0.15
                break

    # Reward for scientific reasoning (checking for thought tags)
    thought_count = len(re.findall(r'<thought>', solution_str, re.IGNORECASE))
    if thought_count >= 3:
        reward += 0.2
    elif thought_count >= 1:
        reward += 0.1

    # Penalty for too many repeated actions
    if actions:
        # Count consecutive repeated actions
        consecutive_repeats = 0
        for i in range(1, len(actions)):
            if actions[i].strip().lower() == actions[i-1].strip().lower():
                consecutive_repeats += 1
        if consecutive_repeats > 3:
            reward -= 0.1

    return max(0.0, min(1.0, reward))
