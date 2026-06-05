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

In real mode, the environment provides the primary reward via tool_rewards.
In mock mode or as fallback, we estimate reward from the agent's trajectory.
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
        extra_info: Additional information including tool_rewards.

    Returns:
        Reward score between 0.0 and 1.0.
    """
    if isinstance(ground_truth, str):
        import json
        try:
            ground_truth = json.loads(ground_truth)
        except Exception:
            ground_truth = {}

    # Primary: use environment reward from tool_rewards if available
    tool_rewards = []
    if extra_info and "tool_rewards" in extra_info:
        tool_rewards = extra_info["tool_rewards"]

    if tool_rewards:
        env_reward = tool_rewards[-1] if tool_rewards else 0.0
        return float(env_reward)

    # Fallback: estimate from trajectory
    return _parse_solution_reward(solution_str, ground_truth)


def _parse_solution_reward(solution_str: str, ground_truth: dict) -> float:
    """Estimate reward from the agent's solution trajectory.

    Args:
        solution_str: The agent's response text.
        ground_truth: Expected task attributes.

    Returns:
        Estimated reward score.
    """
    reward = 0.0

    # Extract actions from <action> tags
    actions = re.findall(r'<action>\s*(.*?)\s*</action>', solution_str, re.IGNORECASE | re.DOTALL)
    action_text = " ".join(actions).lower() if actions else solution_str.lower()

    # Reward for action diversity
    action_types = set()
    action_keywords = ["look", "examine", "open", "take", "put", "use", "toggle", "pour", "mix", "go to"]
    for keyword in action_keywords:
        if keyword in action_text:
            action_types.add(keyword)

    if len(action_types) >= 5:
        reward += 0.4
    elif len(action_types) >= 3:
        reward += 0.2
    elif len(action_types) >= 1:
        reward += 0.1

    # Reward for task-related actions
    task_name = ground_truth.get("task_name", "").lower()
    goal = ground_truth.get("goal", "").lower()

    task_success_indicators = {
        "boil": ["boil", "temperature", "stove", "heat", "pot"],
        "melt": ["melt", "heat", "temperature", "stove"],
        "freeze": ["freeze", "cold", "ice", "temperature"],
        "grow-plant": ["water", "grow", "soil", "seed", "plant"],
        "find-living": ["find", "living", "animal", "plant"],
        "chemistry-mix": ["mix", "react", "combine", "chemical"],
        "power-component": ["connect", "wire", "battery", "power"],
        "test-conductivity": ["test", "conduct", "material"],
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

    # Reward for reasoning (thought tags)
    thought_count = len(re.findall(r'<thought>', solution_str, re.IGNORECASE))
    if thought_count >= 3:
        reward += 0.2
    elif thought_count >= 1:
        reward += 0.1

    # Penalty for repeated actions
    if actions:
        consecutive_repeats = sum(
            1 for i in range(1, len(actions))
            if actions[i].strip().lower() == actions[i - 1].strip().lower()
        )
        if consecutive_repeats > 3:
            reward -= 0.1

    return max(0.0, min(1.0, reward))
