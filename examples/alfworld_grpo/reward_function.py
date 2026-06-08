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
Reward function for ALFWorld GRPO training.

In real mode, the environment provides the primary reward via tool_rewards
(binary: 1.0 if task won, 0.0 otherwise).
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
    """Compute reward for ALFWorld task completion.

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
    action_keywords = ["go to", "take", "put", "open", "close", "toggle", "use",
                       "heat", "clean", "cool", "examine", "look", "inventory"]
    for keyword in action_keywords:
        if keyword in action_text:
            action_types.add(keyword)

    if len(action_types) >= 6:
        reward += 0.3
    elif len(action_types) >= 4:
        reward += 0.2
    elif len(action_types) >= 2:
        reward += 0.1

    # Reward for task-type-specific action sequences
    task_type = ground_truth.get("task_type", "").lower()
    goal = ground_truth.get("goal", "").lower()

    task_success_indicators = {
        "pick_and_place": ["take", "put"],
        "pick_clean_then_place": ["take", "clean", "put"],
        "pick_heat_then_place": ["take", "heat", "put"],
        "pick_cool_then_place": ["take", "cool", "put"],
        "look_at_obj_in_light": ["take", "use", "examine"],
        "pick_two_obj": ["take", "put"],
    }

    required = task_success_indicators.get(task_type, [])
    if required:
        matched = sum(1 for kw in required if kw in action_text)
        if matched >= len(required):
            reward += 0.4
        elif matched >= 1:
            reward += 0.2 * matched / len(required)

    # Reward for goal keyword matching in actions
    goal_words = set(re.findall(r'\b\w+\b', goal.lower()))
    # Filter out common stop words
    stop_words = {"a", "an", "the", "in", "on", "with", "and", "or", "to", "put", "find", "look", "at"}
    goal_words -= stop_words
    if goal_words:
        matched_words = sum(1 for w in goal_words if w in action_text)
        if matched_words >= len(goal_words):
            reward += 0.2
        elif matched_words >= 1:
            reward += 0.1 * matched_words / len(goal_words)

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
