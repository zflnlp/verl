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

For multi-turn mode: returns 0.0 (rewards handled by ScienceWorldInteraction)
For single-turn mode: extracts actions and runs in ScienceWorld environment
"""

import re
from typing import Any, Optional


def _extract_actions(text: str) -> list:
    """Extract all actions from <action> tags in the response."""
    actions = re.findall(r'<action>\s*(.*?)\s*</action>', text, re.IGNORECASE | re.DOTALL)
    return [a.strip() for a in actions if a.strip()]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **kwargs
) -> float:
    """Compute reward for ScienceWorld task completion.

    In multi-turn mode (with ScienceWorldInteraction):
        - Returns 0.0 because rewards are handled by the interaction class
        - The interaction class already runs the environment and returns scores

    In single-turn mode (without interaction):
        - Extracts actions from <action> tags
        - Runs them in the ScienceWorld environment
        - Returns the environment score (0-1)

    Args:
        data_source: Data source identifier (e.g. "scienceworld")
        solution_str: The model's response containing <action> tags
        ground_truth: Dictionary containing task information
        extra_info: Additional information (unused in this version)

    Returns:
        Reward score between 0.0 and 1.0.
    """
    if isinstance(ground_truth, str):
        import json
        try:
            ground_truth = json.loads(ground_truth)
        except Exception:
            ground_truth = {}

    # Check if this is multi-turn mode (interaction_kwargs present)
    if extra_info and extra_info.get("interaction_kwargs"):
        # Multi-turn mode: rewards are handled by ScienceWorldInteraction
        # Return 0.0 here, the interaction class will provide the real rewards
        return 0.0

    # Single-turn mode: extract actions and run in environment
    task_name = ground_truth.get("task_name", "boil")
    variation = ground_truth.get("variation", 0)

    # Extract actions from model response
    actions = _extract_actions(solution_str)

    if not actions:
        # No valid actions found, try to extract from raw text
        # Fallback: use the last non-empty line as action
        lines = solution_str.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith("<") and not line.startswith("#"):
                actions = [line]
                break

    if not actions:
        return 0.0

    # Run in ScienceWorld environment
    try:
        from scienceworld import ScienceWorldEnv

        env = ScienceWorldEnv()
        env.load(task_name, variation)

        final_score = 0.0
        for action in actions:
            obs, score, is_done, info = env.step(action)
            final_score = info.get("score", score)
            if is_done:
                break

        # Normalize: ScienceWorld score is 0-100
        return final_score / 100.0 if final_score > 1.0 else final_score

    except Exception as e:
        print(f"[reward_fn] ScienceWorld error: {e}")
        return 0.0
