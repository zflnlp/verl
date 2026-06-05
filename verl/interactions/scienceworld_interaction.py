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
Interaction class for ScienceWorld benchmark.

ScienceWorld is a text-based science simulation environment where agents
perform experiments and manipulate objects to complete science tasks.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _extract_action(text: str) -> str:
    """Extract action from <action> tags or use raw text."""
    match = re.search(r'<action>\s*(.*?)\s*</action>', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


class ScienceWorldInteraction(BaseInteraction):
    """Interaction class for ScienceWorld benchmark.

    This class manages multi-turn interactions with the ScienceWorld environment,
    where the agent needs to perform scientific experiments to complete tasks.

    Supports two modes:
    - Mock mode: Simulated responses for pipeline testing.
    - Real mode: Direct ScienceWorld Python API integration.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self.use_mock = config.get("use_mock", True)
        self.max_steps = config.get("max_steps", 25)
        # Real mode config
        self.simplifications_preset = config.get("simplifications_preset", "easy")
        self.env_step_limit = config.get("env_step_limit", 100)

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[dict] = None,
        **kwargs
    ) -> str:
        """Start a new ScienceWorld interaction.

        Args:
            instance_id: Optional unique identifier for this interaction.
            ground_truth: Dictionary containing task information:
                - task_name: ScienceWorld task name (e.g. "boil")
                - variation: Variation index (int)
                - goal: Task description

        Returns:
            The instance ID for this interaction.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        gt = ground_truth or {}
        instance = {
            "ground_truth": gt,
            "steps": [],
            "current_observation": "",
            "reward": 0.0,
            "is_done": False,
            "num_steps": 0,
            "task_name": gt.get("task_name", "unknown"),
            "env": None,
            "possible_actions": [],
        }
        self._instance_dict[instance_id] = instance

        if self.use_mock:
            instance["current_observation"] = self._get_mock_initial_observation(gt)
            instance["possible_actions"] = self._get_mock_actions()
        else:
            try:
                from scienceworld import ScienceWorldEnv

                env = ScienceWorldEnv()
                task_name = gt.get("task_name", "boil")
                variation = gt.get("variation", 0)

                env.load(task_name, variation)
                instance["env"] = env
                instance["current_observation"] = env.look()
                instance["possible_actions"] = env.get_possible_actions()
                instance["task_description"] = env.taskdescription()
            except Exception as e:
                logger.error(f"Failed to initialize ScienceWorld env: {e}")
                instance["current_observation"] = f"Error initializing environment: {e}"

        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Tuple[bool, str, float, dict]:
        """Process agent action and return environment response.

        Args:
            instance_id: The interaction instance ID.
            messages: List of conversation messages.

        Returns:
            Tuple of (should_terminate, response_content, turn_reward, additional_data).
        """
        if instance_id not in self._instance_dict:
            logger.error(f"Instance {instance_id} not found")
            return True, "Error: Instance not found", 0.0, {}

        instance = self._instance_dict[instance_id]
        instance["num_steps"] += 1

        # Extract the last assistant message (the agent's action)
        action = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                action = msg.get("content", "")
                break

        # Process the action
        if self.use_mock:
            raw_observation, reward, is_done = self._process_mock_action(action, instance)
        else:
            raw_observation, reward, is_done = self._process_real_action(action, instance)

        # Format observation for next turn
        observation = self._format_observation(instance, raw_observation)

        # Update instance state
        instance["steps"].append({"action": action, "observation": raw_observation})
        instance["current_observation"] = raw_observation
        instance["reward"] = reward

        # Check termination conditions
        should_terminate = is_done or instance["num_steps"] >= self.max_steps

        if should_terminate:
            instance["is_done"] = True
            final_reward = await self.calculate_score(instance_id)
            instance["reward"] = final_reward
            return True, observation, final_reward, {"num_steps": instance["num_steps"]}
        else:
            return False, observation, 0.0, {"num_steps": instance["num_steps"]}

    def _process_real_action(self, action: str, instance: dict) -> Tuple[str, float, bool]:
        """Process an action using the real ScienceWorld environment.

        Args:
            action: The agent's action string (may contain <action> tags).
            instance: The interaction instance state.

        Returns:
            Tuple of (observation, reward, is_done).
        """
        env = instance.get("env")
        if env is None:
            return "Error: Environment not initialized", 0.0, True

        clean_action = _extract_action(action)

        try:
            obs, score, is_done, info = env.step(clean_action)
            # ScienceWorld score is 0-100, normalize to 0-1
            raw_score = info.get("score", score)
            instance["last_score"] = raw_score
            reward = score / 100.0 if score > 1.0 else score
            # Update possible actions for next turn
            instance["possible_actions"] = env.get_possible_actions()
            return obs, reward, is_done
        except Exception as e:
            logger.error(f"ScienceWorld step error: {e}")
            return f"Error executing action: {e}", 0.0, False

    def _format_observation(self, instance: dict, raw_observation: str) -> str:
        """Format observation with context for the next turn.

        Args:
            instance: The interaction instance state.
            raw_observation: The raw observation from the environment.

        Returns:
            Formatted observation string.
        """
        gt = instance.get("ground_truth", {})
        goal = gt.get("goal", instance.get("task_description", "Complete the science task"))
        task_name = instance.get("task_name", "unknown")
        step_count = instance.get("num_steps", 0)
        steps = instance.get("steps", [])

        # Build action history (last 3 steps)
        history_length = min(3, len(steps))
        history_lines = []
        for i, step in enumerate(steps[-history_length:]):
            step_action = _extract_action(step.get("action", ""))
            step_obs = step.get("observation", "")
            history_lines.append(f"Step {step_count - history_length + i + 1}: Action: {step_action}")
            history_lines.append(f"Observation: {step_obs[:300]}")

        action_history = "\n".join(history_lines) if history_lines else "(no history)"

        # Format available actions
        possible_actions = instance.get("possible_actions", [])
        if possible_actions:
            available_actions = "\n".join(f"- {a}" for a in possible_actions[:30])
        else:
            available_actions = "- look around\n- examine <object>\- task"

        return f"""You are an expert scientist working in a laboratory environment.
Your task is: {goal}
Task name: {task_name}

Prior to this step, you have already taken {step_count - 1} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took:
{action_history}

You are now at step {step_count} and your current observation is:
{raw_observation}

Your admissible actions of the current situation are:
{available_actions}

Now it's your turn to take one action for the current step. You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the science task. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward score for the interaction.

        Args:
            instance_id: The interaction instance ID.

        Returns:
            Reward score between 0.0 and 1.0.
        """
        if instance_id not in self._instance_dict:
            return 0.0

        instance = self._instance_dict[instance_id]

        # Use environment reward if available
        if not self.use_mock and instance.get("env") is not None:
            score = instance.get("last_score", 0.0)
            return score / 100.0 if score > 1.0 else score

        # Use accumulated reward
        if instance["reward"] > 0:
            return instance["reward"]

        # Fallback: estimate from trajectory
        return self._estimate_reward(instance)

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Finalize the interaction and release resources.

        Args:
            instance_id: The interaction instance ID.
        """
        if instance_id in self._instance_dict:
            instance = self._instance_dict[instance_id]
            env = instance.get("env")
            if env is not None:
                try:
                    del env
                except Exception:
                    pass
            del self._instance_dict[instance_id]

    # ── Mock helpers ──────────────────────────────────────────────────

    def _get_mock_initial_observation(self, ground_truth: dict) -> str:
        """Generate a mock initial observation for testing."""
        goal = ground_truth.get("goal", "Complete the science task")
        task_name = ground_truth.get("task_name", "unknown")
        return (
            f"You are in a science laboratory.\n"
            f"Your task is: {goal}\n"
            f"Task name: {task_name}\n\n"
            f"You see workbenches with various equipment, chemical supplies, "
            f"and scientific instruments. A sink is available for water."
        )

    def _get_mock_actions(self) -> list:
        """Return mock admissible actions."""
        return [
            "look around",
            "examine workbench",
            "open cabinet",
            "take beaker from workbench",
            "use stove",
            "toggle stove",
            "pour water into beaker",
            "go to sink",
            "wait",
            "task",
            "inventory",
        ]

    def _process_mock_action(self, action: str, instance: dict) -> Tuple[str, float, bool]:
        """Process an action in mock mode."""
        clean_action = _extract_action(action).lower()

        if "look around" in clean_action:
            obs = "You see a well-equipped laboratory with various instruments and supplies."
        elif "examine" in clean_action:
            obs = "The object appears to be in good condition and ready for use."
        elif "take" in clean_action:
            obs = "You pick up the object carefully."
        elif "pour" in clean_action:
            obs = "You pour the liquid carefully into the container."
        elif "use" in clean_action or "toggle" in clean_action:
            obs = "You use the object. Something happens as a result."
        elif "task" in clean_action:
            obs = f"Your task is: {instance['ground_truth'].get('goal', 'Complete the science task')}"
        else:
            obs = "You perform the action. The environment responds accordingly."

        # Mock scoring: reward after enough diverse actions
        import re as _re
        unique_actions = set()
        for step in instance.get("steps", []):
            sa = _extract_action(step.get("action", "")).lower()
            for kw in ["take", "pour", "mix", "toggle", "use", "open", "examine"]:
                if kw in sa:
                    unique_actions.add(kw)

        if len(unique_actions) >= 4:
            return obs + "\n\nCongratulations! Task completed!", 0.8, True
        elif len(unique_actions) >= 2:
            return obs, 0.3, False
        return obs, 0.0, False

    def _estimate_reward(self, instance: dict) -> float:
        """Estimate reward from trajectory when environment reward is unavailable."""
        reward = 0.0
        steps = instance.get("steps", [])

        unique_actions = set()
        for step in steps:
            a = _extract_action(step.get("action", "")).lower()
            for kw in ["take", "pour", "mix", "toggle", "use", "open", "examine", "look"]:
                if kw in a:
                    unique_actions.add(kw)

        reward += min(0.5, len(unique_actions) * 0.1)
        if len(steps) >= 3:
            reward += 0.2
        if len(steps) <= 15:
            reward += 0.1

        return min(1.0, reward)
