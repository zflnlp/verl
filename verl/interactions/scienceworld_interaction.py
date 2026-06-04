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
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ScienceWorldInteraction(BaseInteraction):
    """Interaction class for ScienceWorld benchmark.

    This class manages multi-turn interactions with the ScienceWorld environment,
    where the agent needs to perform scientific experiments to complete tasks.

    - `start_interaction`: Initialize a new ScienceWorld task instance.
    - `generate_response`: Process agent actions and return environment observations.
    - `calculate_score`: Calculate reward based on task completion.
    - `finalize_interaction`: Clean up the interaction state.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self.use_mock = config.get("use_mock", True)
        self.max_steps = config.get("max_steps", 25)
        self.scienceworld_server = config.get("scienceworld_server", "http://localhost:8080")

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
                - task_id: The ScienceWorld task ID
                - task_name: Name of the science task
                - goal: Description of the goal
                - task_variations: Task variation indices

        Returns:
            The instance ID for this interaction.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        self._instance_dict[instance_id] = {
            "ground_truth": ground_truth or {},
            "steps": [],
            "current_observation": "",
            "reward": 0.0,
            "is_done": False,
            "num_steps": 0,
            "task_name": ground_truth.get("task_name", "unknown") if ground_truth else "unknown",
        }

        # Get initial observation
        if self.use_mock:
            self._instance_dict[instance_id]["current_observation"] = self._get_mock_initial_observation(ground_truth)
        else:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    task_name = ground_truth.get("task_name", "") if ground_truth else ""
                    async with session.post(
                        f"{self.scienceworld_server}/init",
                        json={"task": task_name}
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self._instance_dict[instance_id]["current_observation"] = data.get("observation", "You are in a science laboratory.")
                        else:
                            self._instance_dict[instance_id]["current_observation"] = "You are in a science laboratory."
            except Exception as e:
                logger.error(f"Failed to connect to ScienceWorld server: {e}")
                self._instance_dict[instance_id]["current_observation"] = "You are in a science laboratory. (server connection failed)"

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
            raw_observation, reward, is_done = await self._process_real_action(action, instance)

        # Format observation in paper style
        observation = self._format_observation(instance, raw_observation)

        # Update instance state
        instance["steps"].append({"action": action, "observation": raw_observation})
        instance["current_observation"] = raw_observation
        instance["reward"] = reward

        # Check termination conditions
        should_terminate = is_done or instance["num_steps"] >= self.max_steps

        if should_terminate:
            instance["is_done"] = True
            # Calculate final reward
            final_reward = await self.calculate_score(instance_id)
            instance["reward"] = final_reward
            return True, observation, final_reward, {"num_steps": instance["num_steps"]}
        else:
            return False, observation, 0.0, {"num_steps": instance["num_steps"]}

    async def _process_real_action(self, action: str, instance: dict) -> Tuple[str, float, bool]:
        """Process an action using the real ScienceWorld server.

        Args:
            action: The agent's action string.
            instance: The interaction instance state.

        Returns:
            Tuple of (observation, reward, is_done).
        """
        import aiohttp

        try:
            # Clean up action - extract from <action> tags if present
            import re
            action_match = re.search(r'<action>\s*(.*?)\s*</action>', action, re.IGNORECASE | re.DOTALL)
            if action_match:
                clean_action = action_match.group(1).strip()
            else:
                clean_action = action.strip()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.scienceworld_server}/step",
                    json={"action": clean_action}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        observation = data.get("observation", "Nothing happened.")
                        reward = data.get("reward", 0.0)
                        is_done = data.get("done", False)
                        return observation, reward, is_done
                    else:
                        error_text = await resp.text()
                        logger.error(f"ScienceWorld server error: {resp.status} - {error_text}")
                        return f"Error: Server returned {resp.status}", 0.0, False

        except Exception as e:
            logger.error(f"Failed to process action: {e}")
            return f"Error: {str(e)}", 0.0, False

    def _format_observation(self, instance: dict, raw_observation: str) -> str:
        """Format observation in the paper's style.

        Args:
            instance: The interaction instance state.
            raw_observation: The raw observation from the environment.

        Returns:
            Formatted observation string.
        """
        ground_truth = instance.get("ground_truth", {})
        goal = ground_truth.get("goal", "Complete the science task")
        task_name = instance.get("task_name", "unknown")
        step_count = instance.get("num_steps", 0)
        steps = instance.get("steps", [])

        # Build action history
        history_length = min(3, len(steps))
        history_lines = []
        for i, step in enumerate(steps[-history_length:]):
            action = step.get("action", "")
            obs = step.get("observation", "")
            history_lines.append(f"Step {step_count - history_length + i + 1}: Action: {action}")
            history_lines.append(f"Observation: {obs[:200]}...")

        action_history = "\n".join(history_lines) if history_lines else "(no history)"

        # Available actions for ScienceWorld
        available_actions = """- look around: Describe the current room and visible objects
- examine <object>: Examine an object closely
- open <object>: Open a container or door
- close <object>: Close a container or door
- take <object> from <location>: Pick up an object
- put <object> in/on <location>: Place an object somewhere
- use <object> [on <object>]: Use an object (optionally on another)
- toggle <object>: Turn something on or off
- pour <object> into <object>: Pour a liquid
- mix <object>: Mix contents of a container
- go to <location>: Move to a different location
- look at <object>: Look at something specific
- wait: Wait for something to happen
- task: Describe the current task"""

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
        ground_truth = instance.get("ground_truth", {})

        # If we have a reward from the environment, use it
        if instance["reward"] > 0:
            return instance["reward"]

        # Fallback: estimate reward from trajectory
        return self._estimate_reward(instance, ground_truth)

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Finalize the interaction and release resources.

        Args:
            instance_id: The interaction instance ID.
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

    def _get_mock_initial_observation(self, ground_truth: Optional[dict] = None) -> str:
        """Generate a mock initial observation for testing.

        Args:
            ground_truth: Task information.

        Returns:
            Mock observation string.
        """
        goal = ground_truth.get("goal", "Complete the science task") if ground_truth else "Complete the science task"
        task_name = ground_truth.get("task_name", "unknown") if ground_truth else "unknown"

        # Generate task-specific mock observations
        mock_lab_scenes = {
            "boiling-water": """You are in a laboratory. In front of you, you see:
- a stove (turned off)
- a metal pot
- a sink with running water
- a thermometer
- a beaker

On the table there is:
- a notebook""",
            "growing-plants": """You are in a laboratory. In front of you, you see:
- a windowsill with a flowerpot
- a bag of soil
- seeds (sunflower)
- a watering can
- a ruler

On the table there is:
- a growth chart""",
            "chemistry-mix": """You are in a laboratory. In front of you, you see:
- a workbench with test tubes
- a bunsen burner
- various chemicals (baking soda, vinegar, citric acid)
- safety goggles
- a test tube rack

On the table there is:
- a pH meter""",
        }

        # Return task-specific or default observation
        for key, scene in mock_lab_scenes.items():
            if key in task_name.lower():
                return f"""You are an expert scientist working in a laboratory environment.
Your task is: {goal}
Task name: {task_name}

Prior to this step, you have already taken 0 step(s).
Below are the most recent 0 observations and the corresponding actions you took:
(no history)

You are now at step 1 and your current observation is:
{scene}

Your admissible actions of the current situation are:
- look around: Describe the current room and visible objects
- examine <object>: Examine an object closely
- open <object>: Open a container or door
- close <object>: Close a container or door
- take <object> from <location>: Pick up an object
- put <object> in/on <location>: Place an object somewhere
- use <object> [on <object>]: Use an object (optionally on another)
- toggle <object>: Turn something on or off
- pour <object> into <object>: Pour a liquid
- mix <object>: Mix contents of a container
- go to <location>: Move to a different location
- look at <object>: Look at something specific
- wait: Wait for something to happen
- task: Describe the current task

Now it's your turn to take one action for the current step. You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the science task. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

        # Default lab scene
        return f"""You are an expert scientist working in a laboratory environment.
Your task is: {goal}
Task name: {task_name}

Prior to this step, you have already taken 0 step(s).
Below are the most recent 0 observations and the corresponding actions you took:
(no history)

You are now at step 1 and your current observation is:
You are in a well-equipped science laboratory. There are workbenches with various equipment, chemical supplies, and scientific instruments. A sink is available for water.

Your admissible actions of the current situation are:
- look around: Describe the current room and visible objects
- examine <object>: Examine an object closely
- open <object>: Open a container or door
- close <object>: Close a container or door
- take <object> from <location>: Pick up an object
- put <object> in/on <location>: Place an object somewhere
- use <object> [on <object>]: Use an object (optionally on another)
- toggle <object>: Turn something on or off
- pour <object> into <object>: Pour a liquid
- mix <object>: Mix contents of a container
- go to <location>: Move to a different location
- look at <object>: Look at something specific
- wait: Wait for something to happen
- task: Describe the current task

Now it's your turn to take one action for the current step. You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the science task. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

    def _process_mock_action(self, action: str, instance: dict) -> Tuple[str, float, bool]:
        """Process an action in mock mode.

        Args:
            action: The agent's action string.
            instance: The interaction instance state.

        Returns:
            Tuple of (observation, reward, is_done).
        """
        import re

        # Extract action from <action> tags if present
        action_match = re.search(r'<action>\s*(.*?)\s*</action>', action, re.IGNORECASE | re.DOTALL)
        if action_match:
            clean_action = action_match.group(1).strip().lower()
        else:
            clean_action = action.lower().strip()

        # Track progress for scoring
        ground_truth = instance.get("ground_truth", {})
        task_name = instance.get("task_name", "").lower()
        steps = instance.get("steps", [])

        # Simulate different action responses
        if "look around" in clean_action or "look" in clean_action:
            observation = "You see a well-equipped laboratory with various instruments and supplies on the workbenches."
        elif "examine" in clean_action:
            observation = "The object appears to be in good condition and ready for use."
        elif "open" in clean_action:
            observation = "You open the container. Inside you can see the contents clearly."
        elif "close" in clean_action:
            observation = "You close the container."
        elif "take" in clean_action:
            observation = "You pick up the object carefully."
        elif "put" in clean_action:
            observation = "You place the object in the specified location."
        elif "use" in clean_action:
            observation = "You use the object. Something happens as a result."
        elif "toggle" in clean_action:
            observation = "You toggle the device. It changes state."
        elif "pour" in clean_action:
            observation = "You pour the liquid carefully into the container."
        elif "mix" in clean_action:
            observation = "You mix the contents. The mixture changes color/consistency."
        elif "go to" in clean_action:
            observation = "You move to the new location. You can see the area around you."
        elif "wait" in clean_action:
            observation = "You wait for a moment. Time passes."
        elif "task" in clean_action:
            goal = ground_truth.get("goal", "Complete the science task")
            observation = f"Your current task is: {goal}"
        else:
            observation = "You perform the action. The environment responds accordingly."

        # Check for task completion (mock: reward after enough varied actions)
        unique_actions = set()
        for step in steps:
            step_action = step.get("action", "").lower()
            if "<action>" in step_action:
                match = re.search(r'<action>\s*(.*?)\s*</action>', step_action, re.IGNORECASE | re.DOTALL)
                if match:
                    step_action = match.group(1).strip().lower()
            # Categorize actions
            for keyword in ["take", "pour", "mix", "toggle", "use", "open", "examine"]:
                if keyword in step_action:
                    unique_actions.add(keyword)

        # Simple scoring: reward increases with more diverse actions
        if len(unique_actions) >= 4:
            reward = 0.8
            is_done = True
            observation += "\n\nCongratulations! You have successfully completed the science task!"
        elif len(unique_actions) >= 2:
            reward = 0.3
            is_done = False
        else:
            reward = 0.0
            is_done = False

        return observation, reward, is_done

    def _estimate_reward(self, instance: dict, ground_truth: dict) -> float:
        """Estimate reward from trajectory when environment reward is unavailable.

        Args:
            instance: The interaction instance state.
            ground_truth: Task information.

        Returns:
            Estimated reward score.
        """
        import re

        reward = 0.0
        steps = instance.get("steps", [])

        # Count unique action types
        unique_actions = set()
        for step in steps:
            action = step.get("action", "").lower()
            if "<action>" in action:
                match = re.search(r'<action>\s*(.*?)\s*</action>', action, re.IGNORECASE | re.DOTALL)
                if match:
                    action = match.group(1).strip().lower()
            for keyword in ["take", "pour", "mix", "toggle", "use", "open", "examine", "look"]:
                if keyword in action:
                    unique_actions.add(keyword)

        # Reward for action diversity
        reward += min(0.5, len(unique_actions) * 0.1)

        # Reward for reasonable number of steps
        num_steps = len(steps)
        if num_steps >= 3:
            reward += 0.2
        if num_steps <= 15:
            reward += 0.1

        # Bonus for task-related actions
        goal = ground_truth.get("goal", "").lower()
        all_actions = " ".join(step.get("action", "") for step in steps).lower()
        if goal and any(word in all_actions for word in goal.split()[:3]):
            reward += 0.2

        return min(1.0, reward)
