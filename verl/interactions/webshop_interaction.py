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

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class WebShopInteraction(BaseInteraction):
    """Interaction class for WebShop shopping task.

    This class manages multi-turn interactions with the WebShop environment,
    where the agent needs to search, browse, and purchase products.

    - `start_interaction`: Initialize a new shopping task instance.
    - `generate_response`: Process agent actions and return environment observations.
    - `calculate_score`: Calculate reward based on purchase quality.
    - `finalize_interaction`: Clean up the interaction state.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self.webshop_server = config.get("webshop_server", None)
        self.use_mock = config.get("use_mock", True)
        self.max_steps = config.get("max_steps", 15)

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[dict] = None,
        **kwargs
    ) -> str:
        """Start a new WebShop interaction.

        Args:
            instance_id: Optional unique identifier for this interaction.
            ground_truth: Dictionary containing task information:
                - task_id: The WebShop task ID
                - goal: Target product description
                - category: Product category
                - attributes: Required product attributes

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
        }

        # Get initial observation
        if self.use_mock:
            self._instance_dict[instance_id]["current_observation"] = self._get_mock_initial_observation(ground_truth)
        else:
            # TODO: Connect to real WebShop server
            self._instance_dict[instance_id]["current_observation"] = "Welcome to WebShop!"

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
            observation, reward, is_done = self._process_mock_action(action, instance)
        else:
            # TODO: Send action to real WebShop server
            observation, reward, is_done = "Action processed", 0.0, False

        # Update instance state
        instance["steps"].append({"action": action, "observation": observation})
        instance["current_observation"] = observation
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
        goal = ground_truth.get("goal", "a product") if ground_truth else "a product"
        return f"Welcome to WebShop! Your task is to find and purchase {goal}.\n\nYou can use the following actions:\n- search[query]: Search for products\n- click[item_id]: View product details\n- click[buy]: Purchase the current product"

    def _process_mock_action(self, action: str, instance: dict) -> Tuple[str, float, bool]:
        """Process an action in mock mode.

        Args:
            action: The agent's action string.
            instance: The interaction instance state.

        Returns:
            Tuple of (observation, reward, is_done).
        """
        action_lower = action.lower().strip()

        # Check for purchase action
        if "buy" in action_lower or "purchase" in action_lower:
            # Simulate a successful purchase with some reward
            ground_truth = instance.get("ground_truth", {})
            reward = self._calculate_mock_reward(action, ground_truth)
            observation = f"You have purchased the item! Reward: {reward:.2f}"
            return observation, reward, True

        # Check for search action
        if "search" in action_lower:
            observation = "Found 5 products matching your search. Use click[item_id] to view details."
            return observation, 0.0, False

        # Check for click action
        if "click" in action_lower:
            observation = "Product details displayed. You can click[buy] to purchase or go back."
            return observation, 0.0, False

        # Default response
        observation = "I don't understand that action. Please use search[query], click[item_id], or click[buy]."
        return observation, 0.0, False

    def _calculate_mock_reward(self, action: str, ground_truth: dict) -> float:
        """Calculate a mock reward for testing.

        Args:
            action: The agent's action.
            ground_truth: Task information.

        Returns:
            Mock reward score.
        """
        import random

        # Base reward for purchasing
        reward = 0.5

        # Add some randomness for testing
        reward += random.uniform(0.0, 0.5)

        # Check if action mentions goal attributes
        goal = ground_truth.get("goal", "").lower()
        if goal and any(word in action.lower() for word in goal.split()[:3]):
            reward += 0.2

        return min(1.0, reward)

    def _estimate_reward(self, instance: dict, ground_truth: dict) -> float:
        """Estimate reward from trajectory when environment reward is unavailable.

        Args:
            instance: The interaction instance state.
            ground_truth: Task information.

        Returns:
            Estimated reward score.
        """
        reward = 0.0
        steps = instance.get("steps", [])

        # Check if a buy action was taken
        has_buy = any("buy" in step.get("action", "").lower() for step in steps)
        if has_buy:
            reward += 0.5

        # Check for search actions
        has_search = any("search" in step.get("action", "").lower() for step in steps)
        if has_search:
            reward += 0.1

        # Bonus for efficiency
        num_steps = len(steps)
        if num_steps <= 5:
            reward += 0.2
        elif num_steps <= 10:
            reward += 0.1

        # Check goal matching
        goal = ground_truth.get("goal", "").lower()
        all_actions = " ".join(step.get("action", "") for step in steps).lower()
        if goal and any(word in all_actions for word in goal.split()[:3]):
            reward += 0.2

        return min(1.0, reward)
