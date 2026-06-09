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
import re
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
        self.webshop_server = config.get("webshop_server", "http://localhost:3000")
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
            "task_id": ground_truth.get("task_id", "") if ground_truth else "",
        }

        # Get initial observation from WebShop server
        if self.use_mock:
            self._instance_dict[instance_id]["current_observation"] = self._get_mock_initial_observation(ground_truth)
        else:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.webshop_server}/init") as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self._instance_dict[instance_id]["current_observation"] = data.get("observation", "Welcome to WebShop!")
                        else:
                            self._instance_dict[instance_id]["current_observation"] = "Welcome to WebShop!"
            except Exception as e:
                logger.error(f"Failed to connect to WebShop server: {e}")
                self._instance_dict[instance_id]["current_observation"] = "Welcome to WebShop! (server connection failed)"

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
        """Process an action using the real WebShop server.

        Args:
            action: The agent's action string.
            instance: The interaction instance state.

        Returns:
            Tuple of (observation, reward, is_done).
        """
        import aiohttp

        try:
            # Parse the action — paper defines two action types:
            #   search[<query>] and click[<button name>]
            # "buy" is a click target: click[buy]
            action_lower = action.lower().strip()

            if "search" in action_lower:
                match = re.search(r'search\[(.*?)\]', action, re.IGNORECASE)
                query = match.group(1) if match else action
                api_action = {"action": "search", "query": query}
            elif "click" in action_lower:
                match = re.search(r'click\[(.*?)\]', action, re.IGNORECASE)
                target = match.group(1) if match else action
                # click[buy] triggers purchase on the server
                if target.lower() == "buy":
                    api_action = {"action": "buy"}
                else:
                    api_action = {"action": "click", "target": target}
            else:
                # Default: treat as search
                api_action = {"action": "search", "query": action}

            # Send action to WebShop server
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.webshop_server}/step",
                    json=api_action
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        observation = data.get("observation", "No observation returned")
                        reward = data.get("reward", 0.0)
                        is_done = data.get("done", False)
                        return observation, reward, is_done
                    else:
                        error_text = await resp.text()
                        logger.error(f"WebShop server error: {resp.status} - {error_text}")
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
        goal = ground_truth.get("goal", "a product")
        step_count = instance.get("num_steps", 0)
        steps = instance.get("steps", [])

        # Build action history
        history_length = min(3, len(steps))  # Show last 3 steps
        history_lines = []
        for i, step in enumerate(steps[-history_length:]):
            action = step.get("action", "")
            obs = step.get("observation", "")
            history_lines.append(f"Step {step_count - history_length + i + 1}: Action: {action}")
            history_lines.append(f"Observation: {obs[:200]}...")  # Truncate long observations

        action_history = "\n".join(history_lines) if history_lines else "(no history)"

        # Available actions per paper format: search[<query>] and click[<button name>]
        # "buy" is a click target, not a separate action type
        available_actions = "search[<query>]: Search for products using a text query\nclick[<button name>]: Click on interactive elements (e.g., product links, filter buttons, pagination)"

        return f"""You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {goal}.

Prior to this step, you have already taken {step_count - 1} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took:
{action_history}

You are now at step {step_count} and your current observation is:
{raw_observation}

Your admissible actions of the current situation are:
{available_actions}

Now it's your turn to take one action for the current step. You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

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
        return f"""You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {goal}.

Prior to this step, you have already taken 0 step(s).
Below are the most recent 0 observations and the corresponding actions you took:
(no history)

You are now at step 1 and your current observation is:
Welcome to WebShop! You can search for products and browse listings.

Your admissible actions of the current situation are:
- search[<query>]: Search for products using a text query
- click[<button name>]: Click on interactive elements (e.g., product links, filter buttons, pagination)

Now it's your turn to take one action for the current step. You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

    def _process_mock_action(self, action: str, instance: dict) -> Tuple[str, float, bool]:
        """Process an action in mock mode.

        Paper defines two action types: search[<query>] and click[<button name>].
        "buy" is a click target: click[buy].

        Args:
            action: The agent's action string.
            instance: The interaction instance state.

        Returns:
            Tuple of (observation, reward, is_done).
        """
        action_lower = action.lower().strip()

        # search[<query>]
        if "search" in action_lower:
            observation = "Found 5 products matching your search. Use click[item_id] to view details."
            return observation, 0.0, False

        # click[<button name>] — including click[buy]
        if "click" in action_lower:
            # Extract click target
            match = re.search(r'click\[(.*?)\]', action, re.IGNORECASE)
            target = match.group(1).strip() if match else ""

            if target.lower() == "buy":
                # click[buy] triggers purchase
                ground_truth = instance.get("ground_truth", {})
                reward = self._calculate_mock_reward(action, ground_truth)
                observation = f"You have purchased the item! Reward: {reward:.2f}"
                return observation, reward, True
            else:
                observation = "Product details displayed. You can click[buy] to purchase or go back."
                return observation, 0.0, False

        # Default response
        observation = "I don't understand that action. Please use search[query] or click[button]."
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
