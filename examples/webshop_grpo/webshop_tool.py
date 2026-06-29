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
WebShop environment tool for verl GRPO training.

This module wraps the WebShop environment as a verl tool, enabling
multi-turn agent interaction for web shopping tasks.

WebShop is an RL environment where agents interact with a simulated
e-commerce website to find and purchase products based on natural
language instructions.

Reference: https://webshop-pnlpnlpnlp.github.io/
"""

import logging
import os
from typing import Any, Optional

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)


class WebShopTool(BaseTool):
    """WebShop environment tool for multi-turn agent interaction.

    This tool wraps the WebShop environment, allowing the agent to:
    - Search for products
    - Click on products to view details
    - Navigate through pages
    - Buy products

    The tool maintains an environment instance per trajectory and
    calculates rewards based on task completion.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.env_config = config.get("env_config", {})
        self.max_steps = config.get("max_steps", 10)
        self.sessions: dict[str, dict[str, Any]] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a new WebShop environment session.

        Args:
            instance_id: Optional instance ID to use.

        Returns:
            Tuple of (instance_id, initial_observation).
        """
        instance_id, _ = await super().create(instance_id, **kwargs)

        try:
            # Initialize WebShop environment
            # NOTE: You need to install and configure WebShop separately
            # pip install webshop
            from webshop.web_agent_text_env import WebAgentTextEnv

            env = WebAgentTextEnv(
                observation_mode="text",
                server=self.env_config.get("server", "http://localhost:3000"),
            )

            # Get initial observation
            obs, info = env.reset()

            self.sessions[instance_id] = {
                "env": env,
                "steps": 0,
                "max_steps": self.max_steps,
                "task": info.get("task", ""),
                "goal": info.get("goal", ""),
                "done": False,
                "reward": 0.0,
                "purchase_successful": False,
            }

            # Format initial observation for the agent
            response_text = self._format_observation(obs, info)

            return instance_id, ToolResponse(text=response_text)

        except ImportError:
            logger.error(
                "WebShop is not installed. Please install it with: "
                "pip install webshop\n"
                "And start the WebShop server: "
                "python -m webshop.run_server"
            )
            return instance_id, ToolResponse(
                text="Error: WebShop environment is not available. Please check installation."
            )
        except Exception as e:
            logger.error(f"Failed to create WebShop session: {e}")
            return instance_id, ToolResponse(text=f"Error initializing WebShop: {e}")

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """Execute an action in the WebShop environment.

        Args:
            instance_id: The session instance ID.
            parameters: Action parameters containing:
                - action (str): One of "search[query]", "click[button]", "buy"
                - query (str, optional): Search query (for search action)

        Returns:
            Tuple of (observation, step_reward, metrics).
        """
        if instance_id not in self.sessions:
            return ToolResponse(text="Error: Invalid session."), 0.0, {}

        session = self.sessions[instance_id]

        if session["done"]:
            return ToolResponse(text="Episode already finished. Please create a new session."), 0.0, {}

        try:
            # Parse action from parameters
            action = parameters.get("action", "")
            query = parameters.get("query", "")

            # Format action for WebShop
            if action == "search" and query:
                env_action = f"search[{query}]"
            elif action == "click":
                button = parameters.get("button", "")
                env_action = f"click[{button}]"
            elif action == "buy":
                env_action = "buy"
            elif action.startswith("search[") or action.startswith("click["):
                env_action = action
            else:
                return (
                    ToolResponse(text=f"Invalid action: {action}. Use 'search', 'click', or 'buy'."),
                    0.0,
                    {"error": "invalid_action"},
                )

            # Execute action in environment
            obs, reward, done, info = session["env"].step(env_action)

            # Update session state
            session["steps"] += 1
            session["done"] = done or session["steps"] >= session["max_steps"]
            session["reward"] = reward
            session["purchase_successful"] = info.get("purchase_successful", False)

            # Format response
            response_text = self._format_observation(obs, info)
            step_reward = 0.0  # Reward is calculated at the end

            metrics = {
                "steps": session["steps"],
                "done": session["done"],
                "action": env_action,
            }

            return ToolResponse(text=response_text), step_reward, metrics

        except Exception as e:
            logger.error(f"Error executing WebShop action: {e}")
            return ToolResponse(text=f"Error: {e}"), 0.0, {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the final reward for the WebShop session.

        The reward is based on:
        - Whether the agent successfully purchased a product (1.0 or 0.0)
        - How closely the purchased item matches the instruction
        - Number of steps taken (efficiency bonus)

        Args:
            instance_id: The session instance ID.

        Returns:
            Final reward score between 0.0 and 1.0.
        """
        if instance_id not in self.sessions:
            return 0.0

        session = self.sessions[instance_id]

        if not session["done"]:
            return 0.0

        # Get the reward from WebShop environment
        # WebShop returns reward as the match score (0.0 to 1.0)
        base_reward = session["reward"]

        # Optional: Add efficiency bonus for fewer steps
        efficiency_bonus = max(0, 1.0 - session["steps"] / session["max_steps"]) * 0.1

        final_reward = min(1.0, base_reward + efficiency_bonus)

        logger.info(
            f"WebShop session {instance_id}: "
            f"reward={base_reward:.3f}, "
            f"efficiency_bonus={efficiency_bonus:.3f}, "
            f"final_reward={final_reward:.3f}, "
            f"steps={session['steps']}"
        )

        return final_reward

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the WebShop environment session.

        Args:
            instance_id: The session instance ID.
        """
        if instance_id in self.sessions:
            try:
                session = self.sessions[instance_id]
                if "env" in session:
                    session["env"].close()
            except Exception as e:
                logger.warning(f"Error closing WebShop session: {e}")
            finally:
                del self.sessions[instance_id]

    def _format_observation(self, obs: str, info: dict[str, Any]) -> str:
        """Format the WebShop observation for the agent.

        Args:
            obs: Raw observation from WebShop.
            info: Additional info from the environment.

        Returns:
            Formatted observation string.
        """
        # Clean up the observation
        formatted = obs.strip()

        # Add available actions hint
        if "available_actions" in info:
            actions = info["available_actions"]
            formatted += f"\n\nAvailable actions: {actions}"

        return formatted
