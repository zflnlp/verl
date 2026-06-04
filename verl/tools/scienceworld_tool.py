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
Tool class for ScienceWorld benchmark.

Provides a single action tool for interacting with the ScienceWorld environment.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .base_tool import BaseTool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ScienceWorldTool(BaseTool):
    """Tool class for ScienceWorld environment.

    This tool provides a single `action` function for interacting with
    the ScienceWorld simulation environment.
    """

    def __init__(self, config: dict, tool_schema: dict = None):
        super().__init__(config, tool_schema)
        self.use_mock = config.get("use_mock", True)
        self.scienceworld_server = config.get("scienceworld_server", "http://localhost:8080")

    async def execute(self, instance_id: str, tool_name: str, parameters: dict, **kwargs) -> Any:
        """Execute a ScienceWorld tool function.

        Args:
            instance_id: The interaction instance ID.
            tool_name: The name of the tool function to execute.
            parameters: The parameters for the tool function.

        Returns:
            The result of the tool execution.
        """
        if tool_name == "action":
            return await self._action(instance_id, parameters.get("command", ""))
        else:
            return f"Unknown tool: {tool_name}"

    async def _action(self, instance_id: str, command: str) -> str:
        """Execute an action in the ScienceWorld environment.

        Args:
            instance_id: The interaction instance ID.
            command: The action command to execute.

        Returns:
            The result of the action.
        """
        if self.use_mock:
            return self._mock_action(command)
        else:
            return await self._real_action(command)

    async def _real_action(self, command: str) -> str:
        """Execute an action using the real ScienceWorld server.

        Args:
            command: The action command.

        Returns:
            The result from the server.
        """
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.scienceworld_server}/step",
                    json={"action": command}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("observation", f"Action executed: {command}")
                    else:
                        return f"Error executing action: {resp.status}"
        except Exception as e:
            logger.error(f"Action failed: {e}")
            return f"Error: {str(e)}"

    def _mock_action(self, command: str) -> str:
        """Generate mock action response.

        Args:
            command: The action command.

        Returns:
            Mock response.
        """
        command_lower = command.lower().strip()

        # Mock responses for common actions
        if "look around" in command_lower or command_lower == "look":
            return """You are in a science laboratory. You see:
- Workbenches with various equipment
- A sink with running water
- A stove (currently off)
- Shelves with chemicals and supplies
- A window showing the outdoors"""

        elif "examine" in command_lower:
            return "Upon closer examination, the object has the properties you'd expect for this type of scientific equipment."

        elif "open" in command_lower:
            return "You open the container. Inside you can see the contents clearly."

        elif "close" in command_lower:
            return "You close the container securely."

        elif "take" in command_lower:
            return "You pick up the object carefully, holding it securely."

        elif "put" in command_lower:
            return "You place the object in the specified location."

        elif "use" in command_lower:
            return "You use the object. It performs its intended function."

        elif "toggle" in command_lower:
            return "You toggle the device. It is now in a different state."

        elif "pour" in command_lower:
            return "You pour the liquid carefully. It flows into the target container."

        elif "mix" in command_lower:
            return "You mix the contents together. The mixture combines and may change properties."

        elif "go to" in command_lower:
            return "You walk to the specified location. You can now see the area around you."

        elif "wait" in command_lower:
            return "You wait patiently. A moment passes."

        elif "task" in command_lower:
            return "Your task is to complete the science experiment. Follow the scientific method and use the available equipment."

        elif "inventory" in command_lower:
            return "You are carrying: nothing."

        else:
            return f"You attempt to: {command}. The environment responds to your action."
