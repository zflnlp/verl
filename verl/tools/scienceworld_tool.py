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
from typing import Any

from .base_tool import BaseTool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ScienceWorldTool(BaseTool):
    """Tool class for ScienceWorld environment.

    Provides a single `action` function for executing commands
    in the ScienceWorld simulation environment.
    """

    def __init__(self, config: dict, tool_schema: dict = None):
        super().__init__(config, tool_schema)
        self.use_mock = config.get("use_mock", True)

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
            command = parameters.get("command", "")
            if self.use_mock:
                return self._mock_action(command)
            else:
                # In real mode, the interaction class handles env.step()
                # This tool just passes through the command
                return f"Action submitted: {command}"
        else:
            return f"Unknown tool: {tool_name}"

    def _mock_action(self, command: str) -> str:
        """Generate mock action response."""
        cmd = command.lower().strip()

        if "look around" in cmd or cmd == "look":
            return "You are in a science laboratory. You see workbenches with various equipment."
        elif "examine" in cmd:
            return "Upon examination, the object has the properties you'd expect."
        elif "take" in cmd:
            return "You pick up the object carefully."
        elif "pour" in cmd:
            return "You pour the liquid into the container."
        elif "use" in cmd or "toggle" in cmd:
            return "You use the object. It performs its intended function."
        elif "task" in cmd:
            return "Your task is to complete the science experiment."
        elif "inventory" in cmd:
            return "You are carrying: nothing."
        else:
            return f"You attempt to: {command}. The environment responds."
