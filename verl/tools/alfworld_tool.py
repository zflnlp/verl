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
Tool class for ALFWorld benchmark.

Provides a single action tool for interacting with the ALFWorld environment.
"""

import logging
import os
from typing import Any

from .base_tool import BaseTool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AlfworldTool(BaseTool):
    """Tool class for ALFWorld environment.

    Provides a single `action` function for executing commands
    in the ALFWorld household simulation environment.
    """

    def __init__(self, config: dict, tool_schema: dict = None):
        super().__init__(config, tool_schema)
        self.use_mock = config.get("use_mock", True)

    async def execute(self, instance_id: str, tool_name: str, parameters: dict, **kwargs) -> Any:
        """Execute an ALFWorld tool function.

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

        if "look" in cmd and "at" not in cmd:
            return "You are in a room with various household items. You see cabinets, countertops, and appliances."
        elif "inventory" in cmd:
            return "You are carrying: nothing."
        elif "go to" in cmd:
            loc = cmd.replace("go to", "").strip()
            return f"You arrive at {loc}. On the {loc} you see nothing special."
        elif "take" in cmd:
            return "You pick up the object."
        elif "put" in cmd:
            return "You put the object down."
        elif "open" in cmd:
            return "You open the container. Inside, you see nothing."
        elif "close" in cmd:
            return "You close the container."
        elif "toggle" in cmd:
            return "You toggle the object. It is now on."
        elif "use" in cmd:
            return "You use the object."
        elif "heat" in cmd:
            return "You heat the object in the microwave. It is now hot."
        elif "clean" in cmd:
            return "You clean the object in the sink. It is now clean."
        elif "cool" in cmd:
            return "You cool the object in the fridge. It is now cool."
        elif "examine" in cmd:
            return "You examine the object closely. It looks ordinary."
        else:
            return f"You attempt to: {command}. The environment responds."
