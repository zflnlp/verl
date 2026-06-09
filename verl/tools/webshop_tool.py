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
from typing import Any, Dict, List, Optional

from .base_tool import BaseTool

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class WebShopTool(BaseTool):
    """Tool class for WebShop shopping environment.

    This tool provides functions for interacting with the WebShop environment,
    including searching for products, viewing details, and making purchases.
    """

    def __init__(self, config: dict, tool_schema: dict = None):
        super().__init__(config, tool_schema)
        self.use_mock = config.get("use_mock", True)
        self.webshop_server = config.get("webshop_server", "http://localhost:3000")

    async def execute(self, instance_id: str, tool_name: str, parameters: dict, **kwargs) -> Any:
        """Execute a WebShop tool function.

        Args:
            instance_id: The interaction instance ID.
            tool_name: The name of the tool function to execute.
            parameters: The parameters for the tool function.

        Returns:
            The result of the tool execution.
        """
        if tool_name == "search":
            return await self._search(instance_id, parameters.get("query", ""))
        elif tool_name == "click":
            return await self._click(instance_id, parameters.get("target", ""))
        else:
            return f"Unknown tool: {tool_name}"

    async def _search(self, instance_id: str, query: str) -> str:
        """Search for products.

        Args:
            instance_id: The interaction instance ID.
            query: The search query.

        Returns:
            Search results as a string.
        """
        if self.use_mock:
            return self._mock_search(query)
        else:
            return await self._real_search(query)

    async def _click(self, instance_id: str, target: str) -> str:
        """Click on an item or button.

        Args:
            instance_id: The interaction instance ID.
            target: The item ID or button name to click.

        Returns:
            The result of clicking.
        """
        if self.use_mock:
            return self._mock_click(target)
        else:
            return await self._real_click(target)

    async def _real_search(self, query: str) -> str:
        """Search for products using the real WebShop server.

        Args:
            query: The search query.

        Returns:
            Search results from the server.
        """
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.webshop_server}/step",
                    json={"action": "search", "query": query}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("observation", f"Search results for: {query}")
                    else:
                        return f"Error searching: {resp.status}"
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Error: {str(e)}"

    async def _real_click(self, target: str) -> str:
        """Click on an item or button using the real WebShop server.

        Args:
            target: The item ID or button to click (e.g., 'B001' or 'buy').

        Returns:
            The result of clicking.
        """
        import aiohttp

        # "buy" is a special click target that triggers purchase
        if target.lower() == "buy":
            api_action = {"action": "buy"}
        else:
            api_action = {"action": "click", "target": target}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.webshop_server}/step",
                    json=api_action
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("observation", f"Clicked on: {target}")
                    else:
                        return f"Error clicking: {resp.status}"
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return f"Error: {str(e)}"

    def _mock_search(self, query: str) -> str:
        """Generate mock search results.

        Args:
            query: The search query.

        Returns:
            Mock search results.
        """
        # Mock product database
        products = [
            {"id": "B001", "name": "Classic Red T-Shirt", "price": "$19.99", "rating": "4.5/5"},
            {"id": "B002", "name": "Blue Cotton Polo", "price": "$29.99", "rating": "4.2/5"},
            {"id": "B003", "name": "Black Running Shoes", "price": "$89.99", "rating": "4.7/5"},
            {"id": "B004", "name": "Wireless Headphones", "price": "$49.99", "rating": "4.3/5"},
            {"id": "B005", "name": "Summer Floral Dress", "price": "$39.99", "rating": "4.6/5"},
        ]

        # Filter products based on query
        query_lower = query.lower()
        matching = [p for p in products if any(word in p["name"].lower() for word in query_lower.split())]

        if not matching:
            matching = products[:3]  # Return first 3 if no match

        # Format results
        results = [f"Search results for '{query}':"]
        for i, p in enumerate(matching, 1):
            results.append(f"{i}. [{p['id']}] {p['name']} - {p['price']} (Rating: {p['rating']})")

        results.append("\nUse click[item_id] to view details (e.g., click[B001])")
        return "\n".join(results)

    def _mock_click(self, target: str) -> str:
        """Generate mock click response.

        Args:
            target: The item ID or button to click (e.g., 'B001' or 'buy').

        Returns:
            Mock response after clicking.
        """
        # Handle buy as a click target
        if target.lower() == "buy":
            return "Purchase successful! Your order has been placed."

        products = {
            "B001": {
                "name": "Classic Red T-Shirt",
                "price": "$19.99",
                "rating": "4.5/5",
                "description": "A comfortable red t-shirt made from 100% cotton.",
                "colors": ["Red", "Blue", "Black"],
                "sizes": ["S", "M", "L", "XL"],
            },
            "B002": {
                "name": "Blue Cotton Polo",
                "price": "$29.99",
                "rating": "4.2/5",
                "description": "A stylish blue polo shirt for casual occasions.",
                "colors": ["Blue", "White", "Navy"],
                "sizes": ["M", "L", "XL"],
            },
        }

        if target.startswith("B") and target in products:
            p = products[target]
            return f"""Product: {p['name']}
Price: {p['price']}
Rating: {p['rating']}
Description: {p['description']}
Available Colors: {', '.join(p['colors'])}
Available Sizes: {', '.join(p['sizes'])}

Use click[buy] to purchase this item."""
        else:
            return f"Clicked on: {target}. Please use a valid item ID."
