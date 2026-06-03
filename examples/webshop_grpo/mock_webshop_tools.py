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
Mock WebShop tools for testing and development.

These tools simulate WebShop environment behavior without requiring
an actual WebShop server. Useful for:
- Testing the training pipeline
- Development and debugging
- Running on machines without WebShop installed

Usage:
    # Use mock tools instead of real ones
    actor_rollout_ref.rollout.multi_turn.function_tool_path=examples/webshop_grpo/mock_webshop_tools.py
"""

import json
import random
from typing import Any

from verl.tools.function_tool import function_tool


# Mock product database
MOCK_PRODUCTS = [
    {
        "id": 1,
        "name": "Classic Red T-Shirt",
        "color": "red",
        "size": ["S", "M", "L", "XL"],
        "price": 19.99,
        "rating": 4.5,
        "category": "clothing",
    },
    {
        "id": 2,
        "name": "Blue Cotton Polo",
        "color": "blue",
        "size": ["M", "L", "XL"],
        "price": 29.99,
        "rating": 4.2,
        "category": "clothing",
    },
    {
        "id": 3,
        "name": "Black Running Shoes",
        "color": "black",
        "size": ["8", "9", "10", "11"],
        "price": 89.99,
        "rating": 4.7,
        "category": "shoes",
    },
    {
        "id": 4,
        "name": "Wireless Headphones",
        "color": "black",
        "features": ["noise-cancelling", "bluetooth"],
        "price": 49.99,
        "rating": 4.3,
        "category": "electronics",
    },
    {
        "id": 5,
        "name": "Summer Floral Dress",
        "color": "pink",
        "size": ["S", "M", "L"],
        "price": 39.99,
        "rating": 4.6,
        "category": "clothing",
    },
]

# Mock sessions
_mock_sessions: dict[str, dict[str, Any]] = {}


def _get_mock_session(task_id: str) -> dict[str, Any]:
    """Get or create a mock session."""
    if task_id not in _mock_sessions:
        _mock_sessions[task_id] = {
            "task_id": task_id,
            "steps": 0,
            "max_steps": 10,
            "done": False,
            "current_page": "home",
            "selected_product": None,
            "purchased": False,
            "history": [],
        }
    return _mock_sessions[task_id]


def _search_products(query: str) -> list[dict[str, Any]]:
    """Search mock products."""
    query_lower = query.lower()
    results = []
    for product in MOCK_PRODUCTS:
        # Simple keyword matching
        if any(
            keyword in str(product).lower()
            for keyword in query_lower.split()
        ):
            results.append(product)
    return results[:3]  # Return top 3


def _format_product_list(products: list[dict[str, Any]]) -> str:
    """Format product list for display."""
    if not products:
        return "No products found matching your search."

    lines = ["Search Results:"]
    for i, product in enumerate(products, 1):
        lines.append(f"\n{i}. {product['name']}")
        lines.append(f"   Price: ${product['price']:.2f}")
        lines.append(f"   Rating: {product['rating']}/5.0")
        if 'color' in product:
            lines.append(f"   Color: {product['color']}")
        if 'size' in product:
            lines.append(f"   Sizes: {', '.join(product['size'])}")
    lines.append("\nClick on a product number to view details, or search again.")
    return "\n".join(lines)


def _format_product_detail(product: dict[str, Any]) -> str:
    """Format product details for display."""
    lines = [f"Product: {product['name']}"]
    lines.append(f"Price: ${product['price']:.2f}")
    lines.append(f"Rating: {product['rating']}/5.0")
    lines.append(f"Category: {product['category']}")
    if 'color' in product:
        lines.append(f"Color: {product['color']}")
    if 'size' in product:
        lines.append(f"Available Sizes: {', '.join(product['size'])}")
    if 'features' in product:
        lines.append(f"Features: {', '.join(product['features'])}")
    lines.append("\nActions:")
    lines.append("- Select size: click[S], click[M], click[L], etc.")
    lines.append("- Buy now: buy")
    lines.append("- Go back: click[Back]")
    return "\n".join(lines)


@function_tool
def mock_webshop_search(query: str, task_id: str) -> str:
    """Search for products in the mock WebShop.

    This is a mock implementation for testing. It simulates
    product search without requiring a real WebShop server.

    Args:
        query: Search query (e.g., "red t-shirt", "shoes", "headphones").
        task_id: Unique task identifier.

    Returns:
        Search results with product listings.
    """
    session = _get_mock_session(task_id)

    if session["done"]:
        return json.dumps({
            "error": "Session ended",
            "steps": session["steps"],
        })

    session["steps"] += 1
    session["current_page"] = "search_results"
    session["history"].append({"action": f"search[{query}]"})

    products = _search_products(query)
    session["search_results"] = products

    result = {
        "observation": _format_product_list(products),
        "step": session["steps"],
        "max_steps": session["max_steps"],
        "products_found": len(products),
    }

    return json.dumps(result, ensure_ascii=False)


@function_tool
def mock_webshop_click(element: str, task_id: str) -> str:
    """Click on an element in the mock WebShop.

    This is a mock implementation for testing.

    Args:
        element: Element to click (product number, button name, etc.).
        task_id: Unique task identifier.

    Returns:
        Updated page content.
    """
    session = _get_mock_session(task_id)

    if session["done"]:
        return json.dumps({
            "error": "Session ended",
            "steps": session["steps"],
        })

    session["steps"] += 1
    session["history"].append({"action": f"click[{element}]"})

    # Handle product selection
    if element.isdigit():
        product_idx = int(element) - 1
        if "search_results" in session and 0 <= product_idx < len(session["search_results"]):
            product = session["search_results"][product_idx]
            session["selected_product"] = product
            session["current_page"] = "product_detail"

            result = {
                "observation": _format_product_detail(product),
                "step": session["steps"],
                "product": product,
            }
        else:
            result = {
                "observation": f"Invalid product number: {element}. Please select a valid product.",
                "step": session["steps"],
            }
    # Handle size selection
    elif element in ["S", "M", "L", "XL", "XXL"] or element.isdigit():
        if session["selected_product"]:
            session["selected_size"] = element
            result = {
                "observation": f"Selected size: {element}\n\nReady to purchase. Click 'buy' to complete order.",
                "step": session["steps"],
                "selected_size": element,
            }
        else:
            result = {
                "observation": "Please select a product first.",
                "step": session["steps"],
            }
    # Handle other buttons
    elif element.lower() in ["buy now", "buy"]:
        result = {
            "observation": "Please use the 'buy' tool to purchase.",
            "step": session["steps"],
        }
    elif element.lower() == "back":
        session["current_page"] = "search_results"
        session["selected_product"] = None
        result = {
            "observation": "Returned to search results.",
            "step": session["steps"],
        }
    else:
        result = {
            "observation": f"Clicked: {element}",
            "step": session["steps"],
        }

    return json.dumps(result, ensure_ascii=False)


@function_tool
def mock_webshop_buy(task_id: str) -> str:
    """Purchase the selected product in mock WebShop.

    This is a mock implementation for testing.

    Args:
        task_id: Unique task identifier.

    Returns:
        Purchase confirmation and reward.
    """
    session = _get_mock_session(task_id)

    if session["done"]:
        return json.dumps({
            "error": "Session ended",
            "steps": session["steps"],
        })

    session["steps"] += 1
    session["done"] = True
    session["history"].append({"action": "buy"})

    if session["selected_product"]:
        product = session["selected_product"]
        session["purchased"] = True

        # Calculate mock reward (higher for matching products)
        base_reward = 0.7
        if product.get("rating", 0) >= 4.5:
            base_reward += 0.1
        if product.get("price", 100) < 50:
            base_reward += 0.1

        # Add some randomness
        reward = min(1.0, base_reward + random.uniform(-0.1, 0.1))

        result = {
            "observation": (
                f"✓ Purchase successful!\n\n"
                f"Order Summary:\n"
                f"- Product: {product['name']}\n"
                f"- Price: ${product['price']:.2f}\n"
                f"- Size: {session.get('selected_size', 'M')}\n\n"
                f"Thank you for shopping with us!"
            ),
            "reward": round(reward, 3),
            "done": True,
            "purchased": True,
            "product": product,
            "total_steps": session["steps"],
        }
    else:
        result = {
            "observation": "No product selected. Please search and select a product first.",
            "reward": 0.0,
            "done": True,
            "purchased": False,
            "total_steps": session["steps"],
        }

    return json.dumps(result, ensure_ascii=False)


@function_tool
def mock_webshop_get_status(task_id: str) -> str:
    """Get current mock WebShop session status.

    Args:
        task_id: Unique task identifier.

    Returns:
        Session status information.
    """
    session = _get_mock_session(task_id)

    result = {
        "task_id": task_id,
        "steps": session["steps"],
        "max_steps": session["max_steps"],
        "steps_remaining": session["max_steps"] - session["steps"],
        "current_page": session["current_page"],
        "selected_product": session.get("selected_product"),
        "done": session["done"],
        "purchased": session["purchased"],
    }

    return json.dumps(result, ensure_ascii=False)


# Cleanup function
def cleanup_mock_sessions():
    """Clean up all mock sessions."""
    _mock_sessions.clear()
