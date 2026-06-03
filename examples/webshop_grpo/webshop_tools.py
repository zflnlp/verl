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
WebShop function tools for verl GRPO training.

This module provides FunctionTool wrappers for WebShop environment
interactions, making it easy to integrate with verl's tool system.

Usage:
    These tools are automatically loaded when specified in the training config:
    actor_rollout_ref.rollout.multi_turn.function_tool_path=examples/webshop_grpo/webshop_tools.py
"""

import json
import logging
import os
from typing import Any, Optional

from verl.tools.function_tool import function_tool

logger = logging.getLogger(__name__)

# Global WebShop environment sessions
# In production, you'd want a more robust session management
_sessions: dict[str, Any] = {}


def _get_or_create_session(
    task_id: str,
    server: str = "http://localhost:3000",
    max_steps: int = 10,
) -> tuple[Any, dict[str, Any]]:
    """Get or create a WebShop environment session.

    Args:
        task_id: Unique task identifier.
        server: WebShop server URL.
        max_steps: Maximum steps per episode.

    Returns:
        Tuple of (environment, session_info).
    """
    if task_id in _sessions:
        return _sessions[task_id]["env"], _sessions[task_id]

    try:
        from webshop.web_agent_text_env import WebAgentTextEnv

        env = WebAgentTextEnv(
            observation_mode="text",
            server=server,
        )
        obs, info = env.reset()

        session = {
            "env": env,
            "task_id": task_id,
            "steps": 0,
            "max_steps": max_steps,
            "done": False,
            "history": [],
            "last_observation": obs,
        }
        _sessions[task_id] = session

        return env, session

    except ImportError:
        logger.error("WebShop not installed. Run: pip install webshop")
        raise
    except Exception as e:
        logger.error(f"Failed to create WebShop session: {e}")
        raise


def _format_observation(obs: str, step: int, max_steps: int) -> str:
    """Format observation for the agent.

    Args:
        obs: Raw observation text.
        step: Current step number.
        max_steps: Maximum allowed steps.

    Returns:
        Formatted observation string.
    """
    # Clean up observation
    obs = obs.strip()

    # Add step counter
    header = f"[Step {step}/{max_steps}]\n"

    return header + obs


@function_tool
def webshop_search(query: str, task_id: str) -> str:
    """Search for products in the WebShop environment.

    Use this tool to search for products matching the user's requirements.
    The search results will show available products with their details.

    Args:
        query: Search query string (e.g., "red t-shirt size M").
        task_id: The unique task identifier for this shopping session.

    Returns:
        Search results with product listings.
    """
    env, session = _get_or_create_session(task_id)

    if session["done"]:
        return json.dumps({
            "error": "Session ended. Task already completed or max steps reached.",
            "steps_taken": session["steps"],
        })

    try:
        # Execute search action
        action = f"search[{query}]"
        obs, reward, done, info = env.step(action)

        session["steps"] += 1
        session["done"] = done or session["steps"] >= session["max_steps"]
        session["history"].append({"action": action, "observation": obs})
        session["last_observation"] = obs

        result = {
            "observation": _format_observation(obs, session["steps"], session["max_steps"]),
            "reward": reward,
            "done": session["done"],
            "steps_remaining": session["max_steps"] - session["steps"],
        }

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"WebShop search error: {e}")
        return json.dumps({"error": str(e)})


@function_tool
def webshop_click(element: str, task_id: str) -> str:
    """Click on an element in the WebShop environment.

    Use this tool to click on products, buttons, or navigation elements.
    Common elements include product numbers (e.g., "1", "2") and
    buttons like "Buy Now", "Next", "Back".

    Args:
        element: The element to click (e.g., "1", "Buy Now", "Next").
        task_id: The unique task identifier for this shopping session.

    Returns:
        Updated page content after clicking.
    """
    env, session = _get_or_create_session(task_id)

    if session["done"]:
        return json.dumps({
            "error": "Session ended. Task already completed or max steps reached.",
            "steps_taken": session["steps"],
        })

    try:
        # Execute click action
        action = f"click[{element}]"
        obs, reward, done, info = env.step(action)

        session["steps"] += 1
        session["done"] = done or session["steps"] >= session["max_steps"]
        session["history"].append({"action": action, "observation": obs})
        session["last_observation"] = obs

        result = {
            "observation": _format_observation(obs, session["steps"], session["max_steps"]),
            "reward": reward,
            "done": session["done"],
            "steps_remaining": session["max_steps"] - session["steps"],
        }

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"WebShop click error: {e}")
        return json.dumps({"error": str(e)})


@function_tool
def webshop_buy(task_id: str) -> str:
    """Purchase the currently viewed product in WebShop.

    Use this tool when you've found a product that matches the user's
    requirements. This will complete the shopping task.

    Args:
        task_id: The unique task identifier for this shopping session.

    Returns:
        Purchase confirmation and final reward.
    """
    env, session = _get_or_create_session(task_id)

    if session["done"]:
        return json.dumps({
            "error": "Session ended. Task already completed or max steps reached.",
            "steps_taken": session["steps"],
        })

    try:
        # Execute buy action
        action = "buy"
        obs, reward, done, info = env.step(action)

        session["steps"] += 1
        session["done"] = True  # Buying ends the episode
        session["history"].append({"action": action, "observation": obs, "reward": reward})

        result = {
            "observation": _format_observation(obs, session["steps"], session["max_steps"]),
            "reward": reward,
            "done": True,
            "purchase_successful": info.get("purchase_successful", False),
            "final_reward": reward,
            "total_steps": session["steps"],
        }

        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"WebShop buy error: {e}")
        return json.dumps({"error": str(e)})


@function_tool
def webshop_get_status(task_id: str) -> str:
    """Get the current status of a WebShop shopping session.

    Use this tool to check your progress, remaining steps, and
    current page information.

    Args:
        task_id: The unique task identifier for this shopping session.

    Returns:
        Current session status and information.
    """
    if task_id not in _sessions:
        return json.dumps({
            "error": "No active session found for this task_id.",
            "suggestion": "Start a new session with webshop_search or webshop_click.",
        })

    session = _sessions[task_id]

    result = {
        "task_id": task_id,
        "steps_taken": session["steps"],
        "max_steps": session["max_steps"],
        "steps_remaining": session["max_steps"] - session["steps"],
        "done": session["done"],
        "history_length": len(session["history"]),
        "last_observation_preview": session["last_observation"][:200] + "..." if len(session["last_observation"]) > 200 else session["last_observation"],
    }

    return json.dumps(result, ensure_ascii=False)


@function_tool
def webshop_get_history(task_id: str) -> str:
    """Get the action history for a WebShop shopping session.

    Use this tool to review what actions you've taken so far,
    which can help with planning next steps.

    Args:
        task_id: The unique task identifier for this shopping session.

    Returns:
        List of actions taken in this session.
    """
    if task_id not in _sessions:
        return json.dumps({
            "error": "No active session found for this task_id.",
        })

    session = _sessions[task_id]

    result = {
        "task_id": task_id,
        "total_actions": len(session["history"]),
        "history": session["history"],
    }

    return json.dumps(result, ensure_ascii=False)


# Cleanup function for sessions
def cleanup_session(task_id: str) -> None:
    """Clean up a WebShop session.

    Args:
        task_id: The task ID to clean up.
    """
    if task_id in _sessions:
        try:
            session = _sessions[task_id]
            if "env" in session:
                session["env"].close()
        except Exception as e:
            logger.warning(f"Error cleaning up session {task_id}: {e}")
        finally:
            del _sessions[task_id]


def cleanup_all_sessions() -> None:
    """Clean up all WebShop sessions."""
    for task_id in list(_sessions.keys()):
        cleanup_session(task_id)
