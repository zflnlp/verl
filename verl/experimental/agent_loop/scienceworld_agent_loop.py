# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""
ScienceWorld agent loop for multi-turn interaction with ScienceWorld environment.

This agent loop handles text-based interaction with the ScienceWorld environment,
using <action> tags (not OpenAI function calling). It preserves the prompt format
from the TCOD paper.
"""

import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _extract_action(text: str) -> str:
    """Extract action from <action> tags or use raw text."""
    match = re.search(r"<action>\s*(.*?)\s*</action>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


@register("scienceworld_agent")
class ScienceWorldAgentLoop(AgentLoopBase):
    """Agent loop for ScienceWorld multi-turn interaction.

    This agent loop:
    1. Initializes the ScienceWorld environment
    2. Generates model responses to interact with the environment
    3. Extracts actions from <action> tags
    4. Executes actions and returns observations
    5. Calculates cumulative process rewards
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ScienceWorld configuration
        sw_config = self.config.actor_rollout_ref.rollout.get("scienceworld", {})
        self.max_steps = sw_config.get("max_steps", 30)
        self.simplifications_preset = sw_config.get("simplifications_preset", "easy")
        self.use_mock = sw_config.get("use_mock", False)

    async def run(
        self,
        sampling_params: dict[str, Any],
        **kwargs,
    ) -> AgentLoopOutput:
        """Run the ScienceWorld agent loop.

        Args:
            sampling_params: LLM sampling parameters.
            **kwargs: Includes messages, ground_truth, etc.

        Returns:
            AgentLoopOutput with the full interaction history and reward.
        """
        messages = list(kwargs.get("messages", []))
        ground_truth = kwargs.get("ground_truth", {})
        image_data = kwargs.get("image_data", [])
        video_data = kwargs.get("video_data", [])
        audio_data = kwargs.get("audio_data", [])
        mm_processor_kwargs = kwargs.get("mm_processor_kwargs", {})

        task_name = ground_truth.get("task_name", "boil")
        variation = ground_truth.get("variation", 0)

        # Initialize ScienceWorld environment
        env = None
        cumulative_reward = 0.0
        last_score = 0.0
        metrics = {}

        try:
            from scienceworld import ScienceWorldEnv

            env = ScienceWorldEnv()
            env.load(task_name, variation, simplificationStr=self.simplifications_preset)
        except Exception as e:
            logger.error(f"Failed to initialize ScienceWorld: {e}")
            # Return minimal output on error
            return AgentLoopOutput(
                prompt_ids=[],
                response_ids=[],
                response_mask=[],
                reward_score=0.0,
                num_turns=0,
                metrics=self._init_metrics(),
            )

        try:
            possible_actions = env.get_possible_actions() if env else []
            current_obs = env.look() if env else ""

            for step in range(1, self.max_steps + 1):
                # Format the current observation as user message
                task_desc = env.taskdescription() if env else ""
                history_lines = self._build_history(messages)
                action_history = "\n".join(history_lines) if history_lines else "(no history)"

                available_actions = ", ".join(possible_actions[:30]) if possible_actions else "look around, examine <object>, task"
                history_length = min(3, len(history_lines) // 2)

                observation_text = (
                    f"Your ScienceWorld task is: {task_desc}\n"
                    f"Prior to this step, you have already taken {step - 1} step(s).\n"
                    f"Below are the most recent {history_length} observations and the corresponding actions you took:\n"
                    f"{action_history}\n"
                    f"You are now at step {step} and your current observation is:\n"
                    f"{current_obs}\n"
                    f"Your valid actions of the current situation are: [{available_actions}].\n"
                    f"Now it's your turn to take an action. You should first reason step-by-step about the current situation. "
                    f"This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, "
                    f"you should choose a valid action for the current step and present it within <action> </action> tags."
                )

                # Add observation as user message
                messages.append({"role": "user", "content": observation_text})

                # Generate model response
                prompt_ids = await self.apply_chat_template(
                    messages,
                )

                with simple_timer("generate_sequences", metrics):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=str(uuid4()),
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                    )

                response_ids = output.token_ids
                response_mask = [1] * len(response_ids)

                # Decode the response
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                # Extract action from <action> tags
                action = _extract_action(response_text)

                # Execute action in environment
                if env:
                    obs, score, is_done, info = env.step(action)
                    total_score = info.get("score", score)
                    step_reward = total_score - last_score
                    cumulative_reward += step_reward
                    last_score = total_score

                    # Update state for next turn
                    possible_actions = env.get_possible_actions()
                    current_obs = obs

                    # Add model response as assistant message
                    messages.append({"role": "assistant", "content": response_text})

                    if is_done:
                        break
                else:
                    current_obs = f"Error: Environment not initialized"

        finally:
            if env:
                try:
                    env.close()
                except Exception as e:
                    logger.debug(f"Error closing env: {e}")
                try:
                    del env
                except Exception:
                    pass

        # Calculate final reward (normalized to 0-1)
        reward_score = max(0.0, min(1.0, cumulative_reward / 100.0))

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            reward_score=reward_score,
            num_turns=len([m for m in messages if m["role"] == "assistant"]),
            metrics=self._init_metrics(),
        )

    def _build_history(self, messages: list) -> list:
        """Build action history from assistant messages for prompt formatting."""
        history_lines = []
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        for i, msg in enumerate(assistant_messages[-3:]):
            action = _extract_action(msg.get("content", ""))
            history_lines.append(f"Step {i + 1}: Action: {action}")
            history_lines.append(f"Observation: ...")
        return history_lines

    def _init_metrics(self):
        """Initialize metrics dictionary."""
        return {
            "num_preempted": -1,
        }
