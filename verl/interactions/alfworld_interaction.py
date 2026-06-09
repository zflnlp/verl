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
Interaction class for ALFWorld benchmark.

ALFWorld is a text-game environment that aligns with the ALFRED benchmark
for embodied instruction following. Agents complete household tasks such as
picking, placing, cleaning, heating, and cooling objects.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _extract_action(text: str) -> str:
    """Extract action from <action> tags or use raw text."""
    match = re.search(r'<action>\s*(.*?)\s*</action>', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


# Short name → (internal name, integer ID) mapping
TASK_TYPE_MAP = {
    "pick_and_place":           ("pick_and_place_simple", 1),
    "look_at_obj_in_light":     ("look_at_obj_in_light", 2),
    "pick_clean_then_place":    ("pick_clean_then_place_in_recep", 3),
    "pick_heat_then_place":     ("pick_heat_then_place_in_recep", 4),
    "pick_cool_then_place":     ("pick_cool_then_place_in_recep", 5),
    "pick_two_obj":             ("pick_two_obj_and_place", 6),
}


def _build_alfworld_config(game_files_dir: str, task_type_ids: list, train_eval: str = "eval_out_of_distribution") -> dict:
    """Build an ALFWorld config dict programmatically.

    Args:
        game_files_dir: Path to ALFWORLD_DATA directory (contains json_2.1.1/).
        task_type_ids: List of integer task type IDs (1-6).
        train_eval: One of 'train', 'eval_in_distribution', 'eval_out_of_distribution'.

    Returns:
        Config dict compatible with AlfredTWEnv.__init__.
    """
    data_root = game_files_dir.rstrip("/")
    data_path_key = {
        "train": "data_path",
        "eval_in_distribution": "eval_id_data_path",
        "eval_out_of_distribution": "eval_ood_data_path",
    }.get(train_eval, "eval_ood_data_path")

    config = {
        "dataset": {
            "data_path": f"{data_root}/json_2.1.1/train",
            "eval_id_data_path": f"{data_root}/json_2.1.1/valid_seen",
            "eval_ood_data_path": f"{data_root}/json_2.1.1/valid_unseen",
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {
            "domain": f"{data_root}/logic/alfred.pddl",
            "grammar": f"{data_root}/logic/alfred.twl2",
        },
        "env": {
            "type": "AlfredTWEnv",
            "domain_randomization": False,
            "task_types": task_type_ids,
            "expert_timeout_steps": 150,
            "expert_type": "handcoded",
            "goal_desc_human_anns_prob": 0.0,
        },
        "general": {
            "random_seed": 42,
            "use_cuda": True,
            "task": "alfred",
            "training_method": "dagger",
        },
        "dagger": {
            "training": {
                "max_nb_steps_per_episode": 50,
            },
        },
    }
    return config


class AlfworldInteraction(BaseInteraction):
    """Interaction class for ALFWorld benchmark.

    This class manages multi-turn interactions with the ALFWorld environment,
    where the agent needs to complete household tasks by manipulating objects.

    Supports two modes:
    - Mock mode: Simulated responses for pipeline testing.
    - Real mode: Direct ALFWorld Python API integration.

    Config options:
        use_mock: bool = True
        max_steps: int = 30
        game_files_dir: str — Path to ALFWORLD_DATA directory (contains json_2.1.1/)
        config_path: str — Path to ALFWorld base_config.yaml (optional, overrides game_files_dir)
        task_types: list[str] — Short task type names, e.g. ["pick_and_place", "pick_clean_then_place"]
        train_eval: str = "eval_out_of_distribution"
        num_games: int = -1 — Max number of games to load (-1 = all)
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self.use_mock = config.get("use_mock", True)
        self.max_steps = config.get("max_steps", 30)
        # Real mode config
        self.config_path = config.get("config_path", "")
        self.game_files_dir = config.get("game_files_dir", "")
        self.task_types = config.get("task_types", list(TASK_TYPE_MAP.keys()))
        self.train_eval = config.get("train_eval", "eval_out_of_distribution")
        self.num_games = config.get("num_games", -1)

    def _load_alfworld_config(self) -> dict:
        """Load or build the ALFWorld config dict."""
        if self.config_path:
            import yaml
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
            return config

        # Build config from game_files_dir
        task_type_ids = []
        for short_name in self.task_types:
            if short_name in TASK_TYPE_MAP:
                _, tid = TASK_TYPE_MAP[short_name]
                task_type_ids.append(tid)
            else:
                logger.warning(f"Unknown task type: {short_name}, skipping")

        if not task_type_ids:
            task_type_ids = [1, 2, 3, 4, 5, 6]

        return _build_alfworld_config(self.game_files_dir, task_type_ids, self.train_eval)

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[dict] = None,
        **kwargs
    ) -> str:
        """Start a new ALFWorld interaction.

        Args:
            instance_id: Optional unique identifier for this interaction.
            ground_truth: Dictionary containing task information:
                - task_type: ALFWorld task type (e.g. "pick_and_place")
                - goal: Task description
                - game_file: Path to .tw-pddl game file (optional, for specific game)

        Returns:
            The instance ID for this interaction.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        gt = ground_truth or {}
        instance = {
            "ground_truth": gt,
            "steps": [],
            "current_observation": "",
            "reward": 0.0,
            "is_done": False,
            "won": False,
            "num_steps": 0,
            "task_type": gt.get("task_type", "unknown"),
            "env": None,
            "admissible_actions": [],
        }
        self._instance_dict[instance_id] = instance

        if self.use_mock:
            instance["current_observation"] = self._get_mock_initial_observation(gt)
            instance["admissible_actions"] = self._get_mock_admissible_actions()
        else:
            try:
                from alfworld.agents.environment import get_environment

                alfworld_config = self._load_alfworld_config()
                env_type = alfworld_config["env"]["type"]
                alfred_env = get_environment(env_type)(alfworld_config, train_eval=self.train_eval)

                # If a specific game file is requested, restrict game_files before init_env
                game_file = gt.get("game_file", "")
                if game_file and os.path.exists(game_file):
                    alfred_env.game_files = [game_file]
                    alfred_env.num_games = 1

                # Limit number of games if specified
                if self.num_games > 0 and len(alfred_env.game_files) > self.num_games:
                    alfred_env.game_files = alfred_env.game_files[:self.num_games]
                    alfred_env.num_games = self.num_games

                env = alfred_env.init_env(batch_size=1)
                obs, info = env.reset()

                instance["env"] = env
                instance["current_observation"] = obs[0] if isinstance(obs, list) else obs
                instance["admissible_actions"] = info["admissible_commands"][0] \
                    if isinstance(info.get("admissible_commands"), list) else []
            except Exception as e:
                logger.error(f"Failed to initialize ALFWorld env: {e}")
                instance["current_observation"] = f"Error initializing environment: {e}"

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
            raw_observation, reward, is_done = self._process_real_action(action, instance)

        # Format observation for next turn
        observation = self._format_observation(instance, raw_observation)

        # Update instance state
        instance["steps"].append({"action": action, "observation": raw_observation})
        instance["current_observation"] = raw_observation
        instance["reward"] = reward

        # Check termination conditions
        should_terminate = is_done or instance["num_steps"] >= self.max_steps

        if should_terminate:
            instance["is_done"] = True
            final_reward = await self.calculate_score(instance_id)
            instance["reward"] = final_reward
            return True, observation, final_reward, {"num_steps": instance["num_steps"]}
        else:
            return False, observation, 0.0, {"num_steps": instance["num_steps"]}

    def _process_real_action(self, action: str, instance: dict) -> Tuple[str, float, bool]:
        """Process an action using the real ALFWorld environment.

        Args:
            action: The agent's action string (may contain <action> tags).
            instance: The interaction instance state.

        Returns:
            Tuple of (observation, reward, is_done).
        """
        env = instance.get("env")
        if env is None:
            return "Error: Environment not initialized", 0.0, True

        clean_action = _extract_action(action)

        try:
            obs, scores, dones, infos = env.step([clean_action])
            observation = obs[0] if isinstance(obs, list) else obs

            # ALFWorld: scores are won values (1.0 or 0.0)
            won = infos["won"][0] if isinstance(infos.get("won"), list) else infos.get("won", False)
            instance["won"] = won

            # Update admissible actions for next turn
            admissible = infos["admissible_commands"][0] \
                if isinstance(infos.get("admissible_commands"), list) else []
            instance["admissible_actions"] = admissible

            # Binary reward from environment
            reward_val = 1.0 if won else 0.0
            is_done = dones[0] if isinstance(dones, list) else dones

            return observation, reward_val, is_done
        except Exception as e:
            logger.error(f"ALFWorld step error: {e}")
            return f"Error executing action: {e}", 0.0, False

    def _format_observation(self, instance: dict, raw_observation: str) -> str:
        """Format observation with context for the next turn.

        Args:
            instance: The interaction instance state.
            raw_observation: The raw observation from the environment.

        Returns:
            Formatted observation string.
        """
        gt = instance.get("ground_truth", {})
        goal = gt.get("goal", "Complete the household task")
        task_type = instance.get("task_type", "unknown")
        step_count = instance.get("num_steps", 0)
        steps = instance.get("steps", [])

        # Build action history (last 3 steps)
        history_length = min(3, len(steps))
        history_lines = []
        for i, step in enumerate(steps[-history_length:]):
            step_action = _extract_action(step.get("action", ""))
            step_obs = step.get("observation", "")
            history_lines.append(f"Step {step_count - history_length + i + 1}: Action: {step_action}")
            history_lines.append(f"Observation: {step_obs[:300]}")

        action_history = "\n".join(history_lines) if history_lines else "(no history)"

        # Format admissible actions
        admissible_actions = instance.get("admissible_actions", [])
        if admissible_actions:
            available_actions = "\n".join(f"- {a}" for a in admissible_actions[:30])
        else:
            available_actions = "- look\n- inventory\n- go to <location>\- take <object>"

        return f"""You are a household robot agent performing tasks in a simulated home environment.
Your task is: {goal}
Task type: {task_type}

Prior to this step, you have already taken {step_count - 1} step(s).
Below are the most recent {history_length} observations and the corresponding actions you took:
{action_history}

You are now at step {step_count} and your current observation is:
{raw_observation}

Your admissible actions of the current situation are:
{available_actions}

Now it's your turn to take one action for the current step. You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the household task. This reasoning process MUST be enclosed within <thought> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags."""

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

        # Use environment reward if available
        if not self.use_mock:
            if instance.get("won", False):
                return 1.0
            # Check if env reported a score
            if instance["reward"] > 0:
                return instance["reward"]
            return 0.0

        # Use accumulated reward
        if instance["reward"] > 0:
            return instance["reward"]

        # Fallback: estimate from trajectory
        return self._estimate_reward(instance)

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Finalize the interaction and release resources.

        Args:
            instance_id: The interaction instance ID.
        """
        if instance_id in self._instance_dict:
            instance = self._instance_dict[instance_id]
            env = instance.get("env")
            if env is not None:
                try:
                    del env
                except Exception:
                    pass
            del self._instance_dict[instance_id]

    # ── Mock helpers ──────────────────────────────────────────────────

    def _get_mock_initial_observation(self, ground_truth: dict) -> str:
        """Generate a mock initial observation for testing."""
        goal = ground_truth.get("goal", "Complete the household task")
        task_type = ground_truth.get("task_type", "unknown")

        # Different scenes based on task type
        scene_descriptions = {
            "pick_and_place": (
                "You are in the middle of a room. Looking quickly around you, you see "
                "a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, "
                "a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, "
                "a diningtable 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, "
                "a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, "
                "a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, "
                "a stoveburner 1, and a toaster 1."
            ),
            "pick_clean_then_place": (
                "You are in the middle of a room. Looking quickly around you, you see "
                "a bathtubbasin 1, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, "
                "a countertop 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, "
                "a garbagecan 1, a handtowelholder 1, a sinkbasin 1, a toilet 1, "
                "a toiletpaperhanger 1, and a towelholder 1."
            ),
            "pick_heat_then_place": (
                "You are in the middle of a room. Looking quickly around you, you see "
                "a cabinet 13, a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, "
                "a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, "
                "a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, "
                "a countertop 2, a countertop 1, a diningtable 1, a drawer 1, "
                "a fridge 1, a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, "
                "a shelf 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, "
                "a stoveburner 2, a stoveburner 1, and a toaster 1."
            ),
            "pick_cool_then_place": (
                "You are in the middle of a room. Looking quickly around you, you see "
                "a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, "
                "a fridge 1, a garbagecan 1, a microwave 1, a shelf 1, "
                "a sinkbasin 1, a stoveburner 2, a stoveburner 1, and a toaster 1."
            ),
            "look_at_obj_in_light": (
                "You are in the middle of a room. Looking quickly around you, you see "
                "a bed 1, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, "
                "a desk 1, a drawer 2, a drawer 1, a garbagecan 1, a shelf 1, "
                "a sidetable 2, a sidetable 1, and a desklamp 1."
            ),
            "pick_two_obj": (
                "You are in the middle of a room. Looking quickly around you, you see "
                "a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, "
                "a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, "
                "a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, "
                "a diningtable 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, "
                "a garbagecan 1, a microwave 1, a shelf 3, a shelf 2, a shelf 1, "
                "a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, "
                "a stoveburner 1, and a toaster 1."
            ),
        }

        scene = scene_descriptions.get(task_type, scene_descriptions["pick_and_place"])
        return f"-= {task_type.replace('_', ' ').title()} =-\nYou are in the middle of a home.\n\n{scene}\n\nYour task is: {goal}"

    def _get_mock_admissible_actions(self) -> list:
        """Return mock admissible actions."""
        return [
            "look",
            "inventory",
            "go to cabinet 1",
            "go to countertop 1",
            "go to fridge 1",
            "go to sinkbasin 1",
            "go to diningtable 1",
            "go to stoveburner 1",
            "go to microwave 1",
            "go to garbagecan 1",
            "take apple from countertop 1",
            "take cup from cabinet 1",
            "take knife from drawer 1",
            "take mug from shelf 1",
            "take plate from diningtable 1",
            "take sponge from sinkbasin 1",
            "examine cabinet 1",
            "examine countertop 1",
            "examine fridge 1",
        ]

    def _process_mock_action(self, action: str, instance: dict) -> Tuple[str, float, bool]:
        """Process an action in mock mode."""
        clean_action = _extract_action(action).lower()

        if "look" in clean_action and "at" not in clean_action:
            obs = instance.get("current_observation", "You are in a room with various household items.")
        elif "inventory" in clean_action:
            obs = "You are carrying: nothing."
        elif "go to" in clean_action:
            loc = clean_action.replace("go to", "").strip()
            obs = f"You arrive at {loc}. On the {loc} you see nothing special."
        elif "take" in clean_action:
            obs = "You pick up the object."
        elif "put" in clean_action or "move" in clean_action:
            obs = "You put the object down."
        elif "open" in clean_action:
            obs = "You open the container. Inside, you see nothing."
        elif "close" in clean_action:
            obs = "You close the container."
        elif "toggle" in clean_action:
            obs = "You toggle the object."
        elif "use" in clean_action:
            obs = "You use the object."
        elif "heat" in clean_action:
            obs = "You heat the object in the microwave."
        elif "clean" in clean_action:
            obs = "You clean the object in the sink."
        elif "cool" in clean_action:
            obs = "You cool the object in the fridge."
        elif "examine" in clean_action:
            obs = "You examine the object. It looks ordinary."
        elif "slice" in clean_action:
            obs = "You slice the object with the knife."
        else:
            obs = "You perform the action. The environment responds accordingly."

        # Mock scoring: reward after enough diverse meaningful actions
        action_set = set()
        for step in instance.get("steps", []):
            sa = _extract_action(step.get("action", "")).lower()
            for kw in ["go to", "take", "put", "move", "open", "close", "toggle", "use", "heat", "clean", "cool", "examine", "slice"]:
                if kw in sa:
                    action_set.add(kw)

        # Simulate task completion when enough diverse actions are taken
        task_type = instance.get("task_type", "unknown")
        required_actions = self._get_required_actions_for_task(task_type)
        completed_actions = action_set.intersection(required_actions)

        if len(completed_actions) >= len(required_actions):
            return obs + "\n\nYou have completed the task!", 1.0, True
        elif len(completed_actions) >= max(1, len(required_actions) - 1):
            return obs + "\n\nYou are close to completing the task.", 0.3, False

        return obs, 0.0, False

    def _get_required_actions_for_task(self, task_type: str) -> set:
        """Return the mock required action keywords for each task type."""
        task_actions = {
            "pick_and_place": {"take", "put"},
            "pick_clean_then_place": {"take", "clean", "put"},
            "pick_heat_then_place": {"take", "heat", "put"},
            "pick_cool_then_place": {"take", "cool", "put"},
            "look_at_obj_in_light": {"take", "use", "examine"},
            "pick_two_obj": {"take", "put"},
        }
        return task_actions.get(task_type, {"take", "put"})

    def _estimate_reward(self, instance: dict) -> float:
        """Estimate reward from trajectory when environment reward is unavailable."""
        reward = 0.0
        steps = instance.get("steps", [])
        task_type = instance.get("task_type", "unknown")

        # Check action diversity
        action_set = set()
        for step in steps:
            a = _extract_action(step.get("action", "")).lower()
            for kw in ["go to", "take", "put", "move", "open", "close", "toggle", "use", "heat", "clean", "cool", "examine", "look"]:
                if kw in a:
                    action_set.add(kw)

        reward += min(0.4, len(action_set) * 0.05)

        # Check task-type-specific actions
        required = self._get_required_actions_for_task(task_type)
        completed = action_set.intersection(required)
        if len(completed) >= len(required):
            reward += 0.4
        elif len(completed) > 0:
            reward += 0.2 * len(completed) / len(required)

        # Bonus for reasonable number of steps
        if 3 <= len(steps) <= 20:
            reward += 0.1

        # Bonus for reasoning
        thought_count = sum(
            1 for step in steps
            if "<thought>" in step.get("action", "").lower()
        )
        if thought_count >= 3:
            reward += 0.1

        return min(1.0, reward)
