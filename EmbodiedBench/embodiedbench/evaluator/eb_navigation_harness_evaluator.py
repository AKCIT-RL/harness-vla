"""Beta-only EB-Navigation Harness evaluator, an extension outside the paper."""

import json
import os

from PIL import Image
from tqdm import tqdm

from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv, ValidEvalSets
from embodiedbench.main import logger
from harness_vla.memory.global_memory import GlobalMemory
from harness_vla.planner.navigation_planner import (
    NAV_ACTION_TO_INDEX,
    NavigationHarnessPlanner,
)
from harness_vla.trace_io import (
    append_jsonl_record,
    initialize_jsonl,
    write_json_atomic,
)


class EB_NavigationHarnessEvaluator:
    """Run closed-loop LLM planning over the fixed EB-Navigation vocabulary."""

    def __init__(self, config):
        self.config = config
        self.model_name = config["model_name"]
        self.max_turns = int(config.get("max_turns", 12))
        self.max_env_steps = int(config.get("max_env_steps", 30))
        self.eval_set = None
        self.env = None
        self.planner = None
        self.log_path = None

    def _results_dir(self):
        path = os.path.join(self.log_path, "results")
        os.makedirs(path, exist_ok=True)
        return path

    def _episode_id(self):
        if self.env.selected_indexes:
            return self.env.selected_indexes[self.env._current_episode_num - 1] + 1
        return self.env._current_episode_num

    def _agent_position(self, env_feedback=None):
        if isinstance(env_feedback, dict):
            position = env_feedback.get("agent", {}).get("position")
            if position:
                return position
        event = getattr(getattr(self.env, "env", None), "last_event", None)
        metadata = getattr(event, "metadata", {}) if event is not None else {}
        return metadata.get("agent", {}).get("position", {})

    def _feedback(self, info=None, format_error=None):
        info = info or {}
        env_feedback = info.get("env_feedback")
        feedback = {
            "distance": info.get("distance"),
            "agent_position": self._agent_position(env_feedback),
            "env_feedback": env_feedback,
            "last_action_success": info.get("last_action_success"),
            "env_step": info.get("env_step", getattr(self.env, "_current_step", 0)),
        }
        if format_error:
            feedback["format_error"] = format_error
        return feedback

    def _save_frame(self, obs):
        frame = obs.get("head_rgb") if isinstance(obs, dict) else None
        if frame is None:
            return None
        episode_id = self._episode_id()
        image_dir = os.path.join(
            self.log_path, "images", f"episode_{episode_id}"
        )
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(
            image_dir,
            f"episode_{episode_id}_step_{self.env._current_step}_front_rgb.png",
        )
        Image.fromarray(frame).save(image_path)
        return image_path

    def _write_run_summary(self, episodes):
        successes = sum(int(item["success"]) for item in episodes)
        payload = {
            "episodes": episodes,
            "total_episodes": len(episodes),
            "successes": successes,
            "success_rate": successes / len(episodes) if episodes else 0.0,
        }
        write_json_atomic(os.path.join(self._results_dir(), "summary.json"), payload)

    def evaluate(self):
        summaries = []
        progress = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            obs = self.env.reset()
            instruction = self.env.episode_language_instruction
            self.planner.reset()
            episode_id = self._episode_id()
            self._save_frame(obs)
            trace_path = os.path.join(
                self._results_dir(), f"trace_episode_{episode_id}.jsonl"
            )
            initialize_jsonl(trace_path)
            history = []
            feedback = self._feedback()
            done = False
            success = 0
            parse_errors = 0
            minimum_distance = None
            final_distance = None
            turns = 0

            while (
                not done
                and turns < self.max_turns
                and self.env._current_step < self.max_env_steps
            ):
                turns += 1
                invocation, raw_output = self.planner.act(
                    instruction, feedback, history
                )
                record = {
                    "turn": turns,
                    "invocation": invocation,
                    "status": "ok",
                    "feedback": None,
                    "raw_output": raw_output,
                    "thinking": self.planner.last_thinking,
                }
                if invocation is None:
                    parse_errors += 1
                    record["status"] = "parse_error"
                    feedback = self._feedback(
                        format_error=(
                            "Return one JSON object with a known action and optional "
                            "integer steps from 1 to 5."
                        )
                    )
                    record["feedback"] = feedback
                    history.append({"turn": turns, "status": "parse_error"})
                    append_jsonl_record(trace_path, record)
                    continue

                action_index = NAV_ACTION_TO_INDEX[invocation["action"]]
                turn_failed = False
                step_feedback = []
                for _ in range(invocation["steps"]):
                    if done or self.env._current_step >= self.max_env_steps:
                        break
                    final_budget_step = self.env._current_step + 1 >= self.max_env_steps
                    obs, _, done, info = self.env.step(
                        action_index, invocation, int(final_budget_step)
                    )
                    self._save_frame(obs)
                    success = max(success, int(info.get("task_success", 0)))
                    final_distance = info.get("distance")
                    if final_distance is not None:
                        minimum_distance = (
                            final_distance
                            if minimum_distance is None
                            else min(minimum_distance, final_distance)
                        )
                    feedback = self._feedback(info)
                    step_feedback.append(feedback)
                    if not info.get("last_action_success", True):
                        turn_failed = True
                        break

                if turn_failed:
                    record["status"] = "action_failed"
                record["feedback"] = step_feedback
                history.append(
                    {
                        "turn": turns,
                        "invocation": invocation,
                        "status": record["status"],
                        "feedback": feedback,
                    }
                )
                append_jsonl_record(trace_path, record)

            summary = {
                "episode": episode_id,
                "instruction": instruction,
                "success": success,
                "env_steps": int(self.env._current_step),
                "turns": turns,
                "parse_errors": parse_errors,
                "final_distance": final_distance,
                "minimum_distance": minimum_distance,
            }
            summaries.append(summary)
            write_json_atomic(
                os.path.join(self._results_dir(), f"episode_{episode_id}_res.json"),
                summary,
            )
            self._write_run_summary(summaries)
            progress.update()
        progress.close()
        self.env.close()

    def evaluate_main(self):
        eval_sets = list(self.config.get("eval_sets") or ValidEvalSets)
        for eval_set in eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f"Current eval set: {eval_set}")
            run_root = self.config.get("run_root")
            if run_root:
                self.log_path = os.path.join(run_root, eval_set)
            else:
                output_root = self.config.get("output_root", "running/eb_navigation_harness")
                self.log_path = os.path.join(
                    output_root, self.config.get("exp_name", "harness_nav"), eval_set
                )
            os.makedirs(self.log_path, exist_ok=True)
            self.env = EBNavigationEnv(
                eval_set=eval_set,
                exp_name=self.config.get("exp_name", "harness_nav"),
                down_sample_ratio=self.config.get("down_sample_ratio", 1.0),
                resolution=self.config.get("resolution", 300),
                selected_indexes=list(self.config.get("selected_indexes", []) or []),
            )
            self.env.log_path = self.log_path
            self.env._max_episode_steps = self.max_env_steps
            self.planner = NavigationHarnessPlanner(
                model_name=self.model_name,
                base_url=self.config.get("base_url"),
                api_key=self.config.get("api_key"),
                global_memory=GlobalMemory.load(self.config["global_memory_path"]),
                temperature=self.config.get("temperature", 0.0),
                max_tokens=self.config.get("max_tokens", 1024),
                num_ctx=self.config.get("num_ctx"),
                disable_thinking=self.config.get("disable_thinking", False),
                enable_thinking=self.config.get("enable_thinking", False),
                request_timeout=self.config.get("request_timeout", 600.0),
            )
            try:
                self.evaluate()
            finally:
                with open(os.path.join(self.log_path, "config.txt"), "w") as config_file:
                    config_file.write(str(self.config))

    def check_config_valid(self):
        if self.max_turns <= 0 or self.max_env_steps <= 0:
            raise ValueError("max_turns and max_env_steps must be positive")