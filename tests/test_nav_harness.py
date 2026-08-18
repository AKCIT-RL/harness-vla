import json

import numpy as np

from embodiedbench.evaluator.eb_navigation_harness_evaluator import (
    EB_NavigationHarnessEvaluator,
)
from harness_vla.planner.navigation_planner import (
    NAV_ACTION_TO_INDEX,
    clamp_navigation_steps,
    parse_navigation_invocation,
)


def test_parse_clean_navigation_json():
    assert parse_navigation_invocation('{"action": "move_forward", "steps": 3}') == {
        "action": "move_forward",
        "steps": 3,
    }


def test_parse_navigation_json_with_surrounding_text():
    assert parse_navigation_invocation(
        'I will reorient. {"action": "turn_left", "steps": 5} Done.'
    ) == {"action": "turn_left", "steps": 1}


def test_unknown_navigation_action_is_rejected():
    assert parse_navigation_invocation('{"action": "jump", "steps": 2}') is None


def test_out_of_range_steps_are_clamped():
    assert parse_navigation_invocation(
        '{"action": "move_backward", "steps": 99}'
    ) == {"action": "move_backward", "steps": 5}
    assert parse_navigation_invocation(
        '{"action": "move_right", "steps": -4}'
    ) == {"action": "move_right", "steps": 1}


def test_navigation_action_mapping_is_complete_and_valid():
    assert NAV_ACTION_TO_INDEX == {
        "move_forward": 0,
        "move_backward": 1,
        "move_right": 2,
        "move_left": 3,
        "turn_right": 4,
        "turn_left": 5,
        "tilt_up": 6,
        "tilt_down": 7,
    }
    assert set(NAV_ACTION_TO_INDEX.values()) == set(range(8))


def test_clamp_steps_for_moves_and_single_step_actions():
    assert clamp_navigation_steps("move_left", 0) == 1
    assert clamp_navigation_steps("move_left", 4) == 4
    assert clamp_navigation_steps("move_left", 8) == 5
    assert clamp_navigation_steps("tilt_down", 4) == 1


class _FakeNavEnv:
    selected_indexes = [4]
    number_of_episodes = 1
    _current_episode_num = 0
    _current_step = 0
    episode_language_instruction = "Navigate to the mug."

    def reset(self):
        self._current_episode_num = 1
        self._current_step = 0
        return {"head_rgb": np.zeros((8, 8, 3), dtype=np.uint8)}

    def step(self, action, reasoning, final_step):
        self._current_step += 1
        assert action == 0
        assert reasoning == {"action": "move_forward", "steps": 1}
        return (
            {"head_rgb": np.ones((8, 8, 3), dtype=np.uint8)},
            1.0,
            True,
            {
                "task_success": 1,
                "distance": 0.5,
                "last_action_success": True,
                "env_step": 1,
            },
        )

    def close(self):
        pass


class _FakeNavPlanner:
    last_thinking = None

    def reset(self):
        pass

    def act(self, instruction, feedback, history):
        return {"action": "move_forward", "steps": 1}, '{"action":"move_forward"}'


def test_navigation_evaluator_writes_metrics_trace_and_frames(tmp_path):
    evaluator = EB_NavigationHarnessEvaluator(
        {"model_name": "fake", "max_turns": 2, "max_env_steps": 2}
    )
    evaluator.env = _FakeNavEnv()
    evaluator.planner = _FakeNavPlanner()
    evaluator.log_path = str(tmp_path / "base")

    evaluator.evaluate()

    summary = json.loads(
        (tmp_path / "base" / "results" / "summary.json").read_text()
    )
    assert summary["successes"] == 1
    assert summary["episodes"][0]["episode"] == 5
    assert (tmp_path / "base" / "results" / "trace_episode_5.jsonl").exists()
    assert (
        tmp_path
        / "base"
        / "images"
        / "episode_5"
        / "episode_5_step_1_front_rgb.png"
    ).exists()