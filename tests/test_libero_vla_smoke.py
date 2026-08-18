import json

import numpy as np

from harness_vla.eval.libero_vla_smoke import (
    LIBERO_DUMMY_ACTION,
    prepare_pirlinf_observation,
    run_libero_vla_smoke,
)
from harness_vla.backends.pirlinf_backend import PiRLinfChunk


def _raw_observation(value=0):
    front = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3) + value
    return {
        "agentview_image": front,
        "robot0_eye_in_hand_image": front + 20,
        "robot0_eef_pos": np.array([1.0, 2.0, 3.0]),
        "robot0_eef_quat": np.array([0.1, 0.2, 0.3, 0.9]),
        "robot0_gripper_qpos": np.array([0.4, 0.5]),
    }


def _resize(image, height, width):
    assert (height, width) == (224, 224)
    return image


def _uint8(image):
    return np.asarray(image, dtype=np.uint8)


class _FakeEnv:
    def __init__(self, success_after=None):
        self.success_after = success_after
        self.actions = []
        self.policy_steps = 0
        self.reset_calls = 0
        self.initial_state = None

    def reset(self):
        self.reset_calls += 1

    def set_init_state(self, initial_state):
        self.initial_state = initial_state
        return _raw_observation()

    def step(self, action):
        self.actions.append(action)
        if action != LIBERO_DUMMY_ACTION:
            self.policy_steps += 1
        return _raw_observation(self.policy_steps), float(self.policy_steps), False, {}

    def check_success(self):
        return self.success_after is not None and self.policy_steps >= self.success_after


class _FakeBackend:
    def __init__(self, actions):
        self.actions = tuple(tuple(action) for action in actions)
        self.calls = 0
        self.prompts = []

    def infer_chunk(self, observation, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        assert observation.mode == "task"
        return PiRLinfChunk(self.actions, len(self.actions), 0.125)


class _FakePlanner:
    model_name = "fake-gemma"

    def __init__(self, invocation, raw_output="raw planner output"):
        self.invocation = invocation
        self.raw_output = raw_output
        self.last_thinking = "planner thinking"
        self.calls = []

    def act(self, instruction, max_chunks_cap, available_targets=None):
        self.calls.append((instruction, max_chunks_cap, list(available_targets or ())))
        return self.invocation, self.raw_output


class _FakeLiftMonitor:
    def __init__(self, env, observation, target):
        self.env = env
        self.target = target

    def evaluate(self, observation):
        return {
            "predicate": "lift_and_grasp",
            "target_instance": self.target,
            "tau_satisfied": self.env.policy_steps >= 2,
            "task_success_evaluated": False,
        }


class _TransientFailureLiftMonitor(_FakeLiftMonitor):
    def evaluate(self, observation):
        if self.env.policy_steps == 1:
            raise ValueError("target temporarily occluded")
        return super().evaluate(observation)


def _video_writer(path, frames):
    assert frames
    path.write_bytes(b"fake mp4")


def _run(tmp_path, env, backend, **overrides):
    run_root = tmp_path / "run"
    run_root.mkdir()
    kwargs = {
        "env": env,
        "backend": backend,
        "initial_state": "fixed-state",
        "prompt": "pick up the black bowl",
        "run_root": run_root,
        "task_suite": "libero_spatial",
        "task_id": 0,
        "initial_state_index": 0,
        "seed": 7,
        "replan_steps": 3,
        "max_chunks": 2,
        "horizon": 6,
        "resize_with_pad": _resize,
        "convert_to_uint8": _uint8,
        "video_writer": _video_writer,
    }
    kwargs.update(overrides)
    return run_root, run_libero_vla_smoke(**kwargs)


def test_preprocessing_rotates_resizes_and_preserves_pose_contract():
    raw = _raw_observation()
    seen = []

    def resize(image, height, width):
        seen.append(image.copy())
        return np.full((height, width, 3), 300.0)

    observation = prepare_pirlinf_observation(
        raw, resize, lambda image: np.clip(image, 0, 255).astype(np.uint8)
    )

    np.testing.assert_array_equal(seen[0], raw["agentview_image"][::-1, ::-1])
    np.testing.assert_array_equal(
        seen[1], raw["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    assert observation.front_rgb.shape == (224, 224, 3)
    assert observation.front_rgb.dtype == np.uint8
    np.testing.assert_allclose(
        observation.gripper_pose, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9]
    )
    np.testing.assert_allclose(observation.gripper_qpos, [0.4, 0.5])
    assert observation.mode == "task"


def test_raw_actions_are_unconverted_and_success_stops_mid_chunk(tmp_path):
    actions = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7],
    ]
    env = _FakeEnv(success_after=2)
    backend = _FakeBackend(actions)

    _, result = _run(tmp_path, env, backend)

    assert env.reset_calls == 1
    assert env.initial_state == "fixed-state"
    assert env.actions[:10] == [LIBERO_DUMMY_ACTION] * 10
    assert env.actions[10:] == actions[:2]
    assert backend.calls == 1
    assert result["episode"]["task_success"] is True
    assert result["episode"]["termination_reason"] == "tau_satisfied"


def test_budget_exhausted_does_not_become_success(tmp_path):
    action = [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]
    env = _FakeEnv()
    backend = _FakeBackend(action)

    _, result = _run(
        tmp_path,
        env,
        backend,
        replan_steps=1,
        max_chunks=2,
        horizon=10,
    )

    assert backend.calls == 2
    assert env.policy_steps == 2
    assert result["episode"]["task_success"] is False
    assert result["episode"]["tau_satisfied"] is False
    assert result["summary"]["budget_exhausted"] is True


def test_artifacts_are_written_without_images_and_video_name_is_unique(tmp_path):
    env = _FakeEnv(success_after=1)
    backend = _FakeBackend([[0.1] * 7])

    run_root, _ = _run(
        tmp_path,
        env,
        backend,
        task_id=4,
        initial_state_index=7,
        replan_steps=1,
    )

    expected = {
        "trace.jsonl",
        "episode.json",
        "summary.json",
        "run_manifest.json",
        "videos",
    }
    assert {path.name for path in run_root.iterdir()} == expected
    video = run_root / "videos" / "task_004_state_007_success.mp4"
    assert video.read_bytes() == b"fake mp4"
    trace = json.loads((run_root / "trace.jsonl").read_text().strip())
    assert trace["prompt"] == "pick up the black bowl"
    assert trace["full_chunk_length"] == 1
    assert trace["executed_actions"] == [[0.1] * 7]
    assert trace["rewards"] == [1.0]
    assert trace["dones"] == [False]
    assert trace["task_successes"] == [True]
    assert trace["tau_satisfied"] is True
    assert "image" not in json.dumps(trace).lower()
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    assert manifest["run_type"] == "vla_only_smoke"
    assert manifest["harness_complete"] is False
    assert manifest["analytic_primitives_available"] is False
    assert manifest["task_memory"] is False
    assert manifest["global_memory"] is False
    assert manifest["perception"] == "text_only_task_instruction"


def test_planner_invocation_controls_vla_prompt_and_chunk_cap(tmp_path):
    env = _FakeEnv()
    backend = _FakeBackend([[0.1] * 7])
    invocation = {
        "action": "vla_act",
        "prompt": "planner-selected prompt",
        "max_chunks": 1,
        "tau": "task_success",
    }
    planner = _FakePlanner(invocation)

    run_root, result = _run(tmp_path, env, backend, planner=planner)

    assert planner.calls == [("pick up the black bowl", 2, [])]
    assert backend.calls == 1
    assert backend.prompts == ["planner-selected prompt"]
    assert result["episode"]["prompt"] == "planner-selected prompt"
    assert result["episode"]["planner_raw_output"] == "raw planner output"
    assert result["episode"]["planner_thinking"] == "planner thinking"
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    assert manifest["run_type"] == "harness_vla_only_smoke"
    assert manifest["harness_complete"] is False
    assert manifest["analytic_primitives_available"] is False
    assert manifest["task_memory"] is False
    assert manifest["global_memory"] is False
    assert manifest["perception"] == "text_only_task_instruction"
    assert manifest["planner_invocation"] == invocation


def test_planner_parse_error_executes_no_policy_actions(tmp_path):
    env = _FakeEnv()
    backend = _FakeBackend([[0.1] * 7])
    planner = _FakePlanner(None, raw_output="not valid JSON")

    run_root, result = _run(tmp_path, env, backend, planner=planner)

    assert env.actions == [LIBERO_DUMMY_ACTION] * 10
    assert env.policy_steps == 0
    assert backend.calls == 0
    assert result["episode"]["termination_reason"] == "planner_parse_error"
    assert result["episode"]["chunks_executed"] == 0
    trace = json.loads((run_root / "trace.jsonl").read_text().strip())
    assert trace["event"] == "planner_parse_error"
    assert trace["planner_raw_output"] == "not valid JSON"
    assert trace["planner_thinking"] == "planner thinking"


def test_lift_and_grasp_tau_stops_mid_chunk_without_task_success(tmp_path):
    env = _FakeEnv()
    backend = _FakeBackend([[0.1] * 7, [0.2] * 7, [0.3] * 7])
    invocation = {
        "action": "vla_act",
        "prompt": "grasp and lift the bowl",
        "max_chunks": 2,
        "tau": "lift_and_grasp",
        "target": "akita_black_bowl_1",
    }
    planner = _FakePlanner(invocation)

    run_root, result = _run(
        tmp_path,
        env,
        backend,
        planner=planner,
        lift_tau_factory=_FakeLiftMonitor,
    )

    assert env.policy_steps == 2
    assert result["episode"]["task_success"] is False
    assert result["episode"]["tau_satisfied"] is True
    assert result["episode"]["termination_reason"] == "tau_satisfied"
    assert result["summary"]["tau_satisfied"] is True
    assert result["summary"]["tau_success_rate"] == 1.0
    assert (
        run_root / "videos/task_000_state_000_tau_success_task_incomplete.mp4"
    ).read_bytes() == b"fake mp4"
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    assert manifest["tau_monitor_ready"] is True
    assert manifest["tau_setup_error"] is None
    trace = json.loads((run_root / "trace.jsonl").read_text().strip())
    assert trace["tau_satisfied"] is True
    assert trace["tau_evidence"]["predicate"] == "lift_and_grasp"
    assert trace["tau_evidence"]["task_success_evaluated"] is False


def test_transient_tau_evaluation_error_is_traced_and_recovers(tmp_path):
    env = _FakeEnv()
    backend = _FakeBackend([[0.1] * 7, [0.2] * 7, [0.3] * 7])
    planner = _FakePlanner(
        {
            "action": "vla_act",
            "prompt": "grasp and lift the bowl",
            "max_chunks": 2,
            "tau": "lift_and_grasp",
            "target": "akita_black_bowl_1",
        }
    )

    run_root, result = _run(
        tmp_path,
        env,
        backend,
        planner=planner,
        lift_tau_factory=_TransientFailureLiftMonitor,
    )

    assert result["episode"]["tau_satisfied"] is True
    assert env.policy_steps == 2
    trace = json.loads((run_root / "trace.jsonl").read_text().strip())
    assert trace["tau_evaluation_errors"] == [
        {"action_index": 1, "error": "target temporarily occluded"}
    ]