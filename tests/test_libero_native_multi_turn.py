import json

import numpy as np
import pytest

from harness_vla.eval.libero_native_multi_turn import (
    LiberoNativeExecutionState,
    LiberoNativeOffsets,
    make_native_move_executor,
    make_native_release_executor,
    make_native_vla_executor,
    resolve_target_xyz,
    VisualLiftAndGraspMonitor,
)
from harness_vla.primitives.libero_primitives import LiberoPrimitiveError
from harness_vla.backends.pirlinf_backend import PiRLinfChunk


def _observation(step=0, xyz=(0.0, 0.0, 0.5)):
    image = np.full((2, 2, 3), step, dtype=np.uint8)
    return {
        "step": step,
        "agentview_image": image,
        "robot0_eye_in_hand_image": image + 1,
        "robot0_eef_pos": np.asarray(xyz, dtype=float),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.asarray([0.1, 0.2]),
    }


def _resize(image, height, width):
    assert (height, width) == (224, 224)
    return image


def _uint8(image):
    return np.asarray(image, dtype=np.uint8)


class _FakeEnv:
    def __init__(self, *, success_after=None, done_after=None):
        self.actions = []
        self.success_after = success_after
        self.done_after = done_after
        self.current = _observation()

    def step(self, action):
        self.actions.append(list(action))
        xyz = np.asarray(self.current["robot0_eef_pos"], dtype=float)
        xyz = xyz + np.asarray(action[:3], dtype=float) * 0.05
        self.current = _observation(len(self.actions), xyz)
        done = self.done_after is not None and len(self.actions) >= self.done_after
        return self.current, float(len(self.actions)), done, {"step": len(self.actions)}

    def check_success(self):
        return self.success_after is not None and len(self.actions) >= self.success_after


def test_target_modes_offsets_provenance_and_move_regrounding():
    calls = []

    def grounder(observation, target):
        calls.append((observation["step"], target))
        return {
            "world_xyz": [0.02 * len(calls), 0.0, 0.4],
            "provenance": {"frame_id": np.int64(observation["step"])},
        }

    offsets = LiberoNativeOffsets(above_m=0.1, release_pose_m=0.2)
    above = resolve_target_xyz(
        _observation(), "plate", "above", grounder=grounder, offsets=offsets
    )
    release = resolve_target_xyz(
        _observation(1), "plate", "release_pose", grounder=grounder, offsets=offsets
    )
    assert above.xyz == pytest.approx((0.02, 0.0, 0.5))
    assert release.xyz == pytest.approx((0.04, 0.0, 0.6))
    assert release.provenance["vertical_offset_m"] == 0.2
    assert release.provenance["frame_id"] == 1

    env = _FakeEnv()
    frames = []
    move = make_native_move_executor(
        env,
        grounder,
        offsets=offsets,
        position_tolerance=1e-6,
        frame_callback=lambda observation: frames.append(observation["step"]),
    )
    invocation = {
        "action": "move_to",
        "target": "plate",
        "mode": "above",
        "gripper": "close",
    }
    first = move(invocation, env.current, max_steps=10)
    second = move(invocation, first.observation, max_steps=10)

    assert calls[-2:] == [(0, "plate"), (2, "plate")]
    assert first.trace[0]["event"] == "target_resolved"
    assert second.trace[0]["target_xyz"] != first.trace[0]["target_xyz"]
    assert env.actions and all(action[6] == 1.0 for action in env.actions)
    assert frames == list(range(1, len(env.actions) + 1))


def test_move_stops_when_bilateral_grasp_is_lost():
    env = _FakeEnv()
    state = LiberoNativeExecutionState(holding="bowl")
    checks = []

    def grounder(observation, target):
        return {"world_xyz": [0.2, 0.0, 0.5], "provenance": {}}

    def grasp_monitor(env, observation, holding):
        checks.append((observation["step"], holding))
        return False

    move = make_native_move_executor(
        env,
        grounder,
        offsets=LiberoNativeOffsets(above_m=0.1, release_pose_m=0.05),
        position_tolerance=1e-6,
        execution_state=state,
        grasp_monitor=grasp_monitor,
    )
    result = move(
        {
            "action": "move_to",
            "target": "plate",
            "mode": "above",
            "gripper": "close",
        },
        env.current,
        max_steps=5,
    )

    assert result.termination_reason == "grasp_lost"
    assert result.steps_executed == 1
    assert checks == [(1, "bowl")]


def test_visual_transport_keeps_gripper_closed_without_privileged_contact_guard():
    env = _FakeEnv()
    state = LiberoNativeExecutionState(holding="bowl")
    move = make_native_move_executor(
        env,
        lambda observation, target: {
            "world_xyz": [0.1, 0.0, 0.5],
            "provenance": {"privileged_segmentation": False},
        },
        offsets=LiberoNativeOffsets(above_m=0.1, release_pose_m=0.05),
        position_tolerance=1e-6,
        execution_state=state,
        grasp_monitor=False,
    )

    result = move(
        {
            "action": "move_to",
            "target": "plate",
            "mode": "release_pose",
            "gripper": "close",
        },
        env.current,
        max_steps=5,
    )

    assert result.termination_reason == "postcondition_met"
    assert all(action[6] == 1.0 for action in env.actions)


@pytest.mark.parametrize(
    "invocation",
    [
        {
            "action": "move_pose",
            "xyz": [0.05, 0.0, 0.5],
            "pose": [0.0, 0.0, 0.0, 1.0],
            "gripper": "open",
        },
        {"action": "rotate_wrist", "target_yaw": 0.5},
        {"action": "rotate_pitch", "target_pitch": -0.25},
        {"action": "set_gripper", "gripper": "close"},
    ],
)
def test_native_adapter_executes_each_added_analytic_primitive(invocation):
    env = _FakeEnv()
    grounder_calls = []
    execute = make_native_move_executor(
        env,
        lambda observation, target: grounder_calls.append(target),
        offsets=LiberoNativeOffsets(above_m=0.1, release_pose_m=0.05),
        position_tolerance=1e-6,
    )

    result = execute(invocation, env.current, max_steps=1)

    assert result.steps_executed == 1
    assert len(env.actions) == 1
    assert grounder_calls == []


@pytest.mark.parametrize("value", [0.0, -0.1, float("inf"), float("nan")])
def test_offsets_must_be_finite_and_positive(value):
    with pytest.raises(LiberoPrimitiveError):
        LiberoNativeOffsets(above_m=value, release_pose_m=0.1)


def test_release_adapter_delegates_open_action_and_captures_each_frame():
    env = _FakeEnv()
    frames = []
    release = make_native_release_executor(
        env, frame_callback=lambda observation: frames.append(observation["step"])
    )

    result = release({"action": "release"}, env.current, max_steps=2)

    assert result.primitive_success is True
    assert result.termination_reason == "release_completed_task_incomplete"
    assert len(env.actions) == 2
    assert all(action == [0.0] * 6 + [-1.0] for action in env.actions)
    assert frames == [1, 2]


class _Backend:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.observations = []

    def infer_chunk(self, observation, prompt):
        self.observations.append(observation)
        actions = self.chunks[len(self.observations) - 1]
        return PiRLinfChunk(
            tuple(tuple(action) for action in actions), len(actions), 0.01
        )


class _TauMonitor:
    def __init__(self, env, observation, target, *, satisfy_after=None, error=False):
        self.env = env
        self.target = target
        self.satisfy_after = satisfy_after
        self.error = error

    def evaluate(self, observation):
        if self.error:
            raise ValueError("grounding unavailable")
        return {
            "predicate": "lift_and_grasp",
            "tau_satisfied": (
                self.satisfy_after is not None
                and len(self.env.actions) >= self.satisfy_after
            ),
            "target": self.target,
            "height": np.float32(observation["step"]),
        }


def _vla(env, backend, monitor_factory, frames=None):
    return make_native_vla_executor(
        env,
        backend,
        resize_with_pad=_resize,
        convert_to_uint8=_uint8,
        tau_monitor_factory=monitor_factory,
        frame_callback=(
            None
            if frames is None
            else lambda observation: frames.append(observation["step"])
        ),
    )


def _invocation(max_chunks=3):
    return {
        "action": "vla_act",
        "prompt": "pick up the bowl",
        "target": "bowl",
        "max_chunks": max_chunks,
        "tau": "lift_and_grasp",
    }


def _task_success_invocation(max_chunks=3):
    invocation = _invocation(max_chunks)
    invocation.update(
        prompt="place the bowl into plate_1",
        tau="task_success",
    )
    return invocation


def test_vla_reinfers_from_refreshed_observation_and_respects_horizon():
    action = [0.0] * 7
    env = _FakeEnv()
    backend = _Backend([[action], [action], [action]])
    executor = _vla(
        env, backend, lambda env, obs, target: _TauMonitor(env, obs, target)
    )

    result = executor(_invocation(), env.current, max_steps=2)

    assert len(backend.observations) == 2
    assert backend.observations[0].front_rgb[0, 0, 0] == 0
    assert backend.observations[1].front_rgb[0, 0, 0] == 1
    assert len(env.actions) == result.execution.steps_executed == 2
    assert result.execution.termination_reason == "step_budget_exhausted"


def test_vla_tau_stops_mid_chunk_and_is_the_only_source_of_holding():
    env = _FakeEnv()
    backend = _Backend([[[0.0] * 7] * 4])
    frames = []
    executor = _vla(
        env,
        backend,
        lambda env, obs, target: _TauMonitor(
            env, obs, target, satisfy_after=2
        ),
        frames,
    )

    result = executor(_invocation(), env.current, max_steps=4)

    assert len(env.actions) == 2
    assert frames == [1, 2]
    assert result.tau_satisfied is True
    assert result.execution.primitive_success is True
    assert result.execution.task_success is False
    assert result.execution.termination_reason == "lift_and_grasp_satisfied"
    assert result.holding == "bowl"


@pytest.mark.parametrize(
    "env_kwargs,reason",
    [({"success_after": 1}, "task_success"), ({"done_after": 1}, "env_done")],
)
def test_vla_official_stops_are_distinct_from_tau_and_do_not_set_holding(
    env_kwargs, reason
):
    env = _FakeEnv(**env_kwargs)
    backend = _Backend([[[0.0] * 7] * 3])
    executor = _vla(
        env, backend, lambda env, obs, target: _TauMonitor(env, obs, target)
    )

    result = executor(_invocation(), env.current, max_steps=3)

    assert len(env.actions) == 1
    assert result.tau_satisfied is False
    assert result.holding is None
    assert result.execution.termination_reason == reason
    assert result.execution.task_success is (reason == "task_success")
    assert result.execution.primitive_success is (reason == "task_success")


@pytest.mark.parametrize(
    "env_kwargs,reason,expected_holding",
    [
        ({"success_after": 1}, "task_success", None),
        ({"done_after": 1}, "env_done", "bowl"),
        ({}, "step_budget_exhausted", "bowl"),
    ],
)
def test_task_success_tau_uses_only_official_success_and_preserves_holding_on_failure(
    env_kwargs, reason, expected_holding
):
    env = _FakeEnv(**env_kwargs)
    backend = _Backend([[[0.0] * 7] * 2])

    def forbidden_monitor(*args):
        raise AssertionError("task_success must not instantiate a grasp monitor")

    state = LiberoNativeExecutionState(holding="bowl")
    executor = make_native_vla_executor(
        env,
        backend,
        resize_with_pad=_resize,
        convert_to_uint8=_uint8,
        execution_state=state,
        tau_monitor_factory=forbidden_monitor,
    )

    result = executor(_task_success_invocation(max_chunks=1), env.current, max_steps=2)

    expected_success = reason == "task_success"
    assert result.execution.termination_reason == reason
    assert result.execution.task_success is expected_success
    assert result.execution.primitive_success is expected_success
    assert result.tau_satisfied is expected_success
    assert result.holding == state.holding == expected_holding
    evidence = result.execution.trace[0]["tau_evidence"][-1]
    assert evidence["predicate"] == "task_success"
    assert evidence["source"] == "env.check_success"


def test_tau_errors_fail_closed_remain_in_serializable_trace_and_never_hold():
    env = _FakeEnv()
    backend = _Backend([[[0.0] * 7] * 2])
    executor = _vla(
        env,
        backend,
        lambda env, obs, target: _TauMonitor(env, obs, target, error=True),
    )

    result = executor(_invocation(max_chunks=1), env.current, max_steps=2)

    assert result.tau_satisfied is False
    assert result.holding is None
    assert result.execution.primitive_success is False
    assert result.execution.trace[0]["tau_evaluation_errors"] == [
        {"action_index": 1, "error": "grounding unavailable"},
        {"action_index": 2, "error": "grounding unavailable"},
    ]
    json.dumps(result.execution.trace)


def test_visual_tau_uses_only_repeated_rgbd_height_without_contact_state():
    heights = iter((0.40, 0.42, 0.44))
    calls = []

    def grounder(observation, target):
        calls.append((observation["step"], target))
        return {"world_xyz": [0.0, 0.0, next(heights)]}

    monitor = VisualLiftAndGraspMonitor(
        grounder, _observation(), "bowl", minimum_lift_m=0.03
    )
    first = monitor.evaluate(_observation(1))
    second = monitor.evaluate(_observation(2))

    assert first["tau_satisfied"] is False
    assert second["tau_satisfied"] is True
    assert second["privileged_contact_state"] is False
    assert second["lift"]["coordinate_source"] == "visual_rgbd_projection"
    assert calls == [(0, "bowl"), (1, "bowl"), (2, "bowl")]