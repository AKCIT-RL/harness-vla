import json

import numpy as np
import pytest

from harness_vla.eval.libero_analytic_executor import (
    LiberoPrimitiveExecution,
)
from harness_vla.eval.libero_multi_turn_evaluator import (
    LiberoMultiTurnBudgets,
    LiberoVLAExecution,
)
from harness_vla.eval.libero_multi_turn_run import (
    run_libero_multi_turn_episode,
)
from harness_vla.eval.libero_native_multi_turn import LiberoNativeOffsets
from harness_vla.eval.libero_vla_smoke import LIBERO_DUMMY_ACTION
from harness_vla.trace_io import load_complete_jsonl


TARGET = "akita_black_bowl_1"


def _observation(step=0):
    image = np.full((3, 4, 3), step, dtype=np.uint8)
    return {
        "step": step,
        "agentview_image": image,
        "robot0_eye_in_hand_image": image,
        "robot0_eef_pos": np.asarray([0.0, 0.0, 0.5]),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.asarray([0.0, 0.0]),
    }


class _FakeEnv:
    obj_of_interest = [TARGET, "plate_1"]

    def __init__(self):
        self.actions = []
        self.reset_calls = 0
        self.initial_state = None

    def reset(self):
        self.reset_calls += 1

    def set_init_state(self, initial_state):
        self.initial_state = initial_state
        return _observation()

    def step(self, action):
        self.actions.append(list(action))
        return _observation(len(self.actions)), 0.0, False, {}


class _Planner:
    model_name = "fake-gemma"
    think = True

    def __init__(self):
        self.calls = 0

    def act_turn(
        self,
        instruction,
        state,
        *,
        available_targets,
        max_chunks_cap,
    ):
        self.calls += 1
        invocation = {
            "action": "vla_act",
            "prompt": "lift the bowl",
            "target": TARGET,
            "max_chunks": 1,
            "tau": "lift_and_grasp",
        }
        return invocation, json.dumps(invocation)


class _Backend:
    host = "fake-host"
    port = 8123


def _budgets():
    return LiberoMultiTurnBudgets(
        max_turns=2,
        horizon=2,
        max_chunks_cap=1,
        max_move_steps=2,
        release_steps=1,
    )


def _write_video(path, frames):
    assert frames
    path.write_bytes(b"fake video")


def _unused_executor(*args, **kwargs):
    raise AssertionError("unexpected executor dispatch")


def _run(tmp_path, vla_executor, *, privileged_diagnostics=False):
    run_root = tmp_path / "run"
    run_root.mkdir()
    return run_root, run_libero_multi_turn_episode(
        env=_FakeEnv(),
        backend=_Backend(),
        planner=_Planner(),
        initial_state="fixed-state",
        instruction="place the black bowl on the plate",
        run_root=run_root,
        task_suite="libero_spatial",
        task_id=4,
        initial_state_index=7,
        seed=7,
        budgets=_budgets(),
        offsets=LiberoNativeOffsets(above_m=0.1, release_pose_m=0.05),
        position_tolerance=0.01,
        resize_with_pad=lambda image, height, width: image,
        convert_to_uint8=lambda image: image,
        vla_executor=vla_executor,
        move_executor=_unused_executor,
        release_executor=_unused_executor,
        video_writer=_write_video,
        privileged_diagnostics=privileged_diagnostics,
    )


def test_success_writes_complete_artifacts_and_excludes_settling_from_horizon(
    tmp_path,
):
    seen = {}

    def vla(invocation, observation, *, max_steps):
        run_root = tmp_path / "run"
        seen["manifest_during_execution"] = json.loads(
            (run_root / "run_manifest.json").read_text()
        )
        env = seen["env"]
        current = observation
        for _ in range(max_steps):
            current, _, _, _ = env.step([0.1] * 7)
        execution = LiberoPrimitiveExecution(
            observation=current,
            primitive_success=True,
            task_success=True,
            termination_reason="task_success",
            steps_executed=max_steps,
            trace=[{"event": "vla_complete", "steps": max_steps}],
        )
        return LiberoVLAExecution(execution, True, TARGET)

    run_root = tmp_path / "run"
    run_root.mkdir()
    env = _FakeEnv()
    seen["env"] = env
    result = run_libero_multi_turn_episode(
        env=env,
        backend=_Backend(),
        planner=_Planner(),
        initial_state="fixed-state",
        instruction="place the black bowl on the plate",
        run_root=run_root,
        task_suite="libero_spatial",
        task_id=4,
        initial_state_index=7,
        seed=7,
        budgets=_budgets(),
        offsets=LiberoNativeOffsets(above_m=0.1, release_pose_m=0.05),
        position_tolerance=0.01,
        resize_with_pad=lambda image, height, width: image,
        convert_to_uint8=lambda image: image,
        vla_executor=vla,
        move_executor=_unused_executor,
        release_executor=_unused_executor,
        video_writer=_write_video,
    )

    assert seen["manifest_during_execution"]["status"] == "in_progress"
    assert env.reset_calls == 1
    assert env.initial_state == "fixed-state"
    assert env.actions[:10] == [LIBERO_DUMMY_ACTION] * 10
    assert env.actions[10:] == [[0.1] * 7] * 2
    assert result["episode"]["actions_executed"] == 2
    assert result["episode"]["horizon"] == 2
    assert result["manifest"]["status"] == "completed"
    assert {path.name for path in run_root.iterdir()} == {
        "actions.jsonl",
        "episode.json",
        "run_manifest.json",
        "summary.json",
        "trace.jsonl",
        "videos",
    }


def test_video_manifest_flags_config_and_trace_have_expected_contract(tmp_path):
    def vla(invocation, observation, *, max_steps):
        execution = LiberoPrimitiveExecution(
            observation=observation,
            primitive_success=False,
            task_success=False,
            termination_reason="step_budget_exhausted",
            steps_executed=max_steps,
            trace=[{"event": "bounded_execution"}],
        )
        return LiberoVLAExecution(execution, False, None)

    run_root, result = _run(tmp_path, vla)

    video = run_root / "videos/task_004_state_007_failure.mp4"
    assert video.read_bytes() == b"fake video"
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["harness_complete"] is False
    assert manifest["implementation_scope"] == "published_libero_primitive_vocabulary"
    assert manifest["vla_tau_modes"] == ["lift_and_grasp", "task_success"]
    assert "one_primitive_per_turn" in manifest["scientific_classification"][
        "paper_confirmed"
    ]
    assert "in_process_loop" in manifest["scientific_classification"][
        "paper_compatible"
    ]
    assert "seven_primitive_libero_vocabulary_and_roles" in manifest[
        "scientific_classification"
    ]["paper_confirmed"]
    assert "quaternion_xyzw_pose_and_radian_setpoints" in manifest[
        "scientific_classification"
    ]["paper_compatible"]
    assert "guarded_libero_workspace_and_rotation_ranges" in manifest[
        "scientific_classification"
    ]["beta_only"]
    assert "privileged_contact_state" in manifest["scientific_classification"][
        "beta_only"
    ]
    assert manifest["privileged_segmentation"] is True
    assert manifest["privileged_contact_state"] is True
    assert manifest["privileged_diagnostics"] is False
    assert manifest["planner_receives_oracle_coordinates"] is False
    assert manifest["planner_model"] == "fake-gemma"
    assert manifest["think"] is True
    assert manifest["backend"] == "pirlinf_websocket"
    assert manifest["config"] == {
        "settling_steps": 10,
        "budgets": {
            "max_turns": 2,
            "horizon": 2,
            "max_chunks_cap": 1,
            "max_move_steps": 2,
            "release_steps": 1,
        },
        "offsets_m": {"above_m": 0.1, "release_pose_m": 0.05},
        "position_tolerance_m": 0.01,
        "rotation_tolerance_rad": 0.05,
        "camera": "agentview",
    }
    trace_text = (run_root / "trace.jsonl").read_text()
    assert "image" not in trace_text.lower()
    assert load_complete_jsonl(run_root / "trace.jsonl")[0]["turn"] == 1
    assert result["summary"]["budget_exhausted"] is True


def test_explicit_privileged_diagnostics_flag_marks_manifest_and_trace(tmp_path):
    def vla(invocation, observation, *, max_steps):
        execution = LiberoPrimitiveExecution(
            observation=observation,
            primitive_success=True,
            task_success=True,
            termination_reason="task_success",
            steps_executed=1,
            trace=[],
        )
        return LiberoVLAExecution(execution, True, TARGET)

    run_root, result = _run(tmp_path, vla, privileged_diagnostics=True)

    assert result["manifest"]["privileged_diagnostics"] is True
    assert "privileged_diagnostics" in result["manifest"][
        "scientific_classification"
    ]["beta_only"]
    snapshot = load_complete_jsonl(run_root / "trace.jsonl")[0][
        "privileged_diagnostics"
    ][0]
    assert snapshot["event"] == "post_grasp"
    assert snapshot["source"] == "privileged_mujoco_state"
    assert snapshot["beta_only"] is True
    assert snapshot["eef_pose"]["value"] is None


def test_executor_failure_preserves_trace_video_and_incomplete_manifest(tmp_path):
    def failing_vla(invocation, observation, *, max_steps):
        raise RuntimeError("backend disconnected")

    with pytest.raises(RuntimeError, match="backend disconnected"):
        _run(tmp_path, failing_vla)

    run_root = tmp_path / "run"
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["error_type"] == "RuntimeError"
    assert manifest["error"] == "backend disconnected"
    assert (run_root / "trace.jsonl").exists()
    assert (run_root / manifest["video"]).read_bytes() == b"fake video"
    assert not (run_root / "episode.json").exists()
    assert not (run_root / "summary.json").exists()


@pytest.mark.parametrize("position_tolerance", [0.0, float("nan")])
def test_invalid_config_fails_before_execution(tmp_path, position_tolerance):
    run_root = tmp_path / "run"
    run_root.mkdir()
    env = _FakeEnv()

    with pytest.raises(ValueError, match="position_tolerance"):
        run_libero_multi_turn_episode(
            env=env,
            backend=_Backend(),
            planner=_Planner(),
            initial_state="fixed-state",
            instruction="place the bowl",
            run_root=run_root,
            task_suite="libero_spatial",
            task_id=0,
            initial_state_index=0,
            seed=7,
            budgets=_budgets(),
            offsets=LiberoNativeOffsets(above_m=0.1, release_pose_m=0.05),
            position_tolerance=position_tolerance,
            resize_with_pad=lambda image, height, width: image,
            convert_to_uint8=lambda image: image,
            vla_executor=_unused_executor,
            move_executor=_unused_executor,
            release_executor=_unused_executor,
        )

    assert env.reset_calls == 0
    assert not (run_root / "run_manifest.json").exists()