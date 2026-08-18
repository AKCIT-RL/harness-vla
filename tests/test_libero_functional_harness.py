import json

import numpy as np
import pytest

from harness_vla.eval.libero_analytic_executor import LiberoPrimitiveExecution
from harness_vla.eval.libero_memory_lifecycle import (
    bootstrap_memories,
    prepare_deployment,
    promote_global_memory,
)
from harness_vla.eval.libero_multi_turn_evaluator import (
    LiberoMultiTurnBudgets,
    LiberoVLAExecution,
)
from harness_vla.eval.libero_multi_turn_run import (
    _visual_grounder,
    run_libero_multi_turn_episode,
)
from harness_vla.eval.libero_native_multi_turn import LiberoNativeOffsets
from harness_vla.perception.libero_visual_grounding import VisualPixelSelection
from harness_vla.planner.phase_policy import build_phase_manifest
from harness_vla.trace_io import load_complete_jsonl


TARGETS = ["bowl_1", "plate_1"]


def _observation(step=0):
    image = np.full((5, 5, 3), step, dtype=np.uint8)
    return {
        "step": step,
        "agentview_image": image,
        "agentview_depth": np.full((5, 5), 0.5),
    }


class _Env:
    obj_of_interest = TARGETS

    def __init__(self):
        self.reset_calls = 0
        self.actions = []

    def reset(self):
        self.reset_calls += 1

    def set_init_state(self, initial_state):
        return _observation()

    def step(self, action):
        self.actions.append(action)
        return _observation(len(self.actions)), 0.0, False, {}


class _Planner:
    model_name = "fake"
    think = False

    def __init__(self):
        self.prompts = []

    def act_turn(
        self,
        instruction,
        state,
        *,
        available_targets,
        max_chunks_cap,
        memory_context=None,
    ):
        self.prompts.append(
            {
                "state": state,
                "memory_context": memory_context,
                "max_chunks_cap": max_chunks_cap,
            }
        )
        invocation = {
            "action": "vla_act",
            "prompt": "lift the bowl",
            "target": "bowl_1",
            "max_chunks": 1,
            "tau": "lift_and_grasp",
        }
        return invocation, json.dumps(invocation)


class _Backend:
    host = "fake"
    port = 1


def _successful_vla(invocation, observation, *, max_steps):
    execution = LiberoPrimitiveExecution(
        observation=observation,
        primitive_success=True,
        task_success=True,
        termination_reason="task_success",
        steps_executed=1,
        trace=[
            {
                "event": "visual_tau",
                "privileged_contact_state": False,
                "task_success": True,
            }
        ],
    )
    return LiberoVLAExecution(execution, True, invocation["target"])


def _unused(*args, **kwargs):
    raise AssertionError("unused executor")


def _run(
    root,
    env,
    planner,
    manifest,
    phase,
    seed,
    protocol,
    memory=None,
    reset_environment=None,
):
    return run_libero_multi_turn_episode(
        env=env,
        backend=_Backend(),
        planner=planner,
        initial_state="fixed",
        instruction="place the bowl on the plate",
        run_root=root,
        task_suite="libero_spatial",
        task_id=0,
        initial_state_index=0,
        seed=seed,
        budgets=LiberoMultiTurnBudgets(8, 20, 4, 5, 2),
        offsets=LiberoNativeOffsets(0.1, 0.05),
        position_tolerance=0.01,
        resize_with_pad=lambda image, height, width: image,
        convert_to_uint8=lambda image: image,
        grounder=lambda observation, target: {
            "world_xyz": [0.0, 0.0, 0.5],
            "provenance": {"privileged_segmentation": False},
        },
        vla_executor=_successful_vla,
        move_executor=_unused,
        release_executor=_unused,
        video_writer=lambda path, frames: path.write_bytes(b"video"),
        phase_manifest=manifest,
        phase=phase,
        protocol_seed=99,
        deployment_memory_session=memory,
        file_repl_dir=protocol,
        visual_locator=lambda *args: VisualPixelSelection([2, 2], 0.9),
        object_labels={"bowl_1": "black bowl", "plate_1": "white plate"},
        object_roles={"bowl_1": ["manipulable"], "plate_1": ["destination"]},
        reset_environment=reset_environment,
    )


def test_bootstrap_deployment_composes_functional_harness(tmp_path):
    manifest = build_phase_manifest(7, [101], bootstrap_budget=3, deployment_budget=1)
    bootstrap_root = tmp_path / "bootstrap"
    bootstrap_root.mkdir()
    bootstrap_protocol = tmp_path / "bootstrap_protocol"
    bootstrap_env = _Env()
    bootstrap_planner = _Planner()

    bootstrap_result = _run(
        bootstrap_root,
        bootstrap_env,
        bootstrap_planner,
        manifest,
        "bootstrap",
        7,
        bootstrap_protocol,
    )
    assert bootstrap_env.reset_calls == 1
    assert bootstrap_result["manifest"]["reportable"] is False
    assert bootstrap_result["manifest"]["harness_complete"] is False

    task_memory = tmp_path / "task_memory"
    global_ledger = tmp_path / "global_ledger.json"
    memory_result = bootstrap_memories(
        manifest,
        seed=7,
        trace_path=str(bootstrap_root / "trace.jsonl"),
        episode_result_path=str(bootstrap_root / "episode.json"),
        task_memory_dir=str(task_memory),
        global_ledger_path=str(global_ledger),
    )
    candidate = memory_result.audit["global_memory"]["candidate_ids"][0]
    promote_global_memory(
        str(global_ledger),
        candidate_id=candidate,
        semantic_interpretation="Re-ground the symbolic target before retry.",
    )
    session = prepare_deployment(
        manifest,
        seed=101,
        task_memory_dir=str(task_memory),
        global_ledger_path=str(global_ledger),
    )

    deployment_root = tmp_path / "deployment"
    deployment_root.mkdir()
    deployment_protocol = tmp_path / "deployment_protocol"
    deployment_env = _Env()
    deployment_planner = _Planner()
    result = _run(
        deployment_root,
        deployment_env,
        deployment_planner,
        manifest,
        "deployment",
        101,
        deployment_protocol,
        memory=session,
    )

    assert deployment_env.reset_calls == 0
    prompt = deployment_planner.prompts[0]
    assert "symbolic task structure" in prompt["memory_context"]
    assert "Re-ground the symbolic target before retry." in prompt["memory_context"]
    assert "xyz" not in prompt["memory_context"].casefold()
    assert prompt["state"]["budget"]["actions_remaining"] == 1
    assert prompt["max_chunks_cap"] == 1

    assert {
        "command_01.json",
        "state_01.json",
        "log_01.json",
        "ledger.jsonl",
        "status.json",
    } == {path.name for path in deployment_protocol.iterdir()}
    trace = load_complete_jsonl(deployment_root / "trace.jsonl")[0]
    assert trace["protocol"]["turn"]["turn"] == 1
    assert set(trace["protocol"]["turn"]["files"]) == {"command", "state", "log"}
    assert trace["object_labels"]["bowl_1"] == "black bowl"
    assert trace["object_roles"]["plate_1"] == ["destination"]

    run_manifest = result["manifest"]
    assert run_manifest["phase"] == "deployment"
    assert run_manifest["reportable"] is True
    assert run_manifest["harness_complete"] is True
    assert run_manifest["task_memory"] is True
    assert run_manifest["global_memory"] is True
    assert {
        "bootstrap_deployment_separation",
        "file_mediated_repl",
        "visual_rgbd_world_grounding",
        "task_specific_memory",
        "global_memory",
    }.issubset(run_manifest["scientific_classification"]["paper_confirmed"])
    assert run_manifest["privileged_segmentation"] is False
    assert run_manifest["privileged_contact_state"] is False
    assert run_manifest["scientific_classification"]["beta_only"] == [
        "visual_pixel_locator",
        "visual_rgbd_lift_tau",
        "guarded_libero_workspace_and_rotation_ranges",
    ]
    assert run_manifest["memory_hashes"]["unchanged"] is True
    assert run_manifest["memory_hashes"]["hashes_before"] == session.hashes_before

    blocked_root = tmp_path / "blocked_reset"
    blocked_root.mkdir()
    blocked_env = _Env()
    with pytest.raises(ValueError, match="reset is forbidden"):
        _run(
            blocked_root,
            blocked_env,
            _Planner(),
            manifest,
            "deployment",
            101,
            tmp_path / "blocked_protocol",
            memory=session,
            reset_environment=True,
        )
    assert blocked_env.reset_calls == 0


def test_visual_grounder_does_not_read_segmentation(monkeypatch):
    class _Model:
        stat = type("Stat", (), {"extent": 1.0})()
        vis = type("Vis", (), {"map": type("Map", (), {"zfar": 10.0, "znear": 0.1})()})()
        cam_fovy = [45.0]

        @staticmethod
        def camera_name2id(name):
            return 0

    class _Data:
        cam_xmat = [np.eye(3).reshape(-1)]
        cam_xpos = [np.zeros(3)]

    sim = type("Sim", (), {"model": _Model(), "data": _Data()})()
    env = type("Wrapper", (), {"env": type("Inner", (), {"sim": sim})()})()

    class _ForbiddenSegmentation:
        def __array__(self, *args, **kwargs):
            raise AssertionError("segmentation was accessed")

    observation = _observation()
    observation["agentview_segmentation_instance"] = _ForbiddenSegmentation()
    grounder = _visual_grounder(
        env,
        observation,
        "agentview",
        0,
        lambda *args: VisualPixelSelection([2, 2], 0.9),
        {"bowl_1": "black bowl"},
    )

    result = grounder(observation, "bowl_1")
    assert result["provenance"]["privileged_segmentation"] is False
