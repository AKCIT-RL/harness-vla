"""Simulator-free Task Specific Memory integration tests for EB-Manipulation."""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from harness_vla.memory.task_memory import build_task_memory


def _import_evaluator(monkeypatch):
    stubs = {
        "embodiedbench.envs.eb_manipulation.EBManEnv": {
            "EBManEnv": object,
            "ValidEvalSets": ["base"],
        },
        "embodiedbench.envs.eb_manipulation.eb_man_utils": {
            "form_harness_grounding_artifact_for_input": lambda *args, **kwargs: None,
        },
        "embodiedbench.envs.eb_manipulation.rgbd_grounding": {
            "compute_oracle_metrics": lambda *args: {},
            "summarize_oracle_frames": lambda frames: {},
        },
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop(
        "embodiedbench.evaluator.eb_manipulation_harness_evaluator", None
    )
    from embodiedbench.evaluator.eb_manipulation_harness_evaluator import (
        EB_ManipulationHarnessEvaluator,
    )

    return EB_ManipulationHarnessEvaluator


def _write_memory(path):
    records = [
        {
            "turn": 1,
            "invocation": {"action": "move_to", "target": "seed cube"},
            "primitive": "move_to",
            "primitive_postcondition_met": True,
            "is_contact": False,
            "object_labels": {"seed cube": "red cube"},
            "object_roles": {"seed cube": ["manipulable"]},
            "step_results": [],
        },
        {
            "turn": 2,
            "invocation": {
                "action": "vla_act",
                "object": "seed cube",
                "mode": "grasp",
            },
            "primitive": "vla_act",
            "primitive_postcondition_met": True,
            "is_contact": True,
            "object_labels": {"seed cube": "red cube"},
            "object_roles": {"seed cube": ["manipulable"]},
            "step_results": [],
        },
    ]
    decision = build_task_memory(
        records, {"task_success": 1.0, "instruction": "grasp the red cube"}
    )
    assert decision.accepted
    path.mkdir()
    (path / "audit.json").write_text(json.dumps(decision.audit), encoding="utf-8")
    (path / "commands.jsonl").write_text(
        "".join(
            json.dumps(command, sort_keys=True, separators=(",", ":")) + "\n"
            for command in decision.commands
        ),
        encoding="utf-8",
    )
    return decision.audit["source"]["commands_sha256"]


class _NoActionPlanner:
    planner_steps = 0
    output_json_error = 0
    act_calls = 0

    def reset(self):
        self.planner_steps = 0
        self.output_json_error = 0

    def act(self, *args, **kwargs):
        self.act_calls += 1
        raise AssertionError("planner must not run after memory binding rejection")


class _NoStepEnv:
    def __init__(self, log_path):
        self.log_path = str(log_path)
        self.number_of_episodes = 1
        self._current_episode_num = 0
        self._current_step = 0
        self._max_episode_steps = 5
        self.task_class = "dummy"
        self.episode_language_instruction = "grasp the red cube"
        self.step_calls = 0

    def reset(self):
        self._current_episode_num += 1
        return None, SimpleNamespace()

    def step(self, action):
        self.step_calls += 1
        raise AssertionError("env.step must not run after memory binding rejection")

    def close(self):
        pass


def test_ambiguous_memory_binding_aborts_before_planner_compile_or_env_step(
    tmp_path, monkeypatch
):
    memory_path = tmp_path / "memory"
    expected_hash = _write_memory(memory_path)
    evaluator_type = _import_evaluator(monkeypatch)
    evaluator = evaluator_type({
        "model_name": "fake",
        "task_memory_path": str(memory_path),
        "max_turns": 1,
        "save_grounding_audit": False,
    })
    evaluator.env = _NoStepEnv(tmp_path / "run")
    evaluator.log_path = evaluator.env.log_path
    evaluator.planner = _NoActionPlanner()
    evaluator._perceive_grounding = lambda obs: ({
        "planner_coords": {
            "current cube a": [11, 12, 13],
            "current cube b": [21, 22, 23],
        },
        "roles": {
            "current cube a": ["manipulable"],
            "current cube b": ["manipulable"],
        },
        "labels": {
            "current cube a": "red cube",
            "current cube b": "red cube",
        },
        "id_to_sim_name": {},
        "frame_id": 1,
        "coordinate_source": "current_test_scene",
        "objects": {},
    }, {})

    evaluator.evaluate()

    assert evaluator.planner.act_calls == 0
    assert evaluator.env.step_calls == 0
    result = json.loads(
        (Path(evaluator.env.log_path) / "results" / "episode_1_res.json").read_text()
    )
    assert result["task_memory"]["decision"] == "rejected"
    assert result["task_memory"]["hash"] == expected_hash
    assert result["task_memory"]["stage"] == "grounding"
    assert result["task_memory"]["rejection_turn"] == 1
    assert "ambiguous current grounding" in result["task_memory"]["reason"]
    trace = json.loads(
        (Path(evaluator.env.log_path) / "results" / "trace_episode_1.jsonl")
        .read_text()
        .strip()
    )
    assert trace["status"] == "task_memory_rejected"
    assert trace["execution_status"] == "not_executed"
    assert "compiled_actions" not in trace