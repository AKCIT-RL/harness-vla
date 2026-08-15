"""Tests for beta-only per-action logging and turn timing instrumentation."""

import json

from embodiedbench.evaluator.libero_action_log import flatten_turn_actions
from embodiedbench.evaluator.libero_analytic_executor import (
    LiberoPrimitiveExecution,
)
from embodiedbench.evaluator.libero_multi_turn_evaluator import (
    LiberoMultiTurnBudgets,
    LiberoMultiTurnEvaluator,
    LiberoVLAExecution,
)
from embodiedbench.planner.harness.trace_io import load_complete_jsonl


TARGET = "akita_black_bowl_1"


def _budgets(**overrides):
    values = {
        "max_turns": 4,
        "horizon": 20,
        "max_chunks_cap": 4,
        "max_move_steps": 5,
        "release_steps": 3,
    }
    values.update(overrides)
    return LiberoMultiTurnBudgets(**values)


class _Planner:
    last_thinking = "thinking"

    def __init__(self, invocations):
        self.invocations = list(invocations)
        self.calls = 0

    def act_turn(self, instruction, state, *, available_targets, max_chunks_cap):
        invocation = self.invocations[self.calls]
        self.calls += 1
        if invocation is None:
            return None, "not json"
        return invocation, json.dumps(invocation)


def _vla_trace():
    return [
        {
            "chunk_index": 1,
            "inference_duration_s": 0.25,
            "full_chunk_length": 10,
            "executed_actions": [[0.1] * 7, [0.2] * 7],
            "rewards": [0.0, 1.0],
            "dones": [False, False],
            "task_successes": [False, False],
            "tau_evidence": [
                {"tau_satisfied": False},
                {"tau_satisfied": True},
            ],
            "tau_evaluation_errors": [],
            "tau_satisfied": True,
        }
    ]


def test_flatten_vla_chunk_actions_projects_one_record_per_action():
    records = flatten_turn_actions(3, "vla_act", _vla_trace(), ts="2026-01-01T00:00:00+00:00")
    assert len(records) == 2
    first, second = records
    assert first["turn"] == 3
    assert first["primitive"] == "vla_act"
    assert first["source"] == "vla_chunk"
    assert first["chunk_index"] == 1
    assert first["action_index_in_turn"] == 1
    assert first["action"] == [0.1] * 7
    assert first["tau_satisfied"] is False
    assert first["inference_duration_s"] == 0.25
    assert first["turn_ts"] == "2026-01-01T00:00:00+00:00"
    assert second["action_index_in_turn"] == 2
    assert second["reward"] == 1.0
    assert second["tau_satisfied"] is True
    assert "inference_duration_s" not in second


def test_flatten_analytic_steps_and_skips_non_physical_entries():
    trace = [
        {"event": "target_resolved", "target": TARGET},
        {"step": 1, "action": [0.0] * 7, "reward": 0.0, "env_done": False, "info": {}},
        {"step": 2, "action": [0.0] * 7, "reward": 0.0, "env_done": True, "info": {}},
    ]
    records = flatten_turn_actions(1, "move_to", trace)
    assert len(records) == 2
    assert records[0]["source"] == "analytic_step"
    assert records[0]["step"] == 1
    assert records[1]["env_done"] is True
    assert "turn_ts" not in records[0]


def test_flatten_tolerates_missing_or_malformed_traces():
    assert flatten_turn_actions(1, "vla_act", None) == []
    assert flatten_turn_actions(1, "vla_act", "oops") == []
    assert flatten_turn_actions(1, "vla_act", [{"event": "tau_setup_error"}]) == []


def _run_episode(tmp_path, invocations, executors, **budget_overrides):
    trace_path = tmp_path / "trace.jsonl"
    evaluator = LiberoMultiTurnEvaluator(
        _Planner(invocations),
        vla_executor=executors["vla"],
        move_executor=executors["move"],
        release_executor=executors["release"],
        trace_path=trace_path,
    )
    result = evaluator.run(
        "place the bowl on the plate",
        {"frame": 0},
        available_targets=[TARGET, "plate_1"],
        budgets=_budgets(**budget_overrides),
    )
    return result, trace_path, tmp_path / "actions.jsonl"


def _executors():
    def vla(invocation, observation, *, max_steps):
        execution = LiberoPrimitiveExecution(
            observation={"frame": 1},
            primitive_success=True,
            task_success=True,
            termination_reason="task_success",
            steps_executed=2,
            trace=_vla_trace(),
        )
        return LiberoVLAExecution(execution, True, invocation["target"])

    def unused(invocation, observation, *, max_steps):
        raise AssertionError("unexpected executor call")

    return {"vla": vla, "move": unused, "release": unused}


def test_evaluator_writes_timing_and_actions_jsonl(tmp_path):
    invocation = {
        "action": "vla_act",
        "prompt": "pick up the bowl",
        "target": TARGET,
        "max_chunks": 2,
        "tau": "lift_and_grasp",
    }
    result, trace_path, actions_path = _run_episode(
        tmp_path, [invocation], _executors()
    )
    assert result.task_success is True

    records = load_complete_jsonl(trace_path)
    assert len(records) == 1
    timing = records[0]["timing"]
    assert timing["ts"].endswith("+00:00")
    assert timing["planner_duration_s"] >= 0.0
    assert timing["execution_duration_s"] >= 0.0
    assert timing["turn_duration_s"] >= timing["planner_duration_s"]
    assert timing["mean_step_duration_s"] >= 0.0

    actions = load_complete_jsonl(actions_path)
    assert len(actions) == 2
    assert actions[0]["turn"] == 1
    assert actions[0]["primitive"] == "vla_act"
    assert actions[0]["turn_ts"] == timing["ts"]


def test_parse_error_turn_records_timing_without_actions(tmp_path):
    result, trace_path, actions_path = _run_episode(
        tmp_path, [None], _executors(), max_turns=1
    )
    assert result.termination_reason == "planner_parse_error"
    records = load_complete_jsonl(trace_path)
    assert len(records) == 1
    timing = records[0]["timing"]
    assert "execution_duration_s" not in timing
    assert timing["planner_duration_s"] >= 0.0
    assert load_complete_jsonl(actions_path) == []
