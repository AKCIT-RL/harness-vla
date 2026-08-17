"""Tests for the beta-only frozen-protocol battery runner."""

import json

import pytest

from embodiedbench.evaluator.libero_battery import (
    aggregate_battery,
    episode_name,
    expand_protocol_matrix,
    load_protocol,
    run_battery,
)


def _protocol():
    return {
        "protocol_name": "benchmark_frozen_v1",
        "version": 1,
        "bootstrap_seeds": [7],
        "deployment_seeds": [101, 102],
        "budgets": {
            "max_turns": 8,
            "horizon": 220,
            "max_chunks_cap": 20,
            "max_move_steps": 40,
            "release_steps": 10,
        },
        "tasks": [
            {
                "task_suite": "libero_spatial",
                "task_id": 0,
                "initial_state_indices": [0, 1],
            },
            {
                "task_suite": "libero_object",
                "task_id": 5,
                "initial_state_indices": [0],
            },
        ],
        "hidden_tasks": {
            "tasks": [
                {
                    "task_suite": "libero_spatial",
                    "task_id": 7,
                    "initial_state_indices": [0],
                }
            ]
        },
    }


def test_load_protocol_validates_required_keys(tmp_path):
    payload = _protocol()
    del payload["deployment_seeds"]
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="deployment_seeds"):
        load_protocol(path)


def test_load_protocol_accepts_frozen_v1_file(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(_protocol()), encoding="utf-8")
    protocol = load_protocol(path)
    assert protocol["protocol_name"] == "benchmark_frozen_v1"


def test_expand_matrix_is_deterministic_and_paired():
    matrix = expand_protocol_matrix(_protocol(), condition="harness")
    assert len(matrix) == 3 * 2  # three (task, state) pairs x two seeds
    assert matrix[0] == {
        "condition": "harness",
        "task_suite": "libero_spatial",
        "task_id": 0,
        "initial_state_index": 0,
        "seed": 101,
    }
    baseline = expand_protocol_matrix(_protocol(), condition="baseline_vla")
    harness_keys = [
        (m["task_suite"], m["task_id"], m["initial_state_index"], m["seed"])
        for m in matrix
    ]
    baseline_keys = [
        (m["task_suite"], m["task_id"], m["initial_state_index"], m["seed"])
        for m in baseline
    ]
    assert harness_keys == baseline_keys


def test_expand_matrix_rejects_undeclared_seeds():
    with pytest.raises(ValueError, match="not declared"):
        expand_protocol_matrix(_protocol(), condition="harness", seeds=[999])


def test_expand_matrix_includes_hidden_only_when_requested():
    visible = expand_protocol_matrix(_protocol(), condition="harness")
    with_hidden = expand_protocol_matrix(
        _protocol(), condition="harness", include_hidden=True
    )
    assert len(with_hidden) == len(visible) + 2  # one hidden state x two seeds
    assert any(m["task_id"] == 7 for m in with_hidden)
    assert not any(m["task_id"] == 7 for m in visible)


def test_expand_matrix_hidden_only_selects_just_hidden_tasks():
    hidden = expand_protocol_matrix(
        _protocol(), condition="harness", hidden_only=True
    )
    assert len(hidden) == 2  # one hidden state x two seeds
    assert all(m["task_id"] == 7 for m in hidden)


def test_expand_matrix_hidden_only_requires_hidden_tasks():
    protocol = _protocol()
    del protocol["hidden_tasks"]
    with pytest.raises(ValueError, match="no hidden tasks"):
        expand_protocol_matrix(protocol, condition="harness", hidden_only=True)


def test_episode_name_is_deterministic():
    spec = {
        "condition": "harness",
        "task_suite": "libero_spatial",
        "task_id": 0,
        "initial_state_index": 1,
        "seed": 101,
    }
    assert episode_name(spec) == "harness_libero_spatial_task00_state01_seed101"


def _fake_runner(outcomes):
    calls = []

    def runner(spec, episode_root):
        calls.append(spec)
        key = (spec["task_id"], spec["initial_state_index"], spec["seed"])
        outcome = outcomes[key]
        if isinstance(outcome, Exception):
            raise outcome
        episode = {
            "task_success": outcome,
            "termination_reason": (
                "task_success" if outcome else "max_turns_exhausted"
            ),
            "turns_executed": 3,
            "actions_executed": 40,
        }
        (episode_root / "episode.json").write_text(
            json.dumps(episode), encoding="utf-8"
        )
        return episode

    return runner, calls


def test_run_battery_tolerates_crashes_and_aggregates(tmp_path):
    protocol = _protocol()
    protocol["tasks"] = [protocol["tasks"][0]]
    outcomes = {
        (0, 0, 101): True,
        (0, 0, 102): RuntimeError("sim crash"),
        (0, 1, 101): False,
        (0, 1, 102): True,
    }
    runner, calls = _fake_runner(outcomes)
    outcome = run_battery(
        protocol,
        battery_root=tmp_path / "battery",
        episode_runner=runner,
        condition="harness",
    )
    summary = outcome["summary"]
    assert summary["episodes_planned"] == 4
    assert summary["episodes_completed"] == 3
    assert summary["episodes_failed"] == 1
    assert summary["successes"] == 2
    assert summary["task_success_rate"] == pytest.approx(2 / 3)
    assert summary["failures"][0]["error_type"] == "RuntimeError"
    assert summary["mean_turns_to_success"] == 3
    assert (tmp_path / "battery" / "battery_summary.json").exists()
    assert (tmp_path / "battery" / "battery_manifest.json").exists()
    error_root = (
        tmp_path
        / "battery"
        / "episodes"
        / "harness_libero_spatial_task00_state00_seed102"
    )
    assert (error_root / "episode_error.json").exists()
    assert len(calls) == 4


def test_run_battery_resumes_completed_episodes(tmp_path):
    protocol = _protocol()
    protocol["tasks"] = [protocol["tasks"][1]]  # one state, two seeds
    outcomes = {(5, 0, 101): True, (5, 0, 102): True}
    runner, calls = _fake_runner(outcomes)
    battery_root = tmp_path / "battery"
    run_battery(
        protocol,
        battery_root=battery_root,
        episode_runner=runner,
        condition="harness",
    )
    assert len(calls) == 2
    outcome = run_battery(
        protocol,
        battery_root=battery_root,
        episode_runner=runner,
        condition="harness",
    )
    assert len(calls) == 2  # nothing re-executed
    assert outcome["summary"]["episodes_completed"] == 2


def test_repo_frozen_protocol_file_is_valid():
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "eval_protocol_v1.json"
    )
    protocol = load_protocol(path)
    assert protocol["protocol_name"] == "benchmark_frozen_v1"
    assert protocol["frozen"] is True
    assert 7 in protocol["bootstrap_seeds"]
    assert list(range(101, 111)) == protocol["deployment_seeds"]
    matrix = expand_protocol_matrix(protocol, condition="harness")
    assert len(matrix) == 8 * 10  # eight (task, state) pairs x ten seeds
