"""Tests for crash-resistant incremental Harness traces."""

import json

import pytest

from harness_vla.trace_io import (
    append_jsonl_record,
    initialize_jsonl,
    load_complete_jsonl,
    resolve_git_commit,
    summarize_trace_records,
    write_json_atomic,
)


def test_incremental_trace_survives_interruption_after_complete_records(tmp_path):
    trace_path = tmp_path / "results" / "trace.jsonl"
    initialize_jsonl(trace_path)
    for turn in range(1, 4):
        append_jsonl_record(trace_path, {"turn": turn, "status": "complete"})

    assert load_complete_jsonl(trace_path) == [
        {"turn": 1, "status": "complete"},
        {"turn": 2, "status": "complete"},
        {"turn": 3, "status": "complete"},
    ]
    assert trace_path.read_bytes().endswith(b"\n")


def test_reconstruction_ignores_only_truncated_final_line(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_bytes(
        b'{"turn": 1}\n{"turn": 2}\n{"turn": 3, "status":'
    )
    assert load_complete_jsonl(trace_path) == [{"turn": 1}, {"turn": 2}]


def test_reconstruction_rejects_corrupt_complete_line(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_bytes(b'{"turn": 1}\nnot-json\n{"turn": 3}')
    with pytest.raises(json.JSONDecodeError):
        load_complete_jsonl(trace_path)


def test_initialize_replaces_stale_trace(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text('{"turn": 99}\n', encoding="utf-8")
    initialize_jsonl(trace_path)
    assert trace_path.read_bytes() == b""


def test_summary_is_reconstructed_only_from_complete_trace_records(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    initialize_jsonl(trace_path)
    append_jsonl_record(trace_path, {
        "turn": 1,
        "primitive": "move_to",
        "status": "postcondition_met",
        "primitive_postcondition_met": True,
        "termination_reason": "postcondition_met",
        "step_results": [{"task_success": 0.0}],
    })
    append_jsonl_record(trace_path, {
        "turn": 2,
        "primitive": "vla_act",
        "status": "grasp_unverified",
        "primitive_postcondition_met": False,
        "termination_reason": "unverified",
        "step_results": [
            {"task_success": 0.0},
            {"task_success": 1.0},
        ],
    })
    with trace_path.open("ab") as trace_file:
        trace_file.write(b'{"turn": 3, "status":')

    summary = summarize_trace_records(load_complete_jsonl(trace_path))
    assert summary == {
        "turn_count": 2,
        "env_step_count": 3,
        "status_counts": {"postcondition_met": 1, "grasp_unverified": 1},
        "primitive_counts": {"move_to": 1, "vla_act": 1},
        "postconditions_met": 1,
        "postconditions_failed": 1,
        "termination_reasons": {"postcondition_met": 1, "unverified": 1},
        "task_success": 1.0,
    }


def test_atomic_json_never_leaves_temporary_file_after_success(tmp_path):
    output_path = tmp_path / "run_manifest.json"
    write_json_atomic(output_path, {"status": "running"})
    write_json_atomic(output_path, {"status": "completed"})
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "status": "completed"
    }
    assert not (tmp_path / "run_manifest.json.tmp").exists()


def test_git_commit_resolves_from_loose_ref_without_git_executable(tmp_path):
    git_path = tmp_path / ".git"
    reference_path = git_path / "refs" / "heads" / "main"
    reference_path.parent.mkdir(parents=True)
    (git_path / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    reference_path.write_text("abc123\n", encoding="utf-8")
    nested = tmp_path / "EmbodiedBench" / "running"
    nested.mkdir(parents=True)
    assert resolve_git_commit(nested) == "abc123"