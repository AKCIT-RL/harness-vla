import json

import pytest

from harness_vla.eval.libero_memory_lifecycle import (
    bootstrap_memories,
    prepare_deployment,
    promote_global_memory,
)
from harness_vla.planner.phase_policy import build_phase_manifest


def _manifest(bootstrap_budget=8, deployment_budget=4):
    return build_phase_manifest(7, [101, 202], bootstrap_budget, deployment_budget)


def _records(postcondition=True):
    return [{
        "turn": 1,
        "invocation": {"action": "vla_act", "object": "object 1", "mode": "grasp"},
        "primitive": "vla_act",
        "is_contact": True,
        "primitive_postcondition_met": postcondition,
        "termination_reason": "lift_and_grasp_satisfied" if postcondition else "empty_grasp",
        "status": "postcondition_met" if postcondition else "grasp_unverified",
        "object_roles": {"object 1": ["manipulable"]},
        "object_labels": {"object 1": "black bowl"},
        "step_results": [{"task_success": float(postcondition)}],
        "task_success": int(postcondition),
        "episode_status": "completed",
    }]


def _write_episode(tmp_path, *, success=True, records=None):
    trace_path = tmp_path / "trace.jsonl"
    episode_path = tmp_path / "episode.json"
    selected_records = records if records is not None else _records(success)
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in selected_records),
        encoding="utf-8",
    )
    episode_path.write_text(json.dumps({
        "seed": 7,
        "task_success": float(success),
        "instruction": "pick up the black bowl",
        "num_turns": len(selected_records),
    }), encoding="utf-8")
    return trace_path, episode_path


def _bootstrap(tmp_path, *, success=True, records=None):
    trace_path, episode_path = _write_episode(
        tmp_path, success=success, records=records
    )
    task_dir = tmp_path / "task_memory"
    ledger_path = tmp_path / "global_ledger.json"
    result = bootstrap_memories(
        _manifest(),
        seed=7,
        trace_path=str(trace_path),
        episode_result_path=str(episode_path),
        task_memory_dir=str(task_dir),
        global_ledger_path=str(ledger_path),
    )
    return result, task_dir, ledger_path


def test_bootstrap_success_generates_both_memories_and_serializable_audit(tmp_path):
    result, task_dir, ledger_path = _bootstrap(tmp_path)

    assert result.task_memory.accepted is True
    assert (task_dir / "audit.json").is_file()
    assert (task_dir / "commands.jsonl").is_file()
    assert ledger_path.is_file()
    assert result.audit["global_memory"]["counts"]["pending"] == 1
    assert len(result.audit["source"]["trace_sha256"]) == 64
    assert len(result.audit["source"]["episode_result_sha256"]) == 64
    assert result.audit["task_memory"]["audit"]["classification"] == "paper-compatible"
    json.dumps(result.to_dict())


def test_failed_episode_never_promotes_task_memory(tmp_path):
    result, task_dir, ledger_path = _bootstrap(tmp_path, success=False)

    assert result.task_memory.accepted is False
    assert "task_not_successful" in result.task_memory.reasons
    assert not task_dir.exists()
    assert ledger_path.is_file()


def test_explicit_global_promotion_renders_and_tamper_is_rejected(tmp_path):
    result, _, ledger_path = _bootstrap(tmp_path)
    candidate_id = result.audit["global_memory"]["candidate_ids"][0]
    promotion = promote_global_memory(
        str(ledger_path),
        candidate_id=candidate_id,
        semantic_interpretation="Use contact action only after symbolic target grounding.",
    )
    assert promotion["status"] == "promoted"

    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["entries"][0]["reason"] = "manual override"
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="violates promotion policy"):
        prepare_deployment(
            _manifest(), seed=101,
            task_memory_dir=str(tmp_path / "task_memory"),
            global_ledger_path=str(ledger_path),
        )


def test_deployment_context_guards_and_hash_finalizer(tmp_path):
    result, task_dir, ledger_path = _bootstrap(tmp_path)
    promote_global_memory(
        str(ledger_path),
        candidate_id=result.audit["global_memory"]["candidate_ids"][0],
        semantic_interpretation="Ground the symbolic target before contact.",
    )
    session = prepare_deployment(
        _manifest(), seed=101,
        task_memory_dir=str(task_dir), global_ledger_path=str(ledger_path),
    )

    assert "symbolic task structure" in session.context
    assert "Ground the symbolic target before contact." in session.context
    assert "xyz" not in session.context.casefold()
    called = []
    with pytest.raises(ValueError, match="memory writes are forbidden"):
        session.guard_write(lambda: called.append(True))
    assert called == []
    session.guard_budget(4)
    with pytest.raises(ValueError, match="exceed"):
        session.guard_budget(5)
    final = session.finalize()
    assert final["hashes_before"] == final["hashes_after"]
    assert final["unchanged"] is True
    json.dumps(final)


def test_deployment_validates_relative_bootstrap_trace_after_cwd_change(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    trace_path, episode_path = _write_episode(tmp_path)
    result = bootstrap_memories(
        _manifest(), seed=7,
        trace_path=trace_path.name,
        episode_result_path=episode_path.name,
        task_memory_dir="task_memory",
        global_ledger_path="global_ledger.json",
    )
    promote_global_memory(
        "global_ledger.json",
        candidate_id=result.audit["global_memory"]["candidate_ids"][0],
        semantic_interpretation="Ground the symbolic target before contact.",
    )

    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    session = prepare_deployment(
        _manifest(), seed=101,
        task_memory_dir=str(tmp_path / "task_memory"),
        global_ledger_path=str(tmp_path / "global_ledger.json"),
    )

    assert session.seed == 101


def test_deployment_rejects_non_held_out_seed_and_detects_tamper(tmp_path):
    result, task_dir, ledger_path = _bootstrap(tmp_path)
    promote_global_memory(
        str(ledger_path),
        candidate_id=result.audit["global_memory"]["candidate_ids"][0],
        semantic_interpretation="Re-ground before a contact retry.",
    )
    with pytest.raises(ValueError, match="does not belong"):
        prepare_deployment(
            _manifest(), seed=7,
            task_memory_dir=str(task_dir), global_ledger_path=str(ledger_path),
        )

    session = prepare_deployment(
        _manifest(), seed=202,
        task_memory_dir=str(task_dir), global_ledger_path=str(ledger_path),
    )
    (task_dir / "commands.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="deployment memory changed"):
        session.finalize()


def test_bootstrap_seed_and_budget_guards_run_before_writes(tmp_path):
    records = _records() * 2
    records[1] = dict(records[1], turn=2)
    trace_path, episode_path = _write_episode(tmp_path, records=records)
    task_dir = tmp_path / "task_memory"
    ledger_path = tmp_path / "ledger.json"

    with pytest.raises(ValueError, match="does not belong"):
        bootstrap_memories(
            _manifest(), seed=101, trace_path=str(trace_path),
            episode_result_path=str(episode_path), task_memory_dir=str(task_dir),
            global_ledger_path=str(ledger_path),
        )
    assert not task_dir.exists() and not ledger_path.exists()

    with pytest.raises(ValueError, match="exceeds phase budget"):
        bootstrap_memories(
            _manifest(bootstrap_budget=1, deployment_budget=1), seed=7,
            trace_path=str(trace_path), episode_result_path=str(episode_path),
            task_memory_dir=str(task_dir), global_ledger_path=str(ledger_path),
        )
    assert not task_dir.exists() and not ledger_path.exists()