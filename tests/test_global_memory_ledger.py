import json

import pytest

from harness_vla.memory.global_memory import (
    GlobalMemory,
    GlobalMemoryEvidence,
    GlobalMemoryLedger,
    evidence_from_completed_trace,
    evidence_from_completed_trace_bytes,
)


def _write_trace(path, records, *, final_newline=True):
    for record in records:
        if record.get("primitive_postcondition_met") in {True, False}:
            record.setdefault("task_success", 0)
            record.setdefault("episode_status", "completed")
    payload = "\n".join(json.dumps(record) for record in records)
    if final_newline:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def test_completed_trace_produces_trace_backed_failure_candidate(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    _write_trace(trace_path, [{
        "turn": 3,
        "primitive": "vla_act",
        "termination_reason": "empty_grasp",
        "primitive_postcondition_met": False,
    }])

    evidence = evidence_from_completed_trace(trace_path, run_status="completed")

    assert len(evidence) == 1
    assert evidence[0].kind == "failure_model"
    assert evidence[0].turns == (3,)
    assert len(evidence[0].trace_sha256) == 64


def test_reprocessing_same_trace_is_idempotent(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    ledger_path = tmp_path / "ledger.json"
    _write_trace(trace_path, [{
        "turn": 1,
        "primitive": "move_to",
        "termination_reason": "postcondition_met",
        "primitive_postcondition_met": True,
    }])
    candidate = evidence_from_completed_trace(
        trace_path, run_status="completed"
    )[0]
    ledger = GlobalMemoryLedger()

    assert ledger.add(candidate) is True
    assert ledger.add(candidate) is False
    ledger.save(ledger_path)
    loaded = GlobalMemoryLedger.load(ledger_path)

    assert [item.to_dict() for item in loaded.evidence] == [candidate.to_dict()]


def test_incomplete_run_or_trace_cannot_produce_evidence(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    _write_trace(trace_path, [{"turn": 1}], final_newline=False)

    with pytest.raises(ValueError, match="completed run"):
        evidence_from_completed_trace(trace_path, run_status="incomplete")
    with pytest.raises(ValueError, match="incomplete trace"):
        evidence_from_completed_trace(trace_path, run_status="completed")


def test_claimed_completed_run_requires_closed_episode_fields(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(json.dumps({
        "turn": 1,
        "primitive": "move_to",
        "primitive_postcondition_met": True,
    }) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="completed episode records"):
        evidence_from_completed_trace(trace_path, run_status="completed")


def test_evidence_requires_turn_provenance():
    with pytest.raises(ValueError, match="positive trace turns"):
        GlobalMemoryEvidence(
            kind="failure_model",
            text="Observed failure",
            trace_sha256="a" * 64,
            turns=(),
        )


def test_boolean_observation_remains_pending(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    _write_trace(trace_path, [{
        "turn": 2,
        "primitive": "release",
        "termination_reason": "postcondition_met",
        "primitive_postcondition_met": True,
    }])
    complete = evidence_from_completed_trace(
        trace_path, run_status="completed"
    )[0]
    incomplete = GlobalMemoryEvidence(
        kind="success_rule",
        text="Observed success",
        trace_sha256="a" * 64,
        turns=(1,),
    )
    ledger = GlobalMemoryLedger()

    ledger.add(complete)
    ledger.add(incomplete)

    assert ledger.decision_for(complete).status == "pending"
    assert ledger.decision_for(complete).kind == "success_rule"
    assert ledger.decision_for(incomplete).status == "rejected"


def test_global_memory_does_not_render_raw_boolean_candidates(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    ledger_path = tmp_path / "ledger.json"
    _write_trace(trace_path, [
        {
            "turn": 1,
            "primitive": "move_to",
            "termination_reason": "postcondition_met",
            "primitive_postcondition_met": True,
        },
        {
            "turn": 2,
            "primitive": "release",
            "termination_reason": "object_not_released",
            "primitive_postcondition_met": False,
        },
    ])
    ledger = GlobalMemoryLedger()
    for candidate in evidence_from_completed_trace(trace_path, run_status="completed"):
        ledger.add(candidate)
    ledger.save(ledger_path)

    memory = GlobalMemory.from_ledger(ledger_path, include_seed=False)
    rendered = memory.render()

    assert memory.success_rules == []
    assert memory.failure_models == []
    assert "move_to satisfied its postcondition" not in rendered
    assert "release did not satisfy its postcondition" not in rendered


def test_explicit_semantic_promotion_persists_and_renders(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    ledger_path = tmp_path / "ledger.json"
    _write_trace(trace_path, [{
        "turn": 1,
        "primitive": "move_to",
        "termination_reason": "postcondition_met",
        "primitive_postcondition_met": True,
    }])
    ledger = GlobalMemoryLedger()
    candidate = evidence_from_completed_trace(
        trace_path, run_status="completed"
    )[0]
    ledger.add(candidate)

    with pytest.raises(ValueError, match="non-empty"):
        ledger.promote(candidate.identity, semantic_interpretation="  ")
    decision = ledger.promote(
        candidate.identity,
        semantic_interpretation="Stage above the destination before transport.",
    )
    ledger.save(ledger_path)

    loaded = GlobalMemoryLedger.load(ledger_path)
    memory = GlobalMemory.from_ledger(ledger_path, include_seed=False)
    assert loaded.decision_for(candidate).status == "promoted"
    assert decision.semantic_interpretation in memory.render()
    assert candidate.text not in memory.render()


def test_process_trace_is_byte_for_byte_idempotent(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    ledger_path = tmp_path / "ledger.json"
    _write_trace(trace_path, [{
        "turn": 1,
        "primitive": "vla_act",
        "termination_reason": "empty_grasp",
        "primitive_postcondition_met": False,
    }])
    ledger = GlobalMemoryLedger()

    first_audit = ledger.process_trace(
        trace_path, run_status="completed", ledger_path=ledger_path
    )
    first_bytes = ledger_path.read_bytes()
    second_audit = ledger.process_trace(
        trace_path, run_status="completed", ledger_path=ledger_path
    )

    assert ledger_path.read_bytes() == first_bytes
    assert first_audit["counts"] == second_audit["counts"]
    assert first_audit["counts"] == {
        "candidates": 1, "promoted": 0, "rejected": 0, "pending": 1,
    }
    persisted = json.loads(ledger_path.read_text(encoding="utf-8"))
    evidence = persisted["entries"][0]["candidate"]["structured_evidence"]
    assert evidence == {
        "primitive_postcondition_met": False,
        "termination_reason": "empty_grasp",
        "task_success": 0,
        "episode_status": "completed",
    }


def test_tampered_promotion_is_rejected_on_load(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    ledger_path = tmp_path / "ledger.json"
    _write_trace(trace_path, [{
        "turn": 1,
        "primitive": "move_to",
        "primitive_postcondition_met": True,
    }])
    ledger = GlobalMemoryLedger()
    ledger.add(evidence_from_completed_trace(
        trace_path, run_status="completed"
    )[0])
    ledger.save(ledger_path)
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["entries"][0]["status"] = "promoted"
    payload["entries"][0]["reason"] = "manual override"
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="violates promotion policy"):
        GlobalMemoryLedger.load(ledger_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("trace_path", "missing.jsonl", "trace does not exist"),
        ("trace_sha256", "0" * 64, "trace hash mismatch"),
        ("turns", [99], "turn must exist exactly once"),
        ("primitive", "release", "primitive mismatch"),
        (
            "structured_evidence",
            {
                "primitive_postcondition_met": True,
                "termination_reason": "tampered",
                "task_success": 1,
                "episode_status": "completed",
            },
            "structured evidence mismatch",
        ),
    ],
)
def test_tampered_candidate_provenance_fails_closed(
    tmp_path, field, value, message
):
    trace_path = tmp_path / "trace.jsonl"
    ledger_path = tmp_path / "ledger.json"
    _write_trace(trace_path, [{
        "turn": 1,
        "primitive": "move_to",
        "termination_reason": "postcondition_met",
        "primitive_postcondition_met": True,
        "task_success": 1,
        "episode_status": "completed",
    }])
    ledger = GlobalMemoryLedger()
    ledger.add(evidence_from_completed_trace(
        trace_path, run_status="completed"
    )[0])
    ledger.save(ledger_path)
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["entries"][0]["candidate"][field] = value
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        GlobalMemoryLedger.load(ledger_path)


def test_evidence_uses_one_payload_for_records_and_hash(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    _write_trace(trace_path, [{
        "turn": 1,
        "primitive": "release",
        "primitive_postcondition_met": False,
    }])
    payload = trace_path.read_bytes()

    candidate = evidence_from_completed_trace_bytes(
        payload, trace_path=str(trace_path), run_status="completed"
    )[0]

    import hashlib
    assert candidate.trace_sha256 == hashlib.sha256(payload).hexdigest()
    assert candidate.primitive == "release"


def test_multiline_truncated_final_record_is_rejected(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_bytes(
        b'{"turn": 1}\n{\n  "turn": 2,\n  "primitive": "move_to"'
    )

    with pytest.raises(ValueError, match="incomplete trace"):
        evidence_from_completed_trace(trace_path, run_status="completed")


def test_schema_dispatch_rejects_unknown_and_v1_never_promotes(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps({
        "schema_version": 1,
        "evidence": [{
            "kind": "success_rule",
            "text": "Legacy observation",
            "trace_sha256": "a" * 64,
            "turns": [1],
        }],
    }), encoding="utf-8")

    ledger = GlobalMemoryLedger.load(ledger_path)
    assert ledger.decisions[0].status == "rejected"

    ledger_path.write_text(json.dumps({
        "schema_version": 99, "entries": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        GlobalMemoryLedger.load(ledger_path)


def test_deployment_ledger_rejects_writes(tmp_path):
    ledger = GlobalMemoryLedger(read_only=True)
    candidate = GlobalMemoryEvidence(
        kind="failure_model",
        text="Observed failure",
        trace_sha256="a" * 64,
        turns=(1,),
    )

    with pytest.raises(PermissionError, match="read-only"):
        ledger.add(candidate)
    with pytest.raises(PermissionError, match="read-only"):
        ledger.save(tmp_path / "ledger.json")
