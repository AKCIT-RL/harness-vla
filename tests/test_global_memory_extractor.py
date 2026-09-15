"""Automatic Global Memory extractor: digests, candidates, phase-gated promotion."""

import hashlib
import json

import pytest

from harness_vla.envs.partnr.lifecycle import bootstrap_from_traces
from harness_vla.envs.partnr.memory import partnr_global_memory, partnr_global_memory_ledger
from harness_vla.memory.global_memory import GlobalMemoryLedger, validate_trace_hash_provenance
from harness_vla.memory.global_memory_extractor import (
    EpisodeDigest,
    promote_candidates,
    propose_global_memory_candidates,
    summarize_bootstrap_traces,
)
from harness_vla.planner.phase_policy import Phase, build_phase_manifest

PRIMS = ("Navigate", "Pick", "Place", "Open", "Done")


def _step(turn, action, ok, error=None, instances=()):
    return {
        "turn": turn, "action": action, "primitive": action.split("[")[0],
        "ok": ok, "error": error, "instances": set(instances),
    }


def _episode(tmp_path, episode_id, steps, success=True):
    """Synthetic episode whose trace file exists so ledger provenance validates."""
    trace = tmp_path / f"{episode_id}.jsonl"
    payload = "".join(json.dumps(step, default=list) + "\n" for step in steps).encode()
    trace.write_bytes(payload)
    return {
        "episode_id": episode_id, "records": steps, "task_success": success,
        "trace_path": str(trace), "trace_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _passthrough(records):
    return records


# Pick fails (not_reachable) -> Navigate succeeds -> Pick succeeds.
def _recovery_steps(ok_after=True):
    return [
        _step(1, "Pick[<obj:cup>]", False, "not_reachable", instances=["cup_0"]),
        _step(2, "Navigate[<obj:cup>]", ok_after, None if ok_after else "path_blocked"),
        _step(3, "Pick[<obj:cup>]", True),
        _step(4, "Done[]", True),
    ]


def _loop_steps(start=1):
    return [
        _step(start, "Open[<furn:fridge>]", False, "not_reachable", instances=["fridge_3"]),
        _step(start + 1, "Open[<furn:fridge>]", False, "not_reachable"),
        _step(start + 2, "Open[<furn:fridge>]", False, "not_reachable"),
    ]


def _digests(tmp_path, episodes):
    return summarize_bootstrap_traces(episodes, transcript_fn=_passthrough)


def test_digest_captures_failures_recoveries_and_loops(tmp_path):
    episodes = [_episode(tmp_path, "e1", _recovery_steps() + _loop_steps(5), success=False)]
    digest, = _digests(tmp_path, episodes)

    assert isinstance(digest, EpisodeDigest)
    assert digest.task_success is False
    assert digest.failures[0] == ("Pick", "not_reachable", 1)
    assert digest.recoveries[0].failed_primitive == "Pick"
    assert digest.recoveries[0].recovery_primitive == "Navigate"
    assert digest.recoveries[0].recovery_ok is True
    assert digest.recoveries[0].turns == (1, 2)
    assert len(digest.loops) == 1 and digest.loops[0].count == 3
    assert digest.loops[0].turns == (5, 6, 7)
    assert {"cup_0", "fridge_3"} <= set(digest.instances)
    assert json.dumps(digest.to_dict())


def test_rule_based_support_threshold_and_provenance(tmp_path):
    episodes = [
        _episode(tmp_path, "e1", _recovery_steps()),
        _episode(tmp_path, "e2", _recovery_steps()),
        _episode(tmp_path, "e3", _loop_steps(), success=False),
    ]
    digests = _digests(tmp_path, episodes)

    candidates = propose_global_memory_candidates(digests, min_support=2, primitive_names=PRIMS)
    assert len(candidates) == 1
    recovery, = candidates
    assert recovery.source == "rule"
    assert recovery.kind == "failure_model"
    assert recovery.primitive == "Pick"
    assert recovery.evidence_episode_ids == ("e1", "e2")
    assert recovery.support == 2 and recovery.success_rate == 1.0
    assert recovery.evidence_turns == {"e1": (1, 2), "e2": (1, 2)}
    assert "Navigate" in recovery.text and "cup" not in recovery.text.lower()

    assert propose_global_memory_candidates(digests, min_support=3, primitive_names=PRIMS) == []
    assert propose_global_memory_candidates(
        digests[:1], min_support=1, primitive_names=PRIMS
    )[0].support == 1


def test_promotion_requires_support_and_success_rate(tmp_path):
    episodes = [
        _episode(tmp_path, "e1", _recovery_steps(ok_after=False), success=False),
        _episode(tmp_path, "e2", _recovery_steps(ok_after=False), success=False),
        _episode(tmp_path, "e3", _recovery_steps()),
    ]
    digests = _digests(tmp_path, episodes)
    candidates = [
        c for c in propose_global_memory_candidates(digests, min_support=2, primitive_names=PRIMS)
        if c.primitive == "Pick"
    ]
    assert len(candidates) == 1
    assert candidates[0].success_rate == pytest.approx(1 / 3)
    ledger = GlobalMemoryLedger(primitive_names=PRIMS)

    report = promote_candidates(
        ledger, candidates, Phase.BOOTSTRAP, digests=digests, min_support=2, min_success_rate=0.5
    )
    assert report.promoted == [] and "success rate" in report.rejected[0][1]
    assert ledger.audit()["counts"]["candidates"] == 0

    report = promote_candidates(
        ledger, candidates, Phase.BOOTSTRAP, digests=digests, min_support=2, min_success_rate=0.3
    )
    assert report.promoted == [candidates[0].text]
    assert report.evidence_added == 3
    assert ledger.audit()["counts"] == {"candidates": 3, "promoted": 3, "rejected": 0, "pending": 0}
    evidence = ledger.decisions[0].candidate
    assert evidence.trace_path and len(evidence.trace_sha256) == 64
    assert evidence.structured_evidence["extraction"] == "rule"
    assert evidence.structured_evidence["promotion_reason"].startswith("auto:rule support=3")
    assert {d.candidate.structured_evidence["episode_id"] for d in ledger.decisions} == {"e1", "e2", "e3"}

    ledger_path = tmp_path / "ledger.json"
    ledger.save(ledger_path)
    reloaded = GlobalMemoryLedger.load(
        ledger_path, read_only=True, primitive_names=PRIMS,
        provenance_validator=validate_trace_hash_provenance,
    )
    assert reloaded.audit()["counts"]["promoted"] == 3


def test_llm_mode_parses_messy_json_and_drops_bad_evidence(tmp_path):
    episodes = [_episode(tmp_path, "e1", _recovery_steps()), _episode(tmp_path, "e2", _recovery_steps())]
    digests = _digests(tmp_path, episodes)
    calls = []

    def fake_llm(messages):
        calls.append(messages)
        return (
            "Sure! Here are the rules:\n```json\n"
            + json.dumps([
                {"rule": "Before Pick, Navigate to the object so it is reachable.",
                 "kind": "success_rule", "evidence_episode_ids": ["e1", "e2"]},
                {"rule": "Open the fridge before Pick.", "evidence_episode_ids": ["e1", "e9"]},
                {"rule": "Always Navigate to cup_0 first.", "evidence_episode_ids": ["e1", "e2"]},
                {"rule": "No primitive named here.", "evidence_episode_ids": ["e1", "e2"]},
                "garbage",
            ])
            + "\n```\ntrailing text"
        )

    candidates = propose_global_memory_candidates(digests, fake_llm, min_support=2, primitive_names=PRIMS)

    assert len(calls) == 1 and calls[0][0]["role"] == "system"
    assert "e1" in calls[0][1]["content"] and "cup_0" not in calls[0][1]["content"]
    assert [c.text for c in candidates] == ["Before Pick, Navigate to the object so it is reachable."]
    assert candidates[0].source == "llm"
    assert candidates[0].kind == "success_rule"
    assert candidates[0].evidence_episode_ids == ("e1", "e2")

    ledger = GlobalMemoryLedger(primitive_names=PRIMS)
    report = promote_candidates(ledger, candidates, Phase.BOOTSTRAP, digests=digests)
    assert report.promoted == [candidates[0].text]
    assert all(d.candidate.postcondition_met is True for d in ledger.decisions)


def test_llm_success_rule_downgraded_when_evidence_episode_failed(tmp_path):
    episodes = [_episode(tmp_path, "e1", _recovery_steps()), _episode(tmp_path, "e2", _recovery_steps(), success=False)]
    digests = _digests(tmp_path, episodes)
    llm = lambda messages: json.dumps([
        {"rule": "Navigate before Pick.", "kind": "success_rule", "evidence_episode_ids": ["e1", "e2"]}
    ])
    candidate, = propose_global_memory_candidates(digests, llm, primitive_names=PRIMS)
    assert candidate.kind == "failure_model"


def test_rejects_task_specific_leakage(tmp_path):
    episodes = [_episode(tmp_path, "e1", _recovery_steps()), _episode(tmp_path, "e2", _recovery_steps())]
    digests = _digests(tmp_path, episodes)
    llm = lambda messages: json.dumps([
        {"rule": "Navigate to cup_0 before Pick.", "evidence_episode_ids": ["e1", "e2"]},
        {"rule": "Pick from table_13 after Navigate.", "evidence_episode_ids": ["e1", "e2"]},
        {"rule": "Navigate first, then Pick.", "evidence_episode_ids": ["e1", "e2"]},
    ])
    candidates = propose_global_memory_candidates(digests, llm, primitive_names=PRIMS)
    assert [c.text for c in candidates] == ["Navigate first, then Pick."]


def test_promotion_refused_in_deployment_phase(tmp_path):
    episodes = [_episode(tmp_path, "e1", _recovery_steps()), _episode(tmp_path, "e2", _recovery_steps())]
    digests = _digests(tmp_path, episodes)
    candidates = propose_global_memory_candidates(digests, primitive_names=PRIMS)
    assert candidates
    ledger = GlobalMemoryLedger(primitive_names=PRIMS)
    manifest = build_phase_manifest(0, [1, 2], 10, 10)

    for policy in (Phase.DEPLOYMENT, "deployment", manifest.policy_for(Phase.DEPLOYMENT)):
        with pytest.raises(PermissionError, match="deployment"):
            promote_candidates(ledger, candidates, policy, digests=digests)
    assert ledger.audit()["counts"]["candidates"] == 0

    report = promote_candidates(
        ledger, candidates, manifest.policy_for(Phase.BOOTSTRAP), digests=digests
    )
    assert report.promoted


# --------------------------------------------------------------- lifecycle

def _partnr_trace(path, episode_id, actions, success=1.0):
    """PARTNR env JSONL: actions = [(name, args, success_hint, failure_kind)]."""
    state = {"objects": {"cup_0": {}}, "furniture": {"table_1": {}}, "rooms": ["kitchen_1"]}
    records = [{"type": "episode_start", "context": {"episode_id": episode_id, "instruction": "Move the cup to the table"}}]
    for index, (name, args, hint, failure_kind) in enumerate(actions):
        records.append({
            "type": "decision",
            "context": {"instruction": "Move the cup to the table", "decision_index": index,
                        "rooms": ["kitchen_1"], "extras": {"state": state}},
            "decision": {"name": name, "args": args},
        })
        records.append({"type": "result", "result": {"success_hint": hint, "failure_kind": failure_kind}})
    records.append({"type": "episode_end", "task_state_success": success, "summary": {"episode_id": episode_id}})
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return str(path)


def _bootstrap_traces(tmp_path):
    actions = [
        ("Pick", "cup_0", False, "not_reachable"),
        ("Navigate", "cup_0", True, None),
        ("Pick", "cup_0", True, None),
        ("Place", "cup_0, on, table_1, none, none", True, None),
    ]
    return [_partnr_trace(tmp_path / f"trace_{i}.jsonl", f"ep{i}", actions) for i in range(2)]


def test_lifecycle_flag_off_leaves_ledger_unchanged(tmp_path):
    ledger_path = tmp_path / "gm" / "ledger.json"
    audit = bootstrap_from_traces(
        _bootstrap_traces(tmp_path), str(tmp_path / "tm"), str(ledger_path)
    )
    assert audit["auto_extractor"] is None
    assert audit["promoted_rules"] == []
    assert audit["ledger_counts"]["candidates"] == 0
    assert partnr_global_memory(str(ledger_path)).failure_models == []


def test_lifecycle_flag_on_promotes_rule_based_guidance(tmp_path):
    ledger_path = tmp_path / "gm" / "ledger.json"
    audit = bootstrap_from_traces(
        _bootstrap_traces(tmp_path), str(tmp_path / "tm"), str(ledger_path), auto_extract=True
    )
    extracted = audit["auto_extractor"]
    assert extracted["mode"] == "rule" and extracted["digests"] == 2
    assert len(extracted["promoted"]) == 1 and "Pick" in extracted["promoted"][0]
    assert extracted["candidates"][0]["evidence_episode_ids"] == ["ep0", "ep1"]
    assert audit["promoted_rules"] == extracted["promoted"]
    assert audit["ledger_counts"] == {"candidates": 2, "promoted": 2, "rejected": 0, "pending": 0}
    # Reload validates trace provenance (hash) and renders the promoted rule.
    reloaded = partnr_global_memory_ledger(str(ledger_path), read_only=True)
    assert reloaded.audit()["counts"]["promoted"] == 2
    memory = partnr_global_memory(str(ledger_path))
    assert memory.failure_models == extracted["promoted"]
    assert "cup_0" not in memory.render() and "table_1" not in memory.render()
    assert len(memory.provenance_for(extracted["promoted"][0])) == 2


def test_lifecycle_flag_on_with_llm(tmp_path):
    ledger_path = tmp_path / "gm" / "ledger.json"
    llm = lambda messages: json.dumps([
        {"rule": "Navigate to the target before Pick.", "kind": "success_rule",
         "evidence_episode_ids": ["ep0", "ep1"]},
    ])
    audit = bootstrap_from_traces(
        _bootstrap_traces(tmp_path), str(tmp_path / "tm"), str(ledger_path),
        auto_extract=True, auto_extract_llm=llm,
    )
    assert audit["auto_extractor"]["mode"] == "llm"
    assert audit["auto_extractor"]["promoted"] == ["Navigate to the target before Pick."]
    assert partnr_global_memory(str(ledger_path)).success_rules == ["Navigate to the target before Pick."]
