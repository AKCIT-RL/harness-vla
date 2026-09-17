"""Offline bootstrap from env queue traces (central + decentral layouts)."""

import json

from harness_vla.envs.partnr.lifecycle import (
    bootstrap_from_queue,
    collect_queue_episodes,
    deploy_overrides,
    labeled_transcript,
    main,
)
from harness_vla.envs.partnr.memory import load_partnr_task_memory

STATE = {
    "rooms": ["kitchen_1", "living_room_1"],
    "furniture": {"table_1": {"room": "kitchen_1"}, "sofa_3": {"room": "living_room_1"}},
    "objects": {"apple_0": {"room": "kitchen_1", "relation": "on", "furniture": "table_1"}},
    "agents": {},
}
INSTRUCTION = "Put the apple on the sofa"


def _context(index, uid=0):
    return {"episode_id": "4", "instruction": INSTRUCTION, "agent_uid": uid,
            "decision_index": index, "rooms": STATE["rooms"], "extras": {"state": STATE}}


def _decision(name, args):
    return {"name": name, "args": args, "rationale": "", "error_kind": None}


def _result(name, args, ok, uid=None):
    return {"tool": name, "args": args, "response": "ok" if ok else "Unexpected failure!",
            "success_hint": ok, "failure_kind": None if ok else "skill", "agent_uid": uid}


def central_records(success_first_pick=True):
    records = [{"type": "episode_start", "context": _context(0)}]
    plan = [
        ("agent_0", "Navigate", "table_1", True), ("agent_1", "Navigate", "sofa_3", True),
        ("agent_0", "Pick", "apple_0", success_first_pick),
    ]
    if not success_first_pick:
        plan.append(("agent_0", "Pick", "apple_0", True))
    plan += [("agent_0", "Navigate", "sofa_3", True), ("agent_0", "Place", "apple_0, on, sofa_3, None, None", True)]
    for index, (label, name, args, ok) in enumerate(plan):
        records.append({"type": "decision", "context": _context(index),
                        "decisions": {label: _decision(name, args), "all": None}})
        records.append({"type": "result", "agent": label, "result": _result(name, args, ok)})
    records.append({"type": "decision", "context": _context(len(plan)),
                    "decisions": {"all": {"rationale": "done"}}})
    records.append({"type": "episode_end", "summary": {"episode_id": "4", "num_decisions": len(plan) + 1}})
    return records


def bridge_records(uid, plan):
    records = [{"type": "episode_start", "context": _context(0, uid)}]
    for index, (name, args, ok) in enumerate(plan):
        records.append({"type": "decision", "context": _context(index, uid), "decision": _decision(name, args)})
        records.append({"type": "result", "result": _result(name, args, ok, uid)})
    records.append({"type": "episode_end", "summary": {"episode_id": "4", "num_decisions": len(plan)}})
    return records


def write_job(queue, job_id, success, traces):
    job = queue / "jobs" / job_id
    (job / "results" / "val_mini.json.gz" / "stats").mkdir(parents=True)
    (job / "results" / "val_mini.json.gz" / "stats" / "4.json").write_text(json.dumps({
        "success": True,
        "stats": json.dumps({"task_state_success": success, "task_percent_complete": success}),
    }))
    (job / "job_state.json").write_text(json.dumps({"status": "complete"}))
    for uid, records in traces.items():
        path = job / "results" / "harness" / f"agent_{uid}" / "decisions-4.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_labeled_transcript_handles_centralized_and_bridge_layouts():
    history = labeled_transcript(central_records(success_first_pick=False))
    assert [(h["agent"], h["name"], h["ok"]) for h in history[:4]] == [
        ("agent_0", "Navigate", True), ("agent_1", "Navigate", True),
        ("agent_0", "Pick", False), ("agent_0", "Pick", True),
    ]
    assert history[2]["failure_kind"] == "skill"
    solo = labeled_transcript(bridge_records(1, [("Navigate", "sofa_3", True), ("Wait", "", True)]))
    assert [h["agent"] for h in solo] == ["agent_1", "agent_1"]
    assert all(h["status"] == "success" for h in solo)


def test_labeled_transcript_drops_masked_actions_and_matches_lagged_results():
    # Mirrors a real centralized trace: results lag behind decision boundaries,
    # actions issued to busy agents are masked (never executed).
    def dec(index, decisions):
        return {"type": "decision", "context": _context(index),
                "decisions": {k: (_decision(*v) if v else None) for k, v in decisions.items()}}

    def res(label, name, args, ok):
        return {"type": "result", "agent": label, "result": _result(name, args, ok)}

    records = [
        {"type": "episode_start", "context": _context(0)},
        dec(0, {"agent_0": ("Navigate", "table_1"), "agent_1": ("Navigate", "apple_0")}),
        res("agent_0", "Navigate", "table_1", True),
        dec(1, {"agent_0": ("Wait", ""), "agent_1": ("Pick", "apple_0")}),
        {"type": "masked_busy_actions", "dropped": ["agent_1"]},
        res("agent_1", "Navigate", "apple_0", True),
        dec(2, {"agent_0": ("Wait", ""), "agent_1": ("Pick", "apple_0")}),
        {"type": "masked_busy_actions", "dropped": ["agent_0"]},
        res("agent_1", "Pick", "apple_0", True),
        dec(3, {"agent_0": None, "agent_1": ("Place", "apple_0,on,sofa_3,None,None")}),
        res("agent_0", "Wait", "", True),
        res("agent_1", "Place", "apple_0, on, sofa_3, None, None", True),
        {"type": "episode_end", "summary": {"episode_id": "4"}},
    ]
    history = labeled_transcript(records)
    assert [(h["agent"], h["name"], h["ok"]) for h in history] == [
        ("agent_0", "Navigate", True), ("agent_1", "Navigate", True),
        ("agent_0", "Wait", True), ("agent_1", "Pick", True), ("agent_1", "Place", True),
    ]


def test_bootstrap_central_writes_labeled_task_memory_and_global_memory(tmp_path):
    queue = tmp_path / "queue"
    write_job(queue, "boot--r1--ep4--central-vla-boot", 1.0, {-1: central_records(False)})
    write_job(queue, "boot--r2--ep4--central-vla-boot", 1.0, {-1: central_records(True)})
    write_job(queue, "boot--r3--ep4--central-vla-boot", 0.0, {-1: central_records(False)})
    write_job(queue, "boot--r1--ep4--other", 1.0, {-1: central_records(True)})

    episodes = collect_queue_episodes(str(queue), "central-vla-boot")
    assert [e["task_state_success"] for e in episodes] == [1.0, 1.0, 0.0]

    out = tmp_path / "memory" / "central"
    audit = bootstrap_from_queue(str(queue), "central-vla-boot", str(out), mode="central")
    assert audit["successful_episodes"] == 2
    entry = load_partnr_task_memory(str(out / "task_memory"), INSTRUCTION)
    assert entry["support"] == 2
    # Shortest successful skeleton, labeled and instance-free.
    assert entry["commands"] == [
        "agent_0: Navigate[<furn:table>]", "agent_1: Navigate[<furn:sofa>]",
        "agent_0: Pick[<obj:apple>]", "agent_0: Navigate[<furn:sofa>]",
        "agent_0: Place[<obj:apple>, on, <furn:sofa>, None, None]",
    ]
    assert (out / "global_memory" / "global_memory.json").is_file()
    assert (out / "bootstrap_audit.json").is_file()
    assert audit["deployment"]["zero_shot"] is False and audit["deployment"]["phase"] == "deployment"

    overrides = deploy_overrides(str(out), "central")
    assert "evaluation.planner.plan_config.harness.zero_shot=False" in overrides
    assert f"evaluation.planner.plan_config.harness.task_memory_dir={out / 'task_memory'}" in overrides
    assert any(o.endswith("global_memory_path=" + str(out / "global_memory" / "global_memory.json")) for o in overrides)


def test_bootstrap_decentral_builds_one_memory_per_agent(tmp_path):
    queue = tmp_path / "queue"
    robot = [("Navigate", "table_1", True), ("Pick", "apple_0", True),
             ("Navigate", "sofa_3", True), ("Place", "apple_0, on, sofa_3, None, None", True)]
    human = [("Navigate", "sofa_3", True), ("Wait", "", True)]
    write_job(queue, "boot--r1--ep4--decentral-vla-boot", 1.0,
              {0: bridge_records(0, robot), 1: bridge_records(1, human)})

    out = tmp_path / "memory" / "decentral"
    audit = bootstrap_from_queue(str(queue), "decentral-vla-boot", str(out), mode="decentral")
    assert [b["label"] for b in audit["brains"]] == ["agent_0", "agent_1"]
    entry_0 = load_partnr_task_memory(str(out / "task_memory" / "agent_0"), INSTRUCTION)
    assert entry_0["commands"] == [
        "agent_0: Navigate[<furn:table>]", "agent_0: Pick[<obj:apple>]",
        "agent_0: Navigate[<furn:sofa>]", "agent_0: Place[<obj:apple>, on, <furn:sofa>, None, None]",
    ]
    entry_1 = load_partnr_task_memory(str(out / "task_memory" / "agent_1"), INSTRUCTION)
    assert entry_1["commands"] == ["agent_1: Navigate[<furn:sofa>]"]  # Wait dropped
    assert (out / "global_memory" / "agent_0" / "global_memory.json").is_file()
    assert (out / "global_memory" / "agent_1" / "global_memory.json").is_file()

    overrides = deploy_overrides(str(out), "decentral")
    assert f"evaluation.agents.agent_1.planner.plan_config.harness.global_memory_path={out / 'global_memory'}" in overrides

    # Non-empty output is protected unless forced.
    try:
        bootstrap_from_queue(str(queue), "decentral-vla-boot", str(out), mode="decentral")
    except FileExistsError:
        pass
    else:
        raise AssertionError("non-empty --out was overwritten silently")
    bootstrap_from_queue(str(queue), "decentral-vla-boot", str(out), mode="decentral", force=True)


def test_cli_bootstrap_and_deploy_config(tmp_path, capsys):
    queue = tmp_path / "queue"
    write_job(queue, "boot--r1--ep4--central-vla-boot", 1.0, {-1: central_records(True)})
    out = tmp_path / "mem"
    assert main(["bootstrap", "--queue", str(queue), "--condition", "central-vla-boot",
                 "--out", str(out), "--mode", "central"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["successful_episodes"] == 1 and summary["brains"][0]["label"] == "team"
    assert main(["deploy-config", "--memory", str(out), "--mode", "central", "--json"]) == 0
    overrides = json.loads(capsys.readouterr().out)
    assert "evaluation.planner.plan_config.harness.phase=deployment" in overrides
