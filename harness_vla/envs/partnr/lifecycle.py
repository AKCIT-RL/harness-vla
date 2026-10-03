"""Offline bootstrap -> frozen deployment lifecycle for the PARTNR controllers.

Paper-confirmed protocol: memories are written only from the bootstrap (dev)
split; the deployment phase loads them read-only (``phase_policy.Phase``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from harness_vla.envs.partnr.memory import (
    ChatFn,
    _vocabulary,
    build_partnr_task_memory,
    episode_id_of,
    episode_instruction,
    episode_success,
    extract_global_memory_candidates,
    load_episode_trace,
    partnr_global_memory,
    partnr_global_memory_ledger,
    promote_candidates,
    steps_from_transcript,
    symbolic_steps,
    write_partnr_task_memory,
)
from harness_vla.envs.partnr.primitives import active_primitive_names, format_action
from harness_vla.memory import global_memory_extractor as extractor
from harness_vla.planner.phase_policy import Phase
from harness_vla.trace_io import write_json_atomic

GLOBAL_MEMORY_FILENAME = "global_memory.json"


def _apply_success_overrides(episodes, success_overrides):
    overrides = dict(success_overrides or {})
    for episode in episodes:
        episode["episode_id"] = episode_id_of(episode["records"])
        for key in (episode["episode_id"], episode.get("path", "")):
            if key and key in overrides:
                episode["task_state_success"] = float(bool(overrides[key]))
    return episodes


def auto_extract_global_memory(
    episodes,
    ledger,
    *,
    llm: Optional[extractor.LlmFn] = None,
    min_support: int = 2,
    min_success_rate: float = 0.5,
    transcript_fn=symbolic_steps,
) -> Dict[str, Any]:
    """Run the automatic extractor + phase-gated promotion on bootstrap episodes."""
    digests = extractor.summarize_bootstrap_traces(
        episodes, transcript_fn=transcript_fn, success_fn=episode_success
    )
    candidates = extractor.propose_global_memory_candidates(
        digests, llm, min_support=min_support, primitive_names=active_primitive_names()
    )
    report = extractor.promote_candidates(
        ledger,
        candidates,
        Phase.BOOTSTRAP,
        digests=digests,
        min_support=min_support,
        min_success_rate=min_success_rate,
    )
    audit = report.to_dict()
    audit["mode"] = "llm" if llm is not None else "rule"
    audit["digests"] = len(digests)
    audit["candidates"] = [candidate.to_dict() for candidate in candidates]
    return audit


def bootstrap_from_traces(
    trace_paths: Iterable[str],
    task_memory_dir: str,
    global_memory_ledger_path: Optional[str],
    chat_fn: Optional[ChatFn] = None,
    min_support: int = 2,
    success_overrides: Optional[Mapping[str, bool]] = None,
    *,
    auto_extract: bool = False,
    auto_extract_llm: Optional[extractor.LlmFn] = None,
    min_success_rate: float = 0.5,
) -> Dict[str, Any]:
    """Build Task Memory (and optionally Global Memory) from bootstrap traces.

    Returns an audit dict: task keys written, ledger counts, promoted rules
    and the rendered Global Memory path (``<ledger dir>/global_memory.json``).
    With ``auto_extract=True`` the automatic extractor
    (``memory.global_memory_extractor``; rule-based, or LLM-based when
    ``auto_extract_llm`` is given) also runs and its promotions are recorded
    under ``audit["auto_extractor"]``. Default OFF: behaviour unchanged.
    """
    episodes = [load_episode_trace(path) for path in trace_paths]
    task_memory = build_partnr_task_memory(episodes, success_overrides=success_overrides)
    written = write_partnr_task_memory(task_memory, task_memory_dir)
    audit: Dict[str, Any] = {
        "phase": Phase.BOOTSTRAP.value,
        "episodes": len(episodes),
        "task_memory_dir": str(task_memory_dir),
        "task_keys": sorted(task_memory),
        "task_memory_files": written,
        "global_memory_ledger": None,
        "global_memory_path": None,
        "candidates_added": 0,
        "promoted_rules": [],
        "auto_extractor": None,
    }
    if not global_memory_ledger_path:
        return audit
    ledger = partnr_global_memory_ledger(global_memory_ledger_path, read_only=False)
    if chat_fn is not None:
        candidates = extract_global_memory_candidates(
            episodes, chat_fn, success_overrides=success_overrides
        )
        audit["candidates_added"] = sum(1 for candidate in candidates if ledger.add(candidate))
    audit["promoted_rules"] = promote_candidates(ledger, min_support=min_support)
    if auto_extract:
        audit["auto_extractor"] = auto_extract_global_memory(
            _apply_success_overrides(episodes, success_overrides),
            ledger,
            llm=auto_extract_llm,
            min_support=min_support,
            min_success_rate=min_success_rate,
        )
        audit["promoted_rules"] = list(audit["promoted_rules"]) + audit["auto_extractor"]["promoted"]
    ledger.save(global_memory_ledger_path)
    audit["global_memory_ledger"] = str(global_memory_ledger_path)
    audit["ledger_counts"] = ledger.audit()["counts"]
    rendered_path = Path(global_memory_ledger_path).parent / GLOBAL_MEMORY_FILENAME
    partnr_global_memory(global_memory_ledger_path).save(str(rendered_path))
    audit["global_memory_path"] = str(rendered_path)
    return audit


def deployment_config(task_memory_dir: Optional[str], global_memory_path: Optional[str]) -> Dict[str, Any]:
    """Adapter kwargs for the frozen deployment phase (memories read-only)."""
    return {
        "phase": Phase.DEPLOYMENT.value,
        "zero_shot": False,
        "task_memory_dir": str(task_memory_dir) if task_memory_dir else None,
        "global_memory_path": str(global_memory_path) if global_memory_path else None,
        "max_history": 0,
    }


# ------------------------------------------------- queue traces -> memories
#
# Offline bootstrap for the Harness VLA controllers from an env queue run
# (``partnr_harness.queue``): official ``task_state_success`` labels the
# episodes, the decision JSONL provides the transcripts. Two layouts:
#
# * central   -> one team memory: ``<out>/task_memory/`` (labeled skeletons,
#   ``agent_0: Pick[<obj:apple>]``) and ``<out>/global_memory/global_memory.json``;
# * decentral -> one memory per agent: ``<out>/task_memory/<label>/`` and
#   ``<out>/global_memory/<label>/global_memory.json`` built from that agent's
#   own transcript only (matches ``HarnessVlaDecentralized`` path rules).

LEDGER_FILENAME = "global_memory_ledger.json"


def labeled_transcript(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Env JSONL (bridge or centralized layout) -> labeled history records.

    Output items mirror ``HarnessVlaCentralized.history``: ``{decision_index,
    agent, name, args, action, status, ok, error_kind, vocab}`` so the same
    skeleton/promotion helpers apply to offline and in-process bootstrap.
    """
    from harness_vla.envs.partnr.agent import agent_label  # local: avoid cycle

    history: List[Dict[str, Any]] = []
    # Decisions awaiting a result, per label, in issue order (results for one
    # agent arrive in order, but may lag several decision boundaries behind).
    pending: Dict[str, List[Dict[str, Any]]] = {}
    last_issued: Dict[str, Dict[str, Any]] = {}

    def add(label: str, decision: Optional[Mapping], context: Mapping) -> None:
        decision = decision or {}
        state = (context.get("extras") or {}).get("state") or {}
        objects, furniture, rooms = _vocabulary(state)
        rooms |= set(context.get("rooms") or [])
        name = decision.get("name")
        if name is None:
            name, args = "Done", ""
        else:
            args = decision.get("args") or ""
        record = {
            "decision_index": context.get("decision_index"),
            "agent": label,
            "name": name,
            "args": args,
            "action": format_action(name, args),
            "error_kind": decision.get("error_kind"),
            "status": None,
            "ok": name == "Done",
            "vocab": {"objects": objects, "furniture": furniture, "rooms": rooms},
        }
        history.append(record)
        last_issued[label] = record
        if name != "Done":
            pending.setdefault(label, []).append(record)

    def matches(record: Mapping, result: Mapping) -> bool:
        tool = result.get("tool")
        if tool is None:
            return True
        if str(tool) != str(record["name"]):
            return False
        return _norm_args(result.get("args")) == _norm_args(record["args"])

    for record in records:
        kind = record.get("type")
        if kind == "decision":
            context = record.get("context") or {}
            if "decisions" in record:  # centralized: {label: decision|None, "all": Done}
                for label, decision in (record.get("decisions") or {}).items():
                    if label == "all" or decision is None:
                        continue
                    add(str(label), decision, context)
            else:  # bridge: one decision for context.agent_uid
                add(agent_label(context.get("agent_uid")), record.get("decision"), context)
        elif kind == "masked_busy_actions":
            # The env dropped these (issued to a busy agent): not executed.
            for label in record.get("dropped") or []:
                dropped = last_issued.pop(str(label), None)
                if dropped is None:
                    continue
                if dropped in history:
                    history.remove(dropped)
                queue = pending.get(str(label)) or []
                if dropped in queue:
                    queue.remove(dropped)
        elif kind == "result":
            result = record.get("result") or {}
            label = record.get("agent")
            if label is None:
                uid = result.get("agent_uid")
                if uid is not None:
                    label = agent_label(uid)
                else:
                    label = next((k for k, q in pending.items() if q), None)
            queue = pending.get(str(label)) or []
            target = next((item for item in queue if matches(item, result)), None)
            if target is None:
                continue
            queue.remove(target)
            target["status"] = "success" if result.get("success_hint") else "failure"
            target["ok"] = bool(result.get("success_hint"))
            target["failure_kind"] = result.get("failure_kind")
    return history


def _norm_args(args: Any) -> str:
    return ",".join(part.strip() for part in str(args or "").split(","))


def _job_dirs(queue_root: Path, condition: str) -> List[Path]:
    return sorted(
        path for path in (queue_root / "jobs").glob(f"*--{condition}")
        if path.is_dir()
    )


def _official_success(job_dir: Path) -> Optional[float]:
    stats_files = list(job_dir.glob("results/**/stats/*.json"))
    if len(stats_files) != 1:
        return None
    try:
        payload = json.loads(stats_files[0].read_text(encoding="utf-8"))
        stats = json.loads(payload["stats"])
        return float(stats["task_state_success"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def collect_queue_episodes(queue_root: str, condition: str) -> List[Dict[str, Any]]:
    """Completed jobs of ``condition`` -> episodes ``{job_id, episode_id,
    instruction, task_state_success, traces: {label|"team": records}}``."""
    from harness_vla.envs.partnr.agent import agent_label  # local: avoid cycle

    episodes: List[Dict[str, Any]] = []
    for job_dir in _job_dirs(Path(queue_root), condition):
        state = {}
        state_path = job_dir / "job_state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") != "complete":
            continue
        success = _official_success(job_dir)
        traces: Dict[str, Dict[str, Any]] = {}
        for jsonl_path in sorted(job_dir.glob("results/harness/agent_*/decisions-*.jsonl")):
            trace = load_episode_trace(str(jsonl_path))
            uid = jsonl_path.parent.name.split("_", 1)[1]
            label = "team" if uid == "-1" else agent_label(int(uid))
            traces[label] = trace
        if success is None or not traces:
            continue
        first = next(iter(traces.values()))
        episodes.append({
            "job_id": job_dir.name,
            "episode_id": episode_id_of(first["records"]),
            "instruction": episode_instruction(first["records"]),
            "task_state_success": success,
            "traces": traces,
        })
    return episodes


def _bootstrap_memory(
    out_dir: Path,
    episodes: Sequence[Mapping[str, Any]],
    *,
    own_label: Optional[str],
    trace_key: str,
    min_support: int,
    min_success_rate: float,
    llm: Optional[extractor.LlmFn] = None,
) -> Dict[str, Any]:
    """Task Memory (successful episodes) + Global Memory for one brain.

    Global Memory rules come from the rule-based extractor, or from ``llm``
    (one message list -> JSON list) when given.
    """
    from harness_vla.envs.partnr.agent import promote_labeled_transcript  # local: avoid cycle

    task_dir = out_dir / "task_memory" / (own_label or "")
    global_dir = out_dir / "global_memory" / (own_label or "")
    task_dir.mkdir(parents=True, exist_ok=True)
    global_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    extractor_episodes = []
    for episode in episodes:
        trace = episode["traces"].get(trace_key)
        if trace is None:
            continue
        history = labeled_transcript(trace["records"])
        if own_label is not None:
            history = [step for step in history if step.get("agent") == own_label]
        extractor_episodes.append({
            "path": trace["path"],
            "trace_sha256": trace.get("trace_sha256", ""),
            "episode_id": episode["episode_id"],
            "task_state_success": episode["task_state_success"],
            # The extractor only sees the (possibly agent-filtered) history.
            "records": history,
        })
        if float(episode["task_state_success"]) != 1.0:
            continue
        path = promote_labeled_transcript(
            str(task_dir),
            episode["instruction"],
            history,
            source={"episode_id": episode["episode_id"], "job_id": episode["job_id"],
                    "label_source": "official_stats"},
        )
        if path:
            written.append(path)
    ledger_path = global_dir / LEDGER_FILENAME
    ledger = partnr_global_memory_ledger(str(ledger_path), read_only=False)
    extractor_audit = auto_extract_global_memory(
        extractor_episodes,
        ledger,
        llm=llm,
        transcript_fn=steps_from_transcript,
        min_support=min_support,
        min_success_rate=min_success_rate,
    )
    ledger.save(str(ledger_path))
    rendered = global_dir / GLOBAL_MEMORY_FILENAME
    partnr_global_memory(str(ledger_path)).save(str(rendered))
    return {
        "label": own_label or "team",
        "task_memory_dir": str(task_dir),
        "task_memory_files": sorted(set(written)),
        "global_memory_ledger": str(ledger_path),
        "global_memory_path": str(rendered),
        "promoted_rules": extractor_audit["promoted"],
        "auto_extractor": extractor_audit,
    }


def bootstrap_from_queue(
    queue_root: str,
    condition: str,
    out_dir: str,
    *,
    mode: str,
    min_support: int = 2,
    min_success_rate: float = 0.5,
    force: bool = False,
) -> Dict[str, Any]:
    """Offline bootstrap of a queue condition into ``out_dir`` (see module notes)."""
    if mode not in ("central", "decentral"):
        raise ValueError("mode must be 'central' or 'decentral'")
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        if not force:
            raise FileExistsError(f"{out} is not empty (use force=True to rebuild)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    episodes = collect_queue_episodes(queue_root, condition)
    audit: Dict[str, Any] = {
        "phase": Phase.BOOTSTRAP.value,
        "mode": mode,
        "queue_root": str(queue_root),
        "condition": condition,
        "episodes": [
            {"job_id": e["job_id"], "episode_id": e["episode_id"],
             "task_state_success": e["task_state_success"], "traces": sorted(e["traces"])}
            for e in episodes
        ],
        "successful_episodes": sum(1 for e in episodes if float(e["task_state_success"]) == 1.0),
        "brains": [],
    }
    if mode == "central":
        audit["brains"].append(_bootstrap_memory(
            out, episodes, own_label=None, trace_key="team",
            min_support=min_support, min_success_rate=min_success_rate,
        ))
        audit["deployment"] = deployment_config(
            str(out / "task_memory"), str(out / "global_memory" / GLOBAL_MEMORY_FILENAME)
        )
    else:
        labels = sorted({label for e in episodes for label in e["traces"] if label != "team"})
        for label in labels:
            audit["brains"].append(_bootstrap_memory(
                out, episodes, own_label=label, trace_key=label,
                min_support=min_support, min_success_rate=min_success_rate,
            ))
        # Directory (no suffix): each brain appends <label>/global_memory.json.
        audit["deployment"] = deployment_config(str(out / "task_memory"), str(out / "global_memory"))
    write_json_atomic(str(out / "bootstrap_audit.json"), audit)
    return audit


def bootstrap_from_episodes(
    out_dir: str,
    episodes: Sequence[Mapping[str, Any]],
    *,
    min_support: int = 2,
    min_success_rate: float = 0.5,
    force: bool = False,
    auto_extract_llm: Optional[extractor.LlmFn] = None,
) -> Dict[str, Any]:
    """Central bootstrap from already-labelled episodes (environment agnostic).

    ``episodes`` items: ``{"job_id", "episode_id", "instruction",
    "task_state_success" (official 0/1), "traces": {"team": load_episode_trace(...)}}``.
    Same outputs as ``bootstrap_from_queue(mode="central")``; any environment
    that writes the centralized decision layout can use it (install its
    ``Vocabulary`` first with ``primitives.use_vocabulary``).
    """
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        if not force:
            raise FileExistsError(f"{out} is not empty (use force=True to rebuild)")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    audit: Dict[str, Any] = {
        "phase": Phase.BOOTSTRAP.value,
        "mode": "central",
        "episodes": [
            {"job_id": e.get("job_id"), "episode_id": e["episode_id"],
             "task_state_success": e["task_state_success"], "traces": sorted(e["traces"])}
            for e in episodes
        ],
        "successful_episodes": sum(1 for e in episodes if float(e["task_state_success"]) == 1.0),
        "brains": [_bootstrap_memory(
            out, episodes, own_label=None, trace_key="team",
            min_support=min_support, min_success_rate=min_success_rate,
            llm=auto_extract_llm,
        )],
    }
    audit["deployment"] = deployment_config(
        str(out / "task_memory"), str(out / "global_memory" / GLOBAL_MEMORY_FILENAME)
    )
    write_json_atomic(str(out / "bootstrap_audit.json"), audit)
    return audit


def deploy_overrides(memory_dir: str, mode: str) -> List[str]:
    """Hydra overrides turning a bootstrap output into a frozen deployment run."""
    out = Path(memory_dir)
    if mode == "central":
        prefix = ["evaluation.planner.plan_config.harness"]
        config = deployment_config(
            str(out / "task_memory"), str(out / "global_memory" / GLOBAL_MEMORY_FILENAME)
        )
    elif mode == "decentral":
        prefix = [f"evaluation.agents.{label}.planner.plan_config.harness" for label in ("agent_0", "agent_1")]
        config = deployment_config(str(out / "task_memory"), str(out / "global_memory"))
    else:
        raise ValueError("mode must be 'central' or 'decentral'")
    overrides = []
    for base in prefix:
        for key, value in config.items():
            overrides.append(f"{base}.{key}={'null' if value is None else value}")
    return overrides


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m harness_vla.envs.partnr.lifecycle",
        description="Offline bootstrap (queue traces -> memories) and deployment overrides.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    boot = sub.add_parser("bootstrap", help="build Task/Global Memory from a queue condition")
    boot.add_argument("--queue", required=True, help="queue root (contains jobs/)")
    boot.add_argument("--condition", required=True)
    boot.add_argument("--out", required=True, help="memory output directory")
    boot.add_argument("--mode", choices=("central", "decentral"), required=True)
    boot.add_argument("--min-support", type=int, default=2)
    boot.add_argument("--min-success-rate", type=float, default=0.5)
    boot.add_argument("--force", action="store_true", help="rebuild a non-empty --out")
    deploy = sub.add_parser("deploy-config", help="print Hydra overrides for deployment")
    deploy.add_argument("--memory", required=True, help="bootstrap --out directory")
    deploy.add_argument("--mode", choices=("central", "decentral"), required=True)
    deploy.add_argument("--json", action="store_true", help="emit a JSON list")
    args = parser.parse_args(argv)

    if args.command == "bootstrap":
        audit = bootstrap_from_queue(
            args.queue, args.condition, args.out, mode=args.mode,
            min_support=args.min_support, min_success_rate=args.min_success_rate, force=args.force,
        )
        print(json.dumps({
            "episodes": len(audit["episodes"]),
            "successful_episodes": audit["successful_episodes"],
            "brains": [
                {"label": b["label"], "task_memory_files": b["task_memory_files"],
                 "promoted_rules": b["promoted_rules"]}
                for b in audit["brains"]
            ],
            "deployment": audit["deployment"],
            "audit": str(Path(args.out) / "bootstrap_audit.json"),
        }, indent=2))
        return 0
    overrides = deploy_overrides(args.memory, args.mode)
    print(json.dumps(overrides) if args.json else "\n".join(overrides))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
