"""Offline bootstrap -> frozen deployment lifecycle for the PARTNR adapter.

Paper-confirmed protocol: memories are written only from the bootstrap (dev)
split; the deployment phase loads them read-only (``phase_policy.Phase``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from harness_vla.envs.partnr.memory import (
    ChatFn,
    build_partnr_task_memory,
    episode_id_of,
    episode_success,
    extract_global_memory_candidates,
    load_episode_trace,
    partnr_global_memory,
    partnr_global_memory_ledger,
    promote_candidates,
    symbolic_steps,
    write_partnr_task_memory,
)
from harness_vla.envs.partnr.primitives import PARTNR_PRIMITIVE_NAMES
from harness_vla.memory import global_memory_extractor as extractor
from harness_vla.planner.phase_policy import Phase

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
) -> Dict[str, Any]:
    """Run the automatic extractor + phase-gated promotion on bootstrap episodes."""
    digests = extractor.summarize_bootstrap_traces(
        episodes, transcript_fn=symbolic_steps, success_fn=episode_success
    )
    candidates = extractor.propose_global_memory_candidates(
        digests, llm, min_support=min_support, primitive_names=PARTNR_PRIMITIVE_NAMES
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
