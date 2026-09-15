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
    extract_global_memory_candidates,
    load_episode_trace,
    partnr_global_memory,
    partnr_global_memory_ledger,
    promote_candidates,
    write_partnr_task_memory,
)
from harness_vla.planner.phase_policy import Phase

GLOBAL_MEMORY_FILENAME = "global_memory.json"


def bootstrap_from_traces(
    trace_paths: Iterable[str],
    task_memory_dir: str,
    global_memory_ledger_path: Optional[str],
    chat_fn: Optional[ChatFn] = None,
    min_support: int = 2,
    success_overrides: Optional[Mapping[str, bool]] = None,
) -> Dict[str, Any]:
    """Build Task Memory (and optionally Global Memory) from bootstrap traces.

    Returns an audit dict: task keys written, ledger counts, promoted rules
    and the rendered Global Memory path (``<ledger dir>/global_memory.json``).
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
