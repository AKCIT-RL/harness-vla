"""Beta-only multi-seed evaluation battery over a frozen LIBERO protocol.

This module is evaluation infrastructure for the beta, not a mechanism from
arXiv:2607.08448v2. It expands a frozen protocol file into a deterministic
episode matrix, runs one episode at a time through an injected runner,
tolerates per-episode crashes, and aggregates results paired by
``(task_suite, task_id, initial_state_index, seed)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from harness_vla.trace_io import (
    resolve_git_commit,
    write_json_atomic,
)


REQUIRED_PROTOCOL_KEYS = (
    "protocol_name",
    "version",
    "bootstrap_seeds",
    "deployment_seeds",
    "tasks",
    "budgets",
)

EpisodeRunner = Callable[[Dict[str, Any], Path], Dict[str, Any]]


def load_protocol(path) -> Dict[str, Any]:
    """Load and structurally validate a frozen evaluation protocol."""
    protocol_path = Path(path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_PROTOCOL_KEYS if key not in protocol]
    if missing:
        raise ValueError("protocol is missing keys: %s" % ", ".join(missing))
    for task in protocol["tasks"]:
        for key in ("task_suite", "task_id", "initial_state_indices"):
            if key not in task:
                raise ValueError("protocol task is missing key: %s" % key)
    return protocol


def episode_name(spec: Dict[str, Any]) -> str:
    """Deterministic episode folder name enabling crash-tolerant resume."""
    return "%s_%s_task%02d_state%02d_seed%03d" % (
        spec["condition"],
        spec["task_suite"],
        spec["task_id"],
        spec["initial_state_index"],
        spec["seed"],
    )


def expand_protocol_matrix(
    protocol: Dict[str, Any],
    *,
    condition: str,
    seeds: Optional[Sequence[int]] = None,
    include_hidden: bool = False,
    hidden_only: bool = False,
) -> List[Dict[str, Any]]:
    """Expand the frozen protocol into an ordered list of episode specs."""
    if not isinstance(condition, str) or not condition:
        raise ValueError("condition must be a non-empty string")
    selected_seeds = list(protocol["deployment_seeds"] if seeds is None else seeds)
    if not selected_seeds:
        raise ValueError("at least one seed is required")
    declared = set(protocol["deployment_seeds"]) | set(protocol["bootstrap_seeds"])
    undeclared = [seed for seed in selected_seeds if seed not in declared]
    if undeclared:
        raise ValueError(
            "seeds not declared by the frozen protocol: %s"
            % ", ".join(str(seed) for seed in undeclared)
        )
    tasks = list(protocol["tasks"])
    if hidden_only:
        tasks = list(protocol.get("hidden_tasks", {}).get("tasks", []))
        if not tasks:
            raise ValueError("protocol declares no hidden tasks")
    elif include_hidden:
        tasks.extend(protocol.get("hidden_tasks", {}).get("tasks", []))
    matrix = []
    for task in tasks:
        for state_index in task["initial_state_indices"]:
            for seed in selected_seeds:
                matrix.append(
                    {
                        "condition": condition,
                        "task_suite": task["task_suite"],
                        "task_id": task["task_id"],
                        "initial_state_index": state_index,
                        "seed": seed,
                    }
                )
    return matrix


def _episode_status(episode_root: Path) -> Optional[Dict[str, Any]]:
    episode_path = episode_root / "episode.json"
    if not episode_path.exists():
        return None
    try:
        return json.loads(episode_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def aggregate_battery(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-episode outcomes into one battery summary."""
    completed = [entry for entry in results if entry["status"] == "completed"]
    failed = [entry for entry in results if entry["status"] == "error"]
    successes = [
        entry for entry in completed if entry["episode"].get("task_success")
    ]
    termination_reasons: Dict[str, int] = {}
    for entry in completed:
        reason = entry["episode"].get("termination_reason", "missing")
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
    summary = {
        "episodes_planned": len(results),
        "episodes_completed": len(completed),
        "episodes_failed": len(failed),
        "successes": len(successes),
        "task_success_rate": (
            len(successes) / len(completed) if completed else None
        ),
        "termination_reasons": termination_reasons,
        "failures": [
            {
                "episode": entry["name"],
                "error_type": entry.get("error_type"),
                "error": entry.get("error"),
            }
            for entry in failed
        ],
    }
    if successes:
        turns = [entry["episode"].get("turns_executed") for entry in successes]
        actions = [entry["episode"].get("actions_executed") for entry in successes]
        turns = [value for value in turns if isinstance(value, int)]
        actions = [value for value in actions if isinstance(value, int)]
        if turns:
            summary["mean_turns_to_success"] = sum(turns) / len(turns)
        if actions:
            summary["mean_actions_to_success"] = sum(actions) / len(actions)
    return summary


def run_battery(
    protocol: Dict[str, Any],
    *,
    battery_root,
    episode_runner: EpisodeRunner,
    condition: str,
    seeds: Optional[Sequence[int]] = None,
    include_hidden: bool = False,
    hidden_only: bool = False,
    resume: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full episode matrix sequentially and durably persist results.

    ``episode_runner(spec, episode_root)`` must run exactly one episode inside
    ``episode_root`` and return the episode summary dict (the ``episode.json``
    payload). Crashes are recorded per episode and never abort the battery.
    """
    root = Path(battery_root)
    root.mkdir(parents=True, exist_ok=True)
    episodes_root = root / "episodes"
    episodes_root.mkdir(exist_ok=True)
    matrix = expand_protocol_matrix(
        protocol,
        condition=condition,
        seeds=seeds,
        include_hidden=include_hidden,
        hidden_only=hidden_only,
    )
    manifest = {
        "run_type": "libero_battery",
        "classification": "beta_only",
        "protocol_name": protocol["protocol_name"],
        "protocol_version": protocol["version"],
        "protocol": protocol,
        "condition": condition,
        "seeds": list(protocol["deployment_seeds"] if seeds is None else seeds),
        "include_hidden": bool(include_hidden),
        "hidden_only": bool(hidden_only),
        "overrides": dict(overrides or {}),
        "episodes_planned": len(matrix),
        "git_commit": resolve_git_commit(Path(__file__)),
        "status": "in_progress",
    }
    write_json_atomic(root / "battery_manifest.json", manifest)

    results: List[Dict[str, Any]] = []
    for spec in matrix:
        name = episode_name(spec)
        episode_root = episodes_root / name
        existing = _episode_status(episode_root) if resume else None
        if existing is not None:
            results.append(
                {"name": name, "spec": spec, "status": "completed", "episode": existing}
            )
            continue
        episode_root.mkdir(parents=True, exist_ok=True)
        try:
            episode = episode_runner(dict(spec), episode_root)
            results.append(
                {"name": name, "spec": spec, "status": "completed", "episode": episode}
            )
        except Exception as exc:  # crash-tolerant by design
            results.append(
                {
                    "name": name,
                    "spec": spec,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            write_json_atomic(
                episode_root / "episode_error.json",
                {
                    "spec": spec,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        summary = aggregate_battery(results)
        summary["condition"] = condition
        write_json_atomic(root / "battery_summary.json", summary)

    summary = aggregate_battery(results)
    summary["condition"] = condition
    write_json_atomic(root / "battery_summary.json", summary)
    manifest["status"] = "completed"
    manifest["episodes_completed"] = summary["episodes_completed"]
    manifest["episodes_failed"] = summary["episodes_failed"]
    write_json_atomic(root / "battery_manifest.json", manifest)
    return {"summary": summary, "results": results, "manifest": manifest}


__all__ = [
    "aggregate_battery",
    "episode_name",
    "expand_protocol_matrix",
    "load_protocol",
    "run_battery",
]
