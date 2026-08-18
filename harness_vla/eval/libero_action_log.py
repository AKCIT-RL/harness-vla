"""Beta-only flat per-action log derived from persisted primitive traces.

This module is instrumentation added for evaluation batteries; it is not a
mechanism from arXiv:2607.08448v2. It never alters the authoritative
``trace.jsonl`` schema: it only re-projects each turn's primitive trace into
one JSONL record per physical 7D action so profiling and per-action metrics
can be computed without re-parsing nested chunk structures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def flatten_turn_actions(
    turn: int,
    primitive: Optional[str],
    primitive_trace: Any,
    *,
    ts: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Project one turn's primitive trace into flat per-action records.

    Supports both trace shapes produced by the native executors:

    - VLA chunks: dicts with ``executed_actions``/``rewards``/``dones``/
      ``task_successes``/``tau_evidence`` and ``inference_duration_s``;
    - analytic steps: dicts with ``step``/``action``/``reward``/``env_done``.

    Entries without physical actions (grounding events, tau setup errors,
    synthetic test traces) are skipped silently.
    """
    records: List[Dict[str, Any]] = []
    if not isinstance(primitive_trace, list):
        return records
    action_index = 0
    for entry in primitive_trace:
        if not isinstance(entry, dict):
            continue
        if "executed_actions" in entry:
            executed = entry.get("executed_actions") or []
            rewards = entry.get("rewards") or []
            dones = entry.get("dones") or []
            successes = entry.get("task_successes") or []
            evidence = entry.get("tau_evidence") or []
            for position, action in enumerate(executed):
                action_index += 1
                record = {
                    "turn": turn,
                    "primitive": primitive,
                    "source": "vla_chunk",
                    "chunk_index": entry.get("chunk_index"),
                    "action_index_in_turn": action_index,
                    "action": action,
                    "reward": rewards[position] if position < len(rewards) else None,
                    "env_done": dones[position] if position < len(dones) else None,
                    "task_success": (
                        successes[position] if position < len(successes) else None
                    ),
                    "tau_satisfied": (
                        evidence[position].get("tau_satisfied")
                        if position < len(evidence)
                        and isinstance(evidence[position], dict)
                        else None
                    ),
                }
                if position == 0:
                    record["inference_duration_s"] = entry.get(
                        "inference_duration_s"
                    )
                if ts is not None:
                    record["turn_ts"] = ts
                records.append(record)
        elif "action" in entry and "step" in entry:
            action_index += 1
            record = {
                "turn": turn,
                "primitive": primitive,
                "source": "analytic_step",
                "step": entry.get("step"),
                "action_index_in_turn": action_index,
                "action": entry.get("action"),
                "reward": entry.get("reward"),
                "env_done": entry.get("env_done"),
                "task_success": entry.get("task_success"),
            }
            if ts is not None:
                record["turn_ts"] = ts
            records.append(record)
    return records


__all__ = ["flatten_turn_actions"]
