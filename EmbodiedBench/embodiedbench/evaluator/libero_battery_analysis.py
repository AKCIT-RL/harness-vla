"""Beta-only battery analysis: per-primitive metrics, profiling and funnels.

Reads only persisted artifacts (``battery_summary.json``, per-episode
``episode.json``, ``trace.jsonl`` and ``actions.jsonl``) and computes
intermediate metrics for evaluation batteries. Pure functions; no simulator
required. This is instrumentation for the beta, not a paper mechanism.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from embodiedbench.planner.harness.trace_io import load_complete_jsonl


PLANNER_ERROR_STATUSES = ("planner_parse_error",)
COMPILE_ERROR_STATUSES = (
    "primitive_compile_error",
    "release_without_holding",
    "holding_incompatible",
    "grasp_lost",
)


def percentile(values: List[float], fraction: float) -> Optional[float]:
    """Linear-interpolated percentile over a small sample; None when empty."""
    cleaned = sorted(
        float(value) for value in values if value is not None and math.isfinite(value)
    )
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    position = fraction * (len(cleaned) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return cleaned[lower] * (1.0 - weight) + cleaned[upper] * weight


def load_battery_episodes(battery_root) -> List[Dict[str, Any]]:
    """Load every completed or failed episode folder of one battery."""
    root = Path(battery_root)
    episodes_root = root / "episodes"
    if not episodes_root.is_dir():
        raise ValueError("battery root has no episodes directory")
    episodes = []
    for episode_dir in sorted(episodes_root.iterdir()):
        if not episode_dir.is_dir():
            continue
        entry: Dict[str, Any] = {"name": episode_dir.name}
        episode_path = episode_dir / "episode.json"
        error_path = episode_dir / "episode_error.json"
        if episode_path.exists():
            entry["episode"] = json.loads(episode_path.read_text(encoding="utf-8"))
            entry["status"] = "completed"
        elif error_path.exists():
            entry["error"] = json.loads(error_path.read_text(encoding="utf-8"))
            entry["status"] = "error"
        else:
            entry["status"] = "incomplete"
        entry["trace"] = load_complete_jsonl(episode_dir / "trace.jsonl")
        entry["actions"] = load_complete_jsonl(episode_dir / "actions.jsonl")
        episodes.append(entry)
    return episodes


def _turn_records(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        record
        for record in episode.get("trace", [])
        if isinstance(record, dict) and "turn" in record
    ]


def compute_primitive_metrics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Usage, post-condition failure rate and steps per primitive."""
    table: Dict[str, Dict[str, Any]] = {}
    for episode in episodes:
        for record in _turn_records(episode):
            primitive = record.get("primitive")
            if primitive is None:
                continue
            row = table.setdefault(
                primitive,
                {"turns": 0, "postcondition_failures": 0, "steps_executed": 0},
            )
            row["turns"] += 1
            if record.get("primitive_postcondition_met") is False:
                row["postcondition_failures"] += 1
            feedback = record.get("feedback") or {}
            steps = feedback.get("steps_executed")
            if isinstance(steps, int):
                row["steps_executed"] += steps
    total_turns = sum(row["turns"] for row in table.values())
    for primitive, row in table.items():
        row["usage_rate"] = row["turns"] / total_turns if total_turns else None
        row["failure_rate"] = (
            row["postcondition_failures"] / row["turns"] if row["turns"] else None
        )
        row["mean_steps_per_turn"] = (
            row["steps_executed"] / row["turns"] if row["turns"] else None
        )
    return {"total_turns": total_turns, "primitives": table}


def compute_protocol_health(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Planner parse rate and primitive compile rate over all turns."""
    turns = 0
    parse_errors = 0
    compile_errors = 0
    for episode in episodes:
        for record in _turn_records(episode):
            turns += 1
            status = record.get("status")
            if status in PLANNER_ERROR_STATUSES:
                parse_errors += 1
            elif status in COMPILE_ERROR_STATUSES:
                compile_errors += 1
    return {
        "turns": turns,
        "planner_parse_errors": parse_errors,
        "parse_success_rate": (turns - parse_errors) / turns if turns else None,
        "compile_errors": compile_errors,
        "compile_success_rate": (
            (turns - compile_errors) / turns if turns else None
        ),
    }


def compute_termination_distribution(
    episodes: List[Dict[str, Any]]
) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    for episode in episodes:
        if episode.get("status") == "error":
            key = "battery_episode_error"
        elif episode.get("status") == "incomplete":
            key = "incomplete"
        else:
            key = episode["episode"].get("termination_reason", "missing")
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def compute_success_metrics(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [
        episode for episode in episodes if episode.get("status") == "completed"
    ]
    successes = [
        episode
        for episode in completed
        if episode["episode"].get("task_success")
    ]
    turns = [
        episode["episode"].get("turns_executed")
        for episode in successes
        if isinstance(episode["episode"].get("turns_executed"), int)
    ]
    actions = [
        episode["episode"].get("actions_executed")
        for episode in successes
        if isinstance(episode["episode"].get("actions_executed"), int)
    ]
    return {
        "episodes": len(episodes),
        "episodes_completed": len(completed),
        "successes": len(successes),
        "task_success_rate": (
            len(successes) / len(completed) if completed else None
        ),
        "mean_turns_to_success": sum(turns) / len(turns) if turns else None,
        "mean_actions_to_success": (
            sum(actions) / len(actions) if actions else None
        ),
        "p95_actions_to_success": percentile([float(a) for a in actions], 0.95),
    }


def compute_latency_profile(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Layer-2 timing profile from per-turn ``timing`` instrumentation."""
    per_primitive: Dict[str, Dict[str, List[float]]] = {}
    planner_total = 0.0
    execution_total = 0.0
    overhead_total = 0.0
    inference_durations: List[float] = []
    instrumented_turns = 0
    for episode in episodes:
        for record in _turn_records(episode):
            timing = record.get("timing")
            if not isinstance(timing, dict):
                continue
            instrumented_turns += 1
            primitive = record.get("primitive") or "no_invocation"
            samples = per_primitive.setdefault(
                primitive,
                {"planner_s": [], "execution_s": [], "turn_s": []},
            )
            planner_s = timing.get("planner_duration_s")
            execution_s = timing.get("execution_duration_s")
            turn_s = timing.get("turn_duration_s")
            if planner_s is not None:
                samples["planner_s"].append(planner_s)
                planner_total += planner_s
            if execution_s is not None:
                samples["execution_s"].append(execution_s)
                execution_total += execution_s
            if turn_s is not None:
                samples["turn_s"].append(turn_s)
                overhead_total += max(
                    0.0, turn_s - (planner_s or 0.0) - (execution_s or 0.0)
                )
        for action in episode.get("actions", []):
            duration = action.get("inference_duration_s")
            if duration is not None:
                inference_durations.append(float(duration))
    total = planner_total + execution_total + overhead_total
    profile = {
        "instrumented_turns": instrumented_turns,
        "planner_total_s": round(planner_total, 6),
        "execution_total_s": round(execution_total, 6),
        "loop_overhead_total_s": round(overhead_total, 6),
        "component_share": (
            {
                "planner": planner_total / total,
                "execution": execution_total / total,
                "loop_overhead": overhead_total / total,
            }
            if total > 0
            else None
        ),
        "vla_inference_p50_s": percentile(inference_durations, 0.50),
        "vla_inference_p95_s": percentile(inference_durations, 0.95),
        "per_primitive": {},
    }
    for primitive, samples in per_primitive.items():
        profile["per_primitive"][primitive] = {
            "turns": len(samples["turn_s"]),
            "planner_p50_s": percentile(samples["planner_s"], 0.50),
            "planner_p95_s": percentile(samples["planner_s"], 0.95),
            "execution_p50_s": percentile(samples["execution_s"], 0.50),
            "execution_p95_s": percentile(samples["execution_s"], 0.95),
            "turn_p50_s": percentile(samples["turn_s"], 0.50),
            "turn_p95_s": percentile(samples["turn_s"], 0.95),
        }
    return profile


def compute_tau_calibration(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """How often satisfied tau translated into official task success."""
    tau_satisfied_episodes = 0
    tau_and_task_success = 0
    tau_without_task_success = 0
    for episode in episodes:
        if episode.get("status") != "completed":
            continue
        satisfied = any(
            (record.get("feedback") or {}).get("tau_satisfied") is True
            for record in _turn_records(episode)
        )
        if not satisfied:
            continue
        tau_satisfied_episodes += 1
        if episode["episode"].get("task_success"):
            tau_and_task_success += 1
        else:
            tau_without_task_success += 1
    return {
        "episodes_with_tau_satisfied": tau_satisfied_episodes,
        "tau_to_task_success": tau_and_task_success,
        "tau_without_task_success": tau_without_task_success,
        "tau_precision_for_task_success": (
            tau_and_task_success / tau_satisfied_episodes
            if tau_satisfied_episodes
            else None
        ),
    }


def compute_phase_funnel(episodes: List[Dict[str, Any]]) -> Dict[str, int]:
    """Grasp -> transport -> placement -> success funnel per episode."""
    funnel = {"episodes": 0, "grasp": 0, "transport": 0, "placement": 0, "success": 0}
    for episode in episodes:
        if episode.get("status") != "completed":
            continue
        funnel["episodes"] += 1
        records = _turn_records(episode)
        grasp_turn = None
        for record in records:
            if (record.get("feedback") or {}).get("tau_satisfied") is True:
                grasp_turn = record["turn"]
                break
        if grasp_turn is None:
            continue
        funnel["grasp"] += 1
        transported = any(
            record["turn"] > grasp_turn
            and record.get("primitive") in ("move_to", "move_pose")
            and record.get("primitive_postcondition_met") is True
            for record in records
        )
        if transported:
            funnel["transport"] += 1
        placed = any(
            record["turn"] > grasp_turn
            and record.get("primitive") in ("release", "set_gripper")
            and record.get("primitive_postcondition_met") is True
            for record in records
        )
        if placed:
            funnel["placement"] += 1
        if episode["episode"].get("task_success"):
            funnel["success"] += 1
    return funnel


def compute_grounding_stability(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Positional spread of repeated re-groundings of the same target."""
    positions: Dict[str, List[List[float]]] = {}
    for episode in episodes:
        for record in _turn_records(episode):
            for event in record.get("primitive_trace") or []:
                if (
                    isinstance(event, dict)
                    and event.get("event") == "target_resolved"
                    and isinstance(event.get("target_xyz"), list)
                ):
                    key = "%s::%s" % (episode["name"], event.get("target"))
                    positions.setdefault(key, []).append(
                        [float(value) for value in event["target_xyz"]]
                    )
    spreads = []
    per_target = {}
    for key, samples in positions.items():
        if len(samples) < 2:
            continue
        centroid = [
            sum(sample[axis] for sample in samples) / len(samples)
            for axis in range(3)
        ]
        spread = max(
            math.sqrt(
                sum((sample[axis] - centroid[axis]) ** 2 for axis in range(3))
            )
            for sample in samples
        )
        per_target[key] = {"regroundings": len(samples), "max_spread_m": spread}
        spreads.append(spread)
    return {
        "targets_regrounded": len(per_target),
        "max_spread_p50_m": percentile(spreads, 0.50),
        "max_spread_p95_m": percentile(spreads, 0.95),
        "per_target": per_target,
    }


def build_report(battery_root) -> Dict[str, Any]:
    """Compute the full layer-2 report for one battery folder."""
    root = Path(battery_root)
    episodes = load_battery_episodes(root)
    manifest_path = root / "battery_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    return {
        "classification": "beta_only",
        "battery_root": str(root),
        "protocol_name": manifest.get("protocol_name"),
        "condition": manifest.get("condition"),
        "git_commit": manifest.get("git_commit"),
        "success": compute_success_metrics(episodes),
        "termination_reasons": compute_termination_distribution(episodes),
        "protocol_health": compute_protocol_health(episodes),
        "primitives": compute_primitive_metrics(episodes),
        "latency": compute_latency_profile(episodes),
        "tau_calibration": compute_tau_calibration(episodes),
        "phase_funnel": compute_phase_funnel(episodes),
        "grounding_stability": compute_grounding_stability(episodes),
    }


def render_charts(report: Dict[str, Any], output_dir) -> List[Path]:
    """Render the report's headline metrics as PNG charts (matplotlib Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    def _save(figure, name: str) -> None:
        path = out / name
        figure.tight_layout()
        figure.savefig(path, dpi=120)
        plt.close(figure)
        written.append(path)

    primitives = report["primitives"]["primitives"]
    if primitives:
        names = sorted(primitives)
        usage = [primitives[name]["turns"] for name in names]
        failures = [
            primitives[name]["postcondition_failures"] for name in names
        ]
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.bar(names, usage, label="turns", color="#4c72b0")
        axis.bar(names, failures, label="postcondition failures", color="#c44e52")
        axis.set_title("Primitive usage and failures")
        axis.set_ylabel("turns")
        axis.legend()
        axis.tick_params(axis="x", rotation=30)
        _save(figure, "primitive_usage_failures.png")

    reasons = report["termination_reasons"]
    if reasons:
        names = sorted(reasons)
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.barh(names, [reasons[name] for name in names], color="#55a868")
        axis.set_title("Episode termination reasons")
        axis.set_xlabel("episodes")
        _save(figure, "termination_reasons.png")

    share = report["latency"].get("component_share")
    if share:
        figure, axis = plt.subplots(figsize=(5, 5))
        labels = ["planner", "execution", "loop overhead"]
        values = [share["planner"], share["execution"], share["loop_overhead"]]
        axis.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        axis.set_title("Wall-clock share per component")
        _save(figure, "latency_breakdown.png")

    per_primitive = report["latency"]["per_primitive"]
    if per_primitive:
        names = sorted(per_primitive)
        p50 = [per_primitive[name]["turn_p50_s"] or 0.0 for name in names]
        p95 = [per_primitive[name]["turn_p95_s"] or 0.0 for name in names]
        figure, axis = plt.subplots(figsize=(8, 4))
        indices = range(len(names))
        axis.bar([i - 0.2 for i in indices], p50, width=0.4, label="p50")
        axis.bar([i + 0.2 for i in indices], p95, width=0.4, label="p95")
        axis.set_xticks(list(indices))
        axis.set_xticklabels(names, rotation=30)
        axis.set_ylabel("turn duration (s)")
        axis.set_title("Turn latency per primitive")
        axis.legend()
        _save(figure, "latency_per_primitive.png")

    funnel = report["phase_funnel"]
    if funnel.get("episodes"):
        stages = ["episodes", "grasp", "transport", "placement", "success"]
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar(stages, [funnel[stage] for stage in stages], color="#8172b2")
        axis.set_title("Phase funnel (per episode)")
        axis.set_ylabel("episodes reaching stage")
        _save(figure, "phase_funnel.png")

    return written


__all__ = [
    "build_report",
    "compute_grounding_stability",
    "compute_latency_profile",
    "compute_phase_funnel",
    "compute_primitive_metrics",
    "compute_protocol_health",
    "compute_success_metrics",
    "compute_tau_calibration",
    "compute_termination_distribution",
    "load_battery_episodes",
    "percentile",
    "render_charts",
]
