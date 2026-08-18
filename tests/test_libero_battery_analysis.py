"""Tests for the beta-only battery analysis (layer-2 metrics and charts)."""

import json

import pytest

from harness_vla.eval.libero_battery_analysis import (
    build_report,
    compute_grounding_stability,
    compute_latency_profile,
    compute_phase_funnel,
    compute_primitive_metrics,
    compute_protocol_health,
    compute_success_metrics,
    compute_tau_calibration,
    compute_termination_distribution,
    load_battery_episodes,
    percentile,
    render_charts,
)


def _turn(
    turn,
    primitive,
    *,
    postcondition=True,
    status="postcondition_met",
    tau_satisfied=None,
    steps=4,
    timing=None,
    primitive_trace=None,
):
    feedback = {
        "action": primitive,
        "primitive_success": postcondition,
        "task_success": status == "task_success",
        "env_done": False,
        "termination_reason": status,
        "steps_executed": steps,
        "holding": None,
        "recoverable": False,
    }
    if tau_satisfied is not None:
        feedback["tau_satisfied"] = tau_satisfied
    record = {
        "turn": turn,
        "primitive": primitive,
        "primitive_postcondition_met": postcondition,
        "status": status,
        "termination_reason": status,
        "task_success": status == "task_success",
        "feedback": feedback,
        "primitive_trace": primitive_trace or [],
    }
    if timing is not None:
        record["timing"] = timing
    return record


def _timing(planner_s, execution_s):
    return {
        "ts": "2026-01-01T00:00:00+00:00",
        "planner_duration_s": planner_s,
        "execution_duration_s": execution_s,
        "turn_duration_s": planner_s + execution_s + 0.05,
    }


def _success_trace():
    return [
        _turn(
            1,
            "vla_act",
            tau_satisfied=True,
            timing=_timing(2.0, 5.0),
        ),
        _turn(
            2,
            "move_to",
            timing=_timing(1.5, 3.0),
            primitive_trace=[
                {
                    "event": "target_resolved",
                    "target": "plate_1",
                    "mode": "above",
                    "target_xyz": [0.10, 0.20, 0.30],
                    "provenance": "rgbd",
                },
            ],
        ),
        _turn(
            3,
            "move_to",
            timing=_timing(1.0, 2.0),
            primitive_trace=[
                {
                    "event": "target_resolved",
                    "target": "plate_1",
                    "mode": "release_pose",
                    "target_xyz": [0.11, 0.21, 0.31],
                    "provenance": "rgbd",
                },
            ],
        ),
        _turn(4, "release", status="task_success", timing=_timing(1.2, 0.8)),
    ]


def _failure_trace():
    return [
        _turn(1, "vla_act", tau_satisfied=False, postcondition=False,
              status="step_budget_exhausted", timing=_timing(2.5, 8.0)),
        _turn(2, None, postcondition=False, status="planner_parse_error",
              timing={"ts": "2026-01-01T00:01:00+00:00",
                      "planner_duration_s": 3.0,
                      "turn_duration_s": 3.02}),
    ]


def _write_episode(root, name, *, episode=None, trace=None, actions=None, error=None):
    episode_dir = root / "episodes" / name
    episode_dir.mkdir(parents=True)
    if episode is not None:
        (episode_dir / "episode.json").write_text(
            json.dumps(episode), encoding="utf-8"
        )
    if error is not None:
        (episode_dir / "episode_error.json").write_text(
            json.dumps(error), encoding="utf-8"
        )
    if trace is not None:
        with (episode_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
            for record in trace:
                handle.write(json.dumps(record) + "\n")
    if actions is not None:
        with (episode_dir / "actions.jsonl").open("w", encoding="utf-8") as handle:
            for record in actions:
                handle.write(json.dumps(record) + "\n")


@pytest.fixture()
def battery_root(tmp_path):
    root = tmp_path / "battery"
    _write_episode(
        root,
        "harness_libero_spatial_task00_state00_seed101",
        episode={
            "task_success": True,
            "termination_reason": "task_success",
            "turns_executed": 4,
            "actions_executed": 14,
        },
        trace=_success_trace(),
        actions=[
            {"turn": 1, "primitive": "vla_act", "inference_duration_s": 0.30},
            {"turn": 1, "primitive": "vla_act"},
        ],
    )
    _write_episode(
        root,
        "harness_libero_spatial_task00_state00_seed102",
        episode={
            "task_success": False,
            "termination_reason": "planner_parse_error",
            "turns_executed": 2,
            "actions_executed": 10,
        },
        trace=_failure_trace(),
        actions=[],
    )
    _write_episode(
        root,
        "harness_libero_spatial_task00_state01_seed101",
        error={"error_type": "RuntimeError", "error": "sim crash"},
    )
    (root / "battery_manifest.json").write_text(
        json.dumps(
            {
                "protocol_name": "benchmark_frozen_v1",
                "condition": "harness",
                "git_commit": "deadbeef",
            }
        ),
        encoding="utf-8",
    )
    return root


def test_percentile_interpolates_and_handles_empty():
    assert percentile([], 0.5) is None
    assert percentile([3.0], 0.95) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)


def test_load_battery_episodes_classifies_states(battery_root):
    episodes = load_battery_episodes(battery_root)
    statuses = sorted(entry["status"] for entry in episodes)
    assert statuses == ["completed", "completed", "error"]


def test_primitive_metrics(battery_root):
    episodes = load_battery_episodes(battery_root)
    metrics = compute_primitive_metrics(episodes)
    assert metrics["total_turns"] == 5  # parse-error turn has no primitive
    vla = metrics["primitives"]["vla_act"]
    assert vla["turns"] == 2
    assert vla["postcondition_failures"] == 1
    assert vla["failure_rate"] == pytest.approx(0.5)
    assert metrics["primitives"]["move_to"]["usage_rate"] == pytest.approx(2 / 5)


def test_protocol_health(battery_root):
    episodes = load_battery_episodes(battery_root)
    health = compute_protocol_health(episodes)
    assert health["turns"] == 6
    assert health["planner_parse_errors"] == 1
    assert health["parse_success_rate"] == pytest.approx(5 / 6)
    assert health["compile_errors"] == 0


def test_termination_and_success_metrics(battery_root):
    episodes = load_battery_episodes(battery_root)
    reasons = compute_termination_distribution(episodes)
    assert reasons == {
        "task_success": 1,
        "planner_parse_error": 1,
        "battery_episode_error": 1,
    }
    success = compute_success_metrics(episodes)
    assert success["episodes_completed"] == 2
    assert success["task_success_rate"] == pytest.approx(0.5)
    assert success["mean_turns_to_success"] == 4
    assert success["mean_actions_to_success"] == 14


def test_latency_profile(battery_root):
    episodes = load_battery_episodes(battery_root)
    profile = compute_latency_profile(episodes)
    assert profile["instrumented_turns"] == 6
    assert profile["planner_total_s"] == pytest.approx(11.2)
    assert profile["execution_total_s"] == pytest.approx(18.8)
    share = profile["component_share"]
    assert share["planner"] + share["execution"] + share["loop_overhead"] == (
        pytest.approx(1.0)
    )
    assert profile["vla_inference_p50_s"] == pytest.approx(0.30)
    assert "vla_act" in profile["per_primitive"]
    assert "no_invocation" in profile["per_primitive"]


def test_tau_calibration(battery_root):
    episodes = load_battery_episodes(battery_root)
    calibration = compute_tau_calibration(episodes)
    assert calibration["episodes_with_tau_satisfied"] == 1
    assert calibration["tau_to_task_success"] == 1
    assert calibration["tau_precision_for_task_success"] == pytest.approx(1.0)


def test_phase_funnel(battery_root):
    episodes = load_battery_episodes(battery_root)
    funnel = compute_phase_funnel(episodes)
    assert funnel == {
        "episodes": 2,
        "grasp": 1,
        "transport": 1,
        "placement": 1,
        "success": 1,
    }


def test_grounding_stability(battery_root):
    episodes = load_battery_episodes(battery_root)
    stability = compute_grounding_stability(episodes)
    assert stability["targets_regrounded"] == 1
    key = next(iter(stability["per_target"]))
    assert stability["per_target"][key]["regroundings"] == 2
    assert stability["per_target"][key]["max_spread_m"] == pytest.approx(
        (3 * 0.005**2) ** 0.5, rel=1e-6
    )


def test_build_report_and_render_charts(battery_root, tmp_path):
    report = build_report(battery_root)
    assert report["classification"] == "beta_only"
    assert report["protocol_name"] == "benchmark_frozen_v1"
    assert report["condition"] == "harness"
    assert report["success"]["task_success_rate"] == pytest.approx(0.5)
    charts = render_charts(report, tmp_path / "charts")
    names = {path.name for path in charts}
    assert names == {
        "primitive_usage_failures.png",
        "termination_reasons.png",
        "latency_breakdown.png",
        "latency_per_primitive.png",
        "phase_funnel.png",
    }
    for path in charts:
        assert path.stat().st_size > 0
