# benchmark_frozen_v1 — Frozen Evaluation Battery (beta-only)

**Classification: beta-only.** This battery is evaluation infrastructure for
the Harness VLA beta. It is not a mechanism from arXiv:2607.08448v2; the paper
does not prescribe a specific evaluation battery for this implementation.

## Purpose

Meeting feedback addressed by this battery:

1. **More seeds** — 10 declared deployment seeds (101–110) plus the declared
   bootstrap seed (7).
2. **Fixed envs as a stable baseline** — 4 frozen tasks × 2 initial states,
   identical across every run, plus 2 hidden tasks reserved for final
   evaluation only.
3. **Per-action logs** — every episode now writes `actions.jsonl` (one line
   per physical 7D action) next to the authoritative `trace.jsonl`.
4. **Timestamp profiling** — every turn record in `trace.jsonl` carries a
   `timing` block (`ts`, `planner_duration_s`, `execution_duration_s`,
   `turn_duration_s`, `mean_step_duration_s`), and each VLA chunk already
   records `inference_duration_s`.
5. **Intermediate metrics** — the analysis layer reports per-primitive usage
   and failure rates, parse/compile health, tau→task-success calibration,
   phase funnel, grounding stability and latency percentiles, with charts.

## Files

| File | Role |
|---|---|
| [benchmarks/eval_protocol_v1.json](../benchmarks/eval_protocol_v1.json) | The frozen protocol: tasks, states, seeds, budgets, pairing rule. Never edit; create `v2` instead. |
| [EmbodiedBench/run_libero_battery.py](../EmbodiedBench/run_libero_battery.py) | Crash-tolerant sequential battery runner (`harness` or `baseline_vla` condition), resumable per episode. |
| [EmbodiedBench/analyze_libero_battery.py](../EmbodiedBench/analyze_libero_battery.py) | Layer-2 analysis: `analysis/report.json` + PNG charts. |
| [EmbodiedBench/embodiedbench/evaluator/libero_battery.py](../EmbodiedBench/embodiedbench/evaluator/libero_battery.py) | Matrix expansion, resume, aggregation (pure, tested). |
| [EmbodiedBench/embodiedbench/evaluator/libero_battery_analysis.py](../EmbodiedBench/embodiedbench/evaluator/libero_battery_analysis.py) | Metric computations (pure, tested). |
| [EmbodiedBench/embodiedbench/evaluator/libero_action_log.py](../EmbodiedBench/embodiedbench/evaluator/libero_action_log.py) | Flat per-action projection written as `actions.jsonl`. |

## Protocol contents

- **Tasks (visible)**: `libero_spatial` tasks 0 and 3; `libero_object` tasks 0
  and 5 — each with initial states 0 and 1. Task 0/state 0 (bootstrap seed 7)
  and task 0/state 1 (deployment seed 101) are the already-validated lifecycle
  episodes.
- **Hidden tasks**: `libero_spatial` task 7 and `libero_object` task 8
  (state 0). Use only for the final held-out evaluation; never during
  development or tuning.
- **Seeds**: bootstrap 7; deployment 101–110. The runner refuses seeds not
  declared here.
- **Budgets/offsets**: identical to the validated single-episode runner
  (`max_turns 8`, `horizon 220`, `max_chunks_cap 20`, `max_move_steps 40`,
  `release_steps 10`; offsets 0.10/0.05 m; tolerance 0.01 m).

## Pairing rule

A comparison between conditions (or code revisions) is valid **only** over the
identical set of `(task_suite, task_id, initial_state_index, seed)` tuples
completed by both sides. `action_success` never substitutes for
`task_success`: only the official `env.check_success` counts as success.

## How to run

From `EmbodiedBench/`, with the validated LIBERO runtime environment active,
the pi0.5 websocket server running, and Ollama serving the planner model
(one simulator instance at a time):

```bash
# Harness condition, first three deployment seeds
python run_libero_battery.py --condition harness --seeds 101 102 103

# Paired baseline (frozen VLA only, no planner)
python run_libero_battery.py --condition baseline_vla --seeds 101 102 103

# Resume a crashed battery: pass the same --battery-id
python run_libero_battery.py --condition harness --seeds 101 102 103 \
    --battery-id battery_benchmark_frozen_v1_harness_YYYYMMDD_HHMMSS
```

Analysis (CPU only, no simulator):

```bash
python analyze_libero_battery.py \
    --battery-root ../evaluation_runs/battery_benchmark_frozen_v1_harness_...
```

Outputs: `analysis/report.json` plus `primitive_usage_failures.png`,
`termination_reasons.png`, `latency_breakdown.png`,
`latency_per_primitive.png`, `phase_funnel.png`.

## Artifact layout

```
evaluation_runs/<battery_id>/
  battery_manifest.json      # protocol copy, condition, git commit, status
  battery_summary.json       # rewritten after every episode (crash-safe)
  episodes/
    <condition>_<suite>_taskNN_stateNN_seedNNN/
      run_manifest.json      # per-episode provenance
      episode.json           # official outcome (authoritative task_success)
      summary.json
      trace.jsonl            # authoritative per-turn record (+ timing block)
      actions.jsonl          # beta-only flat per-7D-action log
      videos/
  analysis/                  # written by analyze_libero_battery.py
```

## Invariants

- Never edit `eval_protocol_v1.json`; version a new file for any change.
- Never change planner, primitives or budgets in the middle of a battery.
- Never delete a battery folder without extracting its `analysis/report.json`.
- Hidden tasks are consumed exactly once, at final evaluation.
