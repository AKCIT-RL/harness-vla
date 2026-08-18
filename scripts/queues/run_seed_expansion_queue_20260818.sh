#!/usr/bin/env bash
# Seed expansion queue: bring every experiment to >=3 seeds (102, 103 added).
# Protocols v2_1/v3_1 extend seeds only; tasks and budgets identical.
set -u
cd "$(dirname "$0")/../.."
PY=/home/user_marcosQueiroz/.local/share/harness-vla/openpi/examples/libero/.venv/bin/python
export CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/user_marcosQueiroz/.local/share/harness-vla/openpi/third_party/libero:/home/user_marcosQueiroz/harness-vla/EmbodiedBench

run_battery() {
  local battery_id=$1; shift
  for attempt in 1 2; do
    echo "=== $(date -u +%FT%TZ) START $battery_id attempt=$attempt ==="
    (cd EmbodiedBench && "$PY" ../runners/run_libero_battery.py --port 8010 \
      --battery-id "$battery_id" "$@")
    local code=$?
    echo "=== $(date -u +%FT%TZ) END $battery_id attempt=$attempt exit=$code ==="
    local planned done_n
    planned=$(.venv/bin/python -c "import json;print(json.load(open('evaluation_runs/$battery_id/battery_manifest.json'))['episodes_planned'])" 2>/dev/null || echo -1)
    done_n=$(.venv/bin/python -c "import json;d=json.load(open('evaluation_runs/$battery_id/battery_summary.json'));print(d['episodes_completed']+d['episodes_failed'])" 2>/dev/null || echo -2)
    if [ "$planned" = "$done_n" ]; then return 0; fi
    echo "=== incomplete ($done_n/$planned), retrying ==="
  done
  echo "=== $(date -u +%FT%TZ) GAVE UP $battery_id ==="
  return 1
}

# fast batteries first (baselines + nothink), heavy harness after
run_battery battery_probe_v2_seeds_baseline_20260818 \
  --protocol ../benchmarks/eval_protocol_v2_1.json --condition baseline_vla \
  --seeds 102 103
run_battery battery_probe_v3_seeds_baseline_20260818 \
  --protocol ../benchmarks/eval_protocol_v3_1.json --condition baseline_vla \
  --seeds 102 103
run_battery battery_probe_v2_hidden_baseline_seeds_20260818 \
  --protocol ../benchmarks/eval_protocol_v2_1.json --condition baseline_vla \
  --hidden-only --seeds 102 103
run_battery battery_frozen_v1_nothink_seeds_20260818 \
  --protocol ../benchmarks/eval_protocol_v1.json --condition harness \
  --planner-think off --seeds 102 103
run_battery battery_probe_v2_seeds_harness_20260818 \
  --protocol ../benchmarks/eval_protocol_v2_1.json --condition harness \
  --seeds 102 103
run_battery battery_probe_v3_seeds_harness_20260818 \
  --protocol ../benchmarks/eval_protocol_v3_1.json --condition harness \
  --seeds 102 103
run_battery battery_probe_v2_hidden_harness_seeds_20260818 \
  --protocol ../benchmarks/eval_protocol_v2_1.json --condition harness \
  --hidden-only --seeds 102 103

echo "=== $(date -u +%FT%TZ) QUEUE DONE ==="
