#!/usr/bin/env bash
# Planner-size sweep (beta-only ablation): how small can the planner LLM be
# before the harness advantage collapses? Reference: gemma4:12b probe_v2
# batteries (same protocol/seeds). gemma3 is the closest available sibling
# lineage with small tags; note the lineage change in the report.
# Waits for the remaining seed-expansion queue before touching the simulator.
set -u
cd "$(dirname "$0")/../.."
PY=/home/user_marcosQueiroz/.local/share/harness-vla/openpi/examples/libero/.venv/bin/python
export CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/user_marcosQueiroz/.local/share/harness-vla/openpi/third_party/libero:$PWD:$PWD/EmbodiedBench

SEED_LOG=evaluation_runs/seed_expansion_remaining_20260818.log
while ! grep -q "QUEUE DONE" "$SEED_LOG" 2>/dev/null; do
  if ! pgrep -f run_seed_expansion_remaining_20260818.sh >/dev/null; then
    echo "=== seed queue no longer running and no QUEUE DONE marker; proceeding anyway ==="
    break
  fi
  sleep 120
done

run_battery() {
  local battery_id=$1; shift
  for attempt in 1 2; do
    echo "=== $(date -u +%FT%TZ) START $battery_id attempt=$attempt ==="
    "$PY" runners/run_libero_battery.py --port 8010 --battery-id "$battery_id" "$@"
    echo "=== $(date -u +%FT%TZ) END $battery_id attempt=$attempt exit=$? ==="
    local planned done_n
    planned=$(.venv/bin/python -c "import json;print(json.load(open('evaluation_runs/$battery_id/battery_manifest.json'))['episodes_planned'])" 2>/dev/null || echo -1)
    done_n=$(.venv/bin/python -c "import json;d=json.load(open('evaluation_runs/$battery_id/battery_summary.json'));print(d['episodes_completed']+d['episodes_failed'])" 2>/dev/null || echo -2)
    if [ "$planned" = "$done_n" ]; then return 0; fi
    echo "=== incomplete ($done_n/$planned), retrying ==="
  done
  echo "=== $(date -u +%FT%TZ) GAVE UP $battery_id ==="
  return 1
}

run_battery battery_probe_v2_g3_4b_harness_20260818 \
  --protocol benchmarks/eval_protocol_v2_1.json --condition harness \
  --planner-model gemma3:4b --seeds 101 102 103
run_battery battery_probe_v2_g3_1b_harness_20260818 \
  --protocol benchmarks/eval_protocol_v2_1.json --condition harness \
  --planner-model gemma3:1b --seeds 101 102 103

echo "=== $(date -u +%FT%TZ) SWEEP DONE ==="
