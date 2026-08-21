#!/usr/bin/env bash
# Planner-size sweep v2 (beta-only ablation). v1 was invalidated: gemma3 models
# reject Ollama's think flag ("does not support thinking"), so every turn was a
# planner_parse_error. All conditions now run with --planner-think off,
# including the gemma4:12b reference, so the size axis is the only variable.
set -u
cd "$(dirname "$0")/../.."
PY=/home/user_marcosQueiroz/.local/share/harness-vla/openpi/examples/libero/.venv/bin/python
export CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/user_marcosQueiroz/.local/share/harness-vla/openpi/third_party/libero:$PWD:$PWD/EmbodiedBench

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

run_battery battery_probe_v2_g3_4b_nothink_harness_20260821 \
  --protocol benchmarks/eval_protocol_v2_1.json --condition harness \
  --planner-model gemma3:4b --planner-think off --seeds 101 102 103
run_battery battery_probe_v2_g3_1b_nothink_harness_20260821 \
  --protocol benchmarks/eval_protocol_v2_1.json --condition harness \
  --planner-model gemma3:1b --planner-think off --seeds 101 102 103
run_battery battery_probe_v2_12b_nothink_harness_20260821 \
  --protocol benchmarks/eval_protocol_v2_1.json --condition harness \
  --planner-think off --seeds 101 102 103

echo "=== $(date -u +%FT%TZ) SWEEP V2 DONE ==="
