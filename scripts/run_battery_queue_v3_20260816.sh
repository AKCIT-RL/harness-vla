#!/usr/bin/env bash
# Fila v3 (beta-only) — 2026-08-16, iteração 6 (commit 7c6b0b1).
set -u
cd /home/user_marcosQueiroz/harness-vla/EmbodiedBench
export CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/user_marcosQueiroz/.local/share/harness-vla/openpi/third_party/libero:/home/user_marcosQueiroz/harness-vla/EmbodiedBench
PY=/home/user_marcosQueiroz/.local/share/harness-vla/openpi/examples/libero/.venv/bin/python
LOGDIR=/home/user_marcosQueiroz/harness-vla/evaluation_runs

run() {
  local id="$1"; shift
  for attempt in 1 2; do
    echo "=== $(date -u +%FT%TZ) START $id attempt=$attempt ==="
    "$PY" run_libero_battery.py --port 8010 --battery-id "$id" "$@" \
      >> "$LOGDIR/$id.log" 2>&1
    local code=$?
    echo "=== $(date -u +%FT%TZ) END $id attempt=$attempt exit=$code ==="
    "$PY" - "$LOGDIR/$id" <<'EOF' && break
import json, sys
d = json.load(open(sys.argv[1] + "/battery_summary.json"))
m = json.load(open(sys.argv[1] + "/battery_manifest.json"))
sys.exit(0 if d.get("episodes_completed", 0) + d.get("episodes_failed", 0) >= m.get("episodes_planned", 1) else 1)
EOF
  done
}

run battery_probe_v2_harness_fix6_20260816 \
  --protocol ../benchmarks/eval_protocol_v2.json --condition harness --seeds 101

run battery_probe_v3_harness_fix6_20260816 \
  --protocol ../benchmarks/eval_protocol_v3.json --condition harness --seeds 101

run battery_probe_v3_baseline_20260816 \
  --protocol ../benchmarks/eval_protocol_v3.json --condition baseline_vla --seeds 101

run battery_frozen_v1_harness_fix6_nothink_20260816 \
  --condition harness --seeds 101 --planner-think off

echo "=== $(date -u +%FT%TZ) QUEUE DONE ==="
