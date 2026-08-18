#!/usr/bin/env bash
# Fila sequencial de baterias (beta-only) — 2026-08-15, iteração 4 (commit 94a1159).
# Regra: nunca duas instâncias do simulador em paralelo; ordem importa.
set -u
cd /home/user_marcosQueiroz/harness-vla/EmbodiedBench
export CUDA_VISIBLE_DEVICES=5 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/user_marcosQueiroz/.local/share/harness-vla/openpi/third_party/libero:/home/user_marcosQueiroz/harness-vla/EmbodiedBench
PY=/home/user_marcosQueiroz/.local/share/harness-vla/openpi/examples/libero/.venv/bin/python
LOGDIR=/home/user_marcosQueiroz/harness-vla/evaluation_runs

run() {
  local id="$1"; shift
  echo "=== $(date -u +%FT%TZ) START $id ==="
  "$PY" ../runners/run_libero_battery.py --port 8010 --battery-id "$id" "$@" \
    >> "$LOGDIR/$id.log" 2>&1
  echo "=== $(date -u +%FT%TZ) END $id exit=$? ==="
}

# 1. Verificação da correção multi-objeto (item 1)
run battery_probe_v2_harness_fix4_20260815 \
  --protocol ../benchmarks/eval_protocol_v2.json --condition harness --seeds 101

# 2. Regressão na bateria congelada v1 (item 2)
run battery_frozen_v1_harness_fix4_20260815 \
  --condition harness --seeds 101 102 103

# 3. Sonda v3 libero_90: harness e baseline pareados (item 3)
run battery_probe_v3_harness_20260815 \
  --protocol ../benchmarks/eval_protocol_v3.json --condition harness --seeds 101

run battery_probe_v3_baseline_20260815 \
  --protocol ../benchmarks/eval_protocol_v3.json --condition baseline_vla --seeds 101

# 4. Ablação de eficiência: planner sem thinking (item 4)
run battery_frozen_v1_harness_fix4_nothink_20260815 \
  --condition harness --seeds 101 --planner-think off

echo "=== $(date -u +%FT%TZ) QUEUE DONE ==="
