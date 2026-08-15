#!/usr/bin/env bash
# Detached final-eval queue (2026-08-13): survives VS Code closing.
# Lane A (GPU 2): harness seeds 104-110. Lane B (GPU 7): baseline seeds
# 104-110, then probe v3 harness+baseline. Then layer-2 analyses.
set -u

ROOT=/home/user_marcosQueiroz/harness-vla
EB="$ROOT/EmbodiedBench"
RUNS="$ROOT/evaluation_runs"
PY=/home/user_marcosQueiroz/.local/share/harness-vla/openpi/examples/libero/.venv/bin/python
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=/home/user_marcosQueiroz/.local/share/harness-vla/openpi/third_party/libero:$EB

battery() { # gpu protocol condition battery_id extra_args...
    local gpu="$1" proto="$2" cond="$3" bid="$4"; shift 4
    cd "$EB" && CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_libero_battery.py \
        --protocol "../benchmarks/$proto" --condition "$cond" --port 8010 \
        --battery-id "$bid" "$@" >> "$RUNS/$bid.log" 2>&1
    echo "$(date -Is) done $bid exit=$?" >> "$RUNS/final_eval_queue_20260813.status"
}

(
    battery 2 eval_protocol_v1.json harness \
        battery_frozen_v1_harness_final_seeds104_110_20260813 \
        --seeds 104 105 106 107 108 109 110
) &
LANE_A=$!

(
    battery 7 eval_protocol_v1.json baseline_vla \
        battery_frozen_v1_baseline_final_seeds104_110_20260813 \
        --seeds 104 105 106 107 108 109 110
    battery 7 eval_protocol_v3.json harness \
        battery_probe_v3_harness_20260813 --seeds 101
    battery 7 eval_protocol_v3.json baseline_vla \
        battery_probe_v3_baseline_20260813 --seeds 101
) &
LANE_B=$!

wait "$LANE_A" "$LANE_B"

cd "$EB"
for b in battery_frozen_v1_harness_final_seeds104_110_20260813 \
         battery_frozen_v1_baseline_final_seeds104_110_20260813 \
         battery_probe_v3_harness_20260813 \
         battery_probe_v3_baseline_20260813; do
    "$ROOT/.venv/bin/python" analyze_libero_battery.py --battery-root "$RUNS/$b" \
        >> "$RUNS/final_eval_queue_20260813.status" 2>&1 \
        && echo "$(date -Is) analysis ok $b" >> "$RUNS/final_eval_queue_20260813.status"
done
echo "$(date -Is) QUEUE COMPLETE" >> "$RUNS/final_eval_queue_20260813.status"
