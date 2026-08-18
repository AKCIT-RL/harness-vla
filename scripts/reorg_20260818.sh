#!/usr/bin/env bash
# One-shot repo reorganization: extract the agent into a top-level harness_vla/
# package, split our LIBERO evaluators from the vendored EmbodiedBench fork,
# and group runners/scripts. Import rewrites are mechanical (see mapping).
set -euo pipefail
cd "$(dirname "$0")/.."

git tag -f pre-reorg-20260818

H=EmbodiedBench/embodiedbench/planner/harness
EV=EmbodiedBench/embodiedbench/evaluator

# --- 1. harness_vla package skeleton -------------------------------------
mkdir -p harness_vla/{planner,primitives,memory,perception,backends,eval}

git mv "$H/__init__.py" harness_vla/__init__.py
for m in harness_planner libero_vla_planner libero_multi_turn_planner navigation_planner prompts phase_policy; do
  git mv "$H/$m.py" "harness_vla/planner/$m.py"
done
for m in primitives libero_primitives libero_tau; do
  git mv "$H/$m.py" "harness_vla/primitives/$m.py"
done
for m in task_memory global_memory; do
  git mv "$H/$m.py" "harness_vla/memory/$m.py"
done
for m in libero_grounding libero_visual_grounding; do
  git mv "$H/$m.py" "harness_vla/perception/$m.py"
done
for m in pirlinf_backend openvla_backend vla_runtime; do
  git mv "$H/$m.py" "harness_vla/backends/$m.py"
done
for m in evaluation_guards trace_io; do
  git mv "$H/$m.py" "harness_vla/$m.py"
done
[ -f "$H/nav_global_memory.json" ] && mv "$H/nav_global_memory.json" harness_vla/memory/
rm -rf "$H/__pycache__"; rmdir "$H"
for p in planner primitives memory perception backends eval; do
  touch "harness_vla/$p/__init__.py"; git add "harness_vla/$p/__init__.py"
done

# --- 2. our LIBERO evaluators -> harness_vla/eval ------------------------
for m in libero_battery libero_battery_analysis libero_analytic_executor \
         libero_action_log libero_file_repl libero_file_repl_bridge \
         libero_grounding_smoke libero_memory_lifecycle \
         libero_multi_turn_evaluator libero_multi_turn_run \
         libero_native_multi_turn libero_privileged_diagnostics \
         libero_vla_smoke; do
  git mv "$EV/$m.py" "harness_vla/eval/$m.py"
done

# --- 3. LIBERO runners -> runners/ ---------------------------------------
mkdir -p runners
for r in run_libero_battery run_libero_multi_turn run_libero_vla_smoke \
         run_libero_grounding_smoke analyze_libero_battery; do
  git mv "EmbodiedBench/$r.py" "runners/$r.py"
done

# --- 4. scripts/ subfolders ----------------------------------------------
mkdir -p scripts/queues scripts/analysis
git mv scripts/run_battery_queue_20260815.sh scripts/run_battery_queue_v2_20260815.sh \
       scripts/run_battery_queue_v3_20260816.sh scripts/run_final_eval_queue_20260813.sh \
       scripts/run_hidden_eval_queue_20260817.sh scripts/run_seed_expansion_queue_20260818.sh \
       scripts/queues/
git mv scripts/analyze_card_metrics_20260818.py scripts/make_card_charts_20260817.py scripts/analysis/

# --- 5. stray logs out of the vendored fork root --------------------------
mkdir -p evaluation_runs/legacy_logs
mv EmbodiedBench/*.log evaluation_runs/legacy_logs/ 2>/dev/null || true

# --- 6. import rewrite (most specific first) ------------------------------
mapfile -t FILES < <(grep -rl "embodiedbench\.planner\.harness\|embodiedbench\.evaluator\.libero_" \
  --include="*.py" harness_vla tests runners scripts EmbodiedBench 2>/dev/null | grep -v __pycache__)
for f in "${FILES[@]}"; do
  sed -i \
    -e 's/embodiedbench\.planner\.harness\.\(harness_planner\|libero_vla_planner\|libero_multi_turn_planner\|navigation_planner\|prompts\|phase_policy\)/harness_vla.planner.\1/g' \
    -e 's/embodiedbench\.planner\.harness\.\(primitives\|libero_primitives\|libero_tau\)/harness_vla.primitives.\1/g' \
    -e 's/embodiedbench\.planner\.harness\.\(task_memory\|global_memory\)/harness_vla.memory.\1/g' \
    -e 's/embodiedbench\.planner\.harness\.\(libero_grounding\|libero_visual_grounding\)/harness_vla.perception.\1/g' \
    -e 's/embodiedbench\.planner\.harness\.\(pirlinf_backend\|openvla_backend\|vla_runtime\)/harness_vla.backends.\1/g' \
    -e 's/embodiedbench\.planner\.harness\.\(evaluation_guards\|trace_io\)/harness_vla.\1/g' \
    -e 's/embodiedbench\.planner\.harness/harness_vla/g' \
    -e 's/embodiedbench\.evaluator\.\(libero_[a-z_]*\)/harness_vla.eval.\1/g' \
    "$f"
done

echo "reorg done"
