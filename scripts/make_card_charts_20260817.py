"""Generate card charts from battery traces (beta-only instrumentation)."""
import json
import glob
import statistics
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "evaluation_runs/card_assets_20260817"
import os
os.makedirs(OUT, exist_ok=True)

HARNESS_BATTERIES = [
    "battery_frozen_v1_harness_fix5_20260815",
    "battery_probe_v2_harness_fix6_20260816",
    "battery_probe_v3_harness_fix6_20260816",
]

def turns(battery):
    for f in glob.glob(f"evaluation_runs/{battery}/episodes/*/trace.jsonl"):
        for line in open(f):
            yield f, json.loads(line)

# 1) success rates
fig, ax = plt.subplots(figsize=(8, 4))
suites = ["libero_spatial+object\n(frozen_v1, 24 ep)", "libero_goal+10\n(probe_v2, 8 ep)", "libero_90\n(probe_v3, 5 ep)", "HELD-OUT hidden tasks\n(first look, 8 ep)"]
baseline = [50.0, 25.0, 20.0, 12.5]
harness = [95.8, 75.0, 60.0, 75.0]
x = range(len(suites))
ax.bar([i - 0.2 for i in x], baseline, width=0.4, label="VLA-only baseline", color="#999")
ax.bar([i + 0.2 for i in x], harness, width=0.4, label="Harness", color="#2b7bba")
for i, (b, h) in enumerate(zip(baseline, harness)):
    ax.text(i - 0.2, b + 1.5, f"{b:.0f}%", ha="center")
    ax.text(i + 0.2, h + 1.5, f"{h:.0f}%", ha="center")
ax.set_xticks(list(x)); ax.set_xticklabels(suites)
ax.set_ylabel("Task success (%)"); ax.set_ylim(0, 105)
ax.set_title("Paired task success: frozen pi0.5 alone vs. inside the harness")
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/success_rates.png", dpi=150); plt.close(fig)

# 2) primitive mix per battery
fig, ax = plt.subplots(figsize=(7, 4))
labels = ["frozen_v1", "probe_v2", "probe_v3"]
mixes = []
for b in HARNESS_BATTERIES:
    c = Counter()
    for _, d in turns(b):
        inv = d.get("invocation") or {}
        p = inv.get("action")
        if p:
            c[p] += 1
    mixes.append(c)
prims = ["vla_act", "move_to", "release"]
colors = {"vla_act": "#2b7bba", "move_to": "#e6842a", "release": "#54a24b"}
bottoms = [0] * len(labels)
for p in prims:
    vals = [m.get(p, 0) for m in mixes]
    ax.bar(labels, vals, bottom=bottoms, label=p, color=colors[p])
    for i, v in enumerate(vals):
        if v:
            ax.text(i, bottoms[i] + v / 2, str(v), ha="center", va="center", color="white")
    bottoms = [b0 + v for b0, v in zip(bottoms, vals)]
ax.set_ylabel("Primitive invocations (all episodes)")
ax.set_title("What the planner calls: neural vla_act vs. analytic primitives")
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/primitive_mix.png", dpi=150); plt.close(fig)

# 3) wall-clock split per turn, think vs no-think
def timing_means(pattern):
    plan, execu = [], []
    for f in glob.glob(pattern):
        for line in open(f):
            t = json.loads(line).get("timing") or {}
            if t.get("planner_duration_s") is not None:
                plan.append(t["planner_duration_s"])
                execu.append(t.get("execution_duration_s") or 0)
    return statistics.mean(plan), statistics.mean(execu)

p_think, e_think = timing_means("evaluation_runs/battery_frozen_v1_harness_fix5_20260815/episodes/*seed101/trace.jsonl")
p_nt, e_nt = timing_means("evaluation_runs/battery_frozen_v1_harness_fix6_nothink_20260816/episodes/*/trace.jsonl")
fig, ax = plt.subplots(figsize=(6, 4))
cfg = ["think=on\n(8/8 success)", "think=off\n(7/8 success)"]
ax.bar(cfg, [p_think, p_nt], width=0.5, label="Planner (LLM) time", color="#c44")
ax.bar(cfg, [e_think, e_nt], width=0.5, bottom=[p_think, p_nt], label="Robot execution time", color="#2b7bba")
for i, (p, e) in enumerate([(p_think, e_think), (p_nt, e_nt)]):
    ax.text(i, p / 2, f"{p:.0f}s", ha="center", color="white")
    ax.text(i, p + e / 2, f"{e:.0f}s", ha="center", color="white")
ax.set_ylabel("Mean seconds per turn")
ax.set_title("Where a turn's wall-clock goes (same 8 episodes, seed 101)")
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/time_split.png", dpi=150); plt.close(fig)

# 4) simulator steps: VLA vs analytic control
steps = Counter()
for b in HARNESS_BATTERIES:
    for _, d in turns(b):
        inv = d.get("invocation") or {}
        p = inv.get("action")
        n = len(d.get("step_results") or [])
        if p:
            steps["vla_act" if p == "vla_act" else "analytic"] += n
fig, ax = plt.subplots(figsize=(5, 4))
total = sum(steps.values())
ax.pie([steps["vla_act"], steps["analytic"]],
       labels=[f"vla_act\n{steps['vla_act']} steps ({steps['vla_act']/total:.0%})",
               f"analytic (move_to/release)\n{steps['analytic']} steps ({steps['analytic']/total:.0%})"],
       colors=["#2b7bba", "#e6842a"], startangle=90)
ax.set_title("Simulator steps by controller (all harness episodes)")
fig.tight_layout(); fig.savefig(f"{OUT}/steps_by_controller.png", dpi=150); plt.close(fig)

# 5) turns to success histogram
tts = []
for b in HARNESS_BATTERIES:
    for f in glob.glob(f"evaluation_runs/{b}/episodes/*/episode.json"):
        d = json.load(open(f))
        if d.get("task_success"):
            tts.append(d["turns_executed"])
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(tts, bins=range(1, max(tts) + 2), color="#54a24b", edgecolor="white", align="left")
ax.set_xlabel("Turns until task success"); ax.set_ylabel("Episodes")
ax.set_title(f"Turns to success across all successful harness episodes (n={len(tts)})")
fig.tight_layout(); fig.savefig(f"{OUT}/turns_to_success.png", dpi=150); plt.close(fig)

print("mean turns to success:", statistics.mean(tts))
print("primitive mixes:", mixes)
print("steps:", dict(steps))
print(f"time think: plan={p_think:.0f}s exec={e_think:.0f}s | nothink: plan={p_nt:.0f}s exec={e_nt:.0f}s")
print("done ->", OUT)
