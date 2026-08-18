"""Per-task and behavioral metrics from battery traces (beta-only analysis).

Rerunnable: seed-expansion batteries are included automatically when present.
Outputs charts + metrics.json to evaluation_runs/card_metrics_20260818/.
"""
import json
import glob
import os
import statistics
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "evaluation_runs/card_metrics_20260818"
os.makedirs(OUT, exist_ok=True)

HARNESS = [
    "battery_frozen_v1_harness_fix5_20260815",
    "battery_probe_v2_harness_fix6_20260816",
    "battery_probe_v3_harness_fix6_20260816",
    "battery_frozen_v1_hidden_harness_20260817",
    "battery_probe_v2_hidden_harness_20260817",
    "battery_probe_v2_seeds_harness_20260818",
    "battery_probe_v3_seeds_harness_20260818",
    "battery_probe_v2_hidden_harness_seeds_20260818",
]
BASELINE = [
    "battery_frozen_v1_baseline_20260812",
    "battery_probe_v2_baseline_20260812",
    "battery_probe_v3_baseline_20260816",
    "battery_frozen_v1_hidden_baseline_20260817",
    "battery_probe_v2_hidden_baseline_20260817",
    "battery_probe_v2_seeds_baseline_20260818",
    "battery_probe_v3_seeds_baseline_20260818",
    "battery_probe_v2_hidden_baseline_seeds_20260818",
]

def episodes(batteries):
    for b in batteries:
        for ep in sorted(glob.glob(f"evaluation_runs/{b}/episodes/*/episode.json")):
            d = json.load(open(ep))
            d["_root"] = os.path.dirname(ep)
            d["_battery"] = b
            if d.get("seed") is None:  # baseline episodes omit the seed field
                name = os.path.basename(d["_root"])
                if "_seed" in name:
                    d["seed"] = int(name.rsplit("_seed", 1)[1])
            yield d

def turns_of(ep_root):
    path = os.path.join(ep_root, "trace.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path)]

def task_key(d):
    return f"{d['task_suite']}#{d['task_id']}"

def tuple_key(d):
    return (d["task_suite"], d["task_id"], d["initial_state_index"], d.get("seed"))

def turn_steps(turn):
    return sum(len(c.get("executed_actions") or []) for c in (turn.get("primitive_trace") or []))

harness_eps = [d for d in episodes(HARNESS)]
baseline_eps = [d for d in episodes(BASELINE)]
present = sorted({d["_battery"] for d in harness_eps + baseline_eps})
print("batteries found:", *present, sep="\n  ")

# ---------- collect per-episode / per-turn records ----------
records = []  # harness episodes with turn details
for d in harness_eps:
    turns = turns_of(d["_root"])
    rec = {
        "task": task_key(d), "tuple": tuple_key(d), "seed": d.get("seed"),
        "success": bool(d.get("task_success")), "turns": d.get("turns_executed"),
        "battery": d["_battery"], "turn_data": [],
    }
    for t in turns:
        inv = t.get("invocation") or {}
        rec["turn_data"].append({
            "action": inv.get("action"), "target": inv.get("target"),
            "tau": inv.get("tau"), "max_chunks": inv.get("max_chunks"),
            "ok": bool(t.get("primitive_postcondition_met")),
            "status": t.get("status"),
            "chunks": len(t.get("primitive_trace") or []),
            "steps": turn_steps(t),
            "planner_s": (t.get("timing") or {}).get("planner_duration_s"),
            "exec_s": (t.get("timing") or {}).get("execution_duration_s"),
            "think_chars": len(t.get("planner_thinking") or ""),
        })
    records.append(rec)

base_by_tuple = {tuple_key(d): d for d in baseline_eps}

tasks = sorted({r["task"] for r in records})
def per_task(fn):
    out = {}
    for task in tasks:
        vals = [fn(r) for r in records if r["task"] == task]
        vals = [v for v in vals if v is not None]
        out[task] = vals
    return out

def save_bar(fname, title, ylabel, data, fmt="{:.0f}", color="#2b7bba", ylim=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    keys = list(data)
    vals = [data[k] for k in keys]
    ax.bar(range(len(keys)), vals, color=color)
    for i, v in enumerate(vals):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("#", "\n#") for k in keys], fontsize=7)
    ax.set_ylabel(ylabel); ax.set_title(title)
    if ylim: ax.set_ylim(*ylim)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=150); plt.close(fig)

metrics = {"batteries": present}

# A1) success by task, harness vs baseline
fig, ax = plt.subplots(figsize=(11, 4))
h_rate, b_rate, err = [], [], []
for task in tasks:
    hs = [r["success"] for r in records if r["task"] == task]
    h_rate.append(100 * sum(hs) / len(hs))
    bs = [bool(base_by_tuple[r["tuple"]].get("task_success"))
          for r in records if r["task"] == task and r["tuple"] in base_by_tuple]
    b_rate.append(100 * sum(bs) / len(bs) if bs else float("nan"))
    by_seed = defaultdict(list)
    for r in records:
        if r["task"] == task:
            by_seed[r["seed"]].append(r["success"])
    seed_rates = [100 * sum(v) / len(v) for v in by_seed.values()]
    err.append(statistics.stdev(seed_rates) if len(seed_rates) > 1 else 0.0)
x = range(len(tasks))
ax.bar([i - 0.2 for i in x], b_rate, width=0.4, label="VLA-only baseline", color="#999")
ax.bar([i + 0.2 for i in x], h_rate, width=0.4, yerr=err, capsize=3,
       label="Harness (err = std across seeds)", color="#2b7bba")
ax.set_xticks(list(x)); ax.set_xticklabels([t.replace("#", "\n#") for t in tasks], fontsize=7)
ax.set_ylabel("Task success (%)"); ax.set_ylim(0, 115)
ax.set_title("Per-task paired success"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/success_by_task.png", dpi=150); plt.close(fig)
metrics["success_by_task"] = {t: {"harness_pct": h, "baseline_pct": b, "seed_std": e}
                              for t, h, b, e in zip(tasks, h_rate, b_rate, err)}

# A2) mean turns to success per task
tt = per_task(lambda r: r["turns"] if r["success"] else None)
save_bar("turns_by_task.png", "Mean turns to success (harness successes only)",
         "turns", {k: statistics.mean(v) for k, v in tt.items() if v}, fmt="{:.1f}",
         color="#54a24b")
metrics["turns_to_success_by_task"] = {k: statistics.mean(v) for k, v in tt.items() if v}

# A3) primitive mix per task
fig, ax = plt.subplots(figsize=(11, 4))
prims = ["vla_act", "move_to", "release"]
colors = {"vla_act": "#2b7bba", "move_to": "#e6842a", "release": "#54a24b"}
bottoms = [0.0] * len(tasks)
mix = {t: Counter() for t in tasks}
for r in records:
    for td in r["turn_data"]:
        if td["action"]:
            mix[r["task"]][td["action"]] += 1
for p in prims:
    vals = [mix[t].get(p, 0) for t in tasks]
    ax.bar(range(len(tasks)), vals, bottom=bottoms, label=p, color=colors[p])
    bottoms = [b0 + v for b0, v in zip(bottoms, vals)]
ax.set_xticks(range(len(tasks))); ax.set_xticklabels([t.replace("#", "\n#") for t in tasks], fontsize=7)
ax.set_ylabel("invocations"); ax.set_title("Primitive mix per task"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/primitive_mix_by_task.png", dpi=150); plt.close(fig)
metrics["primitive_mix_by_task"] = {t: dict(c) for t, c in mix.items()}

# A4) planner vs execution time per task
fig, ax = plt.subplots(figsize=(11, 4))
plan_m, exec_m = [], []
for task in tasks:
    ps = [td["planner_s"] for r in records if r["task"] == task
          for td in r["turn_data"] if td["planner_s"]]
    es = [td["exec_s"] for r in records if r["task"] == task
          for td in r["turn_data"] if td["exec_s"]]
    plan_m.append(statistics.mean(ps) if ps else 0)
    exec_m.append(statistics.mean(es) if es else 0)
ax.bar(range(len(tasks)), plan_m, label="planner (LLM)", color="#c44")
ax.bar(range(len(tasks)), exec_m, bottom=plan_m, label="robot execution", color="#2b7bba")
ax.set_xticks(range(len(tasks))); ax.set_xticklabels([t.replace("#", "\n#") for t in tasks], fontsize=7)
ax.set_ylabel("mean s/turn"); ax.set_title("Wall-clock per turn, per task"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/time_by_task.png", dpi=150); plt.close(fig)

# B5) vla_act attempt success per task (all attempts + first attempt of episode)
attempt, first = {}, {}
for task in tasks:
    alls, firsts = [], []
    for r in records:
        if r["task"] != task:
            continue
        vla = [td for td in r["turn_data"] if td["action"] == "vla_act"]
        alls.extend(td["ok"] for td in vla)
        if vla:
            firsts.append(vla[0]["ok"])
    attempt[task] = 100 * sum(alls) / len(alls) if alls else None
    first[task] = 100 * sum(firsts) / len(firsts) if firsts else None
fig, ax = plt.subplots(figsize=(11, 4))
x = range(len(tasks))
ax.bar([i - 0.2 for i in x], [first[t] or 0 for t in tasks], width=0.4,
       label="first vla_act of episode", color="#2b7bba")
ax.bar([i + 0.2 for i in x], [attempt[t] or 0 for t in tasks], width=0.4,
       label="all vla_act attempts", color="#89b8d9")
ax.set_xticks(list(x)); ax.set_xticklabels([t.replace("#", "\n#") for t in tasks], fontsize=7)
ax.set_ylabel("tau satisfied (%)"); ax.set_ylim(0, 115)
ax.set_title("Frozen-VLA competence per task: vla_act postcondition satisfaction")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/vla_attempt_success_by_task.png", dpi=150); plt.close(fig)
metrics["vla_attempt_success_by_task"] = {
    t: {"first_pct": first[t], "all_pct": attempt[t]} for t in tasks}

# B6) recovery rate
with_fail = [r for r in records if any(not td["ok"] for td in r["turn_data"])]
saved = [r for r in with_fail if r["success"]]
metrics["recovery"] = {
    "episodes_total": len(records),
    "episodes_with_primitive_failure": len(with_fail),
    "episodes_recovered": len(saved),
    "recovery_rate_pct": 100 * len(saved) / len(with_fail) if with_fail else None,
    "clean_successes": sum(1 for r in records
                           if r["success"] and all(td["ok"] for td in r["turn_data"])),
}
fig, ax = plt.subplots(figsize=(6, 4))
clean = metrics["recovery"]["clean_successes"]
rec_n = len(saved)
fail_n = len(with_fail) - len(saved)
none_n = len(records) - clean - rec_n - fail_n
ax.bar(["clean success\n(no failed\nprimitive)", "recovered\nsuccess", "failure\ndespite replans", "other\nfailures"],
       [clean, rec_n, fail_n, none_n], color=["#54a24b", "#2b7bba", "#c44", "#999"])
for i, v in enumerate([clean, rec_n, fail_n, none_n]):
    ax.text(i, v, str(v), ha="center", va="bottom")
ax.set_ylabel("episodes")
ax.set_title("Recovery: %d/%d episodes with a failed primitive still succeeded (%.0f%%)"
             % (rec_n, len(with_fail), metrics["recovery"]["recovery_rate_pct"] or 0), fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/recovery.png", dpi=150); plt.close(fig)

# B7) max consecutive failed grasps on same target, per task
persist = {}
for task in tasks:
    worst = 0
    for r in records:
        if r["task"] != task:
            continue
        run, prev = 0, None
        for td in r["turn_data"]:
            if td["action"] == "vla_act" and not td["ok"] and td["target"] == prev:
                run += 1
            elif td["action"] == "vla_act" and not td["ok"]:
                run, prev = 1, td["target"]
            else:
                run, prev = 0, None
            worst = max(worst, run)
    persist[task] = worst
save_bar("grasp_persistence_by_task.png",
         "Worst streak of consecutive failed vla_act on the same target",
         "consecutive failures", persist, color="#c44")
metrics["max_consecutive_failed_vla_by_task"] = persist

# B8) turn status distribution
status = Counter(td["status"] for r in records for td in r["turn_data"] if td["status"])
fig, ax = plt.subplots(figsize=(8, 4))
keys = [k for k, _ in status.most_common()]
ax.barh(keys, [status[k] for k in keys], color="#2b7bba")
for i, k in enumerate(keys):
    ax.text(status[k], i, f" {status[k]}", va="center", fontsize=8)
ax.set_xlabel("turns"); ax.set_title("Turn outcome distribution (feedback the planner sees)")
fig.tight_layout(); fig.savefig(f"{OUT}/turn_status_distribution.png", dpi=150); plt.close(fig)
metrics["turn_status_distribution"] = dict(status)

# C9) physical steps to success on paired successes
pairs = []
for r in records:
    b = base_by_tuple.get(r["tuple"])
    if b is None or not r["success"] or not b.get("task_success"):
        continue
    pairs.append((r["tuple"], sum(td["steps"] for td in r["turn_data"]),
                  b.get("actions_executed")))
if pairs:
    fig, ax = plt.subplots(figsize=(6, 5))
    hs = [p[1] for p in pairs]; bs = [p[2] for p in pairs]
    lim = max(hs + bs) * 1.1
    ax.scatter(bs, hs, color="#2b7bba")
    ax.plot([0, lim], [0, lim], "--", color="#999", label="equal steps")
    ax.set_xlabel("baseline steps to success"); ax.set_ylabel("harness steps to success")
    ax.set_title("Robot-time efficiency on episodes both solve (n=%d)" % len(pairs))
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/steps_paired_successes.png", dpi=150); plt.close(fig)
    metrics["steps_paired_successes"] = {
        "n": len(pairs),
        "harness_mean": statistics.mean(hs), "baseline_mean": statistics.mean(bs),
    }

# C10) chunks used vs cap
ratios = [td["chunks"] / td["max_chunks"] for r in records for td in r["turn_data"]
          if td["action"] == "vla_act" and td["max_chunks"]]
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist([100 * v for v in ratios], bins=20, color="#2b7bba", edgecolor="white")
ax.set_xlabel("% of max_chunks used before tau fired (100% = budget exhausted)")
ax.set_ylabel("vla_act invocations")
ax.set_title("Early-termination efficiency of the tau contract")
fig.tight_layout(); fig.savefig(f"{OUT}/chunks_used_hist.png", dpi=150); plt.close(fig)
metrics["chunks_used"] = {
    "mean_ratio": statistics.mean(ratios), "median_ratio": statistics.median(ratios),
    "pct_exhausted": 100 * sum(1 for v in ratios if v >= 1) / len(ratios),
}

# C11) thinking length vs turn outcome
ok_think = [td["think_chars"] for r in records for td in r["turn_data"]
            if td["ok"] and td["think_chars"]]
ko_think = [td["think_chars"] for r in records for td in r["turn_data"]
            if not td["ok"] and td["think_chars"]]
if ok_think and ko_think:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([ok_think, ko_think], labels=["primitive\nsucceeded", "primitive\nfailed"])
    ax.set_ylabel("planner thinking length (chars)")
    ax.set_title("Does the planner think harder before failures?")
    fig.tight_layout(); fig.savefig(f"{OUT}/thinking_vs_outcome.png", dpi=150); plt.close(fig)
    metrics["thinking_chars"] = {
        "success_median": statistics.median(ok_think),
        "failure_median": statistics.median(ko_think),
    }

json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1)
print(json.dumps({k: v for k, v in metrics.items()
                  if k in ("recovery", "chunks_used", "thinking_chars", "steps_paired_successes")}, indent=1))
print("charts ->", OUT)
