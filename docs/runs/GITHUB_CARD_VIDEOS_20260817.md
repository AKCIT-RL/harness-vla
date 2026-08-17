# Follow-up: episode videos for every task in the evaluation

Companion to the results card above (trace-driven iteration, frozen pi0.5 + harness). One representative episode per task, taken directly from the battery artifacts — 11 successes and 2 instructive failures. All clips are the harness condition at the final code state (iterations 1–6).

> Upload note: drag each `.mp4` from `docs/runs/assets/card_videos_20260817/` onto the marked slots; GitHub embeds them as playable videos.

## libero_spatial + libero_object (frozen_v1 suite — 95.8% paired success)

**"pick up the black bowl between the plate and the ramekin and place it on the plate"** — one `vla_act` turn; the planner grounds the correct bowl among distractors.
<!-- drag: libero_spatial_task00_success.mp4 -->

**"pick up the black bowl on the cookie box and place it on the plate"**
<!-- drag: libero_spatial_task03_success.mp4 -->

**"pick up the alphabet soup and place it in the basket"** — the suite where the VLA-only baseline scores 0/12 and the harness 11/12; the difference is the sub-goal prompt the planner gives to `vla_act`.
<!-- drag: libero_object_task00_success.mp4 -->

**"pick up the tomato sauce and place it in the basket"**
<!-- drag: libero_object_task05_success.mp4 -->

## libero_goal + libero_10 (probe_v2 — long horizon, 75% vs 25% baseline)

**"open the middle drawer of the cabinet"** — fixture manipulation: empty-hand `vla_act tau=task_success`, enabled by iteration 3. Before it, this task was 0/2 (the planner tried to grasp the cabinet).
<!-- drag: libero_goal_task00_success.mp4 -->

**"put the bowl on the stove"**
<!-- drag: libero_goal_task01_success.mp4 -->

**"put the black bowl in the bottom drawer of the cabinet and close it"** — compound task: place, then close, across replan turns.
<!-- drag: libero_10_task03_success.mp4 -->

**"put both the alphabet soup and the tomato sauce in the basket"** — *failure, and the most informative clip*: the planner's sequencing is correct (grasp soup → place in basket → release → move to the sauce, no re-grasping thanks to the `turn_history` memory), but the frozen VLA repeatedly misses the physical grasp of the sauce can until the horizon runs out. This is the remaining failure mode: physical, not logical.
<!-- drag: libero_10_task00_failure.mp4 -->

## libero_90 (probe_v3 — unseen suite, 60% vs 20% baseline)

**"turn on the stove"** — the only task the baseline also solves.
<!-- drag: libero_90_task20_success.mp4 -->

**"put the white bowl to the right of the plate"** — relational placement.
<!-- drag: libero_90_task37_success.mp4 -->

**"pick up the book and place it in the left compartment of the caddy"** — precise insertion.
<!-- drag: libero_90_task74_success.mp4 -->

**"stack the middle black bowl on the back black bowl"** — *failure*: stacking demands a precision the frozen VLA doesn't deliver within the 400-step horizon.
<!-- drag: libero_90_task17_failure.mp4 -->

**"turn on the stove and put the frying pan on it"** — *failure*: compound task exhausts the horizon; the first sub-goal (stove) succeeds.
<!-- drag: libero_90_task21_failure.mp4 -->

## What the videos show, in one paragraph

Across suites the pattern is consistent with the charts in the previous comment: the VLA controls the arm almost the whole time, and the harness's contribution is invisible when things go well (one well-prompted `vla_act` solves the task) and decisive when they don't — grounding the right object among distractors, operating fixtures with an empty hand, sequencing multi-object placements, and never repeating a completed sub-goal. The failures that remain are precision-of-motion failures of the frozen policy, which no amount of replanning fixes.
