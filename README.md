# Harness VLA beta

Paper-aligned implementation of [Harness VLA (arXiv:2607.08448v2)](https://arxiv.org/abs/2607.08448v2), developed first on EB-Manipulation.

The repository currently provides the architectural core: an LLM planner that
selects one primitive per turn, a fixed analytic/contact primitive library,
closed-loop feedback, physical post-conditions, incremental traces, RGB-D world
grounding infrastructure, memory lifecycle components, and a frozen pi0.5/RLinf
backend. This is an **architectural reproduction in progress**, not yet a full
functional or experimental reproduction of the paper.

Current evidence includes:

- EB-Manipulation Etapa E: `1/3`, validating the planner/primitive loop with a
	scripted contact executor;
- native LIBERO pi0.5/RLinf smoke baseline: `9/10` on LIBERO-Spatial;
- a verified real-VLA grasp in EB-Manipulation, while cross-embodiment transport
	remains diagnostic rather than a paper-comparable result.

Documentation:

- [Beginner's guide to the architecture and pipeline](docs/HARNESS_VLA_BEGINNERS_GUIDE.md)
- [Current functional architecture and evaluation report](docs/HARNESS_VLA_FUNCTIONAL_ARCHITECTURE_REPORT.md)
- [Current architecture, evidence, and paper gap](docs/HARNESS_VLA_BETA_REPORT.md)
- [Paper-aligned implementation and evaluation roadmap](docs/HARNESS_VLA_IMPLEMENTATION_ROADMAP.md)
- [Development and evaluation practices](docs/HARNESS_VLA_BEST_PRACTICES.md)
- [Immutable experiment reports](docs/runs)

Repository layout:

- `harness_vla/` — the agent: planner, primitive library, memories, perception,
	VLA backends, and our LIBERO evaluation code (`harness_vla/eval/`);
- `EmbodiedBench/` — vendored benchmark fork (environments only; the legacy
	EB-Manipulation/Navigation harness evaluators remain wired into its hydra
	pipeline);
- `runners/` — entry-point scripts (e.g. `runners/run_libero_battery.py`);
- `benchmarks/` — frozen evaluation protocols (never edited, only extended);
- `scripts/queues/`, `scripts/analysis/` — evaluation queues and chart/metric
	scripts;
- `tests/` — unit and integration tests;
- `evaluation_runs/` (gitignored) — run artifacts and traces.


The next reproduction target is a native LIBERO Harness using the same frozen
checkpoint in the direct-VLA baseline and the memory-guided Harness. EB-Navigation
is treated separately as a beta-only generalization probe.