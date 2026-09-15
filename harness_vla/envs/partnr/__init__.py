"""PARTNR (Habitat 3) agent side.

The environment (bridge planner, tools, official metrics) lives in the
``partnr-harness`` superproject and defines the ``AgentHarness`` contract in
``partnr_harness.contracts``. This package implements the agent:

* ``primitives`` — PARTNR primitive vocabulary and instance-id symbolization;
* ``prompts`` — system/turn prompts with full-episode history;
* ``memory`` — PARTNR Task Memory and Global Memory (paper-compatible);
* ``adapter`` — ``PartnrHarnessAdapter(AgentHarness)``;
* ``lifecycle`` — offline bootstrap -> frozen deployment helpers;
* ``legacy`` — beta-only historical prompt harnesses (equivalence control).
"""
