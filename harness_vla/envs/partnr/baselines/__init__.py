"""PARTNR agent baselines (beta-only) implementing ``partnr_harness.contracts.AgentHarness``.

* ``random_harness`` / ``recovery_aware`` — LLM-free smoke harnesses;
* ``random_loop`` — LLM-free client for the inverted-control env API;
* ``react`` — ReAct reimplementation (full transcript, Thought/Action format).

Nothing here imports ``habitat_llm``; only the contract types from
``partnr_harness.contracts`` / ``partnr_harness.signals``.
"""
