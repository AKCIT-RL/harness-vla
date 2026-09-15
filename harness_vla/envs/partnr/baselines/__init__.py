"""PARTNR agent baselines (beta-only) implementing ``partnr_harness.contracts.AgentHarness``.

* ``random_harness`` / ``recovery_aware`` — LLM-free smoke harnesses;
* ``random_loop`` — LLM-free client for the inverted-control env API;
* ``react`` — ReAct reimplementation (full transcript, Thought/Action format).

Nothing here imports ``habitat_llm``; only the contract types and constants
from ``partnr_harness.contracts`` (``random_loop`` duck-types
``partnr_harness.external.PartnrEnvClient``).
"""
