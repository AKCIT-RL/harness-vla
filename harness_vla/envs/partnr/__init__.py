"""PARTNR (Habitat 3) agent side.

The environment (bridge/centralized planners, tools, official metrics) lives
in the ``partnr-harness`` superproject; the ``AgentHarness`` contract it
drives lives in ``akc_harness.contracts``. This package implements the agent:

* ``primitives`` — PARTNR primitive vocabulary, JSON/bracket codec and
  instance-id symbolization;
* ``prompts`` — system/turn prompts (team transcript, memory, JSON output);
* ``memory`` — PARTNR Task Memory and Global Memory (paper-compatible);
* ``llm_client`` — stdlib LLM transport (Ollama native / OpenAI-compatible);
* ``agent`` — ``HarnessVlaCentralized(AgentHarness)``: one brain, all agents;
* ``decentralized`` — ``HarnessVlaDecentralized``: one brain per agent,
  coordination through published transcripts;
* ``lifecycle`` — offline bootstrap -> frozen deployment helpers;
* ``react_llm`` — OpenAI-compatible client for the upstream ReAct baseline;
* ``baselines`` — LLM-free smoke client for the inverted-control env API.
"""
