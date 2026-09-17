"""PARTNR LLM-free smoke agent for the inverted-control env API.

* ``random_loop`` — drives ``reset``/``step`` of a ``PartnrEnvClient``
  (duck-typed from ``partnr_harness.external``) with random valid tools.

Nothing here imports ``habitat_llm``.
"""
