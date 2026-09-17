"""OpenAI-compatible chat client for the UPSTREAM PARTNR planners (ReAct baseline).

Drop-in for ``habitat_llm.llm.base_llm.BaseLLM`` (same ``generate`` API,
duck-typed — no ``habitat_llm`` import) so the stock zero-shot ReAct planner
runs against a local Ollama server with the same transport, seeding and
inference metadata as the Harness VLA agents (``llm_client``).

Classification: beta-only (standard-algorithm baseline support).
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Union

from omegaconf import DictConfig

from harness_vla.envs.partnr.llm_client import LLMClient

Prompt = Union[str, List[Any]]


class OpenAICompatChat:
    """``generate(prompt, stop, max_length)`` over :class:`LLMClient`."""

    def __init__(self, conf: DictConfig):
        self.llm_conf = conf
        self.generation_params = conf.generation_params
        self._client = LLMClient(
            base_url=str(conf.base_url),
            model=str(self.generation_params.model),
            temperature=float(self.generation_params.temperature),
            max_tokens=int(self.generation_params.max_tokens),
            timeout_s=float(conf.get("request_timeout", 120)),
            enable_thinking=bool(conf.get("enable_thinking", False)),
            disable_thinking=bool(conf.get("disable_thinking", False)),
            num_ctx=int(conf.num_ctx) if conf.get("num_ctx") is not None else None,
            max_retries=int(conf.get("max_retries", 3)),
            repeat_penalty=float(conf.get("repeat_penalty", 1.0)),
            repeat_last_n=int(conf.get("repeat_last_n", 64)),
            thinking_budget=int(conf.get("thinking_budget", 0)),
        )
        # Without constrained generation some models quote bracket arguments
        # (Place[obj, "on", ...]), which the native entity resolver rejects.
        self.strip_arg_quotes = bool(conf.get("strip_arg_quotes", False))

    @property
    def last_meta(self):
        return self._client.last_meta

    def generate(
        self,
        prompt: Prompt,
        stop: Optional[Union[str, List[str]]] = None,
        max_length: Optional[int] = None,
        generation_args=None,
    ) -> str:
        if not isinstance(prompt, str):
            raise ValueError("OpenAICompatChat only supports text prompts")
        stop_sequences = [stop] if isinstance(stop, str) else list(stop or [])
        text = self._client.post(
            [{"role": "user", "content": prompt}],
            stop=stop_sequences or None,
            max_tokens=int(max_length) if max_length else None,
        )
        return self._sanitize(text)

    def _sanitize(self, text: str) -> str:
        if self.strip_arg_quotes:
            text = re.sub(
                r"\[([^\]]*)\]",
                lambda m: "[" + m.group(1).replace('"', "").replace("'", "") + "]",
                text,
            )
        return text
