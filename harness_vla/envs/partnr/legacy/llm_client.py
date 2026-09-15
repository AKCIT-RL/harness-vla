"""beta-only: historical PARTNR prompt harness, kept for equivalence control; not the paper harness.

OpenAI-compatible chat client for the native PARTNR planners.

Lets the stock LLM planners (e.g. zero-shot ReAct) run against any
OpenAI-compatible endpoint such as a local Ollama server, using only the
standard library. Used for like-for-like baselines against the harness.
"""

import json
import re
import socket
import time
import urllib.error
import urllib.request
from typing import List, Optional, Union

from omegaconf import DictConfig

from habitat_llm.llm.base_llm import BaseLLM, Prompt
from harness_vla.envs.partnr.legacy.ollama_chat import (
    MAX_RETRY_BACKOFF_S,
    base_seed_from_env,
    chat_ollama,
    chat_with_thinking_fallback,
)


class OpenAICompatChat(BaseLLM):
    def __init__(self, conf: DictConfig):
        self.llm_conf = conf
        self.generation_params = conf.generation_params
        self.base_url = str(conf.base_url).rstrip("/")
        self.timeout_s = float(conf.get("request_timeout", 120))
        self.max_retries = int(conf.get("max_retries", 3))
        self.enable_thinking = bool(conf.get("enable_thinking", False))
        self.disable_thinking = bool(conf.get("disable_thinking", False))
        if self.enable_thinking and self.disable_thinking:
            raise ValueError("enable_thinking and disable_thinking are mutually exclusive")
        self.num_ctx = conf.get("num_ctx")
        self.repeat_penalty = float(conf.get("repeat_penalty", 1.0))
        self.repeat_last_n = int(conf.get("repeat_last_n", 64))
        self.thinking_budget = int(conf.get("thinking_budget", 0))
        self._base_seed = base_seed_from_env()
        self._call_index = 0
        self.last_meta = {}
        # Beta-only leniency: without constrained generation some models quote
        # bracket arguments (Place[obj, "on", ...]), which the native entity
        # resolver rejects. Strip quotes inside [...] argument lists.
        self.strip_arg_quotes = bool(conf.get("strip_arg_quotes", False))

    def _next_seed(self) -> Optional[int]:
        index = self._call_index
        self._call_index += 1
        return None if self._base_seed is None else self._base_seed + index

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
        self.last_meta = {}
        if self.enable_thinking or self.disable_thinking:
            common = dict(
                base_url=self.base_url,
                model=str(self.generation_params.model),
                messages=[{"role": "user", "content": prompt}],
                temperature=float(self.generation_params.temperature),
                max_tokens=int(max_length or self.generation_params.max_tokens),
                timeout_s=self.timeout_s,
                num_ctx=int(self.num_ctx) if self.num_ctx is not None else None,
                max_retries=self.max_retries,
                stop=stop_sequences,
                repeat_penalty=self.repeat_penalty,
                repeat_last_n=self.repeat_last_n,
            )
            try:
                if self.enable_thinking:
                    text, self.last_meta = chat_with_thinking_fallback(
                        seed=self._next_seed(),
                        thinking_budget=self.thinking_budget,
                        content_done=None,
                        **common,
                    )
                else:
                    text, self.last_meta = chat_ollama(
                        think=False, stream=False, seed=self._next_seed(), **common
                    )
            except Exception as error:  # transport exhausted its retries
                self.last_meta = {"llm_failure": True, "llm_error": str(error)}
                return ""
            if not text.strip():
                self.last_meta["llm_failure"] = True
                self.last_meta["llm_error"] = "empty_content"
            return self._sanitize(text)
        payload = {
            "model": self.generation_params.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.generation_params.temperature,
            "max_tokens": max_length or self.generation_params.max_tokens,
        }
        seed = self._next_seed()
        if seed is not None:
            payload["seed"] = seed
        if stop_sequences:
            payload["stop"] = stop_sequences
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception = RuntimeError("no attempt made")
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_s
                ) as response:
                    body = json.loads(response.read().decode())
                break
            except (socket.timeout, urllib.error.URLError) as error:
                last_error = error
                if attempt + 1 < self.max_retries:
                    time.sleep(min(float(2**attempt), MAX_RETRY_BACKOFF_S))
        else:
            self.last_meta = {"llm_failure": True, "llm_error": str(last_error)}
            return ""
        text = body["choices"][0]["message"]["content"]
        return self._sanitize(text)

    def _sanitize(self, text: str) -> str:
        if self.strip_arg_quotes:
            text = re.sub(
                r"\[([^\]]*)\]",
                lambda m: "[" + m.group(1).replace('"', "").replace("'", "") + "]",
                text,
            )
        return text
