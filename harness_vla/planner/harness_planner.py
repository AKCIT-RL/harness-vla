"""Harness VLA beta planner.

A lightweight, OpenAI-compatible planner that drives the closed-loop harness. It
is deliberately standalone (it does not import the repo's ``RemoteModel``, which
pulls in heavy optional dependencies such as ``lmdeploy``) so the beta can run
against a local Ollama server via the OpenAI SDK by only changing ``base_url``.

Each :meth:`HarnessPlanner.act` call asks the model for exactly one JSON
primitive invocation, parses it robustly, and returns it for compilation and
execution by the evaluator.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple
from urllib import request
from urllib.parse import urlparse

from harness_vla.memory.global_memory import GlobalMemory
from harness_vla.primitives.primitives import normalize_invocation
from harness_vla.planner.prompts import (
    build_system_prompt,
    build_turn_prompt,
)


def extract_json_object(text: str) -> Optional[dict]:
    """Best-effort extraction of a single JSON object from model output.

    Handles markdown fences, leading/trailing prose, and simple trailing commas.
    Returns ``None`` if no valid JSON object can be recovered.
    """
    if not text:
        return None
    cleaned = text.strip()
    # Strip markdown code fences.
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    # Try a direct parse first.
    candidates = [cleaned]
    # Then the first balanced {...} block.
    brace = _first_balanced_object(cleaned)
    if brace is not None:
        candidates.append(brace)

    for cand in candidates:
        for attempt in (cand, _strip_trailing_commas(cand)):
            try:
                obj = json.loads(attempt)
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def _first_balanced_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


class HarnessPlanner:
    """LLM planner emitting one JSON primitive invocation per turn.

    Parameters
    ----------
    model_name:
        Model identifier passed to the OpenAI-compatible endpoint (e.g. an
        Ollama tag like ``"qwen2.5:0.5b-instruct"``).
    base_url:
        OpenAI-compatible base URL. Defaults to the ``OPENAI_BASE_URL`` env var
        or Ollama's local endpoint.
    api_key:
        API key. For Ollama any non-empty string works.
    global_memory:
        Seeded :class:`GlobalMemory`; if ``None`` the fixed seed is used.
    temperature / max_tokens:
        Sampling controls forwarded to the chat completion call.
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        global_memory: Optional[GlobalMemory] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        num_ctx: Optional[int] = None,
        disable_thinking: bool = False,
        enable_thinking: bool = False,
        request_timeout: float = 600.0,
        client=None,
    ) -> None:
        if disable_thinking and enable_thinking:
            raise ValueError("disable_thinking and enable_thinking are mutually exclusive")
        self.model_name = model_name
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "http://localhost:11434/v1"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "ollama"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.disable_thinking = disable_thinking
        self.enable_thinking = enable_thinking
        self.last_thinking: Optional[str] = None
        self.request_timeout = request_timeout
        self.global_memory = global_memory or GlobalMemory.seeded()
        self.system_prompt = build_system_prompt(self.global_memory.render())

        if client is not None:
            self.client = client
        else:
            from openai import OpenAI

            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        # Episode-scoped counters (mirrors ManipPlanner's interface).
        self.planner_steps = 0
        self.output_json_error = 0

    def set_global_memory(self, global_memory: GlobalMemory) -> None:
        """Replace memory and rebuild the system prompt from the same snapshot."""
        self.global_memory = global_memory
        self.system_prompt = build_system_prompt(global_memory.render())

    def reset(self) -> None:
        self.planner_steps = 0
        self.output_json_error = 0

    def _chat(self, turn_prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": turn_prompt},
        ]
        if self.disable_thinking or self.enable_thinking:
            return self._chat_ollama(messages, think=self.enable_thinking)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _chat_ollama(self, messages: List[Dict[str, str]], think: bool = False) -> str:
        base_url = self.base_url.rstrip("/")
        parsed_url = urlparse(base_url)
        if parsed_url.path not in ("", "/v1"):
            raise ValueError(
                "thinking control requires an Ollama base URL ending in /v1"
            )
        if parsed_url.path == "/v1":
            base_url = base_url[:-3]
        payload_options = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        }
        if self.num_ctx is not None:
            payload_options["num_ctx"] = self.num_ctx
        payload = json.dumps({
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": think,
            "options": payload_options,
        }).encode("utf-8")
        req = request.Request(
            f"{base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=self.request_timeout) as response:
            result = json.load(response)
        message = result.get("message", {})
        self.last_thinking = message.get("thinking") or None
        return message.get("content", "")

    def act(
        self,
        user_instruction: str,
        object_coords: Dict[str, Sequence[float]],
        pose_action: Sequence[int],
        history: List[Dict],
        object_roles: Optional[Dict[str, Sequence[str]]] = None,
        object_labels: Optional[Dict[str, str]] = None,
        held_object_id: Optional[str] = None,
        attachment_evidence_available: bool = False,
        resolved_task_memory: Optional[Sequence[Dict]] = None,
    ) -> Tuple[Optional[dict], str]:
        """Request one primitive invocation from the model.

        Returns ``(invocation, raw_text)`` where ``invocation`` is a dict with an
        ``action`` field, or ``None`` if parsing failed (counted in
        ``output_json_error``).
        """
        self.planner_steps += 1
        turn_prompt = build_turn_prompt(
            user_instruction,
            object_coords,
            pose_action,
            history,
            object_roles=object_roles,
            object_labels=object_labels,
            held_object_id=held_object_id,
            attachment_evidence_available=attachment_evidence_available,
            resolved_task_memory=resolved_task_memory,
        )
        raw_text = self._chat(turn_prompt)
        parsed = extract_json_object(raw_text)
        if parsed is None:
            self.output_json_error += 1
            return None, raw_text
        # Coerce the many shapes small models emit into a canonical invocation
        # {"action": <name>, ...args}.
        invocation = normalize_invocation(parsed)
        if invocation is None:
            self.output_json_error += 1
            return None, raw_text
        return invocation, raw_text
