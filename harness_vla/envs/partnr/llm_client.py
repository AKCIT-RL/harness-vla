"""LLM transport for the PARTNR Harness VLA agents (stdlib only).

Two backends behind one :class:`LLMClient`:

* **Ollama native** (``/api/chat``) when either thinking flag is set: exposes
  ``num_ctx``, ``think``, seeds and per-call inference metadata (tokens,
  durations, thinking fallback);
* **OpenAI-compatible** (``/v1/chat/completions``) otherwise.

Seeding: the queue runner exports ``PARTNR_LLM_SEED`` per episode; every call
gets ``base + call_index (+ 100000 * agent_uid)`` so repeated runs are
reproducible and the two decentralized agents never share a seed.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

__all__ = [
    "ANSWER_RESERVE_TOKENS",
    "LLMClient",
    "base_seed_from_env",
    "chat_ollama",
    "chat_with_thinking_fallback",
    "ollama_api_base",
    "thinking_repetition_ratio",
]

_SEED_ENV = "PARTNR_LLM_SEED"
_AGENT_SEED_STRIDE = 100_000

# Tokens kept for the answer after the thinking budget when capping num_predict.
ANSWER_RESERVE_TOKENS = 512
# Retry backoff cap (seconds): a hung server must not stretch retries for minutes.
MAX_RETRY_BACKOFF_S = 10.0


def base_seed_from_env() -> Optional[int]:
    """Per-episode base seed set by the queue runner (``PARTNR_LLM_SEED``)."""
    raw = os.environ.get(_SEED_ENV, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def ollama_api_base(base_url: str) -> str:
    """Return an Ollama server root from a root or ``/v1`` base URL."""
    normalized = base_url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.path not in ("", "/v1"):
        raise ValueError("native Ollama chat requires a base URL ending in /v1")
    return normalized[:-3] if parsed.path == "/v1" else normalized


def thinking_repetition_ratio(thinking: str) -> float:
    """Share of repeated lines in ``thinking`` (0.0 when fewer than 5 lines)."""
    lines = [line.strip() for line in thinking.splitlines() if line.strip()]
    if len(lines) < 5:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


# ------------------------------------------------------------ ollama native

def chat_ollama(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    think: bool,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    num_ctx: Optional[int] = None,
    max_retries: int = 3,
    stop: Optional[List[str]] = None,
    seed: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
    repeat_last_n: Optional[int] = None,
    stream: bool = False,
    thinking_budget: int = 0,
    content_done: Optional[Callable[[str], bool]] = None,
    loop_repetition_threshold: float = 0.3,
) -> Tuple[str, Dict[str, Any]]:
    """Call ``/api/chat`` and return action content plus separate metadata.

    The stream is never aborted by closing the connection (Ollama 0.33.x
    wedges after mid-generation disconnects). With ``think=True`` and
    ``thinking_budget > 0`` the budget is enforced server-side via
    ``options.num_predict = min(max_tokens, thinking_budget +
    ANSWER_RESERVE_TOKENS)`` and the stream is read until ``done: true``.
    ``aborted_reason`` is a post-hoc label (``thinking_budget`` when the
    length cap hit with no usable content, ``thinking_loop`` when the thinking
    text is highly repetitive). ``content_done`` is accepted and ignored.
    """
    num_predict = max_tokens
    if think and thinking_budget > 0:
        num_predict = min(max_tokens, thinking_budget + ANSWER_RESERVE_TOKENS)
    options: Dict[str, Any] = {
        "temperature": temperature,
        "num_predict": num_predict,
    }
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    if stop:
        options["stop"] = stop
    if seed is not None:
        options["seed"] = seed
    if repeat_penalty is not None:
        options["repeat_penalty"] = repeat_penalty
    if repeat_last_n is not None:
        options["repeat_last_n"] = repeat_last_n
    payload = {
        "model": model,
        "messages": messages,
        "stream": bool(stream),
        "think": think,
        "options": options,
    }
    request = urllib.request.Request(
        f"{ollama_api_base(base_url)}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    last_error: Exception = RuntimeError("no attempt made")
    attempt_ok = 0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                if stream:
                    thinking, content, body = _consume_stream(response)
                else:
                    body = json.loads(response.read().decode("utf-8"))
                    message = body.get("message", {})
                    thinking = str(message.get("thinking") or "")
                    content = str(message.get("content") or "")
            attempt_ok = attempt + 1
            break
        except (
            json.JSONDecodeError,
            socket.timeout,
            urllib.error.URLError,
            OSError,
        ) as error:
            last_error = error
            if attempt + 1 < max_retries:
                time.sleep(min(float(2**attempt), MAX_RETRY_BACKOFF_S))
    else:
        raise last_error

    request_duration_s = time.monotonic() - started
    final_received = bool(body.get("done")) if stream else True
    thinking_repetition = thinking_repetition_ratio(thinking)
    aborted_reason = _derive_aborted_reason(
        think=think,
        thinking_budget=thinking_budget,
        done_reason=body.get("done_reason"),
        content=content,
        thinking=thinking,
        thinking_repetition=thinking_repetition,
        loop_repetition_threshold=loop_repetition_threshold,
    )
    thinking_len = len(thinking)
    content_len = len(content)
    prompt_tokens = int(body.get("prompt_eval_count") or 0)
    if final_received:
        completion_tokens = int(body.get("eval_count") or 0)
    else:
        completion_tokens = (thinking_len + content_len) // 4
    if final_received and thinking_len + content_len > 0:
        thinking_tokens_est = int(
            round(completion_tokens * thinking_len / (thinking_len + content_len))
        )
    else:
        thinking_tokens_est = thinking_len // 4
    content_tokens_est = max(0, completion_tokens - thinking_tokens_est)
    done_reason = body.get("done_reason") or aborted_reason
    eval_duration_s = _nanoseconds_to_seconds(body.get("eval_duration"))
    prompt_eval_duration_s = _nanoseconds_to_seconds(body.get("prompt_eval_duration"))
    meta = {
        "backend": "ollama_native",
        "model": body.get("model") or model,
        "thinking_enabled": think,
        "thinking": thinking or None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "request_duration_s": request_duration_s,
        "total_duration_s": _nanoseconds_to_seconds(body.get("total_duration")),
        "load_duration_s": _nanoseconds_to_seconds(body.get("load_duration")),
        "prompt_eval_duration_s": prompt_eval_duration_s,
        "eval_duration_s": eval_duration_s,
        "done_reason": done_reason,
        "thinking_len": thinking_len,
        "content_len": content_len,
        "thinking_tokens_est": thinking_tokens_est,
        "content_tokens_est": content_tokens_est,
        "truncated": done_reason == "length",
        "ctx_overflow": (
            num_ctx is not None and prompt_tokens + completion_tokens >= num_ctx
        ),
        "tok_per_s": (
            completion_tokens / eval_duration_s if eval_duration_s else None
        ),
        "prefill_s": prompt_eval_duration_s,
        "thinking_repetition": thinking_repetition,
        "aborted_reason": aborted_reason,
        "stream": bool(stream),
        "sampling": {
            "temperature": temperature,
            "seed": seed,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
            "num_predict": num_predict,
            "max_tokens": max_tokens,
            "num_ctx": num_ctx,
            "think": think,
            "thinking_budget": thinking_budget,
        },
        "thinking_fallback": False,
        "fallback_reason": None,
        "attempt": attempt_ok,
    }
    return content, meta


def fallback_reason(content: str, meta: Dict[str, Any]) -> Optional[str]:
    """Why a thinking-ON answer needs a thinking-OFF retry (None if it doesn't)."""
    aborted = meta.get("aborted_reason")
    if aborted in ("thinking_budget", "thinking_loop"):
        return aborted
    if not content.strip():
        return "truncated" if meta.get("truncated") else "empty_content"
    return None


def chat_with_thinking_fallback(
    *, seed: Optional[int] = None, **kwargs: Any
) -> Tuple[str, Dict[str, Any]]:
    """Thinking-ON streamed call, retried once with thinking OFF when the
    answer is empty/aborted. Token counts and durations are summed."""
    kwargs.pop("think", None)
    kwargs.pop("stream", None)
    content, meta = chat_ollama(think=True, stream=True, seed=seed, **kwargs)
    reason = fallback_reason(content, meta)
    if reason is None:
        meta["attempts_total"] = 1
        return content, meta
    kwargs.pop("content_done", None)
    kwargs.pop("thinking_budget", None)
    content, second = chat_ollama(
        think=False,
        stream=False,
        seed=None if seed is None else seed + 1,
        **kwargs,
    )
    second["thinking_fallback"] = True
    second["fallback_reason"] = reason
    second["thinking"] = meta.get("thinking")
    second["first_attempt"] = {
        "done_reason": meta.get("done_reason"),
        "thinking_tokens_est": meta.get("thinking_tokens_est"),
        "thinking_repetition": meta.get("thinking_repetition"),
        "aborted_reason": meta.get("aborted_reason"),
        "request_duration_s": meta.get("request_duration_s"),
    }
    for key in ("request_duration_s", "prompt_tokens", "completion_tokens", "total_tokens"):
        second[key] = (second.get(key) or 0) + (meta.get(key) or 0)
    second["attempts_total"] = 2
    return content, second


def _derive_aborted_reason(
    *,
    think: bool,
    thinking_budget: int,
    done_reason: Any,
    content: str,
    thinking: str,
    thinking_repetition: float,
    loop_repetition_threshold: float,
) -> Optional[str]:
    """Post-hoc label (never an actual abort) for a finished thinking-ON call."""
    if not think:
        return None
    if thinking_budget > 0 and done_reason == "length" and not _content_usable(content):
        return "thinking_budget"
    if (
        len(thinking.splitlines()) >= 20
        and thinking_repetition > loop_repetition_threshold
    ):
        return "thinking_loop"
    return None


def _content_usable(content: str) -> bool:
    """Non-empty and, when it looks like JSON, parseable."""
    stripped = content.strip()
    if not stripped:
        return False
    if stripped[0] in "{[":
        try:
            json.loads(stripped)
        except json.JSONDecodeError:
            return False
    return True


def _consume_stream(response: Any) -> Tuple[str, str, Dict[str, Any]]:
    """Read an NDJSON ``/api/chat`` stream to the end; returns (thinking, content, last_body)."""
    thinking_parts: List[str] = []
    content_parts: List[str] = []
    body: Dict[str, Any] = {}
    for raw_line in response:
        line = raw_line.decode("utf-8").strip() if isinstance(raw_line, bytes) else str(raw_line).strip()
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = chunk.get("message") or {}
        if message.get("thinking"):
            thinking_parts.append(str(message["thinking"]))
        if message.get("content"):
            content_parts.append(str(message["content"]))
        body = chunk
        if chunk.get("done"):
            break
    return "".join(thinking_parts), "".join(content_parts), body


def _nanoseconds_to_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value) / 1e9
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ client

class LLMClient:
    """One chat completion per call; failures are reported, never raised.

    ``post(messages)`` returns the content string and fills ``last_meta``;
    on transport exhaustion / empty content ``last_meta["llm_failure"]`` is
    True so callers turn it into an ``error_kind="llm_failure"`` decision.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11435/v1",
        model: str = "gemma4:12b",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout_s: float = 240.0,
        enable_thinking: bool = False,
        disable_thinking: bool = False,
        num_ctx: Optional[int] = None,
        max_retries: int = 3,
        repeat_penalty: float = 1.0,
        repeat_last_n: int = 64,
        thinking_budget: int = 0,
    ) -> None:
        if enable_thinking and disable_thinking:
            raise ValueError("enable_thinking and disable_thinking are mutually exclusive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_s = float(timeout_s)
        self.enable_thinking = bool(enable_thinking)
        self.disable_thinking = bool(disable_thinking)
        self.num_ctx = num_ctx
        self.max_retries = int(max_retries)
        self.repeat_penalty = repeat_penalty
        self.repeat_last_n = repeat_last_n
        self.thinking_budget = int(thinking_budget)
        self.last_meta: Dict[str, Any] = {}
        self._base_seed = base_seed_from_env()
        self._call_index = 0
        self._agent_uid: Optional[int] = None

    @property
    def native(self) -> bool:
        return self.enable_thinking or self.disable_thinking

    def start_episode(self, agent_uid: Optional[int] = None) -> None:
        """Re-read the per-episode seed and restart the call counter."""
        self._base_seed = base_seed_from_env()
        self._call_index = 0
        self._agent_uid = agent_uid

    def next_seed(self) -> Optional[int]:
        index = self._call_index
        self._call_index += 1
        if self._base_seed is None:
            return None
        seed = self._base_seed + index
        if self._agent_uid is not None:
            seed += _AGENT_SEED_STRIDE * self._agent_uid
        return seed

    def post(
        self,
        messages: List[Dict[str, str]],
        *,
        stop: Optional[List[str]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        self.last_meta = {}
        max_tokens = int(max_tokens) if max_tokens else self.max_tokens
        if self.native:
            return self._post_native(messages, stop, max_tokens)
        return self._post_openai(messages, stop, max_tokens)

    def _post_native(
        self, messages: List[Dict[str, str]], stop: Optional[List[str]], max_tokens: int
    ) -> str:
        common = dict(
            base_url=self.base_url,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
            timeout_s=self.timeout_s,
            num_ctx=self.num_ctx,
            max_retries=self.max_retries,
            stop=stop,
            repeat_penalty=self.repeat_penalty,
            repeat_last_n=self.repeat_last_n,
        )
        try:
            if self.enable_thinking:
                content, self.last_meta = chat_with_thinking_fallback(
                    seed=self.next_seed(), thinking_budget=self.thinking_budget, **common
                )
            else:
                content, self.last_meta = chat_ollama(
                    think=False, stream=False, seed=self.next_seed(), **common
                )
        except Exception as error:  # transport exhausted its retries
            self.last_meta = {"llm_failure": True, "llm_error": str(error)}
            return ""
        if not content.strip():
            self.last_meta["llm_failure"] = True
            self.last_meta["llm_error"] = "empty_content"
        return content

    def _post_openai(
        self, messages: List[Dict[str, str]], stop: Optional[List[str]], max_tokens: int
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if stop:
            payload["stop"] = list(stop)
        seed = self.next_seed()
        if seed is not None:
            payload["seed"] = seed
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        started = time.monotonic()
        last_error: Exception = RuntimeError("no attempt made")
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    body = json.loads(response.read().decode())
                break
            except (json.JSONDecodeError, socket.timeout, urllib.error.URLError, OSError) as error:
                last_error = error
                if attempt + 1 < self.max_retries:
                    time.sleep(min(float(2**attempt), MAX_RETRY_BACKOFF_S))
        else:
            self.last_meta = {"llm_failure": True, "llm_error": str(last_error)}
            return ""
        try:
            content = str(body["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as error:
            self.last_meta = {"llm_failure": True, "llm_error": f"malformed response: {error!r}"}
            return ""
        usage = body.get("usage") or {}
        self.last_meta = {
            "backend": "openai_compat",
            "model": body.get("model") or self.model,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "request_duration_s": time.monotonic() - started,
            "sampling": {"temperature": self.temperature, "seed": seed, "max_tokens": max_tokens},
        }
        if not content.strip():
            self.last_meta["llm_failure"] = True
            self.last_meta["llm_error"] = "empty_content"
        return content
