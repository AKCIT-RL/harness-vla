"""beta-only: historical PARTNR prompt harness, kept for equivalence control; not the paper harness.

Native Ollama chat transport with beta-only inference instrumentation.
"""

import json
import os
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

_SEED_ENV = "PARTNR_LLM_SEED"

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

    Beta-only: the stream is never aborted by closing the connection. On
    Ollama 0.33.x a client disconnect mid-generation leaves the server in a
    state where subsequent requests hang and fail with 500 "client closing the
    connection" without reaching llama-server; ~10 such aborts wedged a queue
    job for hours. Instead, when ``think=True`` and ``thinking_budget > 0`` the
    budget is enforced server-side via ``options.num_predict =
    min(max_tokens, thinking_budget + ANSWER_RESERVE_TOKENS)`` and the stream
    is always read until ``done: true`` (or the socket timeout).

    ``aborted_reason`` is kept in the metadata as a label derived from the
    finished result: ``"thinking_budget"`` when the generation hit the length
    cap with no usable content, ``"thinking_loop"`` when the thinking text is
    highly repetitive. ``content_done`` is accepted for API compatibility and
    ignored (the model is allowed to finish on its own).
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


def _consume_stream(response) -> Tuple[str, str, Dict[str, Any]]:
    """Accumulate NDJSON deltas until ``done: true`` (or EOF/timeout).

    Never breaks out early: the connection must stay open until the server
    finishes, see ``chat_ollama``.
    """
    thinking_parts: List[str] = []
    content_parts: List[str] = []
    last_chunk: Dict[str, Any] = {}
    for raw_line in response:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.strip()
        if not line:
            continue
        chunk = json.loads(line)
        last_chunk = chunk
        message = chunk.get("message") or {}
        if message.get("thinking"):
            thinking_parts.append(str(message["thinking"]))
        if message.get("content"):
            content_parts.append(str(message["content"]))
        if chunk.get("done"):
            break
    return "".join(thinking_parts), "".join(content_parts), last_chunk


def _nanoseconds_to_seconds(value: Any) -> Optional[float]:
    return float(value) / 1_000_000_000 if value is not None else None
