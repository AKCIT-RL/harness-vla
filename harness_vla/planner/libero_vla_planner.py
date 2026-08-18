"""Zero-shot Ollama planner for the partial native LIBERO VLA smoke."""

from __future__ import annotations

import json
from typing import Dict, Optional, Tuple
from urllib import request
from urllib.parse import urlparse

from harness_vla.planner.harness_planner import extract_json_object


SYSTEM_PROMPT = """You are the zero-shot planner for a partial LIBERO VLA smoke.
This smoke offers only one primitive, vla_act. It is not a complete Harness primitive library.
It does not offer analytic primitives, Task Memory, or Global Memory.
For a complete-task smoke, emit exactly:
{"action":"vla_act","prompt":"<task prompt for the VLA>","max_chunks":<integer>,"tau":"task_success"}
For a local grasp phase, emit exactly:
{"action":"vla_act","prompt":"<local grasp prompt>","max_chunks":<integer>,"tau":"lift_and_grasp","target":"<grounded instance name>"}
The max_chunks value must not exceed the cap supplied by the user. Do not emit
numeric thresholds; they are benchmark configuration, not planner decisions."""


def parse_libero_vla_invocation(
    raw_output: str, max_chunks_cap: int
) -> Optional[Dict[str, object]]:
    """Extract and strictly validate one planner-facing ``vla_act`` call."""
    if (
        isinstance(max_chunks_cap, bool)
        or not isinstance(max_chunks_cap, int)
        or max_chunks_cap < 1
    ):
        raise ValueError("max_chunks_cap must be an integer greater than zero")
    parsed = extract_json_object(raw_output)
    if parsed is None or parsed.get("action") != "vla_act":
        return None
    prompt = parsed.get("prompt")
    max_chunks = parsed.get("max_chunks")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    if (
        isinstance(max_chunks, bool)
        or not isinstance(max_chunks, int)
        or not 1 <= max_chunks <= max_chunks_cap
    ):
        return None
    tau = parsed.get("tau")
    if tau not in ("task_success", "lift_and_grasp"):
        return None
    invocation = {
        "action": "vla_act",
        "prompt": prompt.strip(),
        "max_chunks": max_chunks,
        "tau": tau,
    }
    if tau == "lift_and_grasp":
        target = parsed.get("target")
        if not isinstance(target, str) or not target.strip():
            return None
        invocation["target"] = target.strip()
    return invocation


class LiberoVLAPlanner:
    """Small Ollama ``/api/chat`` client with no memory dependencies."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434/v1",
        think: bool = False,
        request_timeout: float = 600.0,
        required_tau: str = "task_success",
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.model_name = model_name
        self.base_url = base_url
        self.think = bool(think)
        self.request_timeout = request_timeout
        if required_tau not in ("task_success", "lift_and_grasp"):
            raise ValueError("required_tau is not supported")
        self.required_tau = required_tau
        self.last_thinking: Optional[str] = None

    def _endpoint(self) -> str:
        base_url = self.base_url.rstrip("/")
        parsed = urlparse(base_url)
        if parsed.path not in ("", "/v1"):
            raise ValueError("Ollama base URL must end in /v1 or have no path")
        if parsed.path == "/v1":
            base_url = base_url[:-3]
        return base_url + "/api/chat"

    def _chat(self, user_prompt: str) -> str:
        options = {"temperature": 0}
        if self.think:
            options.update({"num_predict": 8192, "num_ctx": 16384})
        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "think": self.think,
                "options": options,
            }
        ).encode("utf-8")
        req = request.Request(
            self._endpoint(),
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=self.request_timeout) as response:
            result = json.load(response)
        message = result.get("message", {})
        self.last_thinking = message.get("thinking") or None
        return message.get("content", "")

    def act(
        self, instruction: str, max_chunks_cap: int, available_targets=None
    ) -> Tuple[Optional[Dict[str, object]], str]:
        """Request the single invocation from task text and initial feedback."""
        targets = [] if available_targets is None else list(available_targets)
        user_prompt = "\n".join(
            [
                "Official task instruction: %s" % instruction,
                "Available max_chunks cap: %d" % max_chunks_cap,
                "Grounded target names: %s" % json.dumps(targets),
                "Required termination predicate for this phase: %s"
                % self.required_tau,
                "Initial feedback: no policy action has been executed.",
                "Return exactly one JSON invocation.",
            ]
        )
        raw_output = self._chat(user_prompt)
        invocation = parse_libero_vla_invocation(raw_output, max_chunks_cap)
        if invocation is not None and invocation["tau"] != self.required_tau:
            invocation = None
        if (
            invocation is not None
            and invocation["tau"] == "lift_and_grasp"
            and invocation["target"] not in targets
        ):
            invocation = None
        return invocation, raw_output


__all__ = ["LiberoVLAPlanner", "SYSTEM_PROMPT", "parse_libero_vla_invocation"]