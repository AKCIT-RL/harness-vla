"""beta-only: historical PARTNR prompt harness, kept for equivalence control; not the paper harness.

Minimal LLM-backed harness: one chat call per decision, with a prompt prefix.

Validates the end-to-end pipeline against any OpenAI-compatible endpoint
(vLLM/ollama/...) using only the standard library. The world description is
truncated to a character budget sized for the served model's context window.
"""

import json
import re
import urllib.request
from typing import List, Optional

from partnr_harness.contracts import (
    ActionResult,
    AgentHarness,
    Decision,
    DoneDecision,
    EpisodeContext,
    StepContext,
    ToolCall,
)
from harness_vla.envs.partnr.legacy.ollama_chat import (
    base_seed_from_env,
    chat_ollama,
    chat_with_thinking_fallback,
)

_TOOL_CALL = re.compile(r"([A-Za-z_]+)\s*\[(.*?)\]", re.DOTALL)
_AGENT_SEED_STRIDE = 100_000

_INSTRUCTIONS = (
    "Reply with EXACTLY one action in the format ToolName[arguments] and "
    "nothing else. Use Done[] when the whole task is finished."
)


class LLMPrefixHarness(AgentHarness):
    def __init__(
        self,
        prefix: str = "You are a robot collaborating on a household task.",
        base_url: str = "http://192.168.1.227:8000/v1",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 64,
        timeout_s: float = 120.0,
        world_char_budget: int = 12000,
        max_consecutive_parse_failures: int = 3,
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
        self.prefix = prefix
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.world_char_budget = world_char_budget
        # Kept for config compatibility; parse failures no longer escalate to Done.
        self.max_consecutive_parse_failures = max_consecutive_parse_failures
        self.enable_thinking = enable_thinking
        self.disable_thinking = disable_thinking
        self.num_ctx = num_ctx
        self.max_retries = max_retries
        self.repeat_penalty = repeat_penalty
        self.repeat_last_n = repeat_last_n
        self.thinking_budget = thinking_budget
        self.last_meta = {}
        self._base_seed = base_seed_from_env()
        self._call_index = 0
        self._agent_uid: Optional[int] = None

    def on_episode_start(self, context: EpisodeContext) -> None:
        self._agent_uid = context.agent_uid
        self._call_index = 0

    # --------------------------------------------------------------- llm call

    def _next_seed(self) -> Optional[int]:
        index = self._call_index
        self._call_index += 1
        if self._base_seed is None:
            return None
        seed = self._base_seed + index
        if self._agent_uid is not None:
            seed += _AGENT_SEED_STRIDE * self._agent_uid
        return seed

    def _content_done(self, content: str) -> bool:
        """Streaming early-stop: True once ``content`` holds a full answer."""
        return _TOOL_CALL.search(content) is not None

    def _post(self, messages: List[dict]) -> str:
        self.last_meta = {}
        if self.enable_thinking or self.disable_thinking:
            common = dict(
                base_url=self.base_url,
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout_s=self.timeout_s,
                num_ctx=self.num_ctx,
                max_retries=self.max_retries,
                repeat_penalty=self.repeat_penalty,
                repeat_last_n=self.repeat_last_n,
            )
            try:
                if self.enable_thinking:
                    content, self.last_meta = chat_with_thinking_fallback(
                        seed=self._next_seed(),
                        thinking_budget=self.thinking_budget,
                        content_done=self._content_done,
                        **common,
                    )
                else:
                    content, self.last_meta = chat_ollama(
                        think=False, stream=False, seed=self._next_seed(), **common
                    )
            except Exception as error:  # transport exhausted its retries
                self.last_meta = {"llm_failure": True, "llm_error": str(error)}
                return ""
            if not content.strip():
                self.last_meta["llm_failure"] = True
                self.last_meta["llm_error"] = "empty_content"
            return content
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode())
            return body["choices"][0]["message"]["content"]
        except Exception as error:
            self.last_meta = {"llm_failure": True, "llm_error": str(error)}
            return ""

    def _llm_failure(self, raw: str, meta: dict) -> ToolCall:
        """Decision returned when the LLM was unreachable or answered nothing."""
        message = f"LLM unavailable: {meta.get('llm_error', 'unknown')}"
        return ToolCall(
            "InvalidResponse",
            message,
            rationale=message[:200],
            raw_output=raw,
            error_kind="llm_failure",
            meta=meta,
        )

    # ----------------------------------------------------------------- prompt

    def _build_user_message(self, context: StepContext) -> str:
        world = context.world_description
        if len(world) > self.world_char_budget:
            world = world[: self.world_char_budget] + "\n[...truncated...]"

        tools = "\n".join(
            f"- {tool.name}: {tool.description.strip().splitlines()[0][:120]}"
            if tool.description
            else f"- {tool.name}"
            for tool in context.tools
        )
        parts = [
            f"Task: {context.instruction}",
            f"Tools:\n{tools}",
            f"House state:\n{world}",
        ]
        if context.last_result is not None:
            parts.append(
                "Previous action "
                f"{context.last_result.tool}[{context.last_result.args}] -> "
                f"{context.last_result.response}"
            )
        if context.partner_recent_actions:
            parts.append(
                "Partner recent actions:\n" + "\n".join(context.partner_recent_actions)
            )
        if context.signals:
            parts.append(
                "Warnings:\n"
                + "\n".join(f"- {signal.message}" for signal in context.signals)
            )
        parts.append(_INSTRUCTIONS)
        return "\n\n".join(parts)

    # ----------------------------------------------------------------- decide

    def decide(self, context: StepContext) -> Decision:
        messages = [
            {"role": "system", "content": self.prefix},
            {"role": "user", "content": self._build_user_message(context)},
        ]
        raw = self._post(messages)
        meta = dict(self.last_meta)
        if meta.get("llm_failure"):
            return self._llm_failure(raw, meta)
        rationale = raw.strip()[:200]

        match = _TOOL_CALL.search(raw)
        if match is None:
            # Invalid tool name: the bridge returns an error result, giving the
            # LLM corrective feedback on the next decision.
            return ToolCall(
                "InvalidResponse",
                rationale,
                rationale=rationale,
                raw_output=raw,
                error_kind="parse_error",
                meta=meta,
            )

        name, args = match.group(1), match.group(2).strip()
        if name.lower() == "done":
            return DoneDecision(rationale=rationale, meta=meta)
        return ToolCall(name, args, rationale=rationale, meta=meta)

    def on_action_result(self, result: ActionResult) -> None:
        pass
