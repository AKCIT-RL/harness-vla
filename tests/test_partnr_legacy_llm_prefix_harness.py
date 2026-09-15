"""beta-only: ported tests for the legacy LLM prefix harness (no LLM server)."""

import json

import pytest

pytest.importorskip("partnr_harness.contracts")

from partnr_harness.contracts import DoneDecision, StepContext, ToolCall, ToolSpec

from harness_vla.envs.partnr.legacy.llm_prefix_harness import LLMPrefixHarness


def _context(world="bedroom_1: table_1", tools=("Navigate", "Explore")):
    return StepContext(
        instruction="move the cup",
        agent_uid=0,
        decision_index=0,
        max_decisions=10,
        world_description=world,
        tools=[ToolSpec(name, f"{name} does things") for name in tools],
        rooms=["bedroom_1"],
    )


@pytest.fixture
def harness(monkeypatch):
    h = LLMPrefixHarness(prefix="You are Robot-0.", world_char_budget=100)
    h.captured = []

    def fake_post(messages):
        h.captured.append(messages)
        return h.next_response

    monkeypatch.setattr(h, "_post", fake_post)
    return h


def test_parses_tool_call_and_uses_prefix(harness):
    harness.next_response = "Navigate[table_1]"
    decision = harness.decide(_context())
    assert isinstance(decision, ToolCall)
    assert decision.name == "Navigate" and decision.args == "table_1"
    system = harness.captured[0][0]
    assert system["role"] == "system" and system["content"] == "You are Robot-0."


def test_done_response(harness):
    harness.next_response = "Done[]"
    assert isinstance(harness.decide(_context()), DoneDecision)


def test_unparseable_stays_feedback_never_done(harness):
    harness.next_response = "I think we should look around the house."
    for _ in range(4):
        decision = harness.decide(_context())
        assert isinstance(decision, ToolCall) and decision.name == "InvalidResponse"
        assert decision.error_kind == "parse_error"


def test_world_description_truncated_to_budget(harness):
    harness.next_response = "Navigate[table_1]"
    harness.decide(_context(world="x" * 10_000))
    user_message = harness.captured[0][1]["content"]
    assert "[...truncated...]" in user_message
    assert len(user_message) < 2_000


def test_payload_is_valid_json_request():
    h = LLMPrefixHarness()
    message = h._build_user_message(_context())
    json.dumps({"messages": [{"role": "user", "content": message}]})
    assert "Task: move the cup" in message
