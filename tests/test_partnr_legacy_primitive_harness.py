"""beta-only: ported tests for the legacy primitive-action harness."""

import pytest

pytest.importorskip("partnr_harness.contracts")

from partnr_harness.contracts import DoneDecision, StepContext, ToolCall, ToolSpec

from harness_vla.envs.partnr.legacy.primitive_harness import (
    PrimitiveHarness,
    extract_json_object,
)

PRIMITIVE_TOOLS = [
    ToolSpec(name, f"{name} tool")
    for name in (
        "Navigate", "Pick", "Place", "Open", "Close", "PowerOn", "PowerOff",
        "Clean", "Fill", "Pour", "Wait",
    )
]


def make_context(**overrides) -> StepContext:
    defaults = dict(
        instruction="Move the laptop to the kitchen table.",
        world_description="living_room_1: table_13, laptop_0",
        tools=PRIMITIVE_TOOLS,
        rooms=["living_room_1"],
        agent_uid=0,
        decision_index=0,
        max_decisions=30,
    )
    defaults.update(overrides)
    return StepContext(**defaults)


def make_harness(replies):
    harness = PrimitiveHarness(base_url="http://unused:1/v1", model="fake")
    replies = list(replies)
    harness._post = lambda messages: replies.pop(0)
    return harness


def test_extract_json_object_variants():
    assert extract_json_object('{"action":"Pick","args":"laptop_0"}') == {
        "action": "Pick",
        "args": "laptop_0",
    }
    fenced = '```json\n{"action":"Pick","args":"laptop_0"}\n```'
    assert extract_json_object(fenced)["action"] == "Pick"
    prose = 'Sure! {"action":"Pick","args":"laptop_0",} done'
    assert extract_json_object(prose)["args"] == "laptop_0"
    gemma_1b = '```json\n{"tool":"Wait", "args":{}】\n```'
    assert extract_json_object(gemma_1b) == {"tool": "Wait", "args": {}}
    assert extract_json_object("no json here") is None


def test_decide_tool_call():
    harness = make_harness(['{"action":"Navigate","args":"table_13"}'])
    decision = harness.decide(make_context())
    assert isinstance(decision, ToolCall)
    assert decision.name == "Navigate"
    assert decision.args == "table_13"
    assert decision.error_kind is None


def test_decide_done():
    harness = make_harness(['{"action":"Done","args":""}'])
    assert isinstance(harness.decide(make_context()), DoneDecision)


def test_place_three_args_padded_with_none():
    harness = make_harness(['{"action":"Place","args":"laptop_0,on,table_13"}'])
    decision = harness.decide(make_context())
    assert decision.args == "laptop_0,on,table_13,None,None"


def test_place_next_to_is_stripped():
    harness = make_harness(
        ['{"action":"Place","args":"laptop_0,on,table_13,next_to,book_2"}']
    )
    decision = harness.decide(make_context())
    assert decision.args == "laptop_0,on,table_13,None,None"


def test_bracketed_action_is_unwrapped():
    harness = make_harness(['{"action":"Pick[laptop_0]","args":""}'])
    decision = harness.decide(make_context())
    assert decision.name == "Pick"
    assert decision.args == "laptop_0"


def test_bracketed_action_with_conflicting_args_prefers_embedded():
    harness = make_harness(['{"action":"Navigate[shelf_1]","args":"table_2"}'])
    decision = harness.decide(make_context())
    assert decision.name == "Navigate"
    assert decision.args == "shelf_1"


def test_plain_bracket_call_fallback():
    harness = make_harness(["Navigate[table_13] -> pending"])
    decision = harness.decide(make_context())
    assert decision.name == "Navigate"
    assert decision.args == "table_13"


def test_unknown_tool_is_compile_error():
    harness = make_harness(['{"action":"Rearrange","args":"laptop_0,on,table_13"}'])
    decision = harness.decide(make_context())
    assert decision.name == "InvalidResponse"
    assert decision.error_kind == "compile_error"


def test_unparseable_output_is_parse_error():
    harness = make_harness(["I will pick up the laptop now."])
    decision = harness.decide(make_context())
    assert decision.name == "InvalidResponse"
    assert decision.error_kind == "parse_error"
