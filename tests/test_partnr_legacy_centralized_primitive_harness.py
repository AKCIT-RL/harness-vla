"""beta-only: ported parser/prompt tests for the legacy centralized primitive harness."""

import pytest

pytest.importorskip("partnr_harness.contracts")

import json

from partnr_harness.contracts import DoneDecision, StepContext, ToolCall
from harness_vla.envs.partnr.legacy.centralized_primitive_harness import (
    CentralizedPrimitiveHarness,
)


def make_harness():
    harness = CentralizedPrimitiveHarness(prefix="test")
    harness.captured_messages = []

    def fake_post(messages):
        harness.captured_messages.append(messages)
        return harness.next_raw

    harness._post = fake_post
    return harness


def make_context(**extras):
    extras.setdefault("centralized", True)
    extras.setdefault(
        "tools_by_agent",
        {
            "agent_0": [
                {"name": "Navigate", "description": "nav"},
                {"name": "Pick", "description": "pick"},
                {"name": "Wait", "description": "wait"},
            ],
            "agent_1": [
                {"name": "Navigate", "description": "nav"},
                {"name": "Open", "description": "open"},
                {"name": "Wait", "description": "wait"},
            ],
        },
    )
    extras.setdefault("deciding_agents", ["agent_0", "agent_1"])
    extras.setdefault("busy_agents", [])
    return StepContext(
        instruction="Put the cup on the table",
        agent_uid=0,
        decision_index=0,
        max_decisions=100,
        world_description="kitchen_1: counter_20, cup_0",
        extras=extras,
    )


def test_valid_two_agent_response():
    harness = make_harness()
    harness.next_raw = json.dumps(
        {
            "agent_0": {"action": "Pick", "args": "cup_0"},
            "agent_1": {"action": "Navigate", "args": "counter_20"},
        }
    )
    decisions = harness.decide(make_context())
    assert isinstance(decisions["agent_0"], ToolCall)
    assert decisions["agent_0"].name == "Pick"
    assert decisions["agent_0"].args == "cup_0"
    assert decisions["agent_1"].name == "Navigate"


def test_global_done():
    harness = make_harness()
    harness.next_raw = json.dumps({"all": {"action": "Done", "args": ""}})
    decisions = harness.decide(make_context())
    assert isinstance(decisions["all"], DoneDecision)


def test_per_agent_done_becomes_done_decision_for_planner_wait():
    harness = make_harness()
    harness.next_raw = json.dumps(
        {
            "agent_0": {"action": "Done", "args": ""},
            "agent_1": {"action": "Wait", "args": ""},
        }
    )
    decisions = harness.decide(make_context())
    assert isinstance(decisions["agent_0"], DoneDecision)
    assert isinstance(decisions["agent_1"], ToolCall)


def test_tool_from_wrong_agent_is_compile_error():
    harness = make_harness()
    harness.next_raw = json.dumps(
        {"agent_1": {"action": "Pick", "args": "cup_0"}}  # Pick not in agent_1 tools
    )
    decisions = harness.decide(make_context())
    call = decisions["agent_1"]
    assert call.name == "InvalidResponse"
    assert call.error_kind == "compile_error"


def test_place_args_are_normalized_to_five_fields():
    harness = make_harness()
    harness.next_raw = json.dumps(
        {"agent_0": {"action": "Pick", "args": "cup_0"}}
    )
    context = make_context()
    context.extras["tools_by_agent"]["agent_0"].append(
        {"name": "Place", "description": "place"}
    )
    decisions = harness.decide(context)
    assert decisions["agent_0"].name == "Pick"

    harness.next_raw = json.dumps(
        {"agent_0": {"action": "Place", "args": "cup_0, on, counter_20"}}
    )
    decisions = harness.decide(context)
    assert decisions["agent_0"].args == "cup_0,on,counter_20,None,None"


def test_unparseable_output_yields_parse_error_never_done():
    # Parse failures no longer escalate to a global Done.
    harness = make_harness()
    harness.max_consecutive_parse_failures = 2
    for raw in ("I think we should pick the cup.", "still not json", "nope"):
        harness.next_raw = raw
        decisions = harness.decide(make_context())
        assert "all" not in decisions
        assert all(
            isinstance(d, ToolCall) and d.error_kind == "parse_error"
            for d in decisions.values()
        )


def test_bracket_action_with_embedded_args():
    harness = make_harness()
    harness.next_raw = json.dumps(
        {"agent_0": {"action": "Navigate[counter_20]", "args": ""}}
    )
    decisions = harness.decide(make_context())
    assert decisions["agent_0"].name == "Navigate"
    assert decisions["agent_0"].args == "counter_20"


def test_prompt_lists_tools_per_agent_and_busy_agents():
    harness = make_harness()
    harness.next_raw = json.dumps({"agent_0": {"action": "Wait", "args": ""}})
    context = make_context(busy_agents=["agent_1"])
    harness.decide(context)
    user_message = harness.captured_messages[0][1]["content"]
    assert "Tools for agent_0" in user_message
    assert "Tools for agent_1" in user_message
    assert "agent_1" in user_message and "still executing" in user_message
    assert '"all"' in user_message  # global Done instructions


def test_prompt_has_no_coordination_heuristics():
    # Coordination strategy is left to the model: no tuned nudges in the prompt.
    # The in-flight skill of busy agents is still surfaced as plain state.
    harness = make_harness()
    harness.next_raw = json.dumps({"agent_0": {"action": "Wait", "args": ""}})
    context = make_context(
        busy_agents=["agent_1"],
        in_flight_actions={"agent_1": "Navigate[table_10]"},
    )
    harness.decide(context)
    user_message = harness.captured_messages[0][1]["content"].lower()
    for banned in ("parallel", "disjoint", "idle", "gripper", "never repeat"):
        assert banned not in user_message
    # Busy agent's in-flight skill is surfaced to the model.
    assert "navigate[table_10]" in user_message
    assert "do not issue actions for them" in user_message
