"""Tests for the PARTNR primitive codec, symbolization and prompts (no env needed)."""

from harness_vla.envs.partnr.primitives import (
    PARTNR_PRIMITIVE_NAMES,
    format_action,
    parse_action,
    symbolize_partnr_action,
)
from harness_vla.envs.partnr.prompts import (
    build_partnr_system_prompt,
    build_partnr_turn_prompt,
    render_state,
)

STATE = {
    "rooms": ["kitchen_1", "living_room_1"],
    "furniture": {
        "table_1": {"room": "kitchen_1"},
        "cabinet_2": {"room": "kitchen_1", "states": {"is_open": False}},
        "sofa_3": {"room": "living_room_1"},
    },
    "objects": {
        "cup_0": {"room": "kitchen_1", "relation": "on", "furniture": "table_1", "held_by": None},
        "plate_1": {"room": None, "relation": None, "furniture": None, "held_by": 1},
    },
    "agents": {
        "0": {"room": "kitchen_1", "holding": None},
        "1": {"room": "living_room_1", "holding": "plate_1"},
    },
}


def test_format_and_parse_roundtrip():
    text = format_action("Place", ["cup_0", "on", "table_1", "None", "None"])
    assert text == "Place[cup_0, on, table_1, None, None]"
    assert parse_action(text) == ("Place", ["cup_0", "on", "table_1", "None", "None"])
    assert parse_action("Wait[]") == ("Wait", [])
    assert parse_action("no action here") is None


def test_parse_json_shapes():
    assert parse_action('{"thought": "go", "action": "Navigate[table_1]"}') == ("Navigate", ["table_1"])
    assert parse_action('{"action": "Pick", "args": "cup_0"}') == ("Pick", ["cup_0"])
    assert parse_action('{"action": "Place", "args": ["cup_0", "on", "table_1"]}') == (
        "Place", ["cup_0", "on", "table_1"],
    )
    assert parse_action('```json\n{"action":"Done[]"}\n```') == ("Done", [])


def test_symbolize_uses_vocabulary_and_fallback():
    symbolic = symbolize_partnr_action(
        "Place[cup_0, on, table_1, next_to, plate_1]",
        objects={"cup_0", "plate_1"}, furniture={"table_1"}, rooms={"kitchen_1"},
    )
    assert symbolic == "Place[<obj:cup>, on, table_1, next_to, <obj:plate>]".replace(
        "table_1", "<furn:table>"
    )
    assert symbolize_partnr_action("Navigate[kitchen_1]", rooms={"kitchen_1"}) == "Navigate[<room:kitchen>]"
    assert symbolize_partnr_action("Navigate[living_room_1]") == "Navigate[<ent:living_room>]"
    assert "Done" in PARTNR_PRIMITIVE_NAMES and "Wait" in PARTNR_PRIMITIVE_NAMES


def test_render_state_is_compact_and_marks_partner_holding():
    text = render_state(STATE, agent_uid=0)
    assert "kitchen_1: table_1, cabinet_2 (is_open=False)" in text
    assert "cup_0: on table_1 (kitchen_1)" in text
    assert "plate_1: held by agent 1" in text
    assert "agent 0 (you): in kitchen_1, holding nothing" in text
    assert "agent 1 (partner): in living_room_1, holding plate_1" in text


def test_system_prompt_omits_empty_memory_and_has_output_schema():
    system = build_partnr_system_prompt("You are Robot-0.", "  - Navigate: nav", "")
    assert system.startswith("You are Robot-0.")
    assert "Global Memory" not in system
    assert '"action": "Name[arg1, arg2]"' in system
    with_memory = build_partnr_system_prompt("p", "", "Global Memory:\n 1. rule")
    assert "Global Memory" in with_memory


def test_turn_prompt_full_history_and_truncation():
    ctx = {
        "instruction": "Move the cup to the sofa.",
        "agent_uid": 0,
        "decision_index": 3,
        "max_decisions": 50,
        "world_description": "ignored when state is present",
        "partner_recent_actions": ["Pick[plate_1] -> successful execution"],
        "signals": [{"kind": "repeat", "message": "You repeated Navigate[table_1]."}],
        "extras": {"state": STATE},
    }
    history = [
        {"decision_index": i, "action": f"Navigate[table_{i}]", "status": "success", "message": "Successful execution"}
        for i in range(3)
    ]
    prompt = build_partnr_turn_prompt(ctx, history, ["Navigate[table_1]", "Pick[cup_0]"], max_history=0)
    for i in range(3):
        assert f"{i + 1}. Navigate[table_{i}] -> success" in prompt
    assert "Decision 4 of 50" in prompt
    assert "Task Specific Memory structural prior" in prompt and "2. Pick[cup_0]" in prompt
    assert "Partner recent actions" in prompt and "You repeated Navigate[table_1]." in prompt
    assert "ignored when state is present" not in prompt

    windowed = build_partnr_turn_prompt(ctx, history, None, max_history=1)
    assert "Navigate[table_0]" not in windowed and "Navigate[table_2]" in windowed
    assert "[2 earlier steps omitted]" in windowed

    tiny = build_partnr_turn_prompt(ctx, [], None, state_char_budget=40)
    assert "[...truncated to budget...]" in tiny
    assert "No actions executed yet this episode." in tiny
