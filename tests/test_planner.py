"""Unit tests for GlobalMemory and the planner JSON parsing (no network)."""

import json
from unittest.mock import patch

from harness_vla.memory.global_memory import (
    GlobalMemory,
    SEED_FAILURE_MODELS,
    SEED_SUCCESS_RULES,
)
from harness_vla.planner.harness_planner import (
    HarnessPlanner,
    extract_json_object,
)
from harness_vla.planner.prompts import build_turn_prompt


# ---- GlobalMemory ----------------------------------------------------------

def test_seeded_memory_has_rules():
    gm = GlobalMemory.seeded()
    assert gm.success_rules == SEED_SUCCESS_RULES
    assert gm.failure_models == SEED_FAILURE_MODELS


def test_render_includes_rules():
    gm = GlobalMemory.seeded()
    text = gm.render()
    assert "Success rules:" in text
    assert "Failure models:" in text
    assert "empty grasp" in text.lower()


def test_set_global_memory_rebuilds_system_prompt():
    planner = HarnessPlanner(
        model_name="fake",
        client=_FakeClient("unused"),
        global_memory=GlobalMemory(
            success_rules=["legacy-memory-sentinel"], failure_models=[]
        ),
    )

    planner.set_global_memory(
        GlobalMemory(success_rules=["promoted-memory-sentinel"], failure_models=[])
    )

    assert "promoted-memory-sentinel" in planner.system_prompt
    assert "legacy-memory-sentinel" not in planner.system_prompt


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "gm.json"
    gm = GlobalMemory(success_rules=["r1"], failure_models=["f1"])
    gm.save(str(path))
    loaded = GlobalMemory.load(str(path))
    assert loaded.success_rules == ["r1"]
    assert loaded.failure_models == ["f1"]


def test_load_missing_falls_back_to_seed(tmp_path):
    gm = GlobalMemory.load(str(tmp_path / "nope.json"))
    assert gm.success_rules == SEED_SUCCESS_RULES


def test_load_corrupt_falls_back(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    gm = GlobalMemory.load(str(path))
    assert gm.success_rules == SEED_SUCCESS_RULES


# ---- JSON extraction -------------------------------------------------------

def test_extract_plain_json():
    obj = extract_json_object('{"action": {"action": "move_to", "xyz": [1,2,3]}}')
    assert obj["action"]["action"] == "move_to"


def test_extract_from_markdown_fence():
    text = 'Sure!\n```json\n{"action": {"action": "release"}}\n```\n'
    obj = extract_json_object(text)
    assert obj["action"]["action"] == "release"


def test_extract_with_surrounding_prose():
    text = 'Here is my plan: {"reasoning": "x", "action": {"action": "set_gripper", "gripper": "close"}} done.'
    obj = extract_json_object(text)
    assert obj["action"]["action"] == "set_gripper"


def test_extract_trailing_comma():
    obj = extract_json_object('{"action": {"action": "move_to", "xyz": [1,2,3],},}')
    assert obj["action"]["action"] == "move_to"


def test_extract_invalid_returns_none():
    assert extract_json_object("no json here") is None
    assert extract_json_object("") is None


# ---- HarnessPlanner with a fake client ------------------------------------

class _FakeMessage:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return type("R", (), {"choices": [_FakeMessage(self._content)]})


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content):
        self.chat = _FakeChat(content)


def _planner(content):
    return HarnessPlanner(model_name="fake", client=_FakeClient(content))


def test_planner_parses_nested_action():
    planner = _planner('{"reasoning": "go", "action": {"action": "move_to", "xyz": [5,5,5]}}')
    inv, raw = planner.act("pick the cube", {"object 1": [5, 5, 5]}, [50, 50, 50, 60, 60, 60, 1], [])
    assert inv["action"] == "move_to"
    assert planner.output_json_error == 0
    assert planner.planner_steps == 1


def test_planner_parses_bare_invocation():
    planner = _planner('{"action": "release", "lift": true}')
    inv, _ = planner.act("x", {}, [0, 0, 0, 0, 0, 0, 1], [])
    assert inv["action"] == "release"


def test_planner_parses_name_as_key_form():
    # Small models often emit {"action": {"vla_act": {...}}} or {"vla_act": {...}}.
    planner = _planner('{"reasoning": "grasp", "action": {"vla_act": {"target": "object 1", "mode": "grasp"}}}')
    inv, _ = planner.act("x", {"object 1": [1, 2, 3]}, [0, 0, 0, 0, 0, 0, 1], [])
    assert inv["action"] == "vla_act"
    assert inv["target"] == "object 1"


def test_planner_parses_top_level_name_key():
    planner = _planner('{"move_to": {"xyz": [5, 5, 5]}}')
    inv, _ = planner.act("x", {}, [0, 0, 0, 0, 0, 0, 1], [])
    assert inv["action"] == "move_to"
    assert inv["xyz"] == [5, 5, 5]


def test_planner_counts_json_error():
    planner = _planner("I cannot help with that")
    inv, _ = planner.act("x", {}, [0, 0, 0, 0, 0, 0, 1], [])
    assert inv is None
    assert planner.output_json_error == 1


def test_planner_accepts_optional_roles_and_labels():
    planner = _planner('{"action": "release"}')
    invocation, _ = planner.act(
        "place the star",
        {"object 1": [1, 2, 3], "object 2": [4, 5, 6]},
        [0, 0, 0, 0, 0, 0, 1],
        [],
        object_roles={"object 1": ["manipulable"], "object 2": ["destination"]},
        object_labels={"object 1": "first star", "object 2": "shape sorter"},
    )
    assert invocation["action"] == "release"


def test_planner_can_disable_ollama_thinking():
    response = type("Response", (), {
        "__enter__": lambda self: self,
        "__exit__": lambda self, *args: None,
        "read": lambda self: b'{"message":{"content":"{\\"action\\":\\"release\\"}"}}',
    })()
    planner = HarnessPlanner(
        model_name="gemma4:12b",
        base_url="http://localhost:11434/v1",
        disable_thinking=True,
        request_timeout=123,
        client=_FakeClient("unused"),
    )

    with patch("harness_vla.planner.harness_planner.request.urlopen", return_value=response) as urlopen:
        invocation, raw = planner.act("x", {}, [0, 0, 0, 0, 0, 0, 1], [])

    assert invocation["action"] == "release"
    assert raw == '{"action":"release"}'
    sent = json.loads(urlopen.call_args.args[0].data)
    assert sent["think"] is False
    assert sent["options"]["num_predict"] == 1024
    assert "num_ctx" not in sent["options"]
    assert urlopen.call_args.args[0].full_url == "http://localhost:11434/api/chat"
    assert urlopen.call_args.kwargs["timeout"] == 123


def test_planner_can_enable_ollama_thinking():
    body = (
        b'{"message":{"thinking":"I should release now.",'
        b'"content":"{\\"action\\":\\"release\\"}"}}'
    )
    response = type("Response", (), {
        "__enter__": lambda self: self,
        "__exit__": lambda self, *args: None,
        "read": lambda self: body,
    })()
    planner = HarnessPlanner(
        model_name="gemma4:12b",
        base_url="http://localhost:11434/v1",
        enable_thinking=True,
        num_ctx=16384,
        client=_FakeClient("unused"),
    )

    with patch("harness_vla.planner.harness_planner.request.urlopen", return_value=response) as urlopen:
        invocation, raw = planner.act("x", {}, [0, 0, 0, 0, 0, 0, 1], [])

    assert invocation["action"] == "release"
    assert raw == '{"action":"release"}'
    assert planner.last_thinking == "I should release now."
    sent = json.loads(urlopen.call_args.args[0].data)
    assert sent["think"] is True
    assert sent["options"]["num_ctx"] == 16384


def test_thinking_modes_are_mutually_exclusive():
    try:
        HarnessPlanner(
            model_name="gemma4:12b",
            disable_thinking=True,
            enable_thinking=True,
            client=_FakeClient("unused"),
        )
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_disable_ollama_thinking_rejects_incompatible_url_path():
    planner = HarnessPlanner(
        model_name="gemma4:12b",
        base_url="http://localhost:11434/openai/v1",
        disable_thinking=True,
        client=_FakeClient("unused"),
    )

    try:
        planner._chat("x")
    except ValueError as error:
        assert "Ollama base URL" in str(error)
    else:
        raise AssertionError("Expected incompatible Ollama URL to be rejected")


def test_turn_prompt_separates_roles_and_uses_labels_without_color_claims():
    prompt = build_turn_prompt(
        "place the star",
        {"object 1": [1, 2, 3], "object 2": [4, 5, 6]},
        [0, 0, 0, 0, 0, 0, 1],
        [],
        object_roles={"object 1": ["manipulable"], "object 2": ["destination"]},
        object_labels={"object 1": "first star", "object 2": "shape sorter"},
    )
    manipulable, destinations = prompt.split("Destination candidates", 1)
    assert "object 1: coords=[1, 2, 3], label=first star" in manipulable
    assert "object 2: coords=[4, 5, 6], label=shape sorter" in destinations
    assert "distinct" in prompt
    assert "color" not in prompt.lower()


def test_turn_prompt_marks_resolved_task_memory_as_current_scene_prior():
    prompt = build_turn_prompt(
        "place the cube",
        {"current cube": [11, 12, 13]},
        [0, 0, 0, 0, 0, 0, 1],
        [],
        resolved_task_memory=[{
            "sequence": 1,
            "source_turn": 1,
            "action": "move_to",
            "target": "current cube",
            "xyz": [11, 12, 13],
        }],
    )

    assert "structural prior (not an execution script)" in prompt
    assert "CURRENT scene" in prompt
    assert "current feedback are authoritative and take precedence" in prompt
    assert '"xyz": [11, 12, 13]' in prompt


def test_turn_prompt_without_task_memory_preserves_zero_shot_output():
    args = (
        "place the cube",
        {"object 1": [1, 2, 3]},
        [0, 0, 0, 0, 0, 0, 1],
        [],
    )

    assert build_turn_prompt(*args) == build_turn_prompt(
        *args, resolved_task_memory=None
    )


def test_turn_prompt_reports_authoritative_attachment_state():
    held = build_turn_prompt(
        "move the cube",
        {"object 1": [1, 2, 3]},
        [0, 0, 0, 0, 0, 0, 0],
        [],
        held_object_id="object 1",
        attachment_evidence_available=True,
    )
    assert "Currently attached object (authoritative simulator state): object 1" in held
    assert "Keep the gripper closed during transport" in held

    empty = build_turn_prompt(
        "move the cube",
        {"object 1": [1, 2, 3]},
        [0, 0, 0, 0, 0, 0, 1],
        [],
        held_object_id=None,
        attachment_evidence_available=True,
    )
    assert "authoritative simulator state): none" in empty

    no_evidence = build_turn_prompt(
        "move the cube",
        {"object 1": [1, 2, 3]},
        [0, 0, 0, 0, 0, 0, 1],
        [],
        held_object_id="object 1",
        attachment_evidence_available=False,
    )
    assert "authoritative simulator state" not in no_evidence


def test_seed_failure_models_cover_transport_detach():
    joined = " ".join(SEED_FAILURE_MODELS).lower()
    assert "detach during transport" in joined
    assert "gripper=\"close\"" in " ".join(SEED_FAILURE_MODELS)


def test_seed_failure_models_cover_repeated_grasp_pose():
    joined = " ".join(SEED_FAILURE_MODELS).lower()
    assert "repeated grasp at the same pose" in joined
    assert "rotate_wrist" in joined


def test_planner_reset():
    planner = _planner("garbage")
    planner.act("x", {}, [0, 0, 0, 0, 0, 0, 1], [])
    planner.reset()
    assert planner.planner_steps == 0
    assert planner.output_json_error == 0


def test_system_prompt_contains_primitives():
    planner = _planner("{}")
    assert "vla_act" in planner.system_prompt
    assert "move_to" in planner.system_prompt
    assert "grasp_verified" in planner.system_prompt
    assert "NOT proof" in planner.system_prompt
    assert 'Legacy "target" is NEVER allowed for place' in planner.system_prompt
    assert "labels, and roles supplied each turn are" in planner.system_prompt
    assert "authoritative" in planner.system_prompt
