import pytest

from harness_vla.planner.libero_vla_planner import (
    LiberoVLAPlanner,
    SYSTEM_PROMPT,
    parse_libero_vla_invocation,
)


def test_parser_accepts_tolerant_valid_invocation():
    raw_output = """Result:
```json
{"action":"vla_act","prompt":"pick the bowl","max_chunks":2,"tau":"task_success",}
```
"""

    assert parse_libero_vla_invocation(raw_output, 3) == {
        "action": "vla_act",
        "prompt": "pick the bowl",
        "max_chunks": 2,
        "tau": "task_success",
    }


def test_parser_accepts_lift_and_grasp_with_nominal_target():
    raw_output = (
        '{"action":"vla_act","prompt":"grasp and lift the bowl",'
        '"max_chunks":2,"tau":"lift_and_grasp","target":"akita_black_bowl_1"}'
    )

    assert parse_libero_vla_invocation(raw_output, 3) == {
        "action": "vla_act",
        "prompt": "grasp and lift the bowl",
        "max_chunks": 2,
        "tau": "lift_and_grasp",
        "target": "akita_black_bowl_1",
    }


@pytest.mark.parametrize(
    "raw_output",
    [
        "not JSON",
        '{"action":"grasp","prompt":"x","max_chunks":1,"tau":"task_success"}',
        '{"action":"vla_act","prompt":" ","max_chunks":1,"tau":"task_success"}',
        '{"action":"vla_act","prompt":"x","max_chunks":true,"tau":"task_success"}',
        '{"action":"vla_act","prompt":"x","max_chunks":1,"tau":"object_held"}',
        '{"action":"vla_act","prompt":"x","max_chunks":1,"tau":"lift_and_grasp"}',
        '{"action":"vla_act","prompt":"x","max_chunks":1,"tau":"lift_and_grasp","target":" "}',
    ],
)
def test_parser_rejects_invalid_invocations(raw_output):
    assert parse_libero_vla_invocation(raw_output, 2) is None


def test_parser_rejects_chunk_count_above_cap():
    raw_output = (
        '{"action":"vla_act","prompt":"pick the bowl",'
        '"max_chunks":3,"tau":"task_success"}'
    )

    assert parse_libero_vla_invocation(raw_output, 2) is None


def test_planner_turn_contains_only_text_task_context():
    class FakePlanner(LiberoVLAPlanner):
        def _chat(self, user_prompt):
            self.seen_prompt = user_prompt
            self.last_thinking = "deliberation"
            return (
                '{"action":"vla_act","prompt":"policy prompt",'
                '"max_chunks":2,"tau":"task_success"}'
            )

    planner = FakePlanner("gemma-test", think=True)
    invocation, raw_output = planner.act(
        "official instruction", 4, ["akita_black_bowl_1", "plate_1"]
    )

    assert invocation["prompt"] == "policy prompt"
    assert raw_output.startswith('{"action":"vla_act"')
    assert "Official task instruction: official instruction" in planner.seen_prompt
    assert "Available max_chunks cap: 4" in planner.seen_prompt
    assert 'Grounded target names: ["akita_black_bowl_1", "plate_1"]' in planner.seen_prompt
    assert "Required termination predicate for this phase: task_success" in planner.seen_prompt
    assert "no policy action has been executed" in planner.seen_prompt
    assert "pose" not in planner.seen_prompt.lower()
    assert "only one primitive, vla_act" in SYSTEM_PROMPT
    assert "complete Harness primitive library" in SYSTEM_PROMPT


def test_planner_rejects_invocation_for_a_different_phase_tau():
    class FakePlanner(LiberoVLAPlanner):
        def _chat(self, user_prompt):
            return (
                '{"action":"vla_act","prompt":"finish task",'
                '"max_chunks":2,"tau":"task_success"}'
            )

    planner = FakePlanner("gemma-test", required_tau="lift_and_grasp")

    invocation, _ = planner.act("pick and place", 4, ["target_1"])

    assert invocation is None


def test_planner_rejects_lift_target_outside_grounded_names():
    class FakePlanner(LiberoVLAPlanner):
        def _chat(self, user_prompt):
            return (
                '{"action":"vla_act","prompt":"grasp",'
                '"max_chunks":2,"tau":"lift_and_grasp","target":"invented_1"}'
            )

    planner = FakePlanner("gemma-test", required_tau="lift_and_grasp")

    invocation, _ = planner.act("pick and place", 4, ["target_1"])

    assert invocation is None