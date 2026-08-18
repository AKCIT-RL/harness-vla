import pytest

from harness_vla.planner.libero_multi_turn_planner import (
    LiberoMultiTurnPlanner,
    MULTI_TURN_SYSTEM_PROMPT,
    parse_libero_multi_turn_invocation,
)


TARGETS = ["akita_black_bowl_1", "plate_1"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            '{"action":"vla_act","prompt":"grasp the bowl",'
            '"target":"akita_black_bowl_1","max_chunks":3,'
            '"tau":"lift_and_grasp"}',
            {
                "action": "vla_act",
                "prompt": "grasp the bowl",
                "target": "akita_black_bowl_1",
                "max_chunks": 3,
                "tau": "lift_and_grasp",
            },
        ),
        (
            '{"action":"vla_act","prompt":"place the bowl into plate_1",'
            '"target":"akita_black_bowl_1","max_chunks":3,'
            '"tau":"task_success"}',
            {
                "action": "vla_act",
                "prompt": "place the bowl into plate_1",
                "target": "akita_black_bowl_1",
                "max_chunks": 3,
                "tau": "task_success",
            },
        ),
        (
            '{"action":"move_to","target":"plate_1",'
            '"mode":"above","gripper":"close"}',
            {
                "action": "move_to",
                "target": "plate_1",
                "mode": "above",
                "gripper": "close",
            },
        ),
        (
            '{"action":"move_pose","xyz":[0.1,-0.2,1.05],'
            '"pose":[0,0,0,1],"gripper":"open"}',
            {
                "action": "move_pose",
                "xyz": [0.1, -0.2, 1.05],
                "pose": [0.0, 0.0, 0.0, 1.0],
                "gripper": "open",
            },
        ),
        (
            '{"action":"rotate_wrist","target_yaw":1.25}',
            {"action": "rotate_wrist", "target_yaw": 1.25},
        ),
        (
            '{"action":"rotate_pitch","target_pitch":-0.5}',
            {"action": "rotate_pitch", "target_pitch": -0.5},
        ),
        (
            '{"action":"set_gripper","gripper":"close"}',
            {"action": "set_gripper", "gripper": "close"},
        ),
        ('{"action":"release"}', {"action": "release"}),
    ],
)
def test_parser_accepts_published_initial_primitives(raw, expected):
    assert parse_libero_multi_turn_invocation(
        raw, available_targets=TARGETS, max_chunks_cap=4
    ) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"action":"grasp","target":"akita_black_bowl_1"}',
        '{"action":"move_to","target":"invented","mode":"above","gripper":"close"}',
        '{"action":"move_to","target":"plate_1","mode":"above","gripper":"open"}',
        '{"action":"move_to","target":"plate_1","mode":"raw","gripper":"close"}',
        '{"action":"move_to","target":"plate_1","mode":"above","gripper":"close","xyz":[0,0,0]}',
        '{"action":"release","task_success":true}',
        '{"action":"vla_act","prompt":"x","target":"plate_1","max_chunks":5,"tau":"lift_and_grasp"}',
        '{"action":"vla_act","prompt":"x","target":"plate_1","max_chunks":1,"tau":"unknown"}',
        '{"action":"move_pose","xyz":[2,0,0.5],"pose":[0,0,0,1],"gripper":"open"}',
        '{"action":"move_pose","xyz":[0,0,0.5],"pose":[0,0,0,2],"gripper":"open"}',
        '{"action":"rotate_wrist","target_yaw":4}',
        '{"action":"rotate_pitch","target_pitch":2}',
        '{"action":"set_gripper","gripper":"hold"}',
        '{"action":"set_gripper","gripper":"open","extra":true}',
    ],
)
def test_parser_rejects_unsafe_or_out_of_contract_invocations(raw):
    assert parse_libero_multi_turn_invocation(
        raw, available_targets=TARGETS, max_chunks_cap=4
    ) is None


@pytest.mark.parametrize("cap", [0, -1, True, 1.5])
def test_parser_rejects_invalid_caps(cap):
    with pytest.raises(ValueError):
        parse_libero_multi_turn_invocation(
            '{"action":"release"}', available_targets=TARGETS, max_chunks_cap=cap
        )


def test_planner_turn_contains_semantic_state_without_pose_or_oracle():
    class FakePlanner(LiberoMultiTurnPlanner):
        def _chat(self, user_prompt):
            self.seen_prompt = user_prompt
            return '{"action":"release"}'

    planner = FakePlanner("gemma-test", think=True)
    invocation, _ = planner.act_turn(
        "place the bowl on the plate",
        {
            "holding": "akita_black_bowl_1",
            "last_action": "move_to",
            "last_feedback": {"primitive_success": True, "task_success": False},
            "budget": {"turns_remaining": 2, "actions_remaining": 30},
        },
        available_targets=TARGETS,
        max_chunks_cap=4,
    )

    assert invocation == {"action": "release"}
    assert "holding" in planner.seen_prompt
    assert "Grounded target names" in planner.seen_prompt
    assert "robot0_eef_pos" not in planner.seen_prompt
    assert "oracle" not in planner.seen_prompt.lower()
    assert "exactly one JSON primitive" in MULTI_TURN_SYSTEM_PROMPT
    assert "do not repeat it" in MULTI_TURN_SYSTEM_PROMPT
    assert "use release only for unconstrained" in MULTI_TURN_SYSTEM_PROMPT
    assert "For tight LIBERO placement" in MULTI_TURN_SYSTEM_PROMPT
    assert "one-chunk probe" in MULTI_TURN_SYSTEM_PROMPT
    assert "contact-rich placement" in MULTI_TURN_SYSTEM_PROMPT
    assert "holding exactly the" in MULTI_TURN_SYSTEM_PROMPT
    assert "name the grounded destination" in MULTI_TURN_SYSTEM_PROMPT
    assert "NEVER the destination" in MULTI_TURN_SYSTEM_PROMPT
    assert '"target":"alphabet_soup_1"' in MULTI_TURN_SYSTEM_PROMPT
    assert 'never "target":"basket_1"' in MULTI_TURN_SYSTEM_PROMPT
    assert "reserve" in MULTI_TURN_SYSTEM_PROMPT
    assert "for the FINAL object only" in MULTI_TURN_SYSTEM_PROMPT
    assert "turn_history" in MULTI_TURN_SYSTEM_PROMPT
    assert "never grasp or place it again" in MULTI_TURN_SYSTEM_PROMPT


def test_planner_turn_includes_symbolic_memory_context_and_rejects_pose_data():
    class FakePlanner(LiberoMultiTurnPlanner):
        def _chat(self, user_prompt):
            self.seen_prompt = user_prompt
            return '{"action":"release"}'

    planner = FakePlanner("gemma-test")
    context = (
        "Task Memory (symbolic task structure):\n"
        "grasp bowl, move above plate, release\n"
        "Promoted Global rules:\nre-ground before retry"
    )
    planner.act_turn(
        "place the bowl",
        {"holding": "bowl"},
        available_targets=TARGETS,
        max_chunks_cap=2,
        memory_context=context,
    )

    assert context in planner.seen_prompt
    with pytest.raises(ValueError, match="symbolic"):
        planner.act_turn(
            "place the bowl",
            {"holding": "bowl"},
            available_targets=TARGETS,
            max_chunks_cap=2,
            memory_context="target xyz: [0, 0, 1]",
        )