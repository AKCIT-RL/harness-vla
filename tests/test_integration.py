"""Integration test: multi-turn planner -> primitive compilation pipeline.

Exercises the sim-independent core of the harness (HarnessPlanner driving the
PrimitiveLibrary over several turns) using a scripted fake LLM client. This
validates the closed-loop contract (one primitive per turn, pose bookkeeping,
history feedback) without launching CoppeliaSim/PyRep.
"""

import json

from harness_vla.planner.harness_planner import HarnessPlanner
from harness_vla.primitives.primitives import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    PoseState,
    PrimitiveLibrary,
)


class _ScriptedClient:
    """Returns a preset sequence of assistant messages, one per create() call."""

    def __init__(self, scripted):
        self._scripted = list(scripted)
        self._i = 0
        self.chat = self  # so client.chat.completions works
        self.completions = self

    def create(self, **kwargs):
        content = self._scripted[min(self._i, len(self._scripted) - 1)]
        self._i += 1
        message = type("M", (), {"content": content})
        choice = type("C", (), {"message": message})
        return type("R", (), {"choices": [choice]})


def test_full_pick_and_place_loop():
    coords = {"object 1": [50, 60, 20], "object 2": [30, 40, 25]}
    script = [
        json.dumps({"reasoning": "stage above cube", "action": {"action": "move_to", "target": "object 1"}}),
        json.dumps({"reasoning": "grasp the cube", "action": {"action": "vla_act", "target": "object 1", "mode": "grasp"}}),
        json.dumps({"reasoning": "transport to destination", "action": {"action": "move_to", "target": "object 2"}}),
        json.dumps({"reasoning": "place the cube", "action": {"action": "vla_act", "object": "object 1", "destination": "object 2", "mode": "place"}}),
        json.dumps({"reasoning": "release", "action": {"action": "release", "lift": True}}),
    ]
    planner = HarnessPlanner(model_name="fake", client=_ScriptedClient(script))
    library = PrimitiveLibrary()

    pose = PoseState(x=0, y=0, z=90, gripper=GRIPPER_OPEN)
    history = []
    executed = []

    for _ in range(len(script)):
        invocation, raw = planner.act("pick object 1 and place on object 2", coords, pose.as_action(), history)
        assert invocation is not None, f"failed to parse: {raw}"
        result = library.compile(invocation, pose, coords)
        executed.append(result)
        pose = result.end_pose  # simulate perfect execution / pose update
        history.append({"action": invocation, "status": "success", "feedback": "ok"})

    assert planner.output_json_error == 0
    assert planner.planner_steps == len(script)

    names = [r.name for r in executed]
    assert names == ["move_to", "vla_act", "move_to", "vla_act", "release"]

    # The grasp must close the gripper; final release must open it.
    grasp = executed[1]
    assert grasp.is_contact
    assert grasp.end_pose.gripper == GRIPPER_CLOSED
    assert executed[-1].end_pose.gripper == GRIPPER_OPEN

    # Total discrete actions the env would receive across the episode.
    total_actions = sum(len(r.actions) for r in executed)
    assert total_actions == 1 + 4 + 1 + 3 + 2  # move, grasp, move, place, release(+lift)


def test_loop_recovers_from_parse_error_turn():
    coords = {"object 1": [10, 10, 10]}
    script = [
        "I'm not sure, let me think...",  # unparseable
        json.dumps({"action": {"action": "move_to", "target": "object 1"}}),
    ]
    planner = HarnessPlanner(model_name="fake", client=_ScriptedClient(script))

    inv1, _ = planner.act("go", coords, [0, 0, 0, 0, 0, 0, 1], [])
    assert inv1 is None
    assert planner.output_json_error == 1

    inv2, _ = planner.act("go", coords, [0, 0, 0, 0, 0, 0, 1], [{"action": None, "status": "parse_error", "feedback": "x"}])
    assert inv2["action"] == "move_to"
