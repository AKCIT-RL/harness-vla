"""The controller adapts to another environment through a duck-typed profile."""

import pytest

from partnr_harness.contracts import (
    EpisodeContext,
    StepContext,
    ToolCall,
    ToolSpec,
)

from harness_vla.envs.partnr import primitives
from harness_vla.envs.partnr.agent import HarnessVlaCentralized, labeled_skeleton
from harness_vla.envs.partnr.memory import extraction_system
from harness_vla.envs.partnr.primitives import Vocabulary

TOOLS = [ToolSpec("GoToShelf", "Go to a shelf"), ToolSpec("Wait", "Wait")]


class Profile:
    role = "You coordinate a warehouse."
    role_suffix = ""
    primitive_reference = "Primitives: GoToShelf[shelf_id], GoToGoal[goal_id], Wait[]."
    guidance = "Never send two AGVs to the same shelf."
    agent_descriptions = {"agent_0": "AGV", "agent_1": "picker"}
    vocabulary = Vocabulary(
        ("GoToShelf", "GoToGoal", "Wait", "Done"),
        domain="a warehouse robot",
        instance_examples="(shelf_40, goal_3)",
        scope="warehouses",
    )


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.last_meta = {}

    def start_episode(self, agent_uid=None):
        pass

    def post(self, messages):
        self.calls.append(messages)
        self.last_meta = {"prompt_tokens": 10, "completion_tokens": 5}
        return self.replies.pop(0)


@pytest.fixture(autouse=True)
def _restore_vocabulary():
    previous = primitives.use_vocabulary(None)
    yield
    primitives.use_vocabulary(previous)


def _agent(replies):
    agent = HarnessVlaCentralized(use_claims=False, profile=Profile())
    agent._llm = FakeLLM(replies)
    tools = [{"name": t.name, "description": t.description} for t in TOOLS]
    agent.on_episode_start(
        EpisodeContext(
            "ep", "Deliver 3 shelves", 0, [1], TOOLS,
            extras={"agents": {"agent_0": {"uid": 0, "tools": tools},
                               "agent_1": {"uid": 1, "tools": tools}}},
        )
    )
    return agent


def _step(state=None):
    return StepContext(
        "Deliver 3 shelves", 0, 0, 50, "", TOOLS, [],
        extras={"deciding_agents": ["agent_0"], "state": state or {},
                "rendered_state": "Requested shelves: shelf_40 at (3, 17).",
                "state_title": "Warehouse state"},
    )


def test_profile_replaces_prompts_and_state_rendering():
    agent = _agent(['{"thought": "x", "agent_0": "GoToShelf[shelf_40]", "agent_1": null}'])
    decisions = agent.decide(_step())
    assert isinstance(decisions["agent_0"], ToolCall) and decisions["agent_0"].name == "GoToShelf"
    system = agent._llm.calls[0][0]["content"]
    user = agent._llm.calls[0][1]["content"]
    assert "You coordinate a warehouse." in system and "GoToGoal[goal_id]" in system
    assert "Never send two AGVs" in system
    assert "Navigate[target]" not in system and "household" not in system.lower()
    assert "Warehouse state (authoritative, current):" in user
    assert "Requested shelves: shelf_40 at (3, 17)." in user


def test_vocabulary_scopes_the_memory_skeleton_and_extraction_prompt():
    _agent([])
    history = [{"agent": "agent_0", "name": "GoToShelf", "args": "shelf_40", "ok": True,
                "vocab": {"objects": ["shelf_40"]}}]
    assert labeled_skeleton(history) == ["agent_0: GoToShelf[<obj:shelf>]"]
    assert "warehouse robot" in extraction_system() and "GoToShelf" in extraction_system()


def test_default_vocabulary_is_unchanged_without_a_profile():
    agent = HarnessVlaCentralized(use_claims=False)
    assert agent.profile is None
    assert primitives.active_primitive_names() == primitives.PARTNR_PRIMITIVE_NAMES
    assert "household" in extraction_system()
