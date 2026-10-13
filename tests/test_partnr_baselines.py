"""PARTNR LLM-free smoke agent driven through the env API (no simulator/LLM)."""

from akc_harness.contracts import (
    ActionResult,
    DoneDecision,
    EpisodeContext,
    EpisodeSummary,
    StepContext,
    ToolSpec,
)
from partnr_harness.external import ExternalLoopHarness

from harness_vla.envs.partnr.baselines.random_loop import RandomLoopAgent

TOOLS = [ToolSpec("Navigate", "nav to"), ToolSpec("Explore", "explore a room"), ToolSpec("Wait", "")]
STATE = {
    "rooms": ["kitchen_1", "bedroom_1"],
    "furniture": {"table_1": {"room": "kitchen_1"}},
    "objects": {},
    "agents": {},
    "text": "w",
}
ACTION_SPACE = [
    {"name": "Explore", "description": "", "args_format": "Explore[<room>]", "arg_names": ["room"]},
    {"name": "Navigate", "description": "", "args_format": "Navigate[<x>]", "arg_names": ["nav_target"]},
    {"name": "Wait", "description": "", "args_format": "Wait[]", "arg_names": []},
]


def episode_context(**extras):
    return EpisodeContext(
        episode_id="7", instruction="tidy up", agent_uid=0, partner_uids=[1], tools=TOOLS, extras=extras
    )


def step_context(index, last_result=None):
    return StepContext(
        instruction="tidy up",
        agent_uid=0,
        decision_index=index,
        max_decisions=10,
        world_description="kitchen_1: table_1\nbedroom_1: bed_1",
        tools=TOOLS,
        rooms=["kitchen_1", "bedroom_1"],
        last_result=last_result,
        extras={"state": STATE},
    )


def drive_bridge(harness, max_decisions=10, response="Successful execution!"):
    """Fake bridge: decide/on_action_result until Done or budget; return decisions."""
    harness.on_episode_start(episode_context(action_space=ACTION_SPACE))
    decisions = []
    last = None
    for index in range(max_decisions):
        decision = harness.decide(step_context(index, last))
        decisions.append(decision)
        if isinstance(decision, DoneDecision):
            break
        last = ActionResult(
            tool=decision.name, args=decision.args, response=response,
            success_hint=True, decision_index=index,
        )
        harness.on_action_result(last)
    harness.on_episode_end(EpisodeSummary("7", len(decisions), isinstance(decisions[-1], DoneDecision)))
    return decisions


def test_random_loop_agent_by_dotted_path_uses_state_args():
    harness = ExternalLoopHarness(
        agent_target="harness_vla.envs.partnr.baselines.random_loop.RandomLoopAgent",
        agent_kwargs={"max_steps": 3, "seed": 1},
        decision_timeout_s=5,
    )
    decisions = drive_bridge(harness)
    assert len(decisions) == 4
    assert isinstance(decisions[-1], DoneDecision)
    for decision in decisions[:-1]:
        assert decision.name in {"Explore", "Navigate", "Wait"}
        if decision.name == "Explore":
            assert decision.args in STATE["rooms"]
        elif decision.name == "Navigate":
            assert decision.args == "table_1"
        else:
            assert decision.args == ""
    assert RandomLoopAgent.__name__ == "RandomLoopAgent"
