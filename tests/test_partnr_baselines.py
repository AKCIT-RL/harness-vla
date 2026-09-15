"""PARTNR agent baselines (beta-only) driven through the env contract, no simulator/LLM."""

from partnr_harness.contracts import (
    ActionResult,
    DoneDecision,
    EpisodeContext,
    EpisodeSummary,
    Signal,
    StepContext,
    ToolCall,
    ToolSpec,
)
from partnr_harness.external import ExternalLoopHarness

from harness_vla.envs.partnr.baselines.random_harness import RandomHarness
from harness_vla.envs.partnr.baselines.random_loop import RandomLoopAgent
from harness_vla.envs.partnr.baselines.react import (
    STOPWORD,
    ReActHarness,
    parse_react_response,
)
from harness_vla.envs.partnr.baselines.recovery_aware import RecoveryAwareHarness

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


def step_context(index, last_result=None, signals=()):
    return StepContext(
        instruction="tidy up",
        agent_uid=0,
        decision_index=index,
        max_decisions=10,
        world_description="kitchen_1: table_1\nbedroom_1: bed_1",
        tools=TOOLS,
        rooms=["kitchen_1", "bedroom_1"],
        last_result=last_result,
        signals=list(signals),
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


# ------------------------------------------------------------- LLM-free

def test_random_harness_explores_then_done():
    decisions = drive_bridge(RandomHarness(max_decisions_before_done=3, seed=0))
    assert [type(d).__name__ for d in decisions] == ["ToolCall"] * 3 + ["DoneDecision"]
    assert all(d.name == "Explore" and d.args in STATE["rooms"] for d in decisions[:-1])


def test_recovery_aware_blacklists_failing_room():
    harness = RecoveryAwareHarness(max_decisions_before_done=5, seed=0)
    harness.on_episode_start(episode_context())
    signal = Signal(kind="skill_failure_loop", message="", data={"args": "kitchen_1"})
    for index in range(4):
        decision = harness.decide(step_context(index, signals=[signal]))
        assert isinstance(decision, ToolCall)
        assert decision.args == "bedroom_1"
        assert "recovered from" in decision.rationale


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


# ---------------------------------------------------------------- ReAct

def _react(text):
    return f"Thought: {text}\nExplore[kitchen_1]\n{STOPWORD}"


def test_parse_react_response_variants():
    parsed = parse_react_response(_react("look around"))
    assert parsed == {"thought": "look around", "name": "Explore", "args": "kitchen_1", "action": "Explore[kitchen_1]"}
    assert parse_react_response(f"Thought: finished\nDone[]\n{STOPWORD}")["name"] == "Done"
    # Trailing chatter after the call: fall back to the last bracket call.
    assert parse_react_response("Thought: x\nNavigate[table_1]\nok then")["name"] == "Navigate"
    assert parse_react_response("Thought: only thinking, no action") is None
    assert parse_react_response("") is None


def test_react_transcript_accumulates_whole_episode_and_prompt_has_react_format():
    replies = iter([
        _react("first"),
        f"Thought: go\nNavigate[table_1]\n{STOPWORD}",
        f"Thought: finished\nDone[]\n{STOPWORD}",
    ])
    prompts = []

    def fake_chat(messages):
        prompts.append(messages)
        return next(replies)

    harness = ReActHarness(chat_fn=fake_chat)
    decisions = drive_bridge(harness, response="Successful execution! Explored kitchen_1")

    assert [type(d).__name__ for d in decisions] == ["ToolCall", "ToolCall", "DoneDecision"]
    assert (decisions[0].name, decisions[0].args, decisions[0].rationale) == ("Explore", "kitchen_1", "first")
    assert decisions[-1].rationale == "finished"
    # One LLM call per decision, system + user roles, ReAct format spec present.
    assert len(prompts) == 3
    assert [m["role"] for m in prompts[0]] == ["system", "user"]
    user0 = prompts[0][1]["content"]
    assert "Task: tidy up" in user0 and "Possible Actions:" in user0
    assert "- Explore: explore a room" in user0 and "Done[]" in user0
    assert f"Thought: <reasoning" in user0 and STOPWORD in user0
    assert "Previous actions" not in user0
    # Later prompts carry the FULL transcript with observations, oldest first.
    user2 = prompts[2][1]["content"]
    assert user2.index("Explore[kitchen_1]") < user2.index("Navigate[table_1]")
    assert "Agent_0_Observation:Successful execution! Explored kitchen_1" in user2
    assert harness.last_meta["transcript_len"] == 2
    assert harness.transcript[-1]["action"] == "Done[]"
    # A new episode starts from an empty transcript.
    harness.on_episode_start(episode_context())
    assert harness.transcript == []


def test_react_unknown_tool_then_valid_retries_once_with_reminder():
    replies = iter([f"Thought: hm\nFly[away]\n{STOPWORD}", _react("ok")])
    prompts = []

    def fake_chat(messages):
        prompts.append(messages)
        return next(replies)

    harness = ReActHarness(chat_fn=fake_chat)
    harness.on_episode_start(episode_context())
    decision = harness.decide(step_context(0))
    assert isinstance(decision, ToolCall) and decision.name == "Explore"
    assert decision.meta["parse_attempts"] == 2
    assert prompts[1][-1]["role"] == "user" and "'Fly' is not one of the Possible Actions" in prompts[1][-1]["content"]


def test_react_parse_failure_twice_declares_done():
    harness = ReActHarness(chat_fn=lambda messages: "I have no idea what to do")
    harness.on_episode_start(episode_context())
    decision = harness.decide(step_context(0))
    assert isinstance(decision, DoneDecision)
    assert decision.rationale == "parse failure"
    assert harness.transcript == []


def test_react_transport_error_is_llm_failure_not_decision():
    def boom(messages):
        raise ConnectionError("down")

    harness = ReActHarness(chat_fn=boom)
    harness.on_episode_start(episode_context())
    decision = harness.decide(step_context(0))
    assert isinstance(decision, ToolCall) and decision.error_kind == "llm_failure"
    assert "down" in decision.meta["llm_error"]


def test_react_partner_prompt_only_with_partner():
    harness = ReActHarness(chat_fn=lambda m: _react("x"))
    harness.on_episode_start(episode_context())
    assert "another agent" in harness.system_prompt()
    harness.on_episode_start(EpisodeContext("7", "tidy up", 0, partner_uids=[], tools=TOOLS))
    assert "another agent" not in harness.system_prompt()
