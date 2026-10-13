"""Tests for the Harness VLA controllers (centralized + decentralized), no env/LLM needed."""

import json

import pytest

from akc_harness.contracts import (
    ActionResult,
    DoneDecision,
    EpisodeContext,
    EpisodeSummary,
    StepContext,
    ToolCall,
    ToolSpec,
)

from harness_vla.envs.partnr.agent import (
    HarnessVlaCentralized,
    labeled_skeleton,
    promote_labeled_transcript,
    resolve_labeled_task_memory,
)
from harness_vla.envs.partnr.decentralized import (
    HarnessVlaDecentralized,
    read_published_transcript,
)

TOOLS = [
    ToolSpec("Navigate", "Navigate[target]"),
    ToolSpec("Pick", "Pick[object]"),
    ToolSpec("Place", "Place[object, relation, furniture, constraint, reference]"),
    ToolSpec("Wait", "Wait[]"),
]
STATE = {
    "rooms": ["kitchen_1", "living_room_1"],
    "furniture": {"table_1": {"room": "kitchen_1"}, "sofa_3": {"room": "living_room_1"}},
    "objects": {"apple_0": {"room": "kitchen_1", "relation": "on", "furniture": "table_1", "held_by": None}},
    "agents": {"0": {"room": "kitchen_1", "holding": None}, "1": {"room": "living_room_1", "holding": None}},
}


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.last_meta = {}

    def start_episode(self, agent_uid=None):
        pass

    def post(self, messages):
        self.calls.append(messages)
        reply = self.replies.pop(0) if self.replies else '{"thought": "x", "agent_0": null}'
        if reply is None:
            self.last_meta = {"llm_failure": True, "llm_error": "boom"}
            return ""
        self.last_meta = {"prompt_tokens": 10, "completion_tokens": 5}
        return reply


def central_episode(**extras):
    tools = [{"name": t.name, "description": t.description} for t in TOOLS]
    return EpisodeContext(
        episode_id="ep4", instruction="Put the apple on the sofa", agent_uid=0, partner_uids=[1],
        tools=TOOLS,
        extras={"centralized": True, "agents": {"agent_0": {"uid": 0, "tools": tools},
                                                "agent_1": {"uid": 1, "tools": tools}}, **extras},
    )


def central_step(deciding=("agent_0", "agent_1"), busy=(), index=0):
    return StepContext(
        instruction="Put the apple on the sofa", agent_uid=0, decision_index=index, max_decisions=100,
        world_description="", tools=TOOLS, rooms=STATE["rooms"],
        extras={"centralized": True, "deciding_agents": list(deciding), "busy_agents": list(busy),
                "in_flight_actions": {label: "Navigate[table_1]" for label in busy},
                "tools_by_agent": {}, "state": STATE},
    )


def make_central(replies, **kwargs):
    agent = HarnessVlaCentralized(use_claims=False, **kwargs)
    agent._llm = FakeLLM(replies)
    return agent


def test_central_decides_for_deciding_agents_only_and_records_team_transcript():
    agent = make_central(['{"thought": "split", "agent_0": "Pick[apple_0]", "agent_1": "Navigate[sofa_3]"}'])
    agent.on_episode_start(central_episode())
    decisions = agent.decide(central_step(deciding=("agent_0",)))
    assert set(decisions) == {"agent_0"}
    assert isinstance(decisions["agent_0"], ToolCall) and decisions["agent_0"].name == "Pick"
    assert [r["agent"] for r in agent.history] == ["agent_0"]
    system = agent._llm.calls[0][0]["content"]
    assert '"agent_0"' in system and '"all"' in system
    user = agent._llm.calls[0][1]["content"]
    assert "Agents awaiting your decision this turn: agent_0" in user
    assert "agent_0, Spot robot" in user


def test_central_all_done_and_per_agent_done_latches_claims():
    agent = HarnessVlaCentralized(use_claims=True)
    agent._llm = FakeLLM([
        '{"subgoals": [{"id": "a", "description": "carry apple", "assigned_to": "agent_0"}]}',
        '{"thought": "done", "agent_0": "Done[]", "agent_1": null}',
        '{"thought": "done", "all": "Done[]"}',
    ])
    agent.on_episode_start(central_episode())
    assert set(agent.claims()) == {"agent_0", "agent_1"}
    assert agent.claims()["agent_1"]["description"] == "assist / standby"
    decisions = agent.decide(central_step())
    assert isinstance(decisions["agent_0"], DoneDecision) and decisions["agent_1"] is None
    assert agent.claims()["agent_0"]["done"] and not agent.all_claims_done()
    decisions = agent.decide(central_step(index=1))
    assert isinstance(decisions["all"], DoneDecision)
    user = agent._llm.calls[1][1]["content"]
    assert "[TODO] carry apple" in user


def test_central_busy_agents_shown_and_results_attributed_by_uid():
    agent = make_central([
        '{"thought": "", "agent_0": "Pick[apple_0]", "agent_1": "Navigate[sofa_3]"}',
        '{"thought": "", "agent_1": "Wait[]"}',
    ])
    agent.on_episode_start(central_episode())
    agent.decide(central_step())
    agent.on_action_result(ActionResult("Navigate", "sofa_3", "Successful execution!", True, 0, agent_uid=1))
    agent.on_action_result(ActionResult("Pick", "apple_0", "Unexpected failure!", False, 0, agent_uid=0))
    by_agent = {r["agent"]: r for r in agent.history}
    assert by_agent["agent_1"]["status"] == "success"
    assert by_agent["agent_0"]["status"] == "failure"
    agent.decide(central_step(deciding=("agent_1",), busy=("agent_0",), index=1))
    user = agent._llm.calls[1][1]["content"]
    assert "agent_0: running Navigate[table_1]" in user
    assert "1. agent_0: Pick[apple_0] -> failure" in user
    assert "1. agent_1: Navigate[sofa_3] -> success" in user


def test_central_parse_failure_retries_then_waits():
    agent = make_central(["garbage", "still garbage", "{}", "nope"], max_retries=3)
    agent.on_episode_start(central_episode())
    decisions = agent.decide(central_step())
    assert all(d.name == "Wait" and d.error_kind == "parse_error" for d in decisions.values())
    assert len(agent._llm.calls) == 4
    assert agent.last_meta["parse_attempts"] == 4


def test_central_unknown_tool_and_llm_failure():
    agent = make_central(['{"agent_0": "Fly[apple_0]"}', None])
    agent.on_episode_start(central_episode())
    decisions = agent.decide(central_step(deciding=("agent_0",)))
    assert decisions["agent_0"].name == "InvalidResponse"
    assert decisions["agent_0"].error_kind == "compile_error"
    decisions = agent.decide(central_step(deciding=("agent_0", "agent_1"), index=1))
    assert all(d.error_kind == "llm_failure" for d in decisions.values())


def test_central_place_padded_and_dict_values_accepted():
    agent = make_central(['{"agent_0": {"action": "Place", "args": "apple_0, on, sofa_3"}}'])
    agent.on_episode_start(central_episode())
    decision = agent.decide(central_step(deciding=("agent_0",)))["agent_0"]
    assert decision.name == "Place" and decision.args == "apple_0,on,sofa_3,None,None"


def test_labeled_skeleton_promotion_and_resolution(tmp_path):
    agent = make_central(['{"agent_0": "Pick[apple_0]", "agent_1": "Navigate[sofa_3]"}',
                          '{"agent_0": "Place[apple_0, on, sofa_3]"}'],
                         phase="bootstrap", task_memory_dir=str(tmp_path))
    agent.on_episode_start(central_episode())
    agent.decide(central_step())
    agent.on_action_result(ActionResult("Pick", "apple_0", "ok", True, 0, agent_uid=0))
    agent.on_action_result(ActionResult("Navigate", "sofa_3", "ok", True, 0, agent_uid=1))
    agent.decide(central_step(deciding=("agent_0",), index=1))
    agent.on_action_result(ActionResult("Place", "apple_0,on,sofa_3,None,None", "ok", True, 1, agent_uid=0))
    assert labeled_skeleton(agent.history) == [
        "agent_0: Pick[<obj:apple>]", "agent_1: Navigate[<furn:sofa>]",
        "agent_0: Place[<obj:apple>, on, <furn:sofa>, None, None]",
    ]
    summary = EpisodeSummary("ep4", 2, True, extras={"bootstrap_label": {"task_success": True, "source": "official"}})
    agent.on_episode_end(summary)
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    entry = json.loads(files[0].read_text())
    resolved = resolve_labeled_task_memory(entry, STATE)
    assert resolved[0] == "agent_0: Pick[apple_0]"
    assert resolved[1] == "agent_1: Navigate[sofa_3]"
    # deployment: prior shows up in the prompt with labels
    deploy = make_central(['{"agent_0": "Pick[apple_0]"}'], zero_shot=False, task_memory_dir=str(tmp_path))
    deploy.on_episode_start(central_episode())
    deploy.decide(central_step(deciding=("agent_0",)))
    assert "1. agent_0: Pick[apple_0]" in deploy._llm.calls[0][1]["content"]
    assert deploy.last_meta["task_memory_used"] is True


def test_deployment_never_promotes(tmp_path):
    agent = make_central(['{"agent_0": "Pick[apple_0]"}'], task_memory_dir=str(tmp_path))
    agent.on_episode_start(central_episode())
    agent.decide(central_step(deciding=("agent_0",)))
    agent.on_action_result(ActionResult("Pick", "apple_0", "ok", True, 0, agent_uid=0))
    agent.on_episode_end(EpisodeSummary("ep4", 1, True, extras={"bootstrap_label": {"task_success": True}}))
    assert list(tmp_path.glob("*.json")) == []


# --------------------------------------------------------------- decentral

def decentral_episode(uid):
    return EpisodeContext(episode_id="ep4", instruction="Put the apple on the sofa", agent_uid=uid,
                          partner_uids=[1 - uid], tools=TOOLS, extras={"state": STATE})


def decentral_step(uid, index=0):
    return StepContext(instruction="Put the apple on the sofa", agent_uid=uid, decision_index=index,
                       max_decisions=50, world_description="", tools=TOOLS, rooms=STATE["rooms"],
                       extras={"state": STATE})


def make_decentral(replies, shared, **kwargs):
    agent = HarnessVlaDecentralized(shared_memory_dir=str(shared), **kwargs)
    agent._llm = FakeLLM(replies)
    return agent


def test_decentral_publishes_actions_without_thought_and_reads_partner(tmp_path):
    robot = make_decentral(['{"thought": "SECRET", "agent_0": "Pick[apple_0]"}'], tmp_path)
    human = make_decentral(['{"thought": "mine", "agent_1": "Navigate[sofa_3]"}'], tmp_path)
    robot.on_episode_start(decentral_episode(0))
    human.on_episode_start(decentral_episode(1))
    assert robot.controlled == ["agent_0"] and robot.partner == "agent_1"
    assert "You are agent_0, a Spot robot" in robot.role and "agent_1 (a human)" in robot.role
    assert '"all"' not in robot.system_prompt

    decision = robot.decide(decentral_step(0))
    assert isinstance(decision, ToolCall) and decision.name == "Pick"
    published = (tmp_path / "ep4" / "agent_0.jsonl").read_text()
    assert "SECRET" not in published and "Pick[apple_0]" in published
    robot.on_action_result(ActionResult("Pick", "apple_0", "Successful execution!", True, 0))
    partner_view = read_published_transcript(tmp_path / "ep4" / "agent_0.jsonl")
    assert len(partner_view) == 1 and partner_view[0]["status"] == "success"

    human.decide(decentral_step(1))
    user = human._llm.calls[0][1]["content"]
    assert "Partner's transcript this episode" in user
    assert "agent_0: Pick[apple_0] -> success" in user
    assert "Transcript of your actions" in user or "No actions executed yet" in user
    assert human.last_meta["partner_transcript_len"] == 1


def test_decentral_single_agent_shape_and_done(tmp_path):
    agent = make_decentral(['{"thought": "", "action": "Navigate[sofa_3]"}', '{"agent_0": "Done[]"}'], tmp_path)
    agent.on_episode_start(decentral_episode(0))
    assert agent.decide(decentral_step(0)).name == "Navigate"
    assert isinstance(agent.decide(decentral_step(0, 1)), DoneDecision)


def test_decentral_idle_becomes_wait(tmp_path):
    agent = make_decentral(['{"thought": "", "agent_0": null}'], tmp_path)
    agent.on_episode_start(decentral_episode(0))
    decision = agent.decide(decentral_step(0))
    assert decision.name == "Wait" and decision.error_kind is None


def test_decentral_memory_dirs_are_per_label_and_partner_plan_shown(tmp_path):
    memory_root = tmp_path / "task_memory"
    (memory_root / "agent_1").mkdir(parents=True)
    promote_labeled_transcript(
        str(memory_root / "agent_1"), "Put the apple on the sofa",
        [{"agent": "agent_1", "name": "Navigate", "args": "sofa_3", "action": "Navigate[sofa_3]",
          "ok": True, "error_kind": None, "vocab": {"furniture": {"sofa_3"}}}],
    )
    robot = make_decentral(['{"agent_0": "Pick[apple_0]"}'], tmp_path, zero_shot=False,
                           task_memory_dir=str(memory_root), global_memory_path=str(tmp_path / "gm"))
    robot.on_episode_start(decentral_episode(0))
    assert robot._own_task_memory_dir() == str(memory_root / "agent_0")
    assert robot._own_global_memory_path() == str(tmp_path / "gm" / "agent_0" / "global_memory.json")
    robot.decide(decentral_step(0))
    user = robot._llm.calls[0][1]["content"]
    assert "Partner's remembered plan" in user and "agent_1: Navigate[sofa_3]" in user
    assert robot.last_meta["partner_task_memory_used"] is True
    assert robot.last_meta["task_memory_used"] is False


def test_read_published_transcript_latest_line_wins_and_skips_partial(tmp_path):
    path = tmp_path / "agent_1.jsonl"
    path.write_text(
        json.dumps({"decision_index": 0, "agent": "agent_1", "action": "Pick[a_0]", "status": None}) + "\n"
        + json.dumps({"decision_index": 0, "agent": "agent_1", "action": "Pick[a_0]", "status": "failure"}) + "\n"
        + '{"decision_index": 1, "agent": "agent_1", "acti'
    )
    records = read_published_transcript(path)
    assert len(records) == 1 and records[0]["status"] == "failure"
