"""Tests for ``PartnrHarnessAdapter`` with the LLM transport mocked (no simulator)."""

import json

import pytest

from partnr_harness.contracts import (
    ActionResult,
    DoneDecision,
    EpisodeContext,
    EpisodeSummary,
    StepContext,
    ToolCall,
    ToolSpec,
)

from harness_vla.envs.partnr.adapter import PartnrHarnessAdapter
from harness_vla.envs.partnr.prompts import build_partnr_turn_prompt, step_context_to_dict

TOOLS = [ToolSpec("Navigate", "nav"), ToolSpec("Pick", "pick"), ToolSpec("Wait", "idle")]


def _ctx(decision_index=0):
    return StepContext(
        instruction="Move the cup to the table",
        agent_uid=0,
        decision_index=decision_index,
        max_decisions=10,
        world_description="table_1 in kitchen_1",
        tools=TOOLS,
        rooms=["kitchen_1"],
    )


def _mock_llm(monkeypatch, adapter, replies):
    """Feed ``replies`` in order; the last one repeats if the adapter asks again."""
    calls = []

    def fake_post(messages):
        calls.append(messages)
        return replies[min(len(calls) - 1, len(replies) - 1)]

    monkeypatch.setattr(adapter._llm, "_post", fake_post)
    return calls


@pytest.fixture
def adapter():
    return PartnrHarnessAdapter(max_retries=3)


def test_decide_returns_tool_call(monkeypatch, adapter):
    _mock_llm(monkeypatch, adapter, [json.dumps({"thought": "x", "action": "Navigate[table_1]"})])
    decision = adapter.decide(_ctx())
    assert isinstance(decision, ToolCall)
    assert (decision.name, decision.args) == ("Navigate", "table_1")
    assert decision.rationale == "x"
    assert decision.error_kind is None
    assert decision.meta["parse_attempts"] == 1
    assert adapter.history[-1]["action"] == "Navigate[table_1]"
    assert adapter.history[-1]["status"] is None


def test_decide_done(monkeypatch, adapter):
    _mock_llm(monkeypatch, adapter, [json.dumps({"thought": "finished", "action": "Done[]"})])
    decision = adapter.decide(_ctx())
    assert isinstance(decision, DoneDecision)
    assert adapter.history[-1]["status"] == "done"
    assert adapter.history[-1]["ok"] is True


def test_invalid_reply_after_retries_falls_back_to_wait(monkeypatch, adapter):
    calls = _mock_llm(monkeypatch, adapter, ["I do not know what to do"])
    decision = adapter.decide(_ctx())
    assert len(calls) == 4  # 1 attempt + max_retries
    assert isinstance(decision, ToolCall)
    assert decision.name == "Wait"
    assert decision.error_kind == "parse_error"
    assert decision.meta["parse_attempts"] == 4
    assert adapter.history[-1]["error_kind"] == "parse_error"


def test_on_action_result_fills_last_record(monkeypatch, adapter):
    _mock_llm(monkeypatch, adapter, [json.dumps({"thought": "x", "action": "Navigate[table_1]"})])
    adapter.decide(_ctx())
    adapter.on_action_result(
        ActionResult(tool="Navigate", args="table_1", response="Node table_1 not found", success_hint=False)
    )
    record = adapter.history[-1]
    assert record["status"] == "failure"
    assert record["ok"] is False
    assert record["message"] == "Node table_1 not found"
    assert record["failure_kind"] == "grounding"

    _mock_llm(monkeypatch, adapter, [json.dumps({"thought": "x", "action": "Pick[cup_0]"})])
    adapter.decide(_ctx(1))
    adapter.on_action_result(
        ActionResult(tool="Pick", args="cup_0", response="Successful execution!", success_hint=True)
    )
    assert adapter.history[-1]["status"] == "success"
    assert adapter.history[-1]["ok"] is True
    assert adapter.history[0]["status"] == "failure"  # earlier record untouched


def test_full_history_in_prompt_with_max_history_zero(monkeypatch, adapter):
    replies = [json.dumps({"thought": "x", "action": f"Navigate[room_{i}]"}) for i in range(5)]
    for i, reply in enumerate(replies):
        _mock_llm(monkeypatch, adapter, [reply])
        adapter.decide(_ctx(i))
        adapter.on_action_result(
            ActionResult(tool="Navigate", args=f"room_{i}", response="Successful execution!", success_hint=True)
        )
    prompt = build_partnr_turn_prompt(step_context_to_dict(_ctx(5)), adapter.history, None, max_history=0)
    for i in range(5):
        assert f"{i + 1}. Navigate[room_{i}] -> success" in prompt
    assert "omitted" not in prompt
    truncated = build_partnr_turn_prompt(step_context_to_dict(_ctx(5)), adapter.history, None, max_history=2)
    assert "Navigate[room_0]" not in truncated
    assert "[3 earlier steps omitted]" in truncated
    # The prompt sent on turn 5 (index 4) carried the full transcript of turns 1-4.
    assert "4. Navigate[room_3] -> success" in adapter.last_prompt
    assert "Navigate[room_4]" not in adapter.last_prompt


# ------------------------------------------------------ bootstrap label

def _run_success_episode(monkeypatch, adapter):
    adapter.on_episode_start(EpisodeContext("11", "Move the cup to the table", 0, tools=TOOLS))
    for i, action in enumerate(["Navigate[table]", "Pick[cup]"]):
        _mock_llm(monkeypatch, adapter, [json.dumps({"thought": "x", "action": action})])
        adapter.decide(_ctx(i))
        adapter.on_action_result(
            ActionResult(tool=action.split("[")[0], args="", response="Successful execution!", success_hint=True)
        )


def _summary(**extras):
    return EpisodeSummary(episode_id="11", num_decisions=2, emitted_done=True, extras=extras)


def test_bootstrap_promotes_only_with_env_label_and_success(monkeypatch, tmp_path):
    adapter = PartnrHarnessAdapter(zero_shot=False, task_memory_dir=str(tmp_path), phase="bootstrap")
    _run_success_episode(monkeypatch, adapter)
    # No label published by the env (expose_bootstrap_label off) -> nothing written.
    adapter.on_episode_end(_summary(decisions_total=2))
    assert list(tmp_path.glob("*.json")) == []
    # Legacy key alone is no longer a channel.
    adapter.on_episode_end(_summary(task_state_success=1.0))
    assert list(tmp_path.glob("*.json")) == []
    # Explicit label, failure -> nothing.
    adapter.on_episode_end(_summary(bootstrap_label={"task_state_success": 0.0, "source": "official_stats"}))
    assert list(tmp_path.glob("*.json")) == []
    # Explicit label, success -> task memory written with provenance.
    adapter.on_episode_end(_summary(bootstrap_label={"task_state_success": 1.0, "source": "official_stats"}))
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    entry = json.loads(files[0].read_text())
    assert entry["sources"][0]["label_source"] == "official_stats"
    assert entry["sources"][0]["episode_id"] == "11"


def test_deployment_ignores_bootstrap_label(monkeypatch, tmp_path):
    adapter = PartnrHarnessAdapter(zero_shot=False, task_memory_dir=str(tmp_path), phase="deployment")
    _run_success_episode(monkeypatch, adapter)
    adapter.on_episode_end(_summary(bootstrap_label={"task_state_success": 1.0, "source": "official_stats"}))
    assert list(tmp_path.glob("*.json")) == []
