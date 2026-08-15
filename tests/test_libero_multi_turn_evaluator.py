import json

import pytest

from embodiedbench.evaluator.libero_analytic_executor import (
    LiberoPrimitiveExecution,
)
from embodiedbench.evaluator.libero_multi_turn_evaluator import (
    LiberoMultiTurnBudgets,
    LiberoMultiTurnEvaluator,
    LiberoVLAExecution,
)
from embodiedbench.planner.harness.trace_io import load_complete_jsonl
from embodiedbench.planner.harness.task_memory import build_task_memory


TARGETS = ["akita_black_bowl_1", "plate_1"]


def _budgets(**overrides):
    values = {
        "max_turns": 6,
        "horizon": 20,
        "max_chunks_cap": 4,
        "max_move_steps": 5,
        "release_steps": 3,
    }
    values.update(overrides)
    return LiberoMultiTurnBudgets(**values)


def _execution(
    frame,
    *,
    primitive_success=True,
    task_success=False,
    reason="postcondition_met",
    steps=1,
):
    return LiberoPrimitiveExecution(
        observation={
            "frame": frame,
            "robot0_eef_pos": [float(frame), 0.0, 0.5],
        },
        primitive_success=primitive_success,
        task_success=task_success,
        termination_reason=reason,
        steps_executed=steps,
        trace=[{"frame": frame, "physical_pose": [float(frame), 0.0, 0.5]}],
    )


class _ScriptedPlanner:
    last_thinking = "scripted thinking"

    def __init__(self, invocations):
        self.invocations = list(invocations)
        self.calls = []

    def act_turn(
        self,
        instruction,
        state,
        *,
        available_targets,
        max_chunks_cap,
    ):
        invocation = self.invocations[len(self.calls)]
        self.calls.append(
            {
                "instruction": instruction,
                "state": state,
                "available_targets": list(available_targets),
                "max_chunks_cap": max_chunks_cap,
                "invocation": invocation,
            }
        )
        if invocation is None:
            return None, "not json"
        return invocation, json.dumps(invocation)


class _FakeExecutors:
    def __init__(self, *, release_task_success=True):
        self.calls = []
        self.release_task_success = release_task_success

    def vla(self, invocation, observation, *, max_steps):
        self.calls.append(("vla_act", invocation, observation, max_steps))
        return LiberoVLAExecution(
            execution=_execution(
                len(self.calls), reason="lift_and_grasp_satisfied", steps=2
            ),
            tau_satisfied=True,
            holding=invocation["target"],
        )

    def move(self, invocation, observation, *, max_steps):
        self.calls.append((invocation["action"], invocation, observation, max_steps))
        return _execution(len(self.calls))

    def release(self, invocation, observation, *, max_steps):
        self.calls.append(("release", invocation, observation, max_steps))
        return _execution(
            len(self.calls),
            task_success=self.release_task_success,
            reason=(
                "task_success"
                if self.release_task_success
                else "release_completed_task_incomplete"
            ),
        )


def _evaluator(tmp_path, planner, executors):
    trace_path = tmp_path / "turns.jsonl"
    evaluator = LiberoMultiTurnEvaluator(
        planner,
        vla_executor=executors.vla,
        move_executor=executors.move,
        release_executor=executors.release,
        trace_path=trace_path,
    )
    return evaluator, trace_path


def _happy_script():
    return [
        {
            "action": "vla_act",
            "prompt": "pick up and lift the bowl",
            "target": "akita_black_bowl_1",
            "max_chunks": 2,
            "tau": "lift_and_grasp",
        },
        {
            "action": "move_to",
            "target": "plate_1",
            "mode": "above",
            "gripper": "close",
        },
        {
            "action": "move_to",
            "target": "plate_1",
            "mode": "release_pose",
            "gripper": "close",
        },
        {"action": "release"},
    ]


def test_scripted_planner_flow_reaches_official_task_success(tmp_path):
    script = _happy_script()
    planner = _ScriptedPlanner(script)
    executors = _FakeExecutors()
    evaluator, _ = _evaluator(tmp_path, planner, executors)

    result = evaluator.run(
        "place the bowl on the plate",
        {"frame": 0, "robot0_eef_pos": [0.0, 0.0, 0.5]},
        available_targets=TARGETS,
        budgets=_budgets(),
    )

    assert result.task_success is True
    assert result.primitive_success is True
    assert result.env_done is False
    assert result.termination_reason == "task_success"
    assert result.turns_executed == 4
    assert result.steps_executed == 5
    assert result.holding is None
    assert [call[0] for call in executors.calls] == [
        "vla_act",
        "move_to",
        "move_to",
        "release",
    ]
    assert all(
        executor_call[1] is planner_call["invocation"]
        for executor_call, planner_call in zip(executors.calls, planner.calls)
    )


def test_all_seven_published_libero_primitives_dispatch_one_per_turn(tmp_path):
    script = [
        {"action": "set_gripper", "gripper": "open"},
        {
            "action": "move_pose",
            "xyz": [0.1, -0.2, 1.05],
            "pose": [0.0, 0.0, 0.0, 1.0],
            "gripper": "open",
        },
        {"action": "rotate_wrist", "target_yaw": 1.0},
        {"action": "rotate_pitch", "target_pitch": -0.5},
        _happy_script()[0],
        _happy_script()[1],
        {"action": "release"},
    ]
    planner = _ScriptedPlanner(script)
    executors = _FakeExecutors()
    evaluator, trace_path = _evaluator(tmp_path, planner, executors)

    result = evaluator.run(
        "place the bowl",
        {"frame": 0},
        available_targets=TARGETS,
        budgets=_budgets(max_turns=7),
    )

    expected = [
        "set_gripper",
        "move_pose",
        "rotate_wrist",
        "rotate_pitch",
        "vla_act",
        "move_to",
        "release",
    ]
    assert result.task_success is True
    assert result.turns_executed == 7
    assert [call[0] for call in executors.calls] == expected
    assert [record["primitive"] for record in load_complete_jsonl(trace_path)] == expected


@pytest.mark.parametrize(
    "invocation",
    [
        {"action": "move_pose", "xyz": [9, 0, 0.5], "pose": [0, 0, 0, 1], "gripper": "open"},
        {"action": "rotate_wrist", "target_yaw": 0.0, "extra": True},
        {"action": "rotate_pitch", "target_pitch": 2.0},
        {"action": "set_gripper", "gripper": "hold"},
    ],
)
def test_new_analytic_guards_fail_closed_before_dispatch(tmp_path, invocation):
    planner = _ScriptedPlanner([invocation])
    executors = _FakeExecutors()
    evaluator, _ = _evaluator(tmp_path, planner, executors)

    result = evaluator.run(
        "stage the gripper",
        {"frame": 0},
        available_targets=TARGETS,
        budgets=_budgets(max_turns=1),
    )

    assert result.termination_reason == "primitive_compile_error"
    assert executors.calls == []


def test_sequence_is_selected_by_planner_without_evaluator_state_machine(tmp_path):
    script = [
        _happy_script()[0],
        _happy_script()[2],
        _happy_script()[1],
        _happy_script()[3],
    ]
    planner = _ScriptedPlanner(script)
    executors = _FakeExecutors()
    evaluator, _ = _evaluator(tmp_path, planner, executors)

    result = evaluator.run(
        "place the bowl",
        {"frame": 0},
        available_targets=TARGETS,
        budgets=_budgets(),
    )

    assert result.task_success is True
    assert [call[1]["mode"] for call in executors.calls[1:3]] == [
        "release_pose",
        "above",
    ]


def test_previous_feedback_to_planner_has_no_physical_pose_leakage(tmp_path):
    planner = _ScriptedPlanner(_happy_script())
    executors = _FakeExecutors()
    evaluator, _ = _evaluator(tmp_path, planner, executors)

    evaluator.run(
        "place the bowl",
        {"robot0_eef_pos": [0.1, 0.2, 0.3], "contact": "oracle"},
        available_targets=TARGETS,
        budgets=_budgets(),
    )

    second_state = planner.calls[1]["state"]
    encoded_state = json.dumps(second_state)
    assert second_state["holding"] == "akita_black_bowl_1"
    assert second_state["last_feedback"]["tau_satisfied"] is True
    assert second_state["last_feedback"]["target"] == "akita_black_bowl_1"
    assert "robot0_eef_pos" not in encoded_state
    assert "physical_pose" not in encoded_state
    assert "contact" not in encoded_state


def test_privileged_diagnostics_persist_only_in_trace(tmp_path):
    planner = _ScriptedPlanner(_happy_script())
    executors = _FakeExecutors()
    events = []

    def diagnose(event, invocation, holding):
        events.append(event)
        return {
            "event": event,
            "source": "privileged_mujoco_state",
            "beta_only": True,
            "holding": holding,
            "contact": {"bilateral": True},
        }

    trace_path = tmp_path / "turns.jsonl"
    evaluator = LiberoMultiTurnEvaluator(
        planner,
        vla_executor=executors.vla,
        move_executor=executors.move,
        release_executor=executors.release,
        trace_path=trace_path,
        diagnostics_callback=diagnose,
    )

    evaluator.run(
        "place the bowl on the plate",
        {"frame": 0},
        available_targets=TARGETS,
        budgets=_budgets(),
    )

    records = load_complete_jsonl(trace_path)
    assert events == ["post_grasp", "pre_release", "post_release"]
    assert records[0]["privileged_diagnostics"][0]["event"] == "post_grasp"
    assert [item["event"] for item in records[-1]["privileged_diagnostics"]] == [
        "pre_release",
        "post_release",
    ]
    assert all("privileged_diagnostics" not in call["state"] for call in planner.calls)
    assert all(
        "privileged" not in json.dumps(call["state"].get("last_feedback"))
        for call in planner.calls
    )


def test_release_without_holding_is_blocked_without_executor_call(tmp_path):
    planner = _ScriptedPlanner([{"action": "release"}])
    executors = _FakeExecutors()
    evaluator, trace_path = _evaluator(tmp_path, planner, executors)

    result = evaluator.run(
        "place the bowl",
        {"frame": 0},
        available_targets=TARGETS,
        budgets=_budgets(max_turns=1),
    )

    assert result.termination_reason == "release_without_holding"
    assert result.steps_executed == 0
    assert executors.calls == []
    assert load_complete_jsonl(trace_path)[0]["feedback"]["steps_executed"] == 0


@pytest.mark.parametrize("holding", ["plate_1"])
def test_task_success_vla_requires_holding_target(tmp_path, holding):
    invocation = {
        "action": "vla_act",
        "prompt": "place the bowl into plate_1",
        "target": "akita_black_bowl_1",
        "max_chunks": 1,
        "tau": "task_success",
    }
    planner = _ScriptedPlanner([invocation])
    executors = _FakeExecutors()
    evaluator, _ = _evaluator(tmp_path, planner, executors)

    result = evaluator.run(
        "place the bowl",
        {"frame": 0},
        available_targets=TARGETS,
        budgets=_budgets(max_turns=1),
        holding=holding,
    )

    assert result.termination_reason == "holding_incompatible"
    assert executors.calls == []


def test_task_success_vla_with_empty_hand_reaches_executor(tmp_path):
    invocation = {
        "action": "vla_act",
        "prompt": "open the middle drawer of the cabinet",
        "target": "plate_1",
        "max_chunks": 1,
        "tau": "task_success",
    }
    planner = _ScriptedPlanner([invocation])
    executors = _FakeExecutors()
    evaluator, _ = _evaluator(tmp_path, planner, executors)

    evaluator.run(
        "open the drawer",
        {"frame": 0},
        available_targets=TARGETS,
        budgets=_budgets(max_turns=1),
        holding=None,
    )

    assert [call[0] for call in executors.calls] == ["vla_act"]


def test_guard_violation_is_recoverable_with_feedback(tmp_path):
    bad = {
        "action": "vla_act",
        "prompt": "place the bowl into plate_1",
        "target": "plate_1",
        "max_chunks": 1,
        "tau": "task_success",
    }
    good = {"action": "release"}
    planner = _ScriptedPlanner([bad, good])
    executors = _FakeExecutors()
    evaluator, trace_path = _evaluator(tmp_path, planner, executors)

    result = evaluator.run(
        "place the bowl",
        {"frame": 0},
        available_targets=TARGETS,
        budgets=_budgets(),
        holding="akita_black_bowl_1",
    )

    assert result.termination_reason == "task_success"
    assert result.turns_executed == 2
    assert [call[0] for call in executors.calls] == ["release"]
    records = load_complete_jsonl(trace_path)
    assert records[0]["feedback"]["termination_reason"] == "holding_incompatible"
    assert records[0]["feedback"]["steps_executed"] == 0
    second_state = planner.calls[1]["state"]
    assert (
        second_state["last_feedback"]["termination_reason"]
        == "holding_incompatible"
    )


@pytest.mark.parametrize(
    "invocation",
    [
        {
            "action": "move_to",
            "target": "plate_1",
            "mode": "above",
            "gripper": "close",
        },
        {
            "action": "move_to",
            "target": "plate_1",
            "mode": "above",
            "gripper": "open",
        },
    ],
)
def test_move_to_with_incompatible_holding_is_blocked(tmp_path, invocation):
    planner = _ScriptedPlanner([invocation])
    executors = _FakeExecutors()
    evaluator, _ = _evaluator(tmp_path, planner, executors)

    result = evaluator.run(
        "place the bowl",
        {"frame": 0},
        available_targets=TARGETS,
        budgets=_budgets(max_turns=1),
        holding=None,
    )

    assert result.termination_reason == "grasp_lost"
    assert result.steps_executed == 0
    assert executors.calls == []


def test_physical_grasp_loss_clears_holding_before_replanning(tmp_path):
    planner = _ScriptedPlanner(
        [
            {
                "action": "move_to",
                "target": "plate_1",
                "mode": "release_pose",
                "gripper": "close",
            },
            {
                "action": "vla_act",
                "prompt": "recover the bowl",
                "target": "akita_black_bowl_1",
                "max_chunks": 1,
                "tau": "lift_and_grasp",
            },
        ]
    )
    executors = _FakeExecutors()

    def lost_move(invocation, observation, *, max_steps):
        executors.calls.append(("move_to", invocation, observation, max_steps))
        return _execution(
            1,
            primitive_success=False,
            reason="grasp_lost",
        )

    evaluator = LiberoMultiTurnEvaluator(
        planner,
        vla_executor=executors.vla,
        move_executor=lost_move,
        release_executor=executors.release,
        trace_path=tmp_path / "turns.jsonl",
    )

    evaluator.run(
        "place the bowl",
        {"frame": 0},
        available_targets=TARGETS,
        budgets=_budgets(max_turns=2),
        holding="akita_black_bowl_1",
    )

    assert planner.calls[1]["state"]["holding"] is None
    assert planner.calls[1]["state"]["last_feedback"]["holding"] is None


def test_release_task_incomplete_can_continue_when_budget_remains(tmp_path):
    planner = _ScriptedPlanner(
        [
            {"action": "release"},
            {
                "action": "vla_act",
                "prompt": "recover the bowl",
                "target": "akita_black_bowl_1",
                "max_chunks": 1,
                "tau": "lift_and_grasp",
            },
        ]
    )
    executors = _FakeExecutors(release_task_success=False)

    def successful_vla(invocation, observation, *, max_steps):
        executors.calls.append(("vla_act", invocation, observation, max_steps))
        return LiberoVLAExecution(
            _execution(2, task_success=True, reason="task_success"),
            tau_satisfied=True,
            holding=invocation["target"],
        )

    trace_path = tmp_path / "turns.jsonl"
    evaluator = LiberoMultiTurnEvaluator(
        planner,
        vla_executor=successful_vla,
        move_executor=executors.move,
        release_executor=executors.release,
        trace_path=trace_path,
    )

    result = evaluator.run(
        "place the bowl",
        {"frame": 0},
        available_targets=TARGETS,
        budgets=_budgets(),
        holding="akita_black_bowl_1",
    )

    assert result.task_success is True
    assert result.turns_executed == 2
    records = load_complete_jsonl(trace_path)
    assert records[0]["feedback"]["recoverable"] is True
    assert records[0]["feedback"]["holding"] is None


def test_max_turns_stops_before_an_additional_planner_call(tmp_path):
    planner = _ScriptedPlanner(_happy_script())
    executors = _FakeExecutors()
    evaluator, _ = _evaluator(tmp_path, planner, executors)

    result = evaluator.run(
        "place the bowl",
        {"frame": 0},
        available_targets=TARGETS,
        budgets=_budgets(max_turns=2),
    )

    assert result.termination_reason == "max_turns_exhausted"
    assert result.turns_executed == 2
    assert len(planner.calls) == 2
    assert len(executors.calls) == 2


def test_horizon_is_passed_as_hard_cap_and_never_exceeded(tmp_path):
    planner = _ScriptedPlanner(_happy_script())
    calls = []

    def vla(invocation, observation, *, max_steps):
        calls.append((invocation, max_steps))
        return LiberoVLAExecution(
            _execution(1, reason="lift_and_grasp_satisfied", steps=max_steps),
            tau_satisfied=True,
            holding=invocation["target"],
        )

    def move(invocation, observation, *, max_steps):
        calls.append((invocation, max_steps))
        return _execution(2, steps=max_steps)

    evaluator = LiberoMultiTurnEvaluator(
        planner,
        vla_executor=vla,
        move_executor=move,
        release_executor=move,
        trace_path=tmp_path / "turns.jsonl",
    )

    result = evaluator.run(
        "place the bowl",
        {"frame": 0},
        available_targets=TARGETS,
        budgets=_budgets(horizon=3),
    )

    assert result.termination_reason == "horizon_exhausted"
    assert result.steps_executed == 3
    assert [max_steps for _, max_steps in calls] == [3]
    assert len(planner.calls) == 1


def test_planner_parse_error_is_traced_without_dispatch(tmp_path):
    planner = _ScriptedPlanner([None])
    executors = _FakeExecutors()
    evaluator, trace_path = _evaluator(tmp_path, planner, executors)

    result = evaluator.run(
        "place the bowl",
        {"frame": 0},
        available_targets=TARGETS,
        budgets=_budgets(max_turns=1),
    )

    assert result.termination_reason == "planner_parse_error"
    assert executors.calls == []
    records = load_complete_jsonl(trace_path)
    assert len(records) == 1
    record = records[0]
    assert record["turn"] == 1
    assert record["planner_raw_output"] == "not json"
    assert record["planner_thinking"] == "scripted thinking"
    assert record["invocation"] is None
    assert record["primitive"] is None
    assert record["primitive_trace"] == []
    assert record["feedback"] == {
        "action": None,
        "primitive_success": False,
        "task_success": False,
        "env_done": False,
        "termination_reason": "planner_parse_error",
        "steps_executed": 0,
        "holding": None,
        "recoverable": False,
    }


def test_trace_has_one_reconstructable_record_per_turn(tmp_path):
    planner = _ScriptedPlanner(_happy_script())
    executors = _FakeExecutors()
    evaluator, trace_path = _evaluator(tmp_path, planner, executors)

    evaluator.run(
        "place the bowl",
        {"frame": 0},
        available_targets=TARGETS,
        budgets=_budgets(),
        object_labels={
            "akita_black_bowl_1": "black bowl",
            "plate_1": "plate",
        },
        object_roles={
            "akita_black_bowl_1": ["manipulable"],
            "plate_1": ["destination"],
        },
    )

    records = load_complete_jsonl(trace_path)
    assert [record["turn"] for record in records] == [1, 2, 3, 4]
    assert [record["invocation"] for record in records] == _happy_script()
    assert all("planner_raw_output" in record for record in records)
    assert all(record["planner_thinking"] == "scripted thinking" for record in records)
    assert all("primitive_trace" in record for record in records)
    assert all(
        set(record["feedback"])
        >= {
            "primitive_success",
            "task_success",
            "env_done",
            "termination_reason",
            "steps_executed",
            "holding",
        }
        for record in records
    )
    assert records[2]["feedback"]["target"] == "plate_1"
    assert records[2]["feedback"]["mode"] == "release_pose"

    decision = build_task_memory(
        records,
        {
            "task_success": 1.0,
            "instruction": "place the bowl",
            "num_turns": 4,
        },
    )
    assert decision.accepted is True
    assert [command["action"] for command in decision.commands] == [
        "vla_act",
        "move_to",
        "move_to",
        "release",
    ]
    assert decision.commands[0]["target"]["label"] == "black bowl"
    assert decision.commands[1]["target"]["label"] == "plate"