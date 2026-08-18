import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from harness_vla.eval.libero_file_repl import (
    LiberoFileProtocolError,
    LiberoFileREPL,
)
from harness_vla.eval.libero_file_repl_bridge import (
    LiberoFileREPLBridge,
    LiberoProtocolMetadata,
)
from harness_vla.eval.libero_analytic_executor import LiberoPrimitiveExecution
from harness_vla.eval.libero_multi_turn_evaluator import LiberoVLAExecution


def _outcome(value, *, done=False):
    return {"state": {"value": value}, "done": done, "error": None}


def _wait_for_json(path, predicate=lambda payload: True, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.01)
            continue
        if predicate(payload):
            return payload
        time.sleep(0.01)
    raise AssertionError("timed out waiting for %s" % path)


def _worker_process(protocol_directory, factory_directory):
    factory_module = factory_directory / "repl_test_factory.py"
    factory_module.write_text(
        """import json
import os

class StatefulOwner:
    def __init__(self):
        self.count = 0

    def execute(self, invocation):
        self.count += 1
        return {\"state\": {\"count\": self.count, \"value\": invocation[\"value\"]},
                \"done\": False, \"error\": None}

    def close(self):
        with open(os.environ[\"REPL_CLOSE_MARKER\"], \"w\", encoding=\"utf-8\") as marker:
            json.dump({\"closed\": True, \"count\": self.count}, marker)

def build():
    return StatefulOwner()
""",
        encoding="utf-8",
    )
    close_marker = protocol_directory / "owner_closed.json"
    environment = os.environ.copy()
    package_root = Path(__file__).parents[1] / "EmbodiedBench"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(factory_directory), str(package_root), environment.get("PYTHONPATH", ""))
    )
    environment["REPL_CLOSE_MARKER"] = str(close_marker)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "harness_vla.eval.libero_file_repl",
            "--directory",
            str(protocol_directory),
            "--factory",
            "repl_test_factory:build",
            "--poll-interval",
            "0.01",
        ],
        env=environment,
    )
    _wait_for_json(
        protocol_directory / "status.json",
        lambda status: status.get("worker") == "running",
    )
    return process, close_marker


def test_one_command_executes_exactly_once_and_commits_all_files(tmp_path):
    calls = []
    repl = LiberoFileREPL(tmp_path, lambda invocation: calls.append(invocation) or _outcome(1))
    repl.step({"name": "move", "target": "bowl"})

    state = repl.process_one()

    assert calls == [{"name": "move", "target": "bowl"}]
    assert state["state"] == {"value": 1}
    assert {path.name for path in tmp_path.iterdir()} == {
        "command_01.json", "state_01.json", "log_01.json", "ledger.jsonl", "status.json"
    }


def test_replay_and_restart_return_persisted_result_without_execution(tmp_path):
    first = LiberoFileREPL(tmp_path, lambda invocation: _outcome(7))
    invocation = {"name": "grasp", "target": "cup"}
    first.step(invocation)
    expected = first.process_one()
    calls = []
    restarted = LiberoFileREPL(tmp_path, lambda invocation: calls.append(invocation))

    assert restarted.step(invocation, turn=1) == expected
    assert restarted.process_one() == expected
    assert calls == []


def test_gap_and_conflicting_replay_fail_closed(tmp_path):
    repl = LiberoFileREPL(tmp_path, lambda invocation: _outcome(1))
    with pytest.raises(LiberoFileProtocolError, match="gap"):
        repl.step({"name": "move"}, turn=2)
    repl.step({"name": "move"})
    repl.process_one()
    with pytest.raises(LiberoFileProtocolError, match="conflicting replay"):
        repl.step({"name": "release"}, turn=1)


def test_states_are_monotonic_and_next_command_waits_for_state(tmp_path):
    repl = LiberoFileREPL(tmp_path, lambda invocation: _outcome(invocation["value"]))
    repl.step({"value": 1})
    with pytest.raises(LiberoFileProtocolError):
        repl.step({"value": 2})
    repl.process_one()
    repl.step({"value": 2})
    repl.process_one()
    assert json.loads((tmp_path / "state_01.json").read_text())["turn"] == 1
    assert json.loads((tmp_path / "state_02.json").read_text())["turn"] == 2


def test_crash_before_commit_can_execute_again_on_retry(tmp_path, monkeypatch):
    calls = []
    repl = LiberoFileREPL(tmp_path, lambda invocation: calls.append(invocation) or _outcome(3))
    repl.step({"name": "move"})
    import harness_vla.eval.libero_file_repl as module

    original = module.write_json_atomic
    crashed = False

    def crash_before_commit(path, payload):
        nonlocal crashed
        if not crashed and path.name == "log_01.json":
            crashed = True
            raise OSError("simulated crash")
        original(path, payload)

    monkeypatch.setattr(module, "write_json_atomic", crash_before_commit)
    with pytest.raises(OSError, match="simulated crash"):
        repl.process_one()
    assert not (tmp_path / "state_01.json").exists()
    result = repl.process_one()
    assert result["state"] == {"value": 3}
    assert len(calls) == 2


def test_restart_repairs_metadata_after_committed_state_without_reexecution(
    tmp_path, monkeypatch
):
    calls = []
    repl = LiberoFileREPL(tmp_path, lambda invocation: calls.append(invocation) or _outcome(4))
    repl.step({"name": "move"})
    import harness_vla.eval.libero_file_repl as module

    original = module.write_text_atomic

    def crash_after_state(path, record):
        raise OSError("simulated metadata crash")

    monkeypatch.setattr(module, "write_text_atomic", crash_after_state)
    with pytest.raises(OSError, match="metadata crash"):
        repl.process_one()
    monkeypatch.setattr(module, "write_text_atomic", original)

    restarted = LiberoFileREPL(tmp_path, lambda invocation: calls.append(invocation))
    state = restarted.process_one()
    assert state["state"] == {"value": 4}
    assert len(calls) == 1
    assert restarted.ledger()[0]["turn"] == 1
    assert json.loads((tmp_path / "status.json").read_text())["last_completed_turn"] == 1


def test_malformed_command_is_rejected_without_execution(tmp_path):
    (tmp_path / "command_01.json").write_text('{"turn": 1}', encoding="utf-8")
    calls = []
    with pytest.raises(LiberoFileProtocolError, match="schema"):
        LiberoFileREPL(tmp_path, lambda invocation: calls.append(invocation)).process_one()
    assert calls == []


def test_files_and_schemas_are_self_contained_and_reconstructable(tmp_path):
    repl = LiberoFileREPL(tmp_path, lambda invocation: _outcome(9, done=True))
    repl.step({"name": "finish"})
    repl.process_one()
    command = json.loads((tmp_path / "command_01.json").read_text())
    state = json.loads((tmp_path / "state_01.json").read_text())
    log = json.loads((tmp_path / "log_01.json").read_text())
    status = json.loads((tmp_path / "status.json").read_text())
    assert command == {"schema_version": 1, "turn": 1, "invocation": {"name": "finish"}}
    assert state["command_sha256"] == log["command_sha256"]
    assert log["invocation"] == command["invocation"]
    assert state["done"] is status["done"] is True
    assert repl.ledger()[0]["turn"] == status["last_completed_turn"] == 1


def test_subprocess_worker_persists_state_replays_once_and_shuts_down(tmp_path):
    protocol_directory = tmp_path / "protocol"
    protocol_directory.mkdir()
    process, close_marker = _worker_process(protocol_directory, tmp_path)
    client = LiberoFileREPL(protocol_directory)
    try:
        client.step({"value": "first"}, turn=1)
        first = _wait_for_json(protocol_directory / "state_01.json")
        assert first["state"] == {"count": 1, "value": "first"}

        assert client.step({"value": "first"}, turn=1) == first
        client.step({"value": "second"}, turn=2)
        second = _wait_for_json(protocol_directory / "state_02.json")
        assert second["state"] == {"count": 2, "value": "second"}

        client.request_shutdown()
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)

    assert json.loads(close_marker.read_text(encoding="utf-8")) == {
        "closed": True,
        "count": 2,
    }
    status = json.loads((protocol_directory / "status.json").read_text(encoding="utf-8"))
    assert status["worker"] == "stopped"
    assert status["last_completed_turn"] == 2
    assert not (protocol_directory / "worker.lock").exists()


@pytest.mark.parametrize("invalid_kind", ["malformed", "gap"])
def test_subprocess_worker_fails_closed_on_invalid_sequence(
    tmp_path, invalid_kind
):
    protocol_directory = tmp_path / "protocol"
    protocol_directory.mkdir()
    process, close_marker = _worker_process(protocol_directory, tmp_path)
    try:
        if invalid_kind == "malformed":
            (protocol_directory / "command_01.json").write_text(
                '{"turn": 1}', encoding="utf-8"
            )
        else:
            (protocol_directory / "command_02.json").write_text(
                json.dumps(
                    {"schema_version": 1, "turn": 2, "invocation": {"value": "late"}}
                ),
                encoding="utf-8",
            )
        assert process.wait(timeout=5) == 1
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)

    status = json.loads((protocol_directory / "status.json").read_text(encoding="utf-8"))
    assert status["worker"] == "failed"
    assert invalid_kind in status["worker_error"].lower() or (
        invalid_kind == "malformed" and "schema" in status["worker_error"].lower()
    )
    assert not list(protocol_directory.glob("state_*.json"))
    assert json.loads(close_marker.read_text(encoding="utf-8"))["count"] == 0


def test_bridge_commits_primitives_and_persists_analytic_holding_effects(tmp_path):
    calls = []

    def execution(reason="postcondition_met"):
        return LiberoPrimitiveExecution(
            observation={"frame": len(calls)},
            primitive_success=True,
            task_success=False,
            termination_reason=reason,
            steps_executed=1,
            trace=[],
        )

    def vla(invocation, observation, *, max_steps):
        calls.append(invocation["action"])
        return LiberoVLAExecution(
            execution("lift_and_grasp_satisfied"), True, "bowl_1"
        )

    def move(invocation, observation, *, max_steps):
        calls.append(invocation["action"])
        return execution()

    def release(invocation, observation, *, max_steps):
        calls.append(invocation["action"])
        return execution("release_completed_task_incomplete")

    bridge = LiberoFileREPLBridge(
        LiberoProtocolMetadata(tmp_path, 17),
        vla_executor=vla,
        move_executor=move,
        release_executor=release,
    )
    invocations = [
        {"action": "vla_act"},
        {"action": "move_to"},
        {"action": "move_pose"},
        {"action": "rotate_wrist"},
        {"action": "rotate_pitch"},
        {"action": "set_gripper", "gripper": "open"},
        {"action": "release"},
    ]
    for turn, invocation in enumerate(invocations, 1):
        bridge.dispatch(
            invocation,
            {"frame": turn},
            max_steps=1,
            turn=turn,
            holding="bowl_1",
        )

    assert calls == [invocation["action"] for invocation in invocations]
    for turn in range(1, 8):
        assert (tmp_path / ("command_%02d.json" % turn)).is_file()
        assert (tmp_path / ("log_%02d.json" % turn)).is_file()
        state = json.loads((tmp_path / ("state_%02d.json" % turn)).read_text())
        expected_holding = None if turn in (6, 7) else "bowl_1"
        assert state["state"]["holding"] == expected_holding