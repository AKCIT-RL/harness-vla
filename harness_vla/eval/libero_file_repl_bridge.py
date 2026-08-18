"""Synchronous bridge between multi-turn primitives and the LIBERO file REPL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from harness_vla.eval.libero_file_repl import LiberoFileREPL


@dataclass(frozen=True)
class LiberoProtocolMetadata:
    directory: Path
    protocol_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if isinstance(self.protocol_seed, bool) or not isinstance(self.protocol_seed, int):
            raise ValueError("protocol_seed must be an integer")


class LiberoFileREPLBridge:
    """Publish and synchronously execute exactly one physical primitive per turn."""

    def __init__(
        self,
        metadata: LiberoProtocolMetadata,
        *,
        vla_executor,
        move_executor,
        release_executor,
    ) -> None:
        self.metadata = metadata
        self._executors = {
            "vla_act": vla_executor,
            "move_to": move_executor,
            "move_pose": move_executor,
            "rotate_wrist": move_executor,
            "rotate_pitch": move_executor,
            "set_gripper": move_executor,
            "release": release_executor,
        }
        self._pending: Dict[str, Any] = {}
        self._results: Dict[int, Any] = {}
        self._repl = LiberoFileREPL(metadata.directory, self._execute_pending)

    def _execution_summary(self, result: Any) -> Mapping[str, Any]:
        execution = getattr(result, "execution", result)
        holding = getattr(result, "holding", self._pending.get("holding"))
        invocation = self._pending["invocation"]
        if execution.primitive_success and (
            invocation["action"] == "release"
            or (
                invocation["action"] == "set_gripper"
                and invocation.get("gripper") == "open"
            )
        ):
            holding = None
        return {
            "primitive_success": bool(execution.primitive_success),
            "task_success": bool(execution.task_success),
            "termination_reason": execution.termination_reason,
            "steps_executed": execution.steps_executed,
            "holding": holding,
            "tau_satisfied": getattr(result, "tau_satisfied", None),
        }

    def _execute_pending(self, invocation: Mapping[str, Any]) -> Dict[str, Any]:
        turn = self._pending["turn"]
        executor = self._executors[invocation["action"]]
        result = executor(
            dict(invocation),
            self._pending["observation"],
            max_steps=self._pending["max_steps"],
        )
        self._results[turn] = result
        summary = dict(self._execution_summary(result))
        return {
            "state": summary,
            "done": bool(summary["task_success"]),
            "error": None,
        }

    def dispatch(
        self,
        invocation: Dict[str, Any],
        observation: Mapping[str, Any],
        *,
        max_steps: int,
        turn: int,
        holding=None,
    ):
        self._pending = {
            "turn": turn,
            "invocation": invocation,
            "observation": observation,
            "max_steps": max_steps,
            "holding": holding,
        }
        self._repl.step(invocation, turn=turn)
        state = self._repl.process_one()
        result = self._results.pop(turn, None)
        self._pending = {}
        if result is None:
            raise RuntimeError("file REPL replay lacks an in-process physical result")
        files = {
            kind: str(self.metadata.directory / ("%s_%02d.json" % (kind, turn)))
            for kind in ("command", "state", "log")
        }
        return result, {
            "protocol_seed": self.metadata.protocol_seed,
            "turn": state["turn"],
            "files": files,
            "command_sha256": state["command_sha256"],
        }


__all__ = ["LiberoFileREPLBridge", "LiberoProtocolMetadata"]