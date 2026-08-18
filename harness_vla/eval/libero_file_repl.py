"""Idempotent file-mediated REPL for synchronous and persistent LIBERO workers.

The indexed files, polling lock, hashes, and local lifecycle form a beta,
paper-compatible protocol. The worker factory, not the protocol, owns live state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, Optional

from harness_vla.trace_io import (
    load_complete_jsonl,
    write_json_atomic,
    write_text_atomic,
)


SCHEMA_VERSION = 1
SHUTDOWN_NAME = "shutdown.json"
WORKER_LOCK_NAME = "worker.lock"


class LiberoFileProtocolError(RuntimeError):
    """Raised when persisted protocol files are inconsistent or invalid."""


def _name(kind: str, turn: int) -> str:
    return "%s_%02d.json" % (kind, turn)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiberoFileProtocolError("malformed protocol file: %s" % path.name) from exc
    if not isinstance(payload, dict):
        raise LiberoFileProtocolError("protocol file must contain an object: %s" % path.name)
    return payload


def _command_digest(command: Dict[str, Any]) -> str:
    encoded = json.dumps(
        command, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _update_status(path: Path, **updates: Any) -> None:
    status = {}
    if path.exists():
        try:
            status = _read_json(path)
        except LiberoFileProtocolError:
            status = {}
    status.update(updates)
    write_json_atomic(path, status)


class LiberoFileREPL:
    """Publish planner turns and execute at most one pending command per call."""

    def __init__(self, directory, executor: Optional[Callable[[Any], Any]] = None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.executor = executor
        self.ledger_path = self.directory / "ledger.jsonl"
        self.status_path = self.directory / "status.json"

    def _commands(self) -> Dict[int, Path]:
        commands = {}
        for path in self.directory.glob("command_*.json"):
            suffix = path.stem[len("command_") :]
            if not suffix.isdigit() or int(suffix) < 1 or path.name != _name("command", int(suffix)):
                raise LiberoFileProtocolError("invalid command filename: %s" % path.name)
            commands[int(suffix)] = path
        return commands

    def _validate_command(self, path: Path, expected_turn: int) -> Dict[str, Any]:
        command = _read_json(path)
        if set(command) != {"schema_version", "turn", "invocation"}:
            raise LiberoFileProtocolError("invalid command schema: %s" % path.name)
        if command["schema_version"] != SCHEMA_VERSION or command["turn"] != expected_turn:
            raise LiberoFileProtocolError("command identity mismatch: %s" % path.name)
        if not isinstance(command["invocation"], dict):
            raise LiberoFileProtocolError("command invocation must be an object: %s" % path.name)
        return command

    def _completed(self) -> Dict[int, Dict[str, Any]]:
        completed = {}
        turn = 1
        while True:
            state_path = self.directory / _name("state", turn)
            log_path = self.directory / _name("log", turn)
            if not state_path.exists():
                future_states = []
                for path in self.directory.glob("state_*.json"):
                    suffix = path.stem[len("state_") :]
                    if suffix.isdigit() and int(suffix) > turn:
                        future_states.append(path)
                if future_states:
                    raise LiberoFileProtocolError("state gap at turn %d" % turn)
                break
            if not log_path.exists():
                raise LiberoFileProtocolError("state without log at turn %d" % turn)
            command_path = self.directory / _name("command", turn)
            if not command_path.exists():
                raise LiberoFileProtocolError("state without command at turn %d" % turn)
            command = self._validate_command(command_path, turn)
            digest = _command_digest(command)
            state = _read_json(state_path)
            log = _read_json(log_path)
            required = {"schema_version", "turn", "command_sha256", "done", "error"}
            if not required.issubset(state) or not required.issubset(log):
                raise LiberoFileProtocolError("invalid result schema at turn %d" % turn)
            if any(item.get("schema_version") != SCHEMA_VERSION for item in (state, log)):
                raise LiberoFileProtocolError("result schema version mismatch at turn %d" % turn)
            if any(item.get("turn") != turn or item.get("command_sha256") != digest for item in (state, log)):
                raise LiberoFileProtocolError("conflicting replay at turn %d" % turn)
            completed[turn] = state
            turn += 1
        return completed

    def _sync_metadata(self, completed: Dict[int, Dict[str, Any]]) -> None:
        if not completed:
            return
        canonical = [
            {
                "schema_version": SCHEMA_VERSION,
                "turn": turn,
                "command_sha256": state["command_sha256"],
                "done": state["done"],
                "error": state["error"],
            }
            for turn, state in completed.items()
        ]
        ledger = load_complete_jsonl(self.ledger_path)
        if ledger != canonical[:len(ledger)] or len(ledger) > len(canonical):
            raise LiberoFileProtocolError("ledger conflicts with committed states")
        if ledger != canonical or (
            self.ledger_path.exists()
            and self.ledger_path.read_bytes()
            and not self.ledger_path.read_bytes().endswith(b"\n")
        ):
            payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in canonical)
            write_text_atomic(self.ledger_path, payload)
        latest = completed[max(completed)]
        status = {
            "schema_version": SCHEMA_VERSION,
            "last_completed_turn": latest["turn"],
            "done": latest["done"],
            "error": latest["error"],
        }
        current = _read_json(self.status_path) if self.status_path.exists() else {}
        if any(current.get(key) != value for key, value in status.items()):
            _update_status(self.status_path, **status)

    def step(self, invocation: Dict[str, Any], *, turn: Optional[int] = None) -> Dict[str, Any]:
        """Atomically publish one command after the preceding state is committed."""
        if not isinstance(invocation, dict):
            raise ValueError("invocation must be an object")
        completed = self._completed()
        self._sync_metadata(completed)
        commands = self._commands()
        next_turn = len(completed) + 1
        requested_turn = next_turn if turn is None else turn
        if isinstance(requested_turn, bool) or not isinstance(requested_turn, int) or requested_turn < 1:
            raise ValueError("turn must be a positive integer")
        command = {
            "schema_version": SCHEMA_VERSION,
            "turn": requested_turn,
            "invocation": invocation,
        }
        path = self.directory / _name("command", requested_turn)
        if requested_turn > next_turn:
            raise LiberoFileProtocolError("command gap at turn %d" % requested_turn)
        if requested_turn < next_turn or requested_turn in commands:
            persisted = self._validate_command(path, requested_turn)
            if _command_digest(persisted) != _command_digest(command):
                raise LiberoFileProtocolError("conflicting replay at turn %d" % requested_turn)
            return completed.get(requested_turn, persisted)
        if commands and max(commands) >= requested_turn:
            raise LiberoFileProtocolError("pending command sequence is inconsistent")
        write_json_atomic(path, command)
        return command

    def process_one(self) -> Optional[Dict[str, Any]]:
        """Execute and commit one pending command, or return the persisted replay."""
        completed = self._completed()
        self._sync_metadata(completed)
        commands = self._commands()
        next_turn = len(completed) + 1
        if any(turn > next_turn for turn in commands):
            raise LiberoFileProtocolError("command gap at turn %d" % next_turn)
        if next_turn not in commands:
            return completed.get(max(completed)) if completed else None
        command = self._validate_command(commands[next_turn], next_turn)
        if self.executor is None:
            raise RuntimeError("process_one requires an executor")
        outcome = self.executor(command["invocation"])
        if not isinstance(outcome, dict):
            raise TypeError("executor result must be an object")
        done = bool(outcome.get("done", False))
        error = outcome.get("error")
        if error is not None and not isinstance(error, str):
            raise TypeError("executor error must be a string or null")
        digest = _command_digest(command)
        state = {
            "schema_version": SCHEMA_VERSION,
            "turn": next_turn,
            "command_sha256": digest,
            "state": outcome.get("state"),
            "done": done,
            "error": error,
        }
        log = {
            "schema_version": SCHEMA_VERSION,
            "turn": next_turn,
            "command_sha256": digest,
            "invocation": command["invocation"],
            "result": outcome,
            "done": done,
            "error": error,
        }
        write_json_atomic(self.directory / _name("log", next_turn), log)
        write_json_atomic(self.directory / _name("state", next_turn), state)
        completed[next_turn] = state
        self._sync_metadata(completed)
        return state

    def ledger(self):
        """Return all complete advisory ledger records."""
        return load_complete_jsonl(self.ledger_path)

    def request_shutdown(self) -> None:
        """Atomically request shutdown of a separate persistent worker."""
        write_json_atomic(
            self.directory / SHUTDOWN_NAME,
            {"schema_version": SCHEMA_VERSION, "shutdown": True},
        )


class LiberoFileWorkerProcess:
    """Beta persistent worker that creates and owns one live executor per episode."""

    def __init__(self, directory, factory: Callable[[], Any], *, poll_interval=0.05):
        if not callable(factory):
            raise TypeError("factory must be callable")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.factory = factory
        self.poll_interval = poll_interval
        self.status_path = self.directory / "status.json"
        self.lock_path = self.directory / WORKER_LOCK_NAME

    def _acquire_lock(self) -> None:
        try:
            descriptor = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise LiberoFileProtocolError("worker lock already held") from exc
        with os.fdopen(descriptor, "w", encoding="ascii") as lock_file:
            lock_file.write(str(os.getpid()) + "\n")
            lock_file.flush()
            os.fsync(lock_file.fileno())

    def _shutdown_requested(self) -> bool:
        path = self.directory / SHUTDOWN_NAME
        if not path.exists():
            return False
        payload = _read_json(path)
        if payload != {"schema_version": SCHEMA_VERSION, "shutdown": True}:
            raise LiberoFileProtocolError("invalid shutdown schema")
        return True

    @staticmethod
    def _executor(owner: Any) -> Callable[[Any], Any]:
        executor = owner if callable(owner) else getattr(owner, "execute", None)
        if not callable(executor):
            raise TypeError("factory result must be callable or define execute()")
        return executor

    def run(self) -> None:
        """Own the executor and process contiguous commands until explicit shutdown."""
        self._acquire_lock()
        owner = None
        failure = None
        try:
            owner = self.factory()
            repl = LiberoFileREPL(self.directory, self._executor(owner))
            _update_status(
                self.status_path,
                schema_version=SCHEMA_VERSION,
                worker="running",
                worker_pid=os.getpid(),
                worker_error=None,
            )
            while True:
                shutdown_requested = self._shutdown_requested()
                completed_before = len(repl._completed())
                repl.process_one()
                completed_after = len(repl._completed())
                if completed_after == completed_before and shutdown_requested:
                    break
                if completed_after == completed_before:
                    time.sleep(self.poll_interval)
        except Exception as exc:
            failure = exc
            _update_status(
                self.status_path,
                schema_version=SCHEMA_VERSION,
                worker="failed",
                worker_pid=os.getpid(),
                worker_error="%s: %s" % (type(exc).__name__, exc),
            )
        finally:
            close = getattr(owner, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    if failure is None:
                        failure = exc
                        _update_status(
                            self.status_path,
                            schema_version=SCHEMA_VERSION,
                            worker="failed",
                            worker_pid=os.getpid(),
                            worker_error="%s: %s" % (type(exc).__name__, exc),
                        )
            if failure is None:
                _update_status(
                    self.status_path,
                    schema_version=SCHEMA_VERSION,
                    worker="stopped",
                    worker_pid=os.getpid(),
                    worker_error=None,
                )
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass
        if failure is not None:
            raise failure


def _load_factory(specification: str) -> Callable[[], Any]:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("factory must use module:attribute syntax")
    factory = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(factory):
        raise TypeError("configured factory must be callable")
    return factory


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run a persistent LIBERO file REPL worker")
    parser.add_argument("--directory", required=True)
    parser.add_argument("--factory", required=True, help="zero-argument module:attribute factory")
    parser.add_argument("--poll-interval", type=float, default=0.05)
    arguments = parser.parse_args(argv)
    try:
        LiberoFileWorkerProcess(
            arguments.directory,
            lambda: _load_factory(arguments.factory)(),
            poll_interval=arguments.poll_interval,
        ).run()
    except Exception:
        return 1
    return 0


LiberoFilePlanner = LiberoFileREPL
LiberoFileWorker = LiberoFileREPL


if __name__ == "__main__":
    raise SystemExit(main())
