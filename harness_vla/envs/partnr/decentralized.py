"""``HarnessVlaDecentralized``: one Harness VLA brain per agent, coordinated
through published transcripts.

Each agent runs its own instance behind the env's ``HarnessBridgePlanner``
(upstream decentralized runner). Inherits everything from
``HarnessVlaCentralized`` but controls a single label, keeps INDIVIDUAL Task
and Global Memory (``<dir>/<label>/``), and learns what the partner did by
reading the partner's published transcript
``<shared_memory_dir>/<episode_id>/<partner_label>.jsonl`` before every
decision. Only actions and their results are published — never thoughts.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from akc_harness.contracts import (
    Decision,
    DoneDecision,
    EpisodeContext,
    StepContext,
    ToolCall,
)

from harness_vla.envs.partnr.agent import (
    HarnessVlaCentralized,
    agent_label,
    resolve_labeled_task_memory,
)
from harness_vla.envs.partnr.memory import load_partnr_task_memory

_LOG = logging.getLogger(__name__)

DEFAULT_DECENTRAL_ROLE = (
    "You are {label}, a {description}, collaborating with {partner} (a "
    "{partner_description}) on a household task. You control ONLY yourself, "
    "one atomic primitive per turn. Your partner acts independently; you learn "
    "what it did from its published transcript."
)
_PUBLISHED_KEYS = ("decision_index", "agent", "action", "status", "message")


class HarnessVlaDecentralized(HarnessVlaCentralized):
    """Per-agent Harness VLA brain; partner awareness via published transcripts."""

    def __init__(
        self,
        *args: Any,
        shared_memory_dir: Optional[str] = None,
        role: Optional[str] = None,
        use_claims: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, role=role or DEFAULT_DECENTRAL_ROLE, use_claims=False, **kwargs)
        if use_claims:
            _LOG.warning("claims are a centralized feature; ignored by HarnessVlaDecentralized")
        self.shared_memory_dir = shared_memory_dir
        self.me: Optional[str] = None
        self.partner: Optional[str] = None
        self._own_file: Optional[Path] = None
        self._partner_file: Optional[Path] = None
        self._role_template = role or DEFAULT_DECENTRAL_ROLE

    # ------------------------------------------------------------ overrides

    def _controlled_labels(self, context: EpisodeContext) -> List[str]:
        return [agent_label(context.agent_uid)]

    def _scoped(self, base: Optional[str]) -> Optional[str]:
        if not base or self.me is None:
            return base
        return str(Path(base) / self.me)

    def _own_task_memory_dir(self) -> Optional[str]:
        return self._scoped(self.task_memory_dir)

    def _own_global_memory_path(self) -> Optional[str]:
        base = self.global_memory_path
        if not base or self.me is None:
            return base
        path = Path(base)
        if path.suffix:  # a file: shared ledger for both agents
            return str(path)
        return str(path / self.me / "global_memory.json")

    def _partner_task_memory(self, state: Optional[Mapping[str, Any]]) -> Optional[List[str]]:
        if self.zero_shot or not self.task_memory_dir or self.partner is None or state is None:
            return None
        if self._episode is None:
            return None
        try:
            entry = load_partnr_task_memory(
                str(Path(self.task_memory_dir) / self.partner), self._episode.instruction
            )
        except (OSError, ValueError, json.JSONDecodeError):
            _LOG.warning("partner task memory unreadable; ignoring", exc_info=True)
            return None
        return resolve_labeled_task_memory(entry, state) or None

    # ------------------------------------------------------------ lifecycle

    def on_episode_start(self, context: EpisodeContext) -> None:
        self.me = agent_label(context.agent_uid)
        partners = [agent_label(uid) for uid in (context.partner_uids or [])]
        self.partner = partners[0] if partners else None
        self.role = self._role_template.format(
            label=self.me,
            description=self.agent_descriptions.get(self.me, "agent"),
            partner=self.partner or "no partner",
            partner_description=self.agent_descriptions.get(self.partner or "", "agent"),
        )
        self._own_file = self._partner_file = None
        if self.shared_memory_dir:
            episode_dir = Path(self.shared_memory_dir) / str(context.episode_id)
            episode_dir.mkdir(parents=True, exist_ok=True)
            self._own_file = episode_dir / f"{self.me}.jsonl"
            try:
                self._own_file.unlink()
            except FileNotFoundError:
                pass
            if self.partner is not None:
                self._partner_file = episode_dir / f"{self.partner}.jsonl"
        super().on_episode_start(context)

    # -------------------------------------------------------- transcripts

    def _publish(self, record: Mapping[str, Any]) -> None:
        if self._own_file is None:
            return
        line = json.dumps({key: record.get(key) for key in _PUBLISHED_KEYS}, ensure_ascii=False)
        try:
            fd = os.open(str(self._own_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, (line + "\n").encode("utf-8"))
            finally:
                os.close(fd)
        except OSError:
            _LOG.warning("could not publish transcript line to %s", self._own_file, exc_info=True)

    def _partner_transcript(self) -> Optional[List[Dict[str, Any]]]:
        """Partner's published records, latest line per decision wins."""
        if self._partner_file is None:
            return None
        return read_published_transcript(self._partner_file)

    # --------------------------------------------------------------- decide

    def decide(self, context: StepContext) -> Decision:  # type: ignore[override]
        view = self._view(context)
        view["deciding"] = [self.me]
        view["busy"] = {}
        decisions = self._decide_team(view)
        decision = decisions.get(self.me)
        if decision is None:
            if isinstance(decisions.get("all"), DoneDecision):
                return decisions["all"]
            # Idle: the bridge treats a missing decision as an error, so emit a
            # short wait explicitly (recorded as such in the transcript).
            wait = self._wait_tool(self.me)
            vocab = self._vocab(view.get("state"), view.get("rooms"))
            self._record(self.me, view.get("decision_index"), wait, "", vocab)
            return ToolCall(wait, "", rationale="idle (no action chosen)", meta=dict(self.last_meta))
        return decision


def read_published_transcript(path: Path) -> List[Dict[str, Any]]:
    """Read ``<label>.jsonl``; the latest line for each ``decision_index`` wins."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError:
        _LOG.warning("could not read published transcript %s", path, exc_info=True)
        return []
    latest: Dict[Any, Dict[str, Any]] = {}
    order: List[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # partially written line
        if not isinstance(record, dict):
            continue
        key = (record.get("decision_index"), record.get("action"))
        if key not in latest:
            order.append(key)
        latest[key] = record
    return [latest[key] for key in order]
