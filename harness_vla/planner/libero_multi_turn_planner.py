"""Zero-shot planner for the native LIBERO multi-turn Harness loop."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence, Tuple

from harness_vla.planner.harness_planner import extract_json_object
from harness_vla.primitives.libero_primitives import (
    LiberoPrimitiveError,
    validate_pose_target,
    validate_rotation_setpoint,
)
from harness_vla.planner.libero_vla_planner import LiberoVLAPlanner


MULTI_TURN_SYSTEM_PROMPT = """You are the planner for a native LIBERO Harness loop.
Emit exactly one JSON primitive per turn and no prose.
The only available primitives are:
{"action":"vla_act","prompt":"<local contact prompt>","target":"<grounded name>","max_chunks":<integer>,"tau":"lift_and_grasp"}
{"action":"vla_act","prompt":"<contact-rich placement prompt naming the grounded destination>","target":"<held grounded name>","max_chunks":<integer>,"tau":"task_success"}
{"action":"move_to","target":"<grounded name>","mode":"above|release_pose","gripper":"close"}
{"action":"move_pose","xyz":[<x>,<y>,<z>],"pose":[<qx>,<qy>,<qz>,<qw>],"gripper":"open|close"}
{"action":"rotate_wrist","target_yaw":<radians>}
{"action":"rotate_pitch","target_pitch":<radians>}
{"action":"set_gripper","gripper":"open|close"}
{"action":"release"}
Choose grounded names only for target-based calls. Explicit move_pose values must
come from current visual RGB-D/world-map evidence, never simulator or oracle poses.
Never emit joint commands, torques, numeric tolerances, success claims, or multiple primitives. Primitive success
does not imply task success. A successful move_to means that target and mode
already hold; do not repeat it unless feedback reports a failure or state
change. After release_pose succeeds, use release only for unconstrained
placement. For tight LIBERO placement while still holding the object, use
vla_act with tau=task_success and name the grounded destination in its prompt.
Use the latest feedback to choose the next action. For a contact attempt, use
the available max_chunks cap; tau already returns early, so do not issue a
one-chunk probe before the same contact prompt.
Use tau=task_success only for contact-rich placement while holding exactly the
target object, or with an empty hand for fixture manipulation (open a drawer,
turn on a stove) with the fixture as target; never use lift_and_grasp on a
fixture or region. The prompt must explicitly name the grounded destination.
This mode terminates only on the environment's official task-success predicate.
tau=task_success fires only when the WHOLE task is complete: for multi-object
tasks handle one object at a time. NEVER use tau=task_success while another
required object is still unplaced — it cannot fire and only burns steps. Place
every non-final object with move_to mode=release_pose (gripper close) on the
destination followed by release, then grasp the next object; reserve
vla_act tau=task_success for the FINAL object only. The semantic state includes
turn_history with all your previous primitives and results: an object already
released at its destination (result release_completed_task_incomplete) is DONE;
never grasp or place it again — proceed to the next required object. If the
same invocation failed twice in a row with the same error, do not repeat it;
change primitive, target or tau.
CRITICAL: in every vla_act the "target" field is ALWAYS the object being
manipulated (the held object during placement), NEVER the destination. The
destination appears only inside the "prompt" text. While holding
alphabet_soup_1 to place it in basket_1, emit
{"action":"vla_act","prompt":"place the alphabet soup in the basket","target":"alphabet_soup_1","max_chunks":20,"tau":"task_success"}
and never "target":"basket_1"; a destination target aborts the episode."""


def _valid_cap(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("%s must be an integer greater than zero" % name)


def parse_libero_multi_turn_invocation(
    raw_output: str,
    *,
    available_targets: Sequence[str],
    max_chunks_cap: int,
) -> Optional[Dict[str, object]]:
    """Extract one invocation and fail closed on unknown fields or targets."""
    _valid_cap(max_chunks_cap, "max_chunks_cap")
    targets = {
        target for target in available_targets if isinstance(target, str) and target
    }
    parsed = extract_json_object(raw_output)
    if not isinstance(parsed, dict):
        return None
    action = parsed.get("action")

    if action == "vla_act":
        if set(parsed) != {"action", "prompt", "target", "max_chunks", "tau"}:
            return None
        prompt = parsed.get("prompt")
        target = parsed.get("target")
        max_chunks = parsed.get("max_chunks")
        if not isinstance(prompt, str) or not prompt.strip():
            return None
        if not isinstance(target, str) or target not in targets:
            return None
        if (
            isinstance(max_chunks, bool)
            or not isinstance(max_chunks, int)
            or not 1 <= max_chunks <= max_chunks_cap
        ):
            return None
        tau = parsed.get("tau")
        if tau not in ("lift_and_grasp", "task_success"):
            return None
        return {
            "action": "vla_act",
            "prompt": prompt.strip(),
            "target": target,
            "max_chunks": max_chunks,
            "tau": tau,
        }

    if action == "move_to":
        if set(parsed) != {"action", "target", "mode", "gripper"}:
            return None
        target = parsed.get("target")
        if not isinstance(target, str) or target not in targets:
            return None
        if parsed.get("mode") not in ("above", "release_pose"):
            return None
        if parsed.get("gripper") != "close":
            return None
        return {
            "action": "move_to",
            "target": target,
            "mode": parsed["mode"],
            "gripper": "close",
        }

    if action == "move_pose":
        if set(parsed) != {"action", "xyz", "pose", "gripper"}:
            return None
        if parsed.get("gripper") not in ("open", "close"):
            return None
        try:
            xyz, pose = validate_pose_target(parsed.get("xyz"), parsed.get("pose"))
        except (LiberoPrimitiveError, TypeError):
            return None
        return {
            "action": "move_pose",
            "xyz": xyz.tolist(),
            "pose": pose.tolist(),
            "gripper": parsed["gripper"],
        }

    rotation_fields = {
        "rotate_wrist": ("target_yaw", "yaw"),
        "rotate_pitch": ("target_pitch", "pitch"),
    }
    if action in rotation_fields:
        field, axis = rotation_fields[action]
        if set(parsed) != {"action", field}:
            return None
        try:
            angle = validate_rotation_setpoint(axis, parsed.get(field))
        except LiberoPrimitiveError:
            return None
        return {"action": action, field: angle}

    if action == "set_gripper":
        if set(parsed) != {"action", "gripper"}:
            return None
        gripper = parsed.get("gripper")
        return parsed if gripper in ("open", "close") else None

    if action == "release":
        return {"action": "release"} if set(parsed) == {"action"} else None
    return None


class LiberoMultiTurnPlanner(LiberoVLAPlanner):
    """Ollama planner that receives semantic state and emits one primitive."""

    _chat_attempts = 3

    def _chat(self, user_prompt: str) -> str:
        import time
        from urllib import error, request

        options = {"temperature": 0}
        if self.think:
            options.update({"num_predict": 8192, "num_ctx": 16384})
        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": MULTI_TURN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "think": self.think,
                "options": options,
            }
        ).encode("utf-8")
        req = request.Request(
            self._endpoint(), data=payload, headers={"Content-Type": "application/json"}
        )
        for attempt in range(self._chat_attempts):
            try:
                with request.urlopen(req, timeout=self.request_timeout) as response:
                    result = json.load(response)
            except (error.URLError, OSError, ValueError):
                time.sleep(min(5 * (attempt + 1), 15))
                continue
            message = result.get("message", {})
            content = message.get("content", "")
            if content.strip():
                self.last_thinking = message.get("thinking") or None
                return content
            # empty content is a transient backend failure; retry
            time.sleep(min(5 * (attempt + 1), 15))
        self.last_thinking = None
        return ""

    def act_turn(
        self,
        instruction: str,
        state: Dict[str, Any],
        *,
        available_targets: Sequence[str],
        max_chunks_cap: int,
        memory_context: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, object]], str]:
        """Choose one primitive from the current semantic state."""
        _valid_cap(max_chunks_cap, "max_chunks_cap")
        if memory_context is not None:
            if not isinstance(memory_context, str) or not memory_context.strip():
                raise ValueError("memory_context must be a non-empty string")
            forbidden = ("xyz", "pose", "coordinate")
            if any(token in memory_context.casefold() for token in forbidden):
                raise ValueError("memory_context must contain symbolic information only")
        prompt_lines = [
                "Official task instruction: %s" % instruction,
                "Grounded target names: %s" % json.dumps(list(available_targets)),
                "Remaining vla_act chunk cap: %d" % max_chunks_cap,
                "Current semantic state: %s"
                % json.dumps(state, sort_keys=True, separators=(",", ":")),
        ]
        if memory_context is not None:
            prompt_lines.extend(("Deployment memory context:", memory_context.strip()))
        prompt_lines.append("Return exactly one JSON primitive.")
        user_prompt = "\n".join(prompt_lines)
        raw_output = self._chat(user_prompt)
        invocation = parse_libero_multi_turn_invocation(
            raw_output,
            available_targets=available_targets,
            max_chunks_cap=max_chunks_cap,
        )
        return invocation, raw_output


__all__ = [
    "LiberoMultiTurnPlanner",
    "MULTI_TURN_SYSTEM_PROMPT",
    "parse_libero_multi_turn_invocation",
]