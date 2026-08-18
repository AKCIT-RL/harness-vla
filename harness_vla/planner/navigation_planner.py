"""Beta-only EB-Navigation planner extension outside the Harness VLA paper."""

import json

from harness_vla.planner.harness_planner import (
    HarnessPlanner,
    extract_json_object,
)


NAV_ACTION_TO_INDEX = {
    "move_forward": 0,
    "move_backward": 1,
    "move_right": 2,
    "move_left": 3,
    "turn_right": 4,
    "turn_left": 5,
    "tilt_up": 6,
    "tilt_down": 7,
}
MOVEMENT_ACTIONS = frozenset(
    {"move_forward", "move_backward", "move_right", "move_left"}
)


def clamp_navigation_steps(action, steps=1):
    """Return a legal repetition count for a canonical navigation action."""
    if action not in MOVEMENT_ACTIONS:
        return 1
    if isinstance(steps, bool):
        steps = 1
    try:
        steps = int(steps)
    except (TypeError, ValueError):
        steps = 1
    return max(1, min(5, steps))


def parse_navigation_invocation(raw_output):
    """Parse and normalize the first navigation invocation in model output."""
    parsed = extract_json_object(raw_output)
    if parsed is None:
        return None
    action = parsed.get("action")
    if not isinstance(action, str):
        return None
    action = action.strip().lower()
    if action not in NAV_ACTION_TO_INDEX:
        return None
    return {
        "action": action,
        "steps": clamp_navigation_steps(action, parsed.get("steps", 1)),
    }


class NavigationHarnessPlanner(HarnessPlanner):
    """Harness planner constrained to the eight EB-Navigation actions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.system_prompt = (
            "You are a closed-loop indoor navigation planner. Choose exactly one "
            "action from this fixed vocabulary: move_forward, move_backward, "
            "move_right, move_left, turn_right, turn_left, tilt_up, tilt_down. "
            "Reply with one JSON object only: {\"action\": \"<name>\", "
            "\"steps\": <integer 1..5>}. steps may be greater than 1 only for "
            "the four move actions; rotations and tilts always execute once. "
            "Use execution feedback and distance to correct the route.\n\n"
            + self.global_memory.render()
        )

    def act(self, instruction, feedback, history):
        self.planner_steps += 1
        turn_prompt = "\n".join(
            [
                f"Instruction: {instruction}",
                "Latest structured feedback:",
                json.dumps(feedback, ensure_ascii=False, sort_keys=True),
                "Recent history:",
                json.dumps(history[-6:], ensure_ascii=False, sort_keys=True),
                "Return the next navigation action as JSON.",
            ]
        )
        raw_output = self._chat(turn_prompt)
        invocation = parse_navigation_invocation(raw_output)
        if invocation is None:
            self.output_json_error += 1
        return invocation, raw_output