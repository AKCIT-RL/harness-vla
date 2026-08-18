"""Local stop predicates for native LIBERO ``vla_act`` execution."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict


class LiberoTauError(ValueError):
    """Raised when local physical evidence is invalid or incomplete."""


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise LiberoTauError("%s must be finite" % name)
    return number


def read_bilateral_contact(env, target_instance: str) -> Dict[str, Any]:
    """Read native robosuite finger-pad contact for one exact LIBERO instance."""
    if not isinstance(target_instance, str) or not target_instance.strip():
        raise LiberoTauError("target_instance must be a non-empty string")
    inner = getattr(env, "env", None)
    if inner is None:
        raise LiberoTauError("LIBERO wrapper does not expose its inner environment")
    matches = [
        model
        for model in getattr(inner, "objects", ())
        if getattr(model, "name", None) == target_instance
    ]
    if len(matches) != 1:
        raise LiberoTauError(
            "target_instance must resolve to exactly one LIBERO object"
        )
    robots = getattr(inner, "robots", ())
    if len(robots) != 1:
        raise LiberoTauError("native LIBERO contact requires exactly one robot")
    important_geoms = getattr(robots[0].gripper, "important_geoms", {})
    left_geoms = important_geoms.get("left_fingerpad")
    right_geoms = important_geoms.get("right_fingerpad")
    object_geoms = getattr(matches[0], "contact_geoms", None)
    if not left_geoms or not right_geoms or not object_geoms:
        raise LiberoTauError("contact geometry groups are incomplete")
    checker = getattr(inner, "check_contact", None)
    if not callable(checker):
        raise LiberoTauError("inner environment does not expose check_contact")
    left = bool(checker(left_geoms, object_geoms))
    right = bool(checker(right_geoms, object_geoms))
    return {
        "target_instance": target_instance,
        "left_finger_contact": left,
        "right_finger_contact": right,
        "bilateral_contact": left and right,
        "source": "robosuite_check_contact",
        "privileged_contact_state": True,
    }


@dataclass(frozen=True)
class LiftAndGraspTau:
    """Evaluate the paper-confirmed lift-and-grasp stop-predicate role.

    The concrete bilateral-contact signal and lift threshold are
    benchmark-specific, paper-compatible implementation choices.
    """

    baseline_target_z_m: float
    minimum_lift_m: float

    def __post_init__(self):
        _finite(self.baseline_target_z_m, "baseline_target_z_m")
        threshold = _finite(self.minimum_lift_m, "minimum_lift_m")
        if threshold <= 0.0:
            raise LiberoTauError("minimum_lift_m must be greater than zero")

    def evaluate(
        self,
        *,
        current_target_z_m: float,
        left_finger_contact: bool,
        right_finger_contact: bool,
    ) -> Dict[str, Any]:
        current = _finite(current_target_z_m, "current_target_z_m")
        if not isinstance(left_finger_contact, bool) or not isinstance(
            right_finger_contact, bool
        ):
            raise LiberoTauError("finger contact evidence must be boolean")
        lift = current - float(self.baseline_target_z_m)
        bilateral_contact = left_finger_contact and right_finger_contact
        lift_met = lift >= float(self.minimum_lift_m)
        return {
            "predicate": "lift_and_grasp",
            "tau_satisfied": bilateral_contact and lift_met,
            "contact": {
                "left_finger": left_finger_contact,
                "right_finger": right_finger_contact,
                "bilateral": bilateral_contact,
            },
            "lift": {
                "baseline_target_z_m": float(self.baseline_target_z_m),
                "current_target_z_m": current,
                "delta_z_m": lift,
                "minimum_lift_m": float(self.minimum_lift_m),
                "threshold_met": lift_met,
                "coordinate_source": "rgbd_projection",
            },
            "task_success_evaluated": False,
        }


__all__ = ["LiberoTauError", "LiftAndGraspTau", "read_bilateral_contact"]