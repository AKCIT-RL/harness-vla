"""Optional beta-only diagnostics from privileged native LIBERO state."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence


def _error(exc: Exception) -> Dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


class LiberoPrivilegedDiagnostics:
    """Read-only snapshots that never participate in policy execution."""

    def __init__(
        self,
        env,
        *,
        available_targets: Sequence[str],
        object_roles: Optional[Mapping[str, Sequence[str]]] = None,
    ) -> None:
        self.env = env
        self.targets = tuple(available_targets)
        self.roles = dict(object_roles or {})
        self.held_object = None

    def __call__(self, event: str, invocation: Dict[str, Any], holding: Optional[str]):
        if event == "post_grasp" and holding is not None:
            self.held_object = holding
        held_name = holding or self.held_object
        destination_name = self._destination(invocation, held_name)
        snapshot = {
            "event": event,
            "source": "privileged_mujoco_state",
            "beta_only": True,
            "eef_pose": self._read(self._eef_pose),
            "holding": self._entity(held_name),
            "destination": self._entity(destination_name),
            "delta_xy_m": None,
            "contact": self._read(self._contact, held_name),
            "on_predicate": self._on_predicate(held_name, destination_name),
        }
        held_pose = snapshot["holding"].get("pose")
        destination_pose = snapshot["destination"].get("pose")
        if held_pose is not None and destination_pose is not None:
            delta_x = held_pose["position"][0] - destination_pose["position"][0]
            delta_y = held_pose["position"][1] - destination_pose["position"][1]
            snapshot["delta_xy_m"] = math.hypot(delta_x, delta_y)
        if event == "post_release":
            self.held_object = None
        return snapshot

    def _read(self, reader, *args):
        try:
            return reader(*args)
        except Exception as exc:
            return {"value": None, "error": _error(exc)}

    def _inner(self):
        inner = getattr(self.env, "env", None)
        if inner is None:
            raise ValueError("LIBERO wrapper does not expose its inner environment")
        return inner

    def _object(self, name: Optional[str]):
        if name is None:
            raise ValueError("object name is unavailable")
        matches = [
            obj
            for obj in getattr(self._inner(), "objects", ())
            if getattr(obj, "name", None) == name
        ]
        if len(matches) != 1:
            raise ValueError("object must resolve to exactly one LIBERO model: %s" % name)
        return matches[0]

    def _destination(self, invocation, held_name):
        target = invocation.get("target")
        if target != held_name and target in self.targets:
            return target
        candidates = [
            name for name in self.targets if "destination" in self.roles.get(name, ())
        ]
        return candidates[0] if len(candidates) == 1 else None

    def _body_pose(self, obj):
        sim = self._inner().sim
        body_name = getattr(obj, "root_body", None)
        if not body_name:
            raise ValueError("LIBERO object does not expose root_body")
        body_id = sim.model.body_name2id(body_name)
        position = [float(value) for value in sim.data.body_xpos[body_id]]
        quaternion = [float(value) for value in sim.data.body_xquat[body_id]]
        return {"position": position, "quaternion_wxyz": quaternion}

    def _entity(self, name):
        if name is None:
            return {"name": None, "pose": None, "error": {"type": "ValueError", "message": "object name is unavailable"}}
        try:
            return {"name": name, "pose": self._body_pose(self._object(name)), "error": None}
        except Exception as exc:
            return {"name": name, "pose": None, "error": _error(exc)}

    def _eef_pose(self):
        inner = self._inner()
        robots = getattr(inner, "robots", ())
        if len(robots) != 1:
            raise ValueError("native LIBERO diagnostics require exactly one robot")
        robot = robots[0]
        site_id = getattr(robot, "eef_site_id", None)
        if site_id is None:
            raise ValueError("LIBERO robot does not expose eef_site_id")
        position = [float(value) for value in inner.sim.data.site_xpos[site_id]]
        matrix = [float(value) for value in inner.sim.data.site_xmat[site_id]]
        return {"position": position, "rotation_matrix": matrix}

    def _contact(self, held_name):
        inner = self._inner()
        held = self._object(held_name)
        robots = getattr(inner, "robots", ())
        if len(robots) != 1:
            raise ValueError("native LIBERO diagnostics require exactly one robot")
        geoms = getattr(robots[0].gripper, "important_geoms", {})
        object_geoms = getattr(held, "contact_geoms", None)
        checker = getattr(inner, "check_contact", None)
        if not callable(checker) or not object_geoms:
            raise ValueError("LIBERO contact API is incomplete")
        left = bool(checker(geoms.get("left_fingerpad"), object_geoms))
        right = bool(checker(geoms.get("right_fingerpad"), object_geoms))
        return {"left_finger": left, "right_finger": right, "bilateral": left and right}

    def _on_predicate(self, held_name, destination_name):
        clauses = {
            "object_above_destination": None,
            "horizontal_center_within_0_03_m": None,
            "objects_in_contact": None,
        }
        errors = {}
        held = destination = None
        held_pose = destination_pose = None
        try:
            held = self._object(held_name)
            held_pose = self._body_pose(held)
            destination = self._object(destination_name)
            destination_pose = self._body_pose(destination)
        except Exception as exc:
            errors["objects"] = _error(exc)
        if held_pose is not None and destination_pose is not None:
            clauses["object_above_destination"] = (
                held_pose["position"][2] >= destination_pose["position"][2]
            )
            try:
                delta_x = held_pose["position"][0] - destination_pose["position"][0]
                delta_y = held_pose["position"][1] - destination_pose["position"][1]
                clauses["horizontal_center_within_0_03_m"] = (
                    math.hypot(delta_x, delta_y) < 0.03
                )
            except Exception as exc:
                errors["horizontal_center_within_0_03_m"] = _error(exc)
        if held is not None and destination is not None:
            try:
                checker = getattr(self._inner(), "check_contact", None)
                if not callable(checker):
                    raise ValueError("inner environment does not expose check_contact")
                clauses["objects_in_contact"] = bool(checker(held, destination))
            except Exception as exc:
                errors["objects_in_contact"] = _error(exc)
        values = tuple(clauses.values())
        return {
            "predicate": "On",
            "clauses": clauses,
            "value": all(values) if all(value is not None for value in values) else None,
            "errors": errors or None,
        }


__all__ = ["LiberoPrivilegedDiagnostics"]