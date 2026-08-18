"""Fixed primitive library for the Harness VLA beta on EB-Manipulation.

The EB-Manipulation environment consumes a 7-D *discrete* action
``[X, Y, Z, Roll, Pitch, Yaw, Gripper]`` where:

* ``X, Y, Z`` are voxel indices in ``[0, VOXEL_SIZE]`` (``VOXEL_SIZE == 100``);
* ``Roll, Pitch, Yaw`` are discrete Euler bins in ``[0, 360 / ROTATION_RESOLUTION]``
  (``ROTATION_RESOLUTION == 3`` degrees per bin, so bins live in ``[0, 120]``);
* ``Gripper`` is ``1`` for *open* and ``0`` for *closed*
  (matches ``is_gripper_open = discrete_action[6]`` in ``eb_man_utils``).

Object coordinates handed to the planner by ``form_object_coord_for_input`` are
already expressed in voxel indices (``{"object 1": [vx, vy, vz], ...}``), so the
primitives operate directly in that space.

Design goals for the beta:

* Each primitive *compiles* to one or more discrete actions. Compilation is a
  pure function of the current end-effector :class:`PoseState`, the primitive
  arguments, and the object coordinate table. This keeps the whole library unit
  testable without launching the (heavy) simulator.
* The single contact-rich primitive ``vla_act`` is a *mock scripted* stand-in for
  a frozen VLA: it expands a high-level ``grasp``/``place``/``push`` intent into a
  short burst of analytic sub-actions. In the full framework this would be a
  frozen VLA policy; here it validates the harness architecture end to end.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Kept in sync with embodiedbench.envs.eb_manipulation.eb_man_utils. Imported
# lazily / defensively so the primitive library can be unit tested without the
# manipulation environment's heavy optional dependencies (pyrep, ultralytics).
try:  # pragma: no cover - exercised indirectly when the env is installed
    from embodiedbench.envs.eb_manipulation.eb_man_utils import (
        ROTATION_RESOLUTION as _ROTATION_RESOLUTION,
        VOXEL_SIZE as _VOXEL_SIZE,
    )
    ROTATION_RESOLUTION = int(_ROTATION_RESOLUTION)
    VOXEL_SIZE = int(_VOXEL_SIZE)
except Exception:  # pragma: no cover - fallback for isolated unit tests
    ROTATION_RESOLUTION = 3
    VOXEL_SIZE = 100

# Discrete rotation bins span [0, 360 / ROTATION_RESOLUTION].
ROT_MAX = 360 // ROTATION_RESOLUTION
GRIPPER_OPEN = 1
GRIPPER_CLOSED = 0

# Canonical primitive names (module-level so parsing helpers can reference them
# without instantiating the library).
ANALYTIC_PRIMITIVE_NAMES: Tuple[str, ...] = (
    "move_to",
    "rotate_wrist",
    "rotate_pitch",
    "set_gripper",
    "release",
)
CONTACT_PRIMITIVE_NAME = "vla_act"
PRIMITIVE_NAMES: Tuple[str, ...] = ANALYTIC_PRIMITIVE_NAMES + (CONTACT_PRIMITIVE_NAME,)

# A conservative top-down-ish neutral orientation used when no orientation is
# known yet. Values are discrete Euler bins; they can be tuned per task without
# changing the harness architecture.
NEUTRAL_ORIENTATION: Tuple[int, int, int] = (60, 60, 60)

# Voxels the gripper is lifted above a target before/after contact (staging).
DEFAULT_APPROACH_DZ = 8
# Voxels used for a short post-grasp lift to expose empty grasps.
DEFAULT_LIFT_DZ = 6

DiscreteAction = List[int]


class PrimitiveError(ValueError):
    """Raised when a primitive invocation cannot be compiled to actions."""


def _physical_object_name(sim_name: str) -> str:
    return sim_name.replace("_visual", "")


def classify_grasp_outcome(
    target_object_id: str,
    grasped_object_names: Optional[Sequence[str]] = None,
    object_position_at_close: Optional[Sequence[float]] = None,
    object_position_after_lift: Optional[Sequence[float]] = None,
    gripper_position_at_close: Optional[Sequence[float]] = None,
    gripper_position_after_lift: Optional[Sequence[float]] = None,
    object_lift_threshold: float = 3.0,
    max_gripper_object_distance: float = 8.0,
    empty_object_motion_threshold: float = 1.0,
    min_gripper_lift: float = 3.0,
    max_comotion_residual: float = 2.0,
    target_sim_name: Optional[str] = None,
) -> Dict:
    """Classify a grasp without conflating action execution with grasp success.

    Simulator attachment is authoritative when present.  If no attachment is
    reported, a conservative geometric fallback checks that the target moved up
    with the lifting gripper.  Missing or ambiguous evidence stays unverified.
    Positions may be world coordinates or voxels, provided thresholds use the
    same units.
    """
    target = str(target_object_id).strip()
    target_sim = str(target_sim_name).strip() if target_sim_name is not None else None
    attached = [str(name).strip() for name in (grasped_object_names or [])]
    metrics = {
        "target_object_id": target,
        "target_sim_name": target_sim,
        "grasped_object_names": attached,
        "matched_grasped_object_name": None,
        "object_lift": None,
        "object_displacement": None,
        "gripper_lift": None,
        "gripper_object_distance": None,
        "comotion_residual": None,
        "geometry_consistent_with_attachment": None,
        "classification_source": "insufficient_evidence",
        "reason": "missing_geometric_evidence",
    }

    def _xyz(value):
        if value is None or len(value) < 3:
            return None
        try:
            return [float(value[i]) for i in range(3)]
        except (TypeError, ValueError):
            return None

    obj_close = _xyz(object_position_at_close)
    obj_after = _xyz(object_position_after_lift)
    grip_close = _xyz(gripper_position_at_close)
    grip_after = _xyz(gripper_position_after_lift)
    def _distance(a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    geometry_outcome = None
    if None not in (obj_close, obj_after, grip_close, grip_after):
        metrics.update({
            "object_lift": obj_after[2] - obj_close[2],
            "object_displacement": _distance(obj_close, obj_after),
            "gripper_lift": grip_after[2] - grip_close[2],
            "gripper_object_distance": _distance(grip_after, obj_after),
            "comotion_residual": _distance(
                [obj_after[i] - obj_close[i] for i in range(3)],
                [grip_after[i] - grip_close[i] for i in range(3)],
            ),
        })
        if (
            metrics["object_lift"] >= object_lift_threshold
            and metrics["gripper_lift"] >= min_gripper_lift
            and metrics["gripper_object_distance"] <= max_gripper_object_distance
            and metrics["comotion_residual"] <= max_comotion_residual
        ):
            geometry_outcome = "grasp_verified"
        elif (
            metrics["gripper_lift"] >= min_gripper_lift
            and metrics["object_displacement"] <= empty_object_motion_threshold
        ):
            geometry_outcome = "empty_grasp"
        else:
            geometry_outcome = "grasp_unverified"

    if attached and target_sim:
        target_physical_name = _physical_object_name(target_sim)
        matched_name = next(
            (
                name
                for name in attached
                if _physical_object_name(name) == target_physical_name
            ),
            None,
        )
        target_attached = matched_name is not None
        metrics["matched_grasped_object_name"] = matched_name
        metrics["classification_source"] = "attachment"
        if geometry_outcome is not None:
            metrics["geometry_consistent_with_attachment"] = (
                (geometry_outcome == "grasp_verified") == target_attached
            )
        if target_attached:
            metrics.update(outcome="grasp_verified", reason="target_attached")
        else:
            metrics.update(outcome="grasp_unverified", reason="wrong_object_attached")
    elif geometry_outcome == "grasp_verified":
        metrics.update(
            outcome=geometry_outcome,
            reason="target_lifted_with_gripper",
            classification_source="geometry",
        )
    elif geometry_outcome == "empty_grasp":
        metrics.update(
            outcome=geometry_outcome,
            reason="gripper_lifted_without_target",
            classification_source="geometry",
        )
    elif geometry_outcome is not None:
        metrics.update(
            outcome=geometry_outcome,
            reason="ambiguous_geometry",
            classification_source="geometry",
        )
    else:
        metrics["outcome"] = "grasp_unverified"
    return metrics


def classify_spatial_postcondition(
    expected_position: Optional[Sequence[float]],
    observed_position: Optional[Sequence[float]],
    tolerance: float,
) -> Dict:
    """Classify whether an observed position reaches an expected position."""
    result = {
        "expected_position": list(expected_position) if expected_position is not None else None,
        "observed_position": list(observed_position) if observed_position is not None else None,
        "tolerance": float(tolerance),
        "distance": None,
        "postcondition_met": None,
        "reason": "missing_spatial_evidence",
    }
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if expected_position is None or observed_position is None:
        return result
    if len(expected_position) < 3 or len(observed_position) < 3:
        return result
    try:
        distance = math.sqrt(sum(
            (float(observed_position[index]) - float(expected_position[index])) ** 2
            for index in range(3)
        ))
    except (TypeError, ValueError):
        return result
    result["distance"] = distance
    result["postcondition_met"] = distance <= tolerance
    result["reason"] = "within_tolerance" if result["postcondition_met"] else "outside_tolerance"
    return result


def summarize_physical_state(
    object_roles: Dict[str, Sequence[str]],
    held_object_id: Optional[str],
    placed_object_ids: Sequence[str],
) -> Dict:
    """Return deterministic held, placed, and remaining manipulable IDs."""
    manipulable = {
        object_id
        for object_id, roles in object_roles.items()
        if "manipulable" in roles
    }
    placed = set(placed_object_ids) & manipulable
    held = held_object_id if held_object_id in manipulable else None
    remaining = manipulable - placed
    if held is not None:
        remaining.discard(held)
    return {
        "held": held,
        "placed": sorted(placed),
        "remaining": sorted(remaining),
    }


def reconcile_held_object(
    held_object_id: Optional[str],
    attachments: Sequence[str],
    attachment_available: bool,
    id_to_sim_name: Dict[str, str],
) -> Tuple[Optional[str], bool]:
    """Re-read authoritative attachment state after a primitive executes.

    Returns ``(held_object_id, evidence_available)``. When attachment evidence
    is available, the tracked held object must correspond to a currently
    attached body; otherwise it is cleared (a detach happened). If nothing is
    tracked but an attachment exists, the matching object ID is adopted.
    """
    if not attachment_available:
        return held_object_id, False
    attached_physical = {_physical_object_name(name) for name in attachments}
    if held_object_id is not None:
        sim_name = id_to_sim_name.get(held_object_id)
        if sim_name is None or _physical_object_name(sim_name) not in attached_physical:
            held_object_id = None
    if held_object_id is None and attached_physical:
        for object_id in sorted(id_to_sim_name):
            if _physical_object_name(id_to_sim_name[object_id]) in attached_physical:
                held_object_id = object_id
                break
    return held_object_id, True


def primitive_termination(
    mode: Optional[str],
    grasp_outcome: Optional[str] = None,
    env_done: bool = False,
    release_executed: bool = False,
    attachment_evidence_available: bool = False,
    grasped_object_names: Optional[Sequence[str]] = None,
    primitive_name: Optional[str] = None,
    spatial_postcondition_met: Optional[bool] = None,
) -> Tuple[str, bool]:
    """Return a primitive termination reason and its postcondition status."""
    if primitive_name == "move_to":
        if spatial_postcondition_met is True:
            return "postcondition_met", True
        if spatial_postcondition_met is False:
            return "target_pose_not_reached", False
        return "unverified", False
    if primitive_name == "release":
        if env_done and not release_executed:
            return "environment_terminated_before_release", False
        if release_executed and attachment_evidence_available:
            if not list(grasped_object_names or []):
                return "postcondition_met", True
            return "unverified", False
        return "unverified", False
    if mode == "grasp":
        if grasp_outcome == "grasp_verified":
            return "postcondition_met", True
        if grasp_outcome == "empty_grasp":
            return "empty_grasp", False
        return "unverified", False
    if mode == "place":
        if env_done and not release_executed:
            return "environment_terminated_before_release", False
        if release_executed and attachment_evidence_available:
            if list(grasped_object_names or []):
                return "unverified", False
            if spatial_postcondition_met is True:
                return "postcondition_met", True
            if spatial_postcondition_met is False:
                return "released_outside_destination_tolerance", False
        return "unverified", False
    return "postcondition_met", True


@dataclass
class PoseState:
    """Current end-effector state in the environment's discrete voxel space."""

    x: int = VOXEL_SIZE // 2
    y: int = VOXEL_SIZE // 2
    z: int = VOXEL_SIZE // 2
    roll: int = NEUTRAL_ORIENTATION[0]
    pitch: int = NEUTRAL_ORIENTATION[1]
    yaw: int = NEUTRAL_ORIENTATION[2]
    gripper: int = GRIPPER_OPEN

    def as_action(self) -> DiscreteAction:
        return [
            int(self.x),
            int(self.y),
            int(self.z),
            int(self.roll),
            int(self.pitch),
            int(self.yaw),
            int(self.gripper),
        ]

    def copy(self) -> "PoseState":
        return copy.copy(self)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def _clamp_xyz(xyz: Sequence[float]) -> List[int]:
    if len(xyz) != 3:
        raise PrimitiveError(f"xyz must have 3 elements, got {xyz!r}")
    return [_clamp(v, 0, VOXEL_SIZE) for v in xyz]


def _clamp_rot(value: float) -> int:
    return _clamp(value, 0, ROT_MAX)


def _normalize_gripper(value: Union[int, str, None], default: int) -> int:
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("open", "opened", "release", "1"):
            return GRIPPER_OPEN
        if v in ("close", "closed", "grip", "grasp", "0"):
            return GRIPPER_CLOSED
        raise PrimitiveError(f"Unknown gripper state: {value!r}")
    if int(value) in (GRIPPER_OPEN, GRIPPER_CLOSED):
        return int(value)
    raise PrimitiveError(f"Gripper must be 0/1 or open/close, got {value!r}")


def normalize_invocation(obj: object) -> Optional[Dict]:
    """Coerce a variety of model output shapes into a canonical invocation.

    Canonical form is ``{"action": <name>, ...args}``. Accepted inputs:

    * ``{"action": <name>, ...}`` (already canonical);
    * ``{"reasoning": ..., "action": {...}}`` (nested — unwrapped recursively);
    * ``{"<primitive_name>": {...args}}`` (name-as-key, common with small models).

    Returns ``None`` if no single primitive can be identified.
    """
    if not isinstance(obj, dict):
        return None
    if "action" in obj:
        val = obj["action"]
        if isinstance(val, str):
            # Bare canonical form; drop non-argument bookkeeping fields.
            return {k: v for k, v in obj.items() if k != "reasoning"}
        if isinstance(val, dict):
            return normalize_invocation(val)
        return None
    # Name-as-key form: exactly one key naming a known primitive.
    prim_keys = [k for k in obj if k in PRIMITIVE_NAMES]
    if len(prim_keys) == 1:
        name = prim_keys[0]
        args = obj[name]
        if isinstance(args, dict):
            return {"action": name, **args}
        return {"action": name}
    return None


@dataclass
class PrimitiveResult:
    """Outcome of compiling a primitive invocation.

    Attributes
    ----------
    name:
        Canonical primitive name.
    actions:
        Ordered list of discrete 7-D actions to feed to ``env.step``.
    end_pose:
        The intended pose after all actions execute (best-effort bookkeeping).
    is_contact:
        Whether this primitive is the contact-rich ``vla_act`` primitive.
    meta:
        Extra structured info recorded into the audit trace.
    """

    name: str
    actions: List[DiscreteAction]
    end_pose: PoseState
    is_contact: bool = False
    meta: Dict = field(default_factory=dict)


class PrimitiveLibrary:
    """Compiles high-level primitive invocations into discrete env actions.

    The library is *fixed*: the planner may only invoke the primitives declared
    in :attr:`PRIMITIVES`. It cannot invent new primitives at deployment time,
    matching the Harness VLA constraint.
    """

    #: Canonical analytic primitives available on EB-Manipulation.
    ANALYTIC_PRIMITIVES = ANALYTIC_PRIMITIVE_NAMES
    #: The single learned contact-rich primitive (mock scripted here).
    CONTACT_PRIMITIVE = CONTACT_PRIMITIVE_NAME
    PRIMITIVES = PRIMITIVE_NAMES

    def __init__(
        self,
        approach_dz: int = DEFAULT_APPROACH_DZ,
        lift_dz: int = DEFAULT_LIFT_DZ,
    ) -> None:
        self.approach_dz = int(approach_dz)
        self.lift_dz = int(lift_dz)

    # -- target resolution ------------------------------------------------

    @staticmethod
    def resolve_target(
        target: Union[str, Sequence[float]],
        object_coords: Optional[Dict[str, Sequence[float]]],
    ) -> List[int]:
        """Resolve a primitive target to voxel ``[x, y, z]``.

        ``target`` may be an explicit ``[x, y, z]`` list or an object key present
        in ``object_coords`` (e.g. ``"object 1"``).
        """
        if isinstance(target, (list, tuple)):
            return _clamp_xyz(target)
        if not isinstance(target, str):
            raise PrimitiveError(f"target must be a name or [x,y,z], got {target!r}")
        if not object_coords:
            raise PrimitiveError(
                f"target {target!r} is a name but no object coordinates were provided"
            )
        key = target.strip()
        if key in object_coords:
            return _clamp_xyz(object_coords[key])
        # Tolerant matching: 'object1' -> 'object 1', case-insensitive.
        normalized = {k.replace(" ", "").lower(): k for k in object_coords}
        probe = key.replace(" ", "").lower()
        if probe in normalized:
            return _clamp_xyz(object_coords[normalized[probe]])
        raise PrimitiveError(
            f"Unknown target {target!r}; known objects: {sorted(object_coords)}"
        )

    # -- primitive compilation -------------------------------------------

    def compile(
        self,
        invocation: Dict,
        pose: PoseState,
        object_coords: Optional[Dict[str, Sequence[float]]] = None,
    ) -> PrimitiveResult:
        """Compile a JSON-style primitive invocation to discrete actions.

        Parameters
        ----------
        invocation:
            Dict with an ``action`` field naming the primitive plus its
            arguments (e.g. ``{"action": "move_to", "xyz": [50, 60, 20]}``).
        pose:
            Current end-effector pose; used to fill unspecified dimensions.
        object_coords:
            Mapping of object names to voxel coordinates for target resolution.
        """
        if not isinstance(invocation, dict):
            raise PrimitiveError(f"invocation must be a dict, got {type(invocation)}")
        normalized = normalize_invocation(invocation)
        if normalized is None:
            raise PrimitiveError(f"could not identify a primitive in {invocation!r}")
        invocation = normalized
        name = invocation.get("action")
        if name is None:
            raise PrimitiveError("invocation missing 'action' field")
        name = str(name).strip()
        if name not in self.PRIMITIVES:
            raise PrimitiveError(
                f"Unknown primitive {name!r}; allowed: {self.PRIMITIVES}"
            )
        handler = getattr(self, f"_compile_{name}")
        return handler(invocation, pose.copy(), object_coords)

    def _compile_move_to(self, inv, pose, object_coords) -> PrimitiveResult:
        target = inv.get("xyz", inv.get("target"))
        if target is None:
            raise PrimitiveError("move_to requires 'xyz' or 'target'")
        x, y, z = self.resolve_target(target, object_coords)
        gripper = _normalize_gripper(inv.get("gripper"), pose.gripper)
        end = pose.copy()
        end.x, end.y, end.z, end.gripper = x, y, z, gripper
        return PrimitiveResult("move_to", [end.as_action()], end)

    def _compile_rotate_wrist(self, inv, pose, object_coords) -> PrimitiveResult:
        if "target_yaw" not in inv and "yaw" not in inv:
            raise PrimitiveError("rotate_wrist requires 'target_yaw'")
        yaw = _clamp_rot(inv.get("target_yaw", inv.get("yaw")))
        end = pose.copy()
        end.yaw = yaw
        return PrimitiveResult("rotate_wrist", [end.as_action()], end)

    def _compile_rotate_pitch(self, inv, pose, object_coords) -> PrimitiveResult:
        if "target_pitch" not in inv and "pitch" not in inv:
            raise PrimitiveError("rotate_pitch requires 'target_pitch'")
        pitch = _clamp_rot(inv.get("target_pitch", inv.get("pitch")))
        end = pose.copy()
        end.pitch = pitch
        return PrimitiveResult("rotate_pitch", [end.as_action()], end)

    def _compile_set_gripper(self, inv, pose, object_coords) -> PrimitiveResult:
        state = inv.get("gripper", inv.get("state"))
        if state is None:
            raise PrimitiveError("set_gripper requires 'gripper' (open/close)")
        gripper = _normalize_gripper(state, pose.gripper)
        end = pose.copy()
        end.gripper = gripper
        return PrimitiveResult("set_gripper", [end.as_action()], end)

    def _compile_release(self, inv, pose, object_coords) -> PrimitiveResult:
        end = pose.copy()
        end.gripper = GRIPPER_OPEN
        actions = [end.as_action()]
        # Optional short retreat lift to clear the released object.
        if inv.get("lift", False):
            lifted = end.copy()
            lifted.z = _clamp(lifted.z + self.lift_dz, 0, VOXEL_SIZE)
            actions.append(lifted.as_action())
            end = lifted
        return PrimitiveResult("release", actions, end)

    def _compile_vla_act(self, inv, pose, object_coords) -> PrimitiveResult:
        """Mock scripted stand-in for a frozen contact-rich VLA policy.

        Supported modes:

        * ``grasp``: stage above the target (open), descend onto it (open),
          close the gripper, then a short lift to expose empty grasps.
        * ``place``: stage above the destination (closed), descend, open to
          release.
        * ``push``: descend onto the target and translate toward an optional
          ``direction`` offset while keeping the gripper closed.
        """
        mode = str(inv.get("mode", inv.get("prompt", "grasp"))).strip().lower()
        # Allow natural-language prompts to imply a mode.
        if "place" in mode or "put" in mode or "drop" in mode:
            mode = "place"
        elif "push" in mode or "wipe" in mode:
            mode = "push"
        else:
            mode = "grasp"

        object_id = inv.get("object")
        destination_id = inv.get("destination")
        if mode == "grasp":
            target = object_id if object_id is not None else inv.get("target", inv.get("xyz"))
        elif mode == "place":
            if object_id is None or destination_id is None:
                raise PrimitiveError("place requires both 'object' and 'destination'; legacy 'target' is not allowed")
            if str(object_id).strip() == str(destination_id).strip():
                raise PrimitiveError("place object and destination must be different")
            target = destination_id
        else:
            # Legacy target/xyz remains accepted for push (and above for grasp).
            target = object_id if object_id is not None else inv.get("target", inv.get("xyz"))
        if target is None:
            raise PrimitiveError("vla_act requires an object, target, or xyz")
        x, y, z = self.resolve_target(target, object_coords)
        approach_dz = int(inv.get("approach_dz", self.approach_dz))
        lift_dz = int(inv.get("lift_dz", self.lift_dz))

        actions: List[DiscreteAction] = []
        cur = pose.copy()

        if mode == "grasp":
            above = cur.copy()
            above.x, above.y = x, y
            above.z = _clamp(z + approach_dz, 0, VOXEL_SIZE)
            above.gripper = GRIPPER_OPEN
            actions.append(above.as_action())

            on = above.copy()
            on.z = z
            on.gripper = GRIPPER_OPEN
            actions.append(on.as_action())

            close = on.copy()
            close.gripper = GRIPPER_CLOSED
            actions.append(close.as_action())

            lift = close.copy()
            lift.z = _clamp(z + lift_dz, 0, VOXEL_SIZE)
            actions.append(lift.as_action())
            end = lift
            meta = {
                "mode": "grasp",
                "target_voxel": [x, y, z],
                "object_id": object_id if object_id is not None else target,
                "destination_id": None,
            }

        elif mode == "place":
            above = cur.copy()
            above.x, above.y = x, y
            above.z = _clamp(z + approach_dz, 0, VOXEL_SIZE)
            above.gripper = GRIPPER_CLOSED
            actions.append(above.as_action())

            on = above.copy()
            on.z = z
            on.gripper = GRIPPER_CLOSED
            actions.append(on.as_action())

            release = on.copy()
            release.gripper = GRIPPER_OPEN
            actions.append(release.as_action())
            end = release
            meta = {
                "mode": "place",
                "target_voxel": [x, y, z],
                "object_id": object_id,
                "destination_id": destination_id,
                "canonical_contract": True,
            }

        else:  # push
            direction = inv.get("direction", [0, 0, 0])
            if len(direction) != 3:
                raise PrimitiveError("push direction must be [dx, dy, dz]")
            on = cur.copy()
            on.x, on.y, on.z = x, y, z
            on.gripper = GRIPPER_CLOSED
            actions.append(on.as_action())

            pushed = on.copy()
            pushed.x = _clamp(x + direction[0], 0, VOXEL_SIZE)
            pushed.y = _clamp(y + direction[1], 0, VOXEL_SIZE)
            pushed.z = _clamp(z + direction[2], 0, VOXEL_SIZE)
            actions.append(pushed.as_action())
            end = pushed
            meta = {
                "mode": "push",
                "target_voxel": [x, y, z],
                "direction": direction,
                "object_id": object_id if object_id is not None else target,
                "destination_id": None,
            }

        return PrimitiveResult("vla_act", actions, end, is_contact=True, meta=meta)


def pose_from_observation(obs: Dict) -> Optional[PoseState]:
    """Best-effort extraction of the current EE pose in voxel space from an obs.

    EB-Manipulation observations expose ``gripper_pose`` (world xyz + quat) and
    ``gripper_open``. We convert the world position to a voxel index and the
    quaternion to discrete Euler bins so primitives can preserve unspecified
    dimensions. Returns ``None`` if the observation lacks the needed fields.
    """
    if not isinstance(obs, dict):
        return None
    gripper_pose = obs.get("gripper_pose")
    if gripper_pose is None:
        return None
    try:  # Imported lazily to avoid hard dependency during unit tests.
        import numpy as np
        from scipy.spatial.transform import Rotation
        from embodiedbench.envs.eb_manipulation.eb_man_utils import (
            point_to_voxel_index,
            ROTATION_RESOLUTION as RR,
        )

        pos = np.asarray(gripper_pose[:3], dtype=float)
        voxel = list(point_to_voxel_index(pos))
        quat = np.asarray(gripper_pose[3:7], dtype=float)
        euler = Rotation.from_quat(quat).as_euler("xyz", degrees=True)
        disc = [(int(round((e + 180) / RR))) for e in euler]
        gripper_open = obs.get("gripper_open", 1.0)
        gripper = GRIPPER_OPEN if float(gripper_open) >= 0.5 else GRIPPER_CLOSED
        return PoseState(
            x=_clamp(voxel[0], 0, VOXEL_SIZE),
            y=_clamp(voxel[1], 0, VOXEL_SIZE),
            z=_clamp(voxel[2], 0, VOXEL_SIZE),
            roll=_clamp_rot(disc[0]),
            pitch=_clamp_rot(disc[1]),
            yaw=_clamp_rot(disc[2]),
            gripper=gripper,
        )
    except Exception:
        return None
