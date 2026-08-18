"""Beta-only one-episode diagnostic for physical grasp feedback.

This runner is deliberately independent of the production evaluator and uses no
planner or LLM. It executes one canonical four-action grasp and, only after a
verified grasp, one canonical three-action place.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import traceback
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv
from embodiedbench.envs.eb_manipulation.eb_man_utils import (
    form_harness_grounding_for_input,
)
from harness_vla.primitives.primitives import (
    PoseState,
    PrimitiveLibrary,
    classify_grasp_outcome,
    pose_from_observation,
)


MAX_ENV_STEPS = 7
CAMERAS = ["front_rgb", "overhead_rgb"]
THRESHOLDS = {
    "object_lift_threshold": 3.0,
    "max_gripper_object_distance": 8.0,
    "empty_object_motion_threshold": 1.0,
    "min_gripper_lift": 3.0,
    "max_comotion_residual": 2.0,
}
RUN_DIR = Path("running/grasp_feedback_physics_1ep") / datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)
TRACE_PATH = RUN_DIR / "trace.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _obs_dict(obs: Any) -> Dict[str, Any]:
    return obs if isinstance(obs, dict) else vars(copy.deepcopy(obs))


def _ground(env: EBManEnv, obs: Any) -> Tuple[Dict, Dict, Dict, Dict]:
    coords, roles, labels, id_to_sim_name = form_harness_grounding_for_input(
        copy.deepcopy(_obs_dict(obs)), env.task_class, ["front_rgb"]
    )
    rounded = {
        object_id: [int(round(component)) for component in coord]
        for object_id, coord in coords.items()
    }
    return rounded, roles, labels, id_to_sim_name


def _pose(obs: Any) -> PoseState:
    return pose_from_observation(_obs_dict(obs)) or PoseState()


def _object_name(obj: Any) -> str:
    getter = getattr(obj, "get_name", None)
    return str(getter() if callable(getter) else getattr(obj, "name", obj))


def _physical_graspable_names(env: EBManEnv) -> Sequence[str]:
    task_environment = getattr(env, "task", None)
    task = getattr(task_environment, "_task", task_environment)
    getter = getattr(task, "get_graspable_objects", None)
    if not callable(getter):
        return []
    return [_object_name(obj) for obj in (getter() or [])]


def _alias_score(visual_name: str, physical_name: str) -> int:
    visual = visual_name.lower()
    physical = physical_name.lower()
    candidates = {
        visual,
        visual.replace("_visual", ""),
        visual.replace("visual", ""),
    }
    if physical in candidates:
        return 3
    if any(physical.startswith(candidate) or candidate.startswith(physical) for candidate in candidates):
        return 2
    visual_instance = "".join(char for char in visual if char.isdigit())
    physical_instance = "".join(char for char in physical if char.isdigit())
    visual_stem = visual.split("_visual", 1)[0]
    if visual_stem in physical and visual_instance == physical_instance:
        return 1
    return 0


def _resolve_physical_alias(
    grounded_visual_name: Optional[str], graspable_names: Iterable[str]
) -> Dict[str, Any]:
    names = list(graspable_names)
    ranked = sorted(
        ((name, _alias_score(grounded_visual_name or "", name)) for name in names),
        key=lambda item: (-item[1], item[0]),
    )
    best_score = ranked[0][1] if ranked else 0
    matches = [name for name, score in ranked if score == best_score and score > 0]
    resolved = matches[0] if len(matches) == 1 else None
    return {
        "grounded_visual_name": grounded_visual_name,
        "physical_graspable_name": resolved,
        "physical_graspable_candidates": names,
        "alias_matches": matches,
        "resolution": "unique_match" if resolved else "unresolved_or_ambiguous",
    }


def _select_entities(coords: Dict, roles: Dict) -> Tuple[str, str, Dict[str, Any]]:
    visible_manipulable = [
        object_id
        for object_id in coords
        if "manipulable" in roles.get(object_id, [])
    ]
    preferred = "object 3"
    if preferred in visible_manipulable:
        object_id = preferred
        object_policy = "preferred_object3"
    elif visible_manipulable:
        object_id = visible_manipulable[0]
        object_policy = "first_visible_manipulable_fallback"
    else:
        raise RuntimeError("No visible manipulable object is available")

    destinations = [
        object_id_candidate
        for object_id_candidate in coords
        if object_id_candidate != object_id
        and "destination" in roles.get(object_id_candidate, [])
    ]
    if not destinations:
        raise RuntimeError("No visible destination distinct from the selected object")
    return object_id, destinations[0], {
        "requested_object": preferred,
        "object_policy": object_policy,
        "visible_manipulable": visible_manipulable,
        "visible_destinations": destinations,
    }


def _attachments(env: EBManEnv, info: Optional[Dict] = None) -> Sequence[str]:
    getter = getattr(env, "get_grasped_object_names", None)
    if callable(getter):
        try:
            return list(getter())
        except Exception:
            pass
    return list((info or {}).get("grasped_objects", []) or [])


def _snapshot(
    env: EBManEnv,
    obs: Any,
    phase: str,
    subaction_index: Optional[int],
    info: Optional[Dict] = None,
) -> Dict[str, Any]:
    coords, roles, labels, id_to_sim_name = _ground(env, obs)
    pose = _pose(obs)
    image_paths = env.save_image(CAMERAS)
    return {
        "phase": phase,
        "subaction_index": subaction_index,
        "env_step": env._current_step,
        "images": image_paths,
        "coords": coords,
        "pose": pose.as_action(),
        "attachments": _attachments(env, info),
        "roles": roles,
        "labels": labels,
        "id_to_sim_name": id_to_sim_name,
    }


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    trace: Dict[str, Any] = {
        "schema": "harness-vla-grasp-feedback-physics-beta-v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "infrastructure": {
            "runner": str(Path(__file__).resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "eval_set": "base",
            "selected_indexes": [0],
            "headless": True,
            "render_mode": None,
            "max_env_steps": MAX_ENV_STEPS,
            "cameras": CAMERAS,
            "llm_used": False,
        },
        "thresholds": THRESHOLDS,
        "invocations": [],
        "snapshots": [],
        "result": {"status": "not_started"},
        "error": None,
    }
    env: Optional[EBManEnv] = None

    try:
        env = EBManEnv(
            eval_set="base",
            render_mode=None,
            img_size=(256, 256),
            down_sample_ratio=1.0,
            selected_indexes=[0],
            headless=True,
            log_path=str(RUN_DIR),
        )
        env._max_episode_steps = MAX_ENV_STEPS
        instruction, obs = env.reset()
        trace["instruction"] = instruction
        trace["episode"] = {
            "dataset_index": 0,
            "episode_number": env._current_episode_num,
            "task_class": env.task_class,
            "task_variation": env.current_task_variation,
        }
        initial = _snapshot(env, obs, "initial", None)
        trace["snapshots"].append(initial)

        coords = initial["coords"]
        roles = initial["roles"]
        object_id, destination_id, selection = _select_entities(coords, roles)
        trace["selection"] = selection
        visual_name = initial["id_to_sim_name"].get(object_id)
        alias = _resolve_physical_alias(
            visual_name, _physical_graspable_names(env)
        )
        trace["target_identity"] = {
            "grounded_id": object_id,
            **alias,
        }
        if alias["physical_graspable_name"] is None:
            raise RuntimeError(
                "Could not uniquely resolve grounded visual target to a physical graspable name"
            )

        library = PrimitiveLibrary(approach_dz=8, lift_dz=6)
        grasp_invocation = {
            "action": "vla_act",
            "mode": "grasp",
            "object": object_id,
        }
        grasp = library.compile(grasp_invocation, _pose(obs), coords)
        if len(grasp.actions) != 4:
            raise RuntimeError(
                f"Canonical grasp compiled to {len(grasp.actions)} actions instead of 4"
            )
        grasp_record = {
            "kind": "grasp",
            "invocation": grasp_invocation,
            "compiled_actions": grasp.actions,
            "meta": grasp.meta,
            "step_results": [],
        }
        trace["invocations"].append(grasp_record)

        close_snapshot = None
        last_info: Dict[str, Any] = {}
        for index, action in enumerate(grasp.actions):
            obs, reward, done, last_info = env.step(action)
            snapshot = _snapshot(env, obs, "grasp", index, last_info)
            trace["snapshots"].append(snapshot)
            grasp_record["step_results"].append({
                "subaction_index": index,
                "action": action,
                "reward": reward,
                "done": done,
                "info": last_info,
                "snapshot_env_step": snapshot["env_step"],
            })
            if index == 2:
                close_snapshot = snapshot

        after_lift = trace["snapshots"][-1]
        grasp_outcome = classify_grasp_outcome(
            target_object_id=object_id,
            target_sim_name=alias["physical_graspable_name"],
            grasped_object_names=after_lift["attachments"],
            object_position_at_close=close_snapshot["coords"].get(object_id),
            object_position_after_lift=after_lift["coords"].get(object_id),
            gripper_position_at_close=close_snapshot["pose"][:3],
            gripper_position_after_lift=after_lift["pose"][:3],
            **THRESHOLDS,
        )
        grasp_record["classification"] = grasp_outcome
        trace["result"] = {
            "status": grasp_outcome["outcome"],
            "grasp": grasp_outcome,
            "place_executed": False,
            "final_info": last_info,
        }

        if grasp_outcome["outcome"] == "grasp_verified":
            current_coords = after_lift["coords"]
            if destination_id not in current_coords:
                raise RuntimeError(
                    f"Selected destination {destination_id!r} is no longer visible before place"
                )
            place_invocation = {
                "action": "vla_act",
                "mode": "place",
                "object": object_id,
                "destination": destination_id,
            }
            place = library.compile(place_invocation, _pose(obs), current_coords)
            if len(place.actions) != 3 or not place.meta.get("canonical_contract"):
                raise RuntimeError("Place did not compile to the canonical three-action contract")
            place_record = {
                "kind": "place",
                "invocation": place_invocation,
                "compiled_actions": place.actions,
                "meta": place.meta,
                "step_results": [],
            }
            trace["invocations"].append(place_record)
            for index, action in enumerate(place.actions):
                obs, reward, done, last_info = env.step(action)
                snapshot = _snapshot(env, obs, "place", index, last_info)
                trace["snapshots"].append(snapshot)
                place_record["step_results"].append({
                    "subaction_index": index,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": last_info,
                    "snapshot_env_step": snapshot["env_step"],
                })
            trace["result"].update({
                "place_executed": True,
                "place": {
                    "destination_id": destination_id,
                    "final_info": last_info,
                },
                "final_info": last_info,
            })
        return_code = 0
    except Exception as exc:
        trace["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        trace["result"]["status"] = "error"
        return_code = 1
    finally:
        trace["finished_at"] = datetime.now(timezone.utc).isoformat()
        if env is not None:
            try:
                env.close()
                trace["infrastructure"]["env_closed"] = True
            except Exception as close_exc:
                trace["infrastructure"]["env_closed"] = False
                trace["infrastructure"]["close_error"] = close_exc
                if trace["error"] is None:
                    trace["error"] = {
                        "type": type(close_exc).__name__,
                        "message": str(close_exc),
                        "traceback": traceback.format_exc(),
                    }
                    trace["result"]["status"] = "error"
                    return_code = 1
        TRACE_PATH.write_text(
            json.dumps(_jsonable(trace), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"TRACE: {TRACE_PATH.resolve()}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
