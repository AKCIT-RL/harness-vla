"""Optional HTTP adapter for a frozen OpenVLA LIBERO policy.

This beta-only backend is an alternative contact policy, not a reproduction of
the Harness VLA paper.  It requests and converts one action at a time so the
evaluator can reobserve after every environment step.
"""

from __future__ import annotations

import base64
import io
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from numbers import Real
from typing import Callable, Mapping, Sequence

import numpy as np
from PIL import Image


SCENE_BOUNDS = np.array([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6], dtype=float)
VOXEL_SIZE = 100
ROTATION_RESOLUTION = 3
ROTATION_BINS = 360 // ROTATION_RESOLUTION


class OpenVLABackendError(RuntimeError):
    """Raised when OpenVLA inference or response validation fails."""


@dataclass(frozen=True)
class OpenVLAObservation:
    """Live inputs required for one closed-loop OpenVLA inference."""

    front_rgb: np.ndarray
    gripper_pose: Sequence[float]
    mode: str


@dataclass(frozen=True)
class OpenVLAAction:
    """One validated model action and its converted EB action."""

    raw_delta: tuple[float, ...]
    converted_action: tuple[int, ...]
    inference_duration_s: float


Transport = Callable[[str, Mapping, float], Mapping]


def _normalized_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    values = np.asarray(quaternion, dtype=float)
    norm = float(np.linalg.norm(values))
    if values.shape != (4,) or not math.isfinite(norm) or norm <= 1e-12:
        raise OpenVLABackendError("gripper quaternion must be finite and non-zero")
    return values / norm


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_x, left_y, left_z, left_w = left
    right_x, right_y, right_z, right_w = right
    return np.array([
        left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
        left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
        left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
        left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
    ])


def _rotvec_to_quaternion(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    scale = math.sin(angle / 2.0) / angle
    return np.concatenate((rotvec * scale, [math.cos(angle / 2.0)]))


def _quaternion_to_euler_xyz(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalized_quaternion(quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


def convert_libero_delta_to_eb(
    raw_delta: Sequence[float],
    gripper_pose: Sequence[float],
    *,
    max_delta_xyz: float,
    max_delta_rotation: float,
    workspace_bounds: Sequence[float],
    gripper_convention: str,
    rotation_frame: str,
) -> list[int]:
    """Compose one LIBERO delta with a live ``[xyz, quat]`` into EB bins.

    Raw deltas are normalized OSC commands in [-1, 1]; robosuite scales them
    by output_max (max_delta_xyz meters, max_delta_rotation radians).
    """
    if len(raw_delta) != 7 or len(gripper_pose) < 7:
        raise OpenVLABackendError("action and gripper_pose must have 7 values")
    pose = np.asarray(gripper_pose[:7], dtype=float)
    if not np.all(np.isfinite(pose)):
        raise OpenVLABackendError("gripper_pose must contain finite values")
    bounds = np.asarray(workspace_bounds, dtype=float)

    delta_xyz = (
        np.clip(np.asarray(raw_delta[:3], dtype=float), -1.0, 1.0) * max_delta_xyz
    )
    target_xyz = np.clip(pose[:3] + delta_xyz, bounds[:3], bounds[3:])
    resolution = (bounds[3:] - bounds[:3]) / VOXEL_SIZE
    voxel = np.clip(
        np.floor((target_xyz - bounds[:3]) / resolution).astype(int),
        0,
        VOXEL_SIZE - 1,
    )

    delta_rotvec = (
        np.clip(np.asarray(raw_delta[3:6], dtype=float), -1.0, 1.0)
        * max_delta_rotation
    )
    current_rotation = _normalized_quaternion(pose[3:7])
    delta_rotation = _rotvec_to_quaternion(delta_rotvec)
    target_rotation = (
        _quaternion_multiply(current_rotation, delta_rotation)
        if rotation_frame == "local"
        else _quaternion_multiply(delta_rotation, current_rotation)
    )
    euler = _quaternion_to_euler_xyz(target_rotation)
    rotation_bins = np.clip(
        np.rint((euler + 180.0) / ROTATION_RESOLUTION).astype(int),
        0,
        ROTATION_BINS,
    )

    gripper_value = float(raw_delta[6])
    if gripper_convention == "libero_minus_open_plus_close":
        eb_gripper = 0 if gripper_value >= 0 else 1
    else:
        eb_gripper = 1 if gripper_value >= 0 else 0
    return [int(value) for value in np.concatenate((voxel, rotation_bins, [eb_gripper]))]


def _default_transport(url: str, payload: Mapping, timeout: float) -> Mapping:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise OpenVLABackendError(f"OpenVLA HTTP request failed: {exc}") from exc
    try:
        decoded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenVLABackendError("OpenVLA response was not valid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise OpenVLABackendError("OpenVLA response must be a JSON object")
    return decoded


class OpenVLAHTTPBackend:
    """HTTP client for one frozen OpenVLA LIBERO delta action per request."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 120.0,
        max_delta_xyz: float = 0.05,
        max_delta_rotation: float = 0.5,
        workspace_bounds: Sequence[float] = SCENE_BOUNDS,
        gripper_convention: str = "libero_minus_open_plus_close",
        rotation_frame: str = "local",
        expected_unnorm_key: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        if not isinstance(url, str) or not url.strip():
            raise ValueError("OpenVLA URL must be a non-empty string")
        if max_delta_xyz <= 0 or max_delta_rotation <= 0:
            raise ValueError("OpenVLA delta clamps must be greater than zero")
        bounds = np.asarray(workspace_bounds, dtype=float)
        if bounds.shape != (6,) or np.any(bounds[:3] >= bounds[3:]):
            raise ValueError("workspace_bounds must contain increasing xyz min/max values")
        if gripper_convention not in {
            "libero_minus_open_plus_close",
            "minus_close_plus_open",
        }:
            raise ValueError("unsupported OpenVLA gripper convention")
        if rotation_frame not in {"local", "world"}:
            raise ValueError("rotation_frame must be 'local' or 'world'")
        self.url = url
        self.timeout = float(timeout)
        self.max_delta_xyz = float(max_delta_xyz)
        self.max_delta_rotation = float(max_delta_rotation)
        self.workspace_bounds = bounds
        self.gripper_convention = gripper_convention
        self.rotation_frame = rotation_frame
        self.expected_unnorm_key = expected_unnorm_key
        self.transport = transport or _default_transport

    def infer_chunk(self, observation: OpenVLAObservation, prompt: str) -> OpenVLAAction:
        """Infer exactly one action from the current image and gripper pose."""
        if not isinstance(observation, OpenVLAObservation):
            raise TypeError("observation must be an OpenVLAObservation")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty instruction")
        if observation.mode not in {"grasp", "place", "push"}:
            raise OpenVLABackendError(f"unsupported OpenVLA mode: {observation.mode!r}")
        payload = {
            "image": self._encode_image(observation.front_rgb),
            "prompt": prompt,
            "mode": observation.mode,
            "max_actions": 1,
        }
        started = time.perf_counter()
        try:
            response = self.transport(self.url, payload, self.timeout)
        except OpenVLABackendError:
            raise
        except Exception as exc:
            raise OpenVLABackendError(f"OpenVLA transport failed: {exc}") from exc
        duration = time.perf_counter() - started
        try:
            if response.get("error"):
                raise OpenVLABackendError(f"OpenVLA server error: {response['error']}")
            if (
                self.expected_unnorm_key is not None
                and response.get("unnorm_key") != self.expected_unnorm_key
            ):
                raise OpenVLABackendError(
                    "OpenVLA unnorm_key mismatch: expected {!r}, got {!r}".format(
                        self.expected_unnorm_key, response.get("unnorm_key")
                    )
                )
            raw_delta = self._validate_action(response)
            converted = self.convert_action(raw_delta, observation.gripper_pose)
        except OpenVLABackendError:
            raise
        except Exception as exc:
            raise OpenVLABackendError(f"OpenVLA action conversion failed: {exc}") from exc
        return OpenVLAAction(raw_delta, tuple(converted), duration)

    @staticmethod
    def _encode_image(front_rgb: np.ndarray) -> Mapping[str, str]:
        image = np.asarray(front_rgb)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise OpenVLABackendError("front_rgb must have shape HxWx3 or HxWx4")
        if image.dtype != np.uint8:
            if not np.issubdtype(image.dtype, np.number) or not np.all(np.isfinite(image)):
                raise OpenVLABackendError("front_rgb must contain finite numeric pixels")
            image = np.clip(image, 0, 255).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(image).convert("RGB").save(buffer, format="PNG")
        return {
            "encoding": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }

    @staticmethod
    def _validate_action(response: Mapping) -> tuple[float, ...]:
        if not isinstance(response, Mapping):
            raise OpenVLABackendError("OpenVLA response must be a JSON object")
        action = response.get("action")
        if not isinstance(action, (list, tuple)) or len(action) != 7:
            raise OpenVLABackendError("OpenVLA response action must contain exactly 7 values")
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in action):
            raise OpenVLABackendError("OpenVLA response action must contain only numbers")
        values = tuple(float(value) for value in action)
        if not all(math.isfinite(value) for value in values):
            raise OpenVLABackendError("OpenVLA response action must contain only finite values")
        return values

    def convert_action(
        self, raw_delta: Sequence[float], gripper_pose: Sequence[float]
    ) -> list[int]:
        """Compose a LIBERO delta with live ``[xyz, quat]`` into EB bins."""
        return convert_libero_delta_to_eb(
            raw_delta,
            gripper_pose,
            max_delta_xyz=self.max_delta_xyz,
            max_delta_rotation=self.max_delta_rotation,
            workspace_bounds=self.workspace_bounds,
            gripper_convention=self.gripper_convention,
            rotation_frame=self.rotation_frame,
        )


__all__ = [
    "convert_libero_delta_to_eb",
    "OpenVLAAction",
    "OpenVLABackendError",
    "OpenVLAHTTPBackend",
    "OpenVLAObservation",
]