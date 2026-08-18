"""Paper-confirmed frozen VLA execution by bounded action chunks.

The OpenPI WebSocket transport and LIBERO-to-EB conversion are paper-compatible.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


class PiRLinfBackendError(RuntimeError):
    """Raised when PiRLinf inference or response validation fails."""


@dataclass(frozen=True)
class PiRLinfObservation:
    """Live inputs required for one PiRLinf chunk inference."""

    front_rgb: np.ndarray
    wrist_rgb: np.ndarray
    gripper_pose: Sequence[float]
    gripper_qpos: Sequence[float]
    mode: str


@dataclass(frozen=True)
class PiRLinfChunk:
    """Validated prefix of one OpenPI action chunk."""

    raw_deltas: tuple[tuple[float, ...], ...]
    full_chunk_length: int
    inference_duration_s: float


def quat_to_axis_angle(quaternion: Sequence[float]) -> np.ndarray:
    """Convert an xyzw quaternion to an axis-angle rotation vector."""
    try:
        values = np.asarray(quaternion, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PiRLinfBackendError(
            "gripper quaternion must be finite and non-zero"
        ) from exc
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise PiRLinfBackendError("gripper quaternion must be finite and non-zero")
    norm = float(np.linalg.norm(values))
    if norm <= 1e-12:
        raise PiRLinfBackendError("gripper quaternion must be finite and non-zero")
    normalized = values / norm
    vector = normalized[:3]
    vector_norm = float(np.linalg.norm(vector))
    angle = 2.0 * math.acos(float(np.clip(normalized[3], -1.0, 1.0)))
    if angle <= 1e-12 or vector_norm <= 1e-12:
        return np.zeros(3, dtype=float)
    return vector / vector_norm * angle


class PiRLinfWebsocketBackend:
    """OpenPI WebSocket client for frozen PiRLinf action chunks."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        replan_steps: int = 5,
        timeout: float = 120.0,
        client=None,
    ) -> None:
        if not isinstance(host, str) or not host.strip():
            raise PiRLinfBackendError("PiRLinf host must be a non-empty string")
        if isinstance(replan_steps, bool) or not isinstance(replan_steps, int) or replan_steps < 1:
            raise PiRLinfBackendError("replan_steps must be an integer greater than zero")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise PiRLinfBackendError("PiRLinf port must be an integer between 1 and 65535")
        try:
            timeout_value = float(timeout)
        except (TypeError, ValueError) as exc:
            raise PiRLinfBackendError("PiRLinf timeout must be greater than zero") from exc
        if not math.isfinite(timeout_value) or timeout_value <= 0:
            raise PiRLinfBackendError("PiRLinf timeout must be greater than zero")
        self.host = host
        self.port = port
        self.replan_steps = replan_steps
        self.timeout = timeout_value
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
        except ImportError as exc:
            raise PiRLinfBackendError(
                "PiRLinf requires the optional 'openpi-client' package"
            ) from exc
        try:
            self._client = WebsocketClientPolicy(host=self.host, port=self.port)
        except Exception as exc:
            raise PiRLinfBackendError(f"PiRLinf client creation failed: {exc}") from exc
        return self._client

    @staticmethod
    def _validate_image(image, name: str) -> np.ndarray:
        values = np.asarray(image)
        if values.ndim != 3 or values.shape[2] != 3:
            raise PiRLinfBackendError(f"{name} must have shape HxWx3")
        if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
            raise PiRLinfBackendError(f"{name} must contain finite numeric pixels")
        values = np.clip(values, 0, 255).astype(np.uint8)
        if values.shape[:2] == (224, 224):
            return values
        try:
            from openpi_client.image_tools import resize_with_pad
        except ImportError as exc:
            raise PiRLinfBackendError(
                f"{name} requires openpi-client resize_with_pad for non-224 images"
            ) from exc
        try:
            return np.asarray(resize_with_pad(values, 224, 224), dtype=np.uint8)
        except Exception as exc:
            raise PiRLinfBackendError(f"failed to resize {name}: {exc}") from exc

    def infer_chunk(self, observation: PiRLinfObservation, prompt: str) -> PiRLinfChunk:
        """Infer and validate one bounded PiRLinf action chunk."""
        if not isinstance(observation, PiRLinfObservation):
            raise PiRLinfBackendError("observation must be a PiRLinfObservation")
        if not isinstance(prompt, str) or not prompt.strip():
            raise PiRLinfBackendError("prompt must be a non-empty instruction")
        if observation.mode not in {"grasp", "place", "push", "task"}:
            raise PiRLinfBackendError(f"unsupported PiRLinf mode: {observation.mode!r}")
        try:
            pose = np.asarray(observation.gripper_pose, dtype=float)
            qpos = np.asarray(observation.gripper_qpos, dtype=float)
        except (TypeError, ValueError) as exc:
            raise PiRLinfBackendError(
                "gripper pose and qpos must contain finite numeric values"
            ) from exc
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise PiRLinfBackendError("gripper_pose must contain exactly 7 finite values")
        if qpos.shape != (2,) or not np.all(np.isfinite(qpos)):
            raise PiRLinfBackendError("gripper_qpos must contain exactly 2 finite values")
        state = np.concatenate((pose[:3], quat_to_axis_angle(pose[3:]), qpos))
        element = {
            "observation/image": self._validate_image(observation.front_rgb, "front_rgb"),
            "observation/wrist_image": self._validate_image(
                observation.wrist_rgb, "wrist_rgb"
            ),
            "observation/state": state,
            "prompt": prompt,
        }
        # EB camera images are passed upright; LIBERO-render's 180-degree rotation is dataset-specific.
        started = time.perf_counter()
        try:
            response = self._get_client().infer(element)
        except PiRLinfBackendError:
            raise
        except Exception as exc:
            raise PiRLinfBackendError(f"PiRLinf WebSocket inference failed: {exc}") from exc
        duration = time.perf_counter() - started
        if not isinstance(response, Mapping) or "actions" not in response:
            raise PiRLinfBackendError("PiRLinf response must contain 'actions'")
        try:
            actions = np.asarray(response["actions"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise PiRLinfBackendError("PiRLinf actions must be a numeric array") from exc
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise PiRLinfBackendError("PiRLinf actions must have shape (N, 7)")
        if actions.shape[0] < self.replan_steps:
            raise PiRLinfBackendError(
                f"PiRLinf returned {actions.shape[0]} actions; expected at least {self.replan_steps}"
            )
        if not np.all(np.isfinite(actions)):
            raise PiRLinfBackendError("PiRLinf actions must contain only finite values")
        raw_deltas = tuple(
            tuple(float(value) for value in action)
            for action in actions[:self.replan_steps]
        )
        return PiRLinfChunk(raw_deltas, int(actions.shape[0]), duration)


__all__ = [
    "PiRLinfBackendError",
    "PiRLinfChunk",
    "PiRLinfObservation",
    "PiRLinfWebsocketBackend",
    "quat_to_axis_angle",
]