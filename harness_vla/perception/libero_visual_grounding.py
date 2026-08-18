"""Visual-only pixel localization with auditable LIBERO RGB-D projection."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from typing import Any, Optional, Protocol, Sequence, Tuple
from urllib import request

import numpy as np
from PIL import Image

from harness_vla.perception.libero_grounding import (
    LiberoCameraCalibration,
    LiberoGroundingError,
    project_mask_to_world,
)


_DEFAULT_PROMPT = """Locate exactly one visible instance matching this description: {target}
Return only a JSON object with exactly these fields:
{{"pixel_uv":[u,v],"confidence":number,"bbox":[left,top,right,bottom]}}
bbox is optional. Coordinates use a normalized 0..1000 image grid, with [0,0] at the
top-left and [1000,1000] at the bottom-right. If the target is absent or ambiguous,
return confidence 0. Do not include markdown or any other fields."""


@dataclass(frozen=True)
class VisualPixelObservation:
    """An RGB frame intentionally excluding simulator segmentation and poses."""

    rgb: Any


@dataclass(frozen=True)
class VisualPixelSelection:
    """One visual locator result in RGB image coordinates."""

    pixel_uv: Sequence[float]
    confidence: float
    bbox: Optional[Sequence[float]] = None
    locator_model: Optional[str] = None
    prompt_hash: Optional[str] = None
    coordinate_transform: Optional[str] = None


class VisualPixelLocator(Protocol):
    """Protocol implemented by visual pixel locators."""

    def locate(
        self, observation: VisualPixelObservation, target_description: str
    ) -> VisualPixelSelection:
        ...


class OllamaVisualPixelLocator:
    """Locate an object pixel using Ollama's Gemma-compatible vision chat API."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 60.0,
        prompt_template: str = _DEFAULT_PROMPT,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.prompt_template = prompt_template

    @staticmethod
    def parse_response(content: str) -> VisualPixelSelection:
        """Parse the locator response without accepting prose or schema drift."""
        try:
            value = json.loads(content, object_pairs_hook=_strict_json_object)
        except (TypeError, json.JSONDecodeError) as exc:
            raise LiberoGroundingError("visual locator returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise LiberoGroundingError("visual locator response must be a JSON object")
        allowed = {"pixel_uv", "confidence", "bbox"}
        if set(value) - allowed or not {"pixel_uv", "confidence"} <= set(value):
            raise LiberoGroundingError("visual locator response has an invalid schema")
        pixel = value["pixel_uv"]
        confidence = value["confidence"]
        bbox = value.get("bbox")
        if not _is_number_pair(pixel) or not _is_number(confidence):
            raise LiberoGroundingError("visual locator pixel/confidence is invalid")
        if bbox is not None and not _is_number_sequence(bbox, 4):
            raise LiberoGroundingError("visual locator bbox is invalid")
        if not 0.0 <= float(confidence) <= 1.0:
            raise LiberoGroundingError("visual locator confidence is invalid")
        return VisualPixelSelection(pixel, float(confidence), bbox)

    def locate(
        self, observation: VisualPixelObservation, target_description: str
    ) -> VisualPixelSelection:
        rgb = _validated_rgb(observation.rgb)
        prompt = self.prompt_template.format(target=target_description.strip())
        image_buffer = BytesIO()
        Image.fromarray(rgb).save(image_buffer, format="PNG")
        encoded_image = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "think": False,
            "messages": [
                {"role": "user", "content": prompt, "images": [encoded_image]}
            ],
            "options": {"temperature": 0, "num_predict": 256},
        }
        api_request = request.Request(
            self.base_url + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(api_request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LiberoGroundingError("Ollama visual locator request failed") from exc
        try:
            content = response_payload["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LiberoGroundingError("Ollama response lacks message content") from exc
        selection = _normalized_selection_to_pixels(
            self.parse_response(content), rgb.shape[1], rgb.shape[0]
        )
        return VisualPixelSelection(
            pixel_uv=selection.pixel_uv,
            confidence=selection.confidence,
            bbox=selection.bbox,
            locator_model=self.model,
            prompt_hash=sha256(prompt.encode("utf-8")).hexdigest(),
            coordinate_transform="normalized_1000_to_image_pixels",
        )

    def __call__(
        self, observation: VisualPixelObservation, target_description: str
    ) -> VisualPixelSelection:
        return self.locate(observation, target_description)


def ground_visual_instance(
    observation: VisualPixelObservation,
    target_description: str,
    calibration: LiberoCameraCalibration,
    metric_depth,
    locator,
):
    """Locate a target visually and robustly back-project nearby metric depth."""
    rgb = _validated_rgb(observation.rgb)
    depth = np.asarray(metric_depth, dtype=float)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
        raise LiberoGroundingError("RGB and metric depth shapes do not match")
    if not target_description or not target_description.strip():
        raise LiberoGroundingError("target description must not be empty")

    if hasattr(locator, "locate"):
        selection = locator.locate(observation, target_description)
    elif callable(locator):
        selection = locator(observation, target_description)
    else:
        raise LiberoGroundingError("visual locator is not callable")
    if not isinstance(selection, VisualPixelSelection):
        raise LiberoGroundingError("visual locator returned an invalid selection")

    height, width = depth.shape
    pixel_u, pixel_v = _validated_pixel(selection.pixel_uv, width, height)
    confidence = float(selection.confidence)
    if not math.isfinite(confidence) or confidence < 0.5 or confidence > 1.0:
        raise LiberoGroundingError("visual locator confidence is too low or invalid")

    sample_mask = np.zeros((height, width), dtype=bool)
    if selection.bbox is None:
        center_u, center_v = int(round(pixel_u)), int(round(pixel_v))
        sample_mask[
            max(0, center_v - 2) : min(height, center_v + 3),
            max(0, center_u - 2) : min(width, center_u + 3),
        ] = True
    else:
        left, top, right, bottom = _validated_bbox(
            selection.bbox, width, height, pixel_u, pixel_v
        )
        sample_mask[top : bottom + 1, left : right + 1] = True

    valid_depth = sample_mask & np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid_depth):
        raise LiberoGroundingError("selected visual region contains no valid depth")
    robust_depth = float(np.median(depth[valid_depth]))
    robust_mask = valid_depth & np.isclose(
        depth, robust_depth, rtol=0.05, atol=max(0.002, robust_depth * 0.01)
    )
    if not np.any(robust_mask):
        raise LiberoGroundingError("selected visual region has ambiguous depth")

    projection_depth = depth
    projection_mask = robust_mask
    if calibration.observation_vertical_flip:
        projection_depth = projection_depth[::-1]
        projection_mask = projection_mask[::-1]
    points = project_mask_to_world(projection_depth, projection_mask, calibration)
    world_xyz = np.median(points, axis=0)
    locator_model = selection.locator_model or getattr(locator, "model", None)
    prompt_hash = selection.prompt_hash or getattr(locator, "prompt_hash", None)
    if not locator_model:
        locator_model = type(locator).__name__
    if not prompt_hash:
        prompt_hash = sha256(target_description.strip().encode("utf-8")).hexdigest()

    return {
        "target_description": target_description,
        "world_xyz": [float(value) for value in world_xyz],
        "provenance": {
            "version": 1,
            "method": "libero_visual_rgbd_grounding",
            "camera": calibration.camera,
            "frame_id": int(calibration.frame_id),
            "pixel_uv": [float(pixel_u), float(pixel_v)],
            "bbox": list(selection.bbox) if selection.bbox is not None else None,
            "confidence": confidence,
            "depth_m": robust_depth,
            "depth_sample_count": int(np.count_nonzero(robust_mask)),
            "locator_model": str(locator_model),
            "locator_prompt_hash": str(prompt_hash),
            "locator_coordinate_transform": selection.coordinate_transform or "none",
            "pixel_selection": "visual_locator",
            "privileged_segmentation": False,
            "coordinate_source": "rgbd_projection",
            "observation_transform": (
                "vertical_flip" if calibration.observation_vertical_flip else "none"
            ),
        },
    }


def _validated_rgb(rgb) -> np.ndarray:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise LiberoGroundingError("RGB must be a uint8 array with shape (H, W, 3)")
    if not array.shape[0] or not array.shape[1]:
        raise LiberoGroundingError("RGB must not be empty")
    return array


def _normalized_selection_to_pixels(
    selection: VisualPixelSelection, width: int, height: int
) -> VisualPixelSelection:
    values = list(selection.pixel_uv)
    if not _is_number_pair(values) or any(value < 0.0 or value > 1000.0 for value in values):
        raise LiberoGroundingError("visual locator normalized pixel is invalid")
    pixel = [
        float(values[0]) * (width - 1) / 1000.0,
        float(values[1]) * (height - 1) / 1000.0,
    ]
    bbox = None
    if selection.bbox is not None:
        bounds = list(selection.bbox)
        if not _is_number_sequence(bounds, 4) or any(
            value < 0.0 or value > 1000.0 for value in bounds
        ):
            raise LiberoGroundingError("visual locator normalized bbox is invalid")
        bbox = [
            float(bounds[0]) * (width - 1) / 1000.0,
            float(bounds[1]) * (height - 1) / 1000.0,
            float(bounds[2]) * (width - 1) / 1000.0,
            float(bounds[3]) * (height - 1) / 1000.0,
        ]
    return VisualPixelSelection(pixel, selection.confidence, bbox)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_number_sequence(value: Any, length: int) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(_is_number(item) for item in value)
    )


def _is_number_pair(value: Any) -> bool:
    return _is_number_sequence(value, 2)


def _strict_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise LiberoGroundingError("visual locator response has duplicate fields")
        value[key] = item
    return value


def _validated_pixel(pixel, width: int, height: int) -> Tuple[float, float]:
    if not _is_number_pair(pixel):
        raise LiberoGroundingError("pixel_uv must contain two finite numbers")
    pixel_u, pixel_v = (float(value) for value in pixel)
    if pixel_u < 0.0 or pixel_u >= width or pixel_v < 0.0 or pixel_v >= height:
        raise LiberoGroundingError("visual locator pixel is outside the RGB frame")
    return pixel_u, pixel_v


def _validated_bbox(bbox, width, height, pixel_u, pixel_v) -> Tuple[int, int, int, int]:
    if not _is_number_sequence(bbox, 4):
        raise LiberoGroundingError("bbox must contain four finite numbers")
    left_f, top_f, right_f, bottom_f = (float(value) for value in bbox)
    if not (0 <= left_f <= pixel_u <= right_f < width):
        raise LiberoGroundingError("bbox horizontal coordinates are invalid")
    if not (0 <= top_f <= pixel_v <= bottom_f < height):
        raise LiberoGroundingError("bbox vertical coordinates are invalid")
    return (
        int(math.floor(left_f)),
        int(math.floor(top_f)),
        int(math.ceil(right_f)),
        int(math.ceil(bottom_f)),
    )


__all__ = [
    "OllamaVisualPixelLocator",
    "VisualPixelLocator",
    "VisualPixelObservation",
    "VisualPixelSelection",
    "ground_visual_instance",
]
