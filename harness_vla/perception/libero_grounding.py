"""Auditable RGB-D instance grounding for native LIBERO observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np


class LiberoGroundingError(ValueError):
    """Raised when an RGB-D estimate cannot be produced safely."""


@dataclass(frozen=True)
class LiberoCameraCalibration:
    camera: str
    frame_id: int
    pixel_to_world: np.ndarray
    observation_vertical_flip: bool = False


def calibration_from_sim(
    sim, camera: str, height: int, width: int, frame_id: int
) -> LiberoCameraCalibration:
    """Read the camera model using the installed robosuite equations."""
    camera_id = sim.model.camera_name2id(camera)
    focal_length = 0.5 * height / np.tan(
        float(sim.model.cam_fovy[camera_id]) * np.pi / 360.0
    )
    intrinsics = np.asarray(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.asarray(sim.data.cam_xmat[camera_id]).reshape(3, 3)
    camera_pose[:3, 3] = np.asarray(sim.data.cam_xpos[camera_id])
    camera_axis_correction = np.diag([1.0, -1.0, -1.0, 1.0])
    camera_pose = camera_pose @ camera_axis_correction
    expanded_intrinsics = np.eye(4)
    expanded_intrinsics[:3, :3] = intrinsics
    world_to_pixel = expanded_intrinsics @ np.linalg.inv(camera_pose)
    return LiberoCameraCalibration(
        camera=camera,
        frame_id=frame_id,
        pixel_to_world=np.linalg.inv(np.asarray(world_to_pixel, dtype=float)),
        observation_vertical_flip=True,
    )


def depth_to_meters(sim, normalized_depth) -> np.ndarray:
    """Convert MuJoCo nonlinear depth using the installed robosuite equation."""
    depth = np.asarray(normalized_depth, dtype=float)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2 or not np.all(np.isfinite(depth)):
        raise LiberoGroundingError("depth must have shape (height, width) and be finite")
    if np.any(depth < 0.0) or np.any(depth > 1.0):
        raise LiberoGroundingError("normalized depth must lie in [0, 1]")
    extent = float(sim.model.stat.extent)
    far = float(sim.model.vis.map.zfar) * extent
    near = float(sim.model.vis.map.znear) * extent
    if near <= 0.0 or far <= near:
        raise LiberoGroundingError("simulator clipping planes are invalid")
    return near / (1.0 - depth * (1.0 - near / far))


def project_mask_to_world(
    metric_depth, mask, calibration: LiberoCameraCalibration
) -> np.ndarray:
    """Back-project every valid masked pixel into world coordinates."""
    depth = np.asarray(metric_depth, dtype=float)
    selected = np.asarray(mask, dtype=bool)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2 or selected.shape != depth.shape:
        raise LiberoGroundingError("depth and mask must have the same 2-D shape")
    transform = np.asarray(calibration.pixel_to_world, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise LiberoGroundingError("pixel_to_world must be a finite 4x4 matrix")

    rows, columns = np.nonzero(selected)
    values = depth[rows, columns]
    valid = np.isfinite(values) & (values > 0.0)
    rows, columns, values = rows[valid], columns[valid], values[valid]
    if not len(values):
        raise LiberoGroundingError("instance mask contains no valid depth pixels")
    homogeneous = np.stack(
        (columns * values, rows * values, values, np.ones_like(values)), axis=1
    )
    points = (transform @ homogeneous.T).T
    points = points[:, :3] / points[:, 3:4]
    if not np.all(np.isfinite(points)):
        raise LiberoGroundingError("projection produced non-finite world points")
    return points


def ground_instance(
    observation: Mapping[str, Any],
    instance_to_id: Mapping[str, int],
    instance_name: str,
    calibration: LiberoCameraCalibration,
    metric_depth,
) -> Dict[str, Any]:
    """Estimate one visible instance and retain its privileged-mask provenance."""
    segmentation_key = "%s_segmentation_instance" % calibration.camera
    if instance_name not in instance_to_id:
        raise LiberoGroundingError("unknown instance %r" % instance_name)
    if segmentation_key not in observation:
        raise LiberoGroundingError("observation lacks %s" % segmentation_key)
    segmentation = np.asarray(observation[segmentation_key])
    if segmentation.ndim == 3 and segmentation.shape[2] == 1:
        segmentation = segmentation[:, :, 0]
    if segmentation.ndim != 2:
        raise LiberoGroundingError("instance segmentation must be 2-D")
    mask = segmentation == int(instance_to_id[instance_name])
    projection_depth = np.asarray(metric_depth)
    projection_mask = mask
    if calibration.observation_vertical_flip:
        projection_depth = projection_depth[::-1]
        projection_mask = projection_mask[::-1]
    points = project_mask_to_world(projection_depth, projection_mask, calibration)
    rows, columns = np.nonzero(mask)
    median = np.median(np.stack((rows, columns), axis=1), axis=0)
    representative_index = int(
        np.argmin((rows - median[0]) ** 2 + (columns - median[1]) ** 2)
    )
    world_xyz = np.median(points, axis=0)
    return {
        "instance_name": instance_name,
        "world_xyz": [float(value) for value in world_xyz],
        "provenance": {
            "version": 1,
            "method": "libero_rgbd_instance_mask_median",
            "camera": calibration.camera,
            "frame_id": int(calibration.frame_id),
            "pixel_uv": [
                int(columns[representative_index]),
                int(rows[representative_index]),
            ],
            "visible_pixel_count": int(mask.sum()),
            "valid_point_count": int(len(points)),
            "pixel_selection": "sim_instance_mask",
            "privileged_segmentation": True,
            "coordinate_source": "rgbd_projection",
            "observation_transform": (
                "vertical_flip"
                if calibration.observation_vertical_flip
                else "none"
            ),
        },
    }


__all__ = [
    "LiberoCameraCalibration",
    "LiberoGroundingError",
    "calibration_from_sim",
    "depth_to_meters",
    "ground_instance",
    "project_mask_to_world",
]