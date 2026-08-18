import numpy as np
import pytest

from harness_vla.perception.libero_grounding import (
    LiberoCameraCalibration,
    LiberoGroundingError,
    calibration_from_sim,
    depth_to_meters,
    ground_instance,
    project_mask_to_world,
)


def _calibration():
    intrinsics = np.asarray(
        [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
    )
    expanded = np.eye(4)
    expanded[:3, :3] = intrinsics
    return LiberoCameraCalibration("agentview", 7, np.linalg.inv(expanded))


class _FakeSim:
    class Model:
        cam_fovy = np.asarray([90.0])

        class Stat:
            extent = 2.0

        class Vis:
            class Map:
                znear = 0.05
                zfar = 5.0

            map = Map()

        stat = Stat()
        vis = Vis()

        @staticmethod
        def camera_name2id(name):
            assert name == "agentview"
            return 0

    class Data:
        cam_xmat = np.asarray([np.eye(3).reshape(-1)])
        cam_xpos = np.asarray([[1.0, 2.0, 3.0]])

    model = Model()
    data = Data()


def test_calibration_and_depth_match_robosuite_equations():
    sim = _FakeSim()
    calibration = calibration_from_sim(sim, "agentview", 100, 200, 3)

    expected_pose = np.diag([1.0, -1.0, -1.0, 1.0])
    expected_pose[:3, 3] = [1.0, 2.0, 3.0]
    expected_intrinsics = np.asarray(
        [[50.0, 0.0, 100.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]]
    )
    expected_expanded = np.eye(4)
    expected_expanded[:3, :3] = expected_intrinsics
    np.testing.assert_allclose(
        calibration.pixel_to_world, expected_pose @ np.linalg.inv(expected_expanded)
    )
    np.testing.assert_allclose(depth_to_meters(sim, [[0.0, 1.0]]), [[0.1, 10.0]])


def test_mask_projection_uses_metric_depth_and_pixel_calibration():
    depth = np.full((4, 5), 2.0)
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 3] = True
    mask[3, 1] = True

    points = project_mask_to_world(depth, mask, _calibration())

    np.testing.assert_allclose(points, [[0.02, -0.02, 2.0], [-0.02, 0.02, 2.0]])


def test_ground_instance_returns_median_and_auditable_provenance():
    segmentation = np.zeros((4, 5, 1), dtype=np.int32)
    segmentation[1, 3, 0] = 4
    segmentation[3, 1, 0] = 4
    observation = {"agentview_segmentation_instance": segmentation}

    result = ground_instance(
        observation,
        {"black_bowl_1": 4},
        "black_bowl_1",
        _calibration(),
        np.full((4, 5), 2.0),
    )

    np.testing.assert_allclose(result["world_xyz"], [0.0, 0.0, 2.0], atol=1e-12)
    provenance = result["provenance"]
    assert provenance["camera"] == "agentview"
    assert provenance["frame_id"] == 7
    assert provenance["visible_pixel_count"] == 2
    assert provenance["valid_point_count"] == 2
    assert provenance["privileged_segmentation"] is True
    assert provenance["coordinate_source"] == "rgbd_projection"
    assert provenance["observation_transform"] == "none"
    assert provenance["pixel_uv"] in ([3, 1], [1, 3])


def test_ground_instance_applies_calibrated_observation_vertical_flip():
    segmentation = np.zeros((4, 5), dtype=np.int32)
    segmentation[0, 2] = 1
    calibration = LiberoCameraCalibration(
        "agentview", 0, _calibration().pixel_to_world, observation_vertical_flip=True
    )

    result = ground_instance(
        {"agentview_segmentation_instance": segmentation},
        {"object_1": 1},
        "object_1",
        calibration,
        np.full((4, 5), 2.0),
    )

    np.testing.assert_allclose(result["world_xyz"], [0.0, 0.02, 2.0])
    assert result["provenance"]["pixel_uv"] == [2, 0]
    assert result["provenance"]["observation_transform"] == "vertical_flip"


@pytest.mark.parametrize(
    "depth,mask",
    [
        (np.ones((2, 2)), np.zeros((2, 2), dtype=bool)),
        (np.ones((2, 2)), np.ones((3, 2), dtype=bool)),
        (np.full((2, 2), np.nan), np.ones((2, 2), dtype=bool)),
    ],
)
def test_invalid_or_empty_projection_fails_closed(depth, mask):
    with pytest.raises(LiberoGroundingError):
        project_mask_to_world(depth, mask, _calibration())


def test_unknown_instance_fails_closed():
    with pytest.raises(LiberoGroundingError, match="unknown instance"):
        ground_instance({}, {}, "missing", _calibration(), np.ones((2, 2)))