"""Tests for the simulator-independent OpenVLA HTTP adapter."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from harness_vla.backends.openvla_backend import (
    OpenVLABackendError,
    OpenVLAHTTPBackend,
    OpenVLAObservation,
)


def _observation(*, pose=None, mode="grasp"):
    return OpenVLAObservation(
        front_rgb=np.full((3, 4, 3), 127, dtype=np.uint8),
        gripper_pose=pose or [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        mode=mode,
    )


def test_request_uses_structured_png_and_one_action():
    calls = []

    def transport(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {"action": [0, 0, 0, 0, 0, 0, -1]}

    backend = OpenVLAHTTPBackend(
        "http://127.0.0.1:8000/predict", timeout=7, transport=transport
    )
    result = backend.infer_chunk(_observation(mode="place"), "place the red cube")

    url, payload, timeout = calls[0]
    assert url == "http://127.0.0.1:8000/predict"
    assert timeout == 7
    assert payload["prompt"] == "place the red cube"
    assert payload["mode"] == "place"
    assert payload["max_actions"] == 1
    assert payload["image"]["encoding"] == "base64"
    assert payload["image"]["media_type"] == "image/png"
    decoded = base64.b64decode(payload["image"]["data"], validate=True)
    assert Image.open(io.BytesIO(decoded)).size == (4, 3)
    assert result.raw_delta == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0)
    assert result.converted_action[6] == 1
    assert result.inference_duration_s >= 0


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"action": [0] * 6},
        {"action": [0] * 8},
        {"action": [0, 0, 0, 0, 0, 0, "open"]},
        {"action": [0, 0, 0, 0, 0, 0, float("nan")]},
    ],
)
def test_response_requires_exactly_seven_finite_floats(response):
    backend = OpenVLAHTTPBackend("http://fake", transport=lambda *_: response)

    with pytest.raises(OpenVLABackendError):
        backend.infer_chunk(_observation(), "grasp the cube")


def test_delta_clamps_workspace_and_uses_official_libero_gripper_sign():
    backend = OpenVLAHTTPBackend(
        "http://fake",
        max_delta_xyz=0.05,
        max_delta_rotation=0.2,
        transport=lambda *_: {},
    )
    pose = [0.69, 0.49, 1.59, 0, 0, 0, 1]

    close_action = backend.convert_action([1, 1, 1, 0, 0, 1, 1], pose)
    open_action = backend.convert_action([1, 1, 1, 0, 0, 1, -1], pose)

    assert close_action[:3] == [99, 99, 99]
    assert close_action[6] == 0
    assert open_action[6] == 1
    expected_yaw_bin = round((np.degrees(0.2) + 180) / 3)
    assert close_action[5] == expected_yaw_bin


def test_axis_angle_is_composed_with_live_quaternion_in_local_frame():
    half_angle = np.deg2rad(45)
    current = [np.sin(half_angle), 0, 0, np.cos(half_angle)]
    pose = [0.2, 0.0, 1.0, *current]
    backend = OpenVLAHTTPBackend(
        "http://fake", max_delta_rotation=np.pi / 6, transport=lambda *_: {}
    )

    converted = backend.convert_action([0, 0, 0, 1, 0, 0, -1], pose)
    assert converted[3:6] == [100, 60, 60]


def test_transport_failure_is_explicit():
    def failing_transport(*_):
        raise OSError("model unavailable")

    backend = OpenVLAHTTPBackend("http://fake", transport=failing_transport)

    with pytest.raises(OpenVLABackendError, match="model unavailable"):
        backend.infer_chunk(_observation(), "grasp the cube")


def test_server_error_and_unnorm_mismatch_are_explicit():
    server_error = OpenVLAHTTPBackend(
        "http://fake", transport=lambda *_: {"error": "inference failed"}
    )
    with pytest.raises(OpenVLABackendError, match="server error: inference failed"):
        server_error.infer_chunk(_observation(), "grasp the cube")

    wrong_stats = OpenVLAHTTPBackend(
        "http://fake",
        expected_unnorm_key="libero_object",
        transport=lambda *_: {
            "action": [0, 0, 0, 0, 0, 0, -1],
            "unnorm_key": "libero_spatial",
        },
    )
    with pytest.raises(OpenVLABackendError, match="unnorm_key mismatch"):
        wrong_stats.infer_chunk(_observation(), "grasp the cube")


def test_invalid_contact_mode_is_rejected_before_transport():
    calls = []
    backend = OpenVLAHTTPBackend(
        "http://fake", transport=lambda *_: calls.append(True)
    )
    with pytest.raises(OpenVLABackendError, match="unsupported OpenVLA mode"):
        backend.infer_chunk(_observation(mode=""), "touch the cube")
    assert calls == []


def test_invalid_live_quaternion_is_explicit():
    backend = OpenVLAHTTPBackend(
        "http://fake",
        transport=lambda *_: {"action": [0, 0, 0, 0, 0, 0, -1]},
    )
    invalid_pose = [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    with pytest.raises(OpenVLABackendError, match="quaternion must be finite and non-zero"):
        backend.infer_chunk(_observation(pose=invalid_pose), "grasp the cube")