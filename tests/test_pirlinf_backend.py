"""Tests for the simulator-independent PiRLinf WebSocket adapter."""

import math

import numpy as np
import pytest

from harness_vla.backends.pirlinf_backend import (
    PiRLinfBackendError,
    PiRLinfObservation,
    PiRLinfWebsocketBackend,
    quat_to_axis_angle,
)


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.elements = []

    def infer(self, element):
        self.elements.append(element)
        return self.response


def _observation(*, image=None, pose=None, qpos=None, mode="grasp"):
    rgb = np.full((224, 224, 3), 127, dtype=np.uint8) if image is None else image
    return PiRLinfObservation(
        front_rgb=rgb,
        wrist_rgb=rgb.copy(),
        gripper_pose=(
            [0.2, -0.1, 1.0, 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
            if pose is None else pose
        ),
        gripper_qpos=[0.03, 0.04] if qpos is None else qpos,
        mode=mode,
    )


def _backend(response=None, *, replan_steps=5):
    actions = np.arange(70, dtype=float).reshape(10, 7) / 70
    client = FakeClient({"actions": actions} if response is None else response)
    return PiRLinfWebsocketBackend(
        "127.0.0.1", 8010, replan_steps=replan_steps, client=client
    ), client


def test_element_has_exact_keys_images_and_eight_dimensional_state():
    backend, client = _backend()

    backend.infer_chunk(_observation(), "grasp the red cube")

    element = client.elements[0]
    assert set(element) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
    }
    assert element["observation/image"].shape == (224, 224, 3)
    assert element["observation/wrist_image"].shape == (224, 224, 3)
    assert element["observation/image"].dtype == np.uint8
    assert element["observation/wrist_image"].dtype == np.uint8
    np.testing.assert_allclose(
        element["observation/state"],
        [0.2, -0.1, 1.0, 0.0, 0.0, math.pi / 2, 0.03, 0.04],
    )
    assert element["prompt"] == "grasp the red cube"


def test_chunk_is_truncated_and_preserves_full_length():
    backend, _ = _backend(replan_steps=3)

    chunk = backend.infer_chunk(_observation(), "grasp the red cube")

    assert len(chunk.raw_deltas) == 3
    assert chunk.full_chunk_length == 10
    assert chunk.raw_deltas[0] == tuple(np.arange(7, dtype=float) / 70)
    assert chunk.inference_duration_s >= 0


def test_full_task_mode_preserves_the_native_libero_prompt():
    backend, client = _backend()

    chunk = backend.infer_chunk(
        _observation(mode="task"), "pick up the bowl and place it on the plate"
    )

    assert len(chunk.raw_deltas) == 5
    assert client.elements[0]["prompt"] == "pick up the bowl and place it on the plate"


def test_replan_steps_must_be_positive():
    with pytest.raises(PiRLinfBackendError, match="greater than zero"):
        PiRLinfWebsocketBackend("127.0.0.1", 8010, replan_steps=0, client=FakeClient({}))


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({}, "contain 'actions'"),
        ({"actions": np.zeros((10, 6))}, "shape"),
        ({"actions": np.zeros((4, 7))}, "expected at least"),
        ({"actions": np.full((10, 7), np.nan)}, "finite"),
    ],
)
def test_invalid_responses_are_explicit(response, message):
    backend, _ = _backend(response)

    with pytest.raises(PiRLinfBackendError, match=message):
        backend.infer_chunk(_observation(), "grasp the red cube")


def test_invalid_image_is_rejected_before_inference():
    backend, client = _backend()

    with pytest.raises(PiRLinfBackendError, match="shape HxWx3"):
        backend.infer_chunk(_observation(image=np.zeros((224, 224))), "grasp")
    assert client.elements == []


def test_zero_quaternion_is_rejected_before_inference():
    backend, client = _backend()

    with pytest.raises(PiRLinfBackendError, match="quaternion must be finite and non-zero"):
        backend.infer_chunk(
            _observation(pose=[0.2, -0.1, 1.0, 0.0, 0.0, 0.0, 0.0]), "grasp"
        )
    assert client.elements == []


def test_quaternion_to_axis_angle_known_rotations():
    np.testing.assert_allclose(quat_to_axis_angle([0, 0, 0, 1]), np.zeros(3))
    np.testing.assert_allclose(
        quat_to_axis_angle([0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]),
        [0, 0, math.pi / 2],
    )