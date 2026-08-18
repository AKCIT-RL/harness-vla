import base64
from collections.abc import Mapping
from io import BytesIO
import json

import numpy as np
import pytest
from PIL import Image

from harness_vla.perception.libero_grounding import (
    LiberoCameraCalibration,
    LiberoGroundingError,
)
from harness_vla.perception import libero_visual_grounding as visual
from harness_vla.perception.libero_visual_grounding import (
    OllamaVisualPixelLocator,
    VisualPixelObservation,
    VisualPixelSelection,
    ground_visual_instance,
)


def _calibration():
    intrinsics = np.asarray(
        [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
    )
    expanded = np.eye(4)
    expanded[:3, :3] = intrinsics
    return LiberoCameraCalibration("agentview", 11, np.linalg.inv(expanded))


def _observation(rgb=None):
    if rgb is None:
        rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    return VisualPixelObservation(rgb)


def _selection(**changes):
    values = {
        "pixel_uv": [2, 2],
        "confidence": 0.9,
        "bbox": [1, 1, 3, 3],
        "locator_model": "fake-gemma",
        "prompt_hash": "abc123",
    }
    values.update(changes)
    return VisualPixelSelection(**values)


def test_world_projection_uses_robust_bbox_depth_and_callable_locator():
    depth = np.full((5, 5), 2.0)
    depth[1, 1] = 20.0

    result = ground_visual_instance(
        _observation(), "black bowl", _calibration(), depth, lambda *_: _selection()
    )

    np.testing.assert_allclose(result["world_xyz"], [0.0, 0.0, 2.0])
    provenance = result["provenance"]
    assert provenance["depth_m"] == 2.0
    assert provenance["depth_sample_count"] == 8
    assert provenance["camera"] == "agentview"
    assert provenance["frame_id"] == 11


class _ExplodingMapping(Mapping):
    def __init__(self, rgb):
        self.rgb = rgb

    def __array__(self, dtype=None):
        return np.asarray(self.rgb, dtype=dtype)

    def __getitem__(self, key):
        raise AssertionError("grounding must not inspect observation keys: %s" % key)

    def __iter__(self):
        raise AssertionError("grounding must not iterate observation keys")

    def __len__(self):
        raise AssertionError("grounding must not inspect observation keys")


def test_observation_never_accesses_segmentation_or_oracle_keys():
    observation = _observation(_ExplodingMapping(np.zeros((5, 5, 3), dtype=np.uint8)))

    result = ground_visual_instance(
        observation, "red mug", _calibration(), np.full((5, 5), 2.0), lambda *_: _selection()
    )

    assert result["provenance"]["privileged_segmentation"] is False


@pytest.mark.parametrize(
    "content",
    [
        '```json\n{"pixel_uv":[1,2],"confidence":0.9}\n```',
        '{"pixel_uv":[1,2],"confidence":0.9,"candidate":2}',
        '{"pixel_uv":[1,2],"confidence":true}',
        '{"pixel_uv":[1,2],"pixel_uv":[2,3],"confidence":0.9}',
        '{"pixel_uv":[1,2],"confidence":1.1}',
        '{"pixel_uv":[1,2],"confidence":0.9} trailing',
        '[{"pixel_uv":[1,2],"confidence":0.9}]',
    ],
)
def test_strict_parser_rejects_ambiguous_or_non_schema_responses(content):
    with pytest.raises(LiberoGroundingError):
        OllamaVisualPixelLocator.parse_response(content)


def test_strict_parser_accepts_exact_schema_with_optional_bbox():
    selection = OllamaVisualPixelLocator.parse_response(
        '{"pixel_uv":[3,4],"confidence":0.75,"bbox":[1,2,5,6]}'
    )

    assert list(selection.pixel_uv) == [3, 4]
    assert selection.confidence == 0.75
    assert list(selection.bbox) == [1, 2, 5, 6]


@pytest.mark.parametrize(
    "selection,depth,error",
    [
        (_selection(pixel_uv=[5, 2], bbox=None), np.ones((5, 5)), "outside"),
        (_selection(confidence=0.49), np.ones((5, 5)), "confidence"),
        (_selection(), np.full((5, 5), np.nan), "valid depth"),
        (_selection(bbox=[0, 0, 1, 1]), np.ones((5, 5)), "bbox"),
    ],
)
def test_invalid_visual_evidence_fails_closed(selection, depth, error):
    with pytest.raises(LiberoGroundingError, match=error):
        ground_visual_instance(
            _observation(), "black bowl", _calibration(), depth, lambda *_: selection
        )


def test_provenance_is_auditable_and_contains_no_oracle_or_image():
    result = ground_visual_instance(
        _observation(),
        "black bowl",
        _calibration(),
        np.full((5, 5), 2.0),
        lambda *_: _selection(),
    )

    provenance = result["provenance"]
    assert provenance["privileged_segmentation"] is False
    assert provenance["coordinate_source"] == "rgbd_projection"
    assert provenance["locator_model"] == "fake-gemma"
    assert provenance["locator_prompt_hash"] == "abc123"
    assert provenance["pixel_uv"] == [2.0, 2.0]
    forbidden_keys = {
        "segmentation_instance",
        "instance_to_id",
        "object_pose",
        "image",
        "rgb",
    }
    assert not forbidden_keys & (set(result) | set(provenance))


class _FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_ollama_request_sends_base64_png_as_gemma_vision_image(monkeypatch):
    captured = {}

    def fake_urlopen(api_request, timeout):
        captured["url"] = api_request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(api_request.data.decode("utf-8"))
        return _FakeHTTPResponse(
            {"message": {"content": '{"pixel_uv":[2,2],"confidence":0.8}'}}
        )

    monkeypatch.setattr(visual.request, "urlopen", fake_urlopen)
    rgb = np.zeros((5, 5, 3), dtype=np.uint8)
    rgb[2, 2] = [12, 34, 56]
    locator = OllamaVisualPixelLocator("gemma3:4b", "http://ollama.test", timeout=3)

    selection = locator.locate(_observation(rgb), "small red mug")

    assert captured["url"] == "http://ollama.test/api/chat"
    payload = captured["payload"]
    assert payload["model"] == "gemma3:4b"
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["options"]["num_predict"] == 256
    assert payload["messages"][0]["role"] == "user"
    assert "small red mug" in payload["messages"][0]["content"]
    encoded = payload["messages"][0]["images"][0]
    png = base64.b64decode(encoded, validate=True)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    decoded = np.asarray(Image.open(BytesIO(png)).convert("RGB"))
    np.testing.assert_array_equal(decoded, rgb)
    np.testing.assert_allclose(selection.pixel_uv, [0.008, 0.008])
    assert selection.coordinate_transform == "normalized_1000_to_image_pixels"
    assert selection.locator_model == "gemma3:4b"
    assert len(selection.prompt_hash) == 64


def test_ollama_normalized_coordinates_are_converted_to_image_pixels(monkeypatch):
    monkeypatch.setattr(
        visual.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeHTTPResponse(
            {
                "message": {
                    "content": '{"pixel_uv":[500,250],"confidence":0.9,"bbox":[250,0,750,500]}'
                }
            }
        ),
    )
    locator = OllamaVisualPixelLocator("gemma4:12b")

    selection = locator.locate(
        _observation(np.zeros((101, 201, 3), dtype=np.uint8)), "black bowl"
    )

    np.testing.assert_allclose(selection.pixel_uv, [100.0, 25.0])
    np.testing.assert_allclose(selection.bbox, [50.0, 0.0, 150.0, 50.0])
