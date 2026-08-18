import pytest

from harness_vla.eval.libero_privileged_diagnostics import (
    LiberoPrivilegedDiagnostics,
)


class _Model:
    def __init__(self):
        self.positions = {
            "bowl_body": [0.2, 0.1, 0.7],
            "plate_body": [0.1, -0.1, 0.6],
        }

    def body_name2id(self, name):
        return list(self.positions).index(name)


class _Data:
    body_xpos = [[0.2, 0.1, 0.7], [0.1, -0.1, 0.6]]
    body_xquat = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    site_xpos = [[0.3, 0.2, 0.8]]
    site_xmat = [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]


class _Object:
    def __init__(self, name, root_body, radius):
        self.name = name
        self.root_body = root_body
        self.horizontal_radius = radius
        self.contact_geoms = [name + "_geom"]


class _Gripper:
    important_geoms = {
        "left_fingerpad": ["left"],
        "right_fingerpad": ["right"],
    }


class _Robot:
    eef_site_id = 0
    gripper = _Gripper()


class _Inner:
    def __init__(self):
        self.sim = type("Sim", (), {"model": _Model(), "data": _Data()})()
        self.objects = [
            _Object("bowl", "bowl_body", 0.05),
            _Object("plate", "plate_body", 0.3),
        ]
        self.robots = [_Robot()]

    def check_contact(self, first, second):
        return first in (["left"], ["right"]) or getattr(first, "name", None) == "bowl"


def test_snapshot_calculates_native_poses_delta_contact_and_on_clauses():
    env = type("Env", (), {"env": _Inner()})()
    diagnostics = LiberoPrivilegedDiagnostics(
        env,
        available_targets=["bowl", "plate"],
        object_roles={"bowl": ["manipulable"], "plate": ["destination"]},
    )

    snapshot = diagnostics("post_grasp", {"action": "vla_act", "target": "bowl"}, "bowl")

    assert snapshot["source"] == "privileged_mujoco_state"
    assert snapshot["beta_only"] is True
    assert snapshot["eef_pose"]["position"] == [0.3, 0.2, 0.8]
    assert snapshot["holding"]["pose"]["position"] == [0.2, 0.1, 0.7]
    assert snapshot["destination"]["pose"]["position"] == [0.1, -0.1, 0.6]
    assert snapshot["delta_xy_m"] == pytest.approx(0.2236067977)
    assert snapshot["contact"] == {
        "left_finger": True,
        "right_finger": True,
        "bilateral": True,
    }
    assert snapshot["on_predicate"] == {
        "predicate": "On",
        "clauses": {
            "object_above_destination": True,
            "horizontal_center_within_0_03_m": False,
            "objects_in_contact": True,
        },
        "value": False,
        "errors": None,
    }


def test_unreadable_clause_is_null_and_does_not_raise():
    env = type("Env", (), {"env": _Inner()})()
    env.env.objects[1].root_body = None
    diagnostics = LiberoPrivilegedDiagnostics(
        env,
        available_targets=["bowl", "plate"],
        object_roles={"plate": ["destination"]},
    )

    snapshot = diagnostics("pre_release", {"action": "release"}, "bowl")

    assert snapshot["on_predicate"]["value"] is None
    assert snapshot["on_predicate"]["clauses"][
        "horizontal_center_within_0_03_m"
    ] is None
    assert "objects" in snapshot["on_predicate"]["errors"]