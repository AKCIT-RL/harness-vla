import pytest

from harness_vla.primitives.libero_tau import (
    LiberoTauError,
    LiftAndGraspTau,
    read_bilateral_contact,
)


class _Model:
    def __init__(self, name):
        self.name = name
        self.contact_geoms = ["%s_geom" % name]


class _Gripper:
    important_geoms = {
        "left_fingerpad": ["left_pad"],
        "right_fingerpad": ["right_pad"],
    }


class _Robot:
    gripper = _Gripper()


class _InnerEnv:
    objects = [_Model("target_1"), _Model("distractor_1")]
    robots = [_Robot()]

    def __init__(self, contacts):
        self.contacts = contacts

    def check_contact(self, finger_geoms, object_geoms):
        return (tuple(finger_geoms), tuple(object_geoms)) in self.contacts


class _Wrapper:
    def __init__(self, contacts):
        self.env = _InnerEnv(contacts)


@pytest.mark.parametrize(
    "left,right,current_z,expected",
    [
        (True, True, 0.94, True),
        (True, True, 0.91, False),
        (True, False, 0.94, False),
        (False, True, 0.94, False),
        (False, False, 0.94, False),
    ],
)
def test_lift_and_grasp_requires_bilateral_contact_and_lift(
    left, right, current_z, expected
):
    tau = LiftAndGraspTau(baseline_target_z_m=0.90, minimum_lift_m=0.03)

    evidence = tau.evaluate(
        current_target_z_m=current_z,
        left_finger_contact=left,
        right_finger_contact=right,
    )

    assert evidence["tau_satisfied"] is expected
    assert evidence["contact"]["bilateral"] is (left and right)
    assert evidence["lift"]["delta_z_m"] == pytest.approx(current_z - 0.90)
    assert evidence["task_success_evaluated"] is False


def test_lift_threshold_is_inclusive_and_explicit():
    tau = LiftAndGraspTau(baseline_target_z_m=0.90, minimum_lift_m=0.03)

    evidence = tau.evaluate(
        current_target_z_m=0.93,
        left_finger_contact=True,
        right_finger_contact=True,
    )

    assert evidence["tau_satisfied"] is True
    assert evidence["lift"]["minimum_lift_m"] == 0.03
    assert evidence["lift"]["coordinate_source"] == "rgbd_projection"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LiftAndGraspTau(float("nan"), 0.03),
        lambda: LiftAndGraspTau(0.90, 0.0),
        lambda: LiftAndGraspTau(0.90, -0.01),
    ],
)
def test_invalid_tau_configuration_fails_closed(factory):
    with pytest.raises(LiberoTauError):
        factory()


def test_invalid_evidence_fails_closed():
    tau = LiftAndGraspTau(0.90, 0.03)
    with pytest.raises(LiberoTauError):
        tau.evaluate(
            current_target_z_m=float("nan"),
            left_finger_contact=True,
            right_finger_contact=True,
        )
    with pytest.raises(LiberoTauError):
        tau.evaluate(
            current_target_z_m=0.94,
            left_finger_contact=1,
            right_finger_contact=True,
        )


def test_contact_adapter_reads_each_pad_against_exact_target():
    contacts = {
        (("left_pad",), ("target_1_geom",)),
        (("right_pad",), ("target_1_geom",)),
    }

    evidence = read_bilateral_contact(_Wrapper(contacts), "target_1")

    assert evidence == {
        "target_instance": "target_1",
        "left_finger_contact": True,
        "right_finger_contact": True,
        "bilateral_contact": True,
        "source": "robosuite_check_contact",
        "privileged_contact_state": True,
    }


@pytest.mark.parametrize("target", ["", "missing"])
def test_contact_adapter_rejects_invalid_or_missing_target(target):
    with pytest.raises(LiberoTauError):
        read_bilateral_contact(_Wrapper(set()), target)