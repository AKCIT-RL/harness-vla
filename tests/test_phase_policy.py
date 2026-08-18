"""Simulator-free tests for bootstrap/deployment phase separation."""

import json

import pytest

from harness_vla.planner.phase_policy import (
    Phase,
    PhaseManifest,
    PhaseOperation,
    PhasePolicy,
    build_phase_manifest,
    validate_phase_manifest,
)


@pytest.fixture
def protocol_manifest():
    return build_phase_manifest(
        bootstrap_seed=0,
        evaluation_seeds=[15, 38],
        bootstrap_budget=8,
        deployment_budget=4,
    )


def test_protocol_fixture_has_paper_confirmed_phase_matrix(protocol_manifest):
    bootstrap = protocol_manifest.policy_for(Phase.BOOTSTRAP)
    deployment = protocol_manifest.policy_for(Phase.DEPLOYMENT)

    assert bootstrap.seeds == (0,)
    assert bootstrap.reset_allowed is True
    assert bootstrap.memory_write_allowed is True
    assert bootstrap.reportable is False
    assert deployment.seeds == (15, 38)
    assert deployment.reset_allowed is False
    assert deployment.memory_write_allowed is False
    assert deployment.reportable is True
    assert bootstrap.budget >= deployment.budget


@pytest.mark.parametrize(
    "phase,seed,operation",
    [
        (Phase.BOOTSTRAP, 0, PhaseOperation.RESET),
        ("bootstrap", 0, PhaseOperation.READ_MEMORY),
        (Phase.BOOTSTRAP, 0, PhaseOperation.WRITE_MEMORY),
        (Phase.DEPLOYMENT, 15, PhaseOperation.READ_MEMORY),
        ("deployment", 15, PhaseOperation.REPORT_METRIC),
        (Phase.DEPLOYMENT, 38, PhaseOperation.REPORT_METRIC),
    ],
)
def test_allowed_operation_matrix(protocol_manifest, phase, seed, operation):
    assert protocol_manifest.guard_operation(phase, seed, operation) is None


def test_unknown_phase_is_rejected(protocol_manifest):
    with pytest.raises(ValueError, match="unknown phase"):
        protocol_manifest.policy_for("training")


def test_bootstrap_and_evaluation_seeds_must_be_disjoint():
    with pytest.raises(ValueError, match="must not appear"):
        build_phase_manifest(0, [0, 15], 8, 4)


@pytest.mark.parametrize(
    "operation,error",
    [
        (PhaseOperation.RESET, "reset is forbidden"),
        (PhaseOperation.WRITE_MEMORY, "memory writes are forbidden"),
    ],
)
def test_deployment_rejects_mutating_operations(protocol_manifest, operation, error):
    with pytest.raises(ValueError, match=error):
        protocol_manifest.guard_operation(Phase.DEPLOYMENT, 15, operation)


def test_bootstrap_seed_cannot_be_reported(protocol_manifest):
    with pytest.raises(ValueError, match="must not be included"):
        protocol_manifest.guard_operation(Phase.BOOTSTRAP, 0, PhaseOperation.REPORT_METRIC)


@pytest.mark.parametrize(
    "phase,seed",
    [(Phase.BOOTSTRAP, 15), (Phase.DEPLOYMENT, 0)],
)
def test_seed_must_belong_to_requested_phase(protocol_manifest, phase, seed):
    with pytest.raises(ValueError, match="does not belong"):
        protocol_manifest.guard_operation(phase, seed, PhaseOperation.READ_MEMORY)


def test_bootstrap_budget_cannot_be_smaller_than_deployment_budget():
    with pytest.raises(ValueError, match="greater than or equal"):
        build_phase_manifest(0, [15, 38], 3, 4)


def test_phase_permissions_cannot_be_configured_to_leak():
    with pytest.raises(ValueError, match="permissions do not match deployment"):
        PhasePolicy(Phase.DEPLOYMENT, (15,), 4, True, True, True)
    with pytest.raises(ValueError, match="permissions do not match bootstrap"):
        PhasePolicy(Phase.BOOTSTRAP, (0,), 8, True, True, True)


def test_manifest_serialization_is_deterministic(protocol_manifest):
    expected = (
        '{"bootstrap_budget":8,"bootstrap_seed":0,'
        '"deployment_budget":4,"evaluation_seeds":[15,38]}'
    )
    assert protocol_manifest.to_json() == expected
    assert json.loads(protocol_manifest.to_json()) == protocol_manifest.to_dict()
    assert validate_phase_manifest(protocol_manifest) is None


def test_manifest_does_not_alias_mutable_seed_input():
    evaluation_seeds = [15, 38]
    manifest = build_phase_manifest(0, evaluation_seeds, 8, 4)

    evaluation_seeds.append(0)

    assert manifest.evaluation_seeds == (15, 38)


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"evaluation_seeds": []}, "must not be empty"),
        ({"evaluation_seeds": [15, 15]}, "must be unique"),
        ({"evaluation_seeds": [15, "38"]}, "must be an integer"),
        ({"bootstrap_budget": 0}, "greater than zero"),
        ({"deployment_budget": True}, "greater than zero"),
    ],
)
def test_manifest_rejects_malformed_values(kwargs, error):
    values = {
        "bootstrap_seed": 0,
        "evaluation_seeds": [15, 38],
        "bootstrap_budget": 8,
        "deployment_budget": 4,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=error):
        build_phase_manifest(**values)


def test_unknown_operation_is_rejected(protocol_manifest):
    with pytest.raises(ValueError, match="unknown phase operation"):
        protocol_manifest.guard_operation(Phase.DEPLOYMENT, 15, "fine_tune")


def test_manifest_type_guard():
    with pytest.raises(TypeError, match="PhaseManifest"):
        validate_phase_manifest({})


def test_direct_manifest_construction_is_validated():
    with pytest.raises(ValueError, match="must not appear"):
        PhaseManifest(0, (0, 15), 8, 4)