"""Unit tests for the simulator-independent VLA chunk runtime."""

from dataclasses import dataclass

import pytest

from harness_vla import VLARuntime


@dataclass(frozen=True)
class _Observation:
    step: int
    task_success: bool = False


class _FakeBackend:
    def __init__(self):
        self.calls = []

    def infer_chunk(self, observation, prompt):
        self.calls.append((observation, prompt))
        return {"chunk": len(self.calls), "from_step": observation.step}


def _run(max_chunks=3, stop_at=None, task_success=False):
    backend = _FakeBackend()
    executed = []

    def execute(chunk):
        executed.append(chunk)
        return _Observation(step=len(executed), task_success=task_success)

    tau = lambda observation: stop_at is not None and observation.step >= stop_at
    result = VLARuntime(backend).run(
        _Observation(step=0, task_success=task_success),
        "grasp the red mug",
        max_chunks,
        tau,
        execute,
    )
    return backend, executed, result


def test_prompt_propagates_and_budget_executes_exactly_n_chunks():
    backend, executed, result = _run(max_chunks=3)

    assert [prompt for _, prompt in backend.calls] == ["grasp the red mug"] * 3
    assert len(executed) == 3
    assert result.chunks_requested == 3
    assert result.chunks_executed == 3
    assert len(result.chunk_records) == 3


def test_tau_early_return_stops_inference_at_k():
    backend, executed, result = _run(max_chunks=5, stop_at=2)

    assert len(backend.calls) == len(executed) == 2
    assert result.chunks_requested == 5
    assert result.chunks_executed == 2
    assert result.tau_satisfied is True
    assert result.termination_reason == "tau_satisfied"


def test_each_inference_receives_refreshed_observation():
    backend, _, result = _run(max_chunks=3)

    assert [observation.step for observation, _ in backend.calls] == [0, 1, 2]
    assert [record.observation_after.step for record in result.chunk_records] == [1, 2, 3]
    assert result.latest_observation.step == 3


def test_budget_exhaustion_is_distinct_and_does_not_change_task_success():
    _, _, result = _run(max_chunks=2, task_success=True)

    assert result.tau_satisfied is False
    assert result.termination_reason == "budget_exhausted"
    assert result.latest_observation.task_success is True
    assert not hasattr(result, "task_success")


@pytest.mark.parametrize(
    "prompt,max_chunks,error",
    [
        ("", 1, ValueError),
        ("   ", 1, ValueError),
        (None, 1, ValueError),
        ("valid", 0, ValueError),
        ("valid", -1, ValueError),
        ("valid", 1.5, ValueError),
        ("valid", True, ValueError),
    ],
)
def test_input_validation(prompt, max_chunks, error):
    with pytest.raises(error):
        VLARuntime(_FakeBackend()).run(
            _Observation(0),
            prompt,
            max_chunks,
            lambda observation: False,
            lambda chunk: _Observation(1),
        )


@pytest.mark.parametrize("argument", ["tau", "executor"])
def test_callbacks_must_be_callable(argument):
    kwargs = {
        "initial_observation": _Observation(0),
        "prompt": "valid",
        "max_chunks": 1,
        "tau": lambda observation: False,
        "executor": lambda chunk: _Observation(1),
    }
    kwargs[argument] = None
    with pytest.raises(TypeError):
        VLARuntime(_FakeBackend()).run(**kwargs)


def test_two_tau_predicates_terminate_at_different_chunks():
    backend_a, _, result_a = _run(max_chunks=5, stop_at=1)
    backend_b, _, result_b = _run(max_chunks=5, stop_at=4)

    assert result_a.chunks_executed == len(backend_a.calls) == 1
    assert result_b.chunks_executed == len(backend_b.calls) == 4