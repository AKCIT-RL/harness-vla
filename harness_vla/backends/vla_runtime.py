"""Backend-neutral runtime contract for a frozen, chunk-producing VLA.

This module owns only the closed-loop execution semantics.  A future adapter
will supply a frozen visual policy through :class:`VLABackend`; the existing
scripted ``vla_act`` compiler is intentionally not adapted because its analytic
action sequence is not a real VLA chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Literal, Protocol, Tuple, TypeVar


ObservationT = TypeVar("ObservationT")
ChunkT = TypeVar("ChunkT")


class VLABackend(Protocol[ObservationT, ChunkT]):
    """Policy adapter that infers exactly one chunk from the live observation."""

    def infer_chunk(self, observation: ObservationT, prompt: str) -> ChunkT:
        """Return one action chunk without executing it."""
        ...


@dataclass(frozen=True)
class VLAChunkRecord(Generic[ObservationT, ChunkT]):
    """Audit record for one inferred and executed chunk."""

    chunk_index: int
    prompt: str
    observation_before: ObservationT
    chunk: ChunkT
    observation_after: ObservationT
    tau_satisfied: bool


@dataclass(frozen=True)
class VLARuntimeResult(Generic[ObservationT, ChunkT]):
    """Result of a bounded VLA call; benchmark task success is not interpreted."""

    chunks_requested: int
    chunks_executed: int
    tau_satisfied: bool
    termination_reason: Literal["tau_satisfied", "budget_exhausted"]
    latest_observation: ObservationT
    chunk_records: Tuple[VLAChunkRecord[ObservationT, ChunkT], ...]


class VLARuntime(Generic[ObservationT, ChunkT]):
    """Infer and execute fresh-observation chunks until ``tau`` or the hard cap."""

    def __init__(self, backend: VLABackend[ObservationT, ChunkT]) -> None:
        self.backend = backend

    def run(
        self,
        initial_observation: ObservationT,
        prompt: str,
        max_chunks: int,
        tau: Callable[[ObservationT], bool],
        executor: Callable[[ChunkT], ObservationT],
    ) -> VLARuntimeResult[ObservationT, ChunkT]:
        """Run a task-conditioned VLA call with a strict chunk budget.

        ``executor`` must execute its single chunk and return a newly observed
        state.  ``tau`` is evaluated on that state after every chunk.  Budget
        exhaustion is a continuation-capable termination cause, not a failure
        signal, and this runtime never reads or changes benchmark task success.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if (
            isinstance(max_chunks, bool)
            or not isinstance(max_chunks, int)
            or max_chunks <= 0
        ):
            raise ValueError("max_chunks must be an integer greater than zero")
        if not callable(tau):
            raise TypeError("tau must be callable")
        if not callable(executor):
            raise TypeError("executor must be callable")

        observation = initial_observation
        records = []
        for chunk_index in range(1, max_chunks + 1):
            observation_before = observation
            chunk = self.backend.infer_chunk(observation_before, prompt)
            observation = executor(chunk)
            satisfied = bool(tau(observation))
            records.append(
                VLAChunkRecord(
                    chunk_index=chunk_index,
                    prompt=prompt,
                    observation_before=observation_before,
                    chunk=chunk,
                    observation_after=observation,
                    tau_satisfied=satisfied,
                )
            )
            if satisfied:
                return VLARuntimeResult(
                    chunks_requested=max_chunks,
                    chunks_executed=chunk_index,
                    tau_satisfied=True,
                    termination_reason="tau_satisfied",
                    latest_observation=observation,
                    chunk_records=tuple(records),
                )

        return VLARuntimeResult(
            chunks_requested=max_chunks,
            chunks_executed=max_chunks,
            tau_satisfied=False,
            termination_reason="budget_exhausted",
            latest_observation=observation,
            chunk_records=tuple(records),
        )


__all__ = [
    "VLABackend",
    "VLAChunkRecord",
    "VLARuntime",
    "VLARuntimeResult",
]