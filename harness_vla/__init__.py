"""Harness VLA beta extension for EmbodiedBench (EB-Manipulation).

A memory-guided agentic planner that composes a small fixed library of analytic
primitives with a single retryable contact-rich primitive (``vla_act``). This is
a simplified, in-process, zero-shot beta of the Harness VLA framework
(arXiv:2607.08448) used as a sanity check on top of EB-Manipulation.

See ``PROVENANCE.md`` and ``docs/HARNESS_VLA_BETA_REPORT.md`` at the repo root
for what is intentionally left out of this beta.
"""

from harness_vla.primitives.primitives import (
    PrimitiveLibrary,
    PrimitiveError,
    PoseState,
)
from harness_vla.memory.global_memory import (
    GlobalMemory,
    GlobalMemoryEvidence,
    GlobalMemoryLedger,
    evidence_from_completed_trace,
)
from harness_vla.backends.vla_runtime import (
    VLABackend,
    VLAChunkRecord,
    VLARuntime,
    VLARuntimeResult,
)

__all__ = [
    "PrimitiveLibrary",
    "PrimitiveError",
    "PoseState",
    "GlobalMemory",
    "GlobalMemoryEvidence",
    "GlobalMemoryLedger",
    "evidence_from_completed_trace",
    "VLABackend",
    "VLAChunkRecord",
    "VLARuntime",
    "VLARuntimeResult",
]
