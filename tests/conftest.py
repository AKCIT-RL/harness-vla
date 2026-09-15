"""Pytest configuration: make harness_vla and the vendored EmbodiedBench importable."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_EB_ROOT = os.path.join(_REPO_ROOT, "EmbodiedBench")
# Environment contract (partnr_harness.contracts) when checked out as submodule.
_PARTNR_ENV_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))
for _p in (_REPO_ROOT, _EB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if (
    os.path.isdir(os.path.join(_PARTNR_ENV_ROOT, "partnr_harness"))
    and _PARTNR_ENV_ROOT not in sys.path
):
    sys.path.append(_PARTNR_ENV_ROOT)
