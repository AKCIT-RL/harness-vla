"""Pytest configuration: make harness_vla and the vendored EmbodiedBench importable."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_EB_ROOT = os.path.join(_REPO_ROOT, "EmbodiedBench")
for _p in (_REPO_ROOT, _EB_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
