"""Path bootstrap so runners work from any CWD (repo root + vendored fork)."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "EmbodiedBench")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
