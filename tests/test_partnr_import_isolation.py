"""Agent side of the PARTNR separation must not reach into the simulator.

``harness_vla.envs.partnr`` may import only the agent contract
(``akc_harness.contracts`` / ``akc_harness.signals``) and, for loop-driving
clients, the public reset/step types in ``partnr_harness.external``. Never
``habitat_llm`` and never the PARTNR env internals (bridge/centralized
planners, state, queue).
"""

import ast
import os
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "harness_vla" / "envs" / "partnr"
ALLOWED_ENV_MODULES = {"partnr_harness.external"}
ALLOWED_CONTRACT_MODULES = {"akc_harness.contracts", "akc_harness.signals"}
FORBIDDEN_PREFIXES = ("habitat_llm", "habitat", "magnum")


def _modules():
    for root, _dirs, files in os.walk(PACKAGE):
        for name in files:
            if name.endswith(".py"):
                yield Path(root) / name


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


@pytest.mark.parametrize("path", sorted(_modules()), ids=lambda p: str(p.relative_to(PACKAGE)))
def test_partnr_agent_module_does_not_import_simulator_or_env_internals(path):
    for module in _imports(path):
        assert not module.startswith(FORBIDDEN_PREFIXES), f"{path.name} imports {module}"
        if module == "partnr_harness" or module.startswith("partnr_harness."):
            assert module in ALLOWED_ENV_MODULES, f"{path.name} imports env internals {module}"
        if module == "akc_harness" or module.startswith("akc_harness."):
            assert module in ALLOWED_CONTRACT_MODULES, f"{path.name} imports {module}"


def test_grep_no_habitat_llm_anywhere_in_partnr_agent_code():
    offenders = []
    for path in _modules():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("import habitat_llm", "from habitat_llm")):
                offenders.append(f"{path.relative_to(PACKAGE)}: {stripped}")
    assert offenders == []


def test_agent_configs_target_agent_or_env_public_planners():
    """Yaml ``_target_``s point to agent modules or the env's public planners."""
    conf = PACKAGE / "conf"
    assert (conf / "__init__.py").is_file(), "pkg://harness_vla.envs.partnr.conf needs __init__.py"
    allowed_env_targets = {
        "partnr_harness.bridge_planner.HarnessBridgePlanner",
        "partnr_harness.centralized_planner.CentralizedHarnessPlanner",
        "partnr_harness.external.ExternalLoopHarness",
    }
    for path in conf.rglob("*.yaml"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "_target_" not in line:
                continue
            target = line.split(":", 1)[1].strip().strip("'\"")
            if target.startswith("partnr_harness."):
                assert target in allowed_env_targets, f"{path.name}: {target}"
            else:
                assert target.startswith("harness_vla."), f"{path.name}: {target}"
            assert "partnr_harness.examples" not in target
