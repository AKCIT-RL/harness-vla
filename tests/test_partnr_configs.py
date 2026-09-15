"""Composition checks for the agent-side Hydra configs (OmegaConf only, no Hydra runtime).

Moved from ``partnr_harness/tests`` when the agent configs left the env repo.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / "harness_vla" / "envs" / "partnr" / "conf"
BRIDGE = "partnr_harness.bridge_planner.HarnessBridgePlanner"

PROFILES = {
    "gemma4_12b_off": ("gemma4:12b", False, True),
    "gemma4_12b_on": ("gemma4:12b", True, False),
    "gemma3_4b_off": ("gemma3:4b", False, True),
    "gemma3_1b_off": ("gemma3:1b", False, True),
}
CONTROLLERS = {
    "planner/harness_primitive_planner": "plan_config.harness",
    "planner/harness_react_planner": "plan_config.harness",
    "llm/ollama_chat": "",
}


def _baseline(name):
    config = OmegaConf.load(CONFIG_DIR / "baselines" / f"{name}.yaml")
    return config, list(config.defaults)


def _planner(name, profile="gemma4_12b_off_4k"):
    planner = OmegaConf.load(CONFIG_DIR / "planner" / f"{name}.yaml")
    profile_config = OmegaConf.load(CONFIG_DIR / "model_pool" / f"{profile}.yaml")
    merged = OmegaConf.merge({"model_pool": profile_config}, planner)
    OmegaConf.resolve(merged)
    return merged


# ------------------------------------------------------------ model pool

@pytest.mark.parametrize("profile,expected", PROFILES.items())
@pytest.mark.parametrize("config_name,controller_path", CONTROLLERS.items())
def test_model_pool_composes_into_each_controller(profile, expected, config_name, controller_path):
    profile_config = OmegaConf.load(CONFIG_DIR / "model_pool" / f"{profile}.yaml")
    controller_config = OmegaConf.load(CONFIG_DIR / f"{config_name}.yaml")
    config = OmegaConf.merge({"model_pool": profile_config}, controller_config)
    OmegaConf.resolve(config)

    controller = OmegaConf.select(config, controller_path) if controller_path else config
    is_llm = config_name.startswith("llm/")
    assert OmegaConf.select(controller, "generation_params.model" if is_llm else "model") == expected[0]
    assert controller.enable_thinking is expected[1]
    assert controller.disable_thinking is expected[2]
    assert controller.num_ctx == profile_config.num_ctx
    assert OmegaConf.select(controller, "generation_params.max_tokens" if is_llm else "max_tokens") == profile_config.max_tokens
    assert OmegaConf.select(controller, "request_timeout" if is_llm else "timeout_s") == 240


@pytest.mark.parametrize("baseline", ("single_robot_primitive_harness", "single_agent_react_ollama"))
def test_beta_baselines_select_default_model_pool_profile(baseline):
    config, defaults = _baseline(baseline)
    assert {"/model_pool": "gemma3_1b_off"} in defaults


# ---------------------------------------------------------------- planners

def test_random_planner_is_the_bridge_with_random_harness():
    planner = OmegaConf.load(CONFIG_DIR / "planner" / "harness_random_planner.yaml")
    assert planner._target_ == BRIDGE
    assert planner.plan_config.harness._target_ == "harness_vla.envs.partnr.baselines.random_harness.RandomHarness"
    assert planner.plan_config.expose_bootstrap_label is False


def test_external_random_planner_targets_agent_side_loop_client():
    planner = OmegaConf.load(CONFIG_DIR / "planner" / "harness_external_random_planner.yaml")
    harness = planner.plan_config.harness
    assert planner._target_ == BRIDGE
    assert planner.plan_config.max_decisions == 8
    assert harness._target_ == "partnr_harness.external.ExternalLoopHarness"
    assert harness.agent_target == "harness_vla.envs.partnr.baselines.random_loop.RandomLoopAgent"
    assert OmegaConf.to_container(harness.agent_kwargs) == {"max_steps": 6, "seed": 0}


def test_react_planner_shares_llm_knobs_with_primitive_planner():
    primitive = OmegaConf.to_container(_planner("harness_primitive_planner").plan_config.harness)
    react = OmegaConf.to_container(_planner("harness_react_planner").plan_config.harness)
    assert react.pop("_target_") == "harness_vla.envs.partnr.baselines.react.ReActHarness"
    primitive.pop("_target_")
    primitive.pop("prefix")
    assert react.pop("max_transcript_chars") == 0
    assert react == primitive


def test_harness_vla_planner_matches_primitive_kwargs_plus_memory_knobs():
    p = OmegaConf.to_container(_planner("harness_primitive_planner").plan_config.harness)
    v = OmegaConf.to_container(_planner("harness_vla_planner").plan_config.harness)
    assert v.pop("_target_") == "harness_vla.envs.partnr.adapter.PartnrHarnessAdapter"
    p.pop("_target_")
    extra = {k: v.pop(k) for k in ("max_history", "zero_shot", "task_memory_dir", "global_memory_path")}
    assert extra == {"max_history": 0, "zero_shot": True, "task_memory_dir": None, "global_memory_path": None}
    assert v == p


def test_centralized_planner_config_resolves_model_pool():
    config = _planner("harness_centralized_primitive_planner", profile="gemma4_12b_on")
    assert config._target_ == "partnr_harness.centralized_planner.CentralizedHarnessPlanner"
    assert (
        config.plan_config.harness._target_
        == "harness_vla.envs.partnr.legacy.centralized_primitive_harness.CentralizedPrimitiveHarness"
    )
    assert config.plan_config.harness.model == "gemma4:12b"
    assert config.plan_config.harness.enable_thinking is True
    assert config.plan_config.harness.num_ctx == 16384
    assert config.plan_config.harness.temperature == 0.6
    assert config.plan_config.harness.thinking_budget == 2048


# --------------------------------------------------------------- baselines

@pytest.mark.parametrize(
    "baseline,planner",
    [
        ("two_agent_external_random", "harness_external_random_planner"),
        ("two_agent_harness_vla", "harness_vla_planner"),
        ("harness_demo", "harness_random_planner"),
        ("two_robot_harness", "harness_random_planner"),
    ],
)
def test_two_agent_baselines_use_the_planner_for_both_agents(baseline, planner):
    config, defaults = _baseline(baseline)
    for agent in ("agent_0", "agent_1"):
        assert {f"override /planner@evaluation.agents.{agent}.planner": planner} in defaults
    if "model_pool" in {list(d)[0].lstrip("/") for d in defaults if isinstance(d, dict)}:
        assert {"/model_pool": "gemma4_12b_off_4k"} in defaults


def test_no_baseline_references_the_env_bridge_without_a_harness():
    """``harness_bridge_planner`` (env side) has ``harness: ???``: agent yamls must not select it bare."""
    for path in (CONFIG_DIR / "baselines").glob("*.yaml"):
        assert "harness_bridge_planner" not in path.read_text(encoding="utf-8"), path.name


def test_centralized_baseline_overrides():
    config, defaults = _baseline("two_agent_centralized_harness")
    assert {"/model_pool": "gemma4_12b_off_4k"} in defaults
    assert {"override /evaluation": "centralized_evaluation_runner_multi_agent"} in defaults
    assert {"override /planner@evaluation.planner": "harness_centralized_primitive_planner"} in defaults
    for agent in ("agent_0", "agent_1"):
        assert {f"override /agent@evaluation.agents.{agent}.config": "harness_primitive_agent"} in defaults
    assert config.world_model.partial_obs is False
    assert config.evaluation.save_video is True
