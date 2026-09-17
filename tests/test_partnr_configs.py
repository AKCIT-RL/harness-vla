"""Composition checks for the agent-side Hydra configs (OmegaConf only, no Hydra runtime)."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / "harness_vla" / "envs" / "partnr" / "conf"
BRIDGE = "partnr_harness.bridge_planner.HarnessBridgePlanner"
CENTRAL = "partnr_harness.centralized_planner.CentralizedHarnessPlanner"

PROFILES = {
    "gemma4_12b_off": ("gemma4:12b", False, True),
    "gemma4_12b_on": ("gemma4:12b", True, False),
    "gemma3_4b_off": ("gemma3:4b", False, True),
    "gemma3_1b_off": ("gemma3:1b", False, True),
}
CONTROLLERS = {
    "planner/harness_vla_central": "plan_config.harness",
    "planner/harness_vla_decentral": "plan_config.harness",
    "llm/ollama_chat": "",
}
LLM_KNOBS = (
    "base_url", "model", "temperature", "repeat_penalty", "repeat_last_n", "thinking_budget",
    "max_tokens", "timeout_s", "world_char_budget", "enable_thinking", "disable_thinking",
    "num_ctx", "max_retries", "max_history", "zero_shot", "phase", "task_memory_dir",
    "global_memory_path",
)


def _baseline(name):
    config = OmegaConf.load(CONFIG_DIR / "baselines" / f"{name}.yaml")
    return config, list(config.defaults)


def _planner(name, profile="gemma4_12b_off_4k"):
    planner = OmegaConf.load(CONFIG_DIR / "planner" / f"{name}.yaml")
    profile_config = OmegaConf.load(CONFIG_DIR / "model_pool" / f"{profile}.yaml")
    return OmegaConf.merge({"model_pool": profile_config}, planner)


def _raw(container, key):
    """Unresolved value (interpolation strings kept verbatim)."""
    return OmegaConf.to_container(container, resolve=False)[key]


# ------------------------------------------------------------ model pool

@pytest.mark.parametrize("profile,expected", PROFILES.items())
@pytest.mark.parametrize("config_name,controller_path", CONTROLLERS.items())
def test_model_pool_composes_into_each_controller(profile, expected, config_name, controller_path):
    profile_config = OmegaConf.load(CONFIG_DIR / "model_pool" / f"{profile}.yaml")
    controller_config = OmegaConf.load(CONFIG_DIR / f"{config_name}.yaml")
    config = OmegaConf.merge({"model_pool": profile_config}, controller_config)
    if "hydra" not in config:
        # ${hydra:runtime.output_dir} only resolves inside a Hydra run.
        controller = OmegaConf.select(config, controller_path) if controller_path else config
        if "shared_memory_dir" in controller:
            controller.shared_memory_dir = "/tmp/memory"
    OmegaConf.resolve(config)

    controller = OmegaConf.select(config, controller_path) if controller_path else config
    is_llm = config_name.startswith("llm/")
    assert OmegaConf.select(controller, "generation_params.model" if is_llm else "model") == expected[0]
    assert controller.enable_thinking is expected[1]
    assert controller.disable_thinking is expected[2]
    assert controller.num_ctx == profile_config.num_ctx
    assert OmegaConf.select(controller, "generation_params.max_tokens" if is_llm else "max_tokens") == profile_config.max_tokens
    assert OmegaConf.select(controller, "request_timeout" if is_llm else "timeout_s") == 240


# ---------------------------------------------------------------- planners

def test_central_planner_targets_env_centralized_planner_with_vla_controller():
    planner = _planner("harness_vla_central")
    harness = planner.plan_config.harness
    assert planner._target_ == CENTRAL
    assert harness._target_ == "harness_vla.envs.partnr.agent.HarnessVlaCentralized"
    assert harness.use_claims is True
    assert planner.plan_config.max_decisions == 100
    assert planner.plan_config.expose_bootstrap_label is False
    assert harness.max_history == 0 and harness.zero_shot is True and harness.phase == "deployment"


def test_decentral_planner_is_bridge_with_per_agent_vla_brain():
    planner = _planner("harness_vla_decentral")
    harness = planner.plan_config.harness
    assert planner._target_ == BRIDGE
    assert harness._target_ == "harness_vla.envs.partnr.decentralized.HarnessVlaDecentralized"
    assert "${hydra:runtime.output_dir}" in _raw(harness, "shared_memory_dir")
    assert planner.plan_config.symmetric_partner_snapshot is True
    assert planner.plan_config.max_decisions == 50


def test_central_and_decentral_share_the_llm_and_memory_knobs():
    central = OmegaConf.load(CONFIG_DIR / "planner" / "harness_vla_central.yaml").plan_config.harness
    decentral = OmegaConf.load(CONFIG_DIR / "planner" / "harness_vla_decentral.yaml").plan_config.harness
    for knob in LLM_KNOBS:
        assert knob in central and knob in decentral, knob
        assert _raw(central, knob) == _raw(decentral, knob), knob


def test_external_random_planner_targets_agent_side_loop_client():
    planner = OmegaConf.load(CONFIG_DIR / "planner" / "harness_external_random_planner.yaml")
    harness = planner.plan_config.harness
    assert planner._target_ == BRIDGE
    assert planner.plan_config.max_decisions == 8
    assert harness._target_ == "partnr_harness.external.ExternalLoopHarness"
    assert harness.agent_target == "harness_vla.envs.partnr.baselines.random_loop.RandomLoopAgent"


def test_react_llm_targets_agent_side_openai_compat_client():
    llm = OmegaConf.load(CONFIG_DIR / "llm" / "ollama_chat.yaml")
    assert llm.llm._target_ == "harness_vla.envs.partnr.react_llm.OpenAICompatChat"


# --------------------------------------------------------------- baselines

def test_only_three_two_agent_baselines_plus_env_smoke_exist():
    names = sorted(p.stem for p in (CONFIG_DIR / "baselines").glob("*.yaml"))
    assert names == [
        "two_agent_external_random", "two_agent_react_ollama",
        "two_agent_vla_central", "two_agent_vla_decentral",
    ]
    planners = sorted(p.stem for p in (CONFIG_DIR / "planner").glob("*.yaml"))
    assert planners == ["harness_external_random_planner", "harness_vla_central", "harness_vla_decentral"]


@pytest.mark.parametrize("baseline,planner", [
    ("two_agent_vla_decentral", "harness_vla_decentral"),
    ("two_agent_external_random", "harness_external_random_planner"),
])
def test_decentralized_baselines_use_the_planner_for_both_agents(baseline, planner):
    config, defaults = _baseline(baseline)
    assert {"override /evaluation": "decentralized_evaluation_runner_multi_agent"} in defaults
    for label in ("agent_0", "agent_1"):
        assert {f"override /planner@evaluation.agents.{label}.planner": planner} in defaults
        assert {f"override /agent@evaluation.agents.{label}.config": "harness_primitive_agent"} in defaults
    assert config.world_model.partial_obs is False


def test_central_baseline_uses_the_centralized_runner():
    config, defaults = _baseline("two_agent_vla_central")
    assert {"override /evaluation": "centralized_evaluation_runner_multi_agent"} in defaults
    assert {"override /planner@evaluation.planner": "harness_vla_central"} in defaults
    assert {"/model_pool": "gemma4_12b_off_4k"} in defaults
    assert config.evaluation.save_video is False


def test_react_baseline_is_upstream_react_with_oracle_skills():
    _config, defaults = _baseline("two_agent_react_ollama")
    assert {"override /planner@evaluation.agents.agent_0.planner": "llm_zero_shot_react_planner"} in defaults
    assert {"override /llm@evaluation.agents.agent_0.planner.plan_config.llm": "ollama_chat"} in defaults
    assert {"/model_pool": "gemma4_12b_off_4k"} in defaults
