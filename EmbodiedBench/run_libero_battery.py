"""Run the frozen multi-seed LIBERO battery (beta-only evaluation harness).

Usage (from EmbodiedBench/, with the LIBERO runtime environment active):

    python run_libero_battery.py --condition harness \
        --protocol ../benchmarks/eval_protocol_v1.json \
        --seeds 101 102 103

    python run_libero_battery.py --condition baseline_vla \
        --protocol ../benchmarks/eval_protocol_v1.json \
        --seeds 101 102 103

Episodes are paired by (task_suite, task_id, initial_state_index, seed);
compare conditions only over identical tuples. Re-running the same battery
root resumes: completed episodes are skipped.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

from embodiedbench.evaluator.libero_battery import (
    load_protocol,
    run_battery,
)
from embodiedbench.evaluator.libero_multi_turn_evaluator import (
    LiberoMultiTurnBudgets,
)
from embodiedbench.evaluator.libero_multi_turn_run import (
    run_libero_multi_turn_episode,
)
from embodiedbench.evaluator.libero_native_multi_turn import LiberoNativeOffsets
from embodiedbench.evaluator.libero_vla_smoke import run_libero_vla_smoke
from embodiedbench.planner.harness.libero_multi_turn_planner import (
    LiberoMultiTurnPlanner,
)
from embodiedbench.planner.harness.pirlinf_backend import PiRLinfWebsocketBackend


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "eval_protocol_v1.json",
    )
    parser.add_argument(
        "--condition", choices=("harness", "baseline_vla"), required=True
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--include-hidden", action="store_true")
    parser.add_argument(
        "--hidden-only",
        action="store_true",
        help="Run only the held-out hidden tasks (final evaluation).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evaluation_runs",
    )
    parser.add_argument(
        "--battery-id",
        default=None,
        help="Battery folder name; reuse an existing one to resume it.",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _create_env(spec, seed):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    from openpi_client import image_tools

    suite_type = benchmark.get_benchmark_dict()[spec["task_suite"]]
    suite = suite_type()
    task = suite.get_task(spec["task_id"])
    initial_states = suite.get_task_init_states(spec["task_id"])
    state_index = spec["initial_state_index"]
    if not 0 <= state_index < len(initial_states):
        raise IndexError("initial_state_index outside the task's initial states")
    bddl_path = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = SegmentationRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    env.seed(seed)
    return env, str(task.language), initial_states[state_index], image_tools


def make_episode_runner(args, protocol):
    planner_config = protocol.get("planner", {})
    backend_config = protocol.get("backend", {})
    budgets_config = protocol["budgets"]
    offsets_config = protocol.get("offsets", {})

    def episode_runner(spec, episode_root):
        env = None
        try:
            env, instruction, initial_state, image_tools = _create_env(
                spec, spec["seed"]
            )
            backend = PiRLinfWebsocketBackend(
                args.host,
                args.port,
                replan_steps=int(backend_config.get("replan_steps", 5)),
            )
            if spec["condition"] == "harness":
                planner = LiberoMultiTurnPlanner(
                    planner_config.get("model", "gemma4:12b"),
                    base_url=planner_config.get(
                        "base_url", "http://localhost:11434/v1"
                    ),
                    think=bool(planner_config.get("think", True)),
                    required_tau="lift_and_grasp",
                )
                budgets = LiberoMultiTurnBudgets(
                    max_turns=budgets_config["max_turns"],
                    horizon=budgets_config["horizon"],
                    max_chunks_cap=budgets_config["max_chunks_cap"],
                    max_move_steps=budgets_config["max_move_steps"],
                    release_steps=budgets_config["release_steps"],
                )
                offsets = LiberoNativeOffsets(
                    above_m=offsets_config.get("above_offset_m", 0.10),
                    release_pose_m=offsets_config.get(
                        "release_pose_offset_m", 0.05
                    ),
                )
                result = run_libero_multi_turn_episode(
                    env=env,
                    backend=backend,
                    planner=planner,
                    initial_state=initial_state,
                    instruction=instruction,
                    run_root=episode_root,
                    task_suite=spec["task_suite"],
                    task_id=spec["task_id"],
                    initial_state_index=spec["initial_state_index"],
                    seed=spec["seed"],
                    budgets=budgets,
                    offsets=offsets,
                    position_tolerance=offsets_config.get(
                        "position_tolerance_m", 0.01
                    ),
                    resize_with_pad=image_tools.resize_with_pad,
                    convert_to_uint8=image_tools.convert_to_uint8,
                )
                return result["episode"]
            result = run_libero_vla_smoke(
                env=env,
                backend=backend,
                initial_state=initial_state,
                prompt=instruction,
                run_root=episode_root,
                task_suite=spec["task_suite"],
                task_id=spec["task_id"],
                initial_state_index=spec["initial_state_index"],
                seed=spec["seed"],
                replan_steps=int(backend_config.get("replan_steps", 5)),
                max_chunks=budgets_config["max_chunks_cap"],
                horizon=budgets_config["horizon"],
                resize_with_pad=image_tools.resize_with_pad,
                convert_to_uint8=image_tools.convert_to_uint8,
                host=args.host,
                port=args.port,
            )
            return result["episode"]
        finally:
            if env is not None:
                env.close()

    return episode_runner


def main():
    args = parse_args()
    protocol = load_protocol(args.protocol)
    battery_id = args.battery_id or "battery_%s_%s_%s" % (
        protocol["protocol_name"],
        args.condition,
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    battery_root = args.output_root / battery_id
    outcome = run_battery(
        protocol,
        battery_root=battery_root,
        episode_runner=make_episode_runner(args, protocol),
        condition=args.condition,
        seeds=args.seeds,
        include_hidden=args.include_hidden,
        hidden_only=args.hidden_only,
        resume=not args.no_resume,
    )
    print(json.dumps(outcome["summary"], indent=2))
    print("ARTIFACTS: %s" % battery_root)


if __name__ == "__main__":
    main()
