"""Run one native LIBERO task/state through the frozen PiRLinf VLA."""

import argparse
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from harness_vla.eval.libero_vla_smoke import run_libero_vla_smoke
from embodiedbench.evaluator.run_artifacts import create_run_root
from harness_vla.planner.libero_vla_planner import LiberoVLAPlanner
from harness_vla.backends.pirlinf_backend import PiRLinfWebsocketBackend


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=220)
    parser.add_argument("--planner-model", default="")
    parser.add_argument(
        "--planner-base-url", default="http://localhost:11434/v1"
    )
    parser.add_argument("--think", action="store_true")
    parser.add_argument(
        "--planner-tau",
        choices=("task_success", "lift_and_grasp"),
        default="task_success",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evaluation_runs",
    )
    parser.add_argument(
        "--test-id",
        default="libero_vla_smoke_%s" % datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    return parser.parse_args()


def create_libero_episode(args):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    from openpi_client import image_tools

    suite_type = benchmark.get_benchmark_dict()[args.task_suite]
    suite = suite_type()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.initial_state_index < len(initial_states):
        raise IndexError("initial-state-index is outside the task's initial states")
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = SegmentationRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    env.seed(args.seed)
    return env, task.language, initial_states[args.initial_state_index], image_tools


def main():
    args = parse_args()
    run_root = create_run_root(args.output_root, args.test_id)
    env = None
    try:
        env, prompt, initial_state, image_tools = create_libero_episode(args)
        backend = PiRLinfWebsocketBackend(
            args.host,
            args.port,
            replan_steps=args.replan_steps,
        )
        planner = None
        if args.planner_model:
            planner = LiberoVLAPlanner(
                args.planner_model,
                base_url=args.planner_base_url,
                think=args.think,
                required_tau=args.planner_tau,
            )
        result = run_libero_vla_smoke(
            env=env,
            backend=backend,
            initial_state=initial_state,
            prompt=str(prompt),
            run_root=run_root,
            task_suite=args.task_suite,
            task_id=args.task_id,
            initial_state_index=args.initial_state_index,
            seed=args.seed,
            replan_steps=args.replan_steps,
            max_chunks=args.max_chunks,
            horizon=args.horizon,
            resize_with_pad=image_tools.resize_with_pad,
            convert_to_uint8=image_tools.convert_to_uint8,
            planner=planner,
            host=args.host,
            port=args.port,
        )
        print("SUCCESS: %s" % result["episode"]["task_success"])
        print("ARTIFACTS: %s" % run_root)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()