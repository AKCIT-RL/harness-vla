"""Run one isolated native LIBERO RGB-D grounding smoke."""

import argparse
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

from harness_vla.eval.libero_grounding_smoke import run_libero_grounding_smoke
from embodiedbench.evaluator.run_artifacts import create_run_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evaluation_runs",
    )
    parser.add_argument(
        "--test-id",
        default="libero_grounding_smoke_%s" % datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.initial_state_index < len(initial_states):
        raise IndexError("initial-state-index is outside the task's initial states")
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    run_root = create_run_root(args.output_root, args.test_id)
    env = SegmentationRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=args.resolution,
        camera_widths=args.resolution,
        camera_depths=True,
        camera_names=[args.camera],
    )
    try:
        env.seed(args.seed)
        result = run_libero_grounding_smoke(
            env=env,
            initial_state=initial_states[args.initial_state_index],
            run_root=run_root,
            task_suite=args.task_suite,
            task_id=args.task_id,
            initial_state_index=args.initial_state_index,
            seed=args.seed,
            camera=args.camera,
            height=args.resolution,
            width=args.resolution,
        )
        print("GROUNDED: %d" % result["summary"]["objects_grounded"])
        print("ARTIFACTS: %s" % run_root)
    finally:
        env.close()


if __name__ == "__main__":
    main()