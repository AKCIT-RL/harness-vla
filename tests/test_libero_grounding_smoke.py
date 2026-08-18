import json

import numpy as np

from harness_vla.eval.libero_grounding_smoke import (
    LIBERO_GROUNDING_DUMMY_ACTION,
    run_libero_grounding_smoke,
)


class _FakeSim:
    class Model:
        cam_fovy = np.asarray([90.0])
        body_names = ["object_1_main"]

        class Stat:
            extent = 2.0

        class Vis:
            class Map:
                znear = 0.05
                zfar = 5.0

            map = Map()

        stat = Stat()
        vis = Vis()

        @staticmethod
        def camera_name2id(name):
            return 0

    class Data:
        cam_xmat = np.asarray([np.eye(3).reshape(-1)])
        cam_xpos = np.asarray([[0.0, 0.0, 0.0]])
        body_xpos = np.asarray([[0.0, 0.0, -0.1]])

    model = Model()
    data = Data()


class _FakeInner:
    sim = _FakeSim()


class _FakeEnv:
    env = _FakeInner()
    obj_of_interest = ["object_1"]
    instance_to_id = {"object_1": 1}

    def __init__(self):
        self.actions = []

    def reset(self):
        pass

    def set_init_state(self, state):
        return self._observation()

    def step(self, action):
        self.actions.append(action)
        return self._observation(), 0.0, False, {}

    @staticmethod
    def _observation():
        segmentation = np.zeros((4, 4, 1), dtype=np.int32)
        segmentation[1, 2, 0] = 1
        return {
            "agentview_image": np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3),
            "agentview_depth": np.zeros((4, 4, 1), dtype=np.float32),
            "agentview_segmentation_instance": segmentation,
        }


def test_grounding_smoke_writes_separated_audit_artifacts(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    env = _FakeEnv()

    written_frames = []
    result = run_libero_grounding_smoke(
        env=env,
        initial_state="fixed",
        run_root=run_root,
        task_suite="libero_spatial",
        task_id=0,
        initial_state_index=0,
        seed=7,
        camera="agentview",
        height=4,
        width=4,
        settle_steps=2,
        video_writer=lambda path, frames: (
            written_frames.extend(frames), path.write_bytes(b"mp4")
        ),
    )

    assert env.actions == [LIBERO_GROUNDING_DUMMY_ACTION] * 2
    assert result["summary"]["objects_grounded"] == 1
    assert result["summary"]["task_success"] is None
    assert result["manifest"]["privileged_segmentation"] is True
    assert result["manifest"]["planner_receives_oracle_coordinates"] is False
    np.testing.assert_array_equal(
        written_frames[0], _FakeEnv._observation()["agentview_image"][::-1]
    )
    assert (run_root / "videos/task_000_state_000_grounding.mp4").read_bytes() == b"mp4"
    grounding = json.loads((run_root / "grounding.json").read_text())
    assert "world_xyz" in grounding["objects"]["object_1"]
    assert "body_center_xyz" not in grounding["objects"]["object_1"]
    assert json.loads((run_root / "summary.json").read_text())["task_success"] is None