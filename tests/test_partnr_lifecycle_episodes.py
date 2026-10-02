"""bootstrap_from_episodes: central bootstrap from labelled episodes of any environment."""

import json

from harness_vla.envs.partnr import primitives
from harness_vla.envs.partnr.lifecycle import bootstrap_from_episodes
from harness_vla.envs.partnr.memory import load_episode_trace
from harness_vla.envs.partnr.primitives import Vocabulary

VOCAB = Vocabulary(("GoToShelf", "GoToGoal", "Wait", "Done"), domain="a warehouse robot",
                   instance_examples="(shelf_1)", scope="warehouses")


def _trace(path, success):
    state = {"objects": {"shelf_1": {}}, "furniture": {"goal_1": {}}}
    context = {"instruction": "Deliver 1 shelf.", "episode_id": "e", "decision_index": 0,
               "extras": {"state": state}}
    records = [
        {"type": "episode_start", "context": context},
        {"type": "decision", "context": context,
         "decisions": {"agent_0": {"name": "GoToShelf", "args": "shelf_1"}}},
        {"type": "result", "agent": "agent_0",
         "result": {"tool": "GoToShelf", "args": "shelf_1", "success_hint": True}},
        {"type": "episode_end", "summary": {"episode_id": "e", "extras": {}}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return {"job_id": path.name, "episode_id": "e", "instruction": "Deliver 1 shelf.",
            "task_state_success": 1.0 if success else 0.0,
            "traces": {"team": load_episode_trace(str(path))}}


def test_central_bootstrap_writes_symbolic_task_memory_for_a_custom_vocabulary(tmp_path):
    previous = primitives.use_vocabulary(VOCAB)
    try:
        episodes = [_trace(tmp_path / "ok.jsonl", True), _trace(tmp_path / "bad.jsonl", False)]
        audit = bootstrap_from_episodes(str(tmp_path / "out"), episodes, min_support=1)
    finally:
        primitives.use_vocabulary(previous)
    assert audit["successful_episodes"] == 1
    files = list((tmp_path / "out" / "task_memory").glob("*.json"))
    assert len(files) == 1
    assert "agent_0: GoToShelf[<obj:shelf>]" in files[0].read_text()
    assert audit["deployment"]["zero_shot"] is False
