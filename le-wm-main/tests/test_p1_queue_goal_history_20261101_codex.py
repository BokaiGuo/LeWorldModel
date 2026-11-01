import json

import pytest

from experiments.spectral_reachability import p1_queue_goal_history_20261101_codex as p
from experiments.spectral_reachability import p1_queue_goal_saved_20261101_codex as saved


def make_queue(tmp_path):
    spec = dict(schema=p.q.SCHEMA + ".spec.v1", experiment_id="goal_dev_fixture",
                **p.bindings(), model_sha256="4" * 64,
                slots=[dict(slot_id="dev", role="development", reset_seeds=[301000001]),
                       dict(slot_id="hold", role="confirmation", reset_seeds=[301000003])])
    path = tmp_path / "queue.json"
    return path, p.q.write_new(path, p.q.freeze(spec))


def test_real_goal_binding_and_tampering(tmp_path):
    queue, queue_sha = make_queue(tmp_path)
    with p.native.locking.gpu_lock(p.native.LOCK):
        report = p.run(queue, queue_sha, "dev", 0)
    assert report["status"] == "complete_development"
    directory = tmp_path / "queue.json.native_reset/dev_0000.warmup"
    report_sha = p.sha(directory / "report.json")
    result = saved.verify(queue, queue_sha, "dev", 0, report_sha)
    assert result["goal_rgb_verified"] and result["frames_verified"] == 3
    reset = json.loads((directory / "reset_receipt.json").read_bytes())
    source = json.loads((directory / "source_receipt.json").read_bytes())
    assert reset["goal_rgb"] == source["history"]["goal_rgb"]
    image_path = directory / "goal.png"
    image_bytes = image_path.read_bytes()
    image_path.write_bytes(image_bytes + b"tamper")
    with pytest.raises(ValueError, match="goal file"):
        saved.verify(queue, queue_sha, "dev", 0, report_sha)
    image_path.write_bytes(image_bytes)
    source_path = directory / "source_receipt.json"
    data = json.loads(source_path.read_bytes())
    data["history"]["goal"][0] += 1
    source_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source/history seal"):
        saved.verify(queue, queue_sha, "dev", 0, report_sha)
    with pytest.raises(ValueError, match="development only"):
        p.run(queue, queue_sha, "hold", 0)


def test_goal_rgb_mismatch_fails_before_history(tmp_path):
    from stable_worldmodel.envs.pusht.env import PushT

    class WrongGoal:
        def __init__(self):
            self.body = PushT(render_mode="rgb_array", resolution=224, with_target=False)

        def __getattr__(self, key):
            return getattr(self.body, key)

        def reset(self, **kwargs):
            obs, info = self.body.reset(**kwargs)
            info["goal"] = info["goal"].copy()
            info["goal"][0, 0, 0] ^= 1
            return obs, info

    queue, queue_sha = make_queue(tmp_path)
    with p.native.locking.gpu_lock(p.native.LOCK):
        report = p.run(queue, queue_sha, "dev", 0, factory=WrongGoal)
    assert report["status"] == "stopped_or_failed" and report["steps_returned"] == 0
    assert "actual native goal" in report["error"]["message"]
    assert not (tmp_path / "queue.json.native_reset/dev_0000.warmup/source_receipt.json").exists()


def test_inplace_goal_drift_preserves_native_return(tmp_path):
    from stable_worldmodel.envs.pusht.env import PushT

    class Drift:
        def __init__(self):
            self.body = PushT(render_mode="rgb_array", resolution=224, with_target=False)

        def __getattr__(self, key):
            return getattr(self.body, key)

        def step(self, action):
            result = self.body.step(action)
            self.body._goal[0, 0, 0] ^= 1
            return result

    queue, queue_sha = make_queue(tmp_path)
    with p.native.locking.gpu_lock(p.native.LOCK):
        report = p.run(queue, queue_sha, "dev", 0, factory=Drift)
    assert report["status"] == "stopped_or_failed" and report["steps_returned"] == 1
    journal = p.events(tmp_path / "queue.json.native_reset/dev_0000.warmup/journal.jsonl")
    assert sum(e["kind"] == "step_native_return" for e in journal) == 1
    assert not any(e["kind"] == "step_return" for e in journal)
