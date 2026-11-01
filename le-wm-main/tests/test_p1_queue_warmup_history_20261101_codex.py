import json

import pytest
from PIL import Image
import numpy as np

from experiments.spectral_reachability import p1_queue_warmup_history_20261101_codex as m


def queue_fixture(tmp_path):
    spec = dict(schema=m.q.SCHEMA + ".spec.v1", experiment_id="dev_not_confirmation",
                **m.bindings(), model_sha256="4" * 64,
                slots=[dict(slot_id="dev", role="development", reset_seeds=[301000001]),
                       dict(slot_id="stopped", role="development", reset_seeds=[301000002]),
                       dict(slot_id="hold", role="confirmation", reset_seeds=[301000003])])
    path = tmp_path / "queue.json"
    digest = m.q.write_new(path, m.q.freeze(spec))
    return path, digest


def test_real_history_and_no_replay(tmp_path):
    path, digest = queue_fixture(tmp_path)
    with m.native.locking.gpu_lock(m.native.LOCK):
        result = m.run(path, digest, "dev", 0)
    assert result["status"] == "complete_development" and result["steps_returned"] == 10
    base = tmp_path / "queue.json.native_reset"
    run = base / "dev_0000.warmup"
    source = json.loads((run / "source_receipt.json").read_bytes())
    assert [f["step"] for f in source["history"]["frames"]] == [0, 5, 10]
    assert source["history"]["actions"] == [[0.0, 0.0]] * 10
    assert all(np.asarray(Image.open(run / f["file"])).shape == (224, 224, 3)
               for f in source["history"]["frames"])
    ledger = m.events(base / "queue_ledger.jsonl")
    assert [x["kind"] for x in ledger] == ["reset_attempt", "reset_return"]
    assert ledger[-1]["payload"]["history_sha256"] == source["history_sha256"]
    assert not m.q.validate(m.q.read_json(path), ledger)["pending"]
    with pytest.raises(ValueError, match="replay prohibited"):
        m.run(path, digest, "dev", 0)
    with pytest.raises(ValueError, match="development only"):
        m.run(path, digest, "hold", 0)


def test_native_truncation_preserves_partial_history(tmp_path):
    from stable_worldmodel.envs.pusht.env import PushT

    class Truncate:
        def __init__(self):
            self.body = PushT(render_mode="rgb_array", resolution=224, with_target=False)
            self.n = 0

        def __getattr__(self, key):
            return getattr(self.body, key)

        def step(self, action):
            self.n += 1
            obs, reward, terminated, _, info = self.body.step(action)
            return obs, reward, terminated, self.n == 3, info

    path, digest = queue_fixture(tmp_path)
    with m.native.locking.gpu_lock(m.native.LOCK):
        report = m.run(path, digest, "stopped", 0, factory=Truncate)
    assert report["status"] == "stopped_or_failed" and report["steps_returned"] == 3
    run = tmp_path / "queue.json.native_reset/stopped_0000.warmup"
    assert not (run / "source_receipt.json").exists()
    assert (run / "rgb_00.png").exists() and not (run / "rgb_05.png").exists()
    ledger = m.events(tmp_path / "queue.json.native_reset/queue_ledger.jsonl")
    assert [x["kind"] for x in ledger] == ["reset_attempt", "reset_failed"]
