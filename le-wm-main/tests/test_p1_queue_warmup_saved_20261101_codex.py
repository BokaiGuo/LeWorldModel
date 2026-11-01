import pytest

from experiments.spectral_reachability import p1_queue_warmup_history_20261101_codex as p
from experiments.spectral_reachability import p1_queue_warmup_saved_20261101_codex as saved


def test_real_saved_and_tampering(tmp_path):
    spec = dict(schema=p.q.SCHEMA + ".spec.v1", experiment_id="dev_saved_fixture",
                **p.bindings(), model_sha256="4" * 64,
                slots=[dict(slot_id="dev", role="development", reset_seeds=[301000001]),
                       dict(slot_id="hold", role="confirmation", reset_seeds=[301000003])])
    queue = tmp_path / "queue.json"
    queue_sha = p.q.write_new(queue, p.q.freeze(spec))
    with p.native.locking.gpu_lock(p.native.LOCK):
        result = p.run(queue, queue_sha, "dev", 0)
    assert result["status"] == "complete_development"
    directory = tmp_path / "queue.json.native_reset/dev_0000.warmup"
    report_sha = p.sha(directory / "report.json")
    audit = saved.verify(queue, queue_sha, "dev", 0, report_sha)
    assert audit["frames_verified"] == 3 and audit["returned_steps"] == 10
    with (directory / "rgb_05.png").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="frame file"):
        saved.verify(queue, queue_sha, "dev", 0, report_sha)
