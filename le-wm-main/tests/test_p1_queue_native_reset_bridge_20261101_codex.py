import json

import pytest

from experiments.spectral_reachability import p1_queue_native_reset_bridge_20261101_codex as b


def make_queue(tmp_path):
    spec = dict(schema=b.q.SCHEMA + ".spec.v1", experiment_id="dev_fixture",
                **b.expected_spec_bindings(), model_sha256="4" * 64,
                slots=[dict(slot_id="dev", role="development", reset_seeds=[301000001]),
                       dict(slot_id="fail", role="development", reset_seeds=[301000002]),
                       dict(slot_id="confirm", role="confirmation", reset_seeds=[301000003])])
    queue_path = tmp_path / "queue.json"
    digest = b.q.write_new(queue_path, b.q.freeze(spec))
    return queue_path, digest


def test_real_reset_link_and_no_replay(tmp_path):
    queue, digest = make_queue(tmp_path)
    with b.native.locking.gpu_lock(b.native.LOCK):
        receipt = b.run(queue, digest, "dev", 0)
    assert receipt["status"] == "reset_returned_history_pending"
    assert receipt["history_sha256"] is None and not receipt["P1_admission"]
    workspace = tmp_path / "queue.json.native_reset"
    events = b.read_events(workspace / "queue_ledger.jsonl")
    assert len(events) == 1 and events[0]["sha256"] == receipt["queue_attempt_event_sha256"]
    assert b.q.validate(b.q.read_json(queue), events)["pending"] == {"dev": 0}
    assert json.loads((workspace / "dev_0000.json").read_bytes()) == receipt
    with pytest.raises(ValueError, match="replay prohibited"):
        b.run(queue, digest, "dev", 0)
    with pytest.raises(ValueError, match="development slot"):
        b.run(queue, digest, "confirm", 0)


def test_failed_reset_is_recorded_and_not_replayed(tmp_path):
    class Failing:
        def reset(self, **kwargs):
            raise RuntimeError("fixture failure")

        def close(self):
            pass

    queue, digest = make_queue(tmp_path)
    result = b.run(queue, digest, "fail", 0, factory=Failing)
    assert result["status"] == "failed" and result["phase"] == "reset_dispatched"
    events = b.read_events(tmp_path / "queue.json.native_reset/queue_ledger.jsonl")
    assert [e["kind"] for e in events] == ["reset_attempt", "reset_failed"]
    assert not b.q.validate(b.q.read_json(queue), events)["pending"]
    with pytest.raises(ValueError, match="replay prohibited"):
        b.run(queue, digest, "fail", 0, factory=Failing)


def test_reservation_crash_blocks_and_wrong_sha_fails(tmp_path):
    queue, digest = make_queue(tmp_path)
    workspace = tmp_path / "queue.json.native_reset"
    workspace.mkdir()
    b.q.write_new(workspace / "dev_0000.reservation.json", {"interrupted": True})
    with pytest.raises(FileExistsError):
        b.run(queue, digest, "dev", 0)
    assert not (workspace / "queue_ledger.jsonl").exists()
    with pytest.raises(ValueError, match="queue file SHA"):
        b.run(queue, "0" * 64, "dev", 0)
