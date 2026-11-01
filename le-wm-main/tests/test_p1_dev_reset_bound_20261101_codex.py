import json

from experiments.spectral_reachability import p1_dev_reset_bound_20261101_codex as m


def test_real_native_reset_and_journal(tmp_path):
    plan = tmp_path / "prepare.json"
    digest = m.queue.write_new(plan, m.contract())
    # The actual integration test takes the project lock and invokes PushT.reset.
    with m.locking.gpu_lock(m.LOCK):
        report = m.run(plan, digest, tmp_path / "real")
    assert not report["errors"]
    assert [row["status"] for row in report["rows"]] == ["returned", "returned"]
    assert all(row["receipt"]["history_sha256"] is None for row in report["rows"])
    events = [json.loads(line) for line in (tmp_path / "real/journal.jsonl").read_bytes().splitlines()]
    assert [row["kind"] for row in events] == [
        "run_started", "reset_attempt", "reset_return", "reset_attempt", "reset_return", "run_finished"]
    assert events[1]["seed"] == events[2]["seed"] == m.SEEDS[0]
    assert not report["P1_admission"]


def test_failed_reset_preserves_denominator(tmp_path):
    class Failing:
        def reset(self, **kwargs):
            raise RuntimeError("fixture failure")

        def close(self):
            pass

    plan = tmp_path / "prepare.json"
    digest = m.queue.write_new(plan, m.contract())
    report = m.run(plan, digest, tmp_path / "fail", factory=Failing)
    assert [row["status"] for row in report["rows"]] == ["failed", "not_started"]
    events = [json.loads(line) for line in (tmp_path / "fail/journal.jsonl").read_bytes().splitlines()]
    assert [row["kind"] for row in events] == ["run_started", "reset_attempt", "reset_failed", "run_finished"]


def test_wrong_prepare_rejected_before_output(tmp_path):
    plan = tmp_path / "prepare.json"
    m.queue.write_new(plan, m.contract())
    import pytest
    with pytest.raises(ValueError, match="prepared file SHA"):
        m.run(plan, "0" * 64, tmp_path / "absent")
    assert not (tmp_path / "absent").exists()
