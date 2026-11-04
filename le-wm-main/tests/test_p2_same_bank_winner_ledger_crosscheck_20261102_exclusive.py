import hashlib
from contextlib import contextmanager

import pytest
import torch

from experiments.spectral_reachability import p2_same_bank_winner_ledger_crosscheck_20261102_exclusive as check


def fixture():
    result, events = {"rows": []}, []
    raw = b"\0" * 80
    digest = hashlib.sha256(raw).hexdigest()
    for i in range(16):
        arms = []
        for arm in check.runner.native.optimizer.ARMS:
            for j in range(9):
                events.append({"record": {"kind": "optimizer_returned",
                                          "payload": {"candidate_index": i, "arm": arm,
                                                      "iteration": j, "score": float(j),
                                                      "action_sha256": digest}}})
            arms.append({"arm": arm, "winner_iteration": 0, "model_score": 0.,
                         "raw_float32_hex": raw.hex()})
        result["rows"].append({"index": i, "arms": arms})
    return result, events


def test_crosscheck_and_forged_winner_rejected():
    result, events = fixture()
    assert check.check_result_events(result, events)["optimizer_returns_checked"] == 288
    result["rows"][0]["arms"][0]["raw_float32_hex"] = (b"\1" * 80).hex()
    with pytest.raises(ValueError, match="winner"):
        check.check_result_events(result, events)


def test_missing_and_later_tie_rejected():
    result, events = fixture()
    events.pop()
    with pytest.raises(ValueError, match="census"):
        check.check_result_events(result, events)
    result, events = fixture()
    events[1]["record"]["payload"]["score"] = 0.
    result["rows"][0]["arms"][0]["winner_iteration"] = 1
    with pytest.raises(ValueError, match="winner"):
        check.check_result_events(result, events)


def test_complete_saved_fixture_crosscheck(tmp_path, monkeypatch):
    runner = check.runner
    monkeypatch.setattr(runner, "RESULTS", tmp_path)
    path = tmp_path / (runner.NAME + "_PLAN_check.json")
    plan_sha = runner.prepare(path)
    @contextmanager
    def session(fixed, journal):
        def transition(state, history, packet):
            successor = state[:, -1] + packet.sum().expand(192) / 100
            return torch.cat((state[:, 1:], successor[:, None]), dim=1), history
        meter = runner.native.CappedMeter(journal)
        yield runner.native.make_objective(fixed, transition, meter, torch.device("cpu")), meter
    output = tmp_path / (runner.NAME + "_RUN_check")
    report, report_sha = runner._execute(path, plan_sha, output, session, "cpu_fixture")
    assert report["status"] == "completed_development_control"
    assert check.crosscheck(output, report_sha, "cpu_fixture")["candidate_arm_winners_bound"] == 32
