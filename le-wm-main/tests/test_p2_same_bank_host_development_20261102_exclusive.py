from contextlib import contextmanager
import json

import pytest
import torch

from experiments.spectral_reachability import p2_same_bank_host_development_20261102_exclusive as runner


def transition(state, history, packet):
    successor = state[:, -1] + packet.sum().expand(192) / 100
    return torch.cat((state[:, 1:], successor[:, None]), dim=1), history


@contextmanager
def cpu_fixture(fixed, journal):
    meter = runner.native.CappedMeter(journal)
    yield runner.native.make_objective(fixed, transition, meter, torch.device("cpu")), meter


@contextmanager
def failing_fixture(fixed, journal):
    def fail(state, history, packet):
        raise RuntimeError("fixture failure")
    meter = runner.native.CappedMeter(journal)
    yield runner.native.make_objective(fixed, fail, meter, torch.device("cpu")), meter


def make_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS", tmp_path)
    path = tmp_path / (runner.NAME + "_PLAN_fixture.json")
    return path, runner.prepare(path)


def test_prepare_and_complete_fixture_with_saved_handoff(tmp_path, monkeypatch):
    path, sha = make_plan(tmp_path, monkeypatch)
    plan = runner.verify(path, sha)
    assert len(plan["input"]["initial_actions_sha256"]) == 16
    assert plan["schedule"]["scores_per_arm"] == 144
    assert plan["limits"]["source_manifest_full_transitive_closure"] is False
    output = tmp_path / (runner.NAME + "_RUN_fixture")
    report, report_sha = runner._execute(path, sha, output, cpu_fixture, "cpu_fixture")
    assert report["status"] == "completed_development_control"
    assert report["ledger_audit"]["profile"] == "cpu_fixture"
    assert not report["ledger_audit"]["native_execution_authenticated"]
    winners = json.loads((output / "winners.json").read_bytes())
    assert len(winners["winners"]) == 2 and winners["physical_execution_authorized"] is False
    assert runner.file_sha(output / "winners.json") == report["winners_sha256"]
    assert runner.file_sha(output / "report.json") == report_sha
    assert runner.consume(output, report_sha, "cpu_fixture")["status"] == "saved_development_ledger_verified_not_admitted"
    assert not report["P1_admission"] and not report["P2_admission"]
    with pytest.raises(ValueError, match="exclusive"):
        runner._execute(path, sha, output, cpu_fixture, "cpu_fixture")


def test_failure_keeps_attempt_and_no_retry(tmp_path, monkeypatch):
    path, sha = make_plan(tmp_path, monkeypatch)
    output = tmp_path / (runner.NAME + "_RUN_failure")
    report, _ = runner._execute(path, sha, output, failing_fixture, "cpu_fixture")
    assert report["status"] == "failed_partial"
    assert report["failure"]["message"] == "fixture failure"
    assert report["attempted_counts"]["callback_attempts"]["composition"]["score"] == 1
    events = [json.loads(line)["record"]["kind"] for line in (output / "journal.jsonl").read_text().splitlines()]
    assert "optimizer_failed" in events and "objective_failed" in events
    assert report["ledger_audit"] is None
    assert not (output / "winners.json").exists()


def test_wrong_authorization_and_plan_sha_fail_before_output(tmp_path, monkeypatch):
    path, sha = make_plan(tmp_path, monkeypatch)
    output = tmp_path / (runner.NAME + "_RUN_rejected")
    with pytest.raises(ValueError, match="authorization"):
        runner.run(path, sha, output, "wrong")
    with pytest.raises(ValueError, match="plan SHA"):
        runner._execute(path, "0" * 64, output, cpu_fixture, "cpu_fixture")
    assert not output.exists()
