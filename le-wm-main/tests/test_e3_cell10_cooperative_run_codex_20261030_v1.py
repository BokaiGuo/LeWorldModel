import json
import subprocess

import pytest

from experiments.spectral_reachability import e3_cell10_cooperative_run_codex_20261030_v1 as m


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "RESULTS", tmp_path)
    monkeypatch.setattr(m, "OUTPUT", tmp_path / "native")
    monkeypatch.setattr(m, "JOURNAL", tmp_path / "parent.jsonl")
    monkeypatch.setattr(m.claim, "DIRECTORY", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    return tmp_path


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_fixed_fresh_child_wiring_and_cost_events(isolated, monkeypatch):
    calls = []
    def fake(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(m.subprocess, "run", fake)
    value = m.run(m.CONFIRM)
    rows = events(m.JOURNAL)
    assert [r["stage"] for r in rows] == ["started", "cpu_precheck_intent", "cpu_precheck_return",
                                             "native_host_intent", "native_host_return", "finished"]
    assert [c[0] for c in calls] == [m._precheck_command(), m._child_command()]
    assert all(c[1]["timeout"] <= 1800 for c in calls)
    assert rows[0]["gpu_time"] == "unknown" and rows[0]["loading_full_cost"] == "unknown"
    assert value["closed_loop_complete"] == "unknown" and not value["scientific_admission"]
    with pytest.raises(FileExistsError):
        m.run(m.CONFIRM)


@pytest.mark.parametrize("failure", ["precheck", "native", "timeout"])
def test_failure_keeps_claim_and_parent_record(isolated, monkeypatch, failure):
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 1 if (failure == "precheck" or len(calls) == 2) else 0)
    monkeypatch.setattr(m.subprocess, "run", fake)
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        m.run(m.CONFIRM)
    assert (isolated / m.claim.MARKER).is_file()
    assert events(m.JOURNAL)[-1]["stage"] == "failure"
    assert len(calls) == (2 if failure == "native" else 1)
    with pytest.raises(FileExistsError):
        m.run(m.CONFIRM)


def test_no_confirmation_or_disabled_cuda_never_claims(isolated, monkeypatch):
    with pytest.raises(ValueError, match="coordination"):
        m.run("not authorized")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="disabled"):
        m.run(m.CONFIRM)
    assert not (isolated / m.claim.MARKER).exists()
    assert not m.JOURNAL.exists()


def test_preexisting_output_is_preserved_and_not_claimed(isolated):
    m.OUTPUT.mkdir()
    with pytest.raises(FileExistsError):
        m.run(m.CONFIRM)
    assert not (isolated / m.claim.MARKER).exists()
