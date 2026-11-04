import json

import pytest

from experiments.spectral_reachability import run_pc_p2_conditional_host as runner


def test_plan_verification_rejects_source_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS", tmp_path)
    path = tmp_path / (runner.NAME + "_PLAN_test.json")
    path.write_text(json.dumps({"source": "original"}))
    digest = runner.host.file_sha(path)
    monkeypatch.setattr(runner, "build_plan", lambda: {"source": "original"})
    assert runner.verify(path, digest)["source"] == "original"
    monkeypatch.setattr(runner, "build_plan", lambda: {"source": "changed"})
    with pytest.raises(ValueError, match="drift"):
        runner.verify(path, digest)


def test_exclusive_output_rejects_existing_and_wrong_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS", tmp_path)
    candidate = tmp_path / (runner.NAME + "_RUN_test")
    assert runner.output_path(candidate, "_RUN_") == candidate
    candidate.mkdir()
    with pytest.raises(ValueError, match="exclusive"):
        runner.output_path(candidate, "_RUN_")
    with pytest.raises(ValueError, match="exclusive"):
        runner.output_path(tmp_path.parent / candidate.name, "_RUN_")
