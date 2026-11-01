import numpy as np
import pytest
import json
from contextlib import contextmanager

from experiments.spectral_reachability import p1_cp5_cem_rgb_mismatch_20261101_leibniz as diagnostic


def test_diagnostic_alignment_and_q():
    predicted = np.ones((3, 192))
    actual = np.zeros((3, 192))
    rows = diagnostic.analyze(predicted, actual, np.zeros(192), np.eye(192),
                              [{"joint_xy": n} for n in (10, 15, 20)])
    assert [r["step"] for r in rows] == [10, 15, 20]
    assert all(r["predicted_q"] == 1 and r["actual_rgb_q"] == 0
               and r["latent_rmse"] == 1 and r["projected_rmse"] == 1 for r in rows)
    assert [r["physical_metrics"]["joint_xy"] for r in rows] == [10, 15, 20]


def test_reject_nonfinite_and_wrong_shape():
    with pytest.raises(ValueError):
        diagnostic.analyze(np.zeros((2, 192)), np.zeros((3, 192)),
                           np.zeros(192), np.eye(192), [{}, {}, {}])
    values = np.zeros((3, 192))
    values[0, 0] = np.nan
    with pytest.raises(ValueError):
        diagnostic.analyze(values, np.zeros((3, 192)),
                           np.zeros(192), np.eye(192), [{}, {}, {}])


def test_fixed_budget():
    v = diagnostic.costs()
    assert (v["encode_calls"], v["predict_calls"], v["predict_samples"],
            v["action_encode_calls"], v["physical_steps"]) == (4, 3, 3, 3, 0)


def test_run_resolves_real_launcher_lock_before_cuda(monkeypatch, tmp_path):
    assert callable(diagnostic.launcher.extra.runner.signal.model_hash)
    results = tmp_path / "results"
    results.mkdir()
    prepared = results / (diagnostic.NAME + "_PREP.json")
    source = tmp_path / "input.txt"
    source.write_text("source")
    value = {"sources": {str(source): diagnostic.family.sha(source)}}
    prepared.write_text(json.dumps(value))
    expected = diagnostic.family.sha(prepared)
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    monkeypatch.setattr(diagnostic, "build", lambda: value)
    seen = []
    @contextmanager
    def fake_lock(path):
        seen.append(path)
        yield
    monkeypatch.setattr(diagnostic.launcher.extra.gpu.locking, "gpu_lock", fake_lock)
    def stop_before_allocation(device):
        raise RuntimeError("sentinel before CUDA allocation")
    monkeypatch.setattr(diagnostic.own.gpu_preflight, "require_cuda_allocation", stop_before_allocation)
    output = results / (diagnostic.NAME + "_RUN")
    result = diagnostic.run(prepared, expected, output, diagnostic.AUTH)
    assert seen == [diagnostic.launcher.extra.gpu.LOCK]
    assert result["status"] == "partial"
    report = json.loads((output / "report.json").read_text())
    assert report["failure"]["message"] == "sentinel before CUDA allocation"
    assert report["ledger"] is None
