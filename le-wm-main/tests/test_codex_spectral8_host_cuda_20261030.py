import copy
import json

import pytest
import torch

from experiments.spectral_reachability import codex_spectral8_host_cuda_20261030 as host
from experiments.spectral_reachability import codex_four_arm_spectral8_suffix_20261030 as spectral
from test_codex_four_arm_spectral8_suffix_20261030 import Affine


def fixture_prepared():
    base = {"task": {"binding": {"root_sha256": "a"*64, "action_domain_sha256": "b"*64},
                     "goal": [1.]*192, "projection": torch.eye(192).tolist(),
                     "root_context": {"latent_history": torch.zeros(3,192).tolist(),
                                      "action_history": torch.zeros(2,10).tolist()},
                     "action_mean": [0.,0.], "action_std": [1.,1.]},
            "expected_model_state_sha256": "c"*64}
    return host.assemble(base)


def test_engine_fixture_persists_eight_candidates_and_meter(tmp_path):
    p = fixture_prepared()
    result = host.execute_engine(Affine(), p, tmp_path, lambda: None, device="cpu", analytic_fixture=True)
    assert result["status"] == "complete"
    assert result["account"]["charged_upper_bounds"] == spectral.required_caps(p["config"])
    events = [json.loads(line) for line in (tmp_path/"candidates.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in events] == ["header","spectral_signal","spectral_bank",
                                          "optimized_candidates","final_selection","terminal"]
    assert len(events[2]["payload"]["raw"]) == len(events[3]["payload"]["raw"]) == 8
    assert events[4]["payload"]["index"] == result["selected_index"]


def test_engine_failure_preserves_failed_ledger_and_noncompleted_trace(tmp_path, monkeypatch):
    p = fixture_prepared()
    def fail(*args, **kwargs):
        raise RuntimeError("SVD failure")
    monkeypatch.setattr(spectral.np.linalg, "svd", fail)
    result = host.execute_engine(Affine(), p, tmp_path, lambda: None, device="cpu", analytic_fixture=True)
    assert result["status"] == "failed_or_partial"
    assert result["account"]["status"] == "failed_or_blocked"
    assert result["account"]["charged_upper_bounds"]["svd_calls"] == 1
    assert json.loads((tmp_path/"candidates.jsonl").read_text().splitlines()[-1])["payload"]["status"] == "failed_or_partial"


def test_prepare_drift_fails_before_cuda(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(host, "build", lambda: {"current": True})
    monkeypatch.setattr(host, "require_cuda_allocation", lambda *_: called.append(True))
    monkeypatch.setattr(host, "_output_path", lambda p, suffix: tmp_path/"out")
    with pytest.raises(ValueError, match="drift"):
        host.run_cuda(tmp_path/"out", {"current": False}, host.AUTH)
    assert not called and not (tmp_path/"out").exists()


def test_run_requires_authorization_before_any_cuda(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(host, "require_cuda_allocation", lambda *_: called.append(True))
    with pytest.raises(ValueError, match="authorization"):
        host.run_cuda(tmp_path/"out", {}, "no")
    assert not called


def test_cli_partial_exit_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(host, "read_prepare", lambda *_: {})
    monkeypatch.setattr(host, "run_cuda", lambda *_: {"status": "failed_or_partial"})
    monkeypatch.setattr("sys.argv", ["entry","run","--prepared","x","--sha256","x",
                                     "--output","x","--authorization",host.AUTH])
    assert host.main() == 1
    assert "failed_or_partial" in capsys.readouterr().out


def test_candidate_sink_failure_blocks_suffix_and_preserves_meter(tmp_path, monkeypatch):
    p = fixture_prepared()
    original = host.transport.Journal.emit
    def failing_emit(self, kind, payload):
        if kind == "spectral_bank":
            raise OSError("persist failed")
        return original(self, kind, payload)
    monkeypatch.setattr(host.transport.Journal, "emit", failing_emit)
    result = host.execute_engine(Affine(), p, tmp_path, lambda: None, device="cpu", analytic_fixture=True)
    assert result["status"] == "failed_or_partial"
    assert result["account"]["status"] == "failed_or_blocked"
    assert result["account"]["charged_upper_bounds"]["svd_calls"] == 1
    assert result["account"]["charged_upper_bounds"]["replay_calls"] == 0


def test_tampered_contract_rejected_before_model(tmp_path):
    p = fixture_prepared()
    p["policy"]["candidate_ids"].reverse()
    result = host.execute_engine(Affine(), p, tmp_path, lambda: None, device="cpu", analytic_fixture=True)
    assert result["status"] == "failed_or_partial"
    assert result["account"]["status"] == "failed_or_blocked"
    assert result["account"]["charged_upper_bounds"]["predict_calls"] == 0
