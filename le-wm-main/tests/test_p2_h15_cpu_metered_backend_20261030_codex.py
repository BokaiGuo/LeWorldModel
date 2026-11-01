import hashlib
import json

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_durable_hook_driver_20261030_codex as durable
from experiments.spectral_reachability import p2_h15_cpu_metered_backend_20261030_codex as backend
from experiments.spectral_reachability import p2_h15_sra_prepare_20261029 as original
from experiments.spectral_reachability import p2_hook_contract_bridge_20261030_codex as prior
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec


def setup(tmp_path, *, svd_cap=2, score_cap=40):
    h = "a" * 64
    c = {"schema": spec.NAME + ".contract.v1", "arms": ["joint", "composition"],
         "scope": "per_root_per_arm_standalone_planning_and_selected_physical_execution",
         "identity": {k: h for k in spec.IDENTITY},
         "axes": {k: {"unit": u, "aggregation": "max" if k == "peak_device_bytes" else "sum",
                      "cap": 1000} for k, u in spec.AXES.items()}}
    c["axes"]["array_svd_calls"]["cap"] = svd_cap
    c["axes"]["selection_evaluations"]["cap"] = score_cap
    digest = spec.sha(spec.canonical(c))
    d = backend.make_driver(tmp_path / "cpu", c, digest, dict.fromkeys(core.AXES, 1000), "joint", "cpu-test")
    return backend.Backend(d), digest


def source(root):
    host = spec.read_bound(prior.HOST / "report.json", prior.HOST_SHA)
    slot = host["slots"][root]
    report = spec.read_bound(prior.HOST / f"root{root}/report.json", slot["report_sha256"])
    raw = (prior.HOST / f"root{root}/events.jsonl").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == report["journal_sha256"]
    records = durable.read_chain(raw)
    attempts = [x["payload"] for x in records if x["payload"].get("kind") == "native_call_attempt"]
    returns = [x["payload"] for x in records if x["payload"].get("kind") == "native_call_return"]
    assert attempts[0]["method"] == "signal" and len(returns) == 43
    return report, attempts[0]["task"]["goal"], returns[0]["result"]


@pytest.mark.parametrize("root", [0, 1])
def test_real_h15_saved_signal_and_scores(tmp_path, root):
    report, goal, signal = source(root)
    b, digest = setup(tmp_path)
    bank = b.compile(signal["jacobian"], signal["nominal"], goal)
    expected = report["result"]["bank"]
    assert bank["labels"] == expected["labels"]
    np.testing.assert_allclose(bank["logits"], expected["logits"], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(bank["raw15"], expected["raw15"], rtol=0, atol=1e-6)
    scores = []
    for i, row in enumerate(report["result"]["candidates"]):
        terminal = row["path"][-1]["latent_history"][-1]
        scores.append(b.score_candidate(i, terminal, goal))
        assert scores[-1] == pytest.approx(row["score"], rel=1e-13, abs=1e-13)
    assert int(np.argmin(scores)) == report["result"]["winner"]
    output = b.finish()
    checked = durable.verify(output["journal"], output["sha256"], digest)
    assert output["status"] == "complete" and output["completed_scores"] == 40
    assert checked["completed"]["array_SVD_calls"] == 2
    assert checked["completed"]["solve_calls"] == 1
    assert checked["completed"]["metric_calls"] == 40
    assert checked["completed"]["predict_samples"] == 0  # only new CPU work
    assert output["FLOPs"] is None and not checked["coverage_verified"]


def test_real_numpy_bank_matches_original_implementation(tmp_path):
    rng = np.random.default_rng(527)
    j, z, g = rng.normal(size=(192, 30)), rng.normal(size=192), rng.normal(size=192)
    b, digest = setup(tmp_path)
    got = b.compile(j, z, g)
    reference = original.spectral_bank(j, z, g)
    assert json.dumps(got, sort_keys=True) == json.dumps(reference, sort_keys=True)
    # Deliberately incomplete score set must not produce a complete run.
    output = b.finish()
    assert durable.verify(output["journal"], output["sha256"], digest)["status"] == "failed_or_blocked"


def test_svd_cap_stops_before_second_svd(tmp_path, monkeypatch):
    b, digest = setup(tmp_path, svd_cap=1)
    calls = []
    fn = np.linalg.svd
    def counted(*args, **kwargs):
        calls.append(1)
        return fn(*args, **kwargs)
    monkeypatch.setattr(np.linalg, "svd", counted)
    with pytest.raises(core.BudgetExceeded):
        b.compile(np.eye(192, 30), np.zeros(192), np.ones(192))
    assert len(calls) == 1
    output = b.finish()
    checked = durable.verify(output["journal"], output["sha256"], digest)
    assert checked["status"] == "failed_or_blocked"
    assert checked["attempted"]["array_SVD_calls"] == 1


def test_score_cap_and_validation_failure_are_fail_stop(tmp_path):
    b, digest = setup(tmp_path, score_cap=0)
    b.compile(np.eye(192, 30), np.zeros(192), np.ones(192))
    with pytest.raises(core.BudgetExceeded):
        b.score_candidate(0, np.zeros(192), np.ones(192))
    checked = durable.verify((r := b.finish())["journal"], r["sha256"], digest)
    assert checked["status"] == "failed_or_blocked" and checked["attempted"]["metric_calls"] == 0


def test_invalid_goal_cannot_retry(tmp_path):
    b, digest = setup(tmp_path)
    b.compile(np.eye(192, 30), np.zeros(192), np.ones(192))
    with pytest.raises(ValueError):
        b.score_candidate(0, np.zeros(192), np.zeros(192))
    with pytest.raises(ValueError, match="fail-stopped"):
        b.score_candidate(0, np.zeros(192), np.ones(192))
    r = b.finish()
    assert durable.verify(r["journal"], r["sha256"], digest)["status"] == "failed_or_blocked"


def test_real_svd_failure_is_paid_and_no_retry(tmp_path, monkeypatch):
    b, digest = setup(tmp_path)
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError("failed CPU SVD")
    monkeypatch.setattr(np.linalg, "svd", fail)
    with pytest.raises(np.linalg.LinAlgError):
        b.compile(np.eye(192, 30), np.zeros(192), np.ones(192))
    with pytest.raises(ValueError, match="fail-stopped"):
        b.compile(np.eye(192, 30), np.zeros(192), np.ones(192))
    r = b.finish()
    checked = durable.verify(r["journal"], r["sha256"], digest)
    assert checked["status"] == "failed_or_blocked"
    assert checked["attempted"]["array_SVD_calls"] == 1
    assert checked["completed"]["array_SVD_calls"] == 0
