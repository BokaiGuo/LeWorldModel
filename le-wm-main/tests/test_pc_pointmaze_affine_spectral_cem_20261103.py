import json
from pathlib import Path

import numpy as np
import pytest

from experiments import pc_pointmaze_affine_spectral_cem_20261103 as experiment
from experiments.pc_pointmaze_affine_spectral_cem_20261103 import spectral_cem
from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import sha256_file


def test_injection_is_scored_and_enters_elites_and_best_plan():
    rng = np.random.default_rng(42)
    noise = rng.normal(size=(4, 16, 2, 2)).astype(np.float32)
    injected = np.zeros((4, 2, 2), np.float32)
    injected[:, 0, 0] = [.3, -.3, .6, -.6]
    seen = []
    def score(proposals):
        seen.append(proposals.copy())
        return np.sum((proposals - injected[2]) ** 2, axis=(1, 2))
    plan, cost, trace = spectral_cem(noise, score, injected)
    np.testing.assert_array_equal(seen[0][:4], injected)
    np.testing.assert_array_equal(plan, injected[2])
    assert cost == 0
    assert trace[0]["elite_indices"][0] == 2
    assert len(seen) == 4


def test_rejects_invalid_injection():
    noise = np.zeros((4, 16, 2, 2))
    with pytest.raises(ValueError):
        spectral_cem(noise, lambda x: np.zeros(len(x)), np.full((4, 2, 2), 2.0))
    with pytest.raises(ValueError):
        spectral_cem(noise, lambda x: np.full(len(x), np.nan), np.zeros((4, 2, 2)))


def test_saved_spectral_ablation_and_history_receipt():
    if not experiment.OUTPUT.exists():
        pytest.skip("development run not available")
    report = json.loads(experiment.OUTPUT.read_text())
    assert report["script_sha256"] == sha256_file(Path(experiment.__file__))
    assert report["protocol_sha256"] == sha256_file(experiment.PROTOCOL)
    assert report["arrays_sha256"] == sha256_file(experiment.ARRAYS)
    assert report["model_before"] == report["model_after"]
    assert [r["root"] for r in report["rows"]] == list(range(1888, 1896))
    with np.load(experiment.ARRAYS, allow_pickle=False) as saved:
        for i, row in enumerate(report["rows"]):
            jac = np.asarray(row["jacobian"])
            assert np.linalg.norm(jac - row["validation_fd"]["0.001"]) / np.linalg.norm(jac) < 0.001
            np.testing.assert_array_equal(row["initial_trace"][0]["proposals"][:4], row["injected"])
            for phase, plan_key in (("initial", "initial_plan"), ("tail", "tail")):
                best = min((float(score), proposal)
                           for entry in row[phase + "_trace"]
                           for score, proposal in zip(entry["scores"], entry["proposals"]))
                assert row[phase + "_cost"] == pytest.approx(best[0])
                np.testing.assert_array_equal(row[plan_key], best[1])
            np.testing.assert_array_equal(saved["sequences"][i, :5],
                                          np.repeat(np.asarray(row["initial_plan"])[0:1], 5, axis=0))
            np.testing.assert_array_equal(saved["sequences"][i, 5:],
                                          np.repeat(np.asarray(row["tail"])[0:1], 5, axis=0))
            np.testing.assert_array_equal(saved["legacy_replay_rgb10"][i], saved[f"r{row['root']}_rgb10"])
            np.testing.assert_array_equal(saved["legacy_replay_endpoints"][i], row["endpoint_eval_only"])
            np.testing.assert_array_equal(saved["position_replay_endpoints"][i, :2],
                                          row["endpoint_eval_only"][:2])
    audit_path = experiment.ROOT / "results/pc_pointmaze_affine_history_audit_20261103.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        assert audit["all_match"] and len(audit["rows"]) == 16
        assert audit["arrays_sha256"] == sha256_file(audit_path.with_suffix(".npz"))
