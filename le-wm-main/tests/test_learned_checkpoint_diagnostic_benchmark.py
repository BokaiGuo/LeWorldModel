import json

import numpy as np

from experiments.empirical_spectral_audit.run_learned_checkpoint_diagnostic_benchmark import (
    Config,
    _candidate_family,
    _eigenvalue_match,
    markdown,
    run,
)


def _write_metrics(path, rows=96, dimension=8):
    rng = np.random.default_rng(11)
    target = rng.normal(size=(rows, dimension))
    operator = np.diag(np.linspace(0.9, 0.2, dimension))
    predicted = target @ operator + 0.02 * rng.normal(size=target.shape)
    downstream = target @ np.linalg.matrix_power(operator, 4) + 0.02 * rng.normal(size=target.shape)
    payload = {
        "protocol": {"checkpoint_epoch": 1, "split": "synthetic-test"},
        "operator_fit_vectors": {
            "target": {"1": target.tolist(), "4": downstream.tolist()},
            "predicted": {"1": predicted.tolist(), "4": downstream.tolist()},
        },
    }
    path.write_text(json.dumps(payload))


def test_orthogonal_conjugates_preserve_eigenvalue_multiset():
    anchor = np.diag([0.9, 0.7, 0.4, 0.2])
    candidates = _candidate_family(anchor, 8, 5)
    assert _eigenvalue_match(anchor, candidates) < 1e-10


def test_benchmark_records_disjoint_three_way_protocol(tmp_path):
    path = tmp_path / "metrics.json"
    _write_metrics(path)
    config = Config(rank=4, candidate_count=7, split_seeds=(0, 1))
    report = run({"synthetic-checkpoint": path}, config)
    assert report["protocol"]["alignment_split"] == "train only"
    assert report["protocol"]["pair_matching_split"] == "validation only"
    assert report["protocol"]["transfer_evaluation_split"] == "test only"
    assert len(report["records"]) == 2
    assert all(record["eigenvalue_match_max_abs"] < 1e-8 for record in report["records"])
    assert all("validation_selection_regret" in record for record in report["records"])


def test_benchmark_explicitly_bounds_checkpoint_probe_claim(tmp_path):
    path = tmp_path / "metrics.json"
    _write_metrics(path)
    report = run({"synthetic-checkpoint": path}, Config(rank=4, candidate_count=5, split_seeds=(0,)))
    assert any("not independently trained" in item for item in report["claim_boundary"])


def test_report_exposes_source_stratification(tmp_path):
    path = tmp_path / "metrics.json"
    _write_metrics(path)
    report = run({"synthetic-checkpoint": path}, Config(rank=4, candidate_count=5, split_seeds=(0, 1)))
    assert "synthetic-checkpoint" in report["source_summary"]
    assert "validation_selection_regret" in report["source_summary"]["synthetic-checkpoint"]
    rendered = markdown(report)
    assert "## Stratified results" in rendered
    assert "Feasible candidates" in rendered
