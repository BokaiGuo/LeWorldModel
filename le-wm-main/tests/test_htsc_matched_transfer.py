from __future__ import annotations

import copy

import pytest

from experiments.decision_benchmark.run_htsc_matched_transfer import evaluate_records


def _record(
    root: str,
    split: str,
    candidate_id: int,
    *,
    wm_score: float,
    fallback_score: float,
    true_cost: float,
    success: bool,
    risk_score: float = 0.1,
    support_passed: bool = True,
) -> dict[str, object]:
    return {
        "root_id": root,
        "split": split,
        "candidate_id": candidate_id,
        "action_sequence": [[float(candidate_id), 0.0]],
        "wm_score": wm_score,
        "fallback_score": fallback_score,
        "true_cost": true_cost,
        "success": success,
        "risk_score": risk_score,
        "support_passed": support_passed,
    }


def _panel() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for root_index in range(2):
        root = f"train-{root_index}"
        rows.extend([
            _record(root, "train", 0, wm_score=0.2, fallback_score=0.1, true_cost=0.1, success=True),
            _record(root, "train", 1, wm_score=0.8, fallback_score=0.9, true_cost=0.9, success=False),
        ])
    # Ten roots make alpha=.1 finite: ceil((10 + 1) * .9) = 10.
    for root_index in range(10):
        root = f"cal-{root_index}"
        rows.extend([
            _record(root, "calibration", 0, wm_score=0.1, fallback_score=0.9, true_cost=0.1, success=True),
            _record(root, "calibration", 1, wm_score=0.9, fallback_score=0.1, true_cost=0.9, success=False),
        ])
    rows.extend([
        _record("test-permit", "test", 0, wm_score=0.0, fallback_score=0.9, true_cost=0.0, success=True),
        _record("test-permit", "test", 1, wm_score=1.0, fallback_score=0.1, true_cost=1.0, success=False),
        _record("test-risk-fallback", "test", 0, wm_score=0.0, fallback_score=0.8, true_cost=1.0, success=False, risk_score=10.0),
        _record("test-risk-fallback", "test", 1, wm_score=1.0, fallback_score=0.1, true_cost=0.0, success=True, risk_score=10.0),
    ])
    return rows


def test_matched_fallback_only_differs_when_frozen_htsc_permits():
    report = evaluate_records(_panel(), alpha=0.1, risk_quantile=0.9, bootstrap_draws=100, seed=7)

    assert report["protocol"]["train_calibration_test_root_disjoint"] is True
    assert report["calibration"]["contrast"]["status"] == "ok"
    assert report["test"]["released_root_count"] == 1
    assert report["test"]["matched_fallback_root_count"] == 2
    assert report["test"]["paired_regret_gain_mean"] == pytest.approx(0.5)
    assert report["test"]["paired_success_gain_mean"] == pytest.approx(0.5)

    roots = {row["root_id"]: row for row in report["test"]["roots"]}
    assert roots["test-permit"]["status"] == "permit"
    assert roots["test-permit"]["htsc_candidate_id"] == 0
    assert roots["test-permit"]["matched_fallback_candidate_id"] == 1
    assert roots["test-risk-fallback"]["status"] == "probe"
    assert roots["test-risk-fallback"]["htsc_candidate_id"] == 1
    assert roots["test-risk-fallback"]["matched_fallback_candidate_id"] == 1


def test_test_outcomes_change_metrics_but_never_the_frozen_gate_or_choices():
    original = evaluate_records(_panel(), alpha=0.1, risk_quantile=0.9, bootstrap_draws=100, seed=7)
    changed = copy.deepcopy(_panel())
    for row in changed:
        if row["split"] == "test":
            row["true_cost"] = 1.0 - float(row["true_cost"])
            row["success"] = not bool(row["success"])
    replay = evaluate_records(changed, alpha=0.1, risk_quantile=0.9, bootstrap_draws=100, seed=7)

    old_decisions = [(row["root_id"], row["status"], row["htsc_candidate_id"], row["matched_fallback_candidate_id"])
                     for row in original["test"]["roots"]]
    new_decisions = [(row["root_id"], row["status"], row["htsc_candidate_id"], row["matched_fallback_candidate_id"])
                     for row in replay["test"]["roots"]]
    assert new_decisions == old_decisions
    assert replay["test"]["paired_regret_gain_mean"] != original["test"]["paired_regret_gain_mean"]


def test_rejects_candidate_sets_that_are_not_fixed_across_roots():
    malformed = _panel()
    malformed[-1]["action_sequence"] = [[99.0, 0.0]]
    with pytest.raises(ValueError, match="fixed action sequence"):
        evaluate_records(malformed, alpha=0.1, risk_quantile=0.9)


def test_paired_source_clusters_are_the_conformal_and_bootstrap_units():
    rows = _panel()
    # Ten calibration roots are enough only if they are independent. Pairing
    # them leaves five source clusters, so 90% split conformal must fail
    # closed instead of borrowing an apparent sample size of ten.
    clusters = {
        **{f"train-{index}": "train-source" for index in range(2)},
        **{f"cal-{index}": f"cal-source-{index // 2}" for index in range(10)},
        "test-permit": "test-source", "test-risk-fallback": "test-source",
    }
    report = evaluate_records(rows, alpha=.1, risk_quantile=.9, bootstrap_draws=100, seed=7, cluster_by_root=clusters)
    assert report["protocol"]["independent_unit"] == "declared cluster"
    assert report["protocol"]["cluster_counts"] == {"train": 1, "calibration": 5, "test": 1}
    assert report["calibration"]["contrast"]["status"] == "insufficient_calibration"
    assert report["test"]["cluster_conformal_outcome_coverage"]["outcome_cluster_count"] == 1


def test_cluster_cannot_cross_protocol_splits():
    clusters = {root: root for root in {str(row["root_id"]) for row in _panel()}}
    clusters["test-permit"] = clusters["cal-0"]
    with pytest.raises(ValueError, match="cannot cross"):
        evaluate_records(_panel(), alpha=.1, risk_quantile=.9, cluster_by_root=clusters)
