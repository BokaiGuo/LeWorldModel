import numpy as np

from experiments.decision_benchmark.audit_htsc_transport_support_shift import (
    _root_latent_rows,
    fit_latent_mahalanobis_support,
    freeze_conformal_abstain_rule,
    mahalanobis_scores,
    stratify_frozen_rule,
)


def test_transport_rule_is_input_only_and_uses_finite_sample_calibration_rank():
    rule = freeze_conformal_abstain_rule(np.arange(14, dtype=np.float64), alpha=0.1)
    assert rule["threshold"] == 13.0
    assert rule["calibration"]["finite_sample_rank"] == 14
    assert rule["calibration"]["finite_sample_marginal_coverage_lower_bound"] == 14 / 15
    assert rule["pre_freeze_outcome_access"] is False
    assert len(rule["rule_sha256"]) == 64


def test_transport_model_scores_out_of_support_features_higher():
    train = np.asarray([[0.0, 0.0], [0.2, 0.1], [-0.1, 0.0], [0.1, -0.2]])
    model = fit_latent_mahalanobis_support(train, relative_ridge=1e-3)
    scores = mahalanobis_scores(np.asarray([[0.0, 0.0], [4.0, 4.0]]), model)
    assert scores.shape == (2,)
    assert scores[1] > scores[0]


def test_transport_stratification_keeps_detection_and_certificate_claims_separate():
    rule = {
        "threshold": 1.0,
        "rule_sha256": "frozen",
    }
    report = stratify_frozen_rule(
        ["a", "b", "c", "d"],
        np.asarray([0.1, 0.2, 2.0, 3.0]),
        rule,
        {"a": True, "b": True, "c": False, "d": False},
    )
    assert report["supported"]["root_count"] == 2
    assert report["abstained"]["root_count"] == 2
    assert report["detection_utility"]["enriches_failures"] is True
    assert report["detection_utility"]["failure_capture_recall"] == 1.0
    assert report["certificate_validity"]["status"] == "not_established"


def test_transport_stratification_rejects_missing_sealed_coverage_labels():
    try:
        stratify_frozen_rule(["root"], np.asarray([0.0]), {"threshold": 1.0}, {})
    except ValueError as error:
        assert "coverage labels missing" in str(error)
    else:
        raise AssertionError("missing external coverage labels were silently accepted")


def test_transport_feature_builder_never_requires_external_outcome_fields():
    rows = [
        {
            "instance_id": "fresh-root",
            "split": "external_outcome",
            "candidate_id": 0,
            "action_sequence": [[0.1, 0.0], [0.1, 0.0]],
            "source_latent_model": [0.0, 1.0],
        },
        {
            "instance_id": "fresh-root",
            "split": "external_outcome",
            "candidate_id": 1,
            "action_sequence": [[0.0, 0.1], [0.0, 0.1]],
            "source_latent_model": [0.0, 1.0],
        },
    ]
    features, bank = _root_latent_rows(rows, allowed_splits={"external_outcome"})
    assert features["fresh-root"].tolist() == [0.0, 1.0]
    assert set(bank) == {"0", "1"}
