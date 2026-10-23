from pathlib import Path

import pytest

from experiments.decision_benchmark.audit_counterfactual_arm_disagreement import audit, run


def test_audit_separates_same_choice_roots_from_effective_disagreement() -> None:
    report = {"outcome": {"per_root": {
        "same": {"cost_candidate_id": "0", "endpoint_candidate_id": "0", "cost_regret": 0.2, "endpoint_regret": 0.2},
        "better": {"cost_candidate_id": "0", "endpoint_candidate_id": "1", "cost_regret": 0.5, "endpoint_regret": 0.0},
        "worse": {"cost_candidate_id": "0", "endpoint_candidate_id": "2", "cost_regret": 0.0, "endpoint_regret": 0.3},
    }}}
    result = audit(report, left_arm="cost", right_arm="endpoint")
    assert result["same_candidate_count"] == 1
    assert result["candidate_disagreement_count"] == 2
    assert result["disagreement_stratum"]["right_better_count"] == 1
    assert result["disagreement_stratum"]["left_better_count"] == 1
    assert result["left_arm_choice_diversity"]["winner_counts"] == {"0": 3}
    assert result["left_arm_choice_diversity"]["winner_entropy_normalized"] == 0.0
    assert result["right_arm_choice_diversity"]["distinct_winner_count"] == 3


def test_audit_rejects_missing_arm_fields() -> None:
    with pytest.raises(ValueError, match="missing finite"):
        audit({"outcome": {"per_root": {"root": {}}}}, left_arm="cost", right_arm="endpoint")


def test_audit_reports_alias_overlap_without_treating_it_as_causal_attribution() -> None:
    report = {"outcome": {"per_root": {
        "same": {"cost_candidate_id": "0", "endpoint_candidate_id": "0", "cost_regret": 0.0, "endpoint_regret": 0.0},
        "changed": {"cost_candidate_id": "0", "endpoint_candidate_id": "1", "cost_regret": 0.4, "endpoint_regret": 0.0},
    }}}
    aliasing = {"audit": {"evaluated_query_count": 2, "top_pairs": [
        {"query_root_id": "changed", "near_decision_alias": True},
        {"query_root_id": "same", "near_decision_alias": True},
    ]}}
    result = audit(report, left_arm="cost", right_arm="endpoint", decision_aliasing=aliasing)
    overlap = result["history_aliasing_overlap"]
    assert overlap["overlap_roots"] == ["changed"]
    assert overlap["fraction_of_arm_disagreements_with_alias_witness"] == 1.0
    assert "cannot establish" in overlap["claim_boundary"]


def test_alias_overlap_rejects_truncated_aliasing_artifact() -> None:
    report = {"outcome": {"per_root": {
        "root": {"cost_candidate_id": "0", "endpoint_candidate_id": "1", "cost_regret": 0.4, "endpoint_regret": 0.0},
    }}}
    with pytest.raises(ValueError, match="retain every"):
        audit(report, left_arm="cost", right_arm="endpoint", decision_aliasing={"audit": {"evaluated_query_count": 2, "top_pairs": []}})


def test_run_reports_missing_alias_artifact_without_creating_output(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text('{"outcome": {"per_root": {"root": {"cost_candidate_id": "0", "endpoint_candidate_id": "0", "cost_regret": 0.0, "endpoint_regret": 0.0}}}}', encoding="utf-8")
    output = tmp_path / "result.json"
    with pytest.raises(FileNotFoundError, match="decision-aliasing audit does not exist"):
        run(
            report_path=report,
            output=output,
            left_arm="cost",
            right_arm="endpoint",
            decision_aliasing_audit=tmp_path / "missing.json",
        )
    assert not output.exists()
