from __future__ import annotations

from experiments.summarize_cube_failure_cause import classify_arm


def _ceiling(fixed: list[bool], expanded: list[bool]) -> dict:
    return {
        "records": [
            {
                "seed": index,
                "fixed": {"reachable": fixed[index]},
                "expanded": {"reachable": expanded[index]},
            }
            for index in range(len(fixed))
        ]
    }


def _panel(contact: list[bool], success: list[bool]) -> dict:
    return {
        "test": [
            {"seed": index, "contact_observed": contact[index], "success": success[index]}
            for index in range(len(contact))
        ]
    }


def test_classifies_fixed_bank_unreachable_before_rgb_transfer() -> None:
    result = classify_arm(
        ceiling=_ceiling([False, False], [False, False]),
        panel=_panel([False, False], [False, False]),
    )
    assert result["status"] == "support_failure"


def test_classifies_contact_failure_when_expanded_support_exists() -> None:
    result = classify_arm(
        ceiling=_ceiling([True, True], [True, True]),
        panel=_panel([False, False], [False, False]),
    )
    assert result["status"] == "contact_failure"


def test_classifies_transfer_or_ranker_failure_after_contact() -> None:
    result = classify_arm(
        ceiling=_ceiling([False, True], [True, True]),
        panel=_panel([True, True], [False, True]),
    )
    assert result["status"] == "transfer_or_ranker_failure"


def test_expanded_support_prevents_fixed_bank_from_masking_transfer_failure() -> None:
    result = classify_arm(
        ceiling=_ceiling([False, False], [True, True]),
        panel=_panel([True, True], [False, False]),
    )
    assert result["status"] == "transfer_or_ranker_failure"
    assert result["fixed_bank_only_support_gap_rate"] == 1.0


def test_mixed_panel_reports_per_root_failure_causes() -> None:
    result = classify_arm(
        ceiling=_ceiling([False, True, True], [False, True, True]),
        panel=_panel([False, True, True], [False, False, True]),
    )
    assert result["failure_counts"] == {
        "support_failure": 1,
        "contact_failure": 0,
        "transfer_or_ranker_failure": 1,
        "success": 1,
    }
    assert result["dominant_failure"] == "support_failure"
    assert result["rows"][0]["failure_cause"] == "support_failure"
    assert result["rows"][1]["failure_cause"] == "transfer_or_ranker_failure"


def test_legacy_ceiling_falls_back_to_expanded_for_factorized_fields() -> None:
    result = classify_arm(
        ceiling=_ceiling([False], [True]),
        panel=_panel([True], [False]),
    )
    assert result["factorized_reachability_rate"] == 1.0
    assert result["reachability_recovered_by_factorized_count"] == 1
    assert result["multiphase_reachability_rate"] == 1.0


def test_new_ceiling_reports_multiphase_reachability_separately() -> None:
    result = classify_arm(
        ceiling={
            "records": [{
                "seed": 0,
                "fixed": {"reachable": False},
                "expanded": {"reachable": False},
                "factorized": {"reachable": False},
                "multiphase": {"reachable": True},
            }],
        },
        panel=_panel([True], [False]),
    )
    assert result["factorized_reachability_rate"] == 0.0
    assert result["multiphase_reachability_rate"] == 1.0
    assert result["reachability_recovered_by_multiphase_count"] == 1


def test_multiphase_support_repair_reclassifies_downstream_failure() -> None:
    result = classify_arm(
        ceiling={
            "records": [{
                "seed": 0,
                "fixed": {"reachable": False},
                "expanded": {"reachable": False},
                "factorized": {"reachable": False},
                "multiphase": {"reachable": True},
            }],
        },
        panel=_panel([True], [False]),
    )
    assert result["status"] == "transfer_or_ranker_failure"
    assert result["rows"][0]["failure_cause"] == "transfer_or_ranker_failure"
