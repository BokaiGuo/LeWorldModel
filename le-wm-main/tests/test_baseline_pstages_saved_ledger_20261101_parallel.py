import hashlib
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/spectral_reachability/baseline_pstages_saved_ledger_20261101_parallel.py"
)
SPEC = importlib.util.spec_from_file_location("baseline_pstages_saved_ledger_parallel", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def manifest(asset_ids, categories=(), operations=(), requested=False, cohort=None, ranking=False):
    return {
        "schema": MODULE.NAME + ".preflight.v1",
        "asset_ids": list(asset_ids),
        "required_categories": list(categories),
        "required_operations": list(operations),
        "comparison": {
            "requested": requested,
            "ranking": ranking,
            "protocol_cohort": cohort,
        },
    }


def test_inventory_verifies_pinned_local_assets_without_authorizing_claims():
    result = MODULE.inventory()
    ids = {row["asset_id"] for row in result["rows"]}
    assert {
        "p2_zero_h3",
        "p2_no_composition_sra",
        "p2_gn_h15",
        "p2_cem_h15",
        "p2_grasp_h1_local",
        "p3_fixed_schedule",
        "p3_action_jacobian_schedule",
        "p3_reacher_cem_feedback",
        "p4_feedback_pair",
        "p5_architecture_readiness",
    } == ids
    assert result["saved_assets_verified"] is True
    assert result["paper_reported_values_included"] is False
    assert result["fair_comparison_verified"] is False
    assert all(row["stage_admission"] is False for row in result["rows"])
    assert all(row["entrypoint_present"] is True for row in result["rows"])


def test_zero_is_only_zero_when_explicitly_absent_by_contract():
    row = next(row for row in MODULE.inventory()["rows"] if row["asset_id"] == "p2_zero_h3")
    assert row["resources"]["predict"]["value"] == 0
    assert row["resources"]["predict"]["status"] == "absent_by_contract"
    assert row["resources"]["physical_step"]["status"] == "measured"
    assert row["resources"]["physical_step"]["value"] == 120


def test_complete_saved_only_preflight_for_explicit_p4_axes():
    request = manifest(
        ["p4_feedback_pair"],
        operations=["rgb_encode", "project", "action_encode", "predict",
                    "backward", "vjp", "jacobian", "physical_reset",
                    "physical_step", "end_to_end_wall", "peak_cuda_allocated"],
    )
    result = MODULE.preflight(request)
    assert result["resource_coverage_complete"] is True
    assert result["executable_saved_only"] is True
    assert result["fair_comparison_verified"] is False
    assert result["P4_admission"] is False


def test_missing_category_is_fail_closed_not_zero_filled():
    result = MODULE.preflight(manifest(["p2_gn_h15"], categories=["prediction"]))
    assert result["resource_coverage_complete"] is False
    assert result["executable_saved_only"] is False
    assert "predict" in result["missing_operations"]["p2_gn_h15"]


def test_unknown_operation_rejected():
    with pytest.raises(ValueError, match="unknown operation"):
        MODULE.preflight(manifest(["p2_zero_h3"], operations=["paper_score"]))


def test_cross_protocol_comparison_and_ranking_rejected():
    request = manifest(
        ["p2_zero_h3", "p2_no_composition_sra"],
        requested=True,
        cohort="p2_zero_h3_four_known_roots",
        ranking=True,
    )
    with pytest.raises(ValueError, match="protocol cohort mismatch"):
        MODULE.preflight(request)


def test_same_cohort_saved_comparison_still_does_not_authorize_ranking():
    cohort = "p2_h15_matched_host_two_known_roots"
    request = manifest(
        ["p2_gn_h15", "p2_cem_h15"],
        operations=["end_to_end_wall", "peak_cuda_allocated"],
        requested=True,
        cohort=cohort,
        ranking=True,
    )
    result = MODULE.preflight(request)
    assert result["resource_coverage_complete"] is True
    assert result["executable_saved_only"] is False
    assert result["local_performance_ranking_authorized"] is False


def test_manifest_is_exact_and_cannot_smuggle_paper_reported_values():
    request = manifest(["p2_zero_h3"])
    request["paper_reported_score"] = 59.0
    with pytest.raises(ValueError, match="exact preflight manifest fields"):
        MODULE.preflight(request)


def test_saved_report_hash_tamper_is_rejected(tmp_path):
    original = MODULE.ASSETS["p2_zero_h3"]
    source = MODULE.ROOT / original["path"]
    tampered = tmp_path / "report.json"
    tampered.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        MODULE._read_json(tampered, original["sha256"])


def test_all_pinned_hashes_match_current_saved_bytes():
    for spec in MODULE.ASSETS.values():
        raw = (MODULE.ROOT / spec["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == spec["sha256"]
