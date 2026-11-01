"""Independent synthetic JSON receipt tests; no neural imports."""
from copy import deepcopy
import json

import pytest

from experiments.spectral_reachability.audit_reacher_d2p_p0_predict_accounting import (
    METHODS, audit, load_report,
)


def fixture():
    rows = []
    for index, seed in enumerate((20261040, 20261041)):
        methods = {}
        for name in METHODS:
            calls, samples, old = {
                "d2p_goal_progress": (122, 536, 260), "local_gauss_newton": (122, 536, 260),
                "action_gd": (32, 32, 16), "grasp_style_restart_gd": (34, 40, 17),
                "cem": (8, 256, 128), "random": (0, 0, 0),
            }[name]
            methods[name] = {"receipt": {"forward_rollout_evaluations": old,
                "backward_vjp_rows": 0, "actual_planning_predict": {
                    "completed": True, "failed_predict_calls": 0, "phase": "planning",
                    "method": name, "horizon_tokens": 2, "predict_calls": calls,
                    "predict_samples": samples, "context_lengths": {"3": calls} if calls else {},
                }}}
        rows.append({"seed": seed, "root_index": index, "split": "calibration",
                     "executed_context": {"root_rgb_sha256": str(seed)}, "methods": methods})
    return {"schema": "d2p_reacher_p0_goal_progress.v1", "status": "calibration_only",
            "gates": {"integrity": True}, "protocol": {"calibration_only": True,
                "outcome_only": False, "future_horizon": 2, "history_len": 3, "methods": list(METHODS)},
            "stage_gate": {"p1_unlocked": False}, "rows": rows}


def test_complete_receipts_and_old_unit_comparison():
    report = fixture()
    original = deepcopy(report)
    result = audit(report, expected_seeds=[20261040, 20261041])
    assert report == original
    assert result["status"] == "accounting_receipts_valid"
    assert result["efficacy_claim"] is False and result["P1_admission"] is False
    assert result["old_and_actual_same_unit"] is False
    assert result["evaluation_receipts_available"] is False
    cem = result["rows"][0]["methods"]["cem"]
    assert cem["legacy_forward_rollout_evaluations"] == 128
    assert cem["predict_calls"] == 8 and cem["predict_samples"] == 256
    assert cem["numeric_equal_to_old"] == {"predict_calls": False, "predict_samples": False}
    assert result["rows"][0]["methods"]["random"]["numeric_equal_to_old"]["predict_calls"]
    json.dumps(result)


@pytest.mark.parametrize("field,value", [
    ("completed", False), ("completed", 1), ("failed_predict_calls", 1),
    ("predict_calls", -1), ("predict_calls", True), ("predict_samples", 121),
    ("predict_samples", 1.5), ("predict_samples", "536"),
    ("phase", "evaluation"), ("method", "cem"), ("horizon_tokens", 3),
    ("horizon_tokens", True), ("context_lengths", {"3": 121}),
    ("context_lengths", {"0": 122}), ("context_lengths", {"03": 122}),
    ("context_lengths", {"3": True}), ("context_lengths", {"2": 122}),
])
def test_bad_phase_receipt_rejected(field, value):
    report = fixture()
    report["rows"][0]["methods"]["d2p_goal_progress"]["receipt"]["actual_planning_predict"][field] = value
    with pytest.raises(ValueError):
        audit(report, expected_seeds=[20261040, 20261041])


@pytest.mark.parametrize("mutation", [
    lambda r: r["rows"].append(deepcopy(r["rows"][0])),
    lambda r: r["rows"].pop(),
    lambda r: r["rows"][1].update(seed=20261040),
    lambda r: r["rows"][1].update(root_index=0),
    lambda r: r["rows"][1].update(executed_context=r["rows"][0]["executed_context"]),
    lambda r: r["rows"][0]["methods"].pop("random"),
    lambda r: r["protocol"]["methods"].append("cem"),
    lambda r: r["protocol"].update(methods=["cem"]),
    lambda r: r["rows"][0].update(split="outcome"),
    lambda r: r.update(status="completed"),
    lambda r: r["gates"].update(integrity=False),
    lambda r: r["gates"].clear(),
    lambda r: r["stage_gate"].update(p1_unlocked=True),
    lambda r: r["rows"][0]["methods"]["cem"]["receipt"].pop("actual_planning_predict"),
    lambda r: r["rows"][0]["methods"]["random"]["receipt"]["actual_planning_predict"].update(predict_samples=1),
])
def test_structure_and_coverage_fail_closed(mutation):
    report = fixture()
    mutation(report)
    with pytest.raises(ValueError):
        audit(report, expected_seeds=[20261040, 20261041])


def test_duplicate_json_keys_are_not_silently_discarded(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"rows": [], "rows": []}')
    with pytest.raises(ValueError, match="duplicate"):
        load_report(path)


def test_json_loader_and_frozen_expected_root_set(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(fixture()))
    report = load_report(path)
    with pytest.raises(ValueError):
        audit(report, expected_seeds=[20261040, 20261040])
    with pytest.raises(ValueError):
        audit(report, expected_seeds=[20261040, 99])
    assert audit(report, expected_seeds=[20261041, 20261040])["root_count"] == 2


def test_sample_equality_does_not_imply_call_or_compute_equality():
    report = fixture()
    for row in report["rows"]:
        receipt = row["methods"]["cem"]["receipt"]
        receipt["actual_planning_predict"]["predict_samples"] = 536
        receipt["forward_rollout_evaluations"] = 268
    result = audit(report, expected_seeds=[20261040, 20261041])
    for row in result["rows"]:
        comparison = row["d2p_vs_cem_resource_coordinates"]
        assert comparison == {"predict_samples_equal": True, "predict_calls_equal": False,
                              "full_compute_match_claim": False}
    assert result["compute_matched_claim"] is False


def test_cli_creates_bound_analysis_without_overwriting(tmp_path, monkeypatch):
    from experiments.spectral_reachability.audit_reacher_d2p_p0_predict_accounting import main
    source, output = tmp_path / "source.json", tmp_path / "analysis.json"
    source.write_text(json.dumps(fixture()))
    monkeypatch.setattr("sys.argv", ["audit", "--report", str(source), "--expected-seeds",
                                    "20261040", "20261041", "--output", str(output)])
    main()
    result = json.loads(output.read_text())
    assert len(result["source_sha256"]) == 64
    assert result["source_report"] == str(source.resolve())
    with pytest.raises(FileExistsError):
        main()
