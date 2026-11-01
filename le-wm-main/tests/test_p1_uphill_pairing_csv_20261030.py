import copy
import hashlib
import json
import sys

import pytest

from experiments.spectral_reachability import p1_uphill_pairing_csv_20261030 as export


def receipt():
    metrics = {name: 1.0 for name in export.FIELDS}
    metrics["native_success"] = False
    anchors, pairs = [], []
    for logical in range(8):
        roles = {}
        for offset, role in enumerate(export.ROLES):
            index = logical * 3 + offset
            plan = {"status": "native_stopped", "endpoint": None, "returned_steps": 7}
            anchors.append({
                "index": index, "identity": {"logical_pair": logical, "role": role},
                "prefix_action_sha256": "action", "prefix_cp5_pixel_sha256": "pixel",
                "root_metrics": metrics, "prefix_metrics": metrics, "prefix_minus_root": metrics,
                "zero_initialization": plan, "warm_initialization": plan,
                "zero_resources": {"actual_usage_verified": False},
                "warm_resources": {"actual_usage_verified": False},
            })
            roles[role] = {"anchor_index": index, "status": "native_stopped",
                           "returned_suffix_steps": 7, "endpoint": None,
                           "endpoint_minus_own_zero_suffix": None,
                           "endpoint_minus_zero_prefix": None,
                           "response_minus_zero_prefix": None}
        for probe in range(42):
            pairs.append({"logical_pair": logical, "probe": str(probe),
                          "probe_sha256": "probe", "roles": copy.deepcopy(roles)})
    return {"schema": "p1_uphill_complete_pairing_20261030", "anchors": anchors,
            "common_suffix_pairs": pairs, "common_suffix_branches": 1008,
            "independent_plans_per_initialization": 24, "P1_admission": False,
            "independent_confirmation": False, "equal_budget_certified": False}


def test_censoring_and_repeated_plans_explicit():
    rows = export.export_rows(receipt())
    assert len(rows) == 1008
    assert rows[0]["common_endpoint_object_position_error"] is None
    assert rows[0]["zero_plan_endpoint_native_success"] is None
    assert rows[0]["independent_plan_repeated_per_probe"]
    assert '"returned_steps": 7' in rows[0]["zero_plan_receipt_json"]
    assert not rows[0]["equal_budget_certified"]


def test_rejects_duplicate_and_mismatched_anchor():
    data = receipt()
    data["common_suffix_pairs"][1]["probe"] = data["common_suffix_pairs"][0]["probe"]
    with pytest.raises(Exception):
        export.export_rows(data)
    data = receipt()
    data["common_suffix_pairs"][0]["roles"]["selected"]["anchor_index"] = 1
    with pytest.raises(Exception):
        export.export_rows(data)


def test_integration_with_frozen_planning_assets():
    from test_p1_uphill_complete_pairing_20261030 import inputs

    prepared, zero, warm, panel = inputs()
    combined = export.pairing.join(prepared, zero, warm, panel)
    rows = export.export_rows(combined)
    assert len(rows) == 1008
    assert len({row["anchor_index"] for row in rows}) == 24
    assert rows[0]["prefix_cp5_pixel_sha256"] == zero["rows"][0]["prefix_cp5_pixel_sha256"]


def test_cli_reports_records_not_independent_samples(tmp_path, monkeypatch, capsys):
    results = tmp_path / "results"
    results.mkdir()
    source = results / "receipt.json"
    raw = json.dumps(receipt()).encode()
    source.write_bytes(raw)
    output = results / "p1_uphill_pairing_test.csv"
    monkeypatch.setattr(export.pairing, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "export", "--receipt", str(source),
        "--receipt-sha256", hashlib.sha256(raw).hexdigest(),
        "--output", str(output),
    ])
    export.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["anchor_records"] == 24
    assert summary["anchor_records_are_independent_samples"] is False
    assert "independent_anchors" not in summary
