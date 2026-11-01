"""Independent bridge tests; stdlib fixtures plus one read-only saved integration."""
from copy import deepcopy
import math

import pytest

from experiments.spectral_reachability import chat_baselines_20261027_bridge as bridge


def contract():
    return {k: {"value": "explicit-" + k, "source": "fixture.json#/" + k}
            for k in bridge.DIMENSIONS}


def test_even_metadata_match_never_authorizes_performance_claim():
    result = bridge.compare_contracts(contract(), contract())
    assert result["metadata_match_only"]
    assert result["direct_comparison_authorized"] is False
    assert result["improvement_percent"] is None
    assert result["scientific_admission"] is False


@pytest.mark.parametrize("field", bridge.DIMENSIONS)
def test_each_dimension_is_fail_closed(field):
    left, right = contract(), contract()
    right[field]["value"] = "different"
    assert bridge.compare_contracts(left, right)["fields"][field] == "mismatch"
    right[field]["value"] = None
    assert bridge.compare_contracts(left, right)["fields"][field] == "unknown"


@pytest.mark.parametrize("unknown", [None, "unknown", "unmeasured", " ", {"calls": 4, "backward": None}, float("nan")])
def test_unknown_equals_unknown_is_not_match(unknown):
    left, right = contract(), contract()
    left["budget"]["value"] = right["budget"]["value"] = unknown
    assert bridge.compare_contracts(left, right)["fields"]["budget"] == "unknown"


def test_empty_feedback_is_explicit_but_missing_source_rejected():
    left, right = contract(), contract()
    left["feedback"]["value"] = right["feedback"]["value"] = []
    assert bridge.compare_contracts(left, right)["fields"]["feedback"] == "match"
    right["feedback"]["source"] = ""
    with pytest.raises(ValueError, match="source locator"):
        bridge.compare_contracts(left, right)


def test_absent_dimension_rejected():
    left = contract()
    del left["checkpoint"]
    with pytest.raises(ValueError, match="complete dimension"):
        bridge.compare_contracts(left, contract())


def test_table_parser_does_not_merge_nested_tables():
    parser = bridge.Tables()
    parser.feed('<table id="outer"><tr><td>A<table id="inner"><tr><td>B</td></tr></table></td></tr></table>')
    assert parser.tables == {"inner": [["B"]], "outer": [["A"]]}


def test_rounding_consistency_is_not_exact_mean_or_confidence_interval():
    result = bridge.weighted_rounding_check([87.6, 48.3, 51.9, 37.2], [80, 157, 96, 103], 53.7)
    assert result["rounding_compatible"] and result["denominator"] == 436
    assert result["weighted_rounded_cells"] != "53.7"
    assert not result["is_confidence_interval"] and not result["new_model_evaluation"]
    assert not bridge.weighted_rounding_check([87.6, 48.3, 51.9, 37.2], [80, 157, 96, 103], 60)["rounding_compatible"]


@pytest.mark.parametrize("values,weights,average", [([], [], 0), ([1], [0], 1), ([1], [True], 1),
                                                   ([1], [1, 2], 1), ([math.inf], [1], 1), ([1], [1], 101)])
def test_bad_rounding_inputs_rejected(values, weights, average):
    with pytest.raises(ValueError):
        bridge.weighted_rounding_check(values, weights, average)


def test_precision_and_boundary_intervals():
    with pytest.raises(ValueError, match="one-decimal"):
        bridge.weighted_rounding_check([1.23], [1], 1.2)
    result = bridge.weighted_rounding_check([0, 100], [1, 1], 50)
    assert result["weighted_rounding_interval"] == ["49.975", "50.025"]


def test_real_saved_bridge_no_new_neural_or_physics():
    report = bridge.build()
    assert report["reported_column"]["numeric_cells"] == 236
    assert report["local_column"]["GRASP"]["summary"]["native_successes"]["value"] == 0
    assert report["local_column"]["GRASP"]["summary"]["native_denominator"] == 2
    assert report["local_column"]["Hi-LeWM"]["value"] is None
    assert report["local_column"]["PlanWAM"]["value"] is None
    assert all(x["rounding_compatible"] for x in report["planwam_arithmetic_supplement"].values())
    assert report["new_neural_calls"] == report["new_physical_steps"] == report["network_calls"] == 0
    assert len(report["input_sha256"]) == 10  # 3 JSON + 4 raw HTML + 3 raw PDF
    assert set(report["comparability_matrix"]) == set(bridge.COUNTS)
    for item in report["comparability_matrix"].values():
        assert item["preflight"]["metadata_match_only"] is False
        assert item["preflight"]["improvement_percent"] is None


def test_input_corruption_rejected(tmp_path):
    # Test file creation is isolated pytest temporary I/O, not a shared artifact.
    path = tmp_path / (bridge.PREFIX + "fixture.json")
    path.write_text("{}")
    with pytest.raises(ValueError, match="SHA changed"):
        bridge.Evidence(tmp_path).raw(path.name, "0" * 64)


def test_duplicate_local_root_cannot_shrink_denominator():
    report = bridge.build()
    rows = deepcopy(report["local_column"]["GRASP"]["rows"])
    saved = {"status": "comparison_recorded_not_admitted", "rows": [], "summary": []}
    saved.update({k: False for k in ("official_reproduction", "equal_total_compute", "independent_confirmation",
                                   "P1_admission", "P2_admission")})
    for row in rows:
        row["arm"] = "GRASP_H3_bounded"
        row["source_slot"] = 0
    saved["rows"] = rows
    with pytest.raises(ValueError, match="both fixed roots"):
        bridge.local_projection(saved, "GRASP_H3_bounded")


def test_output_scope_and_overwrite_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="direct results child"):
        bridge.publish({}, tmp_path / (bridge.PREFIX + "output"))
    monkeypatch.setattr(bridge, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    output = tmp_path / "results" / (bridge.PREFIX + "test")
    bridge.publish({}, output)
    with pytest.raises(ValueError, match="already exists"):
        bridge.publish({}, output)
