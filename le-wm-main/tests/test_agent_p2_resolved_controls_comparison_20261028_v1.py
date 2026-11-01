"""RAM comparison mutations plus fixed actual saved input closure."""
from copy import deepcopy

import pytest

from experiments.spectral_reachability import agent_p2_resolved_controls_comparison_20261028_v1 as m


def asset(arm="a"):
    controls = {field: [{"component": 1}, {"component": 2}] for field in m.v3.CONTROL_FIELDS}
    controls["task_identity"] = [{"/task_id": "t0"}, {"/task_id": "t1"}]
    controls["root_ids"] = [0, 1]
    controls["resource_axes"] = None
    return {"controls": controls, "fixed_rows": [{"root": 0, "arm": arm}, {"root": 1, "arm": arm}]}


def test_algorithm_arm_differences_not_control_fields():
    result = m.compare(asset("beam"), asset("GN"))
    assert set(result["fields"]) == set(m.v3.CONTROL_FIELDS)
    assert result["fields"]["task_identity"]["status"] == "match"
    assert result["fields"]["resource_axes"]["status"] == "unknown"
    assert not result["matched_experiment_authorized"]
    assert not result["all_metadata_controls_match"]


@pytest.mark.parametrize("field", ["action_support", "goal_identity", "checkpoint_identity", "observation_spec", "eval_metric", "horizon"])
def test_one_root_mismatch_survives(field):
    left, right = asset(), asset()
    right["controls"][field][1] = {"component": 3}
    result = m.compare(left, right)
    assert result["fields"][field]["status"] == "mismatch"
    assert [r["status"] for r in result["fields"][field]["by_task"]] == ["match", "mismatch"]


def test_task_order_join_not_position():
    left, right = asset(), asset()
    for field, value in right["controls"].items():
        if isinstance(value, list):
            right["controls"][field] = list(reversed(value))
    right["fixed_rows"].reverse()
    result = m.compare(left, right)
    assert result["fields"]["goal_identity"]["status"] == "match"


def test_equal_slots_without_task_identity_unknown():
    left, right = asset(), asset()
    right["controls"]["task_identity"] = None
    result = m.compare(left, right)
    assert result["fields"]["root_ids"]["status"] == "unknown"
    assert result["fields"]["goal_identity"]["status"] == "unknown"


def test_changed_task_set_no_subset_comparison():
    left, right = asset(), asset()
    right["controls"]["task_identity"][1] = {"/task_id": "other"}
    result = m.compare(left, right)
    assert result["fields"]["task_identity"]["status"] == "mismatch"
    assert result["fields"]["root_ids"]["status"] == "mismatch"
    assert result["fields"]["goal_identity"]["status"] == "unknown"
    assert len(result["right_fixed_rows"]) == 2


def test_partial_component_never_zero():
    left, right = asset(), asset()
    right["controls"]["action_support"][0] = {"component": None}
    assert m.compare(left, right)["fields"]["action_support"]["status"] == "unknown"


def test_repeated_task_conflicting_controls_refused():
    value = asset()
    value["controls"]["task_identity"][1] = {"/task_id": "t0"}
    with pytest.raises(ValueError, match="within-task"):
        m.by_task(value, "goal_identity")


def test_repeated_arms_preserved_when_task_controls_agree():
    value = asset()
    value["controls"]["task_identity"][1] = {"/task_id": "t0"}
    for field in m.v3.CONTROL_FIELDS:
        if field not in ("task_identity", "root_ids", "resource_axes"):
            value["controls"][field][1] = deepcopy(value["controls"][field][0])
    assert len(m.compare(value, value)["left_fixed_rows"]) == 2


def test_actual_full_closure():
    result = m.build()
    assert {name: len(rows) for name, rows in result["fixed_rows"].items()} == {
        "kinodynamic": 2, "no_composition": 2, "sampling": 4}
    pair = result["comparisons"]["kinodynamic__vs__no_composition"]
    assert sum(f["status"] == "match" for f in pair["fields"].values()) == 9
    assert pair["fields"]["resource_axes"]["status"] == "unknown"
    assert all(f["status"] == "unknown" for f in result["comparisons"]["kinodynamic__vs__sampling"]["fields"].values())
    assert result["author_reported_values"] == [] and result["local_performance_ranking"] is None
    assert not result["P2_admission"]


def test_asset_denominator_mutation_refused():
    catalog = m.resolver.load(m.CATALOG, m.CATALOG_SHA)
    inventory = m.resolver.load(m.CONTRACT, m.CONTRACT_SHA)
    catalog["assets"]["sampling"]["fixed_rows"].pop()
    with pytest.raises(ValueError, match="denominator"):
        m.validate_inputs(catalog, inventory)
