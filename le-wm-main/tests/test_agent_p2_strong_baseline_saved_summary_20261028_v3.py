"""RAM and sealed-metadata tests for the matched-baseline contract."""
from copy import deepcopy
import json
import os
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p2_strong_baseline_saved_summary_20261028_v3 as m


def complete(algorithm):
    return {field: {
        "task_identity": "pusht-known",
        "checkpoint_identity": "frozen-checkpoint-sha",
        "action_support": {"shape": [10, 2], "normalizer": "frozen-action-normalizer"},
        "observation_spec": {"rgb": [224, 224, 3], "history": [0, 5, 10]},
        "eval_metric": "native_official_evaluation",
        "horizon": 2,
        "resource_axes": {"predict": 144, "action_encoder": 144, "wall_seconds": 900},
        "root_ids": [0, 1, 2, 3],
        "goal_identity": "goal-sha",
        "source_authority": "source-report-sha",
    }[field] for field in m.CONTROL_FIELDS} | {"algorithm": algorithm, "family": algorithm + "-family"}


def test_different_algorithms_are_comparable_when_controls_match():
    value = m.compare_controls(complete("beam"), complete("sampling"))
    assert value["status"] == "matched_controls_algorithms_may_differ"
    assert value["mismatches"] == {}
    assert value["delta"] is None


def test_family_and_algorithm_are_not_hidden_control_fields():
    left, right = complete("beam"), complete("sampling")
    right["family"] = "different"
    right["algorithm"] = "different"
    assert m.compare_controls(left, right)["status"] == "matched_controls_algorithms_may_differ"


@pytest.mark.parametrize("field", m.CONTROL_FIELDS)
def test_unknown_control_refuses(field):
    left, right = complete("beam"), complete("sampling")
    left[field] = None
    with pytest.raises(ValueError):
        m.compare_controls(left, right)


def test_known_control_mismatch_refuses_direct_comparison():
    left, right = complete("beam"), complete("sampling")
    right["horizon"] = 10
    value = m.compare_controls(left, right)
    assert value["status"] == "control_mismatch_no_direct_comparison"
    assert value["delta"] is None
    assert "horizon" in value["mismatches"]


@pytest.mark.parametrize("field", m.CONTROL_FIELDS)
def test_every_known_control_mismatch_is_exposed(field):
    left, right = complete("beam"), complete("sampling")
    right[field] = {"changed_control": True}
    decision = m.compare_controls(left, right)
    assert decision["status"] == "control_mismatch_no_direct_comparison"
    assert set(decision["mismatches"]) == {field}


def test_predict_api_and_samples_are_separate_controls():
    left, right = complete("beam"), complete("sampling")
    right["resource_axes"]["predictor_samples"] = 288
    right["resource_axes"]["predict_API"] = 144
    assert m.compare_controls(left, right)["status"] == "control_mismatch_no_direct_comparison"


def test_real_assets_are_all_roots_arms_and_fail_closed():
    value = m.build()
    assert value["run_authorized"] is False
    assert set(value["assets"]) == {"kinodynamic", "no_composition", "sampling"}
    assert all(item["row_count"] in {2, 4} for item in value["assets"].values())
    assert all(item["unknown_controls"] for item in value["assets"].values())
    assert all(item["status"] == "unknown_control_refuse" for item in value["comparisons"].values())


def test_missing_asset_hash_refuses(monkeypatch):
    original = m.ASSETS["sampling"]
    monkeypatch.setitem(m.ASSETS, "sampling", (original[0], "0" * 64))
    with pytest.raises(ValueError):
        m.load_assets()


def test_contract_is_json_serializable_and_does_not_import_runtime():
    json.dumps(m.build(), allow_nan=False)
    code = """import sys
class Guard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','gym','gymnasium','pymunk','stable_worldmodel'}: raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from experiments.spectral_reachability import agent_p2_strong_baseline_saved_summary_20261028_v3
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=m.ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("value", [None, {}, [], "unknown", {"nested": None}, ["unknown"], float("nan")])
def test_nested_unknown_refuses(value):
    left, right = complete("beam"), complete("sampling")
    left["checkpoint_identity"] = value
    with pytest.raises(ValueError):
        m.compare_controls(left, right)


def budget():
    # Synthetic explicit RAM counts, not inferred real experiment costs.
    return {axis: {"unit": "seconds" if axis == "wall_seconds" else ("samples" if axis == "predictor_samples" else ("steps" if axis == "physics_steps" else "calls")),
                   "scope": "per_root_per_arm_including_signal_search_replay", "limit": 144, "inherited": 8}
            for axis in m.RESOURCE_AXES}


def bound_request(tmp_path):
    request = m.request_template()
    for name, method in request["methods"].items():
        controls = complete(name)
        controls.pop("algorithm")
        controls.pop("family")
        controls["resource_axes"] = budget()
        method["algorithm"] = name
        method["controls"] = controls
        method["parameters"] = {"mode": name, "tie_break": "stable-index", "padding": "forbidden"}
        method["resource_budget"] = budget()
        witness = {**controls, "parameters": method["parameters"], "resource_budget": method["resource_budget"]}
        path = tmp_path / (name + ".json")
        m.write(path, witness)
        method["evidence"] = {field: {"path": str(path), "sha256": m.sha(path), "pointer": "/" + field} for field in witness}
    return request


def test_prepare_and_saved_verify_end_to_end_ram(tmp_path):
    request = bound_request(tmp_path)
    prepared = m.prepare(request)
    assert all(row["status"] == "matched_controls_algorithms_may_differ" for row in prepared["comparisons"].values())
    assert prepared["run_authorized"] is False and prepared["fairness_claim"] is False
    path, reqpath = tmp_path / "prepared.json", tmp_path / "request.json"
    m.write(path, prepared)
    m.write(reqpath, request)
    assert m.verify(path, m.sha(path), reqpath) == prepared


@pytest.mark.parametrize("damage", ["root", "arm", "method", "evidence", "unknown", "budget", "hash", "scope"])
def test_prepare_mutations_refuse(tmp_path, damage):
    request = bound_request(tmp_path)
    method = request["methods"]["sampling"]
    if damage == "root":
        method["fixed_rows"].pop()
    elif damage == "arm":
        method["fixed_rows"][0]["arm"] = "omitted"
    elif damage == "method":
        del request["methods"]["sampling"]
    elif damage == "evidence":
        method["evidence"]["goal_identity"]["pointer"] = "/task_identity"
    elif damage == "unknown":
        method["controls"]["checkpoint_identity"] = None
    elif damage == "budget":
        del method["resource_budget"]["AD_forward"]
    elif damage == "hash":
        method["evidence"]["parameters"]["sha256"] = "0" * 64
    else:
        method["resource_budget"]["predict_API"]["scope"] = "search_only"
    with pytest.raises(ValueError):
        m.prepare(request)


def test_template_is_incomplete_no_implicit_zero():
    template = m.request_template()
    assert template["methods"]["sampling"]["resource_budget"]["VJP"] is None
    with pytest.raises(ValueError):
        m.prepare(template)


def test_native_resource_scopes_preserved_not_merged():
    assets = m.load_assets()
    costs = assets["sampling"]["native_resource_evidence"]
    assert costs["sampling_search"]["predict"]["completed"] == 504
    assert costs["physical"]["step"]["completed"] == 40
    assert costs["inherited"]["direct_resources"]["predict"]["completed"] == 120
    assert costs["predictor_samples"] is None


def test_all_axes_needed_and_explicit_zero_is_allowed():
    value = budget()
    value["AD_forward"]["limit"] = 0
    m.validate_budget(value)
    del value["AD_forward"]
    with pytest.raises(ValueError):
        m.validate_budget(value)


def test_cli_saved_inventory_roundtrip(tmp_path):
    path = tmp_path / "inventory.json"
    module = "experiments.spectral_reachability." + m.NAME
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    for arguments in (["inventory", "--output", str(path)], ["verify", "--contract", str(path), "--sha", None]):
        arguments = [m.sha(path) if arg is None else arg for arg in arguments]
        result = subprocess.run([sys.executable, "-B", "-m", module, *arguments], cwd=m.ROOT, capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stdout + result.stderr
    assert m.read(path)["run_authorized"] is False


def test_saved_tamper_and_exclusive_output_refused(tmp_path):
    path = tmp_path / "contract.json"
    value = m.build()
    value["run_authorized"] = True
    m.write(path, value)
    with pytest.raises(ValueError, match="reconstruction"):
        m.verify(path, m.sha(path))
    with pytest.raises(ValueError, match="exclusive"):
        m.write(path, value)
