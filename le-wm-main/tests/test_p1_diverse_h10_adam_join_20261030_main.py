import copy
import json

import pytest

from experiments.spectral_reachability import p1_diverse_h10_adam_join_20261030_main as join


def fixture():
    prepared = json.loads((join.ROOT / "results/p1_diverse_h10_20261030_main_prepare_RUN01.json").read_bytes())
    inputs = {name: json.loads((join.ROOT / path).read_bytes())
              for name, (path, _) in join.SOURCES.items()}
    rows = []
    for a in prepared["anchors"]:
        root = join.metrics.physical_metrics(a["root_state"], a["goal"])
        prefix = join.metrics.physical_metrics(a["checkpoint"]["state"], a["goal"])
        controls = {
            role: {"prefix_metrics": next(r["prefix_metrics"] for r in inputs["old"]["rows"]
                                          if r["identity"]["logical_pair"] == a["logical_pair"]
                                          and r["identity"]["role"] == role), "endpoint": root}
            for role in ("zero", "greedy")
        }
        for i in range(42):
            rows.append({"selected": {"logical_pair": a["logical_pair"], "root_metrics": root,
                                      "prefix_metrics": prefix, "probe": "zero" if i == 0 else str(i), "endpoint": root},
                         "controls": controls if i < 41 else {}})
    return prepared, {"rows": rows}, inputs


def test_identity_join_and_censor():
    prepared, table, inputs = fixture()
    result = join.build(prepared, table, **inputs)
    assert len(result["rows"]) == 9
    selected = result["rows"][0]["independent_adam"]
    assert selected["status"] == "native_stopped"
    assert selected["endpoint"] is None
    assert selected["last_observed"]["native_success"]
    assert result["rows"][0]["independent_minus_zero_prefix"] is None
    assert len(result["rows"][0]["common_suffixes"]) == 42
    assert len(result["rows"][1]["common_suffixes"]) == 41
    assert not result["equal_total_budget_certified"]


@pytest.mark.parametrize("change", ["prefix", "checkpoint", "suffix", "control"])
def test_identity_tampering_fails(change):
    prepared, table, inputs = fixture()
    inputs = copy.deepcopy(inputs)
    if change == "prefix":
        prepared["anchors"][0]["prefix5_float32_hex"] = "00"
    elif change == "checkpoint":
        inputs["physical"]["rows"][5]["checkpoints"][1]["snapshot"]["state"][0] += 1
    elif change == "suffix":
        inputs["plan"]["rows"][0]["selected_action_sha256"] = "tampered"
    else:
        inputs["old"]["rows"][7]["identity"]["task_id"] = "tampered"
    with pytest.raises(ValueError):
        join.build(prepared, table, **inputs)
