import copy
import json
import math

import pytest

from experiments.spectral_reachability import reacher_pc_p1_fd_samebank_greedy_alias_20261102_codexexclusive as mod


def rows():
    return [{"proposal_index": i, "immediate_cost": float(i)} for i in range(9)]


def test_argmin_includes_zero_and_ties_choose_lowest_index():
    values = rows()
    values[7]["immediate_cost"] = -2.
    values[3]["immediate_cost"] = -2.
    assert mod.greedy_index(values) == 3
    assert mod.greedy_index(rows()) == 0


@pytest.mark.parametrize("mutation", ["missing", "reordered", "nan", "inf", "bool"])
def test_bad_roster_and_scores_fail_closed(mutation):
    values = rows()
    if mutation == "missing":
        values.pop()
    elif mutation == "reordered":
        values[1], values[2] = values[2], values[1]
    else:
        values[2]["immediate_cost"] = {"nan": math.nan, "inf": math.inf, "bool": True}[mutation]
    with pytest.raises(ValueError):
        mod.greedy_index(values)


def test_unrelated_fields_cannot_change_selection():
    values = rows()
    changed = copy.deepcopy(values)
    changed[0]["physical_outcome"] = 1000000
    changed[4]["final_costs"] = [-1000000]
    assert mod.greedy_index(values) == mod.greedy_index(changed) == 0


def test_real_frozen_source_selects_zero_and_preserves_dtype_and_actions():
    plan = mod.prepare()
    assert [r["greedy_index"] for r in plan["roots"]] == [0, 0]
    assert [r["selected_index"] for r in plan["roots"]] == [4, 4]
    assert all(r["greedy_aliases_parameter_zero"] for r in plan["roots"])
    assert plan["physical_outcome_used_for_selection"] is False
    assert plan["selection_frozen_before_old_outcomes"] is False
    assert plan["parameter_dtype"] == "float64"


def test_alias_requires_exact_index_params_and_action():
    fixed = {"greedy_index": 0, "params": [[0.] * 10], "raw_prefix": [[0., 0.]] * 5}
    mod.require_alias(fixed, fixed)
    for key, value in (("greedy_index", 2), ("params", [[1.] * 10]), ("raw_prefix", [[1., 0.]] * 5)):
        broken = dict(fixed, **{key: value})
        with pytest.raises(ValueError):
            mod.require_alias(broken, fixed)


def test_consume_real_saved_records_and_reject_mutations(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(mod.prepare(), sort_keys=True))
    digest = mod.sha(path)
    report = mod.consume(path, digest)
    assert report["status"] == "verified_existing_arm_alias"
    assert [r["seed"] for r in report["roots"]] == [20263160, 20263161]
    assert all(r["alias_of"] == "parameter_zero" for r in report["roots"])
    assert report["new_physical_executions"] == report["new_cem_executions"] == 0
    altered = json.loads(path.read_text())
    altered["roots"][0]["greedy_index"] = 4
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="plan hash"):
        mod.consume(path, digest)
    with pytest.raises(ValueError, match="plan or source drift"):
        mod.consume(path, mod.sha(path))
    path.write_text(json.dumps(mod.prepare(), sort_keys=True))
    wrong_saved = tmp_path / "saved.json"
    wrong_saved.write_text("{}")
    with pytest.raises(ValueError, match="archived source mismatch"):
        mod.consume(path, mod.sha(path), archive=wrong_saved)
