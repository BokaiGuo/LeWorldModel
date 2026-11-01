"""Pure cohort/metadata contracts, not neural model or efficacy evidence."""
import copy

import numpy as np
import pytest

from experiments.spectral_reachability import reacher_d2p_model_escape_admission as admission


def task(*, eligible=True):
    values = (1.,.8,.6,.6,.3) if eligible else (1.,.8,.6,.4,.3)
    return {name:{"cost":value} for name,value in zip(("zero","gn4","gn8","gn16","cem"),values)}


def pool():
    return [{"seed":i,"source_episode":100+i,"tasks":{name:task() for name in admission.TASKS}} for i in range(32)]


def test_incumbent_monotonicity_and_strict_model_escape():
    assert admission.task_evidence(task())["eligible"] is True
    assert admission.task_evidence(task(eligible=False))["eligible"] is False
    p = task()
    p["cem"]["cost"] = p["gn16"]["cost"]
    assert admission.task_evidence(p)["sampling_escape"] is False


def test_admission_keeps_all_tasks_and_stable_seed_ties():
    result = admission.select(list(reversed(pool())))
    assert result["selected_seeds"] == list(range(8))
    assert result["eligible_count"] == 32
    assert result["task_retention"] == list(admission.TASKS)
    assert result["P1_admission"] is False and result["physical_future_labels_read"] is False


def test_insufficient_eligible_pool_fails_closed_without_easy_root_filling():
    rows = pool()
    for row in rows[7:]:
        for name in ("shoulder_only","wrist_only"): row["tasks"][name] = task(eligible=False)
    result = admission.select(rows)
    assert result["status"] == "insufficient_model_eligible_roots"
    assert result["eligible_count"] == 7 and result["selected_seeds"] == []


def test_full_drive_only_escape_cannot_change_restricted_task_admission():
    rows = pool()
    for row in rows:
        for name in ("shoulder_only","wrist_only"): row["tasks"][name] = task(eligible=False)
    assert admission.select(rows)["eligible_count"] == 0


def test_actual_outcome_fields_cannot_select_roots():
    rows = pool()
    expected = admission.select(rows)
    for row in rows:
        row["observed_terminal_cost"] = -1e99*row["seed"]
        row["physical_future"] = object()
    assert admission.select(rows) == expected


def test_zero_goal_is_ineligible_without_division_by_zero():
    p = {key:{"cost":0.} for key in task()}
    result = admission.task_evidence(p)
    assert result["eligible"] is False and result["normalized_escape"] == 0.


@pytest.mark.parametrize("defect",["partial","duplicate_seed","duplicate_episode","task_removed","bool_seed","nan","negative","lost_incumbent"])
def test_bad_cohort_or_model_receipt_rejected(defect):
    rows = pool()
    if defect == "partial": rows.pop()
    elif defect == "duplicate_seed": rows[0]["seed"] = rows[1]["seed"]
    elif defect == "duplicate_episode": rows[0]["source_episode"] = rows[1]["source_episode"]
    elif defect == "task_removed": rows[0]["tasks"].pop("wrist_only")
    elif defect == "bool_seed": rows[0]["seed"] = True
    elif defect == "nan": rows[0]["tasks"]["shoulder_only"]["gn16"]["cost"] = np.nan
    elif defect == "negative": rows[0]["tasks"]["shoulder_only"]["gn16"]["cost"] = -1.
    else: rows[0]["tasks"]["shoulder_only"]["cem"]["cost"] = .7
    with pytest.raises(ValueError): admission.select(rows)


def test_metadata_pool_admitted_before_model_values_and_records_collisions():
    lengths = np.full(100,100,dtype=np.int64)
    offsets = np.arange(100,dtype=np.int64)*100
    result = admission.metadata_pool(lengths,offsets,{0,1,2,3,4},seed=202656000)
    assert len(result["roots"]) == 32
    episodes = [r["source_episode"] for r in result["roots"]]
    assert len(set(episodes)) == 32 and not set(episodes)&{0,1,2,3,4}
    assert result["source_values_or_outcomes_used"] is False
    for row in result["roots"]:
        rng = np.random.default_rng(row["seed"] ^ 0x5A7E)
        assert int(rng.integers(100)) == row["source_episode"]
        assert int(offsets[row["source_episode"]])+int(rng.integers(81)) == row["source_start_row"]


def test_fixed_metadata_scan_does_not_replace_model_failures_or_expand_unboundedly():
    with pytest.raises(ValueError):
        admission.metadata_pool(np.array([100]),np.array([0]),set(),seed=202656000)
