import numpy as np
import pytest

from experiments.spectral_reachability import p1_full_tail_cross_intervention_20261101_exclusive as run
from experiments.spectral_reachability import p1_full_tail_cross_verify_20261101_exclusive as verify


def test_tail_deduplication_retains_aliases_without_outcome_filtering():
    zero = np.zeros((10, 2), np.float32)
    one = np.full((10, 2), 0.1, np.float32)
    bank = run.unique_tail_bank([("zero", zero), ("zero_alias", zero), ("one", one)])
    assert [x["label"] for x in bank] == ["zero", "one"]
    assert bank[0]["aliases"] == ["zero", "zero_alias"]


def test_canonical_raw_is_stable_and_rejects_nan():
    assert run.canonical_raw({"b": 1, "a": 2}) == b'{"a":2,"b":1}\n'
    with pytest.raises(ValueError):
        run.canonical_raw({"bad": float("nan")})


def test_schedule_is_complete_cartesian_product():
    prefixes = {role: np.full((5, 2), i / 10, np.float32).tolist()
                for i, role in enumerate(run.ROLES)}
    tails = run.unique_tail_bank([
        ("zero", np.zeros((10, 2), np.float32)),
        ("one", np.full((10, 2), 0.1, np.float32)),
    ])
    rows = run.schedule(0, prefixes, tails)
    assert len(rows) == 6
    assert [(x["prefix_role"], x["suffix_role"]) for x in rows] == [
        (role, tail) for role in run.ROLES for tail in ("zero", "one")
    ]


def test_live_contract_has_fixed_30_branch_roster():
    contract = run.build()
    assert contract["planned_branches"] == 30
    assert [len(root["schedule"]) for root in contract["roots"]] == [18, 12]
    assert [len(root["tails"]) for root in contract["roots"]] == [6, 4]
    assert contract["new_neural_calls"] == 0
    assert not contract["outcomes_used_to_include_or_exclude_tails"]
    assert not contract["independent_confirmation"]
    assert not contract["P1_admission"]
    for root in contract["roots"]:
        prepared = run.json.loads(run.Path(root["source_prepared"]).read_bytes())
        data = run.engine.p3.source_data(prepared["source"])
        rebound = (run.root0_goal.rebind(prepared, data) if root["root_slot"] == 0
                   else run.root1_goal.rebind(prepared, data))
        assert rebound == root["goal_binding"]


def test_analysis_uses_all_nonzero_tails_not_best_tail():
    roots = [{
        "root_slot": 0,
        "root": {"goal": [0.0] * 7},
        "prefixes": {role: np.zeros((5, 2), np.float32).tolist() for role in run.ROLES},
        "tails": [
            {"label": "zero", "sha256": "0"},
            {"label": "a", "sha256": "a"},
            {"label": "b", "sha256": "b"},
        ],
    }]
    rows = []
    for role_index, role in enumerate(run.ROLES):
        for tail_index, tail in enumerate(("zero", "a", "b")):
            value = float(role_index + tail_index)
            rows.append({
                "source_slot": 0,
                "prefix_role": role,
                "suffix_role": tail,
                "prefix_checkpoint": {"state": [0.0] * 7},
                "status": "completed15",
                "final_metrics": {
                    "native_distance": value,
                    "object_position_error": value,
                    "wrapped_object_angle_error": value,
                    "native_success": value == 0,
                },
            })
    result = verify.analyze({"roots": roots}, rows)["0"]["whole_bank_aggregate"]
    assert result["tail_count"] == 2
    assert result["selected_minus_zero_mean"]["native_distance"] == -1.0
