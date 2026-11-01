import copy

import pytest

from experiments.spectral_reachability import p1_journal_collision_supplement_20261101_hypatia as mod


def candidate(seed=None, state=None):
    return {"schema": mod.prior.NAME + ".candidates.v1", "candidates": [
        {"candidate_id": "x", "root_seed": None, "reset_seed": seed, "state7": state}]}


def inventory():
    state = [1., 2., 3., 4., 0., 0., 0.]
    return {"history_complete": False, "freshness_certified": False,
            "events": [{"seed": i, "state": state, "operation": "random_reset"} for i in range(16)]}


def test_seed_and_state_hits():
    rows = mod.supplement(candidate(3, [1., 2., 3., 4., 0., 0., 0.]), inventory())
    kinds = {x["kind"] for x in rows[0]["journal_hits"]}
    assert kinds == {"recorded_reset_seed", "exact_state7_bits", "pose_numeric_alias"}


def test_no_hit_does_not_assert_independence():
    assert mod.supplement(candidate(100), inventory())[0]["journal_hits"] == []


def test_reject_claimed_authority_and_outcome_fields():
    inv = inventory()
    inv["history_complete"] = True
    with pytest.raises(Exception):
        mod.supplement(candidate(3), inv)
    bad = candidate(3)
    bad["candidates"][0]["score"] = 1
    with pytest.raises(Exception):
        mod.supplement(bad, inventory())


def test_inventory_count_cannot_silently_shrink():
    inv = copy.deepcopy(inventory())
    inv["events"].pop()
    with pytest.raises(Exception):
        mod.supplement(candidate(3), inv)


def test_real_inventory_hash_and_no_authority():
    inv = mod.base.decode(mod.base.read_bound(mod.INVENTORY, mod.INVENTORY_SHA))
    rows = mod.supplement(candidate(202781010), inv)
    assert any(h["reference"] == "/events/0" for h in rows[0]["journal_hits"])
    assert inv["history_complete"] is False
