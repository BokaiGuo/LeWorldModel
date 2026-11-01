import copy
import json

import pytest

from experiments.spectral_reachability import p3_raw_h3_roster_dependence_20261031 as index


def test_groups_cover_every_fixed_row_and_pair():
    plans = json.loads(index.source.MANIFEST.read_bytes())["plans"]
    result = index.index_roster(plans)
    for key, count in (("execution_groups", 72), ("prefix_groups", 72), ("pair_groups", 36)):
        rows = [i for group in result[key] for i in group["indices"]]
        assert sorted(rows) == list(range(count))
    assert result["unique_root_full_action_executions"] <= 72
    assert result["unique_root_old_new_action_pairs"] <= 36
    assert any(g["multiplicity"] > 1 for g in result["pair_groups"])


def test_rejects_unbound_action_tampering():
    plans = json.loads(index.source.MANIFEST.read_bytes())["plans"]
    bad = copy.deepcopy(plans)
    bad[1]["commands"][16][0] = 0.123
    with pytest.raises(ValueError):
        index.index_roster(bad)
