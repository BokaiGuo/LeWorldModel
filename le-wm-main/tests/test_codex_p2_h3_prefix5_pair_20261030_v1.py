from copy import deepcopy
import pytest

from experiments.spectral_reachability import codex_p2_h3_prefix5_pair_20261030_v1 as m


def test_actual_saved_pair_and_fail_closed():
    result = m.compare()
    assert len(result["roots"]) == 4
    assert not result["P2_admission"] and not result["total_resource_matched"]
    import json
    current = json.loads((m.OUT / "report.json").read_text())["rows"]
    old = json.loads((m.PARENT / "report.json").read_text())["rows"]
    plans = m.control.build()["plans"]
    old_plans = m.control.original.build()["plans"]
    corrupted = deepcopy(current)
    corrupted[0]["trace"][0]["state"][0] += 1
    with pytest.raises(ValueError):
        m.pair_rows(corrupted, old, plans, old_plans)
    corrupted = deepcopy(plans)
    corrupted[0]["commands"][6] = [0.1, 0.]
    with pytest.raises(ValueError):
        m.pair_rows(current, old, corrupted, old_plans)
