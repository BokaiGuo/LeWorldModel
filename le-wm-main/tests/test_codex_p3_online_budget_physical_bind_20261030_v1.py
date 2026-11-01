import copy
import pytest
from experiments.spectral_reachability import codex_p3_online_budget_physical_bind_20261030_v1 as binding


def fixture():
    slots = [{"root_sha256": str(i), "task_id": str(i)} for i in range(2)]
    refs = [{"root_sha256": str(i)} for i in range(2)]
    winners = [dict(index=0, cost=0., commands=[[0., 0.]]*10) for _ in range(2)]
    rows = [dict(point=p, arm=a, winners=winners, attempted={})
            for p in (20, 30) for a in ("uniform", "signal", "reverse")]
    return {"rows": rows}, slots, refs


def test_all_twelve_bound():
    checked, slots, refs = fixture()
    rows = binding.reconstruct(checked, slots, refs)
    assert len(rows) == 12
    assert all(row["physical_authorized"] is False for row in rows)


@pytest.mark.parametrize("bad", ["missing", "root", "shape"])
def test_mismatch_rejected(bad):
    checked, slots, refs = copy.deepcopy(fixture())
    if bad == "missing":
        checked["rows"].pop()
    elif bad == "root":
        refs[0]["root_sha256"] = "different"
    else:
        checked["rows"][0]["winners"][0]["commands"].pop()
    with pytest.raises((ValueError, IndexError)):
        binding.reconstruct(checked, slots, refs)
