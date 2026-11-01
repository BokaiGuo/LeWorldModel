import pytest
from experiments.spectral_reachability import codex_p2_h3_zero_control_20261030_v1 as m


def test_four_actual_roots_fixed_zero():
    b=m.build()
    assert [p["root_index"] for p in b["plans"]] == [0,1,2,3]
    assert all(p["commands"] == [[0.,0.]]*15 and p["method"] == "fixed_zero" for p in b["plans"])
    assert b["preregistered"] is False and b["budgets"]["selected"] == 60


def test_meter_refuses_16th_step():
    class Env:
        def step(self,a): return None
    meter=m.Meter(Env(),lambda e:None)
    meter.bind(dict(root_sha256="r",method="fixed_zero",selection_sha256="s",planned_commands=15))
    meter.phase="selected"
    for _ in range(15): meter.step([0.,0.])
    with pytest.raises(ValueError): meter.step([0.,0.])
