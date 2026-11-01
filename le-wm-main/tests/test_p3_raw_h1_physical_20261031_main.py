import copy
import json

import pytest

from experiments.spectral_reachability import p3_raw_h1_physical_20261031_main as m


def test_real_sealed_roster_binding_and_rejects_drift():
    roster = m.source.load_roster()
    result = json.loads(m.NONLINEAR.read_bytes())
    physical, _ = m.host.inputs()
    plans = m.build_plans(roster, result, physical)
    assert len(plans) == 72
    for old, new in zip(plans[::2], plans[1::2], strict=True):
        assert old["commands"][:5] == new["commands"][:5]
        assert old["variant"] == "old" and new["variant"] == "new"
        assert old["root_sha256"] == new["root_sha256"]
    bad = copy.deepcopy(result)
    bad["rows"][0]["old_action"][0][0] += 0.1
    with pytest.raises(Exception, match="old action binding"):
        m.build_plans(roster, bad, physical)
    bad = copy.deepcopy(result)
    bad["rows"][0]["root"] = 1
    with pytest.raises(Exception, match="model row identity"):
        m.build_plans(roster, bad, physical)
