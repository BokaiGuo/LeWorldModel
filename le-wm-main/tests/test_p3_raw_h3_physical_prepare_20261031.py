import copy
import json

import pytest

from experiments.spectral_reachability import p3_raw_h3_physical_prepare_20261031 as module


def test_h3_all_pairs_retain_full_prefix_and_reject_tamper():
    roster = module.h1.source.load_roster()
    source = json.loads(module.h1.NONLINEAR.read_bytes())
    physical, _ = module.h1.host.inputs()
    plans = module.build_plans(roster, source, physical)
    assert len(plans) == 72
    for old, new in zip(plans[::2], plans[1::2], strict=True):
        assert (old["variant"], new["variant"]) == ("old", "new")
        assert len(old["commands"]) == len(new["commands"]) == 20
        assert old["commands"][:15] == new["commands"][:15]
        assert old["prefix_float32_sha256"] == new["prefix_float32_sha256"]
        assert old["root_sha256"] == new["root_sha256"]
    bad = copy.deepcopy(source)
    index = next(i for i, x in enumerate(roster) if x[3] == 3)
    bad["rows"][index]["old_action"][0][0] += 0.1
    with pytest.raises(Exception, match="old action binding"):
        module.build_plans(roster, bad, physical)
