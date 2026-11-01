import copy

import pytest

from experiments.spectral_reachability import p1_uphill_multistart_pairing_20261030_main as p


def inputs():
    return (p.read(p.ANCHORS, p.ANCHORS_SHA), p.read(p.PREPARED, p.PREPARED_SHA),
            p.read(p.physical.PLAN, p.physical.PLAN_SHA), p.read(p.CONSUME, p.CONSUME_SHA))


def test_pairing_preserves_stopped_endpoint():
    anchors, prepared, plan, observed = inputs()
    result = p.join(anchors, prepared, plan, observed)
    assert len(result["rows"]) == 24
    assert result["rows"][1]["last_native_success"]
    assert result["rows"][1]["endpoint"] is None
    assert result["rows"][0]["endpoint_minus_zero_prefix"] is None
    assert not result["P1_admission"]


@pytest.mark.parametrize("where", ["pixel", "role", "winner"])
def test_rejects_wrong_pairing(where):
    anchors, prepared, plan, observed = inputs()
    prepared, plan = copy.deepcopy(prepared), copy.deepcopy(plan)
    if where == "pixel":
        prepared["roster"][0]["cp5_pixel_sha256"] = "wrong"
    elif where == "role":
        plan["rows"][0]["role"] = "wrong"
    else:
        plan["rows"][0]["selected_index"] = (plan["rows"][0]["selected_index"] + 1) % 4
    with pytest.raises(Exception):
        p.join(anchors, prepared, plan, observed)
