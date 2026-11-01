import copy
import json

import pytest

from experiments.spectral_reachability import p1_uphill_midpoint_trust_physical_20261031_main as p


def test_trust_radius_binding():
    plan = json.loads(p.PLAN.read_bytes())
    entries = p.prior.planner.inputs()[0]
    p.verify_radius(plan, entries)
    bad = copy.deepcopy(plan)
    bad["rows"][0]["second_packet_hex"] = (b"\x00\x00\x80\x3f" * 10).hex()
    with pytest.raises(Exception, match="trust radius"):
        p.verify_radius(bad, entries)
