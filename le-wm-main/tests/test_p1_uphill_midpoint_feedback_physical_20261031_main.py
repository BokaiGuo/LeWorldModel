import copy
import json

import pytest

from experiments.spectral_reachability import p1_uphill_midpoint_feedback_physical_20261031_main as p


def test_all_source_bound_and_physics_not_admitted():
    result = p.build()
    assert len(result["roster"]) == 24
    assert result["P1_admission"] is False


@pytest.mark.parametrize("field", ["first_packet_hex", "midpoint_pixel_sha256", "role"])
def test_rejects_identity_tampering(field):
    old = json.loads(p.OLD.read_bytes())
    plan = json.loads(p.PLAN.read_bytes())
    inputs, _ = p.planner.inputs()
    plan = copy.deepcopy(plan)
    plan["rows"][0][field] = "bad"
    with pytest.raises(Exception):
        p.assemble(old, plan, inputs)
