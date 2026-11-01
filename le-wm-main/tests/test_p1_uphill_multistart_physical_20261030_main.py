import copy
import json

import pytest

from experiments.spectral_reachability import p1_uphill_multistart_physical_20261030_main as subject


def inputs():
    return (json.loads(subject.PLAN.read_bytes()), json.loads(subject.OLD.read_bytes()),
            json.loads(subject.planner.bundle.OUT.read_bytes()))


def test_builds_all_fixed_roster():
    prepared = subject.build()
    assert len(prepared["roster"]) == 24
    assert not prepared["P1_admission"]


@pytest.mark.parametrize("field", ["selected_action_sha256", "initial_f32_sha256", "role"])
def test_rejects_plan_identity_tampering(field):
    plan, old, blind = inputs()
    plan = copy.deepcopy(plan)
    plan["rows"][0][field] = "invalid"
    with pytest.raises(Exception):
        subject.assemble(plan, old, blind)


def test_rejects_prefix_image_mismatch():
    plan, old, blind = inputs()
    blind = copy.deepcopy(blind)
    blind["rows"][0]["cp5_pixel_sha256"] = "invalid"
    with pytest.raises(Exception):
        subject.assemble(plan, old, blind)
