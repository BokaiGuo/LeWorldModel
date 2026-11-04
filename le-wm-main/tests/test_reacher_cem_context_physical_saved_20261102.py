import copy
import json

import pytest

from experiments.spectral_reachability import reacher_cem_context_physical_saved_20261102 as audit


@pytest.fixture(scope="module")
def recorded():
    root = audit.experiment.ROOT / "results"
    plan = json.loads((root / "reacher_cem_context_physical_PLAN_20261102_v1.json").read_text())
    events = audit.experiment.run.saved.verify_journal(
        root / "reacher_cem_context_physical_RUN_20261102_v1/events.jsonl")
    return plan, events, "d640b146bb78125eac19b9a940c50cb70fe05cdf30c3a307ff90ba3f2bd81a8f"


def test_complete_roster(recorded):
    result = audit.validate(*recorded)
    assert result["complete_branches"] == 48 and len(result["pairs"]) == 24
    assert result["independent_confirmation"] is False


@pytest.mark.parametrize("mutation", ["step", "stop", "rgb", "order"])
def test_rejects_tampering(recorded, mutation):
    plan, events, digest = copy.deepcopy(recorded)
    if mutation == "step":
        events[3]["action"][0] += 0.1
    elif mutation == "stop":
        events[4]["terminated"] = True
    elif mutation == "rgb":
        events[33]["outcome"]["trajectory"][0]["rgb"]["sha256"] = "0" * 64
    else:
        events[2], events[3] = events[3], events[2]
    with pytest.raises(ValueError):
        audit.validate(plan, events, digest)
