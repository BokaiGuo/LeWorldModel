import copy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import reacher_response_svd_predicted_tail_saved_20261102 as audit


@pytest.fixture(scope="module")
def recorded():
    root = audit.physical.ROOT / "results"
    plan = json.loads((root / "reacher_response_svd_predicted_tail_PLAN_20261102_v1.json").read_text())
    events = audit.journal.verify_journal(root / "reacher_response_svd_predicted_tail_RUN_20261102_v2/events.jsonl")
    return plan, events, "324dddcc609af1c26e250b1f20ba379ce214ed8f26e1cd15b16906a0a8f7a930"


def test_recorded_run(recorded):
    result = audit.validate(*recorded)
    assert result["counts"] == {"complete": 4, "stopped": 0, "error": 0}
    assert result["P1_admission"] is False


def test_predicted_and_prior_feedback_tail_actions_nearly_identical(recorded):
    plan, _, _ = recorded
    source = audit.physical.checked.FEEDBACK
    assert audit.checks.sha(source) == audit.physical.checked.FEEDBACK_SHA
    rows = json.loads(source.read_text())["results"]
    differences = []
    for root in plan["roots"]:
        for branch in root["branches"]:
            matches = [row for row in rows if row["seed"] == root["seed"]
                       and row["prefix"] == branch["prefix_index"]]
            assert len(matches) == 1
            difference = np.max(np.abs(np.asarray(branch["raw_actions"][5:], np.float32)
                                       - np.asarray(matches[0]["suffix_raw"], np.float32)))
            differences.append(float(difference))
    np.testing.assert_allclose(differences,
                               [5.9604645e-8, 5.9604645e-8, 1.1920929e-7, 1.1920929e-7],
                               rtol=1e-6, atol=0)


@pytest.mark.parametrize("mutation", ["action", "order", "rgb", "stop", "target"])
def test_mutation_rejected(recorded, mutation):
    plan, events, digest = copy.deepcopy(recorded)
    if mutation == "action":
        events[3]["outcome"]["trajectory"][1]["action"][0] += 0.1
    elif mutation == "order":
        events[2], events[4] = events[4], events[2]
    elif mutation == "rgb":
        events[3]["outcome"]["trajectory"][1]["rgb"]["sha256"] = "0" * 64
    elif mutation == "stop":
        events[3]["outcome"]["trajectory"][1]["terminated"] = True
    else:
        events[3]["outcome"]["trajectory"][1]["target_qpos"][0] += 0.1
    with pytest.raises(ValueError):
        audit.validate(plan, events, digest)
