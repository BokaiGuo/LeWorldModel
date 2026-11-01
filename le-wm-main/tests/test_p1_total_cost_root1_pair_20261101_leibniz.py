from copy import deepcopy

import pytest

from experiments.spectral_reachability import p1_total_cost_root1_pair_20261101_leibniz as audit


def compatible():
    goal = [1, 2]
    binding = {"goal_latent_sha256": audit.pair.digest(goal), "root_sha256": "r"}
    task = {"goal": goal, "action_mean": [0, 0], "action_std": [1, 1], "projection": [[1]]}
    protocol = {"common_node_caps": {"predict": 1}, "boundaries": [0, 5, 10, 15],
                "packet_steps": 5, "terminal": 20}
    report = {"cp10_goal_binding": binding, "model_before": {"hash": "m"},
              "model_after": {"hash": "m"}, "protocol": protocol}
    reports = {"structured": deepcopy(report), "cem": deepcopy(report)}
    prepared = {
        "structured": {"cp10_identity": deepcopy(binding), "source": {"tasks": [None, deepcopy(task)]}},
        "cem": {"cem_source_identity": {"rebound_goal_binding": deepcopy(binding)},
                "source": {"tasks": [None, deepcopy(task)]}},
    }
    return reports, prepared


def test_compatible_pair():
    audit.verify_compatibility(*compatible())


@pytest.mark.parametrize("field", ["goal", "model", "caps", "normalizer"])
def test_rejects_pair_mismatch(field):
    reports, prep = compatible()
    if field == "goal":
        reports["cem"]["cp10_goal_binding"]["root_sha256"] = "other"
    elif field == "model":
        reports["cem"]["model_before"]["hash"] = "other"
    elif field == "caps":
        reports["cem"]["protocol"]["common_node_caps"]["predict"] = 2
    else:
        prep["cem"]["source"]["tasks"][1]["action_std"] = [2, 1]
    with pytest.raises(ValueError):
        audit.verify_compatibility(reports, prep)
