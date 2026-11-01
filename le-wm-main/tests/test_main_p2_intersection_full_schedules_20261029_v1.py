"""MAIN independent complete schedule accounting; analytic RAM, no neural model."""
from copy import deepcopy
import hashlib

import pytest

from experiments.spectral_reachability import agent_p2_common_resource_intersection_20261029_v1 as m


def predict(ledger):
    def dispatch():
        for module in ("AE", "predictor", "lowproj"):
            ledger.module(
                module, lambda: b"RAM fixture only",
                lambda value: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest()),
                shapes=deepcopy(m.MODULE_SHAPES[module]),
            )
    ledger.call("predict", dispatch)


def execute(ledger, label):
    assert ledger.admit(label)
    if label == "replay":
        for part in ("whole", "segmented"):
            ledger.call("replay", lambda: [predict(ledger) for _ in range(3)], part=part)
    elif ledger.arm == "grasp":
        for _ in range(3):
            predict(ledger)
        ledger.call("score", lambda: None, trajectories=1,
                    scope="lifted_loss" if label.startswith("joint:") else "serial_terminal")
        ledger.call("backward", lambda: None)
        ledger.auxiliary("Adam_step", lambda: None)
        if label.startswith("joint:"):
            ledger.auxiliary("virtual_noise", lambda: None)
    else:
        horizon = 2 if label.startswith("H2:") else 1 if label.startswith("H1:") else 3
        has_jacobian = label == "bank" or label.startswith(("joint:", "H2:", "H1:"))
        count = (21 if label.startswith("joint:") else 15 if label.startswith("H2:")
                 else 7 if label.startswith("H1:") else 3)
        if has_jacobian:
            ledger.call("J", lambda: [predict(ledger) for _ in range(horizon)],
                        horizon=horizon, dimension=10*horizon)
        for _ in range(count - (horizon if has_jacobian else 0)):
            predict(ledger)
        if label == "bank":
            ledger.auxiliary("root_SVD", lambda: None)
            ledger.auxiliary("root_SVD", lambda: None)
            ledger.auxiliary("root_ridge", lambda: None)
        else:
            ledger.call("score", lambda: None, trajectories=1, scope="root_full_path")
    assert ledger.commit()


@pytest.mark.parametrize("root,arm", m.ORDER)
def test_complete_each_of_six_schedules_without_padding_or_refund(root, arm):
    ledger = m.Ledger(root, arm, clock=lambda: 0., sink=lambda event: None)
    for label in m.schedule(arm):
        execute(ledger, label)
    expected = {"joint": (1902, 255, 85, 0), "composition": (1062, 129, 85, 0),
                "grasp": (1902, 1896, 0, 632)}[arm]
    assert tuple(ledger.used[k] for k in ("predict_API", "AD_primal", "J", "backward")) == expected
    assert ledger.used == m.add(*(m.envelope(arm, label) for label in m.schedule(arm))) | {
        # Score envelopes are caps, not obligations; real RAM callbacks issue
        # exactly one score per non-bank, non-replay source unit.
        "candidate_cost_calls": 632 if arm == "grasp" else 127,
        "candidate_trajectories": 632 if arm == "grasp" else 127,
    }
    assert ledger.committed == list(m.schedule(arm))
    assert ledger.used["replay_trajectories"] == 2
    assert ledger.used["mean_paid_replays"] == 0
    assert ledger.used["FLOPs"] is None
    assert ledger.summary()["stop"] == "natural_completion"
    assert ledger.unit is None and not ledger.stack
    assert all(x["status"] == "returned" for x in ledger.operations)
    assert ledger.side["Adam_step"] == (632 if arm == "grasp" else 0)
    assert ledger.side["virtual_noise"] == (532 if arm == "grasp" else 0)
    assert ledger.side["Jacobian_elements_requested"] == {"joint": 489600, "composition": 247680, "grasp": 0}[arm]
    assert ledger.summary()["actual_cost_equality"] is False
