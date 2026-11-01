import copy

import numpy as np
import pytest

from experiments.spectral_reachability import p1_cp5_cem_baseline_20261101_leibniz as cem
from experiments.spectral_reachability import p1_cp5_cem_physical_20261101_leibniz as physical


def fixture():
    events = []
    result = cem.optimize(lambda raw: float(np.square(raw - .1).mean()),
                          lambda kind, payload: events.append(dict(kind=kind, payload=payload)))
    return result, events


def test_selection_replay_and_tampering():
    result, events = fixture()
    physical.verify_selection(result, events)
    modified = copy.deepcopy(result)
    modified["selected_raw"][0][0][0] += .1
    with pytest.raises(ValueError, match="selection"):
        physical.verify_selection(modified, events)
    modified = copy.deepcopy(events)
    modified[0]["payload"]["raw"][1][0][0][0] += .1
    with pytest.raises(ValueError, match="trace"):
        physical.verify_selection(result, modified)


def test_bounded_action_rejects_nonfinite():
    with pytest.raises(ValueError):
        physical.family.actions([[float("nan"), 0]] * 20, 20)
