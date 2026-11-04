import copy

import numpy as np
import pytest

from experiments.spectral_reachability import compare_reacher_q_l2_cp5_20261102 as compare


def sample(monkeypatch):
    monkeypatch.setattr(compare.context, "unpack_rgb_observation", lambda x: np.asarray(x))
    monkeypatch.setattr(compare.context, "rgb_digest", lambda x: "fixture")
    events, results = [], []
    for seed in (20263160, 20263161):
        for prefix in range(13):
            events.append({"kind": "context", "role": "successor", "seed": seed,
                           "arm": f"prefix_{prefix:02d}",
                           "value": {"frames": [[[1]], [[2]], [[3]]],
                                     "goal": [[4]], "history": [[0.0, 0.5]]}})
            results.append({"seed": seed, "prefix": prefix, "initial": 1.0,
                            "cp5": 0.9, "final": 0.8, "suffix_raw": [0.0]})
    return events, results


def test_pair_accepts_matching_and_rejects_missing_arm(monkeypatch):
    events, results = sample(monkeypatch)
    assert len(compare.paired(events, copy.deepcopy(events), results, copy.deepcopy(results))["rows"]) == 26
    with pytest.raises(Exception, match="roster"):
        compare.paired(events, events[:-1], results, results)


@pytest.mark.parametrize("field", ["frames", "goal", "history"])
def test_pair_rejects_changed_context(monkeypatch, field):
    events, results = sample(monkeypatch)
    altered = copy.deepcopy(events)
    if field == "frames":
        altered[0]["value"][field][0][0][0] = 9
    else:
        altered[0]["value"][field][0][0] = 9
    with pytest.raises(Exception, match="context mismatch"):
        compare.paired(events, altered, results, results)


def test_pair_rejects_changed_physical_distance(monkeypatch):
    events, results = sample(monkeypatch)
    changed = copy.deepcopy(results)
    changed[0]["cp5"] += 0.1
    with pytest.raises(Exception, match="physical state mismatch"):
        compare.paired(events, events, results, changed)
