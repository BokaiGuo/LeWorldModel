from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import p1_cem_trace_audit_20261101_main as audit


@pytest.fixture
def sealed():
    rng = np.random.default_rng(20261101)
    mean = np.zeros((3, 5, 2), np.float32)
    std = np.full_like(mean, .5)
    events, actions, scores = [], [], []
    for i in range(4):
        bank = np.clip(rng.normal(mean, std, (16, 3, 5, 2)), -1, 1).astype(np.float32)
        bank[0] = mean
        events.append(dict(index=2*i, kind="bank",
                           payload=dict(iteration=i, mean=mean.tolist(), std=std.tolist(),
                                        raw=bank.tolist())))
        values = [float(64 - 16*i - j) for j in range(16)]
        elites = sorted(range(16), key=lambda j: (values[j], j))[:4]
        mean = bank[elites].mean(axis=0, dtype=np.float32)
        std = np.clip(bank[elites].std(axis=0, dtype=np.float32), .05, 1.).astype(np.float32)
        events.append(dict(index=2*i+1, kind="update",
                           payload=dict(iteration=i, scores=values, elite_indices=elites,
                                        mean=mean.tolist(), std=std.tolist())))
        actions.extend(bank.tolist())
        scores.extend(values)
    winner = 63
    events.append(dict(index=8, kind="selected_before_replay",
                       payload=dict(index=winner, score=scores[winner], raw=actions[winner])))
    report = dict(status="complete", failure=None, model_before={"sha": "same"},
                  model_after={"sha": "same"}, source_verified=True,
                  result=dict(selected_index=winner, scores=scores,
                              selected_raw=actions[winner], replay_score=scores[winner]))
    return report, events


def test_sealed_trace(sealed):
    report, events = sealed
    assert audit.audit(report, events)["winner"] == 63


@pytest.mark.parametrize("mutation", ["bank", "elite", "winner", "model"])
def test_tampering_fails_closed(sealed, mutation):
    report, events = deepcopy(sealed)
    if mutation == "bank":
        events[0]["payload"]["raw"][1][0][0][0] += 0.01
    elif mutation == "elite":
        events[1]["payload"]["elite_indices"][0] = 99
    elif mutation == "winner":
        report["result"]["selected_index"] = 0
    else:
        report["model_after"] = {}
    with pytest.raises(ValueError):
        audit.audit(report, events)
