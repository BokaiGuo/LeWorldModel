import copy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p3_preoutcome_classified_control_20261102_codexexclusive as ctl
from experiments.spectral_reachability import p3_preoutcome_cuda_host_20261102_codexexclusive as host
from experiments.spectral_reachability import p3_preoutcome_saved_consumer_20261102_codexexclusive as saved


class Model:
    def action_encoder(self, x):
        return x

    def predict(self, context, action):
        return context


@pytest.fixture
def produced(tmp_path):
    bank = [[{"index": i, "commands": [root, i]} for i in range(30)] for root in (0, 1)]
    binding = {"protocol": "test", "model": "model", "roots": "roots", "feedback": "feedback"}
    plan = ctl.freeze(20, host.CAPS, bank, binding)
    def rollout(meter, phase):
        for _ in range(2):
            value = meter.action_encoder(np.zeros((1, 2)), phase=phase)
            meter.predict(value, value, phase=phase)
    def signal(arm, root, meter):
        rollout(meter, "signal")
        return float(2-root)
    def score(arm, root, candidate, meter):
        rollout(meter, "search")
        return float(candidate["index"])
    path = tmp_path / "controller.jsonl"
    models = {arm: Model() for arm in ctl.ARMS}
    ctl.run_pair(plan, bank, models, {arm: lambda: None for arm in ctl.ARMS}, signal, score, path)
    return plan, bank, path


def test_end_to_end_saved_controller(produced):
    plan, bank, path = produced
    result = saved.verify_events(saved.read(path), plan, bank)
    assert result["allocation"] == {"signal": [15, 5], "reverse": [5, 15]}
    assert result["attempted"]["signal"]["predict_calls"] == 44
    assert result["attempted"]["reverse"]["predict_samples"] == 44
    assert not result["signal_or_scores_recomputed_from_model"]


@pytest.mark.parametrize("error", ["allocation", "candidate", "resource", "chronology", "winner"])
def test_semantic_tamper_rejected(produced, error):
    plan, bank, path = produced
    events = copy.deepcopy(saved.read(path))
    if error == "allocation":
        next(e for e in events if e["kind"] == "both_allocations_frozen")["allocations"]["signal"].reverse()
    elif error == "candidate":
        next(e for e in events if e["kind"] == "candidate_intent")["candidate_sha256"] = "0" * 64
    elif error == "resource":
        next(e for e in events if e["kind"] == "call_intent" and "predict_samples" in e["charge"])["charge"]["predict_samples"] = 2
    elif error == "chronology":
        i = next(i for i, e in enumerate(events) if e["kind"] == "candidate_intent")
        events[i], events[i+1] = events[i+1], events[i]
    else:
        events[-1]["result"]["winners"]["signal"][0]["index"] = 9
    with pytest.raises(ValueError):
        saved.verify_events(events, plan, bank)


def test_chain_detects_tamper_and_partial(produced, tmp_path):
    _, _, path = produced
    raw = path.read_bytes()
    corrupt = tmp_path / "corrupt"
    corrupt.write_bytes(raw.replace(b"signal_observed", b"signal_observeX", 1))
    with pytest.raises(ValueError, match="journal hash"):
        saved.read(corrupt)
    corrupt.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match="truncated"):
        saved.read(corrupt)


def test_source_plan_and_actual_inputs():
    plan, tasks, bank = host.build()
    assert len(tasks) == len(bank) == 2
    assert plan["schedule"]["bank_sha256"] == ctl.digest(bank)
    assert plan["schedule"]["caps_per_arm"] == host.CAPS
    assert plan["independent_outcome"] is False
    assert plan["schedule"]["binding"]["model"] == plan["host_plan"]["expected_model_hash"]


def test_requires_explicit_cuda_authority(tmp_path):
    with pytest.raises(ValueError, match="authority"):
        host.launch("missing", "bad", tmp_path / "out", "unapproved")
