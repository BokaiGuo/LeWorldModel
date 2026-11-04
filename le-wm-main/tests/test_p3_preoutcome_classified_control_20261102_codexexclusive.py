import json

import numpy as np
import pytest

from experiments.spectral_reachability import p3_preoutcome_classified_control_20261102_codexexclusive as ctl


class FakeModel:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def predict(self, context, actions):
        self.calls += 1
        if self.fail:
            raise RuntimeError("predict failure")
        return context

    def encode(self, value):
        return value

    def action_encoder(self, value):
        return value


@pytest.fixture
def fixture():
    bank = [[{"index": i, "command": [root, i]} for i in range(20)] for root in (0, 1)]
    caps = {"observation_encode": 0, "action_encode": 0, "predict_calls": 22,
            "predict_samples": 22, "signal_compute": 2, "candidate_evaluate": 20}
    binding = {"protocol": "fixed", "model": "model-sha", "roots": "root-sha", "feedback": "feedback-sha"}
    plan = ctl.freeze(20, caps, bank, binding)
    models = {arm: FakeModel() for arm in ctl.ARMS}
    syncs = {arm: lambda: None for arm in ctl.ARMS}
    return plan, bank, models, syncs


def signal(arm, root, model):
    model.predict(np.zeros((1, 2)), None, phase="signal")
    return [2.0, 1.0][root]


def evaluate(arm, root, candidate, model):
    model.predict(np.zeros((1, 2)), None, phase="search")
    return float(candidate["index"])


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_real_dispatch_counts_and_preoutcome_order(fixture, tmp_path):
    plan, bank, models, syncs = fixture
    path = tmp_path / "journal"
    result = ctl.run_pair(plan, bank, models, syncs, signal, evaluate, path)
    assert result["allocations"] == {"signal": [15, 5], "reverse": [5, 15]}
    for arm in ctl.ARMS:
        assert result["resources"][arm]["attempted"]["predict_samples"] == 22
        assert result["resources"][arm]["returned"]["candidate_evaluate"] == 20
        assert models[arm].calls == 22
    rows = events(path)
    kinds = [x["kind"] for x in rows]
    assert kinds.index("both_allocations_frozen") > max(i for i, x in enumerate(kinds) if x == "signal_observed")
    assert kinds.index("both_allocations_frozen") < kinds.index("candidate_intent")
    assert [x["index"] for x in rows] == list(range(len(rows)))
    for previous, current in zip(rows, rows[1:]):
        assert current["previous_sha256"] == previous["sha256"]
        assert current["sha256"] == ctl.digest({k: v for k, v in current.items() if k != "sha256"})
    assert result["equal_total_compute"] is False


def test_cap_rejects_before_model_dispatch_and_preserves_attempt(fixture, tmp_path):
    plan, bank, models, syncs = fixture
    plan["caps_per_arm"]["predict_calls"] = 0
    path = tmp_path / "journal"
    with pytest.raises(ValueError, match="cap exceeded"):
        ctl.run_pair(plan, bank, models, syncs, signal, evaluate, path)
    assert models["signal"].calls == 0
    rows = events(path)
    assert any(e["kind"] == "cap_rejected" for e in rows)
    failure = rows[-1]
    assert failure["kind"] == "pair_failed"
    assert failure["resources"]["signal"]["attempted"]["signal_compute"] == 1
    assert failure["resources"]["signal"]["returned"]["signal_compute"] == 0


def test_model_failure_no_refund_or_retry(fixture, tmp_path):
    plan, bank, models, syncs = fixture
    models["signal"].fail = True
    path = tmp_path / "journal"
    with pytest.raises(RuntimeError, match="predict failure"):
        ctl.run_pair(plan, bank, models, syncs, signal, evaluate, path)
    r = events(path)[-1]["resources"]["signal"]
    assert r["attempted"]["predict_calls"] == r["attempted"]["predict_samples"] == 1
    assert r["returned"]["predict_calls"] == 0
    assert r["failed"]
    with pytest.raises(FileExistsError):
        ctl.run_pair(plan, bank, models, syncs, signal, evaluate, path)


@pytest.mark.parametrize("problem", ["tie", "mismatch", "no_predict", "phase"])
def test_signal_failures_precede_all_search(fixture, tmp_path, problem):
    plan, bank, models, syncs = fixture
    def bad(arm, root, model):
        if problem == "no_predict":
            return float(root)
        model.predict(np.zeros((1, 2)), None, phase="search" if problem == "phase" else "signal")
        if problem == "tie":
            return 1.0
        return float(root + (1 if arm == "reverse" else 0))
    path = tmp_path / "journal"
    with pytest.raises(ValueError):
        ctl.run_pair(plan, bank, models, syncs, bad, evaluate, path)
    assert "candidate_intent" not in [e["kind"] for e in events(path)]


def test_bank_drift_rejected_before_journal(fixture, tmp_path):
    plan, bank, models, syncs = fixture
    bank[0][0]["command"][0] = 5
    path = tmp_path / "journal"
    with pytest.raises(ValueError, match="bank mismatch"):
        ctl.run_pair(plan, bank, models, syncs, signal, evaluate, path)
    assert not path.exists()


def test_sync_failure_is_attempted_not_returned(fixture, tmp_path):
    plan, bank, models, syncs = fixture
    def fail():
        raise RuntimeError("sync failed")
    syncs["signal"] = fail
    path = tmp_path / "journal"
    with pytest.raises(RuntimeError, match="sync failed"):
        ctl.run_pair(plan, bank, models, syncs, signal, evaluate, path)
    r = events(path)[-1]["resources"]["signal"]
    assert r["attempted"]["predict_calls"] == 1
    assert r["returned"]["predict_calls"] == 0
