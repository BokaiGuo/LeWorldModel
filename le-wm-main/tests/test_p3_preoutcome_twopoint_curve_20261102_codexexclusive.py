import copy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p3_preoutcome_twopoint_curve_20261102_codexexclusive as m


class Model:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def action_encoder(self, value):
        return value

    def predict(self, value, actions):
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated predictor error")
        return value


@pytest.fixture
def fake():
    bank = [[{"index": i, "commands": [root, i]} for i in range(30)] for root in (0, 1)]
    binding = {"protocol": "test", "model": "model", "roots": "roots", "feedback": "feedback"}
    plan = m.ctl.freeze(30, m.CAP30, bank, binding)
    models = {arm: Model() for arm in m.ctl.ARMS}
    def two(meter, phase):
        for _ in range(2):
            a = meter.action_encoder(np.zeros((1, 2)), phase=phase)
            meter.predict(a, a, phase=phase)
    def signal(arm, root, meter):
        two(meter, "signal")
        return float(2-root)
    def score(arm, root, candidate, meter):
        two(meter, "search")
        return float(candidate["index"])
    return plan, bank, models, signal, score


def test_30_actual_dispatch_and_full_saved_accounting(fake, tmp_path):
    plan, bank, models, signal, score = fake
    path = tmp_path / "journal"
    result = m.ctl.run_pair(plan, bank, models, {a: lambda: None for a in m.ctl.ARMS}, signal, score, path)
    events = m.saved.read(path)
    checked = m.saved.verify_events(events, plan, bank)
    assert checked["allocation"] == {"signal": [22, 8], "reverse": [8, 22]}
    for arm in m.ctl.ARMS:
        assert models[arm].calls == 64
        assert result["resources"][arm]["attempted"]["predict_calls"] == 64
        assert result["resources"][arm]["by_phase"]["search"]["attempted"]["predict_calls"] == 60
        assert result["resources"][arm]["by_phase"]["signal"]["attempted"]["predict_calls"] == 4
    kinds = [x["kind"] for x in events]
    assert kinds.index("both_allocations_frozen") < kinds.index("candidate_intent")


def test_wrong_quota_or_order_rejected(fake, tmp_path):
    plan, bank, models, signal, score = fake
    path = tmp_path / "journal"
    m.ctl.run_pair(plan, bank, models, {a: lambda: None for a in m.ctl.ARMS}, signal, score, path)
    events = m.saved.read(path)
    changed = copy.deepcopy(events)
    next(e for e in changed if e["kind"] == "both_allocations_frozen")["allocations"]["signal"] = [23, 7]
    with pytest.raises(ValueError):
        m.saved.verify_events(changed, plan, bank)
    changed = copy.deepcopy(events)
    next(e for e in changed if e["kind"] == "candidate_intent")["candidate_index"] = 3
    with pytest.raises(ValueError):
        m.saved.verify_events(changed, plan, bank)


def test_failed_calls_not_refunded(fake, tmp_path):
    plan, bank, models, signal, score = fake
    models["signal"].fail = True
    path = tmp_path / "journal"
    with pytest.raises(RuntimeError, match="predictor error"):
        m.ctl.run_pair(plan, bank, models, {a: lambda: None for a in m.ctl.ARMS}, signal, score, path)
    final = m.saved.read(path)[-1]
    assert final["kind"] == "pair_failed"
    assert final["resources"]["signal"]["attempted"]["predict_calls"] == 1
    assert final["resources"]["signal"]["returned"]["predict_calls"] == 0


def test_fixed_actual_assets_and_reused_20():
    plan, tasks, bank = m.build()
    assert plan["points"] == [20, 30] and len(tasks) == len(bank) == 2
    assert plan["schedule20"]["bank_sha256"] == plan["schedule30"]["bank_sha256"]
    assert plan["curve_contract_created_after_point20_outcome"]
    assert not plan["independent_roots"] and not plan["equal_total_compute"]


def test_curve_preserves_missing_work():
    plan, _, _ = m.build()
    checked = m.saved.consume(m.POINT20, m.POINT20_REPORT, m.POINT20_PREP, m.base.verify)
    events = m.saved.read(m.POINT20 / "controller.jsonl")
    curve = m.assemble(plan, checked, events, failure={"reason": "fixture"})
    assert curve["status"] == "partial_curve"
    assert curve["points"][1]["resources"] == "unknown_in_full"
    assert curve["points"][0]["resources"]["signal"]["attempted"]["predict_samples"] == 44


def test_no_authority_or_source_drift(fake, tmp_path):
    with pytest.raises(ValueError, match="authority"):
        m.launch("missing", "bad", "output", "wrong")
    plan, bank, models, signal, score = fake
    bank[0][0]["commands"][0] = -1
    with pytest.raises(ValueError, match="bank mismatch"):
        m.ctl.run_pair(plan, bank, models, {a: lambda: None for a in m.ctl.ARMS},
                       signal, score, tmp_path / "journal")
