import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/codex_paired_physical_interface_20261030.py"
spec = importlib.util.spec_from_file_location("pair_interface", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
H = "a" * 64
J = "b" * 64
LABEL = {"state7": [0.] * 7, "reward": -1.}


def inputs():
    identity = {k: H for k in m.IDENTITY}
    ids = {a: copy.deepcopy(identity) for a in m.ARMS}
    frames = {a: [H, H, H] for a in m.ARMS}
    acts = {a: [[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]] for a in m.ARMS}
    caps = {a: {k: 100 for k in m.AXES} for a in m.ARMS}
    return ids, frames, acts, caps


@pytest.fixture
def runner():
    ids, frames, acts, caps = inputs()
    return m.PairedProtocol(ids, frames, acts, 4, 2, caps, m.digest)


def request(p, arm):
    s = p.state(arm)
    return p.observe(arm, s["step"], copy.deepcopy(s["frames"]), copy.deepcopy(s["actions"]))


def selection(p, arm):
    return p.select(arm, request(p, arm), [[0.1, 0.2], [0.3, 0.4]], H)


def step(p, arm, token, rgb=H, **kw):
    event = p.begin_step(arm, token, True)
    p.finish_step(arm, event, rgb, LABEL, **kw)


def test_complete_two_arms_and_fixed_denominator(runner):
    for arm in m.ARMS:
        for _ in range(2):
            token = selection(runner, arm)
            step(runner, arm, token)
            step(runner, arm, token, J)
    out = runner.snapshot()
    assert out["fixed_time_denominator"] == 10
    assert all(r["status"] == "complete" and r["resources_attempted"]["physical_steps"] == 4 for r in out["rows"])
    assert out["equal_total_compute"] is False and out["flops"] is None
    assert out["actual_execution_verified"] is out["dispatch_authorized"] is False


@pytest.mark.parametrize("field", m.IDENTITY)
def test_identity_mismatch(field):
    ids, frames, acts, caps = inputs()
    ids[m.ARMS[1]][field] = J
    with pytest.raises(ValueError, match="identity mismatch"):
        m.PairedProtocol(ids, frames, acts, 4, 2, caps, m.digest)


def test_observation_uses_own_feedback(runner):
    arm, other = m.ARMS
    token = selection(runner, arm)
    step(runner, arm, token)
    step(runner, arm, token, J)
    s = runner.state(arm)
    with pytest.raises(ValueError, match="own observed"):
        runner.observe(arm, 2, runner.state(other)["frames"], s["actions"])
    req = request(runner, arm)
    assert req["frames"][-1] == J and "physical_label" not in req and "reward" not in req


def test_wrong_time_guard_and_selection(runner):
    arm = m.ARMS[0]
    with pytest.raises(ValueError, match="boundary"):
        runner.observe(arm, 2, [H]*3, runner.state(arm)["actions"])
    token = selection(runner, arm)
    with pytest.raises(ValueError, match="guard"):
        runner.begin_step(arm, token, False)
    with pytest.raises(ValueError, match="selection"):
        runner.begin_step(arm, J, True)
    assert runner.state(arm)["attempted"]["physical_steps"] == 0


def test_intent_requires_ack_before_attempt(runner):
    arm = m.ARMS[0]
    token = selection(runner, arm)
    runner.persist = lambda _: "bad"
    with pytest.raises(ValueError, match="acknowledgement"):
        runner.begin_step(arm, token, True)
    assert runner.state(arm)["phase"] == "stopped"
    assert runner.state(arm)["attempted"]["physical_steps"] == 0


def test_selection_ack_failure_closes_arm(runner):
    arm = m.ARMS[0]
    req = request(runner, arm)
    runner.persist = lambda _: "bad"
    with pytest.raises(ValueError, match="acknowledgement"):
        runner.select(arm, req, [[0, 0], [0, 0]], H)
    assert runner.state(arm)["phase"] == "stopped"


def test_early_stop_no_padding(runner):
    arm = m.ARMS[0]
    token = selection(runner, arm)
    step(runner, arm, token, J, terminated=True)
    r = runner.snapshot()["rows"][0]
    assert r["terminal_reason"] == {"terminated": True, "truncated": False}
    assert r["checkpoints"]["1"]["pixel_sha256"] == J and r["checkpoints"]["2"] is None
    with pytest.raises(ValueError):
        runner.begin_step(arm, token, True)
    assert runner.snapshot()["rows"][1]["status"] == "observe"


def test_inflight_step_is_not_completed(runner):
    arm = m.ARMS[0]
    token = selection(runner, arm)
    runner.begin_step(arm, token, True)
    runner.stop(arm, "step_exception_unknown_internal_work")
    r = runner.snapshot()["rows"][0]
    assert r["physical_step_pending"] and r["resources_attempted"]["physical_steps"] == 1
    assert r["resources_completed_lower_bound"]["physical_steps"] == 0 and r["checkpoints"]["1"] is None


def test_resource_cap_pending_and_unknown(runner):
    arm = m.ARMS[0]
    runner.record_resource_attempt(arm, "predict1", {"predict_calls": 2, "predict_samples": 20})
    with pytest.raises(ValueError, match="cap"):
        runner.record_resource_attempt(arm, "too_much", {"predict_calls": 99})
    req = request(runner, arm)
    with pytest.raises(ValueError, match="pending"):
        runner.select(arm, req, [[0, 0], [0, 0]], H)
    runner.record_resource_return(arm, "predict1")
    runner.select(arm, req, [[0, 0], [0, 0]], H)
    assert runner.snapshot()["rows"][0]["resources_completed_lower_bound"]["predict_samples"] == 20


def test_budget_blocks_before_intent():
    ids, frames, actions, caps = inputs()
    caps[m.ARMS[0]]["physical_steps"] = 0
    events = []
    def persist(event):
        events.append(event)
        return m.digest(event)
    p = m.PairedProtocol(ids, frames, actions, 4, 2, caps, persist)
    token = selection(p, m.ARMS[0])
    with pytest.raises(ValueError, match="cap"):
        p.begin_step(m.ARMS[0], token, True)
    assert len(events) == 1 and p.state(m.ARMS[0])["attempted"]["physical_steps"] == 0


@pytest.mark.parametrize("commands", [[[float("nan"), 0], [0, 0]], [[2, 0], [0, 0]], [[0, 0]]])
def test_invalid_commands(runner, commands):
    arm = m.ARMS[0]
    with pytest.raises(ValueError):
        runner.select(arm, request(runner, arm), commands, H)


def test_return_identity_and_one_shot(runner):
    arm = m.ARMS[0]
    token = selection(runner, arm)
    event = runner.begin_step(arm, token, True)
    bad = dict(event, step=2)
    with pytest.raises(ValueError, match="return mismatch"):
        runner.finish_step(arm, bad, H, LABEL)
    runner.finish_step(arm, event, H, LABEL)
    with pytest.raises(ValueError):
        runner.finish_step(arm, event, H, LABEL)


def test_invalid_physical_label_retains_pending(runner):
    arm = m.ARMS[0]
    token = selection(runner, arm)
    event = runner.begin_step(arm, token, True)
    with pytest.raises(ValueError, match="physical label"):
        runner.finish_step(arm, event, H, {"state7": [0.] * 6, "reward": 0.})
    assert runner.snapshot()["rows"][0]["physical_step_pending"]


def test_selection_before_step_and_unequal_costs():
    ids, frames, actions, caps = inputs()
    events = []
    def persist(value):
        events.append(copy.deepcopy(value))
        return m.digest(value)
    p = m.PairedProtocol(ids, frames, actions, 4, 2, caps, persist)
    p.record_resource_attempt(m.ARMS[0], "plan0", {"predict_calls": 3})
    p.record_resource_return(m.ARMS[0], "plan0")
    p.record_resource_attempt(m.ARMS[1], "plan1", {"predict_calls": 7})
    p.record_resource_return(m.ARMS[1], "plan1")
    token = selection(p, m.ARMS[0])
    p.begin_step(m.ARMS[0], token, True)
    assert "request" in events[0] and events[1]["selection_sha256"] == m.digest(events[0])
    result = p.snapshot()
    assert [r["resources_attempted"]["predict_calls"] for r in result["rows"]] == [3, 7]
    assert result["equal_total_compute"] is False
