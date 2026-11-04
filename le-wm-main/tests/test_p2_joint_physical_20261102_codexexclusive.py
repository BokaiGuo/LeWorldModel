import copy
import hashlib
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p2_joint_physical_20261102_codexexclusive as run
from experiments.spectral_reachability import p2_joint_physical_saved_20261102_codexexclusive as saved
from experiments.spectral_reachability import p2_conditional_physical_saved_consumer_20261102_exclusive as old_consumer


def test_real_model_winners_and_common_root_no_physics():
    plan = run.build()
    assert [(x["candidate_index"], x["iteration"]) for x in plan["selected"]] == [(12, 8), (12, 8)]
    assert plan["selected"][1]["method"] == "joint_gauss_newton_not_historical_sra"
    actions = [np.asarray(p["commands"], dtype="<f4").tobytes() for p in plan["plans"]]
    assert len(actions[0]) == len(actions[1]) == 200 and actions[0][:120] == actions[1][:120]
    assert actions[0][120:] != actions[1][120:]
    for p, a, selected in zip(plan["plans"], actions, plan["selected"], strict=True):
        assert hashlib.sha256(a[120:]).hexdigest() == selected["suffix_sha256"]
        assert p["native_root_sha256"] == plan["root_sha256"] and p["task_id"] == plan["task_id"]
    assert plan["equal_total_compute"] is plan["independent_confirmation"] is False


def test_plan_digest_rejects_mutated_actions():
    plan = run.build()
    original = json.loads(run.prior.checked(run.prior.old.PREP, run.prior.old.PREP_SHA))
    run.derived(original, plan)
    plan["plans"][0]["commands"][0][0] += 0.1
    with pytest.raises(ValueError, match="new plan"):
        run.derived(original, plan)


@pytest.mark.parametrize("row,plan,warm", [
    ({"own_returned": 3}, [{"terminated": True, "truncated": False}], []),
    ({"own_returned": 0}, [], [{"terminated": False, "truncated": True}]),
    ({"own_returned": 0, "stop_reason": "original root already task-terminal"}, [], []),
])
def test_censoring_witness(row, plan, warm):
    saved.check_stop(row, plan, warm)


def test_censoring_unproven_is_rejected():
    with pytest.raises(ValueError, match="uncorroborated"):
        saved.check_stop({"own_returned": 3}, [{"terminated": False, "truncated": False}], [])


def historical_fixture():
    """Use archived records as parser fixtures only; never claim these are new execution."""
    directory = old_consumer.RESULT
    report = json.loads((directory / "report.json").read_bytes())
    report["historical_sra_authenticated"] = False
    events = [json.loads(x) for x in (directory / "journal.jsonl").read_bytes().splitlines()]
    events[0].pop("recipe_sha256")
    events[0].update(kind="joint_started", plan_sha256="fixture")
    events[-1]["kind"] = "joint_finished"
    tail = None
    for event in events:
        event["previous_sha256"] = tail
        event["event_sha256"] = run.physical.digest({k: v for k, v in event.items() if k != "event_sha256"})
        tail = event["event_sha256"]
    original = json.loads(run.prior.checked(run.prior.old.PREP, run.prior.old.PREP_SHA))
    p = run.executor._derived(original, run.executor._checked_recipe())
    return directory, report, events, p


def test_saved_consumer_on_archived_parser_fixture():
    directory, report, events, p = historical_fixture()
    result = saved.audit_events(events, report, p, "fixture", directory)
    assert result["png_verified"] == 222
    assert [x["plan_returned"] for x in result["branches"]] == [25, 25]


def test_saved_consumer_rejects_trajectory_or_binding_tamper():
    directory, report, events, p = historical_fixture()
    forged = copy.deepcopy(report)
    forged["rows"][0]["plan"]["commands"][0][0] += 0.1
    with pytest.raises(ValueError, match="branch close"):
        saved.audit_events(events, forged, p, "fixture", directory)
    events[10]["kind"] = "joint_finished"
    with pytest.raises(ValueError, match="event chain"):
        saved.audit_events(events, report, p, "fixture", directory)


def test_authorization_before_execution():
    with pytest.raises(ValueError, match="authorization"):
        run.run_host("irrelevant", "wrong")


def test_native_stop_continues_second_arm_without_padding(monkeypatch):
    from contextlib import nullcontext
    class Meter:
        def __init__(self, store):
            self.receipts, self.secondary = [], []
        def call(self, kind, fn, **kwargs):
            return fn()
    class Env:
        render_action, render_size, render_mode = False, 224, "rgb_array"
        def __init__(self):
            self.unwrapped = self
        def close(self):
            pass
    class Store:
        def __init__(self):
            self.events = []
        def emit(self, name, **payload):
            self.events.append((name, payload))
    rows = [dict(index=i, root_slot=0, condition="off", arm="x", selection_digest=str(i),
                 status="not_started", secondary_errors=[]) for i in range(2)]
    calls = []
    def episode(env, root, source, plan, row, meter, store, prepared):
        calls.append(row["index"])
        if row["index"] == 0:
            raise run.physical.native.allmode.NativeStop("fixture stop")
        row["status"] = "horizon_completed"
    monkeypatch.setattr(run.physical, "Meter", Meter)
    monkeypatch.setattr(run.physical, "rows_for", lambda p: rows)
    monkeypatch.setattr(run.physical, "validate_env", lambda env: None)
    monkeypatch.setattr(run.physical, "native_hooks", lambda *args: nullcontext())
    monkeypatch.setattr(run.physical, "episode", episode)
    store = Store()
    result, _, errors = run.execute_pair({"plans": [{"root_slot": 0}, {"root_slot": 0}]},
                                         {"roots": [None], "source_plan": None}, store, Env, "fixture")
    assert calls == [0, 1] and not errors
    assert result[0]["status"] == "native_stopped" and result[0]["own_returned"] == 0
    assert result[1]["status"] == "horizon_completed"
    assert [x[0] for x in store.events].count("branch_closed") == 2
