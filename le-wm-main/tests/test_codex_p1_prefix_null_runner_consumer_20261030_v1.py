import copy
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_prefix_null_runner_20261030_v1 as runner
from experiments.spectral_reachability import codex_p1_prefix_null_saved_consumer_20261030_v1 as consumer


@pytest.fixture(scope="module")
def sources():
    return runner.material()


def test_real_source_closure_and_pair_bytes(sources):
    prep, data = sources
    assert len(prep["plans"]) == 16 and len(data["roots"]) == 2
    assert prep["physical_authorized"] is False and prep["P1_admission"] is False
    for i in range(8):
        a, b = prep["plans"][2*i:2*i+2]
        assert a["role"] == "selected_prefix" and b["role"] == "zero_prefix"
        assert bytes.fromhex(b["raw15_float32_hex"])[:40] == bytes(40)
        assert a["raw15_float32_hex"][80:] == b["raw15_float32_hex"][80:]
        assert a["task_id"] == b["task_id"]


def test_authorization_rejects_before_io(monkeypatch):
    monkeypatch.setattr(runner, "verify", lambda *_: pytest.fail("unexpected IO"))
    with pytest.raises(ValueError, match="authorization"):
        runner.execute(None, None, None, "wrong")


def test_native_dispatch_wrapper_is_bound(monkeypatch, sources):
    prep, data = sources
    prep = copy.deepcopy(prep)
    prep["deployment_status"] = "certified_for_known_root_exploration"
    called = {}
    monkeypatch.setattr(runner, "verify", lambda *args: (prep, data))
    monkeypatch.setattr(runner.base.native, "constructor_contract", lambda: None)
    monkeypatch.setattr(runner, "Store", lambda path: ("exclusive_store", path))
    def execute(p, d, store, factory, path, digest, authorization):
        called.update(p=p, d=d, store=store, authorization=authorization)
        assert callable(factory)
        return {"status": "fixture_only"}
    monkeypatch.setattr(runner.base, "execute", execute)
    assert runner.execute("/prepare", "a" * 64, "/output", runner.AUTH)["status"] == "fixture_only"
    assert called["p"] is prep and called["d"] is data
    assert called["authorization"] == runner.base.AUTH


def test_pinned_roster_blocks_deployment_before_constructor(monkeypatch, sources):
    monkeypatch.setattr(runner, "verify", lambda *_: sources)
    monkeypatch.setattr(runner.base.native, "constructor_contract",
                        lambda: pytest.fail("constructor boundary reached"))
    with pytest.raises(ValueError, match="deployment-blocked"):
        runner.execute("/prepare", "a" * 64, "/output", runner.AUTH)


def make_account_fixture(prep):
    rows = [dict(index=i, plan=copy.deepcopy(p), status="not_started") for i, p in enumerate(prep["plans"])]
    intent = {"index": 0, "resource": "reset", "branch": 0, "status": "pending"}
    failed = {**intent, "status": "failed", "error": {"message": "out of bounds"}}
    counts = {k: {"attempted": 0, "completed": 0} for k in consumer.AXES}
    counts["reset"]["attempted"] = 1
    report = {"rows": rows, "fixed_rows": 16, "receipts": [failed], "counts": counts}
    journal = [{"kind": "operation_dispatch_intent", "receipt": intent},
               {"kind": "operation_failed", "receipt": failed}]
    return report, journal


def test_failed_attempt_retained_and_cannot_disappear(sources):
    prep, _ = sources
    report, journal = make_account_fixture(prep)
    assert consumer.account(report, journal, prep)["reset"] == {"attempted": 1, "completed": 0}
    assert len(report["rows"]) == 16
    with pytest.raises(ValueError, match="unresolved"):
        consumer.account(report, journal[:1], prep)
    report["counts"]["reset"]["attempted"] = 0
    with pytest.raises(ValueError, match="counts"):
        consumer.account(report, journal, prep)


@pytest.mark.parametrize("mutation", ["different_receipt", "duplicate", "missing_row", "bad_plan", "orphan"])
def test_account_rejects_invalid_evidence(sources, mutation):
    prep, _ = sources
    report, journal = make_account_fixture(prep)
    if mutation == "different_receipt":
        journal[1]["receipt"]["branch"] = 1
    elif mutation == "duplicate":
        journal.append(copy.deepcopy(journal[1]))
    elif mutation == "missing_row":
        report["rows"].pop()
    elif mutation == "bad_plan":
        report["rows"][2]["plan"]["raw15_float32_hex"] = "00"
    else:
        journal[1]["receipt"]["index"] = 1
    with pytest.raises(ValueError):
        consumer.account(report, journal, prep)


def test_sequence_censors_stop_and_out_of_bounds_without_padding():
    sequence = runner.base.flow.sequence
    dispatched = []
    def dispatch(step, action):
        dispatched.append(step)
        return {"terminated": step == 3, "truncated": False}
    result = sequence(np.zeros((15, 2), np.float32), dispatch, lambda step, _: step, lambda *_: True)
    assert dispatched == [1, 2, 3]
    assert result["status"] == "native_stopped"
    assert result["checkpoints"][5] is result["checkpoints"][10] is result["checkpoints"][15] is None
    dispatched.clear()
    def capture(step, _):
        if step == 4:
            raise ValueError("canvas guard")
        return step
    with pytest.raises(ValueError, match="canvas guard"):
        sequence(np.zeros((15, 2), np.float32),
                 lambda step, action: (dispatched.append(step) or
                                       {"terminated": False, "truncated": False}),
                 capture, lambda *_: True)
    assert dispatched == [1, 2, 3, 4]


def test_runner_keeps_out_of_bounds_failure_and_remaining_rows(monkeypatch, tmp_path, sources):
    prep, data = sources
    class Store:
        def __init__(self):
            self.previous = None
            self.events = []
            self.path = tmp_path
            (tmp_path / "journal.jsonl").write_text("fixture")
        def check(self, *args):
            return None
        def put(self, *args, **kwargs):
            return None
        def emit(self, kind, **fields):
            self.events.append((kind, copy.deepcopy(fields)))
            self.previous = "a" * 64
    monkeypatch.setattr(runner.base.native, "validate_env", lambda *_: None)
    monkeypatch.setattr(runner.base.native, "native_hooks", lambda *_: nullcontext())
    def episode(env, root, source_plan, plan, row, meter, store):
        meter.call("public_step", lambda: None, phase="plan", step=1)
        raise ValueError("canvas guard out of bounds")
    monkeypatch.setattr(runner.base.episode, "episode", episode)
    def factory():
        return SimpleNamespace(unwrapped=SimpleNamespace(render_action=False, render_size=224,
                                                          render_mode="rgb_array"), close=lambda: None)
    report = runner.base.execute(prep, data, Store(), factory, "/prepare", "a" * 64,
                                 authorization=runner.base.AUTH)
    assert report["status"] == "partial_or_failed"
    assert report["rows"][0]["status"] == "engineering_failed"
    assert "canvas guard" in report["rows"][0]["first_error"]["message"]
    assert all(row["status"] == "not_started" for row in report["rows"][1:])
    assert all(row["missing_checkpoints"] == [0, 5, 10, 15] for row in report["rows"])
    assert report["counts"]["public_step"] == {"attempted": 1, "completed": 1}


def test_source_drift_blocks_before_execution(monkeypatch):
    monkeypatch.setattr(runner.roster, "verify", lambda *_: (_ for _ in ()).throw(ValueError("drift")))
    with pytest.raises(ValueError, match="drift"):
        runner.material()


def test_checkpoint_evidence_on_existing_saved_trajectory(sources, tmp_path):
    _, data = sources
    directory = runner.roster.ROOT / "results/p1_backtrack_physical_runner_20261030_MAIN02"
    report = runner.base.native.read(directory / "report.json", cap=64 * 1024**2)
    events = consumer.previous.events(directory / "journal.jsonl", report["journal_chain_tail"])
    row = report["rows"][0]
    branch = [e for e in events if e.get("branch") == 0]
    checkpoints = consumer.observed_checkpoints(row, branch, data["roots"][row["root_slot"]], directory)
    assert [c["step"] for c in checkpoints] == [0, 5, 10, 15]
    assert len(checkpoints[0]["pusher_velocity"]) == 2
    altered = copy.deepcopy(row)
    altered["checkpoints"][1]["state"][0] += 1
    with pytest.raises(ValueError, match="checkpoint"):
        consumer.observed_checkpoints(altered, branch, data["roots"][row["root_slot"]], directory)
    with pytest.raises(ValueError, match="cp5 guard"):
        consumer.observed_checkpoints(row, branch, data["roots"][row["root_slot"]], tmp_path)
    missing = [e for e in branch if not (e["kind"] == "fresh_observation" and
                                         e["receipt"]["phase"] == "plan" and
                                         e["receipt"]["step"] == 2 and
                                         e["receipt"]["overlay_index"] == 1)]
    with pytest.raises(ValueError, match="triplet"):
        consumer.observed_checkpoints(row, missing, data["roots"][row["root_slot"]], directory)


def test_saved_checker_on_historical_format_only(sources):
    """Exercise saved parser; the old run is NOT a prefix-null experiment."""
    prepared, data = sources
    directory = runner.roster.ROOT / "results/p1_backtrack_physical_runner_20261030_MAIN02"
    path = directory / "report.json"
    assert runner.base.sha(path) == "26802255a8192e95ae3d67a089ee673ba6e8a7331786481fefde9f4fd1e84de9"
    report = runner.base.native.read(path, cap=64 * 1024**2)
    old_prepare = runner.base.native.read(report["execution_binding"]["prepared"], cap=32 * 1024**2)
    old_prepare["task_rows"] = prepared["task_rows"]
    result = consumer.check(directory, report, old_prepare, data)
    assert result["status"] == "saved_checks_passed_exploratory_only"
    assert len(result["checkpoints"]) == 16 and not result["P1_admission"]
    changed = copy.deepcopy(report)
    changed["rows"][0]["goal"][0] += 1
    with pytest.raises(ValueError):
        consumer.check(directory, changed, old_prepare, data)
