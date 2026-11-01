"""CPU-only bridge tests; fake callbacks and fake physics objects only."""
import pytest

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p1_moving_overlay_recompile_physical_20261027_v1 as physical
from experiments.spectral_reachability import p2_durable_hook_driver_20261030_codex as durable
from experiments.spectral_reachability import p2_h15_observable_budget_bridge_20261030_codex as bridge
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec
from test_p2_h15_strict_saved_consumer_20261030_codex import rewrite_chain
from test_p2_h15_cuda_ledger_runner_20261030_codex import setup as runner_setup
from test_p2_h15_neural_ledger_adapter_20261030_codex import contract as task_contract


class Clock:
    def __init__(self):
        self.value = 100
    def __call__(self):
        return self.value


def setup(tmp_path, *, replay_cap=3, physical_cap=3, wall_cap=100):
    contract = {"schema": spec.NAME + ".contract.v1", "arms": ["joint", "other"],
                "scope": "per_root_per_arm_standalone_planning_and_selected_physical_execution",
                "identity": dict.fromkeys(spec.IDENTITY, "a" * 64),
                "axes": {k: dict(unit=u, aggregation="max" if k == "peak_device_bytes" else "sum", cap=1000)
                         for k, u in spec.AXES.items()}}
    for name, cap in (("replay_calls", replay_cap), ("physical_steps", physical_cap),
                      ("planning_wall_ns", wall_cap)):
        contract["axes"][name]["cap"] = cap
    sha, clock = core.digest(contract), Clock()
    b = bridge.Bridge(tmp_path / "sidecar", contract, sha, "joint", "run", clock=clock)
    d = durable.Driver(tmp_path / "ledger", contract, sha, dict.fromkeys(core.AXES, 1000), "joint", "run")
    b.attach(d)
    return b, d, clock, sha


def close_and_verify(b, d, sha):
    ledger = d.finish()
    sidecar = b.finish(ledger)
    checked = bridge.verify(sidecar["journal"], sidecar["sha256"], ledger["journal"], ledger["sha256"],
                            sha, arm="joint", run_id="run")
    return checked, sidecar, ledger


class OriginalMeter:
    def __init__(self):
        self.active = dict(explicit_render=0, goal_bind=0)
        self.calls = []
    def call(self, kind, fn, **kwargs):
        self.calls.append(kind)
        return fn()


class FakeSpace:
    def __init__(self):
        self.calls = 0
    def step(self, dt):
        self.calls += 1
        return dt


class FakeBody:
    def __init__(self):
        self.with_target = False
        self.space = FakeSpace()
    def _set_state(self, value):
        return value
    def _set_goal_state(self, value):
        return value
    def render(self):
        return None


class FakeEnv:
    def __init__(self):
        self.unwrapped = FakeBody()
    def reset(self):
        return self.unwrapped._set_state(1)


def test_actual_existing_hook_abi_counts_fake_space_steps_and_replay(tmp_path):
    b, d, clock, sha = setup(tmp_path)
    def native_replay():
        return d.call({"stage": "fake_neural"}, "predict", lambda: "ok", shape=[1, 3, 192])
    assert b.replay(native_replay) == "ok"
    clock.value += 20
    assert b.finish_planning(lambda: None) == 20
    env, meter, errors = FakeEnv(), OriginalMeter(), []
    with physical.native_hooks(env, b.physical_meter(meter), errors):
        env.reset()
        assert env.unwrapped.space.step(0.1) == 0.1
        assert env.unwrapped.space.step(0.1) == 0.1
    assert not errors and "step" not in vars(env.unwrapped.space)
    assert env.unwrapped.space.calls == 2
    checked, _, _ = close_and_verify(b, d, sha)
    assert checked["status"] == "complete" and checked["planning_wall_ns"] == 20
    assert checked["counts"] == {"replay_calls": {"attempted": 1, "completed": 1},
                                  "physical_steps": {"attempted": 2, "completed": 2}}
    assert not checked["physical_hook_coverage_verified"] and not checked["budget_fairness"]


@pytest.mark.parametrize("axis", ["replay", "physical"])
def test_zero_cap_blocks_actual_callback_before_dispatch(tmp_path, axis):
    b, d, _, sha = setup(tmp_path, replay_cap=0 if axis == "replay" else 3,
                         physical_cap=0 if axis == "physical" else 3)
    called = []
    if axis == "replay":
        with pytest.raises(core.BudgetExceeded):
            b.replay(lambda: called.append(1))
    else:
        b.finish_planning(lambda: None)
        with pytest.raises(core.BudgetExceeded):
            b.physical_meter(OriginalMeter()).call("physics_step", lambda: called.append(1))
    assert called == []
    checked, _, _ = close_and_verify(b, d, sha)
    assert checked["status"] == "failed_or_blocked"
    assert checked["counts"][axis + ("_calls" if axis == "replay" else "_steps")]["attempted"] == 0


def test_wall_pre_and_inflight_cap_do_not_refund(tmp_path):
    b, d, clock, sha = setup(tmp_path, wall_cap=10)
    def slow():
        clock.value += 11
    with pytest.raises(core.BudgetExceeded, match="during replay"):
        b.replay(slow)
    checked, _, _ = close_and_verify(b, d, sha)
    assert checked["status"] == "failed_or_blocked"
    assert checked["counts"]["replay_calls"] == {"attempted": 1, "completed": 0}
    assert checked["planning_wall_ns"] is None


def test_wall_pre_cap_blocks_before_replay(tmp_path):
    b, d, clock, sha = setup(tmp_path, wall_cap=10)
    clock.value += 10
    called = []
    with pytest.raises(core.BudgetExceeded):
        b.replay(lambda: called.append(1))
    assert not called
    assert close_and_verify(b, d, sha)[0]["status"] == "failed_or_blocked"


def test_expired_wall_blocks_completion_callback(tmp_path):
    b, d, clock, sha = setup(tmp_path, wall_cap=10)
    clock.value += 10
    called = []
    with pytest.raises(ValueError, match="before barrier"):
        b.finish_planning(lambda: called.append(1))
    assert called == []
    assert close_and_verify(b, d, sha)[0]["status"] == "failed_or_blocked"


def test_h15_runner_replay_method_nested_in_real_durable_driver(tmp_path, monkeypatch):
    make, report, task, _, _ = runner_setup(tmp_path, monkeypatch)
    contract = task_contract(task)
    clock = Clock()
    b = bridge.Bridge(tmp_path / "sidecar", contract, core.digest(contract), "joint", "stub", clock=clock)
    r = make()
    b.attach(r.driver)
    assert b.replay(lambda: r.replay(report["result"]["bank"]["raw15"][0], index=0)) == \
        report["result"]["candidates"][0]["path"]
    b.finish_planning(lambda: None)
    out = r.finish()  # Incomplete CPU plan is deliberately failed.
    side = b.finish(out)
    check = bridge.verify(side["journal"], side["sha256"], out["journal"], out["sha256"],
                          core.digest(contract), arm="joint", run_id="stub")
    assert check["counts"]["replay_calls"] == {"attempted": 1, "completed": 1}
    assert check["status"] == "failed_or_blocked"


def test_original_meter_refuses_before_physics_not_counted(tmp_path):
    b, d, _, sha = setup(tmp_path)
    b.finish_planning(lambda: None)
    class RefusingMeter(OriginalMeter):
        def call(self, kind, fn, **kwargs):
            raise ValueError("original refused")
    with pytest.raises(ValueError, match="original refused"):
        b.physical_meter(RefusingMeter()).call("physics_step", lambda: pytest.fail("not dispatched"))
    assert b.counts["physical_steps"]["attempted"] == 0
    assert b.failed and d.failed
    assert close_and_verify(b, d, sha)[0]["status"] == "failed_or_blocked"


def test_callback_failure_and_replay_stop(tmp_path):
    b, d, _, sha = setup(tmp_path)
    def fail():
        raise RuntimeError("native failure")
    with pytest.raises(RuntimeError):
        b.replay(fail)
    with pytest.raises(ValueError, match="active"):
        b.replay(lambda: None)
    result, _, _ = close_and_verify(b, d, sha)
    assert result["counts"]["replay_calls"] == {"attempted": 1, "completed": 0}
    assert result["status"] == "failed_or_blocked"


def test_failed_intent_write_never_dispatches(tmp_path, monkeypatch):
    b, d, _, sha = setup(tmp_path)
    emit = b.journal.emit
    def injected(kind, payload):
        if kind == "intent":
            raise OSError("journal failure")
        return emit(kind, payload)
    monkeypatch.setattr(b.journal, "emit", injected)
    called = []
    with pytest.raises(OSError, match="journal failure"):
        b.replay(lambda: called.append(1))
    assert not called and d.failed
    assert close_and_verify(b, d, sha)[0]["status"] == "failed_or_blocked"


def test_physical_failure_is_nonrefundable(tmp_path):
    b, d, _, sha = setup(tmp_path)
    b.finish_planning(lambda: None)
    def fail():
        raise RuntimeError("physics callback failure")
    with pytest.raises(RuntimeError):
        b.physical_meter(OriginalMeter()).call("physics_step", fail)
    result, _, _ = close_and_verify(b, d, sha)
    assert result["counts"]["physical_steps"] == {"attempted": 1, "completed": 0}
    assert result["status"] == "failed_or_blocked"


def test_rehashed_sidecar_tamper_is_refused(tmp_path):
    b, d, _, sha = setup(tmp_path)
    b.replay(lambda: None)
    b.finish_planning(lambda: None)
    _, side, ledger = close_and_verify(b, d, sha)
    path = b.path / "budget_events.jsonl"
    def mutate(rows):
        rows[-1]["payload"]["counts"]["replay_calls"]["completed"] = 0
    altered = rewrite_chain(path, mutate)
    with pytest.raises(ValueError, match="totals"):
        bridge.verify(path, altered, ledger["journal"], ledger["sha256"], sha, arm="joint", run_id="run")
    with pytest.raises(ValueError, match="digest"):
        bridge.verify(path, side["sha256"], ledger["journal"], ledger["sha256"], sha, arm="joint", run_id="run")


def test_driver_contract_identity_mismatch_fail_stops(tmp_path):
    b, d, _, _ = setup(tmp_path)
    # A second attach is rejected and cannot dispatch anything.
    with pytest.raises(ValueError, match="fresh driver"):
        b.attach(d)
    assert b.failed and d.failed
