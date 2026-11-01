import copy

import pytest

from experiments.spectral_reachability import p2_hook_contract_bridge_20261030_codex as bridge
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import p2_preregistered_resource_vector_20261030_codex as spec


def fixture(cap=10):
    h = "a" * 64
    c = {"schema": spec.NAME + ".contract.v1", "arms": ["joint", "composition"],
         "scope": "per_root_per_arm_standalone_planning_and_selected_physical_execution",
         "identity": {k: h for k in spec.IDENTITY},
         "axes": {k: {"unit": unit, "aggregation": "max" if k == "peak_device_bytes" else "sum",
                      "cap": cap} for k, unit in spec.AXES.items()}}
    return c, spec.sha(spec.canonical(c)), dict.fromkeys(core.AXES, 100)


def test_actual_public_hook_and_sink_with_contract_caps():
    c, digest, caps = fixture()
    events = []
    ledger = bridge.HookBridge(c, digest, caps, sink=events.append)
    ledger.scope = {"phase": "cpu_fixture"}
    assert ledger.call("predict", lambda: 7, shape=[3, 3, 192]) == 7
    ledger.call("vjp", lambda: None, directions=4)
    ledger.call("array_svd", lambda: None)
    ledger.call("solve", lambda: None)
    ledger.call("replay", lambda: None)
    diag = ledger.diagnostic()
    assert diag["observed_hook_counts"]["predict_samples"]["attempted"] == 3
    assert diag["observed_hook_counts"]["vjp_directions"]["completed"] == 4
    assert diag["observed_hook_counts"]["solve_calls"]["completed"] == 1
    assert all(v["coverage"] == "unknown" for v in diag["observed_hook_counts"].values())
    assert diag["unmapped_axes"]["flops"] is None and not diag["contract_receipt_ready"]
    assert [e["kind"] for e in events] == ["resource_attempt", "resource_return"] * 5


def test_over_cap_blocks_before_function_and_no_later_dispatch():
    c, digest, caps = fixture(2)
    events, calls = [], []
    ledger = bridge.HookBridge(c, digest, caps, sink=events.append)
    ledger.scope = {"phase": "cpu_fixture"}
    ledger.call("predict", lambda: calls.append("ok"), shape=[2, 3, 192])
    with pytest.raises(core.BudgetExceeded):
        ledger.call("predict", lambda: calls.append("forbidden"), shape=[1, 3, 192])
    with pytest.raises(ValueError, match="fail-stop"):
        ledger.call("solve", lambda: calls.append("also forbidden"))
    assert calls == ["ok"] and events[-1]["kind"] == "resource_blocked"
    assert ledger.diagnostic()["observed_hook_counts"]["predict_samples"]["attempted"] == 2


def test_failure_is_charged_and_fail_stops():
    c, digest, caps = fixture()
    events = []
    ledger = bridge.HookBridge(c, digest, caps, sink=events.append)
    ledger.scope = {"phase": "cpu_fixture"}
    def fail():
        raise RuntimeError("fixture failure")
    with pytest.raises(RuntimeError):
        ledger.call("solve", fail)
    assert ledger.attempted["solve_calls"] == 1 and ledger.completed["solve_calls"] == 0
    assert events[-1]["kind"] == "resource_failure"
    with pytest.raises(ValueError, match="fail-stop"):
        ledger.call("solve", lambda: None)
    assert not ledger.diagnostic()["contract_receipt_ready"]


def test_sink_failure_prevents_dispatch():
    c, digest, caps = fixture()
    called = []
    def fail_sink(event):
        raise OSError("cannot persist")
    ledger = bridge.HookBridge(c, digest, caps, sink=fail_sink)
    ledger.scope = {"phase": "fixture"}
    with pytest.raises(OSError):
        ledger.call("solve", lambda: called.append(True))
    assert called == []
    assert ledger.attempted["solve_calls"] == 1


def test_caps_do_not_loosen_and_contract_mutation_fails():
    c, digest, caps = fixture()
    caps["solve_calls"] = 2
    assert bridge.projected_caps(c, digest, caps)["solve_calls"] == 2
    altered = copy.deepcopy(c)
    altered["axes"]["solve_calls"]["cap"] = 20
    with pytest.raises(ValueError, match="digest"):
        bridge.projected_caps(altered, digest, caps)
    with pytest.raises(ValueError, match="sink"):
        bridge.HookBridge(c, digest, caps, sink=None)


@pytest.mark.parametrize("resource,kwargs,axis", [
    ("vjp", {"directions": 2}, "vjp_directions"),
    ("array_svd", {}, "array_svd_calls"),
    ("solve", {}, "solve_calls"),
    ("replay", {}, "replay_calls"),
])
def test_each_mapped_operation_is_blocked_before_dispatch(resource, kwargs, axis):
    c, _, caps = fixture()
    c["axes"][axis]["cap"] = 0
    digest = spec.sha(spec.canonical(c))
    called = []
    ledger = bridge.HookBridge(c, digest, caps, sink=lambda e: None)
    ledger.scope = {"phase": "fixture"}
    with pytest.raises(core.BudgetExceeded):
        ledger.call(resource, lambda: called.append(True), **kwargs)
    assert called == []


def test_bridge_is_not_the_exact_sealed_runner_type():
    c, digest, caps = fixture()
    ledger = bridge.HookBridge(c, digest, caps, sink=lambda e: None)
    assert type(ledger) is not core.Ledger


def test_tampered_native_event_rejected():
    c, digest, caps = fixture()
    ledger = bridge.HookBridge(c, digest, caps, sink=lambda e: None)
    ledger.scope = {"phase": "fixture"}
    ledger.call("predict", lambda: None, shape=[1, 3, 192])
    r = ledger.receipt()
    r["events"][0]["delta"]["predict_samples"] = 0
    with pytest.raises(ValueError, match="actual resource units"):
        bridge.summarize(r, digest)


def test_real_h15_saved_hook_recount_is_not_full_coverage():
    result = bridge.saved_h15_diagnostic()
    assert len(result["rows"]) == 2
    for row in result["rows"]:
        d = row["diagnostic"]
        assert d["observed_hook_counts"]["predict_samples"]["attempted"] == 132
        assert d["observed_hook_counts"]["vjp_directions"]["completed"] == 192
        assert d["unmapped_axes"]["flops"] is None
        assert not d["coverage_verified"] and not d["contract_receipt_ready"]
