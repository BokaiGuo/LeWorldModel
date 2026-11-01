"""CPU-only contract regressions for the new CEM runner."""
import json
import importlib.util
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p1_full_cem_host_20261101_exclusive as old
from experiments.spectral_reachability import codex_p1_full_cem_host_v2_20261101_exclusive as v2


def legacy_protocol(root=0):
    report = Path(old.source.ROOT) / "results" / (
        f"codex_p1_full_cem_host_20261101_exclusive_root{root}_RUN_MAIN01/report.json")
    return json.loads(report.read_text())["protocol"]


@pytest.mark.parametrize("root", [0, 1])
def test_protocol_replaces_legacy_claims_and_preserves_limits(root):
    original = legacy_protocol(root)
    value = v2.make_protocol(original, root)
    for field in ("node_caps", "initializer_labels", "common_suffix", "extra_policy"):
        assert field not in value
    assert value["node_caps_by_boundary"] == {str(b): old.node_cost(b) for b in (0, 5, 10, 15)}
    assert value["node_costs"] == value["node_caps_by_boundary"]
    assert value["module_hooks_gate_each_node"] is False
    assert value["module_hook_coverage_verified"] is False
    assert value["cem"]["iterations"] == 4
    assert value["cem"]["samples_per_iteration"] == 16
    assert value["cem"]["elites"] == 4
    assert value["physical_caps"] == original["physical_caps"]
    assert value["cuda_allocated_bytes"] == original["cuda_allocated_bytes"]
    assert not value["fairness_proven"] and not value["full_cost_certified"]


def test_wrong_root_or_drift_fails_closed():
    protocol = legacy_protocol()
    with pytest.raises(ValueError):
        v2.make_protocol(protocol, 1)
    protocol["node_costs"]["0"]["predict_calls"] = 171
    with pytest.raises(ValueError):
        v2.make_protocol(protocol, 0)


def test_v2_rejects_prior_prepares_and_cross_root_names(tmp_path):
    root0 = tmp_path / f"{v2.NAME}_root0_PREP_test.json"
    old_path = tmp_path / "codex_p1_full_cem_host_20261101_exclusive_root0_PREP_MAIN01.json"
    v2.check_prepare_name(root0, 0)
    with pytest.raises(ValueError):
        v2.check_prepare_name(root0, 1)
    with pytest.raises(ValueError):
        v2.check_prepare_name(old_path, 0)


def test_v2_authorization_is_new_and_does_not_reuse_previous():
    assert v2.AUTH != old.AUTH


def test_real_execute_orchestration_receives_v2_protocol_cpu_fake(tmp_path, monkeypatch):
    fixture_path = Path(__file__).with_name("test_codex_p1_full_cem_host_20261101_exclusive.py")
    spec = importlib.util.spec_from_file_location("v1_cem_fixture", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    prepared, data, store, runtime, factory, fake_runner, rebind = fixture.fixture_runtime(
        tmp_path, monkeypatch, stop_step=6)
    legacy = legacy_protocol()
    legacy["node_costs"] = {str(b): old.node_cost(b) for b in v2.BOUNDARIES}
    fake_runner.protocol = lambda: v2.make_protocol(legacy, 0)
    with monkeypatch.context() as patcher:
        patcher.setattr(old, "NAME", v2.NAME)
        report = old.execute_one(prepared, data, store, None, runtime, factory, fake_runner, rebind, 0)
    assert report["schema"] == v2.NAME + ".report.v1"
    assert report["protocol"]["module_hooks_gate_each_node"] is False
    assert "node_caps" not in report["protocol"]
    start = next(event[1] for event in store.events if event[0] == "run_started")
    assert start["contract"] == report["protocol"]
    assert report["rows"][0]["control_returned"] == 6
