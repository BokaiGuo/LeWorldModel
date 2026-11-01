import numpy as np
import pytest

from experiments.spectral_reachability import codex_p1_cp10_rgb_refresh_runner_20261101_exclusive as runner


def test_output_name_obeys_inherited_exclusive_store_contract(monkeypatch):
    monkeypatch.setattr(runner.engine, "NAME", runner.NAME)
    path = runner.output_path("UNITTEST01")
    assert runner.engine.exclusive(path, "run") == path


def test_meter_records_attempt_before_work_and_failure_is_not_return():
    class Store:
        def __init__(self):
            self.events = []

        def emit(self, kind, **fields):
            self.events.append((kind, fields))

    store = Store()
    meter = runner.Meter(store, "role", lambda: None)
    with pytest.raises(RuntimeError):
        meter.call("predict", {"boundary": 5}, lambda: (_ for _ in ()).throw(RuntimeError("failure")))
    assert meter.attempts["predict"] == 1
    assert meter.returns["predict"] == 0
    assert [e[0] for e in store.events] == ["neural_attempt"]


def test_fixed_meter_enforces_cap_without_executing_excess():
    class Store:
        def emit(self, kind, **fields):
            pass

    meter = runner.Meter(Store(), "role", lambda: None)
    for _ in range(6):
        meter.call("predict", {}, lambda: 0)
    with pytest.raises(Exception):
        meter.call("predict", {}, lambda: pytest.fail("must not run over budget"))
    assert meter.attempts["predict"] == meter.returns["predict"] == 6


def test_module_hooks_must_equal_metered_calls():
    runner.check_hooks({"predict": 3, "action_encoder": 3},
                       {"predict": 1, "action_encoder": 1},
                       {"predict": 2, "action_encoder": 2})
    with pytest.raises(Exception):
        runner.check_hooks({"predict": 4, "action_encoder": 3},
                           {"predict": 1, "action_encoder": 1},
                           {"predict": 2, "action_encoder": 2})


def test_activation_pins_verifier_and_all_three_test_sources():
    binding = runner.activation()
    paths = binding["verification_sources"]
    assert len(paths) == 4
    assert all(runner.sha(path) == digest for path, digest in paths.items())
