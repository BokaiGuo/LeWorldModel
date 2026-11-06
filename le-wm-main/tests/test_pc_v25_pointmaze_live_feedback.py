import numpy as np
import pytest
import json

from experiments import pc_v25_pointmaze_live_feedback as wrapper
from experiments.pc_equal_resource_dev_20261103_codex import Meter


def test_v25_contract_name_and_root_validation():
    contract = {"start": [.1, .2], "root_id": 9, "output_name": "pc_v25_new_root",
                "scope": "development", "goal_sha256": "abc"}
    start, root, name = wrapper.validate_contract(contract, "abc")
    np.testing.assert_array_equal(start, [.1, .2])
    assert (root, name) == (9, "pc_v25_new_root")
    with pytest.raises(ValueError):
        wrapper.validate_contract(dict(contract, output_name="pc_v23_old"), "abc")
    with pytest.raises(ValueError):
        wrapper.validate_contract(dict(contract, output_name="pc_v25_../bad"), "abc")


def test_harness_configuration_is_restored_after_failure(monkeypatch):
    harness = wrapper.harness
    original = harness.feedback, harness.PROTOCOL, harness.dependencies, harness.validate_contract

    def fail():
        assert harness.feedback is wrapper.feedback
        assert harness.PROTOCOL == wrapper.PROTOCOL
        raise RuntimeError("injected")

    monkeypatch.setattr(harness, "main", fail)
    with pytest.raises(RuntimeError, match="injected"):
        wrapper.main()
    assert (harness.feedback, harness.PROTOCOL, harness.dependencies, harness.validate_contract) == original


def test_prediction_journal_preserves_failed_call_and_cost(tmp_path):
    meter = Meter(10)

    def fail(actions):
        raise RuntimeError("model failed after invocation")

    path = tmp_path / "predictions"
    predict = wrapper.harness.durable_predictor(path, meter, fail, {"root": 42}, 10)
    actions = np.full((2, 5, 2), 0.25, dtype=np.float32)
    with pytest.raises(RuntimeError, match="model failed"):
        predict(actions)

    attempt = json.loads((path / "000000_attempt.json").read_text())
    failure = json.loads((path / "000000_failed.json").read_text())
    assert attempt["cost"] == meter.token_samples == predict.tokens == 2
    assert meter.calls == predict.calls == 1
    assert failure["type"] == "RuntimeError"
    assert json.loads((path / "binding.json").read_text()) == {"root": 42}
    with np.load(path / attempt["actions"]) as saved:
        np.testing.assert_array_equal(saved["actions"], actions)
    assert not (path / "000000_returned.json").exists()


def test_prediction_journal_records_returned_results(tmp_path):
    meter = Meter(10)
    predict = wrapper.harness.durable_predictor(
        tmp_path / "predictions", meter,
        lambda actions: (np.zeros((len(actions), 2)), np.ones(len(actions))),
        {"step": 0}, 10)
    predict(np.zeros((1, 5, 2), dtype=np.float32))
    assert (tmp_path / "predictions" / "000000_returned.json").exists()
    assert meter.token_samples == predict.tokens == 1
