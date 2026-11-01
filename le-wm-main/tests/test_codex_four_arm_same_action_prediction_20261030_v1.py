import numpy as np
import pytest
import torch

from experiments.spectral_reachability import codex_four_arm_same_action_prediction_20261030_v1 as m


def test_fixed_packet_uses_actual_history_and_normalization():
    seen = {}
    class Model:
        def action_encoder(self, actions):
            seen["actions"] = actions.clone()
            return actions
        def predict(self, states, encoded):
            seen["states"] = states.clone()
            assert encoded is not None
            return states + 2
    task = {"root_context": {"latent_history": np.ones((3, 192), np.float32).tolist(),
                              "action_history": np.ones((2, 10), np.float32).tolist()},
            "action_mean": [0.5, -0.5], "action_std": [0.5, 0.5]}
    raw = np.zeros((5, 2), np.float32)
    result = m.predict_one(Model(), torch, torch.device("cpu"), {"task": task, "actions": raw})
    assert result.shape == (192,) and np.all(result == 3)
    assert seen["actions"].shape == (1, 3, 10)
    assert seen["actions"][0, -1].tolist() == [-1., 1.] * 5
    assert torch.equal(seen["actions"][0, :2], torch.ones((2, 10)))


@pytest.mark.parametrize("value", [
    [[float("nan"), 0.]] * 5, [[1., 0.]] * 4, [[0.1, 0.]] * 5,
])
def test_reject_nonfinite_wrong_shape_or_nonliteral_float32(value):
    with pytest.raises(ValueError, match="literal"):
        m.f32(value, (5, 2))


def test_wrong_authorization_fails_before_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "saved_inputs", lambda: pytest.fail("must not read"))
    with pytest.raises(ValueError, match="authorization"):
        m.run(str(tmp_path / "out"), "wrong")


def test_wrong_saved_report_digest_rejected_before_prepare(monkeypatch):
    monkeypatch.setattr(m, "REPORT_SHA", "0" * 64)
    monkeypatch.setattr(m.prior, "verify", lambda *a: pytest.fail("must not load prepare"))
    with pytest.raises(ValueError, match="fixed report hash"):
        m.saved_inputs()


def test_reject_bad_model_prediction_shape():
    class Model:
        def action_encoder(self, actions):
            return actions
        def predict(self, states, encoded):
            return torch.zeros((1, 192))
    task = {"root_context": {"latent_history": np.zeros((3, 192)).tolist(),
                              "action_history": np.zeros((2, 10)).tolist()},
            "action_mean": [0., 0.], "action_std": [1., 1.]}
    with pytest.raises(ValueError, match="prediction ABI"):
        m.predict_one(Model(), torch, torch.device("cpu"), {"task": task, "actions": np.zeros((5, 2), np.float32)})


def test_goal_error_decomposition_includes_cross_term():
    value = m.errors(np.full(192, 3, np.float32), np.full(192, 2, np.float32),
                     np.zeros(192, np.float32))
    assert value["predicted_goal_mse"] == 9
    assert value["actual_goal_mse"] == 4 and value["mse"] == 1
    assert value["goal_mse_cross_term"] == 4
    assert value["predicted_goal_mse"] == (value["actual_goal_mse"] + value["mse"]
                                            + value["goal_mse_cross_term"])


def test_action_encoder_failure_does_not_record_predict_return():
    class Model:
        def action_encoder(self, actions):
            raise RuntimeError("failed action encoder")
        def predict(self, *args):
            pytest.fail("prediction after failed action encoder")
    task = {"root_context": {"latent_history": np.zeros((3, 192)).tolist(),
                              "action_history": np.zeros((2, 10)).tolist()},
            "action_mean": [0., 0.], "action_std": [1., 1.]}
    events = []
    with pytest.raises(RuntimeError, match="failed action"):
        m.predict_one(Model(), torch, torch.device("cpu"),
                      {"task": task, "actions": np.zeros((5, 2), np.float32)}, events.append)
    assert events == ["action_encoder_intent"]
