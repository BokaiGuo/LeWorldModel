import hashlib

import pytest
import torch

from experiments.spectral_reachability.audit_reacher_d2p_p1_first_chunk_fidelity import (
    _first_successor_prediction,
    _strict_order_agrees,
    _target_delta_from_report,
    _validate_report,
    _validate_runtime_inputs,
)


class _Model:
    @staticmethod
    def encode(info):
        return {"emb": info["pixels"].mean(dim=(-1, -2, -3), keepdim=False).unsqueeze(-1)}

    @staticmethod
    def action_encoder(actions):
        return actions

    @staticmethod
    def predict(embeddings, actions):
        return embeddings[:, -1:] + actions[:, -1:, :1]


class _AllActionSlotsModel(_Model):
    @staticmethod
    def predict(embeddings, actions):
        return embeddings[:, -1:] + actions.sum(dim=1, keepdim=True)[..., :1]


def test_first_chunk_fidelity_predicts_from_observed_root_and_recorded_bounded_packet():
    pixels = torch.tensor([[[[[1.0]]], [[[2.0]]], [[[3.0]]]]])
    history = torch.tensor([[[10.0], [20.0], [30.0]]])
    normalized_packet = torch.tensor([[4.0]])

    predicted = _first_successor_prediction(_Model(), pixels, history, normalized_packet)

    assert predicted.shape == (1,)
    assert float(predicted[0]) == 7.0


def test_first_chunk_fidelity_uses_the_two_actions_that_causally_precede_the_root():
    pixels = torch.zeros((1, 3, 1, 1, 1))
    history = torch.tensor([[[10.0], [20.0], [30.0]]])

    predicted = _first_successor_prediction(
        _AllActionSlotsModel(), pixels, history, torch.tensor([[4.0]]),
    )

    # The root-anchored RGB context is [o_{t-2}, o_{t-1}, o_t], so its
    # causal action window is [a_{t-2}, a_{t-1}, a_t] = [20, 30, 4], not
    # the discarded pre-context action 10.
    assert float(predicted[0]) == 54.0


def test_first_chunk_fidelity_rejects_a_second_tanh_style_shape_error():
    pixels = torch.zeros((1, 3, 1, 1, 1))
    history = torch.zeros((1, 3, 2))
    with pytest.raises(ValueError, match="incompatible"):
        _first_successor_prediction(_Model(), pixels, history, torch.zeros((1, 1)))


def test_first_chunk_fidelity_prediction_uses_the_recorded_token_without_a_preprocessor_private_field():
    pixels = torch.zeros((1, 3, 1, 1, 1))
    history = torch.zeros((1, 3, 1))
    assert float(_first_successor_prediction(_Model(), pixels, history, torch.tensor([[0.5]]))[0]) == 0.5


def test_first_chunk_fidelity_order_requires_the_observed_order_to_match_model_order():
    assert _strict_order_agrees(2.0, 1.0, 3.0, 2.0)
    assert not _strict_order_agrees(2.0, 1.0, 0.5, 1.0)


def test_first_chunk_fidelity_requires_a_one_token_p1_receipt():
    with pytest.raises(ValueError, match="not a Reacher"):
        _validate_report({"schema": "bad"})
    with pytest.raises(ValueError, match="one-token"):
        _validate_report({"schema": "d2p_reacher_p1_reconfiguration.v1", "rows": [{"seed": 1}], "protocol": {"action_block": 5, "reconfiguration_horizon": 2}})


def test_first_chunk_fidelity_requires_the_sealed_target_geometry():
    assert _target_delta_from_report({"protocol": {"target_delta": [0.65, -0.65]}}) == (0.65, -0.65)
    with pytest.raises(ValueError, match="target_delta"):
        _target_delta_from_report({"protocol": {"target_delta": [0.65]}})


def test_first_chunk_fidelity_rejects_a_checkpoint_or_adapter_not_sealed_by_the_report(tmp_path):
    policy = tmp_path / "checkpoint"
    policy.mkdir()
    weights = policy / "weights.pt"
    adapter = tmp_path / "adapter.json"
    weights.write_bytes(b"sealed weights")
    adapter.write_bytes(b"sealed adapter")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"inputs": {"weights_sha256": digest(weights), "adapter_sha256": digest(adapter)}}

    assert _validate_runtime_inputs(report, policy=policy, adapter=adapter)["checkpoint_weights_sha256"] == digest(weights)
    report["inputs"]["adapter_sha256"] = "wrong"
    with pytest.raises(ValueError, match="adapter"):
        _validate_runtime_inputs(report, policy=policy, adapter=adapter)
