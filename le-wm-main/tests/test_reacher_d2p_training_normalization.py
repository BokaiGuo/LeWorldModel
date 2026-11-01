"""Module-state contract tests only, not CPU neural experiments."""
import pytest
import torch

from experiments.spectral_reachability.reacher_d2p_training_normalization import dynamics_mode, normalization_snapshot, verify_normalization


def fixture():
    model = torch.nn.Module()
    for name in ("encoder", "projector", "predictor", "action_encoder", "pred_proj"):
        setattr(model, name, torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2), torch.nn.Dropout(.1)))
    return model


def test_default_keeps_stats_frozen_without_freezing_affine_or_dropout():
    model = fixture()
    before = normalization_snapshot(model)
    dynamics_mode(model, True)
    assert not model.encoder.training and not model.projector.training
    for module in (model.predictor, model.action_encoder, model.pred_proj):
        assert module.training and module[2].training
        assert not module[1].training
        assert module[1].weight.requires_grad and module[1].bias.requires_grad
    assert verify_normalization(model, before)["base_normalization_buffers_unchanged"] is True


def test_legacy_is_explicit_and_eval_remains_eval():
    model = fixture()
    dynamics_mode(model, True, "legacy-domain-updates")
    assert model.pred_proj[1].training
    dynamics_mode(model, False, "legacy-domain-updates")
    assert not any(m.training for module in (model.predictor, model.action_encoder, model.pred_proj) for m in module.modules())


def test_silent_buffer_drift_fails_closed():
    model = fixture()
    before = normalization_snapshot(model)
    model.pred_proj[1].running_mean.add_(1)
    with pytest.raises(RuntimeError, match="drifted"):
        verify_normalization(model, before)
    assert verify_normalization(model, before, "legacy-domain-updates")["base_normalization_buffers_unchanged"] is False


def test_batch_stats_only_normalization_rejected():
    model = fixture()
    model.pred_proj[1] = torch.nn.BatchNorm1d(2, track_running_stats=False)
    with pytest.raises(ValueError, match="fixed deployment"):
        dynamics_mode(model, True)


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        dynamics_mode(fixture(), True, "unbound")
