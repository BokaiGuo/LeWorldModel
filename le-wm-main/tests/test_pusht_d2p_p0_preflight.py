from pathlib import Path

import pytest
import torch

from experiments.spectral_reachability.preflight_pusht_d2p_p0 import (
    _bounded_live_actions,
    _image_tensor,
    _model_action_tokens,
    _reserved_outcome_contract,
)


def test_live_action_parameterization_stays_in_the_declared_environment_box():
    values = _bounded_live_actions(torch.tensor([[-100.0, 0.0], [100.0, 1.0]]))
    assert tuple(values.shape) == (2, 2)
    assert bool(torch.all(values <= 1.0))
    assert bool(torch.all(values >= -1.0))
    assert values[0, 0] < -0.999
    assert values[1, 0] > 0.999


def test_action_token_transform_is_time_major_and_preserves_gradients():
    live = torch.tensor([[0.5, -0.25], [0.1, 0.2]], requires_grad=True)
    tokens = _model_action_tokens(live, action_block=5)
    assert tuple(tokens.shape) == (2, 10)
    assert torch.allclose(tokens[0, :2], tokens[0, 2:4])
    tokens.sum().backward()
    assert live.grad is not None
    assert bool(torch.isfinite(live.grad).all())
    assert bool(torch.all(live.grad != 0.0))


def test_action_token_transform_rejects_invalid_layout():
    with pytest.raises(ValueError, match="layout"):
        _model_action_tokens(torch.zeros((2, 3)), action_block=5)
    with pytest.raises(ValueError, match="layout"):
        _model_action_tokens(torch.zeros((2, 2)), action_block=0)


def test_single_rgb_frame_uses_the_encoder_five_axis_contract():
    image = torch.zeros((3, 4, 4), dtype=torch.uint8).permute(1, 2, 0).numpy()
    tensor = _image_tensor(image, transform=lambda value: value.float(), device=torch.device("cpu"))
    assert tuple(tensor.shape) == (1, 1, 3, 4, 4)


def test_outcome_contract_is_reserved_without_any_model_or_backward_calls():
    contract = _reserved_outcome_contract(root_seed=17, root_count=24)
    assert contract["status"] == "reserved_not_opened"
    assert contract["model_calls"] == 0
    assert contract["backward_calls"] == 0


def test_cuda_launcher_is_explicitly_host_only_and_never_offers_cpu_fallback():
    launcher = Path(__file__).resolve().parents[1] / "experiments/spectral_reachability/run_pusht_d2p_p0_cuda.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "experiments.gpu_preflight --device cuda" in text
    assert "experiments.spectral_reachability.preflight_pusht_d2p_p0" in text
    assert "--device cuda" in text
    assert "no CPU fallback" in text
