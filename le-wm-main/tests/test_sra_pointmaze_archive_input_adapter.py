import torch

from experiments.spectral_reachability.run_sra_p0_pointmaze_cuda import prepare_archive_observation_for_model


class _Dataset:
    proprio_mean = torch.tensor([1.5, -2.0])
    proprio_std = torch.tensor([2.0, 4.0])


def test_archive_adapter_restores_raw_rgb_and_proprio_before_model_encode():
    archive = {
        "visual": torch.full((1, 3, 2, 2), 0.25),
        "proprio": torch.tensor([[2.0, -0.5]]),
    }
    adapted = prepare_archive_observation_for_model(_Dataset(), archive)
    assert torch.equal(adapted["visual"], torch.full((1, 3, 2, 2), 63.75))
    assert torch.allclose(adapted["proprio"], torch.tensor([[5.5, -4.0]]))


def test_archive_adapter_rejects_unexpected_rgb_scale():
    archive = {"visual": torch.full((1, 3, 2, 2), 2.0), "proprio": torch.zeros((1, 2))}
    try:
        prepare_archive_observation_for_model(_Dataset(), archive)
    except ValueError as error:
        assert "[0, 1]" in str(error)
    else:
        raise AssertionError("adapter accepted already-raw RGB and would permit double scaling")
