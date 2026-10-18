import torch

from experiments.train_cube_multiview_spatial_adapter import SpatialAdapter, _evaluate_zero_baseline


def test_spatial_adapter_has_geometry_reconstruction_and_effect_heads():
    model = SpatialAdapter()
    geometry = torch.randn(4, 10)
    action = torch.randn(4, 2)
    reconstruction, effect = model(geometry, action)
    assert reconstruction.shape == geometry.shape
    assert effect.shape == (4, 4)


def test_spatial_adapter_effect_depends_on_action():
    model = SpatialAdapter()
    geometry = torch.randn(2, 10)
    first = torch.zeros(2, 2)
    second = torch.ones(2, 2)
    _, effect_first = model(geometry, first)
    _, effect_second = model(geometry, second)
    assert not torch.allclose(effect_first, effect_second)


def test_zero_baseline_is_fit_on_calibration_targets_only():
    train = torch.tensor([[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0]]).numpy()
    test = torch.zeros((1, 4)).numpy()
    report = _evaluate_zero_baseline(train, test)
    assert report["effect_rmse"] == 2.0
    assert report["effect_mae"] == 2.0
    assert report["sign_accuracy"] == 0.0
