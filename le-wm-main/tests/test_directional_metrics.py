import numpy as np
import torch

from experiments.directional_metrics import (
    band_energy_profile,
    compute_eigenspace,
    feature_space_adapter,
    gaussian_control_like,
    shuffled_control,
    spectral_descriptors,
    summarize_action_conditioned,
    tangent_retention,
    tangent_cosine,
    top_band_lift,
)


def test_eigenspace_shapes_and_orthogonality():
    x = torch.randn(512, 32)
    eig = compute_eigenspace(x)

    assert eig.eigenvalues.shape == (32,)
    assert eig.eigenvectors.shape == (32, 32)
    eye = eig.eigenvectors.T @ eig.eigenvectors
    assert torch.allclose(eye, torch.eye(32), atol=1e-5)


def test_gaussian_control_lift_is_near_one():
    torch.manual_seed(0)
    x = torch.randn(4096, 32)
    eig = compute_eigenspace(torch.randn(4096, 32))
    control = gaussian_control_like(x, seed=1)

    lift = top_band_lift(eig.eigenvectors, control, k=8)
    assert 0.75 <= lift <= 1.25


def test_shuffled_tangent_control_preserves_lift_for_same_eigenspace():
    torch.manual_seed(1)
    x = torch.randn(1024, 16)
    tangents = np.random.default_rng(0).normal(size=(1024, 16)).astype("float32")
    eig = compute_eigenspace(x)

    original = top_band_lift(eig.eigenvectors, tangents, k=4)
    shuffled = top_band_lift(eig.eigenvectors, shuffled_control(tangents, seed=2), k=4)

    assert abs(original - shuffled) < 1e-6



def test_band_profile_sums_to_one_for_full_partition():
    torch.manual_seed(2)
    x = torch.randn(512, 12)
    eig = compute_eigenspace(x)
    profile = band_energy_profile(eig.eigenvectors, x, [(0, 4), (4, 8), (8, 12)])

    assert abs(sum(profile.values()) - 1.0) < 1e-5


def test_tangent_cosine_detects_alignment():
    x = torch.randn(128, 8)

    assert tangent_cosine(x, x) > 0.999
    assert tangent_cosine(x, -x) < -0.999


def test_action_conditioned_summary_reports_weighted_lift():
    rng = np.random.default_rng(3)
    latents = rng.normal(size=(256, 8)).astype("float32")
    tangents = {"dynamics": rng.normal(size=(256, 8)).astype("float32")}
    actions = np.repeat(np.arange(4, dtype="float32"), 64)[:, None]

    summary = summarize_action_conditioned(
        latents, tangents, actions, ks=[2, 4], n_action_bins=4, min_bin_size=32
    )

    assert len(summary["clusters"]) == 4
    assert "dynamics_lift" in summary["weighted_mean"]
    assert set(summary["weighted_mean"]["dynamics_lift"]) == {"2", "4"}


def test_spectral_descriptors_report_shape_metrics():
    desc = spectral_descriptors(torch.tensor([4.0, 1.0, 1.0, 0.0]), tail_ks=[1, 2])

    assert desc["dim"] == 4
    assert desc["stable_rank"] == 1.5
    assert desc["participation_ratio"] == 36.0 / 18.0
    assert np.isclose(desc["tail_mass"]["1"], 2.0 / 6.0)
    assert np.isclose(desc["tail_mass"]["2"], 1.0 / 6.0)


def test_feature_space_adapter_selectively_suppresses_target_tangent():
    target = torch.zeros(64, 4)
    target[:, 0] = 2.0
    non_target = torch.zeros(64, 4)
    non_target[:, 1] = 2.0

    adapter = feature_space_adapter(target, strength=10.0)

    assert tangent_retention(adapter, target) < 0.01
    assert tangent_retention(adapter, non_target) > 0.99
