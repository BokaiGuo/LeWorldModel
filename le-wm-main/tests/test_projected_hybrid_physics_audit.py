import numpy as np

from experiments.empirical_spectral_audit.run_projected_hybrid_physics_audit import (
    finite_horizon_hankel_geometry,
    fit_memory_kernel,
    fit_mode_conditioned_operators,
    frequency_response,
    hybrid_path_gain_summary,
    memory_companion_operator,
    memory_transfer,
    multiscale_transfer_consistency,
    passivity_residual,
    se2_body_twist,
)


def test_se2_body_twist_handles_angle_wrap_and_zero_translation():
    before = np.asarray([0.0, 0.0, np.pi - 0.02])
    after = np.asarray([0.0, 0.0, -np.pi + 0.03])
    twist = se2_body_twist(before, after)
    assert np.allclose(twist[:2], 0.0, atol=1e-9)
    assert np.isclose(twist[2], 0.05, atol=1e-8)


def test_memory_fit_and_companion_transfer_are_finite():
    rng = np.random.default_rng(11)
    eta = rng.normal(size=(160, 2))
    action = rng.normal(size=(160, 1))
    next_eta = np.zeros_like(eta)
    next_eta[1:] = 0.7 * eta[:-1] + 0.15 * eta[:-1] + action[:-1]
    fitted = fit_memory_kernel(eta, next_eta, action, memory_length=2)
    companion = memory_companion_operator(fitted["A"], fitted["kernels"])
    report = memory_transfer(fitted["A"], fitted["kernels"], fitted["B"], np.eye(2), horizon=4)
    assert companion.shape == (4, 4)
    assert np.isfinite(companion).all()
    assert len(report["transfer_operators"]) == 4
    assert np.isfinite(report["max_task_transfer_gain"])


def test_memory_fit_does_not_duplicate_current_state():
    rng = np.random.default_rng(101)
    eta = rng.normal(size=(240, 2))
    action = rng.normal(size=(240, 1))
    next_eta = np.zeros_like(eta)
    for t in range(2, len(eta) - 1):
        next_eta[t] = 0.6 * eta[t] + 0.2 * eta[t - 1] + action[t]
    fitted = fit_memory_kernel(eta, next_eta, action, memory_length=3, ridge=1e-10)
    assert len(fitted["kernels"]) == 2
    assert np.allclose(fitted["A"], 0.6 * np.eye(2), atol=0.08)
    assert np.isfinite(memory_companion_operator(fitted["A"], fitted["kernels"])).all()


def test_hankel_frequency_and_hybrid_path_tools_are_finite():
    A = np.diag([0.8, 0.5])
    B = np.eye(2)
    D = np.eye(2)
    hankel = finite_horizon_hankel_geometry(A, B, D, horizon=4)
    response = frequency_response(A, B, D, np.linspace(0.0, np.pi, 9))
    paths = hybrid_path_gain_summary(
        [["free", "stick"], ["free", "slide", "stick"]],
        {
            "free": {"A": A, "B": B, "D": D},
            "stick": {"A": 0.7 * A, "B": B, "D": D},
            "slide": {"A": 0.9 * A, "B": B, "D": D},
        },
    )
    assert hankel["hankel_singular_values"].shape == (2,)
    assert np.isfinite(response["gains"]).all()
    assert paths["n_paths"] == 2
    assert np.isfinite(paths["max_gain"])


def test_mode_conditioned_fit_reports_support_and_separates_modes():
    rng = np.random.default_rng(12)
    eta = rng.normal(size=(80, 2))
    action = rng.normal(size=(80, 1))
    modes = np.asarray(["stick"] * 40 + ["slide"] * 40)
    next_eta = eta.copy()
    next_eta[:40, 0] += action[:40, 0]
    next_eta[40:, 1] += 2.0 * action[40:, 0]
    fitted = fit_mode_conditioned_operators(eta, next_eta, action, modes)
    assert fitted["stick"]["status"] == "fit"
    assert fitted["slide"]["status"] == "fit"
    assert not np.allclose(fitted["stick"]["B"], fitted["slide"]["B"])


def test_passivity_and_multiscale_consistency_are_auditable():
    energy_before = np.asarray([1.0, 2.0])
    energy_after = np.asarray([0.8, 2.2])
    output = np.asarray([[0.0], [0.0]])
    action = np.asarray([[0.0], [0.0]])
    report = passivity_residual(energy_before, energy_after, output, action)
    assert report["violation_rate"] == 0.5
    consistency = multiscale_transfer_consistency({"1": np.eye(2), "2": 0.5 * np.eye(2)})
    assert consistency["mean_discrepancy"] == 0.0
