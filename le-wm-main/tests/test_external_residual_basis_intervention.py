from experiments.empirical_spectral_audit.run_external_residual_basis_intervention import (
    INTERVENTIONS,
)


def test_residual_intervention_matrix_contains_raw_and_energy_matched_arms():
    assert "residual_top8" in INTERVENTIONS
    assert "residual_covariance_top8" in INTERVENTIONS
    assert INTERVENTIONS["residual_top8_energy_matched_to_dynamics"] == (
        "residual_basis",
        True,
    )
    assert INTERVENTIONS["residual_covariance_top8_energy_matched_to_dynamics"] == (
        "residual_covariance_basis",
        True,
    )
