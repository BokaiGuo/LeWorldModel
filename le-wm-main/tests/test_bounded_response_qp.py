import itertools

import pytest
import torch

from experiments.spectral_reachability.bounded_response_qp import solve_box_quadratic, certified_response_gain, qualify_certified_decision


def test_diagonal_solution_and_certificate():
    gram = torch.diag(torch.tensor([1., 2., 3.], dtype=torch.float64))
    projected = torch.tensor([2., -.5, -4.], dtype=torch.float64)
    action, certificate = solve_box_quadratic(gram, projected)
    assert torch.allclose(action, torch.tensor([1., -.25, -1.], dtype=torch.float64))
    assert certificate["active_set_terminated"]
    assert certificate["objective_duality_gap_upper_bound"] < 1e-12


@pytest.mark.parametrize("seed", range(5))
def test_correlated_qp_matches_exhaustive_small_active_sets(seed):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    gram = matrix.T @ matrix + .01 * torch.eye(3, dtype=torch.float64)
    projected = torch.randn(3, generator=generator, dtype=torch.float64) * 2
    feasible = []
    for status in itertools.product((-1, 0, 1), repeat=3):
        action = torch.tensor(status, dtype=torch.float64)
        free = action == 0
        active = ~free
        if free.any():
            action[free] = torch.linalg.solve(gram[free][:, free], projected[free] - gram[free][:, active] @ action[active])
        if action.abs().max() <= 1 + 1e-12:
            feasible.append(float(action @ gram @ action - 2 * projected @ action))
    action, certificate = solve_box_quadratic(gram, projected)
    actual = float(action @ gram @ action - 2 * projected @ action)
    assert actual == pytest.approx(min(feasible), abs=1e-10)
    assert certificate["objective_duality_gap_upper_bound"] < 1e-9


def test_ill_conditioned_twenty_coordinate_problem_certified():
    generator = torch.Generator().manual_seed(481)
    left, _ = torch.linalg.qr(torch.randn(20, 20, generator=generator, dtype=torch.float64))
    response = torch.diag(torch.logspace(-2, 3, 20, dtype=torch.float64)) @ left
    residual = torch.randn(20, generator=generator, dtype=torch.float64)
    value = certified_response_gain(response, residual)
    assert value["objective_certificate_passed"]
    assert value["qp_certificate"]["objective_duality_gap_upper_bound"] < 1e-6


def test_non_spd_rejected():
    with pytest.raises(torch.linalg.LinAlgError):
        solve_box_quadratic(-torch.eye(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64))


def test_zero_response_certified_gain_zero():
    value = certified_response_gain(torch.zeros(2, 20, dtype=torch.float64), torch.ones(2, dtype=torch.float64))
    assert value["bounded_quadratic_gain"] == 0
    assert value["bounded_quadratic_gain_upper_bound"] == 0


def test_uncertain_gain_intervals_cannot_open_witness():
    rows = [{"proposal_index": 0, "immediate_goal_cost": 1., "refined_goal_cost": .5},
        {"proposal_index": 1, "immediate_goal_cost": .2, "refined_goal_cost": .3},
        {"proposal_index": 2, "immediate_goal_cost": 1.2, "refined_goal_cost": .1}]
    receipt = {"proposal_rows": rows, "non_greedy_margin": 1e-6,
        "dispatch": {"same_bank_greedy_index": 1, "selected_program_index": 2}}
    metrics = {index: {"bounded_quadratic_gain": gain, "bounded_quadratic_gain_upper_bound": upper,
        "objective_certificate_passed": True} for index, (gain, upper) in enumerate([(.4, .6), (.3, .3), (.5, .5)])}
    decision = qualify_certified_decision(receipt, metrics)
    assert not decision["reconfiguration_witness_found"]
    assert decision["selected_program_index"] == 1
    metrics[0]["bounded_quadratic_gain_upper_bound"] = .45
    assert qualify_certified_decision(receipt, metrics)["selected_program_index"] == 2
    metrics[2]["objective_certificate_passed"] = False
    with pytest.raises(ValueError, match="certified"):
        qualify_certified_decision(receipt, metrics)


def test_active_set_cap_is_reported_not_assumed_converged():
    action, receipt = solve_box_quadratic(torch.eye(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64) * 2,
        max_iterations=1)
    assert action.abs().max() <= 1
    assert not receipt["active_set_terminated"]
