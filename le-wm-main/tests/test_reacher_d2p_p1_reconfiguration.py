import pytest
import subprocess
import sys
from pathlib import Path

import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as p1


def test_p1_preflight_direct_script_entrypoint_loads_project_package():
    completed = subprocess.run(
        [sys.executable, str(Path(p1.__file__)), "--help"],
        cwd=Path(p1.__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "CUDA preflight" in completed.stdout


class _Model:
    def encode(self, info):
        return {"emb": info["pixels"]}

    @staticmethod
    def action_encoder(actions):
        return actions

    @staticmethod
    def predict(embeddings, actions):
        return embeddings[:, -1:] + actions[:, -1:]


def test_reconfiguration_preflight_default_preserves_coverage_matched_candidate_budget():
    assert p1.DEFAULT_RECONFIGURATION_PROPOSALS == 128


def test_reconfiguration_preflight_requires_multiple_witnesses_but_not_universal_feasibility():
    assert p1.DEFAULT_MIN_MODEL_WITNESS_ROOTS == 2


def test_widened_reconfiguration_bank_retains_small_bank_prefix_and_zero_baseline():
    small = p1._reconfiguration_proposal_bank(
        proposals=16, horizon=1, action_dim=3, device=torch.device("cpu"), dtype=torch.float32, seed=21,
    )
    wide = p1._reconfiguration_proposal_bank(
        proposals=128, horizon=1, action_dim=3, device=torch.device("cpu"), dtype=torch.float32, seed=21,
    )
    assert torch.equal(small, wide[:16])
    assert torch.count_nonzero(wide[0]) == 0


def test_default_bank_is_uniform_in_physical_action_coordinate_not_unbounded_parameter_coordinate():
    bank = p1._reconfiguration_proposal_bank(
        proposals=8192, horizon=1, action_dim=1, device=torch.device("cpu"), dtype=torch.float32, seed=31,
    )
    physical = torch.tanh(bank[1:])
    assert float(physical.mean()) == pytest.approx(0.0, abs=0.03)
    assert float(physical.std()) == pytest.approx(1.0 / (3.0 ** 0.5), abs=0.025)
    assert float((physical.abs() > 0.9).float().mean()) == pytest.approx(0.1, abs=0.02)


def test_post_prefix_state_uses_the_prefix_as_the_causal_action():
    emb, actions = p1._state_after_prefix(
        _Model(), torch.tensor([[[1.0], [2.0], [3.0]]]), torch.tensor([[[10.0], [20.0], [30.0]]]),
        torch.atanh(torch.tensor([[[0.4]]])), action_lower=torch.tensor([-1.0]), action_upper=torch.tensor([1.0]),
    )
    assert emb.shape == actions.shape == (1, 3, 1)
    assert emb[0, -1, 0].item() == pytest.approx(3.4)
    assert actions[0, -1, 0].item() == pytest.approx(0.4)


def test_recompiled_suffix_starts_after_the_reconfiguration_prefix():
    terminal = p1._terminal_from_state(
        _Model(), torch.tensor([[[2.0], [3.0], [3.4]]]), torch.tensor([[[20.0], [30.0], [0.4]]]),
        torch.atanh(torch.tensor([[[0.2], [0.1]]])), action_lower=torch.tensor([-1.0]), action_upper=torch.tensor([1.0]),
    )
    assert terminal.shape == (1, 1)
    assert terminal.item() == pytest.approx(3.7)


def test_reachability_value_rejects_batched_or_zero_jacobian_states():
    with pytest.raises(ValueError, match="exactly one"):
        p1._reachability_value(
            _Model(), torch.tensor([1.0]), torch.zeros(2, 3, 1), torch.zeros(2, 3, 1),
            future_horizon=1, action_lower=torch.tensor([-1.0]), action_upper=torch.tensor([1.0]), damping=0.1,
        )


def test_reachability_receipt_reports_the_bounded_cost_of_its_analytic_dls_update():
    receipt = p1._reachability_value(
        _Model(), torch.tensor([1.0]), torch.tensor([[[0.0], [0.0], [0.0]]]),
        torch.tensor([[[0.0], [0.0], [0.0]]]), future_horizon=1,
        action_lower=torch.tensor([-1.0]), action_upper=torch.tensor([1.0]), damping=0.1,
    )
    assert receipt["unbounded_dls_parameter_l2"] > 0.0
    assert 0.0 <= receipt["bounded_dls_suffix_goal_cost"] < receipt["predicted_terminal_goal_cost"]


def test_observed_suffix_reuses_the_bounded_single_step_solver_used_for_ranking():
    params, receipt = p1._bounded_single_step_dls_program(
        _Model(), torch.tensor([[[0.0], [0.0], [0.0]]]), torch.tensor([[[1.0]]]),
        torch.tensor([[[0.0], [0.0], [0.0]]]), future_horizon=1,
        action_lower=torch.tensor([-1.0]), action_upper=torch.tensor([1.0]), damping=0.1,
    )
    assert params.shape == (1, 1, 1)
    assert receipt["program_type"] == "bounded_single_step_dls"
    assert receipt["selection_equivalent_suffix_solver"]
    assert receipt["relinearization_steps_executed"] == 1


def test_declared_future_improvement_uses_the_selector_not_the_legacy_surrogate():
    receipt = {
        "future_reachability_value_improvement": -2.0,
        "bounded_dls_suffix_goal_cost_improvement": 0.2,
    }
    assert p1._declared_future_score_improvement(receipt, "analytic_v_reach") == -2.0
    assert p1._declared_future_score_improvement(receipt, "bounded_dls_suffix_goal_cost") == 0.2
