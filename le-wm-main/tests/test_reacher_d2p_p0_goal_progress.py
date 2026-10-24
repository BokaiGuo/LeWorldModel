import json

import numpy as np
import pytest
import torch
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as d2p

from experiments.spectral_reachability.run_reacher_d2p_p0_goal_progress import (
    DEFAULT_D2P_PROPOSAL_SAMPLES,
    DEFAULT_RELINEARIZATION_STEPS,
    _accepted_backtracking_index,
    _bootstrap_mean,
    _d2p_program,
    _p1_admission,
    _restart_d2p_goal_progress,
    _screened_proposal_d2p_goal_progress,
    _validate_outcome_calibration,
    _verify_execution_identity,
)
from experiments.spectral_reachability.reacher_d2p_context import WARMUP_POLICY


def test_d2p_p0_bootstrap_reports_mean_and_interval():
    interval = _bootstrap_mean(np.asarray([-1.0, 0.0, 1.0]), seed=7, draws=100)
    assert len(interval) == 2
    assert interval[0] <= 0.0 <= interval[1]


def test_d2p_default_preserves_the_validated_multilinearization_protocol():
    assert DEFAULT_RELINEARIZATION_STEPS == 4
    assert DEFAULT_D2P_PROPOSAL_SAMPLES == 128
    assert d2p.DEFAULT_D2P_INITIALIZATION == "screened_gaussian_proposals"


def test_d2p_p0_execution_identity_rejects_different_planner_root():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    receipt = _verify_execution_identity(image, image, image.copy(), image.copy())
    assert len(receipt["reset_rgb_sha256"]) == 64
    with pytest.raises(RuntimeError, match="reset RGB"):
        _verify_execution_identity(image + 1, image, image, image)
    with pytest.raises(RuntimeError, match="goal RGB"):
        _verify_execution_identity(image, image + 1, image, image)


def test_d2p_outcome_requires_matching_disjoint_damping_selection(tmp_path):
    receipt = tmp_path / "selection.json"
    receipt.write_text(json.dumps({
        "schema": "d2p_reacher_damping_selection.v1", "status": "pass",
        "selected": {"damping": 0.1}, "calibration_root_seeds": [10, 11],
        "calibration_contract": {
            "action_parameterization": "tanh mapped to environment raw action box after checkpoint normalization",
            "runner": d2p._sha256_file(d2p.Path(d2p.__file__)),
            "relinearization_steps": 1, "d2p_restarts": 1,
            "d2p_initialization": "gaussian_restarts",
            "d2p_proposal_samples": 32, "d2p_proposal_elites": 4,
            "context_mode": "executed_warmup", "rollout_root_conditioning": "observed_post_warmup_rgb_anchor", "execution_root_reconstruction": "evaluator_only_observed_root_snapshot_restore", "warmup_policy": WARMUP_POLICY,
        },
    }))
    binding = _validate_outcome_calibration(
        (receipt,), damping=0.1, relinearization_steps=1, d2p_restarts=1,
        d2p_initialization="gaussian_restarts", d2p_proposal_samples=32,
        d2p_proposal_elites=4, outcome_seeds={20, 21},
    )
    assert binding["calibration_root_seeds"] == [10, 11]
    with pytest.raises(ValueError, match="differs"):
        _validate_outcome_calibration((receipt,), damping=1.0, relinearization_steps=1, d2p_restarts=1, d2p_initialization="gaussian_restarts", d2p_proposal_samples=32, d2p_proposal_elites=4, outcome_seeds={20, 21})
    with pytest.raises(ValueError, match="disjoint"):
        _validate_outcome_calibration((receipt,), damping=0.1, relinearization_steps=1, d2p_restarts=1, d2p_initialization="gaussian_restarts", d2p_proposal_samples=32, d2p_proposal_elites=4, outcome_seeds={11, 21})
    with pytest.raises(ValueError, match="relinearization"):
        _validate_outcome_calibration((receipt,), damping=0.1, relinearization_steps=2, d2p_restarts=1, d2p_initialization="gaussian_restarts", d2p_proposal_samples=32, d2p_proposal_elites=4, outcome_seeds={20, 21})


def test_d2p_outcome_rejects_restart_count_different_from_calibration(tmp_path):
    receipt = tmp_path / "selection.json"
    receipt.write_text(json.dumps({
        "schema": "d2p_reacher_damping_selection.v1", "status": "pass",
        "selected": {"damping": 0.1}, "calibration_root_seeds": [10, 11],
        "calibration_contract": {
            "action_parameterization": "tanh mapped to environment raw action box after checkpoint normalization",
            "runner": d2p._sha256_file(d2p.Path(d2p.__file__)),
            "relinearization_steps": 1, "d2p_restarts": 4,
            "d2p_initialization": "gaussian_restarts",
            "d2p_proposal_samples": 32, "d2p_proposal_elites": 4,
            "context_mode": "executed_warmup", "rollout_root_conditioning": "observed_post_warmup_rgb_anchor", "execution_root_reconstruction": "evaluator_only_observed_root_snapshot_restore", "warmup_policy": WARMUP_POLICY,
        },
    }))
    with pytest.raises(ValueError, match="restart"):
        _validate_outcome_calibration((receipt,), damping=0.1, relinearization_steps=1, d2p_restarts=1, d2p_initialization="gaussian_restarts", d2p_proposal_samples=32, d2p_proposal_elites=4, outcome_seeds={20, 21})


def test_d2p_trust_region_selects_nonascending_actual_model_cost():
    costs = torch.tensor([1.2, 0.8, 1.0, 1.1])
    assert _accepted_backtracking_index(costs, cost_before=1.0) == 1
    with pytest.raises(RuntimeError, match="preserve"):
        _accepted_backtracking_index(torch.tensor([1.2, 1.1]), cost_before=1.0)


def test_d2p_relinearization_recomputes_and_accepts_each_iteration(monkeypatch):
    def terminal_batch(_model, _pixels, _goal, _history, params, **_kwargs):
        return params[:, 0, :1]

    monkeypatch.setattr(d2p, "_terminal_batch", terminal_batch)
    monkeypatch.setattr(d2p, "_goal_latent", lambda _model, _goal: torch.tensor([0.75]))
    params, receipt = d2p._d2p_goal_progress(
        object(), torch.empty(1), torch.empty(1), torch.zeros(1, 3, 1),
        future_horizon=1, action_lower=torch.tensor([-10.0]), action_upper=torch.tensor([10.0]),
        damping=1.0, relinearization_steps=2,
    )

    assert params.shape == (1, 1, 1)
    assert receipt["relinearization_steps_requested"] == 2
    assert receipt["relinearization_steps_executed"] == 2
    assert receipt["actual_model_cost_nonincreasing"] is True
    assert receipt["relinearization_trace"][1]["predicted_cost_before"] == pytest.approx(
        receipt["relinearization_trace"][0]["predicted_cost_after"], abs=1.0e-6
    )


def test_d2p_restarts_keep_zero_baseline_and_select_by_frozen_cost(monkeypatch):
    def terminal_batch(_model, _pixels, _goal, _history, params, **_kwargs):
        return params[:, 0, :1]

    monkeypatch.setattr(d2p, "_terminal_batch", terminal_batch)
    monkeypatch.setattr(d2p, "_goal_latent", lambda _model, _goal: torch.tensor([0.75]))
    params, receipt = _restart_d2p_goal_progress(
        object(), torch.empty(1), torch.empty(1), torch.zeros(1, 3, 1),
        future_horizon=1, action_lower=torch.tensor([-10.0]), action_upper=torch.tensor([10.0]),
        damping=1.0, relinearization_steps=2, restarts=3, seed=11,
    )

    assert params.shape == (1, 1, 1)
    assert receipt["d2p_restarts"] == 3
    assert receipt["restart_initializations"] == ["zero", "provided", "provided"]
    assert receipt["restart_predicted_terminal_costs"][receipt["selected_restart"]] == pytest.approx(
        min(receipt["restart_predicted_terminal_costs"])
    )


def test_d2p_terminal_is_anchored_at_observed_post_warmup_root_not_predicted_root():
    class Model:
        def encode(self, info):
            return {"emb": info["pixels"]}

        @staticmethod
        def action_encoder(actions):
            return actions

        @staticmethod
        def predict(embeddings, actions):
            return embeddings[:, -1:] + actions[:, -1:]

    # The final RGB-context slot represents the actual post-warmup root (=3),
    # not a model-predicted successor of the prior three warmup observations.
    terminal = d2p._terminal_batch(
        Model(), torch.tensor([[[1.0], [2.0], [3.0]]]), torch.empty(1),
        torch.tensor([[[10.0], [20.0], [30.0]]]),
        torch.atanh(torch.tensor([[[0.4], [0.2]]])),
        action_lower=torch.tensor([-1.0]), action_upper=torch.tensor([1.0]),
    )
    assert terminal.shape == (1, 1)
    assert terminal.item() == pytest.approx(3.6)


def test_d2p_screened_proposals_keep_zero_and_refine_only_frozen_elites(monkeypatch):
    def terminal_batch(_model, _pixels, _goal, _history, params, **_kwargs):
        return params[:, 0, :1]

    monkeypatch.setattr(d2p, "_terminal_batch", terminal_batch)
    monkeypatch.setattr(d2p, "_goal_latent", lambda _model, _goal: torch.tensor([0.75]))
    params, receipt = _screened_proposal_d2p_goal_progress(
        object(), torch.empty(1), torch.empty(1), torch.zeros(1, 3, 1),
        future_horizon=1, action_lower=torch.tensor([-10.0]), action_upper=torch.tensor([10.0]),
        damping=1.0, relinearization_steps=2, proposals=5, elites=2, seed=11,
    )

    assert params.shape == (1, 1, 1)
    assert receipt["initialization_strategy"] == "screened_gaussian_proposals"
    assert receipt["proposal_count"] == 5
    assert receipt["proposal_elites"] == 2
    assert len(receipt["proposal_terminal_costs"]) == 5
    assert len(receipt["proposal_selected_indices"]) == 2
    assert receipt["refined_terminal_costs"][receipt["proposal_selected_indices"].index(receipt["selected_proposal_index"])] == pytest.approx(
        min(receipt["refined_terminal_costs"])
    )


def test_d2p_program_rejects_unknown_initialization():
    with pytest.raises(ValueError, match="unknown"):
        _d2p_program(
            object(), torch.empty(1), torch.empty(1), torch.zeros(1, 3, 1),
            future_horizon=1, action_lower=torch.tensor([-1.0]), action_upper=torch.tensor([1.0]),
            damping=1.0, relinearization_steps=1, restarts=1,
            initialization="not-a-strategy", proposal_samples=1, proposal_elites=1, seed=0,
        )


def test_d2p_p1_requires_affirmative_paired_outcome_not_absence_of_no_go():
    assert _p1_admission({"ci95": [0.10, 0.50]}, calibration_only=False)[0] is False
    assert _p1_admission({"ci95": [-0.50, -0.10]}, calibration_only=False)[0] is True
    assert _p1_admission({"ci95": [-0.50, -0.10]}, calibration_only=True)[0] is False
