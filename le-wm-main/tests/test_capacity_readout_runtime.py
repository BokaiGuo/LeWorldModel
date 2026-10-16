import numpy as np
import pytest
import torch

from experiments.capacity_readout_runtime import (
    CapacityReadoutCostModel,
    IndexedCanonicalReadoutCostModel,
    capacity_action_summary,
    load_capacity_readout,
    load_indexed_canonical_readout,
    proposal_relative_action_summary,
)


def _arrays(gate: int = 1, action_frame: str = "raw") -> dict[str, np.ndarray]:
    return {
        "action_coefficient": np.zeros(13, dtype=np.float32),
        "action_mean": np.zeros(12, dtype=np.float32),
        "action_scale": np.ones(12, dtype=np.float32),
        "residual_coefficient": np.zeros(397, dtype=np.float32),
        "residual_mean": np.zeros(396, dtype=np.float32),
        "residual_scale": np.ones(396, dtype=np.float32),
        "gate_enabled": np.asarray([gate], dtype=np.int8),
        "action_frame": np.asarray([action_frame]),
        "feature_family": np.asarray(["source_delta_action"]),
        "goal_conditioning": np.asarray(["none"]),
    }


def test_capacity_action_summary_matches_train_time_shape_and_order() -> None:
    candidates = torch.arange(2 * 3 * 5 * 10, dtype=torch.float32).reshape(2, 3, 5, 10)
    summary = capacity_action_summary(candidates)
    assert summary.shape == (2, 3, 12)
    # Every flattened block contains five 2-D actions.  The first summary
    # component is the mean over all low-level horizon/block action frames.
    expected = candidates.reshape(2, 3, 5, 5, 2).reshape(2, 3, 25, 2).mean(dim=2)
    torch.testing.assert_close(summary[..., :2], expected)


def test_capacity_readout_loader_rejects_non_raw_runtime_artifact(tmp_path) -> None:
    path = tmp_path / "goal_relative.npz"
    values = _arrays(action_frame="goal_relative")
    np.savez(path, **values)
    with pytest.raises(ValueError, match="proposal_relative"):
        load_capacity_readout(path)


def test_proposal_relative_runtime_chart_removes_row_affine_shift() -> None:
    base = torch.arange(2 * 3 * 12, dtype=torch.float32).reshape(2, 3, 12)
    shifted = base.clone()
    shifted[1] = 100.0 + 4.0 * shifted[1]
    reference = base.clone()
    reference[1] = base[0]
    actual = proposal_relative_action_summary(shifted)
    expected = proposal_relative_action_summary(reference)
    torch.testing.assert_close(actual, expected)


class _FakeBase(torch.nn.Module):
    def get_cost(self, info_dict, action_candidates):
        batch, candidates = action_candidates.shape[:2]
        info_dict["emb"] = torch.zeros(batch, candidates, 1, 192, device=action_candidates.device)
        info_dict["predicted_emb"] = torch.ones(batch, candidates, 1, 192, device=action_candidates.device)
        info_dict["goal_emb"] = torch.zeros(batch, candidates, 192, device=action_candidates.device)
        return torch.zeros(batch, candidates, device=action_candidates.device)


class _RankedBase(torch.nn.Module):
    def get_cost(self, info_dict, action_candidates):
        batch, candidates = action_candidates.shape[:2]
        info_dict["emb"] = torch.zeros(batch, candidates, 1, 192, device=action_candidates.device)
        info_dict["predicted_emb"] = torch.ones(batch, candidates, 1, 192, device=action_candidates.device)
        info_dict["goal_emb"] = torch.zeros(batch, candidates, 192, device=action_candidates.device)
        return torch.arange(candidates, dtype=action_candidates.dtype, device=action_candidates.device).repeat(batch, 1)


def test_capacity_runtime_uses_raw_width_and_frozen_gate() -> None:
    values = _arrays(gate=1)
    artifact = {key: torch.as_tensor(value) for key, value in values.items() if key not in {"gate_enabled", "action_frame", "feature_family", "goal_conditioning"}}
    artifact.update({"gate_enabled": True, "action_frame": "raw", "feature_family": "source_delta_action", "goal_conditioning": "none"})
    model = CapacityReadoutCostModel(
        _FakeBase(), artifact, np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    actions = torch.zeros(1, 4, 5, 10)
    costs = model.get_cost({}, actions)
    assert costs.shape == (1, 4)
    with pytest.raises(ValueError, match="action width"):
        model.get_cost({}, torch.zeros(1, 4, 5, 2))


def test_capacity_runtime_applies_base_rank_trust_region() -> None:
    values = _arrays(gate=1)
    values["action_coefficient"][0] = 2.0
    values["residual_coefficient"][0] = 1.0
    artifact = {
        key: torch.as_tensor(value)
        for key, value in values.items()
        if key not in {"gate_enabled", "action_frame", "feature_family", "goal_conditioning"}
    }
    artifact.update({
        "gate_enabled": True,
        "action_frame": "raw",
        "feature_family": "source_delta_action",
        "goal_conditioning": "none",
        "base_cost_correction": True,
        "action_correction_enabled": False,
        "correction_multiplier": 0.5,
        "base_rank_quantile": 0.5,
    })
    model = CapacityReadoutCostModel(
        _RankedBase(), artifact, np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    costs = model.get_cost(
        {"_candidate_bank_mask": torch.ones(1, 4, dtype=torch.bool)},
        torch.zeros(1, 4, 5, 10),
    )
    torch.testing.assert_close(costs, torch.tensor([[0.5, 1.5, 2.0, 3.0]]))


def test_capacity_runtime_masks_unsupported_gaussian_candidates() -> None:
    values = _arrays(gate=1)
    values["residual_coefficient"][0] = 1.0
    artifact = {
        key: torch.as_tensor(value)
        for key, value in values.items()
        if key not in {"gate_enabled", "action_frame", "feature_family", "goal_conditioning"}
    }
    artifact.update({
        "gate_enabled": True,
        "action_frame": "raw",
        "feature_family": "source_delta_action",
        "goal_conditioning": "none",
        "base_cost_correction": True,
        "action_correction_enabled": False,
        "correction_multiplier": 0.5,
        "base_rank_quantile": 1.0,
    })
    model = CapacityReadoutCostModel(
        _RankedBase(), artifact, np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    info = {"_candidate_bank_mask": torch.tensor([[False, True, False]])}
    costs = model.get_cost(info, torch.zeros(1, 3, 5, 10))
    torch.testing.assert_close(costs, torch.tensor([[0.0, 1.5, 2.0]]))


def test_capacity_runtime_uses_base_cost_for_unsupported_without_additive_mode() -> None:
    values = _arrays(gate=1)
    values["residual_coefficient"][0] = 1.0
    artifact = {
        key: torch.as_tensor(value)
        for key, value in values.items()
        if key not in {"gate_enabled", "action_frame", "feature_family", "goal_conditioning"}
    }
    artifact.update({
        "gate_enabled": True,
        "action_frame": "raw",
        "feature_family": "source_delta_action",
        "goal_conditioning": "none",
        "base_cost_correction": False,
        "action_correction_enabled": False,
        "correction_multiplier": 0.5,
        "base_rank_quantile": 1.0,
    })
    model = CapacityReadoutCostModel(
        _RankedBase(), artifact, np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    info = {"_candidate_bank_mask": torch.tensor([[False, True, False]])}
    costs = model.get_cost(info, torch.zeros(1, 3, 5, 10))
    torch.testing.assert_close(costs, torch.tensor([[0.0, 0.5, 2.0]]))


def test_capacity_runtime_rejects_malformed_candidate_bank_mask() -> None:
    values = _arrays(gate=1)
    artifact = {
        key: torch.as_tensor(value)
        for key, value in values.items()
        if key not in {"gate_enabled", "action_frame", "feature_family", "goal_conditioning"}
    }
    artifact.update({
        "gate_enabled": True,
        "action_frame": "raw",
        "feature_family": "source_delta_action",
        "goal_conditioning": "none",
        "base_cost_correction": True,
        "action_correction_enabled": False,
    })
    model = CapacityReadoutCostModel(
        _RankedBase(), artifact, np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    info = {"_candidate_bank_mask": torch.ones(1, 2, dtype=torch.bool)}
    with pytest.raises(ValueError, match="candidate_bank_mask"):
        model.get_cost(info, torch.zeros(1, 3, 5, 10))


def test_capacity_runtime_computes_trust_region_cutoff_on_bank_only() -> None:
    values = _arrays(gate=1)
    values["residual_coefficient"][0] = 1.0
    artifact = {
        key: torch.as_tensor(value)
        for key, value in values.items()
        if key not in {"gate_enabled", "action_frame", "feature_family", "goal_conditioning"}
    }
    artifact.update({
        "gate_enabled": True,
        "action_frame": "raw",
        "feature_family": "source_delta_action",
        "goal_conditioning": "none",
        "base_cost_correction": True,
        "action_correction_enabled": False,
        "correction_multiplier": 0.5,
        "base_rank_quantile": 0.5,
    })
    model = CapacityReadoutCostModel(
        _RankedBase(), artifact, np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    info = {"_candidate_bank_mask": torch.tensor([[False, True, True, False]])}
    costs = model.get_cost(info, torch.zeros(1, 4, 5, 10))
    # The bank costs are 1 and 2, so the q=.5 cutoff is 1.5; only cost 1
    # receives the residual.  Gaussian costs 0 and 3 stay frozen.
    torch.testing.assert_close(costs, torch.tensor([[0.0, 1.5, 2.0, 3.0]]))


def _final_stage_artifact() -> dict:
    values = _arrays(gate=1)
    values["residual_coefficient"][0] = 1.0
    artifact = {
        key: torch.as_tensor(value)
        for key, value in values.items()
        if key not in {"gate_enabled", "action_frame", "feature_family", "goal_conditioning"}
    }
    artifact.update({
        "gate_enabled": True,
        "action_frame": "raw",
        "feature_family": "source_delta_action",
        "goal_conditioning": "none",
        "base_cost_correction": True,
        "action_correction_enabled": False,
        "correction_multiplier": 0.5,
        "base_rank_quantile": 1.0,
        "residual_start_iteration": 7,
    })
    return artifact


def _mlp_proposal_artifact() -> dict:
    values = _arrays(gate=0)
    artifact = {
        key: torch.as_tensor(value)
        for key, value in values.items()
        if key not in {"gate_enabled", "action_frame", "feature_family", "goal_conditioning"}
    }
    artifact.update({
        "gate_enabled": False,
        "action_frame": "raw",
        "feature_family": "source_delta_action",
        "goal_conditioning": "none",
        "mlp_no_dynamics_gate_enabled": True,
        "mlp_residual_start_iteration": 7,
        "mlp_planner_call_index": 1,
        "mlp_no_dynamics_feature_mean": torch.zeros(204),
        "mlp_no_dynamics_feature_scale": torch.ones(204),
        "mlp_no_dynamics_residual_mean": torch.tensor(0.0),
        "mlp_no_dynamics_residual_scale": torch.tensor(1.0),
        "mlp_no_dynamics_layer_0_weight": torch.zeros(64, 204),
        "mlp_no_dynamics_layer_0_bias": torch.zeros(64),
        "mlp_no_dynamics_layer_2_weight": torch.zeros(32, 64),
        "mlp_no_dynamics_layer_2_bias": torch.zeros(32),
        "mlp_no_dynamics_layer_4_weight": torch.zeros(1, 32),
        "mlp_no_dynamics_layer_4_bias": torch.zeros(1),
        "proposal_support_mean": torch.zeros(12),
        "proposal_support_scale": torch.ones(12),
        "proposal_support_radius_sq": torch.tensor([100.0]),
    })
    return artifact


def test_no_dynamics_mlp_is_final_stage_and_support_gated() -> None:
    model = CapacityReadoutCostModel(
        _RankedBase(), _mlp_proposal_artifact(),
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    actions = torch.zeros(1, 3, 5, 10)
    actions[0, 2] = 100.0
    early = model.get_cost(
        {
            "_cem_iteration": torch.full((1, 3), 6),
            "_planner_call_index": torch.full((1, 3), 1),
        }, actions,
    )
    torch.testing.assert_close(early, torch.tensor([[0.0, 1.0, 2.0]]))
    final = model.get_cost(
        {
            "_cem_iteration": torch.full((1, 3), 7),
            "_planner_call_index": torch.full((1, 3), 1),
        }, actions,
    )
    torch.testing.assert_close(final, torch.tensor([[0.5, 0.5, 2.0]]))
    diagnostics = model.runtime_diagnostics()
    assert diagnostics["mlp_evaluated_rows"] == 1
    assert diagnostics["mlp_proposal_support_rate"] == pytest.approx(2 / 3)
    assert diagnostics["mlp_active_rows"] == 1
    assert diagnostics["mlp_row_support_fraction_mean"] == pytest.approx(2 / 3)


def test_no_dynamics_mlp_falls_back_when_row_support_is_below_calibration() -> None:
    artifact = _mlp_proposal_artifact()
    artifact.update({
        "mlp_residual_start_iteration": 0,
        "mlp_planner_call_index": 0,
        "proposal_support_min_fraction": 0.75,
        "proposal_support_radius_sq": torch.tensor([0.1]),
    })
    base = _RankedBase()
    model = CapacityReadoutCostModel(
        base, artifact, np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    actions = torch.full((1, 4, 5, 10), 100.0)
    actions[:, :2] = 0.0
    info = {
        "_cem_iteration": torch.zeros(1, 4, dtype=torch.int64),
        "_planner_call_index": torch.zeros(1, 4, dtype=torch.int64),
    }
    expected = base.get_cost(info, actions)
    actual = model.get_cost(info, actions)
    assert torch.equal(actual, expected)
    diagnostics = model.runtime_diagnostics()
    assert diagnostics["mlp_evaluated_rows"] == 0
    assert diagnostics["mlp_abstained_low_support_rows"] == 1


def test_no_dynamics_mlp_can_blend_frozen_model_cost() -> None:
    artifact = _mlp_proposal_artifact()
    artifact["mlp_base_cost_blend_weight"] = 0.5
    model = CapacityReadoutCostModel(
        _RankedBase(), artifact,
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    actions = torch.zeros(1, 3, 5, 10)
    actions[0, 2] = 100.0
    costs = model.get_cost(
        {
            "_cem_iteration": torch.full((1, 3), 7),
            "_planner_call_index": torch.full((1, 3), 1),
        },
        actions,
    )
    torch.testing.assert_close(costs, torch.tensor([[0.25, 0.75, 2.0]]))
    diagnostics = model.runtime_diagnostics()
    assert diagnostics["mlp_base_cost_blend_weight"] == 0.5
    assert diagnostics["mlp_shuffle_base_cost_control"] is False


def test_no_dynamics_mlp_shuffled_model_cost_is_placebo_control() -> None:
    artifact = _mlp_proposal_artifact()
    artifact.update({
        "mlp_base_cost_blend_weight": 0.5,
        "mlp_shuffle_base_cost_control": True,
    })
    model = CapacityReadoutCostModel(
        _RankedBase(), artifact,
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    actions = torch.zeros(1, 3, 5, 10)
    actions[0, 2] = 100.0
    costs = model.get_cost(
        {
            "_cem_iteration": torch.full((1, 3), 7),
            "_planner_call_index": torch.full((1, 3), 1),
        },
        actions,
    )
    torch.testing.assert_close(costs, torch.tensor([[1.25, 0.25, 2.0]]))
    diagnostics = model.runtime_diagnostics()
    assert diagnostics["mlp_shuffle_base_cost_control"] is True


def test_no_dynamics_mlp_model_cost_can_only_tiebreak_safe_elite() -> None:
    artifact = _mlp_proposal_artifact()
    artifact["mlp_base_cost_elite_k"] = 2
    model = CapacityReadoutCostModel(
        _RankedBase(), artifact,
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    actions = torch.zeros(1, 3, 5, 10)
    actions[0, 2] = 100.0
    costs = model.get_cost(
        {
            "_cem_iteration": torch.full((1, 3), 7),
            "_planner_call_index": torch.full((1, 3), 1),
        },
        actions,
    )
    assert int(torch.argmin(costs, dim=1).item()) == 0
    assert costs[0, 2] == 2.0
    assert model.runtime_diagnostics()["mlp_base_cost_elite_k"] == 2


def test_no_dynamics_mlp_shuffled_elite_control_breaks_tiebreak_pairing() -> None:
    artifact = _mlp_proposal_artifact()
    artifact.update({
        "mlp_base_cost_elite_k": 2,
        "mlp_shuffle_base_cost_control": True,
    })
    model = CapacityReadoutCostModel(
        _RankedBase(), artifact,
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    actions = torch.zeros(1, 3, 5, 10)
    actions[0, 2] = 100.0
    costs = model.get_cost(
        {
            "_cem_iteration": torch.full((1, 3), 7),
            "_planner_call_index": torch.full((1, 3), 1),
        },
        actions,
    )
    assert int(torch.argmin(costs, dim=1).item()) == 1


def test_no_dynamics_mlp_rejects_wrong_replan_phase() -> None:
    model = CapacityReadoutCostModel(
        _RankedBase(), _mlp_proposal_artifact(),
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    costs = model.get_cost(
        {
            "_cem_iteration": torch.full((1, 3), 7),
            "_planner_call_index": torch.zeros(1, 3, dtype=torch.int64),
        },
        torch.zeros(1, 3, 5, 10),
    )
    torch.testing.assert_close(costs, torch.tensor([[0.0, 1.0, 2.0]]))


def test_capacity_runtime_keeps_final_stage_residual_off_before_start() -> None:
    model = CapacityReadoutCostModel(
        _RankedBase(), _final_stage_artifact(),
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    info = {"_cem_iteration": torch.zeros(1, 4, dtype=torch.int64)}
    costs = model.get_cost(info, torch.zeros(1, 4, 5, 10))
    torch.testing.assert_close(costs, torch.tensor([[0.0, 1.0, 2.0, 3.0]]))


def test_capacity_runtime_enables_final_stage_residual_at_start() -> None:
    model = CapacityReadoutCostModel(
        _RankedBase(), _final_stage_artifact(),
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    info = {
        "_cem_iteration": torch.tensor([[0, 6, 7, 8]], dtype=torch.int64),
        "_candidate_bank_mask": torch.ones(1, 4, dtype=torch.bool),
    }
    costs = model.get_cost(info, torch.zeros(1, 4, 5, 10))
    torch.testing.assert_close(costs, torch.tensor([[0.0, 1.0, 2.5, 3.5]]))


def test_capacity_runtime_fails_closed_without_support_metadata() -> None:
    model = CapacityReadoutCostModel(
        _RankedBase(), _final_stage_artifact(),
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    costs = model.get_cost({}, torch.zeros(1, 4, 5, 10))
    torch.testing.assert_close(costs, torch.tensor([[0.0, 1.0, 2.0, 3.0]]))


def test_capacity_runtime_fails_closed_without_iteration_metadata() -> None:
    model = CapacityReadoutCostModel(
        _RankedBase(), _final_stage_artifact(),
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    costs = model.get_cost({}, torch.zeros(1, 4, 5, 10))
    torch.testing.assert_close(costs, torch.tensor([[0.0, 1.0, 2.0, 3.0]]))


def test_capacity_readout_uses_offline_intercept_first_layout() -> None:
    features = torch.tensor([[[2.0, 3.0]]])
    coefficient = torch.tensor([1.0, 4.0, 5.0])
    result = CapacityReadoutCostModel._readout(
        features,
        coefficient,
        torch.zeros(2),
        torch.ones(2),
    )
    torch.testing.assert_close(result, torch.tensor([[24.0]]))


def _low_rank_bilinear_artifact() -> dict:
    values = _arrays(gate=0)
    artifact = {
        key: torch.as_tensor(value)
        for key, value in values.items()
        if key not in {"gate_enabled", "action_frame", "feature_family", "goal_conditioning"}
    }
    rank = 2
    no_dynamics_width = 12 + rank * 12
    full_width = no_dynamics_width + rank + rank * 12 + rank * rank
    no_dynamics_coefficient = torch.zeros(no_dynamics_width + 1)
    # Intercept first; rank the bank by its first raw action-summary entry.
    no_dynamics_coefficient[1] = 1.0
    artifact.update({
        "gate_enabled": False,
        "action_frame": "raw",
        "feature_family": "source_delta_action",
        "goal_conditioning": "none",
        "bilinear_gate_enabled": True,
        "bilinear_selected": "bilinear_no_dynamics_source_action",
        "bilinear_no_dynamics_coefficient": no_dynamics_coefficient,
        "bilinear_no_dynamics_mean": torch.zeros(no_dynamics_width),
        "bilinear_no_dynamics_scale": torch.ones(no_dynamics_width),
        "bilinear_full_coefficient": torch.zeros(full_width + 1),
        "bilinear_full_mean": torch.zeros(full_width),
        "bilinear_full_scale": torch.ones(full_width),
        "bilinear_source_mean": torch.zeros(192),
        "bilinear_source_components": torch.eye(192)[:rank],
        "bilinear_delta_mean": torch.zeros(192),
        "bilinear_delta_components": torch.eye(192)[:rank],
        "bilinear_rank": rank,
    })
    return artifact


def test_low_rank_bilinear_readout_rank_maps_bank_and_preserves_gaussian() -> None:
    model = CapacityReadoutCostModel(
        _RankedBase(), _low_rank_bilinear_artifact(),
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    actions = torch.zeros(1, 3, 5, 10)
    actions[0, 1, :, 0::2] = 1.0
    actions[0, 2, :, 1::2] = 1.0
    info = {"_candidate_bank_mask": torch.tensor([[False, True, True]])}
    costs = model.get_cost(info, actions)
    # The readout reverses the supported bank order on the frozen base-cost
    # scale, while the unsupported Gaussian/mean slot remains exactly frozen.
    torch.testing.assert_close(costs, torch.tensor([[0.0, 2.0, 1.0]]))


def test_low_rank_bilinear_readout_fails_closed_without_bank_mask() -> None:
    model = CapacityReadoutCostModel(
        _RankedBase(), _low_rank_bilinear_artifact(),
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    costs = model.get_cost({}, torch.zeros(1, 3, 5, 10))
    torch.testing.assert_close(costs, torch.tensor([[0.0, 1.0, 2.0]]))


def _indexed_artifact() -> dict:
    action_a = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    action_b = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    mean = torch.zeros(396)
    scale = torch.ones(396)
    coefficient = torch.zeros(397)
    # Intercept-first ridge layout; rank by the mean x action component.
    coefficient[1 + 384] = 1.0
    return {
        "coefficient": coefficient,
        "feature_mean": mean,
        "feature_scale": scale,
        "candidate_ids": np.asarray(["candidate_a", "candidate_b"]),
        "candidate_action_summaries": torch.stack([action_a, action_b]),
        "support_tolerance": 1.0e-5,
        "action_frame": "raw_control",
        "feature_family": "source_delta_indexed_train_action",
        "cost_mode": "bank_rank_affine",
        "gate_enabled": True,
    }


def test_indexed_readout_uses_exact_bank_identity_and_falls_back_off_bank() -> None:
    model = IndexedCanonicalReadoutCostModel(
        _RankedBase(), _indexed_artifact(),
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    actions = torch.zeros(1, 3, 5, 10)
    actions[0, 1, :, 0::2] = 1.0
    actions[0, 2, :, 1::2] = 1.0
    info = {"_candidate_bank_mask": torch.tensor([[False, True, True]])}
    costs = model.get_cost(info, actions)
    # The indexed readout reverses the two supported bank candidates, while
    # the unsupported Gaussian/mean candidate remains on the frozen base cost.
    torch.testing.assert_close(costs, torch.tensor([[0.0, 2.0, 1.0]]))


def test_indexed_readout_fails_closed_without_bank_annotation() -> None:
    model = IndexedCanonicalReadoutCostModel(
        _RankedBase(), _indexed_artifact(),
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    costs = model.get_cost({}, torch.zeros(1, 3, 5, 10))
    torch.testing.assert_close(costs, torch.tensor([[0.0, 1.0, 2.0]]))


def test_indexed_readout_fails_closed_when_calibration_gate_is_disabled() -> None:
    artifact = _indexed_artifact()
    artifact["gate_enabled"] = False
    model = IndexedCanonicalReadoutCostModel(
        _RankedBase(), artifact,
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    actions = torch.zeros(1, 3, 5, 10)
    actions[0, 1, :, 0::2] = 1.0
    actions[0, 2, :, 1::2] = 1.0
    info = {"_candidate_bank_mask": torch.tensor([[False, True, True]])}
    costs = model.get_cost(info, actions)
    torch.testing.assert_close(costs, torch.tensor([[0.0, 1.0, 2.0]]))


def test_indexed_readout_rejects_unsupported_bank_action() -> None:
    model = IndexedCanonicalReadoutCostModel(
        _RankedBase(), _indexed_artifact(),
        np.zeros(10, dtype=np.float32), np.ones(10, dtype=np.float32),
    )
    actions = torch.zeros(1, 2, 5, 10)
    actions[0, 1, :, 0::2] = 0.37
    info = {"_candidate_bank_mask": torch.tensor([[False, True]])}
    costs = model.get_cost(info, actions)
    torch.testing.assert_close(costs, torch.tensor([[0.0, 1.0]]))


def test_indexed_readout_loader_enforces_runtime_contract(tmp_path) -> None:
    artifact = _indexed_artifact()
    path = tmp_path / "indexed.npz"
    np.savez_compressed(
        path,
        coefficient=artifact["coefficient"].numpy(),
        feature_mean=artifact["feature_mean"].numpy(),
        feature_scale=artifact["feature_scale"].numpy(),
        candidate_ids=artifact["candidate_ids"],
        candidate_action_summaries=artifact["candidate_action_summaries"].numpy(),
        support_tolerance=np.asarray([artifact["support_tolerance"]]),
        action_frame=np.asarray([artifact["action_frame"]]),
        feature_family=np.asarray([artifact["feature_family"]]),
        cost_mode=np.asarray([artifact["cost_mode"]]),
        gate_enabled=np.asarray([1], dtype=np.int8),
    )
    loaded = load_indexed_canonical_readout(path)
    assert loaded["candidate_ids"].tolist() == ["candidate_a", "candidate_b"]
    assert loaded["cost_mode"] == "bank_rank_affine"
