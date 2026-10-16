import numpy as np
import pytest
import torch

from experiments.capacity_readout_runtime import (
    CapacityReadoutCostModel,
    IndexedCanonicalReadoutCostModel,
    capacity_action_summary,
    load_capacity_readout,
    load_indexed_canonical_readout,
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
    with pytest.raises(ValueError, match="action_frame='raw'"):
        load_capacity_readout(path)


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
    values["action_coefficient"][-1] = 2.0
    values["residual_coefficient"][-1] = 1.0
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
    values["residual_coefficient"][-1] = 1.0
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
    values["residual_coefficient"][-1] = 1.0
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
    values["residual_coefficient"][-1] = 1.0
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
    values["residual_coefficient"][-1] = 1.0
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
