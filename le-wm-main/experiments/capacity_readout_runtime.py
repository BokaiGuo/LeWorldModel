"""Runtime-safe train-only capacity readout for frozen RGB world models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def capacity_action_summary(action_candidates: torch.Tensor) -> torch.Tensor:
    """Convert raw flattened action blocks to the 12-D train-time summary."""
    if action_candidates.ndim != 4 or action_candidates.shape[-1] % 2:
        raise ValueError("capacity readout expects [B,N,H,F] with even F")
    # The frozen checkpoint receives H flattened frame-skip blocks.  The
    # train-time ``_action_summary`` sees every low-level frame (H * block),
    # not one average action per block; preserving that ordering is essential
    # for a deployable readout with the same feature semantics.
    sequence = action_candidates.reshape(
        *action_candidates.shape[:3], action_candidates.shape[-1] // 2, 2,
    ).reshape(
        *action_candidates.shape[:2], -1, 2,
    )
    midpoint = max(1, sequence.shape[2] // 2)
    second_half = sequence[:, :, midpoint:] if midpoint < sequence.shape[2] else sequence[:, :, -1:]
    return torch.cat(
        [
            sequence.mean(dim=2),
            sequence.std(dim=2),
            sequence[:, :, 0],
            sequence[:, :, -1],
            sequence[:, :, :midpoint].mean(dim=2),
            second_half.mean(dim=2),
        ],
        dim=-1,
    )


def proposal_relative_action_summary(action_summary: torch.Tensor) -> torch.Tensor:
    """Normalize each CEM proposal cloud without using task outcomes."""
    if action_summary.ndim != 3 or action_summary.shape[-1] != 12:
        raise ValueError("proposal-relative summary expects [B, N, 12]")
    if action_summary.shape[1] < 2:
        raise ValueError("proposal-relative summary requires at least two candidates")
    mean = action_summary.mean(dim=1, keepdim=True)
    scale = action_summary.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    return (action_summary - mean) / scale


def load_capacity_readout(path: Path) -> dict[str, torch.Tensor | bool | str | float | int]:
    """Load and validate an audit-produced deployable raw-action artifact."""
    required = {
        "action_coefficient", "action_mean", "action_scale",
        "residual_coefficient", "residual_mean", "residual_scale",
        "gate_enabled", "action_frame", "feature_family",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"capacity readout is missing arrays: {missing}")
        action_frame = str(np.asarray(data["action_frame"]).reshape(-1)[0])
        feature_family = str(np.asarray(data["feature_family"]).reshape(-1)[0])
        if action_frame not in ("raw", "proposal_relative"):
            raise ValueError(
                "runtime capacity readout requires action_frame='raw' or "
                "'proposal_relative'; "
                f"got {action_frame!r}"
            )
        if feature_family != "source_delta_action":
            raise ValueError(
                "runtime capacity readout requires feature_family='source_delta_action'; "
                f"got {feature_family!r}"
            )
        artifact: dict[str, torch.Tensor | bool | str] = {
            key: torch.as_tensor(np.asarray(data[key]).copy(), dtype=torch.float32)
            for key in (
                "action_coefficient", "action_mean", "action_scale",
                "residual_coefficient", "residual_mean", "residual_scale",
            )
        }
        gate = int(np.asarray(data["gate_enabled"]).reshape(-1)[0])
        if gate not in (0, 1):
            raise ValueError(f"gate_enabled must be 0 or 1, got {gate}")
        artifact["gate_enabled"] = bool(gate)
        artifact["action_frame"] = action_frame
        artifact["feature_family"] = feature_family
        artifact["support_scope"] = str(
            np.asarray(data["support_scope"]).reshape(-1)[0]
        ) if "support_scope" in data.files else "explicit_candidate_bank_only"
        if artifact["support_scope"] not in (
            "explicit_candidate_bank_only",
            "full_cem_final_stage",
        ):
            raise ValueError(
                "runtime capacity readout has unsupported support_scope: "
                f"{artifact['support_scope']!r}"
            )
        if "goal_conditioning" in data.files:
            artifact["goal_conditioning"] = str(
                np.asarray(data["goal_conditioning"]).reshape(-1)[0]
            )
        else:
            artifact["goal_conditioning"] = "none"
        artifact["base_cost_correction"] = bool(
            int(np.asarray(data["base_cost_correction"]).reshape(-1)[0])
        ) if "base_cost_correction" in data.files else False
        artifact["action_correction_enabled"] = bool(
            int(np.asarray(data["action_correction_enabled"]).reshape(-1)[0])
        ) if "action_correction_enabled" in data.files else True
        artifact["correction_multiplier"] = float(
            np.asarray(data["correction_multiplier"]).reshape(-1)[0]
        ) if "correction_multiplier" in data.files else 1.0
        artifact["base_rank_quantile"] = float(
            np.asarray(data["base_rank_quantile"]).reshape(-1)[0]
        ) if "base_rank_quantile" in data.files else 1.0
        artifact["residual_start_iteration"] = int(
            np.asarray(data["residual_start_iteration"]).reshape(-1)[0]
        ) if "residual_start_iteration" in data.files else 0
        if "bilinear_source_components" in data.files:
            bilinear_tensor_fields = (
                "bilinear_no_dynamics_coefficient",
                "bilinear_no_dynamics_mean",
                "bilinear_no_dynamics_scale",
                "bilinear_full_coefficient",
                "bilinear_full_mean",
                "bilinear_full_scale",
                "bilinear_source_mean",
                "bilinear_source_components",
                "bilinear_delta_mean",
                "bilinear_delta_components",
            )
            missing_bilinear = sorted(set(bilinear_tensor_fields).difference(data.files))
            if missing_bilinear:
                raise ValueError(
                    f"capacity readout is missing low-rank bilinear arrays: {missing_bilinear}"
                )
            for key in bilinear_tensor_fields:
                artifact[key] = torch.as_tensor(
                    np.asarray(data[key]).copy(), dtype=torch.float32,
                )
            artifact["bilinear_selected"] = str(
                np.asarray(data["bilinear_selected"]).reshape(-1)[0]
            ) if "bilinear_selected" in data.files else "action_only"
            artifact["bilinear_gate_enabled"] = bool(
                int(np.asarray(data["bilinear_gate_enabled"]).reshape(-1)[0])
            ) if "bilinear_gate_enabled" in data.files else False
            artifact["bilinear_rank"] = int(
                np.asarray(data["bilinear_rank"]).reshape(-1)[0]
            ) if "bilinear_rank" in data.files else int(
                artifact["bilinear_source_components"].shape[0]
            )
        mlp_gate = bool(
            int(np.asarray(data["mlp_no_dynamics_gate_enabled"]).reshape(-1)[0])
        ) if "mlp_no_dynamics_gate_enabled" in data.files else False
        artifact["mlp_no_dynamics_gate_enabled"] = mlp_gate
        artifact["mlp_residual_start_iteration"] = int(
            np.asarray(data["mlp_residual_start_iteration"]).reshape(-1)[0]
        ) if "mlp_residual_start_iteration" in data.files else 0
        artifact["mlp_planner_call_index"] = int(
            np.asarray(data["mlp_planner_call_index"]).reshape(-1)[0]
        ) if "mlp_planner_call_index" in data.files else 0
        artifact["proposal_support_min_fraction"] = float(
            np.asarray(data["proposal_support_min_fraction"]).reshape(-1)[0]
        ) if "proposal_support_min_fraction" in data.files else 0.0
        artifact["mlp_base_cost_blend_weight"] = float(
            np.asarray(data["mlp_base_cost_blend_weight"]).reshape(-1)[0]
        ) if "mlp_base_cost_blend_weight" in data.files else 0.0
        artifact["mlp_shuffle_base_cost_control"] = bool(
            int(np.asarray(data["mlp_shuffle_base_cost_control"]).reshape(-1)[0])
        ) if "mlp_shuffle_base_cost_control" in data.files else False
        artifact["mlp_base_cost_elite_k"] = int(
            np.asarray(data["mlp_base_cost_elite_k"]).reshape(-1)[0]
        ) if "mlp_base_cost_elite_k" in data.files else 1
        if "mlp_no_dynamics_feature_mean" in data.files:
            mlp_fields = (
                "mlp_no_dynamics_feature_mean",
                "mlp_no_dynamics_feature_scale",
                "mlp_no_dynamics_residual_mean",
                "mlp_no_dynamics_residual_scale",
                "mlp_no_dynamics_layer_0_weight",
                "mlp_no_dynamics_layer_0_bias",
                "mlp_no_dynamics_layer_2_weight",
                "mlp_no_dynamics_layer_2_bias",
                "mlp_no_dynamics_layer_4_weight",
                "mlp_no_dynamics_layer_4_bias",
                "proposal_support_mean",
                "proposal_support_scale",
                "proposal_support_radius_sq",
            )
            missing_mlp = sorted(set(mlp_fields).difference(data.files))
            if missing_mlp:
                raise ValueError(
                    f"capacity readout is missing deployment MLP arrays: {missing_mlp}"
                )
            for key in mlp_fields:
                artifact[key] = torch.as_tensor(
                    np.asarray(data[key]).copy(), dtype=torch.float32,
                )
        elif mlp_gate:
            raise ValueError("enabled no-dynamics MLP gate has no deployment arrays")
        if not 0.0 <= artifact["correction_multiplier"] <= 1.0:
            raise ValueError("runtime correction_multiplier must be in [0, 1]")
        if not 0.0 < artifact["base_rank_quantile"] <= 1.0:
            raise ValueError("runtime base_rank_quantile must be in (0, 1]")
        if artifact["residual_start_iteration"] < 0:
            raise ValueError("runtime residual_start_iteration must be nonnegative")
        if not 0.0 <= artifact["proposal_support_min_fraction"] <= 1.0:
            raise ValueError("proposal_support_min_fraction must be in [0, 1]")
        if not 0.0 <= artifact["mlp_base_cost_blend_weight"] <= 1.0:
            raise ValueError("mlp_base_cost_blend_weight must be in [0, 1]")
        if artifact["mlp_base_cost_elite_k"] < 1:
            raise ValueError("mlp_base_cost_elite_k must be positive")
        if (
            artifact["mlp_shuffle_base_cost_control"]
            and artifact["mlp_base_cost_blend_weight"] <= 0.0
            and artifact["mlp_base_cost_elite_k"] <= 1
        ):
            raise ValueError("shuffled base-cost control requires blend or elite tie-break")
        if (
            artifact["mlp_base_cost_blend_weight"] > 0.0
            and artifact["mlp_base_cost_elite_k"] > 1
        ):
            raise ValueError("base-cost blend and elite tie-break are mutually exclusive")
        if artifact["goal_conditioning"] not in ("none", "latent_relative"):
            raise ValueError(
                "runtime capacity readout has unsupported goal_conditioning: "
                f"{artifact['goal_conditioning']!r}"
            )
    _validate_capacity_shapes(artifact, path)
    return artifact


def load_indexed_canonical_readout(path: Path) -> dict:
    """Load a train-only candidate-identity readout.

    The artifact deliberately stores the action summaries and their support
    tolerance.  Runtime code may therefore apply the readout only to an
    explicitly annotated candidate-bank member whose action identity is
    recovered from the frozen train-derived bank.  Gaussian proposals are
    handled by the caller's frozen base cost.
    """
    required = {
        "coefficient", "feature_mean", "feature_scale",
        "candidate_ids", "candidate_action_summaries", "support_tolerance",
        "action_frame", "feature_family",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"indexed readout is missing arrays: {missing}")
        action_frame = str(np.asarray(data["action_frame"]).reshape(-1)[0])
        feature_family = str(np.asarray(data["feature_family"]).reshape(-1)[0])
        if action_frame != "raw_control":
            raise ValueError(
                "indexed runtime readout requires action_frame='raw_control'; "
                f"got {action_frame!r}"
            )
        if feature_family != "source_delta_indexed_train_action":
            raise ValueError(
                "indexed runtime readout requires feature_family="
                "'source_delta_indexed_train_action'; "
                f"got {feature_family!r}"
            )
        artifact = {
            "coefficient": torch.as_tensor(
                np.asarray(data["coefficient"]).copy(), dtype=torch.float32,
            ),
            "feature_mean": torch.as_tensor(
                np.asarray(data["feature_mean"]).copy(), dtype=torch.float32,
            ),
            "feature_scale": torch.as_tensor(
                np.asarray(data["feature_scale"]).copy(), dtype=torch.float32,
            ),
            "candidate_ids": np.asarray(data["candidate_ids"]).astype(str),
            "candidate_action_summaries": torch.as_tensor(
                np.asarray(data["candidate_action_summaries"]).copy(),
                dtype=torch.float32,
            ),
            "support_tolerance": float(
                np.asarray(data["support_tolerance"]).reshape(-1)[0]
            ),
            "action_frame": action_frame,
            "feature_family": feature_family,
            "cost_mode": str(
                np.asarray(data["cost_mode"]).reshape(-1)[0]
            ) if "cost_mode" in data.files else "bank_rank_affine",
            "goal_conditioning": str(
                np.asarray(data["goal_conditioning"]).reshape(-1)[0]
            ) if "goal_conditioning" in data.files else "none",
            # Missing gate metadata is treated as disabled.  Older indexed
            # artifacts were exploratory train-only fits and must not silently
            # become live controllers after this safety contract is introduced.
            "gate_enabled": bool(
                int(np.asarray(data["gate_enabled"]).reshape(-1)[0])
            ) if "gate_enabled" in data.files else False,
        }
        if "candidate_bank_hash" in data.files:
            artifact["candidate_bank_hash"] = str(
                np.asarray(data["candidate_bank_hash"]).reshape(-1)[0]
            )
    _validate_indexed_readout_shapes(artifact, path)
    return artifact


def _validate_indexed_readout_shapes(artifact: dict, path: Path) -> None:
    coefficient = artifact["coefficient"]
    feature_mean = artifact["feature_mean"]
    feature_scale = artifact["feature_scale"]
    summaries = artifact["candidate_action_summaries"]
    candidate_ids = artifact["candidate_ids"]
    tensors = (coefficient, feature_mean, feature_scale, summaries)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError(f"{path}: indexed numeric fields must be tensors")
    if feature_mean.ndim != 1 or feature_scale.shape != feature_mean.shape:
        raise ValueError(f"{path}: indexed feature statistics have incompatible shapes")
    if coefficient.shape != (feature_mean.numel() + 1,):
        raise ValueError(
            f"{path}: indexed coefficient shape {tuple(coefficient.shape)}"
        )
    if summaries.ndim != 2 or summaries.shape[1] != 12:
        raise ValueError(
            f"{path}: candidate action summaries must have shape [N, 12], "
            f"got {tuple(summaries.shape)}"
        )
    if len(candidate_ids) != summaries.shape[0] or len(set(candidate_ids.tolist())) != len(candidate_ids):
        raise ValueError(f"{path}: candidate ids and summaries are not one-to-one")
    if torch.any(feature_scale <= 0) or not np.isfinite(artifact["support_tolerance"]):
        raise ValueError(f"{path}: indexed feature scales/tolerance are invalid")
    if artifact["support_tolerance"] <= 0:
        raise ValueError(f"{path}: indexed support_tolerance must be positive")
    if artifact["cost_mode"] != "bank_rank_affine":
        raise ValueError(
            f"{path}: unsupported indexed cost_mode {artifact['cost_mode']!r}"
        )
    if artifact["goal_conditioning"] not in ("none", "latent_relative"):
        raise ValueError(
            f"{path}: unsupported indexed goal_conditioning "
            f"{artifact['goal_conditioning']!r}"
        )
    if not isinstance(artifact["gate_enabled"], bool):
        raise ValueError(f"{path}: indexed gate_enabled must be boolean")


def _validate_capacity_shapes(
    artifact: dict[str, torch.Tensor | bool | str | float | int], path: Path,
) -> None:
    action_mean = artifact["action_mean"]
    action_scale = artifact["action_scale"]
    action_coefficient = artifact["action_coefficient"]
    residual_mean = artifact["residual_mean"]
    residual_scale = artifact["residual_scale"]
    residual_coefficient = artifact["residual_coefficient"]
    tensors = (action_mean, action_scale, action_coefficient,
               residual_mean, residual_scale, residual_coefficient)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError(f"{path}: readout numeric fields must be tensors")
    assert isinstance(action_mean, torch.Tensor)
    assert isinstance(action_scale, torch.Tensor)
    assert isinstance(action_coefficient, torch.Tensor)
    assert isinstance(residual_mean, torch.Tensor)
    assert isinstance(residual_scale, torch.Tensor)
    assert isinstance(residual_coefficient, torch.Tensor)
    if action_mean.ndim != 1 or action_scale.shape != action_mean.shape:
        raise ValueError(f"{path}: action feature statistics have incompatible shapes")
    if action_coefficient.shape != (action_mean.numel() + 1,):
        raise ValueError(f"{path}: action coefficient shape {tuple(action_coefficient.shape)}")
    if residual_mean.ndim != 1 or residual_scale.shape != residual_mean.shape:
        raise ValueError(f"{path}: residual feature statistics have incompatible shapes")
    if residual_coefficient.shape != (residual_mean.numel() + 1,):
        raise ValueError(
            f"{path}: residual coefficient shape {tuple(residual_coefficient.shape)}"
        )
    if torch.any(action_scale <= 0) or torch.any(residual_scale <= 0):
        raise ValueError(f"{path}: readout feature scales must be positive")
    if "bilinear_source_components" in artifact:
        source_components = artifact["bilinear_source_components"]
        delta_components = artifact["bilinear_delta_components"]
        source_mean = artifact["bilinear_source_mean"]
        delta_mean = artifact["bilinear_delta_mean"]
        if not all(isinstance(value, torch.Tensor) for value in (
            source_components, delta_components, source_mean, delta_mean,
        )):
            raise TypeError(f"{path}: bilinear PCA fields must be tensors")
        assert isinstance(source_components, torch.Tensor)
        assert isinstance(delta_components, torch.Tensor)
        assert isinstance(source_mean, torch.Tensor)
        assert isinstance(delta_mean, torch.Tensor)
        if source_components.ndim != 2 or delta_components.shape != source_components.shape:
            raise ValueError(f"{path}: source/delta PCA components must have matching rank-2 shapes")
        rank, latent_width = source_components.shape
        if rank < 1 or source_mean.shape != (latent_width,) or delta_mean.shape != (latent_width,):
            raise ValueError(f"{path}: bilinear PCA means/components are incompatible")
        if int(artifact.get("bilinear_rank", rank)) != rank:
            raise ValueError(f"{path}: bilinear_rank does not match PCA components")
        no_dynamics_width = 12 + rank * 12
        full_width = no_dynamics_width + rank + rank * 12 + rank * rank
        for prefix, width in (
            ("bilinear_no_dynamics", no_dynamics_width),
            ("bilinear_full", full_width),
        ):
            coefficient = artifact[f"{prefix}_coefficient"]
            mean = artifact[f"{prefix}_mean"]
            scale = artifact[f"{prefix}_scale"]
            if not all(isinstance(value, torch.Tensor) for value in (coefficient, mean, scale)):
                raise TypeError(f"{path}: {prefix} fields must be tensors")
            assert isinstance(coefficient, torch.Tensor)
            assert isinstance(mean, torch.Tensor)
            assert isinstance(scale, torch.Tensor)
            if coefficient.shape != (width + 1,) or mean.shape != (width,) or scale.shape != (width,):
                raise ValueError(f"{path}: {prefix} readout shapes are incompatible")
            if torch.any(scale <= 0):
                raise ValueError(f"{path}: {prefix} scales must be positive")
        if artifact.get("bilinear_selected", "action_only") not in (
            "action_only",
            "bilinear_no_dynamics_source_action",
            "bilinear_full_predicted_dynamics",
        ):
            raise ValueError(f"{path}: unsupported bilinear_selected value")
    if "mlp_no_dynamics_feature_mean" in artifact:
        feature_mean = artifact["mlp_no_dynamics_feature_mean"]
        feature_scale = artifact["mlp_no_dynamics_feature_scale"]
        support_mean = artifact["proposal_support_mean"]
        support_scale = artifact["proposal_support_scale"]
        support_radius = artifact["proposal_support_radius_sq"]
        if not all(isinstance(value, torch.Tensor) for value in (
            feature_mean, feature_scale, support_mean, support_scale, support_radius,
        )):
            raise TypeError(f"{path}: deployment MLP statistics must be tensors")
        latent_width = feature_mean.numel() - 12
        if latent_width < 1 or feature_scale.shape != feature_mean.shape:
            raise ValueError(f"{path}: deployment MLP feature statistics are incompatible")
        expected = {
            "mlp_no_dynamics_layer_0_weight": (64, latent_width + 12),
            "mlp_no_dynamics_layer_0_bias": (64,),
            "mlp_no_dynamics_layer_2_weight": (32, 64),
            "mlp_no_dynamics_layer_2_bias": (32,),
            "mlp_no_dynamics_layer_4_weight": (1, 32),
            "mlp_no_dynamics_layer_4_bias": (1,),
        }
        for key, shape in expected.items():
            value = artifact[key]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"{path}: {key} must have shape {shape}")
        if support_mean.shape != (12,) or support_scale.shape != (12,):
            raise ValueError(f"{path}: proposal support must use 12-D action summaries")
        if support_radius.numel() != 1 or float(support_radius.item()) <= 0:
            raise ValueError(f"{path}: proposal support radius must be positive")
        if torch.any(feature_scale <= 0) or torch.any(support_scale <= 0):
            raise ValueError(f"{path}: deployment MLP/support scales must be positive")


class CapacityReadoutCostModel(torch.nn.Module):
    """Apply a train-only action/residual readout to frozen RGB predictions."""

    def __init__(
        self,
        base: torch.nn.Module,
        artifact: dict[str, torch.Tensor | bool | str | float | int],
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ):
        super().__init__()
        self.base = base
        for name in (
            "action_coefficient", "action_mean", "action_scale",
            "residual_coefficient", "residual_mean", "residual_scale",
        ):
            value = artifact[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"capacity artifact field {name!r} is not a tensor")
            self.register_buffer(name, value.detach().clone())
        goal_conditioning = artifact.get("goal_conditioning", "none")
        if goal_conditioning not in ("none", "latent_relative"):
            raise ValueError(f"unsupported runtime goal conditioning: {goal_conditioning!r}")
        self.goal_conditioning = str(goal_conditioning)
        action_frame = str(artifact.get("action_frame", "raw"))
        if action_frame not in ("raw", "proposal_relative"):
            raise ValueError(f"unsupported runtime action frame: {action_frame!r}")
        self.action_frame = action_frame
        self.support_scope = str(
            artifact.get("support_scope", "explicit_candidate_bank_only")
        )
        if self.support_scope not in (
            "explicit_candidate_bank_only",
            "full_cem_final_stage",
        ):
            raise ValueError(
                "unsupported runtime capacity support_scope: "
                f"{self.support_scope!r}"
            )
        self.base_cost_correction = bool(artifact.get("base_cost_correction", False))
        self.action_correction_enabled = bool(artifact.get("action_correction_enabled", True))
        self.correction_multiplier = float(artifact.get("correction_multiplier", 1.0))
        self.base_rank_quantile = float(artifact.get("base_rank_quantile", 1.0))
        self.residual_start_iteration = int(artifact.get("residual_start_iteration", 0))
        if not 0.0 <= self.correction_multiplier <= 1.0:
            raise ValueError("correction_multiplier must be in [0, 1]")
        if not 0.0 < self.base_rank_quantile <= 1.0:
            raise ValueError("base_rank_quantile must be in (0, 1]")
        if self.residual_start_iteration < 0:
            raise ValueError("residual_start_iteration must be nonnegative")
        raw_mean = np.asarray(action_mean, dtype=np.float32)
        raw_std = np.asarray(action_std, dtype=np.float32)
        if raw_mean.ndim != 1 or raw_std.shape != raw_mean.shape or raw_mean.size % 2:
            raise ValueError("runtime action statistics must be matching 1-D even-width arrays")
        if np.any(raw_std <= 0):
            raise ValueError("runtime action standard deviations must be positive")
        self.register_buffer("action_raw_mean", torch.as_tensor(raw_mean))
        self.register_buffer("action_raw_std", torch.as_tensor(raw_std))
        self.gate_enabled = bool(artifact["gate_enabled"])
        self.bilinear_gate_enabled = bool(artifact.get("bilinear_gate_enabled", False))
        self.bilinear_selected = str(artifact.get("bilinear_selected", "action_only"))
        self.has_bilinear_readout = "bilinear_source_components" in artifact
        if self.has_bilinear_readout:
            for name in (
                "bilinear_no_dynamics_coefficient",
                "bilinear_no_dynamics_mean",
                "bilinear_no_dynamics_scale",
                "bilinear_full_coefficient",
                "bilinear_full_mean",
                "bilinear_full_scale",
                "bilinear_source_mean",
                "bilinear_source_components",
                "bilinear_delta_mean",
                "bilinear_delta_components",
            ):
                value = artifact[name]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"capacity artifact field {name!r} is not a tensor")
                self.register_buffer(name, value.detach().clone())
        if self.bilinear_gate_enabled and (
            not self.has_bilinear_readout or self.bilinear_selected == "action_only"
        ):
            raise ValueError("enabled bilinear gate requires a selected low-rank readout")
        self.mlp_no_dynamics_gate_enabled = bool(
            artifact.get("mlp_no_dynamics_gate_enabled", False)
        )
        self.mlp_residual_start_iteration = int(
            artifact.get("mlp_residual_start_iteration", 0)
        )
        self.mlp_planner_call_index = int(artifact.get("mlp_planner_call_index", 0))
        self.proposal_support_min_fraction = float(
            artifact.get("proposal_support_min_fraction", 0.0)
        )
        self.mlp_base_cost_blend_weight = float(
            artifact.get("mlp_base_cost_blend_weight", 0.0)
        )
        self.mlp_shuffle_base_cost_control = bool(
            artifact.get("mlp_shuffle_base_cost_control", False)
        )
        self.mlp_base_cost_elite_k = int(
            artifact.get("mlp_base_cost_elite_k", 1)
        )
        if self.mlp_residual_start_iteration < 0 or self.mlp_planner_call_index < 0:
            raise ValueError("MLP residual iteration/call gates must be nonnegative")
        if not 0.0 <= self.proposal_support_min_fraction <= 1.0:
            raise ValueError("proposal support fraction must be in [0, 1]")
        if not 0.0 <= self.mlp_base_cost_blend_weight <= 1.0:
            raise ValueError("MLP base-cost blend weight must be in [0, 1]")
        if self.mlp_base_cost_elite_k < 1:
            raise ValueError("MLP base-cost elite k must be positive")
        if (
            self.mlp_shuffle_base_cost_control
            and self.mlp_base_cost_blend_weight <= 0.0
            and self.mlp_base_cost_elite_k <= 1
        ):
            raise ValueError("shuffled base-cost control requires blend or elite tie-break")
        if self.mlp_base_cost_blend_weight > 0.0 and self.mlp_base_cost_elite_k > 1:
            raise ValueError("base-cost blend and elite tie-break are mutually exclusive")
        self.has_mlp_no_dynamics = "mlp_no_dynamics_feature_mean" in artifact
        if self.has_mlp_no_dynamics:
            for name in (
                "mlp_no_dynamics_feature_mean",
                "mlp_no_dynamics_feature_scale",
                "mlp_no_dynamics_residual_mean",
                "mlp_no_dynamics_residual_scale",
                "mlp_no_dynamics_layer_0_weight",
                "mlp_no_dynamics_layer_0_bias",
                "mlp_no_dynamics_layer_2_weight",
                "mlp_no_dynamics_layer_2_bias",
                "mlp_no_dynamics_layer_4_weight",
                "mlp_no_dynamics_layer_4_bias",
                "proposal_support_mean",
                "proposal_support_scale",
                "proposal_support_radius_sq",
            ):
                value = artifact[name]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"capacity artifact field {name!r} is not a tensor")
                self.register_buffer(name, value.detach().clone())
        if self.mlp_no_dynamics_gate_enabled and not self.has_mlp_no_dynamics:
            raise ValueError("enabled no-dynamics MLP gate requires deployment arrays")
        self.bilinear_evaluated_rows = 0
        self.bilinear_bank_argmin_changes = 0
        self.bilinear_bank_beats_gaussian_rows = 0
        self.mlp_evaluated_rows = 0
        self.mlp_abstained_low_support_rows = 0
        self.mlp_active_rows = 0
        self.mlp_supported_candidates = 0
        self.mlp_total_candidates = 0
        self.mlp_support_fraction_sum = 0.0
        self.mlp_support_fraction_min = 1.0
        self.mlp_support_fraction_max = 0.0
        self.mlp_argmin_changes = 0

    def runtime_diagnostics(self) -> dict[str, int | float | str | bool]:
        rows = int(self.bilinear_evaluated_rows)
        return {
            "bilinear_gate_enabled": self.bilinear_gate_enabled,
            "bilinear_selected": self.bilinear_selected,
            "evaluated_candidate_rows": rows,
            "bank_argmin_changes": int(self.bilinear_bank_argmin_changes),
            "bank_argmin_change_rate": (
                float(self.bilinear_bank_argmin_changes / rows) if rows else 0.0
            ),
            "bank_beats_gaussian_rows": int(self.bilinear_bank_beats_gaussian_rows),
            "bank_beats_gaussian_rate": (
                float(self.bilinear_bank_beats_gaussian_rows / rows) if rows else 0.0
            ),
            "mlp_no_dynamics_gate_enabled": self.mlp_no_dynamics_gate_enabled,
            "mlp_residual_start_iteration": self.mlp_residual_start_iteration,
            "mlp_planner_call_index": self.mlp_planner_call_index,
            "mlp_evaluated_rows": int(self.mlp_evaluated_rows),
            "mlp_abstained_low_support_rows": int(
                self.mlp_abstained_low_support_rows
            ),
            "mlp_active_rows": int(self.mlp_active_rows),
            "mlp_min_support_fraction": float(
                self.proposal_support_min_fraction
            ),
            "mlp_base_cost_blend_weight": float(
                self.mlp_base_cost_blend_weight
            ),
            "mlp_shuffle_base_cost_control": bool(
                self.mlp_shuffle_base_cost_control
            ),
            "mlp_base_cost_elite_k": int(self.mlp_base_cost_elite_k),
            "mlp_proposal_support_rate": float(
                self.mlp_supported_candidates / max(self.mlp_total_candidates, 1)
            ),
            "mlp_row_support_fraction_mean": float(
                self.mlp_support_fraction_sum / max(self.mlp_active_rows, 1)
            ),
            "mlp_row_support_fraction_min": float(
                self.mlp_support_fraction_min if self.mlp_active_rows else 0.0
            ),
            "mlp_row_support_fraction_max": float(
                self.mlp_support_fraction_max if self.mlp_active_rows else 0.0
            ),
            "mlp_argmin_changes": int(self.mlp_argmin_changes),
        }

    @staticmethod
    def _readout(
        features: torch.Tensor,
        coefficient: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        scaled = (features - mean.to(features)) / scale.to(features)
        ones = torch.ones(*scaled.shape[:-1], 1, device=scaled.device, dtype=scaled.dtype)
        # ``_fit_ridge`` stores the intercept first: [bias, feature_0, ...].
        # Keeping the bias last here silently permutes every learned weight
        # and was the cause of the apparent Reacher online negative transfer.
        return torch.cat([ones, scaled], dim=-1) @ coefficient.to(features)

    def _bilinear_features(
        self,
        source: torch.Tensor,
        delta: torch.Tensor,
        action_summary: torch.Tensor,
        *,
        include_predicted_dynamics: bool,
    ) -> torch.Tensor:
        source_projected = (
            source - self.bilinear_source_mean.to(source)
        ) @ self.bilinear_source_components.to(source).transpose(0, 1)
        delta_projected = (
            delta - self.bilinear_delta_mean.to(delta)
        ) @ self.bilinear_delta_components.to(delta).transpose(0, 1)
        source_action = (
            source_projected.unsqueeze(-1) * action_summary.unsqueeze(-2)
        ).flatten(start_dim=-2)
        blocks = [action_summary, source_action]
        if include_predicted_dynamics:
            delta_action = (
                delta_projected.unsqueeze(-1) * action_summary.unsqueeze(-2)
            ).flatten(start_dim=-2)
            source_delta = (
                source_projected.unsqueeze(-1) * delta_projected.unsqueeze(-2)
            ).flatten(start_dim=-2)
            blocks.extend([delta_projected, delta_action, source_delta])
        return torch.cat(blocks, dim=-1)

    def _mlp_no_dynamics_residual(self, features: torch.Tensor) -> torch.Tensor:
        scaled = (
            features - self.mlp_no_dynamics_feature_mean.to(features)
        ) / self.mlp_no_dynamics_feature_scale.to(features)
        hidden = torch.nn.functional.silu(
            torch.nn.functional.linear(
                scaled,
                self.mlp_no_dynamics_layer_0_weight.to(features),
                self.mlp_no_dynamics_layer_0_bias.to(features),
            )
        )
        hidden = torch.nn.functional.silu(
            torch.nn.functional.linear(
                hidden,
                self.mlp_no_dynamics_layer_2_weight.to(features),
                self.mlp_no_dynamics_layer_2_bias.to(features),
            )
        )
        normalized = torch.nn.functional.linear(
            hidden,
            self.mlp_no_dynamics_layer_4_weight.to(features),
            self.mlp_no_dynamics_layer_4_bias.to(features),
        ).squeeze(-1)
        return (
            normalized * self.mlp_no_dynamics_residual_scale.to(features)
            + self.mlp_no_dynamics_residual_mean.to(features)
        )

    @staticmethod
    def _rank_affine_bank_cost(
        task_cost: torch.Tensor,
        base_cost: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Map a bank-only task ranking onto the frozen cost scale."""
        count = support_mask.sum(dim=1, keepdim=True)
        safe_count = count.clamp_min(1).to(task_cost)
        task_mean = (task_cost * support_mask).sum(dim=1, keepdim=True) / safe_count
        base_mean = (base_cost * support_mask).sum(dim=1, keepdim=True) / safe_count
        task_var = (((task_cost - task_mean) * support_mask).square()).sum(
            dim=1, keepdim=True,
        ) / safe_count
        base_var = (((base_cost - base_mean) * support_mask).square()).sum(
            dim=1, keepdim=True,
        ) / safe_count
        mapped = base_mean + (task_cost - task_mean) / task_var.sqrt().clamp_min(
            1.0e-6
        ) * base_var.sqrt()
        return torch.where(count >= 2, mapped, base_cost)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        # The frozen model call populates the source/predicted latent tensors.
        base_cost = self.base.get_cost(info_dict, action_candidates)
        if action_candidates.shape[-1] != self.action_raw_mean.numel():
            raise ValueError(
                "runtime action width does not match readout statistics: "
                f"got {action_candidates.shape[-1]}, expected {self.action_raw_mean.numel()}"
            )
        # A gated readout is calibrated on an explicit action support set.
        # Applying it to an unannotated Gaussian CEM proposal silently turns
        # support mismatch into online negative transfer.  The caller must
        # provide the bank mask; otherwise the only valid runtime behavior is
        # the frozen base model.
        if (
            (self.gate_enabled or self.bilinear_gate_enabled)
            and not self.mlp_no_dynamics_gate_enabled
            and "_candidate_bank_mask" not in info_dict
            and self.support_scope != "full_cem_final_stage"
        ):
            return base_cost
        raw_candidates = (
            action_candidates * self.action_raw_std.to(action_candidates)
            + self.action_raw_mean.to(action_candidates)
        )
        action_summary = capacity_action_summary(raw_candidates)
        if self.action_frame == "proposal_relative":
            action_summary = proposal_relative_action_summary(action_summary)
        source = info_dict["emb"][:, :, -1, :]
        predicted = info_dict["predicted_emb"][:, :, -1, :]
        if self.mlp_no_dynamics_gate_enabled:
            iteration = info_dict.get("_cem_iteration")
            planner_call = info_dict.get("_planner_call_index")
            if iteration is None or planner_call is None:
                return base_cost
            if (
                not torch.is_tensor(iteration)
                or not torch.is_tensor(planner_call)
                or tuple(iteration.shape) != tuple(base_cost.shape)
                or tuple(planner_call.shape) != tuple(base_cost.shape)
            ):
                raise ValueError(
                    "_cem_iteration and _planner_call_index must match candidate cost shape"
                )
            active_iteration = (
                iteration.to(base_cost.device) >= self.mlp_residual_start_iteration
            ) & (
                planner_call.to(base_cost.device) == self.mlp_planner_call_index
            )
            support_distance = torch.sum(
                (
                    (action_summary - self.proposal_support_mean.to(action_summary))
                    / self.proposal_support_scale.to(action_summary)
                ).square(),
                dim=-1,
            )
            raw_support_mask = (
                support_distance <= self.proposal_support_radius_sq.to(action_summary)
            ) & active_iteration
            action_cost = self._readout(
                action_summary,
                self.action_coefficient,
                self.action_mean,
                self.action_scale,
            )
            residual = self._mlp_no_dynamics_residual(
                torch.cat([source, action_summary], dim=-1)
            )
            task_cost = action_cost + residual
            support_fraction = raw_support_mask.to(base_cost).mean(dim=1)
            active_rows = active_iteration.any(dim=1)
            valid_rows = (raw_support_mask.sum(dim=1) >= 2) & (
                support_fraction >= self.proposal_support_min_fraction
            )
            self.mlp_abstained_low_support_rows += int(
                (active_rows & ~valid_rows).sum().item()
            )
            active_fraction = support_fraction[active_rows]
            self.mlp_active_rows += int(active_rows.sum().item())
            if active_fraction.numel():
                self.mlp_support_fraction_sum += float(active_fraction.sum().item())
                self.mlp_support_fraction_min = min(
                    self.mlp_support_fraction_min,
                    float(active_fraction.min().item()),
                )
                self.mlp_support_fraction_max = max(
                    self.mlp_support_fraction_max,
                    float(active_fraction.max().item()),
                )
            support_mask = raw_support_mask & valid_rows.unsqueeze(1)
            mapped = self._rank_affine_bank_cost(task_cost, base_cost, support_mask)
            model_component = (
                torch.roll(base_cost, shifts=1, dims=1)
                if self.mlp_shuffle_base_cost_control
                else base_cost
            )
            if self.mlp_base_cost_elite_k > 1:
                elite_k = min(self.mlp_base_cost_elite_k, base_cost.shape[1])
                elite_indices = torch.topk(
                    task_cost.masked_fill(~support_mask, torch.inf),
                    k=elite_k,
                    dim=1,
                    largest=False,
                ).indices
                elite_model_cost = torch.gather(model_component, 1, elite_indices)
                selected_in_elite = torch.argmin(elite_model_cost, dim=1, keepdim=True)
                selected_indices = torch.gather(
                    elite_indices, 1, selected_in_elite
                )
                supported_floor = mapped.masked_fill(~support_mask, torch.inf).amin(
                    dim=1, keepdim=True
                )
                selected_cost = supported_floor - 1.0e-4
                final_supported_cost = mapped.scatter(
                    1, selected_indices, selected_cost
                )
            else:
                final_supported_cost = (
                    (1.0 - self.mlp_base_cost_blend_weight) * mapped
                    + self.mlp_base_cost_blend_weight * model_component
                )
            supported_base = base_cost.masked_fill(~support_mask, torch.inf)
            supported_task = final_supported_cost.masked_fill(~support_mask, torch.inf)
            base_best = torch.argmin(supported_base, dim=1)
            task_best = torch.argmin(supported_task, dim=1)
            self.mlp_evaluated_rows += int(valid_rows.sum().item())
            self.mlp_supported_candidates += int(raw_support_mask.sum().item())
            self.mlp_total_candidates += int(active_iteration.sum().item())
            self.mlp_argmin_changes += int(
                ((base_best != task_best) & valid_rows).sum().item()
            )
            return torch.where(support_mask, final_supported_cost, base_cost)
        if self.bilinear_gate_enabled:
            candidate_bank_mask = info_dict["_candidate_bank_mask"]
            if not torch.is_tensor(candidate_bank_mask):
                raise TypeError("_candidate_bank_mask must be a tensor when provided")
            if tuple(candidate_bank_mask.shape) != tuple(base_cost.shape):
                raise ValueError(
                    "_candidate_bank_mask must match candidate cost shape: "
                    f"got {tuple(candidate_bank_mask.shape)}, expected {tuple(base_cost.shape)}"
                )
            support_mask = candidate_bank_mask.to(device=base_cost.device, dtype=torch.bool)
            include_dynamics = self.bilinear_selected == "bilinear_full_predicted_dynamics"
            features = self._bilinear_features(
                source,
                predicted - source,
                action_summary,
                include_predicted_dynamics=include_dynamics,
            )
            prefix = "bilinear_full" if include_dynamics else "bilinear_no_dynamics"
            task_cost = self._readout(
                features,
                getattr(self, f"{prefix}_coefficient"),
                getattr(self, f"{prefix}_mean"),
                getattr(self, f"{prefix}_scale"),
            )
            mapped = self._rank_affine_bank_cost(task_cost, base_cost, support_mask)
            supported_base = base_cost.masked_fill(~support_mask, torch.inf)
            supported_task = task_cost.masked_fill(~support_mask, torch.inf)
            base_best = torch.argmin(supported_base, dim=1)
            task_best = torch.argmin(supported_task, dim=1)
            self.bilinear_evaluated_rows += int(base_cost.shape[0])
            self.bilinear_bank_argmin_changes += int((base_best != task_best).sum().item())
            unsupported_base = base_cost.masked_fill(support_mask, torch.inf)
            bank_beats_gaussian = mapped.masked_fill(~support_mask, torch.inf).amin(dim=1) < (
                unsupported_base.amin(dim=1)
            )
            self.bilinear_bank_beats_gaussian_rows += int(bank_beats_gaussian.sum().item())
            return torch.where(support_mask, mapped, base_cost)
        latent_features = torch.cat([source, predicted - source, action_summary], dim=-1)
        if self.goal_conditioning == "latent_relative":
            goal_embedding = info_dict["goal_emb"]
            if goal_embedding.ndim == 4:
                goal = goal_embedding[:, :, -1, :]
            elif goal_embedding.ndim in (2, 3):
                # Depending on the stable-worldmodel checkpoint, the goal is
                # either [B,N,D], [B,D], or a temporal [B,T,D] tensor.  In
                # all 3-D cases the candidate axis is already present after
                # the frozen model call; the final feature axis is D.
                goal = goal_embedding
                if goal.ndim == 3 and goal.shape[1] != source.shape[1]:
                    goal = goal[:, -1, :].unsqueeze(1)
            else:
                raise ValueError(
                    "unsupported goal_emb rank for capacity readout: "
                    f"{goal_embedding.ndim}"
                )
            if goal.ndim == 2:
                goal = goal.unsqueeze(1)
            goal = goal.expand_as(source)
            latent_features = torch.cat(
                [latent_features, goal - source, predicted - goal], dim=-1
            )
        action_cost = self._readout(
            action_summary, self.action_coefficient, self.action_mean, self.action_scale,
        )
        if not self.gate_enabled:
            return base_cost if self.base_cost_correction else action_cost
        residual_cost = self._readout(
            latent_features,
            self.residual_coefficient,
            self.residual_mean,
            self.residual_scale,
        )
        if not self.action_correction_enabled:
            action_cost = torch.zeros_like(base_cost)
        correction = self.correction_multiplier * (action_cost + residual_cost)
        # A final-stage residual must not perturb the proposal distribution
        # that generated the later CEM candidates.  The iteration-aware solver
        # supplies this field; absent metadata is treated as unsupported and
        # therefore falls back to the frozen base cost.
        if self.residual_start_iteration > 0:
            iteration = info_dict.get("_cem_iteration")
            if iteration is None:
                return base_cost if self.base_cost_correction else action_cost
            if not torch.is_tensor(iteration) or tuple(iteration.shape) != tuple(base_cost.shape):
                raise ValueError(
                    "_cem_iteration must be a tensor matching candidate cost shape"
                )
            correction = torch.where(
                iteration.to(device=correction.device) >= self.residual_start_iteration,
                correction,
                torch.zeros_like(correction),
            )
        support_mask = None
        # The calibration panel contains only the predeclared candidate bank.
        # CEM also proposes Gaussian candidates at runtime, but those actions
        # have no train/calibration support for this readout.  The solver
        # annotates bank positions with this mask; unsupported Gaussian
        # candidates must remain exactly on the frozen base cost.
        candidate_bank_mask = info_dict.get("_candidate_bank_mask")
        if candidate_bank_mask is not None:
            if not torch.is_tensor(candidate_bank_mask):
                raise TypeError("_candidate_bank_mask must be a tensor when provided")
            if tuple(candidate_bank_mask.shape) != tuple(base_cost.shape):
                raise ValueError(
                    "_candidate_bank_mask must match candidate cost shape: "
                    f"got {tuple(candidate_bank_mask.shape)}, expected {tuple(base_cost.shape)}"
                )
            support_mask = candidate_bank_mask.to(device=correction.device, dtype=torch.bool)
            correction = torch.where(
                support_mask,
                correction,
                torch.zeros_like(correction),
            )
        if self.base_rank_quantile < 1.0:
            if support_mask is None:
                rank_cost = base_cost
            else:
                support_counts = support_mask.sum(dim=1)
                if torch.any(support_counts <= 0):
                    # A solver may intentionally disable the bank after its
                    # support-probe iterations.  In that phase the residual
                    # has no calibrated candidates and must be a no-op.
                    correction = torch.zeros_like(correction)
                    readout_cost = correction
                    return base_cost + readout_cost if self.base_cost_correction else readout_cost
                if torch.any(support_counts != support_counts[0]):
                    raise ValueError("_candidate_bank_mask must select the same number of candidates per batch")
                rank_cost = base_cost.masked_select(support_mask).reshape(base_cost.shape[0], -1)
            cutoff = torch.quantile(
                rank_cost,
                self.base_rank_quantile,
                dim=1,
                keepdim=True,
            )
            correction = torch.where(
                (base_cost <= cutoff) if support_mask is None else (support_mask & (base_cost <= cutoff)),
                correction,
                torch.zeros_like(correction),
            )
        readout_cost = correction
        if self.base_cost_correction:
            # ``correction`` is already zero for unsupported proposals.
            return base_cost + readout_cost
        if support_mask is not None:
            # The readout is calibrated only on the explicit bank.  Off-bank
            # Gaussian proposals must retain the frozen model cost rather than
            # becoming artificial zero-cost candidates.
            return torch.where(support_mask, readout_cost, base_cost)
        return readout_cost


class IndexedCanonicalReadoutCostModel(torch.nn.Module):
    """Use a train-derived candidate identity while failing closed off-bank.

    The readout predicts the train-only task cost for a bank candidate.  Since
    that task cost need not share the frozen latent cost's numerical scale,
    supported bank costs are rank-affinely mapped onto the base-cost scale
    within each candidate set.  This keeps the intervention a ranking change,
    rather than allowing an arbitrary target-scale shift to dominate CEM.
    """

    def __init__(
        self,
        base: torch.nn.Module,
        artifact: dict,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ):
        super().__init__()
        self.base = base
        for name in ("coefficient", "feature_mean", "feature_scale", "candidate_action_summaries"):
            value = artifact[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"indexed artifact field {name!r} must be a tensor")
            self.register_buffer(name, value.detach().clone())
        self.support_tolerance = float(artifact["support_tolerance"])
        self.cost_mode = str(artifact.get("cost_mode", "bank_rank_affine"))
        self.goal_conditioning = str(artifact.get("goal_conditioning", "none"))
        self.gate_enabled = bool(artifact.get("gate_enabled", False))
        if self.cost_mode != "bank_rank_affine":
            raise ValueError(f"unsupported indexed cost_mode {self.cost_mode!r}")
        if self.goal_conditioning not in ("none", "latent_relative"):
            raise ValueError(f"unsupported indexed goal_conditioning {self.goal_conditioning!r}")
        raw_mean = np.asarray(action_mean, dtype=np.float32)
        raw_std = np.asarray(action_std, dtype=np.float32)
        if raw_mean.ndim != 1 or raw_std.shape != raw_mean.shape or raw_mean.size % 2:
            raise ValueError("runtime action statistics must be matching 1-D even-width arrays")
        if np.any(raw_std <= 0):
            raise ValueError("runtime action standard deviations must be positive")
        self.register_buffer("action_raw_mean", torch.as_tensor(raw_mean))
        self.register_buffer("action_raw_std", torch.as_tensor(raw_std))

    @staticmethod
    def _readout(
        features: torch.Tensor,
        coefficient: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        scaled = (features - mean.to(features)) / scale.to(features)
        ones = torch.ones(*scaled.shape[:-1], 1, device=scaled.device, dtype=scaled.dtype)
        # The offline feature-family fitter stores the intercept first.
        return torch.cat([ones, scaled], dim=-1) @ coefficient.to(features)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        base_cost = self.base.get_cost(info_dict, action_candidates)
        if not self.gate_enabled:
            return base_cost
        candidate_bank_mask = info_dict.get("_candidate_bank_mask")
        if candidate_bank_mask is None:
            # A missing support annotation is not evidence of support.
            return base_cost
        if not torch.is_tensor(candidate_bank_mask):
            raise TypeError("_candidate_bank_mask must be a tensor when provided")
        if tuple(candidate_bank_mask.shape) != tuple(base_cost.shape):
            raise ValueError(
                "_candidate_bank_mask must match candidate cost shape: "
                f"got {tuple(candidate_bank_mask.shape)}, expected {tuple(base_cost.shape)}"
            )
        if action_candidates.shape[-1] != self.action_raw_mean.numel():
            raise ValueError(
                "indexed runtime action width does not match readout statistics: "
                f"got {action_candidates.shape[-1]}, expected {self.action_raw_mean.numel()}"
            )
        raw_candidates = (
            action_candidates * self.action_raw_std.to(action_candidates)
            + self.action_raw_mean.to(action_candidates)
        )
        action_summary = capacity_action_summary(raw_candidates)
        bank = self.candidate_action_summaries.to(action_summary)
        distances = torch.linalg.vector_norm(
            action_summary.unsqueeze(-2) - bank.view(1, 1, *bank.shape),
            dim=-1,
        )
        nearest_distance, nearest_index = distances.min(dim=-1)
        supported = candidate_bank_mask.to(device=base_cost.device, dtype=torch.bool)
        supported = supported & (nearest_distance <= self.support_tolerance)
        if not bool(torch.any(supported)):
            return base_cost

        source = info_dict["emb"][:, :, -1, :]
        predicted = info_dict["predicted_emb"][:, :, -1, :]
        matched_summary = bank[nearest_index]
        features = torch.cat([source, predicted - source, matched_summary], dim=-1)
        if self.goal_conditioning == "latent_relative":
            goal_embedding = info_dict["goal_emb"]
            if goal_embedding.ndim == 4:
                goal = goal_embedding[:, :, -1, :]
            elif goal_embedding.ndim == 3:
                goal = goal_embedding
                if goal.shape[1] != source.shape[1]:
                    goal = goal[:, -1, :].unsqueeze(1)
            elif goal_embedding.ndim == 2:
                goal = goal_embedding.unsqueeze(1)
            else:
                raise ValueError(
                    "unsupported goal_emb rank for indexed readout: "
                    f"{goal_embedding.ndim}"
                )
            goal = goal.expand_as(source)
            features = torch.cat([features, goal - source, predicted - goal], dim=-1)
        task_cost = self._readout(
            features, self.coefficient, self.feature_mean, self.feature_scale,
        )

        # Put the train-only task ranking on the frozen base-cost scale.  Do
        # this per batch and over supported bank entries only; Gaussian
        # proposals never influence the calibration of this intervention.
        count = supported.sum(dim=1, keepdim=True)
        safe_count = count.clamp_min(1).to(task_cost)
        task_mean = (task_cost * supported).sum(dim=1, keepdim=True) / safe_count
        base_mean = (base_cost * supported).sum(dim=1, keepdim=True) / safe_count
        task_var = (((task_cost - task_mean) * supported).square()).sum(dim=1, keepdim=True) / safe_count
        base_var = (((base_cost - base_mean) * supported).square()).sum(dim=1, keepdim=True) / safe_count
        task_scale = task_var.sqrt().clamp_min(1.0e-6)
        base_scale = base_var.sqrt()
        mapped = base_mean + (task_cost - task_mean) / task_scale * base_scale
        mapped = torch.where(count >= 2, mapped, base_cost)
        return torch.where(supported, mapped, base_cost)
