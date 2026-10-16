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


def load_capacity_readout(path: Path) -> dict[str, torch.Tensor | bool | str | float]:
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
        if action_frame != "raw":
            raise ValueError(
                "runtime capacity readout requires action_frame='raw'; "
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
        if not 0.0 <= artifact["correction_multiplier"] <= 1.0:
            raise ValueError("runtime correction_multiplier must be in [0, 1]")
        if not 0.0 < artifact["base_rank_quantile"] <= 1.0:
            raise ValueError("runtime base_rank_quantile must be in (0, 1]")
        if artifact["residual_start_iteration"] < 0:
            raise ValueError("runtime residual_start_iteration must be nonnegative")
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
    artifact: dict[str, torch.Tensor | bool | str | float], path: Path,
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


class CapacityReadoutCostModel(torch.nn.Module):
    """Apply a train-only action/residual readout to frozen RGB predictions."""

    def __init__(
        self,
        base: torch.nn.Module,
        artifact: dict[str, torch.Tensor | bool | str | float],
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

    @staticmethod
    def _readout(
        features: torch.Tensor,
        coefficient: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        scaled = (features - mean.to(features)) / scale.to(features)
        ones = torch.ones(*scaled.shape[:-1], 1, device=scaled.device, dtype=scaled.dtype)
        return torch.cat([scaled, ones], dim=-1) @ coefficient.to(features)

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
        if self.gate_enabled and "_candidate_bank_mask" not in info_dict:
            return base_cost
        raw_candidates = (
            action_candidates * self.action_raw_std.to(action_candidates)
            + self.action_raw_mean.to(action_candidates)
        )
        action_summary = capacity_action_summary(raw_candidates)
        source = info_dict["emb"][:, :, -1, :]
        predicted = info_dict["predicted_emb"][:, :, -1, :]
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
