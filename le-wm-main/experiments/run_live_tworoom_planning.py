"""Live RGB planning on TwoRoom, independent of the PushT branch.

The environment exposes a state goal and a private rendered target image, while
LeWM's cost model consumes the canonical ``goal`` pixel key.  The wrapper below
copies only that rendered target into the observation info before the standard
World preprocessing.  It never exposes simulator state to the policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Keep the historical ``python experiments/run_*.py`` entry point usable as
# well as the package form.  No experiment logic or import resolution changes
# when the runner is invoked with ``python -m experiments...``.
if __package__ in (None, ""):
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.capacity_readout_runtime import (
    CapacityReadoutCostModel,
    IndexedCanonicalReadoutCostModel,
    capacity_action_summary as _capacity_action_summary,
    load_capacity_readout,
    load_indexed_canonical_readout,
)
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from stable_worldmodel.solver.utils import prepare_init_action


class TwoRoomGoalAliasWrapper(gym.Wrapper):
    """Provide TwoRoom's rendered target as ``info['goal']``."""

    def _goal_image(self) -> np.ndarray:
        image = getattr(self.env.unwrapped, "_target_img", None)
        if image is None:
            raise RuntimeError("TwoRoom did not create a rendered target image")
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))
        return image.astype(np.uint8, copy=True)

    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        info = dict(info)
        info["goal"] = self._goal_image()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["goal"] = self._goal_image()
        return observation, reward, terminated, truncated, info


class CandidateBankCEMSolver(swm.solver.CEMSolver):
    """CEM with a fixed, predeclared candidate bank.

    The bank is injected before model scoring and consumes slots from the same
    ``num_samples`` budget as ordinary Gaussian candidates.  By default it is
    present in every iteration; ``candidate_bank_iterations=1`` gives a
    one-shot support probe and then restores the full Gaussian budget.
    """

    def __init__(self, *args, candidate_bank: torch.Tensor,
                 candidate_bank_iterations: int | None = None,
                 record_candidate_trace: bool = False,
                 final_action_selection: str = "mean", **kwargs):
        super().__init__(*args, **kwargs)
        if candidate_bank.ndim != 3:
            raise ValueError("candidate_bank must have shape [bank, horizon, action_dim]")
        self.candidate_bank = candidate_bank.detach().clone()
        if self.candidate_bank.shape[0] >= self.num_samples:
            raise ValueError("candidate bank must leave at least one Gaussian slot")
        if candidate_bank_iterations is not None and candidate_bank_iterations < 1:
            raise ValueError("candidate_bank_iterations must be positive or None")
        self.candidate_bank_iterations = candidate_bank_iterations
        self.record_candidate_trace = bool(record_candidate_trace)
        self.candidate_trace: list[dict[str, torch.Tensor]] = []
        self.solve_calls = 0
        if final_action_selection not in {"mean", "best"}:
            raise ValueError("final_action_selection must be 'mean' or 'best'")
        self.final_action_selection = final_action_selection

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        planner_call_index = int(self.solve_calls)
        self.solve_calls += 1
        total_envs = len(next(iter(info_dict.values())))
        init_action = prepare_init_action(
            self.model, info_dict, init_action, self.horizon,
            n_envs=total_envs, action_dim=self.action_dim,
        )
        mean, var = self.init_action_distrib(total_envs, init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)
        bank = self.candidate_bank.to(device=self.device, dtype=self.dtype)
        expected = (self.horizon, self.action_dim)
        if tuple(bank.shape[1:]) != expected:
            raise ValueError(f"candidate bank shape {tuple(bank.shape[1:])} != {expected}")
        outputs = {"costs": [], "mean": [], "var": []}
        trace_batches: list[dict[str, torch.Tensor]] = []

        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]
            expanded_infos = {}
            for key, value in info_dict.items():
                value_batch = value[start_idx:end_idx]
                if torch.is_tensor(value_batch):
                    target_dtype = self.dtype if value_batch.is_floating_point() else None
                    value_batch = value_batch.to(device=self.device, dtype=target_dtype).unsqueeze(1).expand(
                        current_bs, self.num_samples, *value_batch.shape[1:]
                    )
                elif isinstance(value_batch, np.ndarray):
                    value_batch = np.repeat(value_batch[:, None, ...], self.num_samples, axis=1)
                expanded_infos[key] = value_batch

            final_cost = None
            final_candidates = None
            final_candidate_costs = None
            final_bank_mask = None
            for _step in range(self.n_steps):
                use_bank = (
                    self.candidate_bank_iterations is None
                    or _step < self.candidate_bank_iterations
                )
                active_bank = bank if use_bank else bank[:0]
                gaussian_count = self.num_samples - 1 - active_bank.shape[0]
                bank_batch = active_bank.unsqueeze(0).expand(current_bs, -1, -1, -1)
                gaussian = torch.randn(
                    current_bs, gaussian_count, self.horizon, self.action_dim,
                    generator=self.torch_gen, device=self.device, dtype=self.dtype,
                )
                gaussian = gaussian * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates = torch.cat([batch_mean.unsqueeze(1), bank_batch, gaussian], dim=1)
                # The residual was calibrated on the fixed action bank, not
                # on Gaussian CEM proposals.  Unsupported proposals use the
                # immutable frozen cost in CapacityReadoutCostModel.
                bank_mask = torch.zeros(
                    current_bs, self.num_samples, dtype=torch.bool, device=self.device,
                )
                bank_mask[:, 1 : 1 + active_bank.shape[0]] = True
                expanded_infos["_candidate_bank_mask"] = bank_mask
                expanded_infos["_cem_iteration"] = torch.full(
                    (current_bs, self.num_samples), _step,
                    dtype=torch.int64, device=self.device,
                )
                expanded_infos["_planner_call_index"] = torch.full(
                    (current_bs, self.num_samples), planner_call_index,
                    dtype=torch.int64, device=self.device,
                )
                costs = self.model.get_cost(expanded_infos, candidates)
                if costs.ndim != 2 or costs.shape != (current_bs, self.num_samples):
                    raise RuntimeError(f"unexpected candidate cost shape: {tuple(costs.shape)}")
                topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
                batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                topk_candidates = candidates[batch_indices, topk_inds]
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)
                if (
                    self.final_action_selection == "best"
                    and _step == self.n_steps - 1
                ):
                    batch_mean = topk_candidates[:, 0]
                final_cost = topk_vals.mean(dim=1).cpu().tolist()
                final_candidates = candidates
                final_candidate_costs = costs
                final_bank_mask = bank_mask
            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_cost or [])
            if self.record_candidate_trace:
                if final_candidates is None or final_candidate_costs is None or final_bank_mask is None:
                    raise RuntimeError("CEM candidate trace was requested but no iteration ran")
                trace_batches.append({
                    "candidates": final_candidates.detach().cpu(),
                    "costs": final_candidate_costs.detach().cpu(),
                    "bank_mask": final_bank_mask.detach().cpu(),
                    **{
                        key: expanded_infos[key].detach().cpu()
                        for key in ("emb", "predicted_emb", "goal_emb")
                        if torch.is_tensor(expanded_infos.get(key))
                    },
                })

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        if self.record_candidate_trace:
            trace_keys = tuple(trace_batches[0])
            self.candidate_trace.append({
                key: torch.cat([batch[key] for batch in trace_batches], dim=0)
                for key in trace_keys
            })
        return outputs


class IterationAwareCEMSolver(swm.solver.CEMSolver):
    """Standard CEM that exposes the current proposal iteration to the cost.

    The sampling/update equations intentionally mirror the upstream solver.
    The only additional behavior is ``info_dict['_cem_iteration']`` so a
    final-stage residual can remain disabled while the base model establishes
    the proposal distribution.
    """

    def __init__(self, *args, record_trace: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.record_trace = bool(record_trace)
        self.trace: list[dict] = []

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        total_envs = len(next(iter(info_dict.values())))
        init_action = prepare_init_action(
            self.model, info_dict, init_action, self.horizon,
            n_envs=total_envs, action_dim=self.action_dim,
        )
        mean, var = self.init_action_distrib(total_envs, init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)
        outputs = {"costs": [], "mean": [], "var": []}
        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]
            expanded_infos = {}
            for key, value in info_dict.items():
                value_batch = value[start_idx:end_idx]
                if torch.is_tensor(value_batch):
                    target_dtype = self.dtype if value_batch.is_floating_point() else None
                    value_batch = value_batch.to(
                        device=self.device, dtype=target_dtype,
                    ).unsqueeze(1).expand(
                        current_bs, self.num_samples, *value_batch.shape[1:]
                    )
                elif isinstance(value_batch, np.ndarray):
                    value_batch = np.repeat(value_batch[:, None, ...], self.num_samples, axis=1)
                expanded_infos[key] = value_batch
            final_cost = None
            for step in range(self.n_steps):
                expanded_infos["_cem_iteration"] = torch.full(
                    (current_bs, self.num_samples), step,
                    dtype=torch.int64, device=self.device,
                )
                candidates = torch.randn(
                    current_bs, self.num_samples, self.horizon, self.action_dim,
                    generator=self.torch_gen, device=self.device, dtype=self.dtype,
                )
                candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates[:, 0] = batch_mean
                costs = self.model.get_cost(expanded_infos, candidates)
                if costs.ndim != 2 or costs.shape != (current_bs, self.num_samples):
                    raise RuntimeError(f"unexpected candidate cost shape: {tuple(costs.shape)}")
                topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)
                row_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)
                topk_candidates = candidates[row_indices, topk_inds]
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)
                final_cost = topk_vals.mean(dim=1).cpu().tolist()
            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_cost or [])
        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        if self.record_trace:
            self.trace.append({
                "actions": outputs["actions"].tolist(),
                "costs": list(outputs["costs"]),
            })
        return outputs


class SpectralCostModel(torch.nn.Module):
    """Keep the frozen predictor but replace terminal utility by a projection."""

    def __init__(self, base: torch.nn.Module, basis: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("basis", basis.detach().clone())

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        # The base call performs the immutable RGB encoding and predictor
        # rollout.  Its Euclidean cost is deliberately ignored here.
        self.base.get_cost(info_dict, action_candidates)
        predicted = info_dict["predicted_emb"]
        goal = info_dict["goal_emb"][..., -1:, :]
        if goal.ndim == 3:
            goal = goal.unsqueeze(1)
        diff = predicted[:, :, -1:, :] - goal
        projected = diff @ self.basis.to(device=diff.device, dtype=diff.dtype)
        return projected.square().sum(dim=(-1, -2))


class PhysicalReadoutCostModel(torch.nn.Module):
    """Use a frozen train-split latent-to-physical readout as terminal cost."""

    def __init__(self, base: torch.nn.Module, coefficient: torch.Tensor,
                 feature_mean: torch.Tensor, feature_scale: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("coefficient", coefficient.detach().clone())
        self.register_buffer("feature_mean", feature_mean.detach().clone())
        self.register_buffer("feature_scale", feature_scale.detach().clone())

    def _readout(self, value: torch.Tensor) -> torch.Tensor:
        scaled = (value - self.feature_mean.to(value)) / self.feature_scale.to(value)
        ones = torch.ones(*scaled.shape[:-1], 1, device=value.device, dtype=value.dtype)
        return torch.cat([scaled, ones], dim=-1) @ self.coefficient.to(value)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        self.base.get_cost(info_dict, action_candidates)
        predicted = info_dict["predicted_emb"][:, :, -1, :]
        goal = info_dict["goal_emb"][..., -1, :]
        if goal.ndim == 2:
            goal = goal.unsqueeze(1)
        predicted_physical = self._readout(predicted)
        goal_physical = self._readout(goal)
        return (predicted_physical - goal_physical).square().sum(dim=-1)


class ActionConditionedReadoutCostModel(torch.nn.Module):
    """Use action-conditioned successor readout for predicted terminal states."""

    def __init__(self, base: torch.nn.Module, action_readout: tuple[torch.Tensor, ...],
                 goal_readout: tuple[torch.Tensor, ...]):
        super().__init__()
        self.base = base
        for prefix, values in (("action", action_readout), ("goal", goal_readout)):
            self.register_buffer(f"{prefix}_coefficient", values[0].detach().clone())
            self.register_buffer(f"{prefix}_mean", values[1].detach().clone())
            self.register_buffer(f"{prefix}_scale", values[2].detach().clone())

    def _readout(self, value: torch.Tensor, prefix: str) -> torch.Tensor:
        mean = getattr(self, f"{prefix}_mean").to(value)
        scale = getattr(self, f"{prefix}_scale").to(value)
        coefficient = getattr(self, f"{prefix}_coefficient").to(value)
        scaled = (value - mean) / scale
        ones = torch.ones(*scaled.shape[:-1], 1, device=value.device, dtype=value.dtype)
        return torch.cat([scaled, ones], dim=-1) @ coefficient

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        self.base.get_cost(info_dict, action_candidates)
        predicted = info_dict["predicted_emb"][:, :, -1, :]
        goal = info_dict["goal_emb"][..., -1, :]
        if goal.ndim == 2:
            goal = goal.unsqueeze(1)
        # The export and the frozen action encoder both use a ten-dimensional
        # normalized five-frame action block for the first predicted transition.
        first_block = action_candidates[:, :, 0, :]
        predicted_features = torch.cat([predicted, first_block], dim=-1)
        predicted_physical = self._readout(predicted_features, "action")
        goal_physical = self._readout(goal, "goal")
        return (predicted_physical - goal_physical).square().sum(dim=-1)


class LatentActionRankerCostModel(torch.nn.Module):
    """Apply a calibration-only latent/action utility readout.

    The base LeWM encoder and predictor remain frozen.  The ranker was fitted
    on root-disjoint simulator labels and is intentionally kept separate from
    the model checkpoint.  Its feature construction mirrors
    ``audit_tworoom_latent_action_ranker.py``: normalized base cost, the raw
    first macro-action, predicted-goal latent difference, and its contextual
    interactions.
    """

    def __init__(
        self,
        base: torch.nn.Module,
        coefficient: torch.Tensor,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ):
        super().__init__()
        self.base = base
        self.register_buffer("coefficient", coefficient.detach().clone())
        self.register_buffer("feature_mean", feature_mean.detach().clone())
        self.register_buffer("feature_scale", feature_scale.detach().clone())
        self.register_buffer(
            "action_mean", torch.as_tensor(action_mean, dtype=torch.float32).clone(),
        )
        self.register_buffer(
            "action_std", torch.as_tensor(action_std, dtype=torch.float32).clone(),
        )

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        base_cost = self.base.get_cost(info_dict, action_candidates)
        predicted = info_dict["predicted_emb"][:, :, -1, :]
        goal = info_dict["goal_emb"][..., -1, :]
        if goal.ndim == 2:
            goal = goal.unsqueeze(1)
        goal = goal.expand_as(predicted)
        current = info_dict["emb"][:, :, -1, :]
        delta = predicted - goal
        context = current - goal
        z_cost = (base_cost - base_cost.mean(dim=1, keepdim=True)) / base_cost.std(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        if action_candidates.shape[-1] % 2:
            raise ValueError("latent action ranker expects an even flattened action block")
        block_frames = action_candidates.shape[-1] // 2
        raw_macro = action_candidates[:, :, 0, :].reshape(
            *action_candidates.shape[:2], block_frames, 2,
        )
        mean = self.action_mean.to(action_candidates).reshape(1, 1, -1, 2)
        std = self.action_std.to(action_candidates).reshape(1, 1, -1, 2)
        if mean.shape[2] != block_frames:
            raise ValueError(
                "ranker action statistics do not match candidate block: "
                f"stats={mean.shape[2]}, candidates={block_frames}"
            )
        raw_xy = (raw_macro * std + mean).mean(dim=2)
        features = torch.cat([
            z_cost.unsqueeze(-1),
            raw_xy,
            delta,
            delta.square(),
            delta * context,
        ], dim=-1)
        scaled = (features - self.feature_mean.to(features)) / self.feature_scale.to(features)
        ones = torch.ones(*scaled.shape[:-1], 1, device=scaled.device, dtype=scaled.dtype)
        return torch.cat([scaled, ones], dim=-1) @ self.coefficient.to(features)


def _candidate_bank(
    action_mean: np.ndarray,
    action_std: np.ndarray,
    horizon: int,
    action_block: int,
) -> torch.Tensor:
    """Build fixed raw-action macro candidates, then normalize them."""
    if action_mean.size != action_std.size or action_mean.size % action_block:
        raise ValueError("action statistics are incompatible with action_block")
    raw_dim = action_mean.size // action_block
    if raw_dim != 2:
        raise ValueError(f"TwoRoom candidate bank expects 2-D actions, got {raw_dim}")
    directions = [(0.0, 0.0)]
    for magnitude in (0.25, 0.50, 0.75):
        directions.extend([
            (magnitude, 0.0), (-magnitude, 0.0),
            (0.0, magnitude), (0.0, -magnitude),
        ])
    directions.extend([(0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)])
    sequences: list[np.ndarray] = []
    for dx, dy in directions:
        frame = np.asarray([dx, dy], dtype=np.float32)
        block = np.tile(frame, (action_block, 1)).reshape(-1)
        normalized_block = (block - action_mean) / np.maximum(action_std, 1e-6)
        sequences.append(np.tile(normalized_block[None, :], (horizon, 1)))
    return torch.as_tensor(np.stack(sequences), dtype=torch.float32)


def _load_spectral_basis(path: Path, dimension: int) -> torch.Tensor:
    with np.load(path) as data:
        latents = np.asarray(data["latents"], dtype=np.float64)
    if latents.ndim != 2 or latents.shape[1] < dimension:
        raise ValueError(f"spectral basis requires at least {dimension} latent dimensions")
    centered = latents - latents.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvectors = np.linalg.eigh(covariance)[1][:, ::-1].copy()
    return torch.as_tensor(eigenvectors[:, :dimension].copy(), dtype=torch.float32)


def _load_physical_readout(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with np.load(path) as data:
        required = {"coefficient", "feature_mean", "feature_scale"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"readout artifact is missing arrays: {missing}")
        return tuple(torch.as_tensor(np.asarray(data[key]).copy(), dtype=torch.float32) for key in (
            "coefficient", "feature_mean", "feature_scale",
        ))


def _load_latent_action_ranker(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        required = {"coefficient", "feature_mean", "feature_scale"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"latent action ranker is missing arrays: {missing}")
        return tuple(
            torch.as_tensor(np.asarray(data[key]).copy(), dtype=torch.float32)
            for key in ("coefficient", "feature_mean", "feature_scale")
        )


def _image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ]
    )


def _load_action_stats(path: Path, environment: str):
    payload = json.loads(path.read_text())
    rows = [row for row in payload["reports"] if row["environment"] == environment]
    if len(rows) != 1:
        raise ValueError(f"expected one report for {environment}, got {len(rows)}")
    row = rows[0]
    return (
        np.asarray(row["calibration_action_mean"], dtype=np.float32),
        np.asarray(row["calibration_action_std"], dtype=np.float32),
    )


def _planning_policy(args, device: str, action_mean: np.ndarray, action_std: np.ndarray):
    model = load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    # TwoRoom's frozen checkpoint was trained with a fixed action-token block.
    # Reject incompatible short blocks before CEM starts; otherwise the failure
    # appears deep inside Conv1d as an opaque channel-mismatch exception.
    patch_embed = getattr(getattr(model, "action_encoder", None), "patch_embed", None)
    expected_channels = getattr(patch_embed, "in_channels", None)
    if expected_channels is not None and expected_channels % 2 == 0:
        expected_action_block = int(expected_channels // 2)
        if int(args.action_block) != expected_action_block:
            raise ValueError(
                "incompatible action_block for frozen TwoRoom checkpoint: "
                f"got {args.action_block}, expected {expected_action_block} "
                f"(action encoder input channels={expected_channels})"
            )
    residual_start_iteration = 0
    if args.cost_metric == "spectral_top_32":
        if args.spectral_basis is None:
            raise ValueError("--spectral-basis is required for spectral_top_32")
        model = SpectralCostModel(
            model,
            _load_spectral_basis(Path(args.spectral_basis), dimension=32),
        )
    elif args.cost_metric == "physical_readout":
        if args.readout_artifact is None:
            raise ValueError("--readout-artifact is required for physical_readout")
        coefficient, feature_mean, feature_scale = _load_physical_readout(
            Path(args.readout_artifact)
        )
        model = PhysicalReadoutCostModel(model, coefficient, feature_mean, feature_scale)
    elif args.cost_metric == "action_conditioned_readout":
        if args.readout_artifact is None or args.goal_readout_artifact is None:
            raise ValueError(
                "--readout-artifact and --goal-readout-artifact are required "
                "for action_conditioned_readout"
            )
        action_readout = _load_physical_readout(Path(args.readout_artifact))
        goal_readout = _load_physical_readout(Path(args.goal_readout_artifact))
        model = ActionConditionedReadoutCostModel(model, action_readout, goal_readout)
    elif args.cost_metric == "latent_action_ranker":
        if args.ranker_artifact is None:
            raise ValueError("--ranker-artifact is required for latent_action_ranker")
        coefficient, feature_mean, feature_scale = _load_latent_action_ranker(
            Path(args.ranker_artifact)
        )
        model = LatentActionRankerCostModel(
            model, coefficient, feature_mean, feature_scale,
            action_mean, action_std,
        )
    elif args.cost_metric == "capacity_safe_readout":
        if args.capacity_readout_artifact is None:
            raise ValueError(
                "--capacity-readout-artifact is required for capacity_safe_readout"
            )
        artifact = load_capacity_readout(Path(args.capacity_readout_artifact))
        residual_start_iteration = int(artifact.get("residual_start_iteration", 0))
        if args.capacity_correction_multiplier is not None:
            if not 0.0 <= args.capacity_correction_multiplier <= 1.0:
                raise ValueError("--capacity-correction-multiplier must be in [0, 1]")
            artifact["correction_multiplier"] = float(args.capacity_correction_multiplier)
        if args.capacity_base_rank_quantile is not None:
            if not 0.0 < args.capacity_base_rank_quantile <= 1.0:
                raise ValueError("--capacity-base-rank-quantile must be in (0, 1]")
            artifact["base_rank_quantile"] = float(args.capacity_base_rank_quantile)
        model = CapacityReadoutCostModel(model, artifact, action_mean, action_std)
    elif args.cost_metric == "indexed_canonical_readout":
        if args.indexed_canonical_readout_artifact is None:
            raise ValueError(
                "--indexed-canonical-readout-artifact is required for "
                "indexed_canonical_readout"
            )
        if not args.candidate_bank:
            raise ValueError(
                "indexed_canonical_readout requires --candidate-bank so runtime "
                "support is explicit"
            )
        artifact = load_indexed_canonical_readout(
            Path(args.indexed_canonical_readout_artifact)
        )
        model = IndexedCanonicalReadoutCostModel(
            model, artifact, action_mean, action_std,
        )
    if args.candidate_bank and residual_start_iteration > 0:
        raise ValueError(
            "final-stage capacity residual requires the standard Gaussian CEM "
            "solver; candidate-bank mode does not expose iteration metadata"
        )
    if args.candidate_bank:
        solver_class = CandidateBankCEMSolver
    elif residual_start_iteration > 0:
        solver_class = IterationAwareCEMSolver
    else:
        solver_class = IterationAwareCEMSolver if args.record_solver_trace else swm.solver.CEMSolver
    solver_kwargs = {}
    if solver_class is IterationAwareCEMSolver:
        solver_kwargs["record_trace"] = args.record_solver_trace
    if args.candidate_bank:
        solver_kwargs["candidate_bank"] = _candidate_bank(
            action_mean, action_std, args.horizon, args.action_block,
        )
        solver_kwargs["candidate_bank_iterations"] = args.candidate_bank_iterations
    solver = solver_class(
        model=model,
        batch_size=1,
        num_samples=args.num_samples,
        var_scale=1.0,
        n_steps=args.n_steps,
        topk=args.topk,
        device=device,
        seed=args.seed,
        **solver_kwargs,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(
            horizon=args.horizon,
            receding_horizon=args.receding_horizon,
            history_len=3,
            action_block=args.action_block,
            warm_start=True,
        ),
        process={"action": ActionNormalizer(action_mean, action_std)},
        transform={"pixels": _image_transform(), "goal": _image_transform()},
    )
    return policy


def _run(args, policy_kind: str, action_mean: np.ndarray, action_std: np.ndarray):
    if policy_kind == "random":
        policy = swm.policy.RandomPolicy(seed=args.seed)
    else:
        policy = _planning_policy(args, args.device, action_mean, action_std)
    world = swm.World(
        env_name="swm/TwoRoom-v1",
        num_envs=args.episodes,
        image_shape=(224, 224),
        max_episode_steps=args.max_episode_steps,
        goal_conditioned=True,
        pre_wrappers=[TwoRoomGoalAliasWrapper],
        render_mode="rgb_array",
    )
    try:
        world.set_policy(policy)
        # Do not auto-reset early successes: both arms must consume the same
        # initial seed set for a valid paired consequence comparison.
        result = world.evaluate(
            episodes=args.episodes,
            seed=args.seed,
            reset_mode="wait",
        )
        if policy_kind != "random" and args.record_solver_trace:
            result["cem_trace"] = list(getattr(policy.solver, "trace", []))
    finally:
        world.close()
    successes = np.asarray(result["episode_successes"], dtype=np.int64)
    output = {
        "success_rate": float(successes.mean()),
        "episode_successes": successes.tolist(),
        "seeds": np.asarray(result["seeds"], dtype=np.int64).tolist(),
    }
    if "cem_trace" in result:
        output["cem_trace"] = result["cem_trace"]
    return output


def _canonicalize(result: dict) -> dict:
    """Sort outcomes by seed because wait-mode reports completion order."""
    order = np.argsort(np.asarray(result["seeds"], dtype=np.int64))
    result = dict(result)
    result["seeds"] = np.asarray(result["seeds"], dtype=np.int64)[order].tolist()
    result["episode_successes"] = np.asarray(result["episode_successes"], dtype=np.int64)[order].tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/tworoom/lewm")
    parser.add_argument(
        "--adapter-artifact",
        type=Path,
        default=Path("results/live_cross_environment_prediction_20261015/tworoom_cuda/cross_environment_prediction.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--candidate-bank", action="store_true", help="inject a fixed TwoRoom macro-action bank into CEM")
    parser.add_argument(
        "--candidate-bank-iterations", type=int, default=None,
        help="number of initial CEM iterations containing the fixed bank; None means every iteration",
    )
    parser.add_argument(
        "--cost-metric",
        choices=(
            "euclidean", "spectral_top_32", "physical_readout",
            "action_conditioned_readout", "latent_action_ranker",
            "capacity_safe_readout", "indexed_canonical_readout",
        ),
        default="euclidean",
    )
    parser.add_argument("--spectral-basis", type=Path, default=None)
    parser.add_argument("--readout-artifact", type=Path, default=None)
    parser.add_argument("--goal-readout-artifact", type=Path, default=None)
    parser.add_argument("--ranker-artifact", type=Path, default=None)
    parser.add_argument("--capacity-readout-artifact", type=Path, default=None)
    parser.add_argument(
        "--indexed-canonical-readout-artifact", type=Path, default=None,
    )
    parser.add_argument(
        "--capacity-correction-multiplier", type=float, default=None,
        help="frozen ablation override; do not tune from live outcome data",
    )
    parser.add_argument(
        "--capacity-base-rank-quantile", type=float, default=None,
        help="frozen ablation override; do not tune from live outcome data",
    )
    parser.add_argument(
        "--record-solver-trace", action="store_true",
        help="diagnostic only: record CEM actions/costs for solver comparison",
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for live TwoRoom planning but unavailable")
    if args.topk >= args.num_samples or args.episodes < 2:
        raise ValueError("require episodes >= 2 and topk < num_samples")
    capacity_artifact_start_iteration = None
    indexed_artifact_gate_enabled = None
    if args.cost_metric == "capacity_safe_readout":
        if args.capacity_readout_artifact is None:
            raise ValueError(
                "--capacity-readout-artifact is required for capacity_safe_readout"
            )
        # Read the frozen protocol metadata in the top-level driver as well as
        # inside _planning_policy().  The latter owns model construction, but
        # the former writes the provenance receipt after both paired arms run;
        # keeping this value explicit prevents a local-variable leak from
        # silently discarding an otherwise completed held-out experiment.
        capacity_artifact_start_iteration = int(
            load_capacity_readout(Path(args.capacity_readout_artifact)).get(
                "residual_start_iteration", 0
            )
        )
    if args.cost_metric == "indexed_canonical_readout":
        if args.indexed_canonical_readout_artifact is None:
            raise ValueError(
                "--indexed-canonical-readout-artifact is required for "
                "indexed_canonical_readout"
            )
        if not args.candidate_bank:
            raise ValueError(
                "indexed_canonical_readout requires --candidate-bank so runtime "
                "support is explicit"
            )
        indexed_artifact_gate_enabled = bool(
            load_indexed_canonical_readout(
                Path(args.indexed_canonical_readout_artifact)
            ).get("gate_enabled", False)
        )
    action_mean, action_std = _load_action_stats(args.adapter_artifact, "swm/TwoRoom-v1")
    print("Running live RGB TwoRoom: frozen LeWM CEM versus random, same seeds.")
    if args.cost_metric == "indexed_canonical_readout":
        print(
            "Indexed readout calibration gate: "
            f"{'enabled' if indexed_artifact_gate_enabled else 'disabled; falling back to frozen base cost'}"
        )
    random_result = _canonicalize(_run(args, "random", action_mean, action_std))
    planning_result = _canonicalize(_run(args, "planning", action_mean, action_std))
    if random_result["seeds"] != planning_result["seeds"]:
        raise RuntimeError("random and planning did not evaluate the same seed set")
    random_success = np.asarray(random_result["episode_successes"], dtype=np.int64)
    planning_success = np.asarray(planning_result["episode_successes"], dtype=np.int64)
    paired_delta = planning_success - random_success
    payload = {
        "schema": "live_tworoom_planning.v1",
        "protocol": {
            "checkpoint": args.policy,
            "checkpoint_frozen": True,
            "runtime_input": "rendered RGB current frame and rendered RGB goal only",
            "runtime_state_access": False,
            "same_seed_pairing": True,
            "device": args.device,
            "cem": {"num_samples": args.num_samples, "n_steps": args.n_steps, "topk": args.topk},
            "horizon": args.horizon,
            "receding_horizon": args.receding_horizon,
            "action_block": args.action_block,
            "candidate_bank": bool(args.candidate_bank),
            "candidate_bank_definition": "zero, axis-aligned magnitudes 0.25/0.50/0.75, and four diagonals" if args.candidate_bank else None,
            "candidate_bank_iterations": args.candidate_bank_iterations,
            "residual_support_scope": (
                "predeclared_candidate_bank_only" if args.candidate_bank else "all_candidates"
            ),
            "cost_metric": args.cost_metric,
            "spectral_basis": str(args.spectral_basis) if args.spectral_basis else None,
            "readout_artifact": str(args.readout_artifact) if args.readout_artifact else None,
            "goal_readout_artifact": str(args.goal_readout_artifact) if args.goal_readout_artifact else None,
            "ranker_artifact": str(args.ranker_artifact) if args.ranker_artifact else None,
            "capacity_readout_artifact": (
                str(args.capacity_readout_artifact)
                if args.capacity_readout_artifact else None
            ),
            "indexed_canonical_readout_artifact": (
                str(args.indexed_canonical_readout_artifact)
                if args.indexed_canonical_readout_artifact else None
            ),
            "capacity_correction_multiplier_override": args.capacity_correction_multiplier,
            "capacity_base_rank_quantile_override": args.capacity_base_rank_quantile,
            "capacity_residual_start_iteration": (
                capacity_artifact_start_iteration
            ),
            "indexed_readout_gate_enabled": indexed_artifact_gate_enabled,
            "indexed_readout_gate_fallback": (
                None if indexed_artifact_gate_enabled is None
                else not indexed_artifact_gate_enabled
            ),
        },
        "random": random_result,
        "frozen_lewm": planning_result,
        "paired": {
            "seed_order": random_result["seeds"],
            "delta_lewm_minus_random": paired_delta.tolist(),
            "wins": int(np.sum(paired_delta > 0)),
            "losses": int(np.sum(paired_delta < 0)),
            "ties": int(np.sum(paired_delta == 0)),
            "success_rate_delta": float(paired_delta.mean()),
        },
        "claim_boundary": "Independent live TwoRoom RGB planning consequence; not a universal world-model claim.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "planning.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"random": random_result, "frozen_lewm": planning_result}, indent=2))


if __name__ == "__main__":
    main()
