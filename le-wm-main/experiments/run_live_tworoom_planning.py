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
    apply_capacity_bilinear_mode,
    capacity_action_summary as _capacity_action_summary,
    load_capacity_readout,
    load_indexed_canonical_readout,
)
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from stable_worldmodel.solver.utils import prepare_init_action


def _rgb_red_centroid(image: np.ndarray) -> np.ndarray:
    """Return the visible red marker centroid from one raw RGB frame."""
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"expected one RGB frame, got shape {image.shape}")
    red = image[..., 0].astype(np.float32)
    green = image[..., 1].astype(np.float32)
    blue = image[..., 2].astype(np.float32)
    mask = (red > 160.0) & (red > 1.5 * green) & (red > 1.5 * blue)
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return np.asarray([image.shape[1] / 2.0, image.shape[0] / 2.0], dtype=np.float32)
    return np.asarray([xx.mean(), yy.mean()], dtype=np.float32)


def _raw_frame(value: np.ndarray, index: int = 0) -> np.ndarray:
    """Extract one current/goal RGB frame from World policy info."""
    value = np.asarray(value)
    if value.ndim == 5:
        value = value[index, -1]
    elif value.ndim == 4:
        value = value[index]
    return value


def _rgb_reactive_action(current: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, str]:
    """Pure-RGB one-pulse controller used only by the fallback gate.

    It mirrors the separately audited reactive ceiling: when the markers are
    separated across the visible wall, move toward the visible door first;
    otherwise move toward the goal marker.  No simulator state is read.
    """
    source = _rgb_red_centroid(current)
    target = _rgb_red_centroid(goal)
    # Reuse the independently tested detector at runtime without importing it
    # at module import time (that module imports this runner for the wrapper).
    from experiments.audit_tworoom_rgb_reactive_controller import _door_point

    door, vertical = _door_point(current)
    side_gap = abs(float(target[0] - source[0])) if vertical else abs(float(target[1] - source[1]))
    waypoint = target
    phase = "goal"
    if side_gap > 35.0 and np.linalg.norm(source - door) > 18.0:
        waypoint = door
        phase = "door"
    delta = waypoint - source
    norm = float(np.linalg.norm(delta))
    if norm < 1.0e-6:
        return np.zeros(2, dtype=np.float32), phase
    return (delta / norm).astype(np.float32), phase


def _effect_policy_action(
    reactive: np.ndarray, effect_map: np.ndarray,
) -> np.ndarray | None:
    """Invert a verified local RGB effect map into a bounded task action.

    ``effect_map`` follows the row convention ``action @ map = observed RGB
    displacement``.  The inverse is used only after the caller has passed the
    support/conditioning/residual gate; this helper itself remains a pure
    numerical operation and fails closed on singular or non-progressing maps.
    """
    desired = np.asarray(reactive, dtype=np.float64).reshape(-1)
    mapping = np.asarray(effect_map, dtype=np.float64)
    if desired.shape != (2,) or mapping.shape != (2, 2):
        return None
    if not np.isfinite(desired).all() or not np.isfinite(mapping).all():
        return None
    action = desired @ np.linalg.pinv(mapping)
    if not np.isfinite(action).all():
        return None
    norm = float(np.linalg.norm(action))
    if norm > 1.0:
        action = action / norm
    predicted = action @ mapping
    if not np.isfinite(predicted).all() or float(np.dot(predicted, desired)) <= 0.0:
        return None
    return action.astype(np.float32)


class RGBReactivePolicy:
    """Raw-RGB reactive baseline with the same World evaluation interface."""

    def __init__(self):
        self.env = None
        self.calls = 0

    def set_env(self, env):
        self.env = env

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        n_envs = self.env.num_envs
        actions = np.zeros((n_envs, 2), dtype=np.float32)
        for index in range(n_envs):
            action, _ = _rgb_reactive_action(
                _raw_frame(info_dict["pixels"], index),
                _raw_frame(info_dict["goal"], index),
            )
            actions[index] = action
        self.calls += n_envs
        return actions


class LocalEffectRankedCostModel:
    """Use a verified local RGB effect chart to rank supported candidates."""

    def __init__(self, base_model, action_mean, action_std, blend_weight: float):
        if not 0.0 <= float(blend_weight) <= 1.0:
            raise ValueError("local effect blend weight must be in [0, 1]")
        self.base_model = base_model
        self.action_mean = torch.as_tensor(action_mean, dtype=torch.float32)
        self.action_std = torch.as_tensor(action_std, dtype=torch.float32)
        self.blend_weight = float(blend_weight)
        self.local_rank_rows = 0
        self.local_rank_decisions = 0
        self.local_rank_argmin_changes = 0

    @staticmethod
    def _rank_affine(local_cost, base_cost, support):
        count = support.sum(dim=1, keepdim=True)
        safe = count.to(base_cost).clamp_min(1.0)
        local_mean = (local_cost * support).sum(dim=1, keepdim=True) / safe
        base_mean = (base_cost * support).sum(dim=1, keepdim=True) / safe
        local_var = (((local_cost - local_mean) * support).square()).sum(
            dim=1, keepdim=True,
        ) / safe
        base_var = (((base_cost - base_mean) * support).square()).sum(
            dim=1, keepdim=True,
        ) / safe
        mapped = base_mean + (local_cost - local_mean) * (
            base_var.sqrt() / local_var.sqrt().clamp_min(1.0e-6)
        )
        return torch.where(count >= 2, mapped, base_cost)

    def get_cost(self, info_dict, action_candidates):
        base_cost = self.base_model.get_cost(info_dict, action_candidates)
        local_map = info_dict.get("_local_effect_map")
        local_target = info_dict.get("_local_effect_target")
        local_valid = info_dict.get("_local_effect_valid")
        bank_mask = info_dict.get("_candidate_bank_mask")
        if local_map is None or local_target is None or local_valid is None or bank_mask is None:
            return base_cost
        device = action_candidates.device
        local_map = torch.as_tensor(local_map, device=device, dtype=action_candidates.dtype)
        local_target = torch.as_tensor(local_target, device=device, dtype=action_candidates.dtype)
        local_valid = torch.as_tensor(local_valid, device=device, dtype=torch.bool)
        bank_mask = torch.as_tensor(bank_mask, device=device, dtype=torch.bool)
        if local_map.ndim != 4 or local_target.ndim != 3 or local_valid.ndim != 2:
            raise ValueError("local effect context has invalid rank")
        support = local_valid & bank_mask
        if not bool(support.any()):
            return base_cost
        raw = action_candidates * self.action_std.to(device) + self.action_mean.to(device)
        first_action = raw[:, :, 0, :2]
        predicted_effect = torch.einsum("bni,bnij->bnj", first_action, local_map)
        target_direction = local_target / local_target.norm(
            dim=-1, keepdim=True,
        ).clamp_min(1.0e-6)
        local_cost = -torch.sum(predicted_effect * target_direction, dim=-1)
        mapped = self._rank_affine(local_cost, base_cost, support)
        self.local_rank_rows += int(support.any(dim=1).sum().item())
        blended = (
            (1.0 - self.blend_weight) * base_cost
            + self.blend_weight * mapped
        )
        result = torch.where(support, blended, base_cost)
        active_rows = support.any(dim=1)
        base_argmin = torch.argmin(base_cost, dim=1)
        result_argmin = torch.argmin(result, dim=1)
        self.local_rank_decisions += int(active_rows.sum().item())
        self.local_rank_argmin_changes += int(
            ((base_argmin != result_argmin) & active_rows).sum().item()
        )
        return result

    def runtime_diagnostics(self):
        base = getattr(self.base_model, "runtime_diagnostics", None)
        diagnostics = dict(base() if callable(base) else {})
        diagnostics.update({
            "local_effect_ranker_enabled": True,
            "local_effect_ranker_blend_weight": self.blend_weight,
            "local_effect_ranker_rows": int(self.local_rank_rows),
            "local_effect_ranker_decisions": int(self.local_rank_decisions),
            "local_effect_ranker_argmin_changes": int(self.local_rank_argmin_changes),
            "local_effect_ranker_argmin_change_rate": float(
                self.local_rank_argmin_changes / max(self.local_rank_decisions, 1)
            ),
        })
        return diagnostics


class RiskGatedRGBPolicy:
    """Use frozen LeWM, but veto strongly conflicting actions with RGB evidence.

    The gate is deliberately frozen before held-out evaluation: a model action
    is replaced only when its 2-D direction has cosine below ``conflict_cos``
    with the one-pulse RGB direction, or when its norm is below
    ``model_min_norm``.  The next environment observation is consumed by the
    next call, so this is an online action-effect re-observation protocol.
    """

    def __init__(self, base_policy, conflict_cos: float = 0.0,
                 model_min_norm: float = 0.08,
                 use_effect_map: bool = False,
                 use_effect_policy: bool = False,
                 effect_map_intercept: bool = False,
                 door_phase_reactive: bool = False,
                 phase_transition_fallback: bool = False,
                 use_local_effect_ranker: bool = False):
        self.base_policy = base_policy
        self.conflict_cos = float(conflict_cos)
        self.model_min_norm = float(model_min_norm)
        self.use_effect_map = bool(use_effect_map)
        self.use_effect_policy = bool(use_effect_policy)
        self.effect_map_intercept = bool(effect_map_intercept)
        self.door_phase_reactive = bool(door_phase_reactive)
        self.phase_transition_fallback = bool(phase_transition_fallback)
        self.use_local_effect_ranker = bool(use_local_effect_ranker)
        self.env = None
        self.gate_calls = 0
        self.fallback_count = 0
        self.phases: list[str] = []
        self.cosines: list[float] = []
        self.effect_actions_by_env: list[list[np.ndarray]] = []
        self.effect_deltas_by_env: list[list[np.ndarray]] = []
        self.effect_actions_by_env_phase: list[dict[str, list[np.ndarray]]] = []
        self.effect_deltas_by_env_phase: list[dict[str, list[np.ndarray]]] = []
        self.previous_frames: list[np.ndarray | None] = []
        self.previous_actions: list[np.ndarray | None] = []
        self.previous_phases: list[str | None] = []
        self.effect_policy_count = 0
        self.effect_policy_fallback_count = 0
        self.effect_policy_validation_fallback_count = 0
        self.effect_policy_phase_block_count = 0
        self.gate_calls_by_env: list[int] = []
        self.fallback_counts_by_env: list[int] = []
        self.phase_counts_by_env: list[dict[str, int]] = []

    def set_env(self, env):
        self.env = env
        self.base_policy.set_env(env)
        self.previous_frames = [None for _ in range(int(env.num_envs))]
        self.previous_actions = [None for _ in range(int(env.num_envs))]
        self.effect_actions_by_env = [[] for _ in range(int(env.num_envs))]
        self.effect_deltas_by_env = [[] for _ in range(int(env.num_envs))]
        self.effect_actions_by_env_phase = [
            {"door": [], "goal": []} for _ in range(int(env.num_envs))
        ]
        self.effect_deltas_by_env_phase = [
            {"door": [], "goal": []} for _ in range(int(env.num_envs))
        ]
        self.gate_calls_by_env = [0 for _ in range(int(env.num_envs))]
        self.fallback_counts_by_env = [0 for _ in range(int(env.num_envs))]
        self.phase_counts_by_env = [
            {"door": 0, "goal": 0} for _ in range(int(env.num_envs))
        ]
        self.previous_phases = [None for _ in range(int(env.num_envs))]
        self.effect_policy_count = 0
        self.effect_policy_fallback_count = 0
        self.effect_policy_validation_fallback_count = 0
        self.effect_policy_phase_block_count = 0

    def runtime_diagnostics(self):
        """Expose wrapped solver diagnostics without changing action selection.

        Hybrid runs used to export ``null`` here, which made it impossible to
        verify whether a local ranker actually scored any supported bank rows.
        Keep this method diagnostic-only and namespace the wrapper counters so
        the frozen planner's own counters remain distinguishable.
        """
        base = getattr(self.base_policy, "solver", None)
        model = getattr(base, "model", None)
        diagnostics_fn = getattr(model, "runtime_diagnostics", None)
        diagnostics = dict(diagnostics_fn() if callable(diagnostics_fn) else {})
        diagnostics.update({
            "hybrid_gate_calls": int(self.gate_calls),
            "hybrid_fallback_count": int(self.fallback_count),
            "hybrid_fallback_rate": float(
                self.fallback_count / max(self.gate_calls, 1)
            ),
            "hybrid_local_effect_ranker_enabled": bool(
                self.use_local_effect_ranker
            ),
            "hybrid_effect_policy_phase_block_count": int(
                self.effect_policy_phase_block_count
            ),
        })
        return diagnostics

    def _effect_histories(
        self, env_index: int, phase: str | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if phase is None:
            return self.effect_actions_by_env[env_index], self.effect_deltas_by_env[env_index]
        if phase not in {"door", "goal"}:
            raise ValueError(f"unknown TwoRoom phase: {phase}")
        return (
            self.effect_actions_by_env_phase[env_index][phase],
            self.effect_deltas_by_env_phase[env_index][phase],
        )

    def _effect_map(
        self, env_index: int, phase: str | None = None,
    ) -> tuple[np.ndarray | None, int]:
        actions_history, deltas_history = self._effect_histories(env_index, phase)
        if len(actions_history) < 4:
            return None, 0
        actions = np.asarray(actions_history, dtype=np.float64)
        deltas = np.asarray(deltas_history, dtype=np.float64)
        rank = int(np.linalg.matrix_rank(actions))
        if rank < 2 or not np.isfinite(actions).all() or not np.isfinite(deltas).all():
            return None, rank
        design = (
            np.column_stack([actions, np.ones(len(actions))])
            if self.effect_map_intercept else actions
        )
        mapping, _, _, _ = np.linalg.lstsq(design, deltas, rcond=1.0e-6)
        mapping = mapping[:2]
        return mapping.astype(np.float32) if np.isfinite(mapping).all() else None, rank

    def _effect_map_quality(
        self, env_index: int, phase: str | None = None,
    ) -> dict[str, Any]:
        """Return a frozen support/conditioning gate for a local effect map."""
        actions_history, deltas_history = self._effect_histories(env_index, phase)
        diagnostics: dict[str, Any] = {
            "rows": int(len(actions_history)),
            "rank": 0,
            "condition_number": None,
            "normalized_residual": None,
            "heldout_normalized_residual": None,
            "heldout_usable": False,
            "usable": False,
            "reason": "insufficient_rows",
        }
        # Four rows are enough to fit a map but not enough for a stable
        # deployment decision.  Six rows and rank-two support are a frozen
        # minimum; the remaining gates are dimensionless and outcome-free.
        if len(actions_history) < 6:
            return diagnostics
        actions = np.asarray(actions_history, dtype=np.float64)
        deltas = np.asarray(deltas_history, dtype=np.float64)
        rank = int(np.linalg.matrix_rank(actions))
        diagnostics["rank"] = rank
        if rank < 2 or not np.isfinite(actions).all() or not np.isfinite(deltas).all():
            diagnostics["reason"] = "rank_or_finite_support_failed"
            return diagnostics
        condition = float(np.linalg.cond(actions))
        design = (
            np.column_stack([actions, np.ones(len(actions))])
            if self.effect_map_intercept else actions
        )
        mapping_full, _, _, _ = np.linalg.lstsq(design, deltas, rcond=1.0e-6)
        mapping = mapping_full[:2]
        residual = design @ mapping_full - deltas
        normalized_residual = float(
            np.linalg.norm(residual) / max(np.linalg.norm(deltas), 1.0e-6)
        )
        diagnostics["condition_number"] = condition
        diagnostics["normalized_residual"] = normalized_residual
        if not np.isfinite(condition) or condition > 100.0:
            diagnostics["reason"] = "ill_conditioned_support"
            return diagnostics
        if not np.isfinite(normalized_residual) or normalized_residual > 0.75:
            diagnostics["reason"] = "effect_map_residual_too_large"
            return diagnostics
        # A clean in-sample fit does not certify the next pulse. Reserve the
        # latest two observations as an outcome-free rolling verification set.
        if len(actions_history) < 8:
            diagnostics["reason"] = "insufficient_holdout_rows"
            diagnostics["usable"] = True
            return diagnostics
        train_actions = actions[:-2]
        train_deltas = deltas[:-2]
        heldout_actions = actions[-2:]
        heldout_deltas = deltas[-2:]
        if int(np.linalg.matrix_rank(train_actions)) < 2:
            diagnostics["reason"] = "heldout_train_support_failed"
            diagnostics["usable"] = True
            return diagnostics
        train_design = (
            np.column_stack([train_actions, np.ones(len(train_actions))])
            if self.effect_map_intercept else train_actions
        )
        heldout_mapping, _, _, _ = np.linalg.lstsq(
            train_design, train_deltas, rcond=1.0e-6,
        )
        heldout_design = (
            np.column_stack([heldout_actions, np.ones(len(heldout_actions))])
            if self.effect_map_intercept else heldout_actions
        )
        heldout_residual = heldout_design @ heldout_mapping - heldout_deltas
        heldout_normalized = float(
            np.linalg.norm(heldout_residual)
            / max(np.linalg.norm(heldout_deltas), 1.0e-6)
        )
        diagnostics["heldout_normalized_residual"] = heldout_normalized
        if not np.isfinite(heldout_normalized) or heldout_normalized > 0.75:
            diagnostics["reason"] = "heldout_effect_verification_failed"
            diagnostics["usable"] = True
            return diagnostics
        diagnostics["usable"] = True
        diagnostics["heldout_usable"] = True
        diagnostics["reason"] = "support_and_fit_quality_passed"
        return diagnostics

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        planner_info = dict(info_dict)
        if self.use_local_effect_ranker:
            n_envs = len(np.asarray(info_dict["pixels"]))
            effect_maps = np.zeros((n_envs, 2, 2), dtype=np.float32)
            effect_targets = np.zeros((n_envs, 2), dtype=np.float32)
            effect_valid = np.zeros(n_envs, dtype=bool)
            terminated_now = np.asarray(
                info_dict.get("terminated", np.zeros(n_envs, dtype=bool)),
                dtype=bool,
            ).reshape(-1)
            for index in range(n_envs):
                if terminated_now[index]:
                    continue
                target, phase = _rgb_reactive_action(
                    _raw_frame(info_dict["pixels"], index),
                    _raw_frame(info_dict["goal"], index),
                )
                if self.door_phase_reactive and phase == "door":
                    continue
                local_map, _ = self._effect_map(index, phase)
                quality = self._effect_map_quality(index, phase)
                if local_map is not None and quality.get("heldout_usable", False):
                    effect_maps[index] = local_map
                    effect_targets[index] = target
                    effect_valid[index] = True
            planner_info["_local_effect_map"] = effect_maps
            planner_info["_local_effect_target"] = effect_targets
            planner_info["_local_effect_valid"] = effect_valid
        model_actions = np.asarray(
            self.base_policy.get_action(planner_info, **kwargs), dtype=np.float32,
        )
        if model_actions.ndim == 1:
            model_actions = model_actions[None, :]
        output = model_actions.copy()
        terminated = np.asarray(
            info_dict.get("terminated", np.zeros(len(output), dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        needs_flush = np.asarray(
            info_dict.get("_needs_flush", np.zeros(len(output), dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        for index in range(len(output)):
            if needs_flush[index]:
                self.previous_frames[index] = None
                self.previous_actions[index] = None
                self.previous_phases[index] = None
            if self.previous_frames[index] is not None and self.previous_actions[index] is not None:
                previous_source = _rgb_red_centroid(self.previous_frames[index])
                current_source = _rgb_red_centroid(_raw_frame(info_dict["pixels"], index))
                delta = current_source - previous_source
                action_xy = np.asarray(self.previous_actions[index], dtype=np.float32)[:2]
                if np.isfinite(delta).all() and np.isfinite(action_xy).all():
                    self.effect_actions_by_env[index].append(action_xy.copy())
                    self.effect_deltas_by_env[index].append(delta.astype(np.float32))
                    previous_phase = self.previous_phases[index] or "goal"
                    self.effect_actions_by_env_phase[index][previous_phase].append(action_xy.copy())
                    self.effect_deltas_by_env_phase[index][previous_phase].append(delta.astype(np.float32))
        self.gate_calls += int(np.sum(~terminated & np.isfinite(output[:, :2]).all(axis=1)))
        for index in range(len(output)):
            # WorldModelPolicy intentionally returns NaN for already terminated
            # envs.  They are ignored by the evaluator and must not pollute the
            # diagnostic cosine distribution.
            if terminated[index] or not np.isfinite(output[index, :2]).all():
                continue
            self.gate_calls_by_env[index] += 1
            reactive, phase = _rgb_reactive_action(
                _raw_frame(info_dict["pixels"], index),
                _raw_frame(info_dict["goal"], index),
            )
            model_xy = output[index, :2]
            model_norm = float(np.linalg.norm(model_xy))
            reactive_norm = float(np.linalg.norm(reactive))
            cosine = float(np.dot(model_xy, reactive) / max(model_norm * reactive_norm, 1.0e-8))
            fallback = model_norm < self.model_min_norm or cosine < self.conflict_cos
            previous_phase = self.previous_phases[index]
            if (
                self.phase_transition_fallback
                and previous_phase is not None
                and previous_phase != phase
            ):
                # Protect the first block after a visual phase transition,
                # while retaining the learned/effect-policy action in a stable
                # phase.  This is weaker than disabling the entire door phase.
                fallback = True
            if self.door_phase_reactive and phase == "door":
                # TwoRoom has a hybrid task: crossing the door is a distinct
                # waypoint phase.  A frozen terminal-cost ranker is not
                # authorized to override the RGB waypoint controller there;
                # this is a predeclared phase-conditioned fail-closed arm.
                fallback = True
                # This phase gate must dominate the later local-effect repair
                # branch.  Previously ``fallback=True`` was interpreted by
                # that branch as permission to replace the model action with
                # an inverse local map, silently undoing the declared
                # door-phase fail-closed policy.
                self.effect_policy_phase_block_count += 1
            effect_map, effect_rank = self._effect_map(index, phase)
            effect_quality = self._effect_map_quality(index, phase)
            phase_effect_release_allowed = not (
                self.door_phase_reactive and phase == "door"
            )
            if (
                phase_effect_release_allowed
                and (self.use_effect_map or self.use_effect_policy)
                and effect_map is not None
                and effect_quality["usable"]
            ):
                predicted_effect = model_xy @ effect_map
                # The reactive vector is the desired local image-space
                # direction.  Reject a model action whose observed local
                # action-effect map predicts non-progress, even if its raw
                # action direction appears plausible.
                if float(np.dot(predicted_effect, reactive)) <= 0.0:
                    fallback = True
                if self.use_effect_policy:
                    phase_actions, phase_deltas = self._effect_histories(index, phase)
                    validated_map = effect_map
                    if len(phase_actions) >= 8:
                        train_actions = np.asarray(phase_actions[:-2], dtype=np.float64)
                        train_deltas = np.asarray(phase_deltas[:-2], dtype=np.float64)
                        train_design = (
                            np.column_stack([train_actions, np.ones(len(train_actions))])
                            if self.effect_map_intercept else train_actions
                        )
                        validated_map, _, _, _ = np.linalg.lstsq(
                            train_design, train_deltas, rcond=1.0e-6,
                        )
                        validated_map = validated_map[:2]
                    effect_action = _effect_policy_action(reactive, validated_map)
                    if effect_action is None:
                        self.effect_policy_fallback_count += 1
                        fallback = True
                    elif not effect_quality["heldout_usable"]:
                        self.effect_policy_validation_fallback_count += 1
                        fallback = True
                    elif fallback:
                        # The local map is a repair arm, not a replacement for
                        # the long-horizon planner. Preserve a model action
                        # that already passes the RGB conflict gate; only use
                        # the verified inverse when the model action was
                        # independently rejected.
                        output[index, :2] = effect_action
                        self.effect_policy_count += 1
                        fallback = False
            self.cosines.append(cosine)
            self.phases.append(phase)
            self.phase_counts_by_env[index][phase] += 1
            if fallback:
                output[index, :2] = reactive
                self.fallback_count += 1
                self.fallback_counts_by_env[index] += 1
            self.previous_frames[index] = _raw_frame(info_dict["pixels"], index).copy()
            self.previous_actions[index] = output[index].copy()
            self.previous_phases[index] = phase
        # Keep a small diagnostic snapshot on the policy; this is not used for
        # action choice and is exported by the runner for mechanism analysis.
        self.effect_map_rows = [len(history) for history in self.effect_actions_by_env]
        self.effect_map_phase = [
            self.previous_phases[index] for index in range(len(output))
        ]
        self.effect_map_rank = [
            self._effect_map(index, self.previous_phases[index])[1]
            if self.previous_phases[index] is not None else 0
            for index in range(len(output))
        ]
        self.effect_map_quality = [
            self._effect_map_quality(index, self.previous_phases[index])
            if self.previous_phases[index] is not None
            else {"rows": 0, "rank": 0, "usable": False, "reason": "no_phase_observation"}
            for index in range(len(output))
        ]
        self.effect_map = [
            None
            if self.previous_phases[index] is None or self._effect_map(index, self.previous_phases[index])[0] is None
            else self._effect_map(index, self.previous_phases[index])[0].tolist()
            for index in range(len(output))
        ]
        return output


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
                 final_action_selection: str = "mean",
                 vision_goal_bank: bool = False,
                 vision_door_bank: bool = False,
                 action_mean: np.ndarray | None = None,
                 action_std: np.ndarray | None = None,
                 **kwargs):
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
        self.vision_goal_bank = bool(vision_goal_bank)
        self.vision_door_bank = bool(vision_door_bank)
        if self.vision_door_bank and not self.vision_goal_bank:
            raise ValueError("vision_door_bank requires vision_goal_bank")
        if self.vision_goal_bank:
            if action_mean is None or action_std is None:
                raise ValueError("vision_goal_bank requires action statistics")
            self.action_mean = np.asarray(action_mean, dtype=np.float32)
            self.action_std = np.asarray(action_std, dtype=np.float32)

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
            # Phase-conditioned readouts receive only an RGB-derived phase
            # label.  The label is recomputed from current/goal pixels and is
            # expanded across candidates; no simulator state enters the cost.
            if "pixels" in info_dict and "goal" in info_dict:
                phase = _task_phase_from_rgb(
                    info_dict["pixels"][start_idx:end_idx],
                    info_dict["goal"][start_idx:end_idx],
                )
                expanded_infos["_task_phase"] = phase[:, None].expand(
                    current_bs, self.num_samples,
                )
                waypoint_direction = _task_waypoint_direction_from_rgb(
                    info_dict["pixels"][start_idx:end_idx],
                    info_dict["goal"][start_idx:end_idx],
                )
                expanded_infos["_task_waypoint_direction"] = waypoint_direction[:, None, :].expand(
                    current_bs, self.num_samples, 2,
                )

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
                if self.vision_goal_bank and use_bank:
                    if "pixels" not in info_dict or "goal" not in info_dict:
                        raise KeyError("vision_goal_bank requires transformed pixels and goal inputs")
                    bank_builder = (
                        _vision_door_candidate_bank if self.vision_door_bank
                        else _vision_goal_candidate_bank
                    )
                    bank_batch = bank_builder(
                        info_dict["pixels"][start_idx:end_idx],
                        info_dict["goal"][start_idx:end_idx],
                        horizon=self.horizon,
                        action_block=self.action_dim // 2,
                        action_mean=self.action_mean,
                        action_std=self.action_std,
                    ).to(device=self.device, dtype=self.dtype)
                    bank_count = bank_batch.shape[1]
                else:
                    bank_count = active_bank.shape[0]
                    bank_batch = active_bank.unsqueeze(0).expand(current_bs, -1, -1, -1)
                gaussian_count = self.num_samples - 1 - bank_count
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
                bank_mask[:, 1 : 1 + bank_count] = True
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
        ranker_cost = torch.cat([scaled, ones], dim=-1) @ self.coefficient.to(features)
        bank_mask = info_dict.get("_candidate_bank_mask")
        if bank_mask is None:
            raise ValueError(
                "latent_action_ranker requires an explicit candidate bank; "
                "unrestricted Gaussian proposals are outside calibration support"
            )
        bank_mask = bank_mask.to(device=ranker_cost.device, dtype=torch.bool)
        if bank_mask.shape != ranker_cost.shape:
            raise ValueError("candidate_bank_mask shape does not match ranker costs")
        bank_count = bank_mask.sum(dim=1, keepdim=True).clamp_min(1).to(base_cost.dtype)
        mask_float = bank_mask.to(base_cost.dtype)
        base_mean = (base_cost * mask_float).sum(dim=1, keepdim=True) / bank_count
        base_scale = torch.sqrt(
            (((base_cost - base_mean) * mask_float).square().sum(dim=1, keepdim=True) / bank_count)
        ).clamp_min(1.0e-6)
        ranker_mean = (ranker_cost * mask_float).sum(dim=1, keepdim=True) / bank_count
        ranker_scale = torch.sqrt(
            (((ranker_cost - ranker_mean) * mask_float).square().sum(dim=1, keepdim=True) / bank_count)
        ).clamp_min(1.0e-6)
        calibrated_ranker = (ranker_cost - ranker_mean) / ranker_scale * base_scale + base_mean
        return torch.where(bank_mask, calibrated_ranker, base_cost)


def _candidate_bank(
    action_mean: np.ndarray,
    action_std: np.ndarray,
    horizon: int,
    action_block: int,
    variant: str = "legacy",
) -> torch.Tensor:
    """Build fixed raw-action macro candidates, then normalize them.

    ``legacy`` is the previously frozen 11-template bank.  ``angular16`` is
    an explicit coverage intervention: 16 evenly spaced directions at two
    magnitudes plus zero, using the same macro-action interface and CEM slot
    budget.  It is deliberately selected without simulator state or outcomes.
    """
    if action_mean.size != action_std.size or action_mean.size % action_block:
        raise ValueError("action statistics are incompatible with action_block")
    raw_dim = action_mean.size // action_block
    if raw_dim != 2:
        raise ValueError(f"TwoRoom candidate bank expects 2-D actions, got {raw_dim}")
    if variant == "legacy":
        directions = [(0.0, 0.0)]
        for magnitude in (0.25, 0.50, 0.75):
            directions.extend([
                (magnitude, 0.0), (-magnitude, 0.0),
                (0.0, magnitude), (0.0, -magnitude),
            ])
        directions.extend([(0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)])
    elif variant in {"angular16", "vision_goal16", "vision_door16"}:
        directions = [(0.0, 0.0)]
        angle_count = 16 if variant == "angular16" else 8
        angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
        for magnitude in (0.50, 0.75):
            directions.extend([
                (float(magnitude * np.cos(angle)), float(magnitude * np.sin(angle)))
                for angle in angles
            ])
    else:
        raise ValueError(f"unknown TwoRoom candidate-bank variant: {variant}")
    sequences: list[np.ndarray] = []
    for dx, dy in directions:
        frame = np.asarray([dx, dy], dtype=np.float32)
        block = np.tile(frame, (action_block, 1)).reshape(-1)
        normalized_block = (block - action_mean) / np.maximum(action_std, 1e-6)
        sequences.append(np.tile(normalized_block[None, :], (horizon, 1)))
    return torch.as_tensor(np.stack(sequences), dtype=torch.float32)


def _red_centroid(image: torch.Tensor) -> torch.Tensor:
    """Estimate the red agent marker centroid from a normalized RGB image."""
    if image.ndim == 5:
        image = image[:, -1]
    if image.ndim != 4 or image.shape[1] < 3:
        raise ValueError("expected image batch [batch, channels, height, width]")
    red, green, blue = image[:, 0], image[:, 1], image[:, 2]
    mask = (red - green > 1.0) & (red - blue > 1.0)
    height, width = mask.shape[-2:]
    yy = torch.arange(height, device=image.device, dtype=image.dtype).view(1, height, 1)
    xx = torch.arange(width, device=image.device, dtype=image.dtype).view(1, 1, width)
    weights = mask.to(image.dtype)
    mass = weights.sum(dim=(-2, -1))
    safe_mass = mass.clamp_min(1.0)
    x = (weights * xx).sum(dim=(-2, -1)) / safe_mass
    y = (weights * yy).sum(dim=(-2, -1)) / safe_mass
    return torch.stack([x, y], dim=-1)


def _vision_goal_candidate_bank(
    pixels: torch.Tensor,
    goal: torch.Tensor,
    *,
    horizon: int,
    action_block: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> torch.Tensor:
    """Build root-local candidates from RGB red-marker displacement only.

    This is a perception-side candidate proposal, not a simulator-state oracle:
    it consumes the same transformed current/goal images as the policy.  The
    fixed offsets hedge against walls while preserving a small CEM bank.
    """
    if pixels.ndim not in (4, 5) or goal.ndim not in (4, 5):
        raise ValueError("vision goal bank expects transformed current and goal images")
    current = _red_centroid(pixels)
    target = _red_centroid(goal)
    displacement = target - current
    angles = torch.atan2(displacement[:, 1], displacement[:, 0])
    offsets = torch.arange(8, device=pixels.device, dtype=pixels.dtype) * (2.0 * np.pi / 8.0)
    angles = angles[:, None] + offsets[None, :]
    magnitudes = (0.50, 0.75)
    raw_frames = [torch.zeros((pixels.shape[0], 1, 2), device=pixels.device, dtype=pixels.dtype)]
    for magnitude in magnitudes:
        raw_frames.append(torch.stack([
            magnitude * torch.cos(angles), magnitude * torch.sin(angles),
        ], dim=-1))
    frames = torch.cat(raw_frames, dim=1)
    raw_dim = action_block * 2
    raw = frames[:, :, None, :].expand(-1, -1, action_block, -1).reshape(
        pixels.shape[0], frames.shape[1], raw_dim,
    )
    mean = torch.as_tensor(action_mean, device=pixels.device, dtype=pixels.dtype)
    std = torch.as_tensor(action_std, device=pixels.device, dtype=pixels.dtype).clamp_min(1.0e-6)
    normalized = (raw - mean) / std
    return normalized[:, :, None, :].expand(-1, -1, horizon, -1).contiguous()


def _longest_false_run(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return start/end of the longest false run for each batch row."""
    starts: list[int] = []
    ends: list[int] = []
    for row in mask.detach().cpu().numpy():
        best_start, best_end = 0, 0
        start = None
        for index, value in enumerate(row.tolist() + [True]):
            if not value and start is None:
                start = index
            elif value and start is not None:
                if index - start > best_end - best_start:
                    best_start, best_end = start, index
                start = None
        starts.append(best_start)
        ends.append(best_end)
    return (
        torch.as_tensor(starts, device=mask.device, dtype=torch.float32),
        torch.as_tensor(ends, device=mask.device, dtype=torch.float32),
    )


def _longest_true_run_within(mask: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    starts: list[int] = []
    ends: list[int] = []
    for row, valid_row in zip(mask.detach().cpu().numpy(), valid.detach().cpu().numpy()):
        indices = np.flatnonzero(valid_row)
        best_start, best_end = 0, 0
        if len(indices):
            lo, hi = int(indices[0]), int(indices[-1])
            start = None
            for index in range(lo, hi + 2):
                value = bool(row[index]) if index <= hi else False
                if value and start is None:
                    start = index
                elif not value and start is not None:
                    if index - start > best_end - best_start:
                        best_start, best_end = start, index
                    start = None
        starts.append(best_start)
        ends.append(best_end)
    return (
        torch.as_tensor(starts, device=mask.device, dtype=torch.float32),
        torch.as_tensor(ends, device=mask.device, dtype=torch.float32),
    )


def _vision_door_point(pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer wall-door point and wall orientation from transformed RGB."""
    frame = pixels[:, -1] if pixels.ndim == 5 else pixels
    _, _, height, width = frame.shape
    dark = frame.mean(dim=1) < -0.5
    center_x, center_y = width // 2, height // 2
    vertical_line = dark[:, :, max(0, center_x - 3):min(width, center_x + 4)].float().mean(dim=-1)
    horizontal_line = dark[:, max(0, center_y - 3):min(height, center_y + 4), :].float().mean(dim=1)
    is_vertical = vertical_line.mean(dim=-1) >= horizontal_line.mean(dim=-1)
    vertical_dark = dark[:, :, center_x]
    horizontal_dark = dark[:, center_y, :]
    vertical_open = ~vertical_dark
    horizontal_open = ~horizontal_dark
    vertical_valid = vertical_dark.to(torch.int64).cumsum(dim=1) > 0
    vertical_valid = vertical_valid & (torch.flip(vertical_dark, dims=[1]).to(torch.int64).cumsum(dim=1).flip(dims=[1]) > 0)
    horizontal_valid = horizontal_dark.to(torch.int64).cumsum(dim=1) > 0
    horizontal_valid = horizontal_valid & (torch.flip(horizontal_dark, dims=[1]).to(torch.int64).cumsum(dim=1).flip(dims=[1]) > 0)
    v_start, v_end = _longest_true_run_within(vertical_open, vertical_valid)
    h_start, h_end = _longest_true_run_within(horizontal_open, horizontal_valid)
    door = torch.stack([
        torch.where(is_vertical, torch.full_like(v_start, float(center_x)), (h_start + h_end) / 2.0),
        torch.where(is_vertical, (v_start + v_end) / 2.0, torch.full_like(h_start, float(center_y))),
    ], dim=-1)
    return door, is_vertical


def _task_phase_from_rgb(pixels: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Return 0=door, 1=goal using the same RGB waypoint rule as the gate."""
    current = _red_centroid(pixels)
    target = _red_centroid(goal)
    door, is_vertical = _vision_door_point(pixels)
    side_gap = torch.where(
        is_vertical,
        (target[:, 0] - current[:, 0]).abs(),
        (target[:, 1] - current[:, 1]).abs(),
    )
    far_from_door = torch.linalg.vector_norm(current - door, dim=-1) > 18.0
    return ((side_gap > 35.0) & far_from_door).logical_not().to(torch.int64)


def _task_waypoint_direction_from_rgb(
    pixels: torch.Tensor, goal: torch.Tensor,
) -> torch.Tensor:
    """Return the normalized RGB direction to the active task waypoint."""
    current = _red_centroid(pixels)
    target = _red_centroid(goal)
    door, is_vertical = _vision_door_point(pixels)
    side_gap = torch.where(
        is_vertical,
        (target[:, 0] - current[:, 0]).abs(),
        (target[:, 1] - current[:, 1]).abs(),
    )
    far_from_door = torch.linalg.vector_norm(current - door, dim=-1) > 18.0
    door_phase = (side_gap > 35.0) & far_from_door
    waypoint = torch.where(door_phase[:, None], door, target)
    direction = waypoint - current
    return direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1.0e-8)


class VisionGeometryCostModel:
    """RGB-only task-geometry baseline for separating metric from latent ranking."""

    def __init__(self, action_mean: np.ndarray, action_std: np.ndarray, action_block: int):
        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_std = np.asarray(action_std, dtype=np.float32)
        self.action_block = int(action_block)

    @torch.inference_mode()
    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        pixels = info_dict["pixels"][:, 0]
        goal = info_dict["goal"][:, 0]
        current = _red_centroid(pixels)
        target = _red_centroid(goal)
        door, is_vertical = _vision_door_point(pixels)
        raw_macro = action_candidates.reshape(
            *action_candidates.shape[:3], self.action_block, 2,
        )
        mean = torch.as_tensor(self.action_mean, device=action_candidates.device, dtype=action_candidates.dtype)
        std = torch.as_tensor(self.action_std, device=action_candidates.device, dtype=action_candidates.dtype)
        raw_macro = (
            raw_macro * std.reshape(1, 1, self.action_block, 2)
            + mean.reshape(1, 1, self.action_block, 2)
        ).mean(dim=3)
        positions = current[:, None, None, :] + torch.cumsum(
            raw_macro * (10.5 * self.action_block), dim=2,
        )
        final_distance = torch.linalg.vector_norm(
            positions[:, :, -1] - target[:, None], dim=-1,
        )
        side_gap = torch.where(
            is_vertical,
            (target[:, 0] - current[:, 0]).abs(),
            (target[:, 1] - current[:, 1]).abs(),
        )
        midpoint = positions[:, :, max(0, positions.shape[2] // 2 - 1)]
        door_distance = torch.linalg.vector_norm(midpoint - door[:, None], dim=-1)
        return torch.where(side_gap[:, None] > 35.0, door_distance + final_distance, final_distance)


def _vision_door_candidate_bank(
    pixels: torch.Tensor,
    goal: torch.Tensor,
    *,
    horizon: int,
    action_block: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> torch.Tensor:
    """Build RGB-only two-stage candidates that explicitly traverse a visible door."""
    if pixels.ndim not in (4, 5) or goal.ndim not in (4, 5):
        raise ValueError("vision door bank expects transformed current and goal images")
    current = _red_centroid(pixels)
    target = _red_centroid(goal)
    batch = pixels.shape[0]
    door, _ = _vision_door_point(pixels)
    door_direction = door - current
    goal_direction = target - current
    door_angle = torch.atan2(door_direction[:, 1], door_direction[:, 0])
    goal_angle = torch.atan2(goal_direction[:, 1], goal_direction[:, 0])
    offsets = torch.arange(8, device=pixels.device, dtype=pixels.dtype) * (2.0 * np.pi / 8.0)
    stage_angles = door_angle[:, None] + offsets[None, :]
    goal_angles = goal_angle[:, None] + offsets[None, :]
    stage = torch.stack([0.75 * torch.cos(stage_angles), 0.75 * torch.sin(stage_angles)], dim=-1)
    second = torch.stack([0.75 * torch.cos(goal_angles), 0.75 * torch.sin(goal_angles)], dim=-1)
    direct = torch.stack([0.50 * torch.cos(goal_angles), 0.50 * torch.sin(goal_angles)], dim=-1)
    split = max(1, horizon // 2)
    staged = torch.cat([
        stage[:, :, None, :].expand(-1, -1, split, -1),
        second[:, :, None, :].expand(-1, -1, horizon - split, -1),
    ], dim=2)
    zero = torch.zeros((batch, 1, horizon, 2), device=pixels.device, dtype=pixels.dtype)
    direct = direct[:, :, None, :].expand(-1, -1, horizon, -1)
    raw_frames = torch.cat([zero, staged, direct], dim=1)
    raw_dim = action_block * 2
    raw = raw_frames[:, :, :, None, :].expand(-1, -1, -1, action_block, -1).reshape(
        batch, raw_frames.shape[1], horizon, raw_dim,
    )
    mean = torch.as_tensor(action_mean, device=pixels.device, dtype=pixels.dtype).view(1, 1, 1, -1)
    std = torch.as_tensor(action_std, device=pixels.device, dtype=pixels.dtype).clamp_min(1.0e-6).view(1, 1, 1, -1)
    return (raw - mean) / std


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
    if args.cost_metric == "vision_geometry":
        model = VisionGeometryCostModel(action_mean, action_std, args.action_block)
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
        if not args.candidate_bank:
            raise ValueError("latent_action_ranker requires --candidate-bank for support gating")
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
        artifact = apply_capacity_bilinear_mode(
            artifact, args.capacity_bilinear_mode,
        )
        if args.capacity_bilinear_base_cost_blend_weight is not None:
            if not 0.0 <= args.capacity_bilinear_base_cost_blend_weight <= 1.0:
                raise ValueError(
                    "capacity bilinear base-cost blend weight must be in [0, 1]"
                )
            artifact["bilinear_base_cost_blend_weight"] = float(
                args.capacity_bilinear_base_cost_blend_weight
            )
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
    if args.local_effect_ranker:
        model = LocalEffectRankedCostModel(
            model, action_mean, action_std, args.local_effect_blend_weight,
        )
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
            variant=args.candidate_bank_variant,
        )
        solver_kwargs["candidate_bank_iterations"] = args.candidate_bank_iterations
        solver_kwargs["vision_goal_bank"] = args.candidate_bank_variant in {"vision_goal16", "vision_door16"}
        solver_kwargs["vision_door_bank"] = args.candidate_bank_variant == "vision_door16"
        solver_kwargs["action_mean"] = action_mean
        solver_kwargs["action_std"] = action_std
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
    elif policy_kind == "reactive":
        policy = RGBReactivePolicy()
    else:
        policy = _planning_policy(args, args.device, action_mean, action_std)
        if policy_kind == "hybrid":
            policy = RiskGatedRGBPolicy(
                policy,
                conflict_cos=args.hybrid_conflict_cos,
                model_min_norm=args.hybrid_model_min_norm,
                use_effect_map=args.hybrid_use_effect_map,
                use_effect_policy=args.hybrid_use_effect_policy,
                effect_map_intercept=args.hybrid_effect_map_intercept,
                door_phase_reactive=args.hybrid_door_phase_reactive,
                phase_transition_fallback=args.hybrid_phase_transition_fallback,
                use_local_effect_ranker=args.local_effect_ranker,
            )
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
        if (
            policy_kind in {"planning", "hybrid"}
            and args.record_solver_trace
            and hasattr(policy, "solver")
        ):
            result["cem_trace"] = list(getattr(policy.solver, "trace", []))
    finally:
        world.close()
    successes = np.asarray(result["episode_successes"], dtype=np.int64)
    output = {
        "success_rate": float(successes.mean()),
        "episode_successes": successes.tolist(),
        "seeds": np.asarray(result["seeds"], dtype=np.int64).tolist(),
    }
    if isinstance(policy, RiskGatedRGBPolicy):
        output["risk_gate"] = {
            "gate_calls": int(policy.gate_calls),
            "fallback_count": int(policy.fallback_count),
            "fallback_rate": float(policy.fallback_count / max(policy.gate_calls, 1)),
            "conflict_cos": float(policy.conflict_cos),
            "model_min_norm": float(policy.model_min_norm),
            "use_effect_map": bool(policy.use_effect_map),
            "use_effect_policy": bool(policy.use_effect_policy),
            "effect_map_intercept": bool(policy.effect_map_intercept),
            "door_phase_reactive": bool(policy.door_phase_reactive),
            "phase_transition_fallback": bool(policy.phase_transition_fallback),
            "effect_policy_count": int(policy.effect_policy_count),
            "effect_policy_fallback_count": int(policy.effect_policy_fallback_count),
            "effect_policy_validation_fallback_count": int(
                policy.effect_policy_validation_fallback_count
            ),
            "effect_policy_phase_block_count": int(
                policy.effect_policy_phase_block_count
            ),
            "use_local_effect_ranker": bool(policy.use_local_effect_ranker),
            "effect_map_rows_by_env": list(getattr(policy, "effect_map_rows", [])),
            "effect_map_phase_by_env": list(getattr(policy, "effect_map_phase", [])),
            "effect_map_rank_by_env": list(getattr(policy, "effect_map_rank", [])),
            "effect_map_quality_by_env": list(getattr(policy, "effect_map_quality", [])),
            "effect_map": getattr(policy, "effect_map", None),
            "by_env": [
                {
                    "gate_calls": int(calls),
                    "fallback_count": int(fallbacks),
                    "fallback_rate": float(fallbacks / max(calls, 1)),
                    "phase_counts": dict(phases),
                    "effect_map_rows": int(policy.effect_map_rows[index]),
                    "effect_map_phase": policy.effect_map_phase[index],
                    "effect_map_rank": int(policy.effect_map_rank[index]),
                }
                for index, (calls, fallbacks, phases) in enumerate(zip(
                    policy.gate_calls_by_env,
                    policy.fallback_counts_by_env,
                    policy.phase_counts_by_env,
                ))
            ],
            "phase_counts": {
                "door": int(sum(phase == "door" for phase in policy.phases)),
                "goal": int(sum(phase == "goal" for phase in policy.phases)),
            },
            "cosine_mean": float(np.mean(policy.cosines)) if policy.cosines else None,
            "cosine_p10": float(np.percentile(policy.cosines, 10)) if policy.cosines else None,
        }
    # Preserve deployment evidence for train-only readout interventions.  The
    # TwoRoom runner previously omitted this field, making an unchanged score
    # indistinguishable from a readout that never reached the planner.
    policy_diagnostics = getattr(policy, "runtime_diagnostics", None)
    if callable(policy_diagnostics):
        output["runtime_diagnostics"] = policy_diagnostics()
    else:
        solver_model = getattr(getattr(policy, "solver", None), "model", None)
        if callable(getattr(solver_model, "runtime_diagnostics", None)):
            output["runtime_diagnostics"] = solver_model.runtime_diagnostics()
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
        "--candidate-bank-variant", choices=("legacy", "angular16", "vision_goal16", "vision_door16"), default="legacy",
        help="fixed bank definition; angular16 is a predeclared coverage intervention",
    )
    parser.add_argument(
        "--candidate-bank-iterations", type=int, default=None,
        help="number of initial CEM iterations containing the fixed bank; None means every iteration",
    )
    parser.add_argument(
        "--cost-metric",
        choices=(
            "euclidean", "spectral_top_32", "physical_readout",
            "action_conditioned_readout", "latent_action_ranker",
            "capacity_safe_readout", "indexed_canonical_readout", "vision_geometry",
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
        "--capacity-bilinear-base-cost-blend-weight",
        type=float,
        default=None,
        help=(
            "fixed conservative blend of frozen base cost into the bilinear "
            "bank ranking; omitted means the artifact's declared value"
        ),
    )
    parser.add_argument(
        "--capacity-bilinear-mode",
        choices=(
            "artifact",
            "action_only",
            "full_predicted_dynamics",
            "consensus",
            "rank_fusion",
        ),
        default="artifact",
        help=(
            "explicit train-only bilinear deployment mode; artifact preserves "
            "the saved gate, while overrides only select already-exported "
            "coefficients"
        ),
    )
    parser.add_argument(
        "--record-solver-trace", action="store_true",
        help="diagnostic only: record CEM actions/costs for solver comparison",
    )
    parser.add_argument(
        "--hybrid-conflict-cos", type=float, default=0.0,
        help="predeclared RGB fallback threshold; 0 means conflict >90 degrees",
    )
    parser.add_argument(
        "--hybrid-model-min-norm", type=float, default=0.08,
        help="predeclared minimum raw model action norm before RGB fallback",
    )
    parser.add_argument(
        "--hybrid-use-effect-map", action="store_true",
        help="use an online RGB action-to-marker-displacement map as an additional fallback gate",
    )
    parser.add_argument(
        "--hybrid-use-effect-policy", action="store_true",
        help="use the verified local RGB effect map to invert the task-directed action",
    )
    parser.add_argument(
        "--hybrid-effect-map-intercept", action="store_true",
        help=(
            "fit an affine phase-local RGB effect map; the intercept absorbs "
            "common motion and is excluded from action inversion"
        ),
    )
    parser.add_argument(
        "--hybrid-door-phase-reactive", action="store_true",
        help="fail closed to RGB reactive control during the detected door phase",
    )
    parser.add_argument(
        "--hybrid-phase-transition-fallback", action="store_true",
        help="fail closed for one block only when the RGB phase changes",
    )
    parser.add_argument(
        "--local-effect-ranker", action="store_true",
        help=(
            "rank supported candidate-bank actions with a held-out-verified "
            "local RGB effect chart before execution"
        ),
    )
    parser.add_argument(
        "--local-effect-blend-weight", type=float, default=0.5,
        help="fixed blend weight for the local-effect candidate ranker",
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
    reactive_result = _canonicalize(_run(args, "reactive", action_mean, action_std))
    hybrid_result = _canonicalize(_run(args, "hybrid", action_mean, action_std))
    if random_result["seeds"] != planning_result["seeds"]:
        raise RuntimeError("random and planning did not evaluate the same seed set")
    if any(result["seeds"] != random_result["seeds"] for result in (reactive_result, hybrid_result)):
        raise RuntimeError("paired policies did not evaluate the same seed set")
    random_success = np.asarray(random_result["episode_successes"], dtype=np.int64)
    planning_success = np.asarray(planning_result["episode_successes"], dtype=np.int64)
    reactive_success = np.asarray(reactive_result["episode_successes"], dtype=np.int64)
    hybrid_success = np.asarray(hybrid_result["episode_successes"], dtype=np.int64)
    paired_delta = planning_success - random_success
    paired_hybrid_delta = hybrid_success - planning_success
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
            "candidate_bank_definition": (
                "zero, axis-aligned magnitudes 0.25/0.50/0.75, and four diagonals"
                if args.candidate_bank and args.candidate_bank_variant == "legacy"
                else "zero plus 16 directions at magnitudes 0.50 and 0.75"
                if args.candidate_bank_variant == "angular16"
                else "RGB red-marker goal-relative zero plus 8 offsets at magnitudes 0.50 and 0.75"
                if args.candidate_bank_variant == "vision_goal16"
                else "RGB visible-door then goal two-stage candidates"
                if args.candidate_bank else None
            ),
            "candidate_bank_variant": args.candidate_bank_variant if args.candidate_bank else None,
            "candidate_bank_iterations": args.candidate_bank_iterations,
            "residual_support_scope": (
                "predeclared_candidate_bank_only" if args.candidate_bank else "all_candidates"
            ),
            "cost_metric": args.cost_metric,
            "hybrid_gate": {
                "conflict_cos": args.hybrid_conflict_cos,
                "model_min_norm": args.hybrid_model_min_norm,
                "use_effect_map": bool(args.hybrid_use_effect_map),
                "use_effect_policy": bool(args.hybrid_use_effect_policy),
                "effect_map_intercept": bool(args.hybrid_effect_map_intercept),
                "door_phase_reactive": bool(args.hybrid_door_phase_reactive),
                "phase_transition_fallback": bool(args.hybrid_phase_transition_fallback),
                "decision_rule": "fallback if model norm < threshold or cosine(model, RGB reactive) < threshold",
                "runtime_inputs": "raw current RGB and raw rendered RGB goal only; no simulator state",
            },
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
            "capacity_bilinear_base_cost_blend_weight": (
                args.capacity_bilinear_base_cost_blend_weight
            ),
            "capacity_bilinear_mode": args.capacity_bilinear_mode,
            "local_effect_ranker": bool(args.local_effect_ranker),
            "local_effect_blend_weight": float(args.local_effect_blend_weight),
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
        "rgb_reactive": reactive_result,
        "hybrid_rgb_fallback": hybrid_result,
        "paired": {
            "seed_order": random_result["seeds"],
            "delta_lewm_minus_random": paired_delta.tolist(),
            "wins": int(np.sum(paired_delta > 0)),
            "losses": int(np.sum(paired_delta < 0)),
            "ties": int(np.sum(paired_delta == 0)),
            "success_rate_delta": float(paired_delta.mean()),
            "hybrid_minus_frozen_lewm": paired_hybrid_delta.tolist(),
            "hybrid_wins": int(np.sum(paired_hybrid_delta > 0)),
            "hybrid_losses": int(np.sum(paired_hybrid_delta < 0)),
            "hybrid_ties": int(np.sum(paired_hybrid_delta == 0)),
            "hybrid_success_rate_delta": float(paired_hybrid_delta.mean()),
        },
        "claim_boundary": "Independent live TwoRoom RGB planning consequence; not a universal world-model claim.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "planning.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"random": random_result, "frozen_lewm": planning_result}, indent=2))


if __name__ == "__main__":
    main()
