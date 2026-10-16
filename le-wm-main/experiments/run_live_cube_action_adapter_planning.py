"""Paired live RGB planning control for the frozen OGB-Cube checkpoint.

The checkpoint remains frozen.  The optional action interface map is fitted
by ``audit_frozen_cross_environment_prediction.py`` on calibration episodes
and loaded here without refitting.  Baseline and adapted controllers use the
same simulator seeds and the same CEM configuration; no planning outcome is
used to choose the map or tune the controller.

This is an independent closed-loop consequence test for the Cube prediction
repair.  It is not a universal world-model or causal-physics evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

# Keep the historical direct-script entry point usable as well as the package
# invocation used by the CUDA protocol.
if __package__ in (None, ""):
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import gymnasium as gym
from torchvision.transforms import v2 as transforms

from experiments.checkpoint_compat import load_compatible_pretrained


class ActionNormalizer:
    """Picklable raw-action <-> checkpoint-normalized-action transform."""

    def __init__(
        self,
        block_mean: np.ndarray,
        block_std: np.ndarray,
        *,
        clip_output: bool = False,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ):
        self.block_mean = np.asarray(block_mean, dtype=np.float32)
        self.block_std = np.asarray(block_std, dtype=np.float32).clip(1e-6)
        if self.block_mean.ndim != 1 or self.block_mean.size % 5 != 0:
            raise ValueError("Cube action statistics must be flattened frame-skip blocks")
        raw_dim = self.block_mean.size // 5
        self.raw_mean = self.block_mean.reshape(5, raw_dim).mean(axis=0)
        self.raw_std = self.block_std.reshape(5, raw_dim).mean(axis=0).clip(1e-6)
        if action_low >= action_high:
            raise ValueError("action_low must be smaller than action_high")
        self.clip_output = bool(clip_output)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.inverse_calls = 0
        self.inverse_values = 0
        self.clipped_values = 0
        self.max_preclip_abs = 0.0
        self.max_postclip_abs = 0.0

    def _stats(self, width: int) -> tuple[np.ndarray, np.ndarray]:
        if width == self.raw_mean.size:
            return self.raw_mean, self.raw_std
        if width == self.block_mean.size:
            return self.block_mean, self.block_std
        raise ValueError(f"unsupported action width {width}")

    def transform(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        mean, std = self._stats(value.shape[-1])
        return (value - mean) / std

    def inverse_transform(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        mean, std = self._stats(value.shape[-1])
        raw = value * std + mean
        self.inverse_calls += 1
        self.inverse_values += int(raw.size)
        if raw.size:
            self.max_preclip_abs = max(self.max_preclip_abs, float(np.max(np.abs(raw))))
        if self.clip_output:
            clipped = np.clip(raw, self.action_low, self.action_high)
            self.clipped_values += int(np.sum((raw < self.action_low) | (raw > self.action_high)))
            raw = clipped
        if raw.size:
            self.max_postclip_abs = max(self.max_postclip_abs, float(np.max(np.abs(raw))))
        return raw

    def diagnostics(self) -> dict[str, Any]:
        return {
            "action_output_clip": self.clip_output,
            "action_space_bounds": [self.action_low, self.action_high],
            "inverse_transform_calls": self.inverse_calls,
            "inverse_transform_values": self.inverse_values,
            "preclip_values_outside_bounds": self.clipped_values,
            "preclip_fraction": float(self.clipped_values / max(self.inverse_values, 1)),
            "max_preclip_abs": self.max_preclip_abs,
            "max_postclip_abs": self.max_postclip_abs,
        }


class CubeGoalAliasWrapper(gym.Wrapper):
    """Expose OGB-Cube's rendered task goal under LeWM's ``goal`` key.

    OGB-Cube returns the rendered target as ``goal_rendered`` when
    ``reset(options={'render_goal': True})`` is used.  ``WorldModelPolicy``
    consumes the canonical ``goal`` key, so the alias must be inserted before
    ``stable_worldmodel.World``'s ``MegaWrapper`` formats and validates the
    transition.  The goal is captured once per episode and then reused on
    every step; no simulator state is exposed to the policy.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._goal = None

    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        # Current stable_worldmodel's CubeEnv keeps the rendered target on the
        # environment and returns only the state target in ``info['target']``;
        # newer OGBench builds may also return ``goal_rendered`` directly.
        rendered = info.get("goal_rendered")
        if rendered is None:
            rendered = getattr(self.env.unwrapped, "_cur_goal_rendered", None)
        if rendered is None:
            raise RuntimeError(
                "OGB-Cube task reset did not provide goal_rendered; "
                "planning requires options={'render_goal': True}."
            )
        self._goal = np.asarray(rendered).copy()
        info = dict(info)
        info["goal"] = self._goal.copy()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self._goal is None:
            raise RuntimeError("CubeGoalAliasWrapper stepped before reset.")
        info = dict(info)
        info["goal"] = self._goal.copy()
        return observation, reward, terminated, truncated, info


class ActionAdaptedCostModel(torch.nn.Module):
    """Delegate a frozen LeWM cost while mapping target actions to its frame."""

    def __init__(self, base: torch.nn.Module, matrix: np.ndarray, bias: np.ndarray):
        super().__init__()
        self.base = base
        matrix_tensor = torch.as_tensor(matrix, dtype=torch.float32)
        bias_tensor = torch.as_tensor(bias, dtype=torch.float32)
        if matrix_tensor.ndim != 2 or matrix_tensor.shape[0] != matrix_tensor.shape[1]:
            raise ValueError("action adapter matrix must be square")
        if bias_tensor.shape != (matrix_tensor.shape[0],):
            raise ValueError("action adapter bias has incompatible shape")
        self.register_buffer("action_matrix", matrix_tensor)
        self.register_buffer("action_bias", bias_tensor)

    @property
    def raw_action_dim(self) -> int:
        return int(self.action_matrix.shape[0])

    def _map_actions(self, actions: torch.Tensor) -> torch.Tensor:
        width = int(actions.shape[-1])
        raw_dim = self.raw_action_dim
        if width == raw_dim:
            return actions @ self.action_matrix.T + self.action_bias
        if width % raw_dim != 0:
            raise ValueError(f"action width {width} is incompatible with raw dim {raw_dim}")
        blocks = actions.reshape(*actions.shape[:-1], width // raw_dim, raw_dim)
        mapped = blocks @ self.action_matrix.T + self.action_bias
        return mapped.reshape(*actions.shape[:-1], width)

    def get_cost(self, info_dict: dict[str, Any], action_candidates: torch.Tensor) -> torch.Tensor:
        mapped_info = dict(info_dict)
        if "action" in mapped_info and torch.is_tensor(mapped_info["action"]):
            mapped_info["action"] = self._map_actions(mapped_info["action"])
        return self.base.get_cost(mapped_info, self._map_actions(action_candidates))


class PhaseConditionedCEMSolver(swm.solver.CEMSolver):
    """CEM with a frozen, model-ranked contact-phase candidate bank.

    The bank is constructed before the test rollout and contains only generic
    normalized action phases.  It does not inspect cube position, simulator
    state, or test outcomes.  The frozen world model ranks the bank by its
    normal cost, then ordinary CEM continues from the selected full-horizon
    candidate.  This isolates candidate-support failure from prediction-cost
    failure without turning privileged geometry into a runtime input.
    """

    def __init__(self, *args: Any, phase_bank: np.ndarray, **kwargs: Any):
        super().__init__(*args, **kwargs)
        phase_bank_tensor = torch.as_tensor(phase_bank, dtype=torch.float32)
        if phase_bank_tensor.ndim != 3:
            raise ValueError("phase bank must have shape (bank, horizon, action_dim)")
        self.registered_phase_bank = phase_bank_tensor
        self.bank_indices: list[int] = []
        self.bank_costs: list[float] = []

    def _expanded_info(self, info_dict: dict[str, Any], bank_size: int) -> dict[str, Any]:
        total_envs = len(next(iter(info_dict.values())))
        expanded: dict[str, Any] = {}
        for key, value in info_dict.items():
            if torch.is_tensor(value):
                target_dtype = self.dtype if value.is_floating_point() else None
                batch = value.to(device=self.device, dtype=target_dtype)
                expanded[key] = batch.unsqueeze(1).expand(total_envs, bank_size, *batch.shape[1:])
            elif isinstance(value, np.ndarray):
                expanded[key] = np.repeat(value[:, None, ...], bank_size, axis=1)
            elif isinstance(value, list):
                expanded[key] = [value for _ in range(bank_size)]
            else:
                expanded[key] = value
        return expanded

    @torch.inference_mode()
    def solve(self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None) -> dict:
        total_envs = len(next(iter(info_dict.values())))
        if init_action is not None:
            # With receding-horizon execution, preserve the previously selected
            # phase sequence across visual re-observations.  Re-ranking a new
            # bank at every pulse would erase the approach/contact phase and
            # make the intervention internally inconsistent.
            return super().solve(info_dict, init_action=init_action)
        bank = self.registered_phase_bank.to(self.device, dtype=self.dtype)
        if bank.shape[1] != self.horizon or bank.shape[2] != self.action_dim:
            raise ValueError(
                f"phase bank shape {tuple(bank.shape)} does not match "
                f"({bank.shape[0]}, {self.horizon}, {self.action_dim})"
            )
        selected_parts = []
        # Match the released CEM implementation's batch slicing.  LeWM's
        # goal embedding has a singleton leading dimension that is broadcast
        # correctly for current_bs=1; evaluating all episodes at once changes
        # that broadcast contract.
        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx
            sliced_info: dict[str, Any] = {}
            for key, value in info_dict.items():
                if torch.is_tensor(value):
                    sliced_info[key] = value[start_idx:end_idx]
                elif isinstance(value, np.ndarray):
                    sliced_info[key] = value[start_idx:end_idx]
                elif isinstance(value, list):
                    sliced_info[key] = value[start_idx:end_idx]
                else:
                    sliced_info[key] = value
            candidates = bank.unsqueeze(0).expand(current_bs, -1, -1, -1)
            costs = self.model.get_cost(
                self._expanded_info(sliced_info, int(bank.shape[0])), candidates
            )
            if costs.shape != (current_bs, bank.shape[0]):
                raise ValueError(f"phase-bank costs have unexpected shape {tuple(costs.shape)}")
            best = torch.argmin(costs, dim=1)
            rows = torch.arange(current_bs, device=self.device)
            selected_parts.append(candidates[rows, best])
            self.bank_indices.extend(best.detach().cpu().tolist())
            self.bank_costs.extend(costs[rows, best].detach().float().cpu().tolist())
        selected = torch.cat(selected_parts, dim=0)
        # CEM still performs the declared optimization; the bank only provides
        # a support-aware initial mean and never sees evaluator outcomes.
        return super().solve(info_dict, init_action=selected)

    def phase_diagnostics(self) -> dict[str, Any]:
        counts = np.bincount(self.bank_indices, minlength=self.registered_phase_bank.shape[0])
        return {
            "bank_size": int(self.registered_phase_bank.shape[0]),
            "selected_index_counts": counts.astype(int).tolist(),
            "selection_count": int(len(self.bank_indices)),
            "selected_cost_mean": float(np.mean(self.bank_costs)) if self.bank_costs else None,
        }


def _cube_phase_bank(
    action_normalizer: ActionNormalizer,
    *,
    horizon: int,
    action_block: int,
) -> np.ndarray:
    """Build generic x/y align-descend-close-push candidates in model coordinates."""
    raw_dim = int(action_normalizer.raw_mean.size)
    if raw_dim != 5:
        raise ValueError(f"Cube phase bank expects five raw action dimensions, got {raw_dim}")
    if horizon < 5 or action_block < 1:
        raise ValueError("phase bank requires horizon >= 5 and positive action_block")
    directions = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    candidates = []
    for direction in directions:
        raw_tokens = np.zeros((horizon, raw_dim), dtype=np.float32)
        raw_tokens[0, :2] = direction       # lateral alignment phase
        raw_tokens[1:3, 2] = -1.0            # descend to the table/cube plane
        raw_tokens[3, 4] = -1.0              # close the gripper
        raw_tokens[4, :2] = direction       # short push phase
        model_tokens = action_normalizer.transform(raw_tokens)
        block_tokens = np.repeat(model_tokens[:, None, :], action_block, axis=1)
        candidates.append(block_tokens.reshape(horizon, raw_dim * action_block))
    return np.asarray(candidates, dtype=np.float32)


def _image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ]
    )


def _load_adapter(path: Path, target_environment: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    reports = [row for row in payload["reports"] if row["environment"] == target_environment]
    if len(reports) != 1:
        raise ValueError(f"expected exactly one report for {target_environment} in {path}")
    report = reports[0]
    protocol = report["summary"].get("action_adapter_protocol")
    if not protocol or protocol.get("fit", {}).get("calibration_windows", 0) < 2:
        raise ValueError("adapter artifact lacks a valid calibration-only fit")
    if not protocol.get("weights_frozen", False):
        raise ValueError("adapter artifact does not declare frozen checkpoint weights")
    return (
        np.asarray(protocol["matrix"], dtype=np.float32),
        np.asarray(protocol["bias"], dtype=np.float32),
        np.asarray(report["calibration_action_mean"], dtype=np.float32),
        np.asarray(report["calibration_action_std"], dtype=np.float32),
    )


def _run_controller(
    *,
    policy_path: str,
    environment: str,
    device: str,
    seeds: list[int],
    action_mean: np.ndarray,
    action_std: np.ndarray,
    matrix: np.ndarray | None,
    bias: np.ndarray | None,
    horizon: int,
    receding_horizon: int,
    action_block: int,
    num_samples: int,
    n_steps: int,
    topk: int,
    var_scale: float,
    max_episode_steps: int,
    clip_env_actions: bool,
    phase_conditioned: bool = False,
    reset_options: list[dict[str, Any]] | None = None,
    capture_initial_conditions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    model = load_compatible_pretrained(policy_path, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    planning_model: torch.nn.Module = model
    if matrix is not None and bias is not None:
        planning_model = ActionAdaptedCostModel(model, matrix, bias).to(device).eval()
    config = swm.PlanConfig(
        horizon=horizon,
        receding_horizon=receding_horizon,
        history_len=3,
        action_block=action_block,
        warm_start=True,
    )
    action_normalizer = ActionNormalizer(
        action_mean,
        action_std,
        clip_output=clip_env_actions,
    )
    solver_kwargs = {
        "model": planning_model,
        "batch_size": 1,
        "num_samples": num_samples,
        "var_scale": var_scale,
        "n_steps": n_steps,
        "topk": topk,
        "device": device,
        "seed": seeds[0],
    }
    if phase_conditioned:
        solver = PhaseConditionedCEMSolver(
            **solver_kwargs,
            phase_bank=_cube_phase_bank(
                action_normalizer,
                horizon=horizon,
                action_block=action_block,
            ),
        )
    else:
        solver = swm.solver.CEMSolver(**solver_kwargs)
    process = {"action": action_normalizer}
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform={"pixels": _image_transform(), "goal": _image_transform()},
    )
    world = swm.World(
        env_name=environment,
        num_envs=len(seeds),
        image_shape=(224, 224),
        max_episode_steps=max_episode_steps,
        goal_conditioned=True,
        pre_wrappers=[CubeGoalAliasWrapper],
        mode="task",
        ob_type="pixels",
        reward_task_id=2,
        terminate_at_goal=True,
        render_mode="rgb_array",
    )
    seed_vector = None
    try:
        world.set_policy(policy)
        # Wait for the original env pool to finish instead of auto-resetting
        # completed envs.  This preserves one-to-one seed pairing between the
        # baseline and adapter arms.
        # CubeEnv intentionally samples task/visual variations with an
        # unseeded variation space.  For a genuine paired intervention, the
        # first arm snapshots qpos/qvel and variation values, and the second
        # arm restores those exact initial conditions instead of merely
        # reusing the same integer seed.
        world.reset(
            seed=seeds[0],
            options=reset_options if reset_options is not None else {"render_goal": True},
        )
        if reset_options is not None and len(reset_options) != len(seeds):
            raise ValueError("strict paired reset options must match the number of seeds")
        if capture_initial_conditions is not None:
            for env in world.envs.envs:
                raw = env.unwrapped
                capture_initial_conditions.append(
                    {
                        "render_goal": True,
                        "variation_values": copy.deepcopy(raw.variation_space.value),
                        "state": np.concatenate(
                            [raw.data.qpos.copy(), raw.data.qvel.copy()]
                        ),
                    }
                )
        world._run(max_steps=max_episode_steps, mode="wait")
        # These fields are evaluator-only diagnostics.  They are read after
        # the rollout and never enter WorldModelPolicy or CEM scoring.
        final_info = world.infos
        block_pos = np.asarray(final_info["privileged/block_0_pos"], dtype=np.float64)
        if block_pos.ndim == 3:
            block_pos = block_pos[:, -1, :]
        reward = np.asarray(final_info["reward"], dtype=np.float64)
        reward = reward[:, -1] if reward.ndim > 1 else reward
        success = np.asarray(final_info["success"], dtype=np.float64)
        success = success[:, -1] if success.ndim > 1 else success
        final_distances = []
        for env in world.envs.envs:
            raw = env.unwrapped
            target_block = int(getattr(raw, "_target_block", 0))
            goal_xyz = np.asarray(raw.cur_task_info["goal_xyzs"][target_block], dtype=np.float64)
            final_distances.append(float(np.linalg.norm(block_pos[len(final_distances)] - goal_xyz)))
        evaluator_metrics = {
            "final_block_goal_distance_m": final_distances,
            "final_reward": reward.tolist(),
            "final_success_info": success.astype(int).tolist(),
            "runtime_policy_state_access": False,
        }
        seed_vector = np.asarray(world.envs.seeds, dtype=np.int64).copy()
    finally:
        world.close()
    seeds = seed_vector if seed_vector is not None else np.asarray(result["seeds"], dtype=np.int64)
    # world.close() does not destroy the recorded seed vector in the current
    # EnvPool implementation; keep a defensive fallback for older versions.
    successes = np.asarray(evaluator_metrics["final_success_info"], dtype=np.float64)
    return {
        "success_rate": float(successes.mean()),
        "episode_successes": successes.astype(int).tolist(),
        "seeds": seeds.tolist(),
        "evaluator_metrics": evaluator_metrics,
        "action_transform_diagnostics": action_normalizer.diagnostics(),
        "phase_conditioned": bool(phase_conditioned),
        "phase_bank_diagnostics": (
            solver.phase_diagnostics() if phase_conditioned else None
        ),
    }


def _canonicalize(result: dict[str, Any]) -> dict[str, Any]:
    """Sort episode outcomes by seed so completion order cannot affect pairing."""
    order = np.argsort(np.asarray(result["seeds"], dtype=np.int64))
    result = dict(result)
    result["seeds"] = np.asarray(result["seeds"], dtype=np.int64)[order].tolist()
    result["episode_successes"] = np.asarray(result["episode_successes"], dtype=np.int64)[order].tolist()
    if "evaluator_metrics" in result:
        metrics = dict(result["evaluator_metrics"])
        for key in ("final_block_goal_distance_m", "final_reward", "final_success_info"):
            if key in metrics:
                metrics[key] = np.asarray(metrics[key])[order].tolist()
        result["evaluator_metrics"] = metrics
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/cube/lewm")
    parser.add_argument("--environment", default="swm/OGBCube-v0")
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--receding-horizon",
        type=int,
        default=5,
        help="Number of executed action-blocks before replanning; matches the released Cube eval protocol.",
    )
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=8)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--var-scale",
        type=float,
        default=1.0,
        help="Predeclared CEM candidate standard deviation in normalized action coordinates.",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=75,
        help="Predeclared evaluator episode budget; does not alter policy inputs.",
    )
    parser.add_argument(
        "--clip-env-actions",
        action="store_true",
        help=(
            "Clip inverse-transformed actions to the OGBench normalized action "
            "bounds [-1, 1] before sending them to the environment. This is "
            "an explicit protocol intervention and is off by default for "
            "backward-compatible comparison with prior artifacts."
        ),
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for planning audit but is unavailable")
    if args.episodes < 2 or args.topk >= args.num_samples:
        raise ValueError("episodes must be >=2 and topk must be smaller than num-samples")
    if args.var_scale <= 0.0:
        raise ValueError("var-scale must be positive")
    if args.max_episode_steps < args.horizon * args.action_block:
        raise ValueError("max-episode-steps must cover at least one planned horizon")
    matrix, bias, action_mean, action_std = _load_adapter(args.adapter_artifact, args.environment)
    seeds = [args.seed + i for i in range(args.episodes)]
    print("Running paired live RGB Cube planning: baseline versus calibration-only action adapter.")
    initial_conditions: list[dict[str, Any]] = []
    baseline = _canonicalize(_run_controller(
        policy_path=args.policy, environment=args.environment, device=args.device,
        seeds=seeds, action_mean=action_mean, action_std=action_std,
        matrix=None, bias=None, horizon=args.horizon,
        receding_horizon=args.receding_horizon, action_block=args.action_block,
        num_samples=args.num_samples, n_steps=args.n_steps, topk=args.topk,
        var_scale=args.var_scale,
        max_episode_steps=args.max_episode_steps,
        clip_env_actions=args.clip_env_actions,
        capture_initial_conditions=initial_conditions,
    ))
    adapted = _canonicalize(_run_controller(
        policy_path=args.policy, environment=args.environment, device=args.device,
        seeds=seeds, action_mean=action_mean, action_std=action_std,
        matrix=matrix, bias=bias, horizon=args.horizon,
        receding_horizon=args.receding_horizon, action_block=args.action_block,
        num_samples=args.num_samples, n_steps=args.n_steps, topk=args.topk,
        var_scale=args.var_scale,
        max_episode_steps=args.max_episode_steps,
        clip_env_actions=args.clip_env_actions,
        reset_options=initial_conditions,
    ))
    if baseline["seeds"] != adapted["seeds"]:
        raise RuntimeError("baseline and adapter did not evaluate the same seed set")
    baseline_success = np.asarray(baseline["episode_successes"], dtype=np.int64)
    adapted_success = np.asarray(adapted["episode_successes"], dtype=np.int64)
    paired_delta = adapted_success - baseline_success
    baseline_distance = np.asarray(
        baseline["evaluator_metrics"]["final_block_goal_distance_m"], dtype=np.float64
    )
    adapted_distance = np.asarray(
        adapted["evaluator_metrics"]["final_block_goal_distance_m"], dtype=np.float64
    )
    distance_delta = adapted_distance - baseline_distance
    payload = {
        "schema": "live_cube_action_adapter_planning.v1",
        "protocol": {
            "checkpoint": args.policy,
            "checkpoint_frozen": True,
            "adapter_fit_split": "calibration episodes from adapter artifact",
            "planning_test_outcomes_not_used_for_adapter": True,
            "environment": args.environment,
            "device": args.device,
            "same_seed_pairing": True,
            "strict_initial_state_pairing": True,
            "strict_initial_state_source": "baseline qpos/qvel plus variation_values restored for adapter",
            "seeds": seeds,
            "horizon": args.horizon,
            "receding_horizon": args.receding_horizon,
            "action_block": args.action_block,
            "cem": {"num_samples": args.num_samples, "n_steps": args.n_steps, "topk": args.topk},
            "cem_var_scale": args.var_scale,
            "max_episode_steps": args.max_episode_steps,
            "adapter_artifact": str(args.adapter_artifact),
            "action_output_clip": args.clip_env_actions,
            "action_space_bounds": [-1.0, 1.0],
        },
        "baseline": baseline,
        "action_adapter": adapted,
        "paired": {
            "seed_order": baseline["seeds"],
            "delta_adapter_minus_baseline": paired_delta.tolist(),
            "wins": int(np.sum(paired_delta > 0)),
            "losses": int(np.sum(paired_delta < 0)),
            "ties": int(np.sum(paired_delta == 0)),
            "success_rate_delta": float(paired_delta.mean()),
            "final_distance_adapter_minus_baseline_m": distance_delta.tolist(),
            "distance_wins": int(np.sum(distance_delta < 0.0)),
            "distance_losses": int(np.sum(distance_delta > 0.0)),
            "distance_ties": int(np.sum(distance_delta == 0.0)),
            "mean_final_distance_delta_m": float(distance_delta.mean()),
        },
        "claim_boundary": (
            "Independent live RGB closed-loop Cube consequence only; no universal "
            "world-model, causal-physics, cross-environment, or commutator claim."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "planning.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"baseline": baseline, "action_adapter": adapted}, indent=2))


if __name__ == "__main__":
    main()
