"""Closed-loop PushT planning comparison.

Runs the same seeded simulator episodes for random control, vanilla LeWM MPC,
and optionally spectral-cost LeWM MPC.  It records success, return, final
physical pose distance, action clipping, and wall-clock planning time.  The
script is intentionally separate from the dataset replay evaluator: actions
are selected from the current observation and executed in the live simulator.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.online_planning_protocol import (
    expected_action_block,
    validate_action_protocol,
)
from experiments.spectral_planner import wrap_model_with_spectral_cost


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--policies", nargs="+", default=["zero", "random", "heuristic", "vanilla", "spectral"], choices=["zero", "random", "heuristic", "vanilla", "spectral"])
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--seed", type=int, default=5072)
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--receding-horizon", type=int, default=1)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--cem-steps", type=int, default=2)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--spectral-latents", default=None)
    p.add_argument("--action-normalization", choices=["live_zscore", "identity", "raw_div100"], default="live_zscore")
    return p.parse_args()


def image_transform():
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=224),
    ])


class ClippedPolicy(swm.policy.BasePolicy):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.type = getattr(inner, "type", "clipped")
        self.last_plan_seconds = 0.0

    def set_env(self, env):
        self.env = env
        self.inner.set_env(env)

    def get_action(self, info_dict, **kwargs):
        started = time.perf_counter()
        action = self.inner.get_action(info_dict, **kwargs)
        self.last_plan_seconds = time.perf_counter() - started
        return np.clip(action, -1.0, 1.0)


class PublicPushTActionTransform:
    """Map live relative actions to the released raw/100 action units.

    PushT's live action is a relative displacement in ``[-1, 1]`` and the
    simulator's pixel action scale is 100.  Thus ``raw_action / 100`` equals
    the live relative action before the public mean/std z-score.
    """

    mean = np.asarray([-0.0087, 0.0068], dtype=np.float64)
    std = np.asarray([0.2019, 0.2002], dtype=np.float64)

    def transform(self, value):
        return (np.asarray(value) - self.mean) / self.std

    def inverse_transform(self, value):
        return np.asarray(value) * self.std + self.mean


class IdentityActionTransform:
    def transform(self, value):
        return np.asarray(value)

    def inverse_transform(self, value):
        return np.asarray(value)


class RawDiv100ActionTransform:
    """External-data sensitivity arm: raw/100 followed by public z-score."""

    mean = PublicPushTActionTransform.mean
    std = PublicPushTActionTransform.std

    def transform(self, value):
        return (np.asarray(value) / 100.0 - self.mean) / self.std

    def inverse_transform(self, value):
        return (np.asarray(value) * self.std + self.mean) * 100.0


class ZeroPolicy(swm.policy.BasePolicy):
    def __init__(self):
        super().__init__()
        self.type = "zero"

    def get_action(self, obs, **kwargs):
        del obs, kwargs
        return np.zeros_like(self.env.action_space.sample())


class PrivilegedPushTHeuristic(swm.policy.BasePolicy):
    """A privileged simulator-state controller used only as a pipeline check."""

    def __init__(self):
        super().__init__()
        self.type = "privileged_heuristic"

    def get_action(self, info_dict, **kwargs):
        del kwargs
        state = np.asarray(info_dict["state"][:, 0], dtype=np.float64)
        goal = np.asarray(info_dict["goal_state"][:, 0], dtype=np.float64)
        agent = state[:, :2]
        block = state[:, 2:4]
        delta = goal[:, 2:4] - block
        norm = np.linalg.norm(delta, axis=1, keepdims=True)
        direction = delta / np.maximum(norm, 1e-6)
        behind = block - 50.0 * direction
        push_through = block + 100.0 * direction
        to_behind = np.linalg.norm(behind - agent, axis=1, keepdims=True)
        target = np.where(to_behind > 25.0, behind, push_through)
        return np.clip((target - agent) / 100.0, -1.0, 1.0).astype(np.float32)


def action_transform(args: argparse.Namespace):
    if args.action_normalization == "identity":
        return IdentityActionTransform()
    if args.action_normalization == "raw_div100":
        return RawDiv100ActionTransform()
    return PublicPushTActionTransform()


def make_policy(args: argparse.Namespace, name: str, action_dim: int = 2):
    if name == "random":
        return swm.policy.RandomPolicy(seed=args.seed)
    if name == "zero":
        return ZeroPolicy()
    if name == "heuristic":
        return PrivilegedPushTHeuristic()
    model = load_compatible_pretrained("pusht/lewm", device=args.device)
    validate_action_protocol(
        model, action_dim=action_dim, action_block=args.action_block
    )
    if name == "spectral":
        if not args.spectral_latents:
            raise ValueError("--spectral-latents is required for spectral policy")
        model = wrap_model_with_spectral_cost(
            model, args.spectral_latents, max_samples=200000, seed=args.seed,
            weight_mode="reachability_controllability_nuisance", normalize_weights=True,
        ).to(args.device).eval()
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=1,
        num_samples=args.samples,
        n_steps=args.cem_steps,
        topk=args.topk,
        device=args.device,
        seed=args.seed,
    )
    config = swm.PlanConfig(
        horizon=args.horizon,
        receding_horizon=args.receding_horizon,
        action_block=args.action_block,
    )
    return ClippedPolicy(swm.policy.WorldModelPolicy(
        solver=solver, config=config, process={"action": action_transform(args)},
        transform={"pixels": image_transform(), "goal": image_transform()},
    ))


def run_one(args: argparse.Namespace, name: str) -> dict:
    world = swm.World("swm/PushT-v1", num_envs=1, image_shape=(224, 224), max_episode_steps=100)
    action_dim = int(np.prod(world.envs.single_action_space.shape))
    policy = make_policy(args, name, action_dim=action_dim)
    world.set_policy(policy)
    episodes = []
    current = {"return": 0.0, "steps": 0, "plan_seconds": 0.0, "final_distance": float("nan"), "block_xy_distance": float("nan"), "clip_fraction": 0.0}
    pending_reset = False

    def on_step(w):
        nonlocal current, pending_reset
        if pending_reset:
            current = {"return": 0.0, "steps": 0, "plan_seconds": 0.0, "final_distance": float("nan"), "block_xy_distance": float("nan"), "clip_fraction": 0.0}
            pending_reset = False
        current["return"] += float(w.rewards[0])
        current["steps"] += 1
        current["plan_seconds"] += float(getattr(policy, "last_plan_seconds", 0.0))
        action = np.asarray(w.infos["action"][0, 0], dtype=np.float64)
        current["clip_fraction"] += float(np.mean(np.abs(action) >= 0.999))
        state = np.asarray(w.infos["state"][0, 0], dtype=np.float64)
        goal = np.asarray(w.infos["goal_state"][0, 0], dtype=np.float64)
        current["final_distance"] = float(np.linalg.norm(state - goal))
        current["block_xy_distance"] = float(np.linalg.norm(state[2:4] - goal[2:4]))

    def on_done(env_idx, ep_idx, w):
        nonlocal pending_reset
        del env_idx
        row = dict(current)
        row["success"] = bool(w.terminateds[0])
        row["episode"] = int(ep_idx)
        episodes.append(row)
        pending_reset = True

    start = time.perf_counter()
    try:
        world._run(episodes=args.episodes, seed=args.seed, mode="auto", on_step=on_step, on_done=on_done)
    finally:
        world.close()
    elapsed = time.perf_counter() - start
    success = np.asarray([row["success"] for row in episodes], dtype=np.float64)
    return {
        "policy": name,
        "episodes": episodes,
        "summary": {
            "n": len(episodes),
            "success_rate": float(success.mean()) if len(success) else float("nan"),
            "mean_return": float(np.mean([x["return"] for x in episodes])) if episodes else float("nan"),
            "mean_final_distance": float(np.mean([x["final_distance"] for x in episodes])) if episodes else float("nan"),
            "mean_block_xy_distance": float(np.mean([x["block_xy_distance"] for x in episodes])) if episodes else float("nan"),
            "mean_clip_fraction": float(np.mean([x["clip_fraction"] / max(x["steps"], 1) for x in episodes])) if episodes else float("nan"),
            "wall_seconds": elapsed,
        },
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    reports = [run_one(args, policy) for policy in args.policies]
    report = {"protocol": vars(args), "reports": reports, "sanity_checks": {"all_requested_policies_present": len(reports) == len(args.policies), "finite_summaries": bool(all(np.isfinite(r["summary"]["success_rate"]) for r in reports))}}
    (out / "online_planning.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))
    (out / "online_planning.md").write_text("# Online PushT planning\n\n" + json.dumps(report["reports"], indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
