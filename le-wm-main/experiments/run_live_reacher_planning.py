"""Paired live RGB planning on the non-contact Reacher benchmark.

The released Reacher checkpoint is trained for the qpos-match task, while the
raw environment does not expose a rendered goal image.  This runner creates a
goal image by rendering a deterministic target joint configuration after each
reset and restores the live state before the first policy action.  The target
is derived only from the reset state and is therefore identical for the random
and frozen-LeWM arms under the same seed.

Runtime policy inputs are rendered current RGB, rendered target RGB, and the
history/action fields assembled by stable-worldmodel.  Simulator qpos/qvel and
target qpos are used only by the evaluator/wrapper and are never inserted into
the policy info dict.
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path

# Keep the historical ``python experiments/run_*.py`` entry point usable as
# well as the package form.
if __package__ in (None, ""):
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import torch
# Import PyTorch before configuring MuJoCo's headless renderer.  In the
# managed CUDA runtime, exporting MUJOCO_GL in the shell before importing
# PyTorch can make the NVML-based availability probe return zero devices;
# configuring EGL after PyTorch is loaded avoids that launcher-specific
# interaction while still making rendering deterministic and headless.
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import gymnasium as gym
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
from torchvision.transforms import v2 as transforms

try:
    # Package invocation (``python -m experiments.run_live_reacher_planning``)
    # is the canonical path, while retaining direct-script compatibility for
    # the historical experiment entry point.
    from experiments.dmcontrol_compat import install_missing_field_compat
except ModuleNotFoundError:
    from dmcontrol_compat import install_missing_field_compat
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.capacity_readout_runtime import CapacityReadoutCostModel, load_capacity_readout
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer


class ReacherGoalImageWrapper(gym.Wrapper):
    """Inject a rendered qpos-match goal without exposing privileged state."""

    def __init__(self, env: gym.Env, target_delta=(0.65, -0.65)):
        super().__init__(env)
        self.target_delta = np.asarray(target_delta, dtype=np.float64)
        self._goal = None

    def _raw(self):
        # World applies this wrapper before MegaWrapper; the wrapped object is
        # the Gymnasium OrderEnforcing layer around ReacherDMControlWrapper.
        return self.env.unwrapped

    def _render_target_and_restore(self, current_qpos: np.ndarray, current_qvel: np.ndarray) -> np.ndarray:
        raw = self._raw()
        physics = raw.env.physics
        target_qpos = np.asarray(current_qpos, dtype=np.float64) + self.target_delta
        # Keep the target inside the standard Reacher joint range.
        target_qpos = (target_qpos + np.pi) % (2.0 * np.pi) - np.pi
        raw.env.task.target_qpos = target_qpos.copy()
        physics.data.qpos[:] = target_qpos
        physics.data.qvel[:] = 0.0
        physics.forward()
        goal = np.asarray(raw.render(width=224, height=224)).copy()
        physics.data.qpos[:] = current_qpos
        physics.data.qvel[:] = current_qvel
        physics.forward()
        return goal

    def _with_goal(self, info: dict, goal: np.ndarray) -> dict:
        info = dict(info)
        info["goal"] = goal.copy()
        # Do not pass target_qpos, qpos, qvel, or target_pos through info.
        for key in ("qpos", "qvel", "target_pos", "finger_pos"):
            info.pop(key, None)
        return info

    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        raw = self._raw()
        current_qpos = np.asarray(raw.env.physics.data.qpos).copy()
        current_qvel = np.asarray(raw.env.physics.data.qvel).copy()
        self._goal = self._render_target_and_restore(current_qpos, current_qvel)
        return observation, self._with_goal(info, self._goal)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if self._goal is None:
            raise RuntimeError("Reacher goal wrapper stepped before reset")
        return observation, reward, terminated, truncated, self._with_goal(info, self._goal)


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


def _planning_policy(args, action_mean: np.ndarray, action_std: np.ndarray):
    model = load_compatible_pretrained(args.policy, device=args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    if args.cost_metric == "capacity_safe_readout":
        if args.capacity_readout_artifact is None:
            raise ValueError(
                "--capacity-readout-artifact is required for capacity_safe_readout"
            )
        artifact = load_capacity_readout(Path(args.capacity_readout_artifact))
        if bool(artifact["gate_enabled"]):
            raise ValueError(
                "gated capacity readouts require the candidate-bank runner, which "
                "supplies _candidate_bank_mask; use run_live_reacher_sequence_"
                "candidate_planning.py instead of standard CEM"
            )
        model = CapacityReadoutCostModel(model, artifact, action_mean, action_std)
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=1,
        num_samples=args.num_samples,
        var_scale=1.0,
        n_steps=args.n_steps,
        topk=args.topk,
        device=args.device,
        seed=args.seed,
    )
    return swm.policy.WorldModelPolicy(
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


def _run(args, policy_kind: str, action_mean: np.ndarray, action_std: np.ndarray):
    if policy_kind == "random":
        policy = swm.policy.RandomPolicy(seed=args.seed)
    else:
        policy = _planning_policy(args, action_mean, action_std)
    world = swm.World(
        env_name="swm/ReacherDMControl-v0",
        num_envs=args.episodes,
        image_shape=(224, 224),
        max_episode_steps=args.max_episode_steps,
        goal_conditioned=True,
        pre_wrappers=[partial(ReacherGoalImageWrapper, target_delta=args.target_delta)],
        render_mode="rgb_array",
        task="qpos_match",
    )
    try:
        world.set_policy(policy)
        result = world.evaluate(
            episodes=args.episodes,
            seed=args.seed,
            reset_mode="wait",
        )
    finally:
        world.close()
    successes = np.asarray(result["episode_successes"], dtype=np.int64)
    return {
        "success_rate": float(successes.mean()),
        "episode_successes": successes.tolist(),
        "seeds": np.asarray(result["seeds"], dtype=np.int64).tolist(),
    }


def _canonicalize(result: dict) -> dict:
    order = np.argsort(np.asarray(result["seeds"], dtype=np.int64))
    result = dict(result)
    result["seeds"] = np.asarray(result["seeds"], dtype=np.int64)[order].tolist()
    result["episode_successes"] = np.asarray(result["episode_successes"], dtype=np.int64)[order].tolist()
    return result


def main() -> None:
    install_missing_field_compat()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="/home/gbk/.stable_worldmodel/checkpoints/reacher/lewm")
    parser.add_argument(
        "--adapter-artifact",
        type=Path,
        default=Path("results/live_cross_environment_prediction_20261015/reacher_cuda/cross_environment_prediction.json"),
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
    parser.add_argument(
        "--cost-metric",
        choices=("euclidean", "capacity_safe_readout"),
        default="euclidean",
    )
    parser.add_argument("--capacity-readout-artifact", type=Path, default=None)
    parser.add_argument(
        "--target-delta",
        type=float,
        nargs=2,
        default=(0.65, -0.65),
        metavar=("DELTA_Q1", "DELTA_Q2"),
        help="Predeclared target-qpos offset used only to render the evaluator-side goal image.",
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for live Reacher planning but unavailable")
    if args.topk >= args.num_samples or args.episodes < 2:
        raise ValueError("require episodes >= 2 and topk < num_samples")
    environment = "swm/ReacherDMControl-v0"
    action_mean, action_std = _load_action_stats(args.adapter_artifact, environment)
    print("Running live RGB Reacher: frozen LeWM CEM versus random, same seeds.")
    random_result = _canonicalize(_run(args, "random", action_mean, action_std))
    planning_result = _canonicalize(_run(args, "planning", action_mean, action_std))
    if random_result["seeds"] != planning_result["seeds"]:
        raise RuntimeError("random and planning did not evaluate the same seed set")
    random_success = np.asarray(random_result["episode_successes"], dtype=np.int64)
    planning_success = np.asarray(planning_result["episode_successes"], dtype=np.int64)
    paired_delta = planning_success - random_success
    payload = {
        "schema": "live_reacher_planning.v1",
        "protocol": {
            "checkpoint": args.policy,
            "checkpoint_frozen": True,
            "task": "qpos_match",
            "runtime_input": "rendered RGB current frame, rendered RGB target-qpos frame, and action/history only",
            "runtime_state_access": False,
            "same_seed_pairing": True,
            "goal_construction": "same-seed reset qpos plus fixed delta [0.65,-0.65], rendered evaluator-side, then live state restored",
            "target_delta": [float(x) for x in args.target_delta],
            "device": args.device,
            "dmcontrol_missing_field_compat": "installed",
            "cem": {"num_samples": args.num_samples, "n_steps": args.n_steps, "topk": args.topk},
            "horizon": args.horizon,
            "receding_horizon": args.receding_horizon,
            "action_block": args.action_block,
            "cost_metric": args.cost_metric,
            "capacity_readout_artifact": (
                str(args.capacity_readout_artifact)
                if args.capacity_readout_artifact else None
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
        "claim_boundary": "Independent live Reacher RGB planning consequence; not a universal world-model claim.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "planning.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"random": random_result, "frozen_lewm": planning_result}, indent=2))


if __name__ == "__main__":
    main()
