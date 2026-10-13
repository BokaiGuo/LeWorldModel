"""Evaluate immutable visual learned-model snapshots on independent evidence splits.

This runner closes the evidence gap in ``real_checkpoint_selection.v1``.  The
diagnostic export used to fit ``A, B, D`` is not reused for either of the two
external quantities below:

* ``action_inverse_risk`` is computed on a validation source-root split with a
  matched ``+a/-a/zero`` intervention fan;
* ``planning_risk`` is computed by RGB-only closed-loop rollouts on a fixed,
  fresh episode set.

Simulator state is used only to construct intervention labels and to score the
final planning consequence.  It is never passed to the model or the runtime
controller.  The output is deliberately a compact evidence record, rather
than a replacement checkpoint-selection record; the latter is assembled only
after this runner has completed for every checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_causal_visual_pusht_world_model import (
    Normalization,
    VisualCausalWorldModel,
    balanced_source_cases,
    evaluate_visual_planning,
    make_counterfactual_dataset,
    physical_metrics,
)


def _model_from_payload(payload: dict[str, Any], device: str) -> tuple[VisualCausalWorldModel, Normalization, dict[str, Any]]:
    saved = dict(payload["args"])
    model = VisualCausalWorldModel(
        int(saved["latent_dim"]),
        str(saved["vision_encoder"]),
        bool(saved["temporal_context"]),
        str(saved.get("utility_mode", "scalar")),
        str(saved.get("adapter_architecture", "legacy")),
        bool(saved.get("structured_readout", False)),
        bool(saved.get("contact_frame_dynamics", False)),
        bool(saved.get("mode_conditioned_dynamics", False)),
        bool(saved.get("current_only_readout", False)),
        bool(saved.get("torque_features", False)),
        bool(saved.get("temporal_dynamics", False)),
        bool(saved.get("previous_action_dynamics", False)),
        bool(saved.get("explicit_angle_effect", False)),
        bool(saved.get("explicit_angle_readout", False)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    normalizer = Normalization(**payload["normalization"])
    mean, std = normalizer.tensors(device)
    model.set_state_normalization(mean, std)
    return model, normalizer, saved


def _validation_intervention(
    model: VisualCausalWorldModel,
    normalizer: Normalization,
    saved: dict[str, Any],
    env: gym.Env,
    *,
    device: str,
) -> dict[str, Any]:
    """Run the immutable validation action fan used by the training protocol."""
    data_seed = int(saved.get("data_seed", saved["seed"]))
    val_roots = balanced_source_cases(
        env,
        int(saved["val_roots"]),
        int(saved["root_rollout_steps"]),
        data_seed + 10_000,
        float(saved["contact_root_fraction"]),
        int(saved["contact_approach_steps"]),
    )
    data = make_counterfactual_dataset(
        env,
        val_roots,
        int(saved["image_size"]),
        data_seed + 30_000,
        int(saved["action_repeat"]),
        int(saved.get("counterfactual_fan_size", 0)),
        str(saved["train_action_support"]),
        float(saved["train_action_max_magnitude"]),
    )
    metrics = physical_metrics(model, data, normalizer, device)
    inverse_risk = float(metrics["paired_plus_minus_effect_error"])
    ranking_risk = float(1.0 - metrics["paired_action_motion_top1"])
    if not np.isfinite(inverse_risk) or not np.isfinite(ranking_risk):
        raise RuntimeError("validation intervention produced a non-finite action risk")
    return {
        "action_inverse_risk": inverse_risk,
        "action_ranking_risk": ranking_risk,
        "action_identifiability": float(metrics["paired_action_motion_top1"]),
        "paired_plus_minus_effect_error": inverse_risk,
        "paired_action_motion_top1": float(metrics["paired_action_motion_top1"]),
        "paired_orientation_top1": float(metrics["paired_orientation_top1"]),
        "paired_roots": int(metrics["paired_roots"]),
        "contact_paired_plus_minus_effect_error": float(metrics["contact_paired_plus_minus_effect_error"]),
        "contact_paired_action_motion_top1": float(metrics["contact_paired_action_motion_top1"]),
        "contact_paired_roots": int(metrics["contact_paired_roots"]),
        "protocol": {
            "split": "immutable validation source roots from checkpoint data protocol",
            "intervention": "+a/-a/zero matched counterfactual fan",
            "root_count": len(val_roots),
            "row_count": int(len(data["action"])),
            "simulator_state_used_for": "offline intervention labels only",
            "model_runtime_input": "rendered current/previous RGB and action only",
        },
    }


def _planning_consequence(
    model: VisualCausalWorldModel,
    normalizer: Normalization,
    saved: dict[str, Any],
    env: gym.Env,
    *,
    device: str,
    planning_seed: int,
    planning_episodes: int,
    planning_steps: int,
    cem_samples: int,
    cem_iterations: int,
) -> dict[str, Any]:
    """Evaluate the frozen snapshot on a common fresh RGB-only episode panel."""
    runtime = SimpleNamespace(**saved)
    runtime.device = device
    runtime.seed = int(planning_seed)
    runtime.planning_episodes = int(planning_episodes)
    runtime.planning_steps = int(planning_steps)
    runtime.cem_samples = int(cem_samples)
    runtime.cem_iterations = int(cem_iterations)
    runtime.action_repeat = 1
    runtime.contact_servo = True
    runtime.contact_distance = float(saved.get("contact_distance", 42.0))
    runtime.contact_tolerance = float(saved.get("contact_tolerance", 24.0))
    runtime.visual_contact_gate = False
    runtime.direct_utility = True
    runtime.task_space_objective = False
    runtime.successor_orientation_phase = False
    runtime.geometry_state_readout = False
    runtime.orientation_action_repeat = 1
    runtime.reversible_orientation_phase = False
    runtime.phase_template_only = False
    episodes = evaluate_visual_planning(env, model, normalizer, runtime)
    if not episodes:
        raise RuntimeError("planning evaluator returned no episodes")
    success = np.asarray([bool(row["strict_success"]) for row in episodes], dtype=np.float64)
    xy = np.asarray([float(row["block_xy_distance"]) for row in episodes], dtype=np.float64)
    angle = np.asarray([float(row["angle_error"]) for row in episodes], dtype=np.float64)
    # This is a predeclared external task consequence.  Lower is better and
    # all candidates use the same fresh episode seeds and runtime budget.
    pose_cost = xy / 224.0 + angle / np.pi
    planning_risk = float(np.mean(1.0 - success + pose_cost))
    return {
        "planning_risk": planning_risk,
        "planning_failure_rate": float(1.0 - success.mean()),
        "strict_success_rate": float(success.mean()),
        "mean_block_xy_distance": float(xy.mean()),
        "mean_angle_error": float(angle.mean()),
        "mean_pose_cost": float(pose_cost.mean()),
        "episodes": [
            {
                "episode": int(row["episode"]),
                "strict_success": bool(row["strict_success"]),
                "block_xy_distance": float(row["block_xy_distance"]),
                "angle_error": float(row["angle_error"]),
                "steps": int(row["steps"]),
            }
            for row in episodes
        ],
        "protocol": {
            "split": "fresh fixed planning episode panel",
            "planning_seed": int(planning_seed),
            "planning_episodes": int(planning_episodes),
            "planning_steps": int(planning_steps),
            "cem_samples": int(cem_samples),
            "cem_iterations": int(cem_iterations),
            "runtime_input": "previous/current rendered RGB and goal RGB only",
            "runtime_state_access": False,
            "runtime_successor_verification": False,
            "test_planning_used_for_snapshot_selection": False,
            "simulator_state_used_for": "final consequence scoring only",
        },
    }


def evaluate_snapshot(
    snapshot: Path,
    output: Path,
    *,
    device: str,
    planning_seed: int,
    planning_episodes: int,
    planning_steps: int,
    cem_samples: int,
    cem_iterations: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    payload = torch.load(snapshot, map_location=device, weights_only=False)
    model, normalizer, saved = _model_from_payload(payload, device)
    env = gym.make(
        "swm/PushT-v1",
        render_mode="rgb_array",
        max_episode_steps=max(int(planning_steps) * 2, 256),
    )
    try:
        intervention = _validation_intervention(model, normalizer, saved, env, device=device)
        planning = _planning_consequence(
            model,
            normalizer,
            saved,
            env,
            device=device,
            planning_seed=planning_seed,
            planning_episodes=planning_episodes,
            planning_steps=planning_steps,
            cem_samples=cem_samples,
            cem_iterations=cem_iterations,
        )
    finally:
        env.close()
    epoch = int(payload.get("snapshot", {}).get("epoch", saved.get("epochs", -1)))
    report = {
        "schema": "learned_snapshot_external_evidence.v1",
        "checkpoint": str(snapshot.resolve()),
        "architecture": str(saved.get("adapter_architecture", "unknown")),
        "seed": int(saved["seed"]),
        "environment": "PushT",
        "training_stage": epoch,
        "latent_dimension": int(saved["latent_dim"]),
        "action_distribution": f"{saved['train_action_support']}_counterfactual_v1",
        "action_risk": intervention,
        "planning_consequence": planning,
        "protocol": {
            "checkpoint_immutable": True,
            "diagnostic_export_not_reused_for_external_metrics": True,
            "action_risk_split": intervention["protocol"],
            "planning_split": planning["protocol"],
            "wall_seconds": time.perf_counter() - started,
        },
        "claim_boundary": (
            "External evidence for this immutable learned visual checkpoint: matched action intervention risk "
            "and fresh RGB-only planning consequence on PushT. This does not establish cross-environment transfer."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "external_evidence.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "external_evidence.md").write_text(
        "# Learned snapshot external evidence\n\n"
        f"- Checkpoint: `{snapshot}`\n"
        f"- Action inverse risk: `{intervention['action_inverse_risk']:.6g}`\n"
        f"- Action identifiability: `{intervention['action_identifiability']:.6g}`\n"
        f"- Planning risk: `{planning['planning_risk']:.6g}`\n"
        f"- RGB-only strict success: `{planning['strict_success_rate']:.3f}`\n"
        "\nState is used only for offline labels/consequence scoring; it is not a runtime input.\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--planning-seed", type=int, default=20261013)
    parser.add_argument("--planning-episodes", type=int, default=12)
    parser.add_argument("--planning-steps", type=int, default=100)
    parser.add_argument("--cem-samples", type=int, default=96)
    parser.add_argument("--cem-iterations", type=int, default=3)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    report = evaluate_snapshot(
        args.snapshot,
        args.output,
        device=args.device,
        planning_seed=args.planning_seed,
        planning_episodes=args.planning_episodes,
        planning_steps=args.planning_steps,
        cem_samples=args.cem_samples,
        cem_iterations=args.cem_iterations,
    )
    print(json.dumps({
        "checkpoint": report["checkpoint"],
        "training_stage": report["training_stage"],
        "action_inverse_risk": report["action_risk"]["action_inverse_risk"],
        "planning_risk": report["planning_consequence"]["planning_risk"],
        "strict_success_rate": report["planning_consequence"]["strict_success_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
