"""Evaluate one saved causal visual world-model checkpoint under a fixed interface.

This runner deliberately separates checkpoint training from controller ablations:
the loaded model and normalization are immutable, while action repeat, temporal
input, CEM budget, and the optional visual contact servo are recorded in the
output.  It therefore supports matched planning-interface comparisons without
silently re-training a different dynamics model for each controller.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401: registers PushT
import torch

try:  # package import for tests and composed experiment runners
    from .run_causal_visual_pusht_world_model import (
        PUSHT_RENDER_SIZE,
        Normalization,
        VisualCausalWorldModel,
        evaluate_visual_planning,
    )
except ImportError:  # direct ``python experiments/evaluate_*.py`` execution
    from run_causal_visual_pusht_world_model import (  # type: ignore[no-redef]
        PUSHT_RENDER_SIZE,
        Normalization,
        VisualCausalWorldModel,
        evaluate_visual_planning,
    )


def checkpoint_utility_mode(payload: dict) -> str:
    """Recover the architecture contract for old checkpoints without guessing.

    Early scalar checkpoints predate the explicit ``utility_mode`` field, and
    the exploratory phase checkpoint also has no such field.  The final layer
    width is an immutable property of the saved state dict, so it is the only
    admissible backward-compatible source of truth.
    """
    saved = payload.get("args", {})
    declared = saved.get("utility_mode")
    width = int(payload["model"]["utility_head.4.weight"].shape[0])
    inferred = {1: "scalar", 2: "phase"}.get(width)
    if inferred is None:
        raise ValueError(f"unsupported utility-head width: {width}")
    if declared is not None and declared != inferred:
        raise ValueError(f"checkpoint utility_mode={declared!r} conflicts with saved head width {width}")
    return inferred


def checkpoint_adapter_architecture(payload: dict) -> str:
    """Infer the adapter architecture from immutable state-dict keys."""
    declared = payload.get("args", {}).get("adapter_architecture")
    inferred = "mode_uncertain" if any(key.startswith("effect_log_variance.") for key in payload["model"]) else "legacy"
    if declared is not None and declared != inferred:
        raise ValueError(f"checkpoint adapter_architecture={declared!r} conflicts with saved state dict")
    return inferred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--eval-seed", type=int, default=None,
        help=(
            "seed for the held-out rendered-RGB episode set; defaults to the immutable "
            "checkpoint seed for backward compatibility"
        ),
    )
    parser.add_argument(
        "--orientation-checkpoint", default=None,
        help="optional immutable RGB-only local controller used after the visual orientation-phase latch",
    )
    parser.add_argument(
        "--orientation-action-repeat", type=int, default=None,
        help="simulator steps per action after the RGB orientation latch; defaults to --action-repeat",
    )
    parser.add_argument(
        "--reversible-orientation-phase", action="store_true",
        help="return to the base translation controller if RGB geometry leaves the orientation tube",
    )
    parser.add_argument(
        "--phase-template-only", action="store_true",
        help="during RGB terminal phase, rank only the fixed feasible contact-torque fan",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--planning-episodes", type=int, default=12)
    parser.add_argument("--planning-steps", type=int, default=100)
    parser.add_argument("--planning-horizon", type=int, default=6)
    parser.add_argument("--cem-samples", type=int, default=192)
    parser.add_argument("--cem-iterations", type=int, default=4)
    parser.add_argument("--action-repeat", type=int, default=1)
    parser.add_argument("--contact-servo", action="store_true")
    parser.add_argument("--contact-distance", type=float, default=42.0)
    parser.add_argument("--contact-tolerance", type=float, default=24.0)
    parser.add_argument("--visual-contact-gate", action="store_true")
    parser.add_argument("--uncertainty-risk-weight", type=float, default=0.25)
    parser.add_argument("--separation-risk-weight", type=float, default=0.5)
    parser.add_argument("--contact-preservation-weight", type=float, default=1.0)
    parser.add_argument("--orientation-phase-distance", type=float, default=35.0)
    parser.add_argument("--orientation-translation-weight", type=float, default=0.03)
    parser.add_argument("--visual-terminal-xy", type=float, default=20.0)
    parser.add_argument("--visual-terminal-angle", type=float, default=math.pi / 9.0)
    parser.add_argument("--orientation-action-max", type=float, default=1.0)
    parser.add_argument("--direct-utility", action="store_true", help="use one-step direct utility/task-space candidate ranking")
    parser.add_argument("--task-space-objective", action="store_true", help="use fixed raw PushT task-space successor ranking")
    parser.add_argument("--successor-orientation-phase", action="store_true", help="rank terminal-phase candidates by learned successor orientation")
    parser.add_argument("--geometry-state-readout", action="store_true", help="use RGB mask centroids for runtime agent/block positions")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action_repeat < 1:
        raise ValueError("action_repeat must be positive")
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    saved = payload["args"]
    runtime = SimpleNamespace(**saved)
    for name in (
        "planning_episodes", "planning_steps", "planning_horizon", "cem_samples", "cem_iterations",
        "action_repeat", "contact_servo", "contact_distance", "contact_tolerance", "visual_contact_gate", "uncertainty_risk_weight",
        "separation_risk_weight", "contact_preservation_weight", "direct_utility", "task_space_objective", "successor_orientation_phase", "geometry_state_readout", "device",
        "orientation_phase_distance",
        "orientation_translation_weight",
        "visual_terminal_xy", "visual_terminal_angle",
        "orientation_action_max",
        "reversible_orientation_phase",
        "phase_template_only",
    ):
        setattr(runtime, name, getattr(args, name))
    # Keep test-case generation independent of model initialization.  This makes
    # separately trained checkpoints comparable episode-by-episode without
    # granting either controller access to simulator state at runtime.
    runtime.seed = int(saved.get("seed", 0) if args.eval_seed is None else args.eval_seed)
    runtime.utility_mode = checkpoint_utility_mode(payload)
    runtime.adapter_architecture = checkpoint_adapter_architecture(payload)
    model = VisualCausalWorldModel(
        int(runtime.latent_dim), str(runtime.vision_encoder), bool(runtime.temporal_context), runtime.utility_mode,
        runtime.adapter_architecture, bool(getattr(runtime, "structured_readout", False)),
        bool(getattr(runtime, "contact_frame_dynamics", False)),
        bool(getattr(runtime, "mode_conditioned_dynamics", False)),
        bool(getattr(runtime, "current_only_readout", False)),
        bool(getattr(runtime, "torque_features", False)),
        bool(getattr(runtime, "temporal_dynamics", False)),
        bool(getattr(runtime, "previous_action_dynamics", False)),
        bool(getattr(runtime, "explicit_angle_effect", False)),
        bool(getattr(runtime, "explicit_angle_readout", False)),
    ).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    normalizer = Normalization(**payload["normalization"])
    mean, std = normalizer.tensors(args.device)
    model.set_state_normalization(mean, std)
    orientation_repeat = args.action_repeat if args.orientation_action_repeat is None else args.orientation_action_repeat
    if orientation_repeat < 1:
        raise ValueError("orientation_action_repeat must be positive")
    runtime.orientation_action_repeat = int(orientation_repeat)
    env = gym.make(
        "swm/PushT-v1", render_mode="rgb_array",
        max_episode_steps=runtime.planning_steps * max(runtime.action_repeat, runtime.orientation_action_repeat),
    )
    orientation_model = orientation_normalizer = None
    if args.orientation_checkpoint:
        orientation_payload = torch.load(args.orientation_checkpoint, map_location=args.device, weights_only=False)
        orientation_saved = orientation_payload["args"]
        orientation_model = VisualCausalWorldModel(
            int(orientation_saved["latent_dim"]), str(orientation_saved["vision_encoder"]),
            bool(orientation_saved["temporal_context"]), checkpoint_utility_mode(orientation_payload),
            checkpoint_adapter_architecture(orientation_payload), bool(orientation_saved.get("structured_readout", False)),
            bool(orientation_saved.get("contact_frame_dynamics", False)),
            bool(orientation_saved.get("mode_conditioned_dynamics", False)),
            bool(orientation_saved.get("current_only_readout", False)), bool(orientation_saved.get("torque_features", False)),
            bool(orientation_saved.get("temporal_dynamics", False)),
            bool(orientation_saved.get("previous_action_dynamics", False)),
            bool(orientation_saved.get("explicit_angle_effect", False)),
            bool(orientation_saved.get("explicit_angle_readout", False)),
        ).to(args.device)
        orientation_model.load_state_dict(orientation_payload["model"])
        orientation_model.eval()
        orientation_normalizer = Normalization(**orientation_payload["normalization"])
        orientation_mean, orientation_std = orientation_normalizer.tensors(args.device)
        orientation_model.set_state_normalization(orientation_mean, orientation_std)
    try:
        episodes = evaluate_visual_planning(
            env, model, normalizer, runtime, orientation_model, orientation_normalizer,
        )
    finally:
        env.close()
    success = np.asarray([row["strict_success"] for row in episodes], dtype=np.float64)
    report = {
        "protocol": {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "eval_seed": int(runtime.seed),
                "orientation_checkpoint": str(Path(args.orientation_checkpoint).resolve()) if args.orientation_checkpoint else None,
            "checkpoint_immutable": True,
            "render_coordinate_scale": float(PUSHT_RENDER_SIZE),
            "runtime_input": "previous/current rendered image and goal image only" if runtime.temporal_context else "current rendered image and goal image only",
            "runtime_state_access": False,
            "runtime_successor_verification": False,
            "interface": {
                "contact_servo": bool(runtime.contact_servo),
                "contact_distance": float(runtime.contact_distance),
                "contact_tolerance": float(runtime.contact_tolerance),
                "visual_contact_gate": bool(runtime.visual_contact_gate),
                "action_repeat": int(runtime.action_repeat),
                "orientation_action_repeat": int(runtime.orientation_action_repeat),
                "reversible_orientation_phase": bool(runtime.reversible_orientation_phase),
                "phase_template_only": bool(runtime.phase_template_only),
                "planning_horizon": int(runtime.planning_horizon),
                "cem_samples": int(runtime.cem_samples),
                "cem_iterations": int(runtime.cem_iterations),
                "direct_utility": bool(runtime.direct_utility),
                "task_space_objective": bool(runtime.task_space_objective),
                "successor_orientation_phase": bool(runtime.successor_orientation_phase),
                "geometry_state_readout": bool(runtime.geometry_state_readout),
                "utility_mode": str(runtime.utility_mode),
                "adapter_architecture": str(runtime.adapter_architecture),
                "orientation_phase_distance": float(runtime.orientation_phase_distance),
                "orientation_translation_weight": float(runtime.orientation_translation_weight),
                "visual_terminal_xy": float(runtime.visual_terminal_xy),
                "visual_terminal_angle": float(runtime.visual_terminal_angle),
                "orientation_action_max": float(runtime.orientation_action_max),
            },
        },
        "planning": {
            "episodes": episodes,
            "strict_success_count": int(success.sum()),
            "strict_success_rate": float(success.mean()),
            "mean_block_xy_distance": float(np.mean([row["block_xy_distance"] for row in episodes])),
            "mean_angle_error": float(np.mean([row["angle_error"] for row in episodes])),
        },
        "claim_boundary": "A matched controller-interface evaluation of one newly trained visual world model; not frozen-LeWM evidence or a universal world-model result.",
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoint_planning.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "checkpoint_planning.md").write_text(
        "# Matched causal visual checkpoint planning evaluation\n\n"
        f"- Strict success: {report['planning']['strict_success_count']}/{len(episodes)}.\n"
        f"- Mean terminal block XY: {report['planning']['mean_block_xy_distance']:.3f}.\n"
        f"- Mean terminal angle: {report['planning']['mean_angle_error']:.3f}.\n"
    )
    print(json.dumps({
        "output": str(output),
        "strict_success_count": report["planning"]["strict_success_count"],
        "strict_success_rate": report["planning"]["strict_success_rate"],
        "mean_block_xy_distance": report["planning"]["mean_block_xy_distance"],
        "mean_angle_error": report["planning"]["mean_angle_error"],
    }, indent=2))


if __name__ == "__main__":
    main()
