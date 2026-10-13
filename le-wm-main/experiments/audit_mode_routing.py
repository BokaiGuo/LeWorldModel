"""Audit oracle-versus-predicted contact-mode routing on a fixed checkpoint.

The audit regenerates the checkpoint's held-out validation roots and reports
the runtime soft mixture alongside an oracle-mode diagnostic.  Simulator state
is used only to recreate offline labels and to score the audit; no state or
successor is supplied to the model's runtime pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import stable_worldmodel  # noqa: F401
import torch

from run_causal_visual_pusht_world_model import (
    Normalization,
    VisualCausalWorldModel,
    balanced_source_cases,
    bootstrap_closed_loop_source_cases,
    make_counterfactual_dataset,
    physical_metrics,
    terminal_pose_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(checkpoint: Path, device: str) -> tuple[VisualCausalWorldModel, Normalization, dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    saved = dict(payload["args"])
    model = VisualCausalWorldModel(
        int(saved["latent_dim"]), str(saved["vision_encoder"]), bool(saved["temporal_context"]),
        str(saved.get("utility_mode", "scalar")), str(saved.get("adapter_architecture", "legacy")),
        bool(saved.get("structured_readout", False)), bool(saved.get("contact_frame_dynamics", False)),
        bool(saved.get("mode_conditioned_dynamics", False)), bool(saved.get("current_only_readout", False)),
        bool(saved.get("torque_features", False)), bool(saved.get("temporal_dynamics", False)),
        bool(saved.get("previous_action_dynamics", False)), bool(saved.get("explicit_angle_effect", False)),
        bool(saved.get("explicit_angle_readout", False)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    normalizer = Normalization(**payload["normalization"])
    mean, std = normalizer.tensors(device)
    model.set_state_normalization(mean, std)
    return model, normalizer, saved


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, normalizer, saved = load_model(args.checkpoint, args.device)
    runtime = SimpleNamespace(**saved)
    data_seed = int(saved.get("data_seed", saved["seed"]))
    val_roots_count = int(saved["val_roots"])
    terminal_count = int(round(val_roots_count * float(saved.get("terminal_root_fraction", 0.0))))
    bootstrap_count = int(round(val_roots_count * float(saved.get("bootstrap_root_fraction", 0.0))))
    env = gym.make(
        "swm/PushT-v1", render_mode="rgb_array",
        max_episode_steps=int(saved["planning_steps"]) * int(saved["action_repeat"]),
    )
    try:
        roots = balanced_source_cases(
            env, val_roots_count - terminal_count - bootstrap_count,
            int(saved["root_rollout_steps"]), data_seed + 10_000,
            float(saved["contact_root_fraction"]), int(saved["contact_approach_steps"]),
        )
        roots.extend(terminal_pose_cases(env, terminal_count, data_seed + 410_000))
        if bootstrap_count:
            bootstrap = Path(str(saved["bootstrap_checkpoint"]))
            if not bootstrap.is_absolute():
                bootstrap = Path.cwd() / bootstrap
            roots.extend(bootstrap_closed_loop_source_cases(
                env, bootstrap, bootstrap_count, data_seed + 510_000,
                int(saved["bootstrap_rollout_episodes"]), int(saved["bootstrap_rollout_steps"]),
                float(saved["bootstrap_contact_distance"]), args.device,
            ))
        data = make_counterfactual_dataset(
            env, roots, int(saved["image_size"]), data_seed + 30_000,
            int(saved["action_repeat"]), int(saved["counterfactual_fan_size"]),
            str(saved.get("train_action_support", "isotropic")),
            float(saved.get("train_action_max_magnitude", 0.85)),
        )
    finally:
        env.close()
    metrics = physical_metrics(model, data, normalizer, args.device)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_roots": len(roots),
        "validation_rows": len(data["action"]),
        "metrics": metrics,
        "protocol": {
            "data_seed": data_seed,
            "source_train_validation_disjoint": True,
            "runtime_policy_state_access": False,
            "oracle_mode_pass": "diagnostic_only",
            "runtime_pass": "predicted_soft_mode_mixture",
        },
    }
    (args.output / "mode_routing_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
