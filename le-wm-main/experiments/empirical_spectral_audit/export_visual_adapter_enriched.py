"""Export real trained visual PushT adapters into the common enriched schema.

This exporter is intentionally separate from the JEPA/DINO replay evaluators.
The checkpoints consumed here are the intervention-supervised visual adapters
under ``results/strong_claim_protocol_v2_20261009/checkpoints``.  Their learned
transition interface is a six-dimensional normalized observable-pose chart;
that chart is the exported latent, while the full simulator state is retained
only as an offline target.  No state is passed to the model during prediction.

The same deterministic action protocol and the same checkpoint-specific
episode split are used for every candidate.  ``--split fit`` and
``--split outcome`` are intended to be exported separately, so the downstream
record builder can enforce episode-disjoint metric fitting and held-out risk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.empirical_spectral_audit.enriched_record_schema import make_payload, sha256_file  # noqa: E402
from experiments.empirical_spectral_audit.visual_adapter_export_metadata import (  # noqa: E402
    architecture_name,
    training_stage,
)
from experiments.run_causal_visual_pusht_world_model import (  # noqa: E402
    Normalization,
    VisualCausalWorldModel,
    observable_pose_vector,
    resize_image,
    tensor_images,
)


HORIZONS = (1, 2, 4)
ACTION_DISTRIBUTION = "pusht_deterministic_isotropic_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("fit", "outcome"), required=True)
    parser.add_argument("--roots", type=int, default=64)
    parser.add_argument("--root-offset", type=int, default=0)
    parser.add_argument(
        "--training-stage",
        type=int,
        default=None,
        help="Recorded checkpoint stage; inferred from epoch_XXXX in snapshot paths when omitted.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20261011)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(checkpoint: Path, device: torch.device) -> tuple[VisualCausalWorldModel, Normalization, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    saved = dict(payload["args"])
    model = VisualCausalWorldModel(
        latent_dim=int(saved["latent_dim"]),
        vision_encoder=str(saved["vision_encoder"]),
        temporal_context=bool(saved.get("temporal_context", False)),
        utility_mode=str(saved.get("utility_mode", "scalar")),
        adapter_architecture=str(saved.get("adapter_architecture", "legacy")),
        structured_readout=bool(saved.get("structured_readout", False)),
        contact_frame_dynamics=bool(saved.get("contact_frame_dynamics", False)),
        mode_conditioned_dynamics=bool(saved.get("mode_conditioned_dynamics", False)),
        current_only_readout=bool(saved.get("current_only_readout", False)),
        torque_features=bool(saved.get("torque_features", False)),
        temporal_dynamics=bool(saved.get("temporal_dynamics", False)),
        previous_action_dynamics=bool(saved.get("previous_action_dynamics", False)),
        explicit_angle_effect=bool(saved.get("explicit_angle_effect", False)),
        explicit_angle_readout=bool(saved.get("explicit_angle_readout", False)),
        explicit_goal_angle_readout=bool(saved.get("explicit_goal_angle_readout", False)),
        rich_temporal_dynamics=bool(saved.get("rich_temporal_dynamics", False)),
        invariant_angle_readout=bool(saved.get("invariant_angle_readout", False)),
        contact_interface_dynamics=bool(saved.get("contact_interface_dynamics", False)),
        angle_readout_current_only=bool(saved.get("angle_readout_current_only", False)),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    normalizer = Normalization(**payload["normalization"])
    mean, std = normalizer.tensors(str(device))
    model.set_state_normalization(mean, std)
    return model, normalizer, saved


def _action(root_id: int, horizon: int, seed: int) -> np.ndarray:
    """Fixed action draw shared by every architecture and seed."""
    rng = np.random.default_rng(seed + 104729 * int(root_id) + 1009 * int(horizon))
    direction = rng.normal(size=2).astype(np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    magnitude = float(rng.uniform(0.18, 0.85))
    return (magnitude * direction).astype(np.float32)


def _seed(checkpoint: Path, saved: dict[str, Any]) -> str:
    match = re.search(r"seed[_-](\d+)", str(checkpoint))
    return match.group(1) if match else str(saved.get("seed", "unknown"))


def _raw_state(state: np.ndarray) -> np.ndarray:
    return np.asarray(state, dtype=np.float32).reshape(-1)


def _make_record(
    env: gym.Env,
    model: VisualCausalWorldModel,
    normalizer: Normalization,
    root_id: int,
    state: np.ndarray,
    goal: np.ndarray,
    device: torch.device,
    action_seed: int,
    episode_id_base: int,
) -> dict[str, Any]:
    source_state = np.asarray(state, dtype=np.float32).copy()
    goal_state = np.asarray(goal, dtype=np.float32).copy()
    env.reset(seed=0, options={"state": source_state.copy(), "goal_state": goal_state.copy()})
    source_image = env.render().copy()
    # The split manifest stores root states, not predecessor frames.  Feeding
    # the current frame as the predecessor is a declared, identical
    # reconstruction for every candidate, never a hidden simulator feature.
    # All checkpoints in this pool use the declared 64x64 image interface.
    image = tensor_images(resize_image(source_image, 64)[None], str(device))
    previous_image = image.clone()
    mean, std = normalizer.tensors(str(device))

    raw_actions: list[list[float]] = []
    normalized_actions: list[list[float]] = []
    actions: list[np.ndarray] = []
    for horizon in range(1, max(HORIZONS) + 1):
        action = _action(root_id, horizon, action_seed)
        actions.append(action)
        raw_actions.append(action.tolist())
        normalized_actions.append(action.tolist())

    with torch.inference_mode():
        source_latent = model.encode_state(image, previous_image)[0]
        current = source_latent.unsqueeze(0)
        temporal = model.encode_dynamics_context(image, previous_image)
        previous_action = torch.zeros((1, 2), dtype=torch.float32, device=device)

        target_states: list[np.ndarray] = []
        env.reset(seed=0, options={"state": source_state.copy(), "goal_state": goal_state.copy()})
        for action in actions:
            observation, _, terminated, truncated, _ = env.step(action)
            target_states.append(np.asarray(observation["state"], dtype=np.float32).copy())
            if terminated or truncated:
                # Keep the horizon interface defined even when the simulator
                # terminates early; the fixed protocol still records the
                # actual terminal state at subsequent horizons.
                pass

        predictions: list[torch.Tensor] = []
        for action in actions:
            action_tensor = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            predicted = model.predict_next(
                current,
                action_tensor,
                temporal_context=temporal,
                previous_action=previous_action,
            )
            predictions.append(predicted[0].detach().clone())
            current = predicted
            previous_action = action_tensor
            temporal = None

        source_norm = source_latent.detach().cpu().numpy()
        latent_by_horizon: dict[str, dict[str, Any]] = {}
        for horizon, (prediction, target_state) in enumerate(zip(predictions, target_states), start=1):
            if horizon not in HORIZONS:
                continue
            target_pose = torch.as_tensor(observable_pose_vector(target_state), dtype=torch.float32, device=device)
            target_norm = (target_pose - mean) / std
            latent_by_horizon[str(horizon)] = {
                "predicted": prediction.cpu().numpy().tolist(),
                "target": target_norm.cpu().numpy().tolist(),
                "target_state_raw": _raw_state(target_state).tolist(),
            }

    return {
        "sample_id": f"{episode_id_base + root_id}:0",
        "episode_id": int(episode_id_base + root_id),
        "source_frame": 0,
        "source_latent": source_norm.tolist(),
        "source_state_raw": _raw_state(source_state).tolist(),
        "source_proprio_raw": source_state[:4].tolist(),
        "action_blocks_raw": raw_actions,
        "action_blocks_normalized": normalized_actions,
        "latent_by_horizon": latent_by_horizon,
    }


def main() -> None:
    args = parse_args()
    if args.roots < 8 or args.root_offset < 0:
        raise ValueError("roots must be >= 8 and root-offset must be nonnegative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    checkpoint = args.checkpoint.resolve()
    manifest_path = args.split_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    manifest_split = "train" if args.split == "fit" else "validation"
    roots = manifest[manifest_split]["roots"]
    selected = roots[args.root_offset : args.root_offset + args.roots]
    if len(selected) < args.roots:
        raise ValueError(f"requested {args.roots} roots, only {len(selected)} available")

    model, normalizer, saved = _load_model(checkpoint, device)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", max_episode_steps=32)
    try:
        rows = [
            _make_record(
                env,
                model,
                normalizer,
                int(root["root_id"]),
                np.asarray(root["state"], dtype=np.float32),
                np.asarray(root["goal"], dtype=np.float32),
                device,
                args.seed,
                1_000_000 if args.split == "outcome" else 0,
            )
            for root in selected
        ]
    finally:
        env.close()

    payload = make_payload(
        architecture=architecture_name(saved, checkpoint),
        checkpoint=str(checkpoint),
        environment="PushT",
        seed=_seed(checkpoint, saved),
        training_stage=training_stage(checkpoint, saved, args.training_stage),
        latent_dimension=6,
        action_distribution=ACTION_DISTRIBUTION,
        split={
            "role": args.split,
            "source_manifest": str(manifest_path),
            "manifest_split": manifest_split,
            "root_offset": int(args.root_offset),
            "root_count": len(rows),
            "episode_ids": [int(row["episode_id"]) for row in rows],
        },
        horizons=HORIZONS,
        enriched_records=rows,
        checkpoint_sha256=sha256_file(checkpoint),
        dataset_sha256=sha256_file(manifest_path),
        study_scope="trained visual PushT adapter temporal latent diagnostic",
        claim_boundary=(
            "Newly trained intervention-supervised visual-adapter evidence only; "
            "this export does not establish online planning, causal action effects, "
            "or a universal world model."
        ),
        runtime="trained_visual_adapter_runtime.v1",
        protocol={
            "model_interface": "six-dimensional normalized observable-pose chart",
            "runtime_model_input": "rendered source image and reconstructed same-source predecessor image; action only",
            "physical_state_role": "offline simulator target and reset specification only",
            "action_protocol": ACTION_DISTRIBUTION,
            "action_repeat": 1,
            "target_chart": "observable_pose_vector(agent_xy, block_xy, sin_angle, cos_angle)",
            "previous_frame_reconstruction": "same source frame because manifest stores root state but not predecessor image",
            "fit_outcome_policy": "export split roles separately; downstream record builder enforces disjoint episode ids",
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "architecture": payload["architecture"], "seed": payload["seed"], "split": args.split, "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
