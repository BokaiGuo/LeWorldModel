"""CUDA audit of whether frozen RGB goal distance orders Reacher qpos error.

P1 uses an actual-rendered successor's frozen encoder distance to a rendered
goal image as its observability coordinate.  This diagnostic holds the model
frozen and executes a predeclared raw-action bank from exact observed roots
to determine whether that coordinate itself orders the evaluator-only qpos
task distance.  It neither selects planning candidates nor changes a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __package__ in (None, ""):
    import sys
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import gymnasium as gym
import numpy as np
import torch

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor, _spearman
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, _load_action_stats, install_missing_field_compat
from experiments.run_live_reacher_sequence_candidate_planning import _qpos_distance
from experiments.spectral_reachability.calibrate_reacher_d2p_qpos_readout import decode_qpos, wrapped_delta
from experiments.spectral_reachability.reacher_d2p_context import collect_executed_context, restore_execution_snapshot


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _packets(seed: int, *, packets: int, action_block: int = 5) -> np.ndarray:
    """Predeclare zero plus bounded physical packets independently of roots."""

    if packets < 4:
        raise ValueError("goal-readout audit requires at least four packets")
    rng = np.random.default_rng(int(seed))
    values = rng.uniform(-1.0, 1.0, size=(packets, action_block, 2)).astype(np.float32)
    values[0] = 0.0
    return values


def _embedding(model: torch.nn.Module, preprocessor: _Preprocessor, image: np.ndarray, device: torch.device) -> torch.Tensor:
    # DM Control's OpenGL-backed render can return a vertically flipped view.
    # That view has a negative first-axis stride, which PyTorch cannot wrap.
    # Materialize a C-contiguous RGB buffer before the preprocessor converts it
    # to a tensor; this preserves pixels while making the renderer boundary
    # explicit and reproducible.
    pixels = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
    prepared = preprocessor._prepare_info({"pixels": pixels[None, None]})
    with torch.no_grad():
        return model.encode({"pixels": prepared["pixels"].to(device)})["emb"][0, 0].detach()


def _rgb_goal_cost(image: np.ndarray, goal: np.ndarray) -> float:
    """A no-learning rendered-RGB control for the encoder goal coordinate."""

    observed = np.ascontiguousarray(np.asarray(image, dtype=np.float32)) / 255.0
    target = np.ascontiguousarray(np.asarray(goal, dtype=np.float32)) / 255.0
    if observed.shape != target.shape or observed.ndim != 3 or observed.shape[-1] != 3:
        raise ValueError("Reacher RGB goal cost requires same-shaped HWC RGB images")
    return float(np.square(observed - target).mean())


def _load_qpos_readout(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"feature_mean", "feature_scale", "coefficient", "task"}
        missing = required.difference(data.files)
        if missing or str(np.asarray(data["task"]).reshape(-1)[0]) != "reacher_absolute_qpos_sincos_ridge_v1":
            raise ValueError("qpos readout does not satisfy the frozen Reacher angle-readout contract")
        artifact = {key: np.asarray(data[key], dtype=np.float64).copy() for key in ("feature_mean", "feature_scale", "coefficient")}
    width = artifact["feature_mean"].size
    if artifact["feature_scale"].shape != (width,) or artifact["coefficient"].shape != (width + 1, 4) or np.any(artifact["feature_scale"] <= 0.0):
        raise ValueError("qpos readout arrays have incompatible shapes")
    return artifact


def _decoded_qpos_goal_cost(observed: np.ndarray, target: np.ndarray) -> float:
    delta = wrapped_delta(np.asarray(observed).reshape(1, 2), np.asarray(target).reshape(1, 2))
    return float(np.linalg.norm(delta[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--adapter-artifact", type=Path, required=True)
    parser.add_argument("--qpos-readout", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=8)
    parser.add_argument("--packets", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026102540)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("goal-readout alignment audit requires host CUDA exactly; CPU fallback is forbidden")
    if args.roots < 4 or args.output.exists():
        raise ValueError("at least four roots and a new output directory are required")
    policy, adapter, output = args.policy.resolve(), args.adapter_artifact.resolve(), args.output.resolve()
    qpos_readout_path = args.qpos_readout.resolve() if args.qpos_readout is not None else None
    qpos_readout = _load_qpos_readout(qpos_readout_path) if qpos_readout_path is not None else None
    install_missing_field_compat(); device = torch.device("cuda"); torch.empty(0, device=device)
    mean, std = _load_action_stats(adapter, "swm/ReacherDMControl-v0")
    model = load_compatible_pretrained(policy, device=device).eval(); model.requires_grad_(False); model.interpolate_pos_encoding = True
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    bank = _packets(args.seed, packets=args.packets)
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    rows = []
    try:
        for offset in range(args.roots):
            seed = int(args.seed + offset)
            context = collect_executed_context(env, seed=seed, history_len=3, action_block=5)
            goal_latent = _embedding(model, preprocessor, context["goal"], device)
            decoded_goal_qpos = decode_qpos(qpos_readout, goal_latent.detach().cpu().numpy()[None])[0] if qpos_readout is not None else None
            qpos_values, encoder_costs, rgb_costs, decoded_costs = [], [], [], []
            for packet in bank:
                root, goal = restore_execution_snapshot(env, seed=seed, snapshot=context["execution_snapshot"])
                if not np.array_equal(root, context["root_frame"]) or not np.array_equal(goal, context["goal"]):
                    raise RuntimeError("goal-readout audit failed exact root restoration")
                for action in packet:
                    env.step(action)
                raw = env.unwrapped
                qpos_values.append(_qpos_distance(np.asarray(raw.env.physics.data.qpos), np.asarray(raw.env.task.target_qpos)))
                observed = np.asarray(env.render(), dtype=np.uint8)
                latent = _embedding(model, preprocessor, observed, device)
                encoder_costs.append(float((latent - goal_latent).square().mean()))
                rgb_costs.append(_rgb_goal_cost(observed, context["goal"]))
                if qpos_readout is not None:
                    decoded_observed = decode_qpos(qpos_readout, latent.detach().cpu().numpy()[None])[0]
                    decoded_costs.append(_decoded_qpos_goal_cost(decoded_observed, decoded_goal_qpos))
            row = {"seed": seed, "spearman_encoder_goal_cost_vs_qpos_distance": _spearman(np.asarray(encoder_costs), np.asarray(qpos_values)), "spearman_rgb_goal_cost_vs_qpos_distance": _spearman(np.asarray(rgb_costs), np.asarray(qpos_values)), "encoder_goal_cost": encoder_costs, "rgb_goal_cost": rgb_costs, "qpos_distance": qpos_values}
            if qpos_readout is not None:
                row.update({"spearman_decoded_qpos_goal_cost_vs_qpos_distance": _spearman(np.asarray(decoded_costs), np.asarray(qpos_values)), "decoded_qpos_goal_cost": decoded_costs})
            rows.append(row)
    finally:
        env.close()
    correlations = np.asarray([row["spearman_encoder_goal_cost_vs_qpos_distance"] for row in rows], dtype=np.float64)
    rgb_correlations = np.asarray([row["spearman_rgb_goal_cost_vs_qpos_distance"] for row in rows], dtype=np.float64)
    aggregate = {"encoder_spearman_by_root": correlations.tolist(), "encoder_mean_spearman": float(correlations.mean()), "encoder_positive_spearman_every_root": bool(np.all(correlations > 0.0)), "rgb_spearman_by_root": rgb_correlations.tolist(), "rgb_mean_spearman": float(rgb_correlations.mean()), "rgb_positive_spearman_every_root": bool(np.all(rgb_correlations > 0.0))}
    if qpos_readout is not None:
        decoded_correlations = np.asarray([row["spearman_decoded_qpos_goal_cost_vs_qpos_distance"] for row in rows], dtype=np.float64)
        aggregate.update({"decoded_qpos_spearman_by_root": decoded_correlations.tolist(), "decoded_qpos_mean_spearman": float(decoded_correlations.mean()), "decoded_qpos_positive_spearman_every_root": bool(np.all(decoded_correlations > 0.0))})
    result = {"schema": "reacher_d2p_goal_readout_alignment_audit.v3", "status": "completed", "scope": "fresh root-paired CUDA audit of frozen-encoder, no-learning rendered-RGB, and optional source-calibrated frozen-latent goal coordinates against evaluator-only qpos distance", "inputs": {"policy": str(policy), "weights_sha256": _sha256(policy / "weights.pt"), "adapter_artifact": str(adapter), "adapter_sha256": _sha256(adapter), "qpos_readout": str(qpos_readout_path) if qpos_readout_path is not None else None, "qpos_readout_sha256": _sha256(qpos_readout_path) if qpos_readout_path is not None else None}, "protocol": {"device": str(device), "roots": int(args.roots), "root_seeds": [row["seed"] for row in rows], "physical_packet_bank": "predeclared raw uniform packets plus raw-zero", "packets_per_root": int(args.packets)}, "aggregate": aggregate, "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
