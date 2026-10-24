"""CUDA-only, evaluator-side latent rollout-fidelity audit for a sealed P0 receipt.

For every recorded D2P and CEM action witness, reconstruct the observed root,
execute its already-fixed raw action packets, and compare each frozen predicted
latent to the encoder latent of the actually rendered successor RGB.  No qpos
or physical outcome is read, and no candidate is selected or changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

import gymnasium as gym
import numpy as np
import torch

from experiments.audit_live_reacher_shared_candidate_ranking import _Preprocessor
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_live_cube_action_adapter_planning import ActionNormalizer
from experiments.run_live_reacher_planning import ReacherGoalImageWrapper, _load_action_stats, install_missing_field_compat
from experiments.spectral_reachability import run_reacher_d2p_p0_goal_progress as runner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rollout_predictions(
    model: torch.nn.Module, pixels: torch.Tensor, history: torch.Tensor, normalized_future: torch.Tensor,
) -> list[torch.Tensor]:
    """Mirror the P0 root-anchored packet recurrence and retain each successor."""

    if normalized_future.ndim != 2 or normalized_future.shape[-1] != history.shape[-1]:
        raise ValueError("normalized future has incompatible shape")
    encoded = runner._latent_context(model, pixels, history)
    history_size = int(pixels.shape[1])
    action_context = torch.cat([history[:, 1:], normalized_future[:1][None]], dim=1)
    emb_context = encoded
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for step in range(normalized_future.shape[0]):
            action_emb = model.action_encoder(action_context[:, -history_size:])
            terminal = model.predict(emb_context[:, -history_size:], action_emb)[:, -1]
            predictions.append(terminal[0].detach())
            if step + 1 < normalized_future.shape[0]:
                emb_context = torch.cat([emb_context, terminal[:, None]], dim=1)
                action_context = torch.cat([action_context, normalized_future[step + 1:step + 2][None]], dim=1)
    return predictions


def _contiguous_rgb(frame: np.ndarray) -> np.ndarray:
    """Materialize renderer views before PyTorch consumes their pixel buffer."""

    image = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
    if image.ndim != 3:
        raise ValueError(f"expected HWC RGB frame, got shape {image.shape}")
    return image


def _mean_squared_terminal_cost(terminal: torch.Tensor, goal_latent: torch.Tensor) -> torch.Tensor:
    """Use the runner's frozen terminal-cost definition for replay binding."""

    if terminal.shape != goal_latent.shape:
        raise ValueError("terminal and goal latent shapes must match")
    return (terminal - goal_latent).square().mean()


def _encode_frame(model: torch.nn.Module, preprocessor: _Preprocessor, frame: np.ndarray, device: torch.device) -> torch.Tensor:
    image = _contiguous_rgb(frame)
    prepared = preprocessor._prepare_info({"pixels": image[None, None]})
    with torch.no_grad():
        return model.encode({"pixels": prepared["pixels"].to(device)})["emb"][0, 0].detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.atol <= 0 or not torch.cuda.is_available():
        raise RuntimeError("rollout-fidelity audit requires host CUDA and positive tolerance; CPU fallback is forbidden")
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    protocol, inputs = report.get("protocol", {}), report.get("inputs", {})
    if report.get("schema") != "d2p_reacher_p0_goal_progress.v1" or not protocol.get("outcome_only"):
        raise ValueError("rollout-fidelity audit requires an outcome-only P0 receipt")
    if inputs.get("runner_sha256") != runner._sha256_file(Path(runner.__file__)):
        raise ValueError("receipt runner differs from current root-anchored adapter")
    install_missing_field_compat()
    device = torch.device("cuda")
    policy, adapter = Path(inputs["checkpoint"]), Path(inputs["adapter_artifact"])
    mean, std = _load_action_stats(adapter, "swm/ReacherDMControl-v0")
    model = load_compatible_pretrained(policy, device="cuda").eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    if runner._parameter_hash(model) != report.get("parameter_hash_before"):
        raise RuntimeError("frozen checkpoint hash differs from sealed P0 receipt")
    preprocessor = _Preprocessor(ActionNormalizer(mean, std))
    future_horizon, action_block = int(protocol["future_horizon"]), int(protocol["action_block"])
    history_len = int(protocol["history_len"])
    rows: list[dict[str, Any]] = []
    env = ReacherGoalImageWrapper(gym.make("swm/ReacherDMControl-v0", render_mode="rgb_array"), target_delta=(0.65, -0.65))
    try:
        for row in report["rows"]:
            seed = int(row["seed"])
            context = runner._root_context(env, seed, history_len, action_block)
            pixels, goal_tensor, history = runner._prepare(
                preprocessor, context["frames"], context["root_frame"], context["goal"],
                np.asarray(context["history_action_blocks"], dtype=np.float32), device,
            )
            goal_latent = runner._goal_latent(model, goal_tensor)
            methods: dict[str, Any] = {}
            for name in ("d2p_goal_progress", "cem"):
                recorded = row["methods"][name]
                normalized = np.asarray(recorded["action_witness_normalized"], dtype=np.float32)
                raw = np.asarray(recorded["action_witness_raw"], dtype=np.float32)
                if normalized.shape != (future_horizon, history.shape[-1]) or raw.shape != (future_horizon * action_block, 2):
                    raise ValueError(f"seed {seed} {name}: witness shape is incompatible")
                predictions = _rollout_predictions(model, pixels, history, torch.as_tensor(normalized, device=device))
                replayed_terminal_cost = float(_mean_squared_terminal_cost(predictions[-1], goal_latent).detach())
                recorded_terminal_cost = float(recorded["predicted_terminal_cost"])
                root, goal = runner.restore_execution_snapshot(env, seed=seed, snapshot=row["execution_snapshot"])
                if not np.array_equal(root, context["root_frame"]) or not np.array_equal(goal, context["goal"]):
                    raise RuntimeError(f"seed {seed} {name}: execution root identity differs")
                errors: list[float] = []
                for step in range(future_horizon):
                    for action in raw[step * action_block:(step + 1) * action_block]:
                        env.step(action)
                    actual = _encode_frame(model, preprocessor, np.asarray(env.render(), dtype=np.uint8), device)
                    errors.append(float(torch.linalg.vector_norm(predictions[step] - actual).detach()))
                methods[name] = {
                    "latent_l2_by_packet": errors,
                    "mean_latent_l2": float(np.mean(errors)),
                    "replayed_terminal_cost": replayed_terminal_cost,
                    "recorded_terminal_cost": recorded_terminal_cost,
                    "terminal_cost_abs_error": abs(replayed_terminal_cost - recorded_terminal_cost),
                }
            rows.append({"seed": seed, "methods": methods})
    finally:
        env.close()
    d2p = np.asarray([row["methods"]["d2p_goal_progress"]["latent_l2_by_packet"] for row in rows], dtype=np.float64)
    cem = np.asarray([row["methods"]["cem"]["latent_l2_by_packet"] for row in rows], dtype=np.float64)
    result = {
        "schema": "d2p_reacher_p0_rollout_fidelity_audit.v1",
        "status": "pass",
        "scope": "sealed-witness CUDA replay; physical qpos is never read and results cannot tune opened P0 choices",
        "source_report": str(report_path),
        "source_report_sha256": _sha256(report_path),
        "runner_sha256": runner._sha256_file(Path(runner.__file__)),
        "device": {"name": torch.cuda.get_device_name(device), "torch": torch.__version__, "cuda": torch.version.cuda},
        "gates": {
            "cuda_only": True,
            "all_latent_errors_finite": bool(np.isfinite(d2p).all() and np.isfinite(cem).all()),
            "terminal_cost_matches_sealed_receipt": bool(all(
                method["terminal_cost_abs_error"] <= args.atol
                for row in rows for method in row["methods"].values()
            )),
        },
        "mean_latent_l2_by_packet": {"d2p_goal_progress": d2p.mean(axis=0).tolist(), "cem": cem.mean(axis=0).tolist()},
        "rows": rows,
        "claim_boundary": "This audit measures prediction fidelity of executed witnesses, not planning efficacy or a correction to P0 selection.",
    }
    if not all(result["gates"].values()):
        result["status"] = "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "gates": result["gates"], "mean_latent_l2_by_packet": result["mean_latent_l2_by_packet"]}, sort_keys=True))


if __name__ == "__main__":
    main()
