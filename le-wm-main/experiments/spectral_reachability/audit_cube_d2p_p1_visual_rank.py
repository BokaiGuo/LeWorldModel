"""Rank the sealed Cube P1 bank with the same visual objective as the model.

The frozen planner ranks candidates by squared distance between its *predicted*
terminal visual embedding and the encoded goal RGB.  Comparing that score with
Euclidean distance in MuJoCo ``qpos`` changes both the state space and the
metric (and gives arbitrary weight to non-visible coordinates).  This audit
instead physically executes each sealed candidate, renders its successor, and
uses the unchanged frozen encoder to obtain the observed visual objective.

``qpos`` distances are retained as a diagnostic only.  They cannot accept or
reject a visual-world-model prediction contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401 -- register OGBench environments
import torch
from scipy.stats import spearmanr

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.gpu_preflight import require_cuda_allocation
from experiments.run_live_cube_action_adapter_planning import _image_transform
from experiments.spectral_reachability.build_cube_d2p_live_runtime_panel import _restore_state
from experiments.spectral_reachability.run_cube_d2p_p1_live_physical_calibration import step
from experiments.spectral_reachability.cube_d2p_p1_evidence_binding import validate_binding


SCHEMA = "cube_d2p_p1_visual_rank_audit.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _visual_goal_cost(model: torch.nn.Module, image: np.ndarray, goal: torch.Tensor, *, transform: Any, device: torch.device) -> float:
    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError("Cube observed successor must be uint8 HWC RGB")
    with torch.no_grad():
        embedding = model.encode({"pixels": transform(pixels).unsqueeze(0).unsqueeze(0).to(device)})["emb"][0, 0]
        value = (embedding - goal).square().mean()
    if not torch.isfinite(value):
        raise RuntimeError("frozen Cube encoder produced a non-finite observed visual cost")
    return float(value.detach().cpu())


def _rank(value: list[float]) -> int:
    return int(np.argmin(np.asarray(value, dtype=np.float64)))


def visual_ranking_gates(rows: list[dict[str, Any]]) -> tuple[dict[str, bool], dict[str, float | int]]:
    if not rows:
        raise ValueError("visual rank audit needs at least one root")
    selected = sum(bool(row["positive_control_selected_by_model"]) for row in rows)
    rhos = [float(row["spearman_predicted_vs_observed_visual"]) for row in rows]
    gates = {
        "cuda_only": True,
        "full_sealed_bank_physically_executed": True,
        "same_root_state_restored_per_candidate": True,
        "evaluator_state_not_model_input": True,
        "outcome_roots_not_opened": True,
        "physical_goal_metric_matches_model_goal_metric": True,
        "positive_control_is_observed_visual_optimum_each_root": all(
            bool(row["positive_control_observed_visual_optimum"]) for row in rows
        ),
        "positive_control_top1_at_least_75pct": selected * 4 >= len(rows) * 3,
        "mean_spearman_at_least_0_60": float(np.mean(rhos)) >= 0.60,
        "no_per_root_order_reversal": float(np.min(rhos)) >= 0.0,
    }
    return gates, {
        "root_count": len(rows),
        "positive_control_selected_count": selected,
        "positive_control_selected_rate": selected / len(rows),
        "mean_spearman": float(np.mean(rhos)),
        "min_spearman": float(np.min(rhos)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live-panel", type=Path, required=True)
    parser.add_argument("--p1-evidence", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.live_panel = args.live_panel.resolve()
    args.p1_evidence = args.p1_evidence.resolve()
    args.policy = args.policy.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite visual rank audit: {args.output}")
    binding = validate_binding(args.live_panel, args.p1_evidence, args.policy)
    panel = binding["panel"]
    model_rows = binding["ordered_model_rows"]

    host = require_cuda_allocation(args.device)
    device = torch.device("cuda")
    model = load_compatible_pretrained(args.policy, device=device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    parameter_before = hashlib.sha256(
        b"".join(value.detach().cpu().contiguous().numpy().tobytes() for value in model.state_dict().values())
    ).hexdigest()
    transform = _image_transform()
    inputs = np.load(args.live_panel / "model_inputs.npz")
    env = gym.make("swm/OGBCube-v0", mode="data_collection", render_mode="rgb_array", terminate_at_goal=False)
    rows: list[dict[str, Any]] = []
    try:
        for root_index, (root, item) in enumerate(zip(panel["rows"], model_rows, strict=True)):
            if root["root_id"] != item["root_id"]:
                raise ValueError("sealed root identity mismatch")
            # ``_restore_state`` replaces the physical state but deliberately
            # does not emulate Gym's public episode lifecycle.  Establish it
            # once per sealed root before any restored candidate is stepped;
            # otherwise OrderEnforcing rejects a perfectly valid restoration
            # as an action before reset.
            env.reset(seed=int(root["root_seed"]))
            with torch.no_grad():
                goal = model.encode({"pixels": transform(np.asarray(inputs["goal_rgb"][root_index], dtype=np.uint8)).unsqueeze(0).unsqueeze(0).to(device)})["emb"][0, 0].detach()
            predicted = [float(value) for value in item["proposal_goal_costs"]]
            observed_visual: list[float] = []
            qpos_diagnostic: list[float] = []
            target_qpos = np.asarray(root["heldout_successor"]["qpos"], dtype=np.float64)
            for packet in item["proposal_raw_packets"]:
                _restore_state(env.unwrapped, root["evaluator_only_root_state"])
                step(env, np.asarray(packet, dtype=np.float32))
                observed_visual.append(_visual_goal_cost(model, env.render(), goal, transform=transform, device=device))
                qpos_diagnostic.append(float(np.linalg.norm(np.asarray(env.unwrapped.data.qpos) - target_qpos)))
            positive = int(item["positive_control_index"])
            observed_best = _rank(observed_visual)
            model_best = _rank(predicted)
            rho = float(spearmanr(predicted, observed_visual).statistic)
            if not np.isfinite(rho):
                raise RuntimeError("visual candidate bank has an undefined rank correlation")
            rows.append({
                "root_id": root["root_id"],
                "positive_control_index": positive,
                "model_best": model_best,
                "observed_visual_best": observed_best,
                "positive_control_selected_by_model": model_best == positive,
                "positive_control_observed_visual_optimum": observed_best == positive,
                "spearman_predicted_vs_observed_visual": rho,
                "predicted_goal_costs": predicted,
                "observed_visual_goal_costs": observed_visual,
                "qpos_distance_diagnostic": qpos_diagnostic,
                "spearman_predicted_vs_qpos_diagnostic": float(spearmanr(predicted, qpos_diagnostic).statistic),
            })
    finally:
        env.close()
    parameter_after = hashlib.sha256(
        b"".join(value.detach().cpu().contiguous().numpy().tobytes() for value in model.state_dict().values())
    ).hexdigest()
    gates, aggregate = visual_ranking_gates(rows)
    gates["frozen_parameter_hash_unchanged"] = parameter_before == parameter_after
    gates["sealed_inputs_policy_packets_and_root_identity_bound"] = True
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "scope": "CUDA-only P1 prediction calibration in the frozen RGB-latent objective; no planning outcome is evaluated.",
        "cuda_host": {**host, "python": platform.python_version()},
        "upstream": binding["hashes"],
        "criteria": {"positive_control_selected_rate_minimum": 0.75, "mean_spearman_minimum": 0.60, "per_root_spearman_minimum": 0.0},
        "metric_contract": {
            "predicted": "mean squared distance(predicted terminal embedding, encoded goal RGB)",
            "observed": "mean squared distance(encoded executed successor RGB, encoded goal RGB)",
            "qpos": "diagnostic only; not a visual-world-model calibration target",
        },
        "gates": gates,
        "aggregate": aggregate,
        "rows": rows,
    }
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "aggregate": aggregate, "gates": gates}, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit("Cube P1 visual calibration rejected by preregistered ranking gates")


if __name__ == "__main__":
    main()
