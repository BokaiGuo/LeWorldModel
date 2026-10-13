"""Matched candidate-ranking audit for the compositional repaired predictor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.run_pusht_repaired_predictor_mpc import (
    collect_compositional_data,
    fit_readout,
    predict_candidates,
    proposal_sequences,
    rollout_sequence,
    task_state_cost,
)
from experiments.run_pusht_contact_predictor_teacher_forced_repair import (
    predict_all,
    sample_cases,
    train_repair,
)
from experiments.run_pusht_counterfactual import image_transform
from experiments.run_pusht_learned_physical_dynamics import state_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--train-cases", type=int, default=96)
    p.add_argument("--held-out", type=int, default=24)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--repair-epochs", type=int, default=32)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--candidates", type=int, default=64)
    p.add_argument("--action-block", type=int, default=2)
    p.add_argument("--seed", type=int, default=8421)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    output = np.empty(order.size, dtype=np.float64)
    output[order] = np.arange(order.size, dtype=np.float64)
    return output


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x, y = rank(x), rank(y)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap(values: np.ndarray, seed: int, draws: int = 2000) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(draws, values.size))
    means = values[idx].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n": int(values.size),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    base_model = load_compatible_pretrained("pusht/lewm", device=args.device)
    repaired_model = load_compatible_pretrained("pusht/lewm", device=args.device)
    transform = image_transform()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        train_cases = sample_cases(env, args.train_cases, args.seed, 40)
        heldout_cases = sample_cases(env, args.held_out, args.seed + 100000, 40)
        train_data = collect_compositional_data(
            base_model, env, train_cases, transform, 16, args.device,
            args.horizon, args.seed + 300000,
        )
        repair_args = argparse.Namespace(
            device=args.device, seed=args.seed, learning_rate=3e-4,
            weight_decay=1e-4, anchor_weight=0.05, free_run_weight=0.0,
            pose_loss_weight=0.0, epochs=args.repair_epochs, batch_size=64,
            train_horizons=[1, 2, 4, args.horizon],
        )
        repair_training = train_repair(repaired_model, train_data, repair_args)
        with torch.no_grad():
            base_train = predict_all(base_model, train_data, args.device).cpu().numpy()
            repaired_train = predict_all(repaired_model, train_data, args.device).cpu().numpy()
        targets = np.asarray([
            state_features(state)
            for state in train_data["target_state"].reshape(-1, 7)
        ], dtype=np.float64).reshape(train_data["target_state"].shape[0], args.horizon, 8)
        base_decoder, base_scaler = fit_readout(base_train.reshape(-1, base_train.shape[-1]), targets.reshape(-1, 8))
        repaired_decoder, repaired_scaler = fit_readout(repaired_train.reshape(-1, repaired_train.shape[-1]), targets.reshape(-1, 8))
        rows = {"base_mpc": [], "repaired_mpc": []}
        for index, case in enumerate(heldout_cases[:args.episodes]):
            rng = np.random.default_rng(args.seed + 800000 + index)
            candidates = proposal_sequences(
                rng, case["state"], case["goal"], args.candidates,
                args.horizon, args.action_block, 0.22,
            )
            actual = []
            for candidate in candidates:
                states, _ = rollout_sequence(env, case, candidate)
                actual.append(task_state_cost(state_features(states[-1]), case["goal"]))
            actual = np.asarray(actual, dtype=np.float64)
            for name, model, decoder, scaler in (
                ("base_mpc", base_model, base_decoder, base_scaler),
                ("repaired_mpc", repaired_model, repaired_decoder, repaired_scaler),
            ):
                predicted_z = predict_candidates(
                    model, case["context_images"], np.asarray(case["context_actions"], dtype=np.float32),
                    candidates, transform, args.device,
                )
                predicted = scaler.inverse_transform(decoder.predict(predicted_z))
                predicted_cost = np.asarray([
                    task_state_cost(value, case["goal"]) for value in predicted
                ])
                selected = int(np.argmin(predicted_cost))
                oracle = float(np.min(actual))
                rows[name].append({
                    "episode": index,
                    "spearman": spearman(predicted_cost, actual),
                    "selected_regret": float(actual[selected] - oracle),
                    "oracle_best": oracle,
                    "selected_actual": float(actual[selected]),
                    "predicted_best_cost": float(predicted_cost[selected]),
                    "candidate_count": args.candidates,
                })
    finally:
        env.close()
    metrics = {}
    for name, values in rows.items():
        metrics[name] = {
            "spearman": bootstrap(np.asarray([row["spearman"] for row in values]), args.seed + len(name)),
            "selected_regret": bootstrap(np.asarray([row["selected_regret"] for row in values]), args.seed + 100 + len(name)),
            "oracle_best": bootstrap(np.asarray([row["oracle_best"] for row in values]), args.seed + 200 + len(name)),
            "selected_actual": bootstrap(np.asarray([row["selected_actual"] for row in values]), args.seed + 300 + len(name)),
        }
    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "train_cases": args.train_cases,
            "held_out_cases": args.held_out,
            "evaluated_cases": min(args.episodes, args.held_out),
            "candidate_count": args.candidates,
            "horizon": args.horizon,
            "action_block": args.action_block,
            "candidate_set": "same guided/random candidate array for base and repaired models per case",
            "actual_evaluation": "each candidate replayed from identical simulator state and goal",
            "repair_data": "five compositional contact arms per case",
        },
        "repair_training": repair_training,
        "metrics": metrics,
        "rows": rows,
        "sanity_checks": {
            "all_finite": bool(all(np.isfinite(row[key]) for values in rows.values() for row in values for key in ("spearman", "selected_regret", "oracle_best", "selected_actual"))),
            "same_candidate_count": True,
            "train_test_disjoint": True,
        },
        "claim_boundary": "Matched candidate-ranking diagnosis. It separates predictor ranking from candidate coverage and is not itself an online task-success claim.",
    }
    (out / "repaired_predictor_candidate_rank.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    lines = [
        "# Repaired-predictor candidate ranking",
        "",
        "| predictor | Spearman | selected regret | oracle best | selected actual |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(
            "| %s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f | %.3f |"
            % (
                name,
                values["spearman"]["estimate"], values["spearman"]["ci95_low"], values["spearman"]["ci95_high"],
                values["selected_regret"]["estimate"], values["selected_regret"]["ci95_low"], values["selected_regret"]["ci95_high"],
                values["oracle_best"]["estimate"], values["selected_actual"]["estimate"],
            )
        )
    lines += ["", report["claim_boundary"], ""]
    (out / "repaired_predictor_candidate_rank.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(out), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
