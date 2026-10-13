"""Decompose contact-transition failure in the PushT physical adapter.

The report separates contact classification from contact-conditioned motion
regression.  It is deliberately simulator-state only: it diagnoses why the
adapter fails on physical counterfactuals and does not make a claim about the
frozen LeWM latent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel  # noqa: F401
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pusht_contact_aware_closure import (
    block_motion,
    contact_balanced_data,
    fit_model,
    pose_vector,
    sample_case,
    transition_features,
)

gym.logger.min_level = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train", type=int, default=2048)
    parser.add_argument("--held-out", type=int, default=128)
    parser.add_argument("--trees", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7317)
    parser.add_argument("--include-agent-velocity", action="store_true")
    return parser.parse_args()


def finite_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()) if array.size else float("nan"),
        "median": float(np.median(array)) if array.size else float("nan"),
        "p90": float(np.quantile(array, 0.90)) if array.size else float("nan"),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")
    try:
        train = contact_balanced_data(
            env, args.train, args.seed, rng, args.include_agent_velocity
        )
        model = fit_model(
            train, args.trees, args.seed, 0.85, args.include_agent_velocity
        )
        rows = []
        for index in range(args.held_out):
            case_seed = args.seed + 20000 + index
            case = sample_case(env, case_seed)
            direction = rng.normal(size=2)
            direction /= max(np.linalg.norm(direction), 1e-8)
            action = (0.55 * direction).astype(np.float32)
            reset_observation, _ = env.reset(
                seed=case_seed,
                options={"state": case["state"], "goal_state": case["goal"]},
            )
            initial_state = np.asarray(reset_observation["state"], dtype=np.float64).copy()
            current = pose_vector(initial_state, args.include_agent_velocity)
            features = transition_features(current, action)
            probability = float(model.contact.predict_proba(features[None])[0, 1])
            raw_delta = np.asarray(model.dynamics.predict(features[None])[0], dtype=np.float64)
            observation, _, _, _, info = env.step(action)
            true = pose_vector(observation["state"], args.include_agent_velocity)
            true_contact = block_motion(initial_state, observation["state"])
            raw_prediction = current + raw_delta
            gated_delta = raw_delta.copy()
            if probability < 0.85:
                gated_delta[2:6] = 0.0
            gated_prediction = current + gated_delta
            rows.append(
                {
                    "true_contact": true_contact,
                    "contact_probability": probability,
                    "initial_agent_block_distance": float(
                        np.linalg.norm(current[2:4] - current[:2])
                    ),
                    "action_relative_dot": float(
                        np.dot(action, current[2:4] - current[:2])
                    ),
                    "action_relative_cross": float(
                        action[0] * (current[3] - current[1])
                        - action[1] * (current[2] - current[0])
                    ),
                    "raw_block_error": float(np.linalg.norm(raw_prediction[2:4] - true[2:4])),
                    "gated_block_error": float(np.linalg.norm(gated_prediction[2:4] - true[2:4])),
                    "constant_block_error": float(np.linalg.norm(current[2:4] - true[2:4])),
                    "raw_block_delta_norm": float(np.linalg.norm(raw_delta[2:4])),
                    "true_block_delta_norm": float(np.linalg.norm(true[2:4] - current[2:4])),
                    "raw_block_prediction": raw_prediction[2:4].tolist(),
                    "current_block": current[2:4].tolist(),
                    "true_block": true[2:4].tolist(),
                }
            )
    finally:
        env.close()

    thresholds = []
    labels = np.asarray([row["true_contact"] for row in rows], dtype=bool)
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
        predicted = np.asarray(
            [row["contact_probability"] >= threshold for row in rows], dtype=bool
        )
        tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[False, True]).ravel()
        selected_errors = []
        for row in rows:
            prediction = (
                np.asarray(row["raw_block_prediction"], dtype=np.float64)
                if row["contact_probability"] >= threshold
                else np.asarray(row["current_block"], dtype=np.float64)
            )
            truth = np.asarray(row["true_block"], dtype=np.float64)
            selected_errors.append(float(np.linalg.norm(prediction - truth)))
        thresholds.append(
            {
                "threshold": threshold,
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                "precision": float(tp / max(tp + fp, 1)),
                "recall": float(tp / max(tp + fn, 1)),
                "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
                "selected_motion_proxy_mean": float(np.mean(selected_errors)),
            }
        )

    by_contact = {}
    for name, predicate in (("contact", True), ("no_contact", False)):
        subset = [row for row in rows if row["true_contact"] is predicate]
        by_contact[name] = {
            "n": len(subset),
            "contact_probability": finite_summary([row["contact_probability"] for row in subset]),
            "raw_block_error": finite_summary([row["raw_block_error"] for row in subset]),
            "gated_block_error": finite_summary([row["gated_block_error"] for row in subset]),
            "constant_block_error": finite_summary([row["constant_block_error"] for row in subset]),
            "raw_block_delta_norm": finite_summary([row["raw_block_delta_norm"] for row in subset]),
            "true_block_delta_norm": finite_summary([row["true_block_delta_norm"] for row in subset]),
        }

    report = {
        "protocol": {
            "environment": "swm/PushT-v1",
            "train_transitions": args.train,
            "held_out": args.held_out,
            "seed": args.seed,
            "state_representation": "pose + agent velocity" if args.include_agent_velocity else "pose only",
            "thresholds": [row["threshold"] for row in thresholds],
        },
        "threshold_sweep": thresholds,
        "by_contact": by_contact,
        "rows": rows,
        "claim_boundary": "Simulator-state adapter failure decomposition only; no frozen-latent or universal planning claim.",
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "contact_failure_diagnosis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    lines = [
        "# PushT contact-failure diagnosis",
        "",
        "| threshold | precision | recall | F1 | TP | FP | FN |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in thresholds:
        lines.append(
            "| %.2f | %.3f | %.3f | %.3f | %d | %d | %d |"
            % (item["threshold"], item["precision"], item["recall"], item["f1"], item["tp"], item["fp"], item["fn"])
        )
    lines += ["", "| subset | n | raw error | gated error | constant error | raw delta | true delta |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("contact", "no_contact"):
        item = by_contact[name]
        lines.append(
            "| %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f |"
            % (name, item["n"], item["raw_block_error"]["mean"], item["gated_block_error"]["mean"], item["constant_block_error"]["mean"], item["raw_block_delta_norm"]["mean"], item["true_block_delta_norm"]["mean"])
        )
    lines += ["", report["claim_boundary"], ""]
    (output / "contact_failure_diagnosis.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "threshold_sweep": thresholds, "by_contact": by_contact}, indent=2))


if __name__ == "__main__":
    main()
