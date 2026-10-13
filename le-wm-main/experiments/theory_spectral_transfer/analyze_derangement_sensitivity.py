"""Audit action-derangement sensitivity across replay seeds.

The intervention seeds are independent protocol perturbations of the same
fixed checkpoint and episode manifest.  We therefore report paired episode
statistics within each seed and summarize the seed-level range; seeds are not
treated as additional independent episodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from action_interventions import derangement_indices
from analyze_action_intervention_statistics import holm_adjust, mse_vectors, paired_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observed", type=Path, action="append", required=True)
    parser.add_argument("--intervention", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--interventions-per-label", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-trials", type=int, default=20000)
    parser.add_argument("--permutation-trials", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if "operator_fit_vectors" not in payload:
        raise ValueError(f"{path} lacks operator_fit_vectors")
    return payload


def protocol_episodes(payload: dict) -> list:
    return payload.get("protocol", {}).get("episodes_used", [])


def validate_intervention_permutation(payload: dict) -> dict:
    protocol = payload.get("protocol", {})
    seed = protocol.get("action_permutation_seed")
    if seed is None:
        raise ValueError("intervention is missing action_permutation_seed")
    episodes = protocol_episodes(payload)
    mode = protocol.get("action_mode")
    if mode == "episode_shuffle":
        count = len(episodes)
        if count < 2:
            raise ValueError("derangement audit needs at least two episodes")
        permutation = derangement_indices(count, int(seed))
        return {
            "mode": mode,
            "seed": int(seed),
            "episode_count": count,
            "permutation_is_bijection": sorted(permutation) == list(range(count)),
            "fixed_points": sum(i == j for i, j in enumerate(permutation)),
        }
    if mode == "within_episode_shuffle":
        permutations = protocol.get("action_block_permutations")
        block_count = int(protocol.get("action_block_count", 0))
        if not permutations or block_count < 2:
            raise ValueError("within-episode intervention lacks block permutations")
        if len(permutations) != len(episodes):
            raise ValueError("block permutation count does not match episodes_used")
        checks = [
            sorted(permutation) == list(range(block_count))
            and all(i != j for i, j in enumerate(permutation))
            for permutation in permutations
        ]
        return {
            "mode": mode,
            "seed": int(seed),
            "episode_count": len(episodes),
            "block_count": block_count,
            "permutation_is_bijection": all(sorted(p) == list(range(block_count)) for p in permutations),
            "fixed_points": sum(sum(i == j for i, j in enumerate(p)) for p in permutations),
            "episodes_with_valid_derangement": sum(checks),
        }
    raise ValueError(f"unsupported intervention mode: {mode}")


def main() -> None:
    args = parse_args()
    if len(args.observed) != len(args.label):
        raise ValueError("Provide one observed path per label")
    expected = len(args.label) * args.interventions_per_label
    if len(args.intervention) != expected:
        raise ValueError(
            f"Expected {expected} intervention paths ({len(args.label)} labels x "
            f"{args.interventions_per_label} seeds), got {len(args.intervention)}"
        )

    first_intervention = load(args.intervention[0])
    action_mode = first_intervention.get("protocol", {}).get("action_mode")
    intervention_name = {
        "episode_shuffle": "cross_episode_action_derangement",
        "within_episode_shuffle": "within_episode_action_block_derangement",
    }.get(action_mode)
    if intervention_name is None:
        raise ValueError(f"unsupported intervention mode: {action_mode}")

    output = {
        "protocol": {
            "intervention": intervention_name,
            "unit": "episode",
            "bootstrap_trials": args.bootstrap_trials,
            "permutation_trials": args.permutation_trials,
            "seed": args.seed,
            "seed_aggregation": "per-seed paired statistics; report seed range, no pooling as independent episodes",
            "multiple_testing": "Holm family-wise-error correction across all model, horizon, and intervention-seed cells in this artifact",
        },
        "models": {},
        "evidence_boundary": (
            "Action-permutation sensitivity in fixed-checkpoint replays under the recorded intervention mode; "
            "not causal task performance, training robustness, or model ranking."
        ),
    }

    for model_index, (label, observed_path) in enumerate(zip(args.label, args.observed)):
        observed = load(observed_path)
        intervention_paths = args.intervention[
            model_index * args.interventions_per_label : (model_index + 1) * args.interventions_per_label
        ]
        model = {
            "sources": {"observed": str(observed_path), "interventions": [str(p) for p in intervention_paths]},
            "target_vectors_identical": True,
            "episode_order_identical": True,
            "derangement_checks": [],
            "horizons": {},
        }
        observed_episodes = protocol_episodes(observed)
        horizons = sorted(observed["operator_fit_vectors"]["target"], key=int)
        for path in intervention_paths:
            intervention = load(path)
            protocol = intervention.get("protocol", {})
            model["episode_order_identical"] = model["episode_order_identical"] and (
                observed_episodes == protocol_episodes(intervention)
            )
            if not model["episode_order_identical"]:
                raise ValueError(f"Episode order differs for {label}: {path}")
            model["derangement_checks"].append(validate_intervention_permutation(intervention))
            for horizon in horizons:
                _, target_observed = mse_vectors(observed, horizon)
                _, target_intervention = mse_vectors(intervention, horizon)
                identical = torch.equal(target_observed, target_intervention)
                model["target_vectors_identical"] = model["target_vectors_identical"] and identical
                if not identical:
                    raise ValueError(f"Target vectors differ for {label}, horizon {horizon}, {path}")

        for horizon in horizons:
            observed_mse, _ = mse_vectors(observed, horizon)
            rows = []
            for path in intervention_paths:
                intervention = load(path)
                intervention_mse, _ = mse_vectors(intervention, horizon)
                stats = paired_stats(
                    intervention_mse - observed_mse,
                    args.bootstrap_trials,
                    args.permutation_trials,
                    args.seed + 1000 * model_index + 10 * int(horizon) + int(intervention["protocol"]["action_permutation_seed"]),
                )
                rows.append({
                    "source": str(path),
                    "action_permutation_seed": int(intervention["protocol"]["action_permutation_seed"]),
                    "observed_mse_mean": float(observed_mse.mean()),
                    "intervention_mse_mean": float(intervention_mse.mean()),
                    "intervention_minus_observed": stats,
                })
            deltas = [row["intervention_minus_observed"]["mean_delta"] for row in rows]
            model["horizons"][horizon] = {
                "per_seed": rows,
                "seed_mean_delta": sum(deltas) / len(deltas),
                "seed_min_delta": min(deltas),
                "seed_max_delta": max(deltas),
                "all_seed_deltas_positive": all(value > 0 for value in deltas),
            }
        output["models"][label] = model

    tests = []
    for model in output["models"].values():
        for horizon in model["horizons"].values():
            tests.extend(row["intervention_minus_observed"] for row in horizon["per_seed"])
    for result, adjusted in zip(tests, holm_adjust([row["sign_flip_p_two_sided"] for row in tests])):
        result["holm_adjusted_p"] = adjusted
        result["holm_reject_0_05"] = adjusted < 0.05

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    markdown = [
        "# Random action-derangement sensitivity",
        "",
        "This artifact reports paired episode-level checkpoint replays under the "
        f"recorded `{output['protocol']['intervention']}` intervention. Seeds are protocol perturbations, "
        "not additional independent episodes.",
        "",
        "| Model | Horizon | Observed MSE | Mean intervention ΔMSE across seeds | Seed range | All seed Δ > 0 |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for label, model in output["models"].items():
        for horizon, values in model["horizons"].items():
            observed_values = [row["observed_mse_mean"] for row in values["per_seed"]]
            markdown.append(
                f"| {label} | {horizon} | {sum(observed_values) / len(observed_values):.6f} | "
                f"{values['seed_mean_delta']:.6f} | "
                f"[{values['seed_min_delta']:.6f}, {values['seed_max_delta']:.6f}] | "
                f"{'yes' if values['all_seed_deltas_positive'] else 'no'} |"
            )
    markdown.extend([
        "",
        "## Protocol checks",
        "",
        "Every intervention is required to preserve target vectors and episode "
        "order, use a bijection, and have zero fixed points. Paired bootstrap "
        "intervals and two-sided sign-flip tests are retained in the JSON artifact.",
        "",
        "## Evidence boundary",
        "",
        output["evidence_boundary"],
    ])
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n")
    compact = {
        label: {
            horizon: {
                "seed_mean_delta": round(values["seed_mean_delta"], 6),
                "seed_range": [round(values["seed_min_delta"], 6), round(values["seed_max_delta"], 6)],
                "all_seed_deltas_positive": values["all_seed_deltas_positive"],
            }
            for horizon, values in model["horizons"].items()
        }
        for label, model in output["models"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
