"""Paired validation of covariance--predictor alignment as a model-selection signal.

This benchmark is deliberately narrower than a world-model claim.  It tests four
questions on a known linear process:

1. Can an alignment score predict held-out multi-step transfer?
2. Does alignment-aware selection beat spectrum-only selection when every
   candidate has exactly the same predictor eigenvalues?
3. Can two candidates have matched one-step prediction error but different
   downstream transfer?
4. Is the result stable over independent data draws?

The target process is diagonal in the covariance basis. Candidate predictors
are orthogonal conjugates of the same diagonal spectrum, so their spectra are
identical by construction.  The training split is used only to estimate the
covariance-relative alignment score.  One-step matching is a predeclared pair;
the transfer audit is evaluated on a fresh test split.

This is mechanism and model-selection evidence, not evidence for nonlinear
LeWM, physical causality, or online planning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    dimension: int = 6
    train_samples: int = 512
    validation_samples: int = 1024
    test_samples: int = 4096
    repeats: int = 128
    horizon: int = 8
    # Primary selection family: one fixed eigenspace plane and the same
    # eigenvalue pairing.  This is the predeclared regime in which the scalar
    # commutator is expected to be informative.  A cross-plane stress family
    # is intentionally kept separate below because a scalar commutator is not
    # a universal selector when eigengaps and eigenvalue pairings vary.
    angle_min: int = 5
    angle_max: int = 45
    angle_step: int = 5
    seed: int = 20261008


def rotation(dimension: int, i: int, j: int, degrees: float) -> np.ndarray:
    matrix = np.eye(dimension)
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    matrix[i, i] = matrix[j, j] = c
    matrix[i, j] = -s
    matrix[j, i] = s
    return matrix


def covariance_and_spectrum(config: Config) -> tuple[np.ndarray, np.ndarray]:
    covariance = np.diag(np.array([9.0, 4.0, 2.0, 1.0, 0.5, 0.25])[: config.dimension])
    spectrum = np.diag(np.array([0.90, 0.75, 0.60, 0.45, 0.30, 0.15])[: config.dimension])
    return covariance, spectrum


def candidates(config: Config, spectrum: np.ndarray) -> list[dict[str, object]]:
    planes = [(0, 2)]
    result: list[dict[str, object]] = []
    for i, j in planes:
        for degrees in range(config.angle_min, config.angle_max + 1, config.angle_step):
            result.append(
                {
                    "plane": [i, j],
                    "degrees": degrees,
                    "operator": rotation(config.dimension, i, j, degrees) @ spectrum @ rotation(config.dimension, i, j, degrees).T,
                }
            )
    return result


def sample_states(rng: np.random.Generator, covariance: np.ndarray, count: int) -> np.ndarray:
    return rng.multivariate_normal(np.zeros(covariance.shape[0]), covariance, size=count)


def alignment_score(covariance: np.ndarray, operator: np.ndarray) -> float:
    commutator = covariance @ operator - operator @ covariance
    denominator = np.linalg.norm(covariance, "fro") * np.linalg.norm(operator, "fro")
    return float(np.linalg.norm(commutator, "fro") / max(denominator, 1e-12))


def mse(states: np.ndarray, predicted: np.ndarray, target: np.ndarray) -> float:
    error = predicted - target
    return float(np.mean(np.sum(error * error, axis=1)))


def evaluate_candidate(
    states: np.ndarray,
    target_operator: np.ndarray,
    operator: np.ndarray,
    horizon: int,
) -> dict[str, float]:
    target_one = states @ target_operator.T
    prediction_one = states @ operator.T
    target_h = states @ np.linalg.matrix_power(target_operator, horizon).T
    prediction_h = states @ np.linalg.matrix_power(operator, horizon).T
    return {
        "one_step_mse": mse(states, prediction_one, target_one),
        "transfer_mse": mse(states, prediction_h, target_h),
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty_like(order, dtype=float)
        result[order] = np.arange(len(values), dtype=float)
        return result

    rx, ry = ranks(x), ranks(y)
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 1 else float("nan")


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, draws: int = 4000) -> dict[str, float]:
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def run(config: Config) -> dict[str, object]:
    covariance, target_operator = covariance_and_spectrum(config)
    candidate_rows = candidates(config, target_operator)
    population = []
    for row in candidate_rows:
        operator = row["operator"]
        transfer = evaluate_candidate(
            sample_states(np.random.default_rng(0), covariance, 16384),
            target_operator,
            operator,
            config.horizon,
        )
        population.append(
            {
                "plane": row["plane"],
                "degrees": row["degrees"],
                "alignment": alignment_score(covariance, operator),
                **transfer,
            }
        )

    rng = np.random.default_rng(config.seed)
    selection_regret_alignment: list[float] = []
    selection_regret_spectrum: list[float] = []
    alignment_rank_correlations: list[float] = []
    one_step_differences: list[float] = []
    fixed_pair_transfer_differences: list[float] = []
    fixed_pair_alignment_differences: list[float] = []
    selection_wins = 0
    spectrum_wins = 0

    # Predeclared matched pair: population one-step errors are nearly equal,
    # but covariance-relative alignment and h-step transfer differ sharply.
    pair_a = next(row for row in candidate_rows if row["plane"] == [0, 2] and row["degrees"] == 10)
    pair_b = {
        "plane": [3, 4],
        "degrees": 70,
        "operator": rotation(config.dimension, 3, 4, 70) @ target_operator @ rotation(config.dimension, 3, 4, 70).T,
    }

    for repeat in range(config.repeats):
        train = sample_states(rng, covariance, config.train_samples)
        validation = sample_states(rng, covariance, config.validation_samples)
        test = sample_states(rng, covariance, config.test_samples)

        train_scores = np.array([alignment_score(np.cov(train, rowvar=False), row["operator"]) for row in candidate_rows])
        validation_errors = np.array([
            evaluate_candidate(validation, target_operator, row["operator"], config.horizon)["one_step_mse"]
            for row in candidate_rows
        ])
        test_rows = [evaluate_candidate(test, target_operator, row["operator"], config.horizon) for row in candidate_rows]
        transfer_errors = np.array([row["transfer_mse"] for row in test_rows])

        selected_alignment = int(np.argmin(train_scores))
        selected_spectrum = int(rng.integers(0, len(candidate_rows)))
        best = float(transfer_errors.min())
        selection_regret_alignment.append(float(transfer_errors[selected_alignment] - best))
        selection_regret_spectrum.append(float(transfer_errors[selected_spectrum] - best))
        selection_wins += int(selected_alignment == int(np.argmin(transfer_errors)))
        spectrum_wins += int(selected_spectrum == int(np.argmin(transfer_errors)))
        alignment_rank_correlations.append(spearman(train_scores, transfer_errors))

        pair_eval_a = evaluate_candidate(test, target_operator, pair_a["operator"], config.horizon)
        pair_eval_b = evaluate_candidate(test, target_operator, pair_b["operator"], config.horizon)
        one_step_differences.append(pair_eval_a["one_step_mse"] - pair_eval_b["one_step_mse"])
        fixed_pair_transfer_differences.append(pair_eval_a["transfer_mse"] - pair_eval_b["transfer_mse"])
        fixed_pair_alignment_differences.append(
            alignment_score(np.cov(train, rowvar=False), pair_a["operator"])
            - alignment_score(np.cov(train, rowvar=False), pair_b["operator"])
        )

    # A spectrum-only score is exactly tied by construction.  Random tie
    # breaking is the strongest honest baseline for this candidate family.
    summary_rng = np.random.default_rng(config.seed + 1)
    summary = {
        "alignment_selection_regret": bootstrap_mean(np.asarray(selection_regret_alignment), summary_rng),
        "spectrum_only_selection_regret": bootstrap_mean(np.asarray(selection_regret_spectrum), summary_rng),
        "alignment_selection_win_rate": selection_wins / config.repeats,
        "spectrum_only_selection_win_rate": spectrum_wins / config.repeats,
        "alignment_transfer_rank_spearman": bootstrap_mean(np.asarray(alignment_rank_correlations), summary_rng),
        "matched_pair_one_step_mse_A_minus_B": bootstrap_mean(np.asarray(one_step_differences), summary_rng),
        "matched_pair_transfer_mse_A_minus_B": bootstrap_mean(np.asarray(fixed_pair_transfer_differences), summary_rng),
        "matched_pair_alignment_A_minus_B": bootstrap_mean(np.asarray(fixed_pair_alignment_differences), summary_rng),
    }

    return {
        "protocol": {
            "estimand": "alignment-aware model selection for held-out multi-step transfer",
            "target_process": "diagonal covariance-relative linear dynamics",
            "candidate_count": len(candidate_rows),
            "all_candidate_predictor_spectra_identical": True,
            "spectrum_only_baseline": "random tie-break among equal-spectrum candidates",
            "alignment_fit_split": "training covariance only",
            "transfer_evaluation_split": "independent test states",
            "horizon": config.horizon,
            "repeats": config.repeats,
            "claim_boundary": "controlled linear mechanism and model-selection evidence; not nonlinear LeWM, physical causality, or planning evidence",
        },
        "population_candidates": population,
        "matched_pair": {
            "A": {"plane": pair_a["plane"], "degrees": pair_a["degrees"]},
            "B": {"plane": pair_b["plane"], "degrees": pair_b["degrees"]},
            "population_A": next(row for row in population if row["plane"] == [0, 2] and row["degrees"] == 10),
            "population_B": {
                "plane": pair_b["plane"],
                "degrees": pair_b["degrees"],
                "alignment": alignment_score(covariance, pair_b["operator"]),
                **evaluate_candidate(
                    sample_states(np.random.default_rng(1), covariance, 16384),
                    target_operator,
                    pair_b["operator"],
                    config.horizon,
                ),
            },
        },
        "summary": summary,
    }


def markdown(report: dict[str, object]) -> str:
    summary = report["summary"]
    protocol = report["protocol"]
    lines = [
        "# Covariance--predictor diagnostic benchmark",
        "",
        "This is a controlled linear model-selection and held-out transfer audit.",
        "It is not evidence for nonlinear LeWM, physical causality, or online planning.",
        "",
        "## Protocol",
        "",
        f"- Candidates: {protocol['candidate_count']} orthogonal conjugates with identical predictor spectra.",
        f"- Alignment score: training-split normalized commutator.",
        f"- Transfer evaluation: independent test states at horizon {protocol['horizon']}.",
        f"- Repeated trials: {protocol['repeats']}.",
        "- Spectrum-only selection: random tie-break because all candidate spectra are identical.",
        "",
        "## Primary results",
        "",
        "| Quantity | Mean | 95% bootstrap CI |",
        "|---|---:|---:|",
    ]
    for label, key in [
        ("Alignment selection regret", "alignment_selection_regret"),
        ("Spectrum-only selection regret", "spectrum_only_selection_regret"),
        ("Alignment transfer-rank Spearman", "alignment_transfer_rank_spearman"),
        ("Matched pair one-step MSE A-B", "matched_pair_one_step_mse_A_minus_B"),
        ("Matched pair transfer MSE A-B", "matched_pair_transfer_mse_A_minus_B"),
        ("Matched pair alignment A-B", "matched_pair_alignment_A_minus_B"),
    ]:
        value = summary[key]
        lines.append(f"| {label} | {value['mean']:.6g} | [{value['ci_low']:.6g}, {value['ci_high']:.6g}] |")
    lines.extend(
        [
            "",
            f"Alignment selection win rate: {summary['alignment_selection_win_rate']:.4f}.",
            f"Spectrum-only selection win rate: {summary['spectrum_only_selection_win_rate']:.4f}.",
            "",
            "## Interpretation",
            "",
            "The predeclared matched pair has the same predictor spectrum and nearly matched one-step error, but different covariance-relative alignment and held-out multi-step transfer. The model-selection audit then tests whether training-only alignment selects lower-regret candidates than an equal-spectrum spectrum-only baseline; the spectrum-only score is exactly tied and therefore has no defined rank correlation.",
            "",
            "These results support a controlled diagnostic principle only. They do not establish that the same score predicts nonlinear learned-world-model planning or physical task success.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=1024)
    parser.add_argument("--test-samples", type=int, default=4096)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    config = Config(
        repeats=args.repeats,
        train_samples=args.train_samples,
        validation_samples=args.validation_samples,
        test_samples=args.test_samples,
        horizon=args.horizon,
        seed=args.seed,
    )
    report = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".md").write_text(markdown(report))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
