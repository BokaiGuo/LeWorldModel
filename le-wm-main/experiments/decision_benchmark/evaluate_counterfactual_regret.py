"""Evaluate fixed, prediction-only baselines on a shared candidate bank.

The candidate bank is generated once and every method selects from exactly the
same candidates for each root.  Real task cost is read only after selection
for scoring.  This runner never tunes a metric, selects a checkpoint, or
changes the candidate bank using outcome rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.empirical_spectral_audit.enriched_record_schema import load_payload
from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _fit_readouts


def _bootstrap(values: np.ndarray, groups: list[str], seed: int, samples: int = 1000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    unique = np.asarray(sorted(set(groups)))
    if len(unique) < 2:
        mean = float(np.mean(values))
        return {"estimate": mean, "lower": mean, "upper": mean, "groups": int(len(unique))}
    rng = np.random.default_rng(seed)
    by_group = [values[np.asarray(groups) == group] for group in unique]
    draws = []
    for _ in range(samples):
        indices = rng.integers(0, len(unique), len(unique))
        draws.append(float(np.mean(np.concatenate([by_group[index] for index in indices]))))
    return {
        "estimate": float(np.mean(values)),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "groups": int(len(unique)),
    }


def _candidate_hash(records: list[dict[str, Any]]) -> str:
    canonical = [
        (str(row["instance_id"]), int(row["candidate_id"]), row["action_sequence"])
        for row in sorted(records, key=lambda item: (str(item["instance_id"]), int(item["candidate_id"])))
    ]
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode()).hexdigest()[:20]


def evaluate(
    dataset: Path,
    arrays: Path,
    *,
    fit_export: Path | None = None,
    split: str = "outcome",
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    records = [row for row in payload["records"] if row.get("split") == split]
    if not records:
        raise ValueError(f"candidate dataset has no {split!r} records")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(str(row["instance_id"]), []).append(row)
    if any(len(rows) != len(next(iter(grouped.values()))) for rows in grouped.values()):
        raise ValueError("candidate sets are not equal-sized")
    bank_hash = _candidate_hash(records)
    arrays_payload = np.load(arrays, allow_pickle=False)
    readouts = None
    if fit_export is not None:
        fit = load_payload(fit_export)
        readouts = _fit_readouts(fit["enriched_records"])

    method_choices: dict[str, list[int]] = {"random": [], "action_alignment": [], "predicted_task_decoder": []}
    actual_costs: dict[str, list[float]] = {method: [] for method in method_choices}
    actual_success: dict[str, list[float]] = {method: [] for method in method_choices}
    regrets: dict[str, list[float]] = {method: [] for method in method_choices}
    roots: list[str] = []
    details: list[dict[str, Any]] = []
    for instance_index, (instance_id, candidates) in enumerate(sorted(grouped.items())):
        candidates = sorted(candidates, key=lambda row: int(row["candidate_id"]))
        if any(int(row["total_action_steps"]) != int(candidates[0]["total_action_steps"]) for row in candidates):
            raise ValueError(f"unequal action duration in {instance_id}")
        source = np.asarray(candidates[0]["source_state_raw"], dtype=np.float64)[:2]
        goal = np.asarray(candidates[0]["goal_raw"], dtype=np.float64)[:2]
        costs = np.asarray([float(row["real_task_cost"]) for row in candidates], dtype=np.float64)
        success = np.asarray([float(bool(row["real_success"])) for row in candidates], dtype=np.float64)
        oracle_cost = float(np.min(costs))
        action_scores = np.asarray([
            float(np.sum(np.asarray(row["action_sequence"], dtype=np.float64), axis=0) @ (goal - source))
            for row in candidates
        ])
        predicted_scores = np.full(len(candidates), -np.inf, dtype=np.float64)
        if readouts is not None:
            horizon = max(readouts)
            decoder, latent_center, target_center = readouts[horizon]
            decoded = []
            for row in candidates:
                key = str(row["predicted_latent_ref"]).split("::", 1)[-1]
                latent = np.asarray(arrays_payload[key], dtype=np.float64).reshape(1, -1)
                displacement = (latent - latent_center.reshape(1, -1)) @ decoder + target_center.reshape(1, -1)
                decoded.append(float(displacement[0] @ (goal - source)))
            predicted_scores = np.asarray(decoded)
        choices = {
            "random": int(np.random.default_rng(9100 + instance_index).integers(0, len(candidates))),
            "action_alignment": int(np.argmax(action_scores)),
            "predicted_task_decoder": int(np.argmax(predicted_scores)),
        }
        roots.append(instance_id)
        detail = {"instance_id": instance_id, "candidate_count": len(candidates), "oracle_cost": oracle_cost, "choices": {}}
        for method, choice in choices.items():
            method_choices[method].append(choice)
            actual_costs[method].append(float(costs[choice]))
            actual_success[method].append(float(success[choice]))
            regrets[method].append(float(costs[choice] - oracle_cost))
            detail["choices"][method] = {
                "candidate_id": int(candidates[choice]["candidate_id"]),
                "cost": float(costs[choice]),
                "success": bool(success[choice]),
            }
        details.append(detail)

    summaries = {}
    for method in method_choices:
        summaries[method] = {
            "real_task_cost": _bootstrap(np.asarray(actual_costs[method]), roots, 100 + len(method), bootstrap_samples),
            "decision_regret": _bootstrap(np.asarray(regrets[method]), roots, 200 + len(method), bootstrap_samples),
            "success_rate": _bootstrap(np.asarray(actual_success[method]), roots, 300 + len(method), bootstrap_samples),
            "selected_candidate_ids": method_choices[method],
        }
    report = {
        "schema": "pointmaze_counterfactual_regret.v1",
        "dataset": str(dataset.resolve()),
        "arrays": str(arrays.resolve()),
        "split": split,
        "candidate_bank_hash": bank_hash,
        "instance_count": len(grouped),
        "method_summaries": summaries,
        "details": details,
        "protocol": {
            "same_initial_state": True,
            "same_goal": True,
            "same_total_action_duration": True,
            "shared_candidate_bank": True,
            "outcome_used_for_selection": False,
            "selection_rules_frozen": True,
            "predicted_task_decoder": "train-only affine latent-to-displacement readout" if readouts is not None else "not_run_missing_fit_export",
            "runtime_privileged_state": False,
        },
        "claim_boundary": "Paired equal-duration counterfactual decision benchmark; no closed-loop RGB claim and no realized visual-latent consistency claim.",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--fit-export", type=Path)
    parser.add_argument("--split", default="outcome")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(**{key: value for key, value in vars(args).items() if key != "output"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({method: summary["decision_regret"] for method, summary in report["method_summaries"].items()}, indent=2))


if __name__ == "__main__":
    main()
