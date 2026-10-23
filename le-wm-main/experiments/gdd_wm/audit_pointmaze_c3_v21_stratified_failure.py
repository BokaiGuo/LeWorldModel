"""Read-only failure decomposition for V21's sealed C3 panel.

V21's global fixed-bank certificate released no roots.  This audit does not
retune or reopen V21: it uses its already sealed calibration/outcome labels to
separate the predeclared observable high-motion repair stratum from the
direct-control fallback stratum.  It asks whether global calibration was the
bottleneck, or whether each stratum individually lacks winner separation.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.decision_benchmark import run_pointmaze_c3_calibrated_reliability_v21_cuda as v21
from experiments.decision_benchmark import run_pointmaze_temporal_ood_v6_interaction as interaction
from experiments.decision_benchmark.task_contrast import calibrate_contrast_bound
from experiments.gpu_preflight import require_cuda_recurrent_cell


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "results" / "pointmaze_c3_calibrated_reliability_v21_20261023_cuda"
OUTPUT = PANEL / "v21_stratified_failure_audit.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _labels(path: Path) -> dict[str, dict[str, Any]]:
    value = _read(path)
    return dict(value.get("labels", value))


def _root_metrics(rows: list[dict[str, Any]], costs: np.ndarray, labels: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["instance_id"])].append(index)
    high = interaction._history_gate_mask(rows, v21.POLICY)
    result: list[dict[str, Any]] = []
    for root, indices in sorted(groups.items()):
        if len(indices) < 2 or not np.all(high[indices] == high[indices[0]]):
            raise ValueError("V21 root lacks a candidate-invariant motion route")
        predicted, actual = costs[indices], np.asarray([
            float(labels[interaction._key(rows[index])]["real_task_cost"]) for index in indices
        ])
        winner = int(np.argmin(predicted))
        margin = float(np.min(np.delete(predicted, winner) - predicted[winner]))
        contrasts = np.abs((actual[:, None] - actual[None, :]) - (predicted[:, None] - predicted[None, :]))
        result.append({"root_id": root, "stratum": "high_motion_repair" if high[indices[0]] else "direct_fallback",
                       "root_max_contrast_error": float(np.max(contrasts)), "winner_margin": margin,
                       "ranking_correct": bool(int(np.argmin(predicted)) == int(np.argmin(actual)))})
    return result


def _stratum(calibration: list[dict[str, Any]], outcome: list[dict[str, Any]], name: str) -> dict[str, Any]:
    fit = [row for row in calibration if row["stratum"] == name]
    test = [row for row in outcome if row["stratum"] == name]
    values = np.asarray([row["root_max_contrast_error"] for row in fit], dtype=np.float64)
    # Each root is already reduced to its simultaneous all-pair maximum, so
    # use a two-row synthetic matrix only to reuse the exact conformal rank.
    ids = np.asarray([row["root_id"] for row in fit], dtype=object)
    # A two-candidate synthetic root has exact contrast error equal to the
    # observed root maximum: true costs [0, e], predicted costs [0, 0].
    # This only reuses the split-conformal rank calculation; it never claims
    # that the original 16-way bank has two candidates.
    duplicated = np.column_stack((np.zeros_like(values), values)).reshape(-1)
    groups = np.repeat(ids, 2)
    bound = calibrate_contrast_bound(duplicated, np.zeros_like(duplicated), groups, ids, alpha=0.1)
    q = bound["bound"]
    if q is None:
        release = np.zeros(len(test), dtype=bool)
    else:
        release = np.asarray([row["winner_margin"] > float(q) for row in test], dtype=bool)
    coverage = None if q is None else float(np.mean([row["root_max_contrast_error"] <= float(q) for row in test]))
    return {"calibration_roots": len(fit), "outcome_roots": len(test), "contrast_bound": q,
            "outcome_release_count": int(release.sum()), "outcome_release_rate": float(release.mean()),
            "outcome_empirical_coverage": coverage,
            "outcome_mean_winner_margin": float(np.mean([row["winner_margin"] for row in test])),
            "outcome_max_winner_margin": float(np.max([row["winner_margin"] for row in test])),
            "outcome_ranking_accuracy": float(np.mean([row["ranking_correct"] for row in test]))}


def audit() -> dict[str, Any]:
    context = _read(PANEL / "temporal_ood_v6_context.json")["records"]
    frozen = _read(ROOT / "results" / "pointmaze_mixed_velocity_repairability_v20_20261023_cuda" / "temporal_ood_v6_frozen_selection.json")
    calibration_rows = [dict(row) for row in context if row["split"] == "calibration"]
    outcome_rows = [dict(row) for row in context if row["split"] == "outcome"]
    calibration_labels = _labels(PANEL / "temporal_ood_v6_calibration_outcomes.json")
    outcome_labels = _labels(PANEL / "v21_outcome_labels.json")
    _rec_c, action_c, route_c = v21._costs(calibration_rows, frozen, device="cuda")
    _rec_o, action_o, route_o = v21._costs(outcome_rows, frozen, device="cuda")
    calibration = _root_metrics(calibration_rows, route_c, calibration_labels)
    outcome = _root_metrics(outcome_rows, route_o, outcome_labels)
    result = {"schema": "pointmaze_c3_v21_stratified_failure_audit.v1", "runtime": require_cuda_recurrent_cell("cuda"),
              "predeclared_route": v21.POLICY, "by_stratum": {
                  "high_motion_repair": _stratum(calibration, outcome, "high_motion_repair"),
                  "direct_fallback": _stratum(calibration, outcome, "direct_fallback"),
              }, "claim_boundary": "Post-outcome V21 failure diagnosis only; it cannot alter V21's sealed global gate or establish a new C3 result."}
    return result


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite immutable audit {OUTPUT}")
    result = audit()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["by_stratum"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
