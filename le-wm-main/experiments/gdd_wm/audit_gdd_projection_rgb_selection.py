"""Evaluate a frozen GDD projection on independent RGB successor predictions.

The projection is fitted before this audit.  This runner joins only RGB
successor *predictions* (not realized outcomes) to the held-out candidate
package and evaluates candidate selection against realized task cost.  It is
an offline RGB-compatible consequence; it is deliberately not called a
closed-loop planning result.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def _sign(value: float, margin: float) -> int:
    if abs(value) <= margin:
        return 0
    return 1 if value > 0 else -1


def audit(dataset_path: Path, projection_path: Path, feature_table_path: Path, rgb_reports: list[Path], margin: float = 1e-3) -> dict[str, Any]:
    dataset = _load(dataset_path)
    rows = dataset.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("dataset has no records")
    projection_report = _load(projection_path)
    projection = np.asarray(projection_report.get("projection"), dtype=np.float64)
    if projection.ndim != 2 or not np.isfinite(projection).all():
        raise ValueError("projection report has no finite matrix")
    with np.load(feature_table_path, allow_pickle=False) as table:
        keys = np.asarray(table["keys"])
        goal_features = np.asarray(table["goal_features"], dtype=np.float64)
    key_lookup = {item.decode() if isinstance(item, bytes) else str(item): index for index, item in enumerate(keys)}

    rgb_by_source: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for report_path in rgb_reports:
        report = _load(report_path)
        gates = report.get("gates", {})
        if not all(gates.get(key) is True for key in ("realized_rgb_latents_available", "pca_projected_successor_comparison", "renderer_consistency")):
            raise ValueError(f"RGB gates are not all true: {report_path}")
        source = str(Path(report["protocol"]["candidate_dataset"]).resolve())
        table: dict[tuple[str, int], dict[str, Any]] = {}
        for item in report.get("rows", []):
            key = (str(item["instance_id"]), int(item["candidate_id"]))
            if key in table:
                raise ValueError(f"duplicate RGB row {key}")
            table[key] = item
        rgb_by_source[source] = table

    scores: list[float] = []
    baseline_scores: list[float] = []
    joined_rows: list[dict[str, Any]] = []
    for row in rows:
        source = str(Path(str(row["predicted_latent_source_dataset"])).resolve())
        table = rgb_by_source.get(source)
        if table is None:
            raise ValueError(f"no RGB report for predicted source {source}")
        key = (str(row["instance_id"]), int(row["source_candidate_id"]))
        rgb = table.get(key)
        if rgb is None:
            raise ValueError(f"missing RGB prediction for {key}")
        latent = np.asarray(rgb["predicted_terminal_projection"], dtype=np.float64).reshape(-1)
        goal = np.asarray(row["goal_raw"], dtype=np.float64).reshape(-1)
        feature = np.concatenate([latent, goal])
        if feature.size != projection.shape[1]:
            raise ValueError(f"feature/projection dimension mismatch: {feature.size} vs {projection.shape[1]}")
        record_key = "::".join((str(row["instance_id"]), str(row["teacher_set_id"]), str(row["candidate_id"])))
        if record_key not in key_lookup:
            raise ValueError(f"feature table missing {record_key}")
        anchor = goal_features[key_lookup[record_key]]
        if anchor.shape != feature.shape:
            raise ValueError("feature-table goal anchor dimension mismatch")
        centered = feature - anchor
        score = float(np.linalg.norm(projection @ centered) ** 2 / projection.shape[0])
        baseline = float(np.linalg.norm(centered) ** 2 / centered.size)
        scores.append(score)
        baseline_scores.append(baseline)
        joined_rows.append(dict(row))

    def evaluate(values: list[float], split: str | None = None) -> dict[str, Any]:
        groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for i, row in enumerate(joined_rows):
            if split is not None and str(row["split"]) != split:
                continue
            groups[(str(row["split"]), str(row["instance_id"]), str(row["candidate_bank_context_id"]))].append(i)
        selected_regrets: list[float] = []
        teacher_regrets: list[float] = []
        pairs = correct = 0
        for group in groups.values():
            if len(group) < 2:
                continue
            costs = np.asarray([float(joined_rows[i]["real_task_cost"]) for i in group])
            teacher = np.asarray([float(joined_rows[i]["teacher_score"]) for i in group])
            predicted = np.asarray([values[i] for i in group])
            best = int(np.argmin(predicted))
            teacher_best = int(np.argmax(teacher))
            selected_regrets.append(float(costs[best] - costs.min()))
            teacher_regrets.append(float(costs[teacher_best] - costs.min()))
            for left in range(len(group)):
                for right in range(left + 1, len(group)):
                    delta = teacher[left] - teacher[right]
                    if abs(delta) <= margin:
                        continue
                    pairs += 1
                    pred_delta = predicted[left] - predicted[right]
                    correct += int((delta > 0 and pred_delta < 0) or (delta < 0 and pred_delta > 0))
        return {
            "groups": len(groups),
            "mean_real_task_regret": None if not selected_regrets else float(np.mean(selected_regrets)),
            "teacher_oracle_regret": None if not teacher_regrets else float(np.mean(teacher_regrets)),
            "pairwise_teacher_agreement": None if pairs == 0 else float(correct / pairs),
            "comparable_pairs": pairs,
        }

    return {
        "schema": "gdd_wm_rgb_projection_selection.v0.1",
        "status": "completed_rgb_candidate_consequence",
        "protocol": {
            "dataset": str(dataset_path.resolve()),
            "projection": str(projection_path.resolve()),
            "feature_table": str(feature_table_path.resolve()),
            "rgb_reports": [str(path.resolve()) for path in rgb_reports],
            "feature_source": "independent_frozen_RGB_successor_predicted_terminal_projection_plus_goal_raw",
            "outcome_used_for_projection_or_feature_selection": False,
            "privileged_state_used_in_features": False,
        },
        "metrics": {
            "all": evaluate(scores),
            "by_split": {split: evaluate(scores, split) for split in sorted({str(row["split"]) for row in joined_rows})},
            "baseline_all": evaluate(baseline_scores),
            "baseline_by_split": {split: evaluate(baseline_scores, split) for split in sorted({str(row["split"]) for row in joined_rows})},
        },
        "claim_boundary": "offline RGB-compatible candidate consequence; not closed-loop planning or universal transfer",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--projection", required=True, type=Path)
    parser.add_argument("--feature-table", required=True, type=Path)
    parser.add_argument("--rgb-report", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.dataset, args.projection, args.feature_table, args.rgb_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
