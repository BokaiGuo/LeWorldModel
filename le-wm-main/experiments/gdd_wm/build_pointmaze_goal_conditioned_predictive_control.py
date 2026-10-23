"""Build a clearly labelled execution-supervised goal-conditioned GDD control.

Unlike the outcome-free predictive-residual teacher, this control asks the
task-relevant question: where does the frozen predictor say the point will end
relative to the declared goal?  A latent-to-XY readout is fitted only from
*train-root source states*; candidate score is then negative predicted terminal
distance to goal.  It never reads ``real_task_cost`` when fitting or scoring.

The source-state readout is privileged supervision, so this module deliberately
labels its output ``execution_supervised_gdd_control``.  It is a semantic
diagnostic/control, not an independent D-JEPA teacher or a D2 authorization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.gdd_wm.build_pointmaze_frozen_predictive_teacher_package import (
    _canonical_hash,
    _fit_ridge,
    _load_context,
    _predict,
    _sha256_file,
    _source_latent,
    _terminal_latent,
    _validate_contexts,
)


def build(input_paths: list[Path], output: Path, *, ridge: float = 1.0e-3) -> dict[str, Any]:
    if len(input_paths) < 3 or len(set(input_paths)) != len(input_paths):
        raise ValueError("three distinct repeated candidate contexts are required")
    contexts = [(path.resolve(), *_load_context(path.resolve())) for path in input_paths]
    validated = _validate_contexts(contexts, 3)
    cache: dict[Path, Any] = {}
    rows: list[dict[str, Any]] = []
    source_latents: list[np.ndarray] = []
    terminal_latents: list[np.ndarray] = []
    source_xy: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    manifest: list[dict[str, Any]] = []
    for context_index, (path, metadata, source_rows) in enumerate(contexts):
        context_id = f"context_{context_index:02d}_{_sha256_file(path)[:12]}"
        manifest.append({"context_id": context_id, "candidate_dataset": str(path), "sha256": _sha256_file(path)})
        for source_row in source_rows:
            source = _source_latent(source_row)
            terminal = _terminal_latent(source_row, path, cache)
            state = np.asarray(source_row.get("source_state_raw"), dtype=np.float64)
            goal = np.asarray(source_row.get("goal_raw"), dtype=np.float64)
            if state.shape != (4,) or goal.shape != (2,) or source.shape != terminal.shape:
                raise ValueError("PointMaze control requires 4D source state, 2D goal, and matched latent dimensions")
            row = dict(source_row)
            row.update({
                "candidate_bank_context_id": context_id,
                "teacher_set_id": f"{source_row['instance_id']}__{context_id}",
                "predicted_latent_source_dataset": str(path),
                "action_sequence_sha256": _canonical_hash(source_row["action_sequence"]),
            })
            rows.append(row)
            source_latents.append(source)
            terminal_latents.append(terminal)
            source_xy.append(state[:2])
            goals.append(goal)
    source_matrix = np.stack(source_latents)
    terminal_matrix = np.stack(terminal_latents)
    xy_matrix = np.stack(source_xy)
    goal_matrix = np.stack(goals)
    train = np.asarray([i for i, row in enumerate(rows) if row["split"] == "train"], dtype=np.int64)
    if len(train) < source_matrix.shape[1] + 1:
        raise ValueError("insufficient train-root source labels for latent-to-XY control readout")
    beta, mean, scale = _fit_ridge(source_matrix[train], xy_matrix[train], ridge)
    predicted_xy = _predict(terminal_matrix, beta, mean, scale)
    scores = -np.linalg.norm(predicted_xy - goal_matrix, axis=1)
    for row, score, xy in zip(rows, scores, predicted_xy):
        row.update({
            "teacher_model_id": "pointmaze_goal_conditioned_predictive_xy_control_v1",
            "teacher_score_source": "train_only_source_state_readout_of_frozen_predicted_terminal_latent",
            "teacher_score": float(score),
            "teacher_predicted_terminal_xy": np.asarray(xy, dtype=np.float64).tolist(),
            "teacher_fit_split": "train",
        })
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "latent_to_xy_readout.npz", beta=beta, mean=mean, scale=scale)
    metadata = {
        "schema": "gdd_wm_pointmaze_goal_conditioned_predictive_control.v0.1",
        "teacher_model_id": "pointmaze_goal_conditioned_predictive_xy_control_v1",
        "teacher_fit_split": "train",
        "teacher_fit_root_ids": sorted({str(row["instance_id"]) for row in rows if row["split"] == "train"}),
        "teacher_training_uses_execution_outcomes": False,
        "teacher_input_fields": ["frozen_predicted_terminal_latent", "goal_raw"],
        "teacher_training_target_fields": ["train_root_source_state_xy"],
        "control_class": "execution_supervised_gdd_control",
        "context_manifest": manifest,
        "routing_signature": {
            "environment": "PointMaze", "world_model_sha256": validated["world_model_sha256"],
            "horizon": int(rows[0]["predicted_horizon"]), "candidate_family": validated["candidate_family"],
            "action_frame": "world_fixed", "task_cost_version": "external_audit_only_v1",
        },
        "claim_boundary": "Execution-supervised semantic control only; not an independent D-JEPA teacher, D2 authorization, or planning result.",
    }
    dataset = output / "candidate_dataset.json"
    dataset.write_text(json.dumps({"metadata": metadata, "records": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "schema": metadata["schema"], "status": "built_execution_supervised_control",
        "dataset": str(dataset), "record_count": len(rows), "train_root_count": len(metadata["teacher_fit_root_ids"]),
        "train_source_xy_rmse": float(np.sqrt(np.mean((_predict(source_matrix[train], beta, mean, scale) - xy_matrix[train]) ** 2))),
        "claim_boundary": metadata["claim_boundary"],
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.output, ridge=args.ridge), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
