"""Build a train-only GDD projection package on the deployment action bank.

This is an execution-supervised support-matched control.  Each source context
is expanded to the exact 32-action visual bank used by the live planner, and
each action is encoded as the same repeated 20-step pulse released online.
The frozen adapter is not changed; the projection is fit only on train-root
simulator costs and is never presented as D-JEPA teacher validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import (
    _load_adapter,
    _predict,
)
from experiments.empirical_spectral_audit.pointmaze_action_charts import action_blocks_for_chart
from experiments.run_live_pointmaze_planning import (
    RGBMazeMap,
    action_bank,
    fit_visual_calibration,
    load_episode,
    DatasetMatchedRenderer,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _simulate(source_state: np.ndarray, action: np.ndarray, target: np.ndarray, *, steps: int = 5) -> float:
    import mujoco

    from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import _legacy_pointmaze_xml

    model = mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())
    data = mujoco.MjData(model)
    state = np.asarray(source_state, dtype=np.float64).reshape(-1)
    data.qpos[:] = state[:2]
    data.qvel[:] = state[2:4]
    mujoco.mj_forward(model, data)
    for _ in range(int(steps)):
        data.ctrl[:] = np.asarray(action, dtype=np.float64)
        mujoco.mj_step(model, data)
    return float(np.linalg.norm(np.asarray(data.qpos[:2]) - np.asarray(target, dtype=np.float64)))


def _fit_projection(features: np.ndarray, rows: list[dict[str, Any]], train_indices: np.ndarray, rank: int, epochs: int) -> np.ndarray:
    centered = features[train_indices]
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    projection = vectors[:rank].copy()
    groups: dict[tuple[str, str], list[int]] = {}
    for index in train_indices.tolist():
        groups.setdefault((str(rows[index]["instance_id"]), str(rows[index]["teacher_set_id"])), []).append(index)
    pairs: list[tuple[int, int]] = []
    for group in groups.values():
        for left_pos, left in enumerate(group):
            for right in group[left_pos + 1:]:
                delta = float(rows[left]["teacher_score"]) - float(rows[right]["teacher_score"])
                if abs(delta) > 1.0e-3:
                    pairs.append((left, right) if delta > 0 else (right, left))
    if not pairs:
        raise ValueError("support-matched train split has no preference pairs")
    for _ in range(int(epochs)):
        gradient = np.zeros_like(projection)
        for preferred, rejected in pairs:
            xp = projection @ features[preferred]
            xr = projection @ features[rejected]
            violation = 0.1 + float(xp @ xp - xr @ xr) / rank
            weight = 1.0 / (1.0 + np.exp(-np.clip(violation, -60.0, 60.0)))
            gradient += weight * (
                2.0 * np.outer(xp, features[preferred]) / rank
                - 2.0 * np.outer(xr, features[rejected]) / rank
            )
        projection -= 0.05 * gradient / max(len(pairs), 1)
        projection /= max(float(np.mean(np.linalg.svd(projection, compute_uv=False))), 1.0e-8)
    return projection


def build(
    input_path: Path,
    dataset_root: Path,
    adapter_checkpoint: Path,
    output_dir: Path,
    *,
    calibration_episodes: list[int],
    device: str = "cuda",
    rank: int = 4,
    epochs: int = 200,
) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    source_rows = payload.get("records")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("input must contain non-empty records")
    adapter, adapter_payload = _load_adapter(adapter_checkpoint, device)
    pca_basis = np.asarray(adapter_payload["pca_basis"], dtype=np.float64)
    action_chart = str(adapter_payload.get("adapter_training", {}).get("action_feature_chart", "raw"))
    history_windows = int(adapter_payload.get("adapter_training", {}).get("history_windows", 1))
    calibration = fit_visual_calibration(dataset_root, calibration_episodes, 768)
    images, _ = load_episode(dataset_root, calibration_episodes[0])
    renderer = DatasetMatchedRenderer.from_episode(images, calibration)
    maze_map = RGBMazeMap.from_renderer(renderer)

    # Preserve one source row per root/context.  The action bank is expanded
    # deterministically to the exact 32 actions used by the live planner.
    contexts: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    for row in source_rows:
        key = (str(row["instance_id"]), str(row["teacher_set_id"]))
        contexts.setdefault(key, dict(row))
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    features: list[np.ndarray] = []
    keys: list[str] = []
    for context_index, base in enumerate(contexts.values()):
        source_state = np.asarray(base["source_state_raw"], dtype=np.float64)
        current = source_state[:2]
        global_goal = np.asarray(base.get("goal_raw", [1.2, 1.2]), dtype=np.float64)[:2]
        route = maze_map.path_waypoints(current, global_goal)
        target = np.asarray(route[0] if route else global_goal, dtype=np.float64)
        actions = action_bank(current, target, 8, [0.25, 0.5, 0.75, 1.0])
        context = np.asarray(base.get("source_history_model", base["source_latent_model"]), dtype=np.float64).reshape(1, -1)
        if history_windows == 1 and context.shape[1] != pca_basis.shape[0]:
            context = np.asarray(base["source_latent_model"], dtype=np.float64).reshape(1, -1)
        if context.shape[1] != history_windows * pca_basis.shape[0]:
            raise ValueError("source history width does not match adapter contract")
        for action_index, action in enumerate(actions):
            sequence = np.tile(action[None, :], (20, 1))
            repeated = sequence.reshape(4, 5, 2).reshape(1, 4, 10)
            row = dict(base)
            row.update({
                "candidate_id": f"support_action_{action_index:02d}",
                "source_candidate_id": int(action_index),
                "action_sequence": sequence.tolist(),
                "goal_raw": target.tolist(),
                "waypoint_target_raw": target.tolist(),
                "waypoint_route_waypoints": [np.asarray(point).tolist() for point in route],
                "candidate_generator": "visual_fixed_bank_repeated_20_step_v1",
                "action_support": "visual_dynamics_fixed_bank",
                "teacher_model_id": "execution_support_matched_teacher_v1",
                "teacher_score_source": "train_root_execution_cost",
                "teacher_runtime_forbidden": True,
                "real_task_cost": _simulate(source_state, action, target),
                "split": str(base.get("split", "train")),
            })
            action_features = action_blocks_for_chart([row], repeated, action_chart)
            model_input = np.concatenate([context, action_features.reshape(1, -1)], axis=1)
            prediction = _predict(adapter, model_input, int(pca_basis.shape[0]), device, adapter_payload["adapter_training"])
            horizon = max(prediction)
            latent = np.asarray(prediction[horizon][0], dtype=np.float64)
            row["predicted_horizon"] = int(horizon)
            row["teacher_score"] = float(-row["real_task_cost"])
            key = f"{row['instance_id']}::{row['teacher_set_id']}::{row['candidate_id']}"
            arrays[f"predicted::{key}"] = latent.astype(np.float32)
            row["predicted_latent_ref"] = f"arrays.npz::predicted::{key}"
            row["predicted_terminal_latent_ref"] = row["predicted_latent_ref"]
            row["predicted_latent_source_dataset"] = str((output_dir / "candidate_dataset.json").resolve())
            rows.append(row)
            features.append(np.concatenate([latent, target]))
            keys.append(key)

    feature_array = np.stack(features)
    train_indices = np.asarray([i for i, row in enumerate(rows) if row["split"] == "train"], dtype=np.int64)
    if len(train_indices) < rank:
        raise ValueError("not enough train rows for projection rank")
    projection = _fit_projection(feature_array, rows, train_indices, rank, epochs)
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    out_dataset = output_dir / "candidate_dataset.json"
    metadata = dict(payload.get("metadata", {}))
    metadata.update({
        "schema": "gdd_wm_pointmaze_support_matched_projection.v0.1",
        "candidate_generator": "visual_fixed_bank_repeated_20_step_v1",
        "action_support": "visual_dynamics_fixed_bank",
        "teacher_model_id": "execution_support_matched_teacher_v1",
        "teacher_type": "execution_supervised_support_matched",
        "teacher_runtime_forbidden": True,
        "dynamics_frozen": True,
        "fit_split": "train_roots_only",
        "adapter_checkpoint": str(adapter_checkpoint.resolve()),
        "source_dataset_sha256": _sha256(input_path),
        "calibration_episodes": calibration_episodes,
        "claim_boundary": "support-matched execution teacher control; not D-JEPA validation",
    })
    out_dataset.write_text(json.dumps({"metadata": metadata, "records": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    goal_features = np.zeros_like(feature_array)
    table = output_dir / "student_features.npz"
    np.savez_compressed(table, keys=np.asarray(keys), features=feature_array, goal_features=goal_features,
                        metadata_json=np.asarray(json.dumps({
                            "schema": "gdd_wm_student_feature_table.v0.1",
                            "dataset_sha256": _sha256(out_dataset),
                            "key_schema": "instance_id::teacher_set_id::candidate_id",
                            "candidate_set_context_input_forbidden": True,
                            "teacher_rank_input_forbidden": True,
                            "held_out_outcome_input_forbidden": True,
                            "dynamics_frozen": True,
                            "feature_source": "frozen_adapter_prediction_plus_waypoint",
                            "action_support": "visual_dynamics_fixed_bank",
                        }, sort_keys=True)))
    projection_path = output_dir / "projection.json"
    projection_path.write_text(json.dumps({
        "schema": "gdd_wm_support_matched_projection.v0.1",
        "projection": projection.tolist(),
        "protocol": {"fit_split": "train", "action_support": "visual_dynamics_fixed_bank", "teacher_runtime_forbidden": True},
        "summary": {"record_count": len(rows), "train_row_count": int(len(train_indices)), "root_count": len(contexts), "projection_rank": int(rank)},
        "claim_boundary": metadata["claim_boundary"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {"schema": metadata["schema"], "candidate_dataset": str(out_dataset.resolve()), "arrays": str(arrays_path.resolve()),
                "feature_table": str(table.resolve()), "projection": str(projection_path.resolve()), "record_count": len(rows),
                "root_count": len(contexts), "claim_boundary": metadata["claim_boundary"]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--adapter-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-episodes", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(build(args.input, args.dataset_root, args.adapter_checkpoint, args.output,
                           calibration_episodes=args.calibration_episodes, device=args.device,
                           rank=args.rank, epochs=args.epochs), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
