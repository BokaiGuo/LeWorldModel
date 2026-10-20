"""Build an execution-supervised, waypoint-conditioned GDD projection package.

The original PointMaze GDD artifact was trained on global-goal candidate
sequences while the RGB planner deploys a local waypoint.  This builder keeps
the frozen visual dynamics and adapter fixed, regenerates the same paired
candidate family around the RGB-inferred first waypoint, and fits the
projection only on train roots using the executed waypoint cost.  It is an
execution-supervised GDD control, not D-JEPA validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    DatasetMatchedRenderer,
    fit_visual_calibration,
    green_center,
    load_episode,
    paired_gdd_action_sequences,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _simulate(source_state: np.ndarray, sequence: np.ndarray, target: np.ndarray, *, label_blocks: int) -> float:
    import mujoco

    from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import (
        _legacy_pointmaze_xml,
    )

    model = mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())
    data = mujoco.MjData(model)
    source_state = np.asarray(source_state, dtype=np.float64).reshape(-1)
    data.qpos[:] = source_state[:2]
    data.qvel[:] = source_state[2:4]
    mujoco.mj_forward(model, data)
    for action_index, action in enumerate(np.asarray(sequence, dtype=np.float64)):
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        if action_index + 1 >= 5 * int(label_blocks):
            break
    return float(np.linalg.norm(np.asarray(data.qpos[:2]) - np.asarray(target, dtype=np.float64)))


def _build_waypoint_map(dataset_root: Path, calibration_episodes: list[int], calibration_frames: int) -> tuple[Any, Any]:
    calibration = fit_visual_calibration(dataset_root, calibration_episodes, calibration_frames)
    images, _ = load_episode(dataset_root, calibration_episodes[0])
    renderer = DatasetMatchedRenderer.from_episode(images, calibration)
    return calibration, RGBMazeMap.from_renderer(renderer)


def build(
    input_path: Path,
    dataset_root: Path,
    adapter_checkpoint: Path,
    output_dir: Path,
    *,
    calibration_episodes: list[int],
    calibration_frames: int = 768,
    device: str = "cuda",
    projection_rank: int = 4,
    projection_epochs: int = 200,
    label_blocks: int = 1,
) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("input must be a JSON object with records")
    rows = [dict(row) for row in payload["records"]]
    if not rows:
        raise ValueError("input candidate dataset is empty")
    if int(label_blocks) < 1 or int(label_blocks) > 4:
        raise ValueError("label_blocks must lie in [1, 4]")

    adapter, adapter_payload = _load_adapter(adapter_checkpoint, device)
    pca_basis = np.asarray(adapter_payload["pca_basis"], dtype=np.float64)
    action_chart = str(adapter_payload.get("adapter_training", {}).get("action_feature_chart", "raw"))
    history_windows = int(adapter_payload.get("adapter_training", {}).get("history_windows", 1))
    _, maze_map = _build_waypoint_map(dataset_root, calibration_episodes, calibration_frames)
    arrays: dict[str, np.ndarray] = {}
    features: list[np.ndarray] = []
    keys: list[str] = []

    for row in rows:
        source_state = np.asarray(row["source_state_raw"], dtype=np.float64)
        source = source_state[:2]
        global_goal = np.asarray(row.get("goal_raw", [1.2, 1.2]), dtype=np.float64)[:2]
        waypoints = maze_map.path_waypoints(source, global_goal)
        target = np.asarray(waypoints[0] if waypoints else global_goal, dtype=np.float64)
        sequences = paired_gdd_action_sequences(source, target)
        # The joined D1 package carries a stable action hash as candidate_id;
        # source_candidate_id is the canonical within-bank index used by the
        # frozen adapter export.
        candidate_index = int(row.get("source_candidate_id", row.get("candidate_id")))
        if not 0 <= candidate_index < len(sequences):
            raise ValueError(f"candidate_id {candidate_index} is outside waypoint bank")
        sequence = sequences[candidate_index]
        repeated = sequence.reshape(4, 5, 2).reshape(1, 4, 10)
        context = np.asarray(row.get("source_history_model", row["source_latent_model"]), dtype=np.float64).reshape(1, -1)
        if history_windows > 1 and context.shape[1] != history_windows * pca_basis.shape[0]:
            raise ValueError("source history width does not match adapter history contract")
        if history_windows == 1 and context.shape[1] != pca_basis.shape[0]:
            context = np.asarray(row["source_latent_model"], dtype=np.float64).reshape(1, -1)
        action_features = action_blocks_for_chart([row], repeated, action_chart)
        model_input = np.concatenate([context, action_features.reshape(1, -1)], axis=1)
        prediction = _predict(
            adapter,
            model_input,
            int(pca_basis.shape[0]),
            device,
            adapter_payload["adapter_training"],
        )
        horizon = max(prediction)
        predicted = np.asarray(prediction[horizon][0], dtype=np.float32)
        key = "::".join((str(row["instance_id"]), str(row["teacher_set_id"]), str(row["candidate_id"])))
        arrays[f"predicted::{key}"] = predicted
        row["action_sequence"] = sequence.tolist()
        row["goal_raw"] = target.tolist()
        row["waypoint_target_raw"] = target.tolist()
        row["waypoint_route_waypoints"] = [np.asarray(point).tolist() for point in waypoints]
        row["predicted_horizon"] = int(horizon)
        row["predicted_terminal_latent_ref"] = f"arrays.npz::predicted::{key}"
        row["predicted_latent_ref"] = f"arrays.npz::predicted::{key}"
        row["predicted_latent_source_dataset"] = str((output_dir / "candidate_dataset.json").resolve())
        row["real_task_cost"] = _simulate(source_state, sequence, target, label_blocks=label_blocks)
        row["teacher_score"] = float(-row["real_task_cost"])
        row["teacher_score_source"] = "execution_waypoint_cost_train_only_for_projection"
        row["teacher_model_id"] = "execution_waypoint_teacher_v1"
        row["waypoint_task_label_used_for_fit"] = bool(str(row.get("split")) == "train")
        keys.append(key)
        features.append(np.concatenate([predicted.astype(np.float64), target]))

    feature_array = np.stack(features)
    goal_features = np.zeros_like(feature_array)
    train_indices = np.asarray([i for i, row in enumerate(rows) if str(row.get("split")) == "train"], dtype=np.int64)
    if len(train_indices) < projection_rank:
        raise ValueError("not enough train rows for waypoint projection")
    centered = feature_array[train_indices]
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    projection = vectors[:projection_rank].copy()
    groups: dict[tuple[str, str], list[int]] = {}
    for index in train_indices.tolist():
        groups.setdefault((str(rows[index]["instance_id"]), str(rows[index]["teacher_set_id"])), []).append(index)
    pairs: list[tuple[int, int]] = []
    for group in groups.values():
        for left_pos, left in enumerate(group):
            for right in group[left_pos + 1 :]:
                delta = float(rows[left]["teacher_score"]) - float(rows[right]["teacher_score"])
                if abs(delta) > 1.0e-3:
                    pairs.append((left, right) if delta > 0 else (right, left))
    if not pairs:
        raise ValueError("no train preference pairs for waypoint projection")
    for _ in range(int(projection_epochs)):
        gradient = np.zeros_like(projection)
        for preferred, rejected in pairs:
            xp = projection @ feature_array[preferred]
            xr = projection @ feature_array[rejected]
            violation = 0.1 + float(xp @ xp - xr @ xr) / projection_rank
            weight = 1.0 / (1.0 + np.exp(-np.clip(violation, -60.0, 60.0)))
            gradient += weight * (
                2.0 * np.outer(xp, feature_array[preferred]) / projection_rank
                - 2.0 * np.outer(xr, feature_array[rejected]) / projection_rank
            )
        projection -= 0.05 * gradient / max(len(pairs), 1)
        projection /= max(float(np.mean(np.linalg.svd(projection, compute_uv=False))), 1.0e-8)

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    metadata = dict(payload.get("metadata", {}))
    metadata.update({
        "schema": "gdd_wm_pointmaze_waypoint_execution_teacher.v0.1",
        "teacher_model_id": "execution_waypoint_teacher_v1",
        "teacher_type": "execution_supervised_waypoint_conditioned",
        "goal_conditioning": "active_rgb_waypoint",
        "waypoint_target_source": "calibration_only_rgb_maze_map",
        "dynamics_frozen": True,
        "teacher_runtime_forbidden": True,
        "outcome_used_for_projection_selection": "train_roots_only",
        "source_dataset_sha256": _sha256(input_path),
        "adapter_checkpoint": str(adapter_checkpoint.resolve()),
        "calibration_episodes": calibration_episodes,
        "calibration_frames": int(calibration_frames),
        "projection_rank": int(projection_rank),
        "projection_epochs": int(projection_epochs),
        "label_blocks": int(label_blocks),
        "teacher_horizon": f"{int(label_blocks)}_executed_5_step_blocks",
        "claim_boundary": "execution-supervised waypoint teacher control; not D-JEPA validation",
    })
    output_dataset = output_dir / "candidate_dataset.json"
    output_dataset.write_text(json.dumps({"metadata": metadata, "records": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        output_dir / "student_features.npz",
        keys=np.asarray(keys),
        features=feature_array,
        goal_features=goal_features,
        metadata_json=np.asarray(json.dumps({
            "schema": "gdd_wm_student_feature_table.v0.1",
            "dataset_sha256": _sha256(output_dataset),
            "key_schema": "instance_id::teacher_set_id::candidate_id",
            "candidate_set_context_input_forbidden": True,
            "teacher_rank_input_forbidden": True,
            "held_out_outcome_input_forbidden": True,
            "dynamics_frozen": True,
            "goal_anchor": "zero",
            "goal_conditioning": "active_rgb_waypoint",
        }, sort_keys=True)),
    )
    projection_report = {
        "schema": "gdd_wm_waypoint_projection.v0.1",
        "projection": projection.tolist(),
        "protocol": {
            "dataset": str(output_dataset.resolve()),
            "feature_table": str((output_dir / "student_features.npz").resolve()),
            "fit_split": "train",
            "dynamics_frozen": True,
            "teacher_runtime_forbidden": True,
            "goal_conditioning": "active_rgb_waypoint",
            "outcome_used_for_projection_selection": "train_roots_only",
        },
        "summary": {
            "record_count": len(rows),
            "train_row_count": int(len(train_indices)),
            "train_pair_count": len(pairs),
            "projection_rank": int(projection_rank),
        },
        "claim_boundary": "execution-supervised waypoint geometry; no closed-loop claim before independent evaluation",
    }
    projection_path = output_dir / "projection.json"
    projection_path.write_text(json.dumps(projection_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "gdd_wm_pointmaze_waypoint_projection_package.v0.1",
        "candidate_dataset": str(output_dataset.resolve()),
        "arrays": str(arrays_path.resolve()),
        "feature_table": str((output_dir / "student_features.npz").resolve()),
        "projection": str(projection_path.resolve()),
        "record_count": len(rows),
        "root_count": len({str(row["instance_id"]) for row in rows}),
        "split_root_counts": {split: len({str(row["instance_id"]) for row in rows if str(row.get("split")) == split}) for split in sorted({str(row.get("split")) for row in rows})},
        "claim_boundary": metadata["claim_boundary"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-episodes", type=int, nargs="+", required=True)
    parser.add_argument("--calibration-frames", type=int, default=768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--projection-rank", type=int, default=4)
    parser.add_argument("--projection-epochs", type=int, default=200)
    parser.add_argument("--label-blocks", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(build(
        args.input, args.dataset_root, args.adapter_checkpoint, args.output,
        calibration_episodes=args.calibration_episodes,
        calibration_frames=args.calibration_frames,
        device=args.device,
        projection_rank=args.projection_rank,
        projection_epochs=args.projection_epochs,
        label_blocks=args.label_blocks,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
