"""Verify PointMaze candidate successors through realized RGB latents.

This is the missing D1 bridge for the PointMaze decision benchmark.  Candidate
actions are executed by the same legacy MuJoCo dynamics used for the offline
labels.  The realized terminal position is then rendered with a dataset-matched
PointMaze renderer: the static scene is recovered from the observed RGB
episode and the moving point is placed using the calibrated state-to-pixel
map.  This avoids the known MuJoCo offscreen-camera/material mismatch while
keeping the successor dynamics independent of the frozen visual encoder.

The script reports two separate gates:

* renderer consistency: the dataset-matched renderer reconstructs held-out
  observed frames from their physical state;
* RGB successor consistency: the frozen JEPA latent of the realized terminal
  RGB is compared with the adapter's predicted terminal latent.

No simulator state is passed to the encoder or adapter.  State is used only
to execute the fixed candidate bank and to construct the post-hoc realized
RGB frame.  This artifact therefore establishes visual successor verification
under a declared simulator renderer, not physical causality or universal
online planning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.empirical_spectral_audit.evaluate_pointmaze_adapter_external_evidence import (
    _legacy_pointmaze_xml,
)
from experiments.theory_spectral_transfer.run_jepa_wm_pointmaze_temporal_eval import (
    canonical_pooled_latent,
)


STATE_TO_PIXEL = np.asarray([27.0, 27.77], dtype=np.float64)
GREEN_MASK_THRESHOLD = 100.0
MARKER_RADIUS = 9


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _green_center(image: np.ndarray) -> tuple[float, float]:
    image = np.asarray(image)
    mask = (
        (image[..., 1] > GREEN_MASK_THRESHOLD)
        & (image[..., 1] > image[..., 0] * 1.25)
        & (image[..., 1] > image[..., 2] * 1.15)
        & (image[..., 0] < GREEN_MASK_THRESHOLD)
    )
    y, x = np.where(mask)
    if len(x) == 0:
        raise ValueError("PointMaze RGB frame has no detectable green moving-point marker")
    return float(x.mean()), float(y.mean())


def _static_scene_and_marker(
    images: np.ndarray,
    marker_image: np.ndarray,
    marker_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Recover a static RGB scene and a source marker residual template."""

    images = np.asarray(images, dtype=np.int16)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"expected THWC RGB episode, got {images.shape}")
    base = np.median(images, axis=0).astype(np.int16)
    source = np.asarray(marker_image)
    source_center = _green_center(source)
    predicted_center = STATE_TO_PIXEL * np.asarray(marker_state[:2], dtype=np.float64)
    offset = np.asarray(source_center) - predicted_center
    center = STATE_TO_PIXEL * np.asarray(marker_state[:2], dtype=np.float64) + offset
    x0, y0 = np.rint(center).astype(int)
    r = MARKER_RADIUS
    if x0 - r < 0 or y0 - r < 0 or x0 + r >= images.shape[2] or y0 + r >= images.shape[1]:
        raise ValueError("source marker patch falls outside RGB frame")
    residual = source[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1] - base[y0 - r : y0 + r + 1, x0 - r : x0 + r + 1]
    return base, residual, (float(offset[0]), float(offset[1]))


def _render_dataset_matched(
    base: np.ndarray,
    residual: np.ndarray,
    state: np.ndarray,
    offset: tuple[float, float],
) -> np.ndarray:
    """Render a state using the observed PointMaze RGB scene convention."""

    output = np.asarray(base, dtype=np.int16).copy()
    center = STATE_TO_PIXEL * np.asarray(state[:2], dtype=np.float64) + np.asarray(offset)
    x0, y0 = np.rint(center).astype(int)
    r = MARKER_RADIUS
    h, w = output.shape[:2]
    y_start, y_stop = max(0, y0 - r), min(h, y0 + r + 1)
    x_start, x_stop = max(0, x0 - r), min(w, x0 + r + 1)
    dy_start, dy_stop = y_start - (y0 - r), y_stop - (y0 - r)
    dx_start, dx_stop = x_start - (x0 - r), x_stop - (x0 - r)
    output[y_start:y_stop, x_start:x_stop] += residual[dy_start:dy_stop, dx_start:dx_stop]
    return np.clip(output, 0, 255).astype(np.uint8)


def _renderer_reconstruction(
    images: np.ndarray,
    states: np.ndarray,
    base: np.ndarray,
    residual: np.ndarray,
    offset: tuple[float, float],
) -> dict[str, float]:
    errors = []
    for image, state in zip(images, states):
        reconstructed = _render_dataset_matched(base, residual, state, offset)
        errors.append(float(np.mean(np.abs(reconstructed.astype(np.float64) - image.astype(np.float64)))))
    values = np.asarray(errors, dtype=np.float64)
    return {
        "frames": int(len(values)),
        "mean_abs_pixel_error_0_255": float(np.mean(values)),
        "p95_abs_pixel_error_0_255": float(np.quantile(values, 0.95)),
        "max_abs_pixel_error_0_255": float(np.max(values)),
        "normalized_mean_l1": float(np.mean(values) / max(float(np.ptp(images)), 1.0)),
    }


def _simulate_blocks(model: Any, source_state: np.ndarray, sequence: np.ndarray) -> list[np.ndarray]:
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(source_state[:2], dtype=np.float64)
    data.qvel[:] = np.asarray(source_state[2:4], dtype=np.float64)
    mujoco.mj_forward(model, data)
    sequence = np.asarray(sequence, dtype=np.float64)
    if len(sequence) % 5:
        raise ValueError("candidate action sequence must contain complete five-step blocks")
    for step_index, action in enumerate(sequence):
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
        if (step_index + 1) % 5 == 0:
            yield np.concatenate([np.asarray(data.qpos[:2]), np.asarray(data.qvel[:2])])


def _load_visual_encoder(repo_root: Path, checkpoint: Path, dataset_root: Path, device: str, dinov2_source: Path):
    sys.path.insert(0, str(repo_root / "external/jepa-wms"))
    from app.plan_common.datasets.point_maze_dset import PointMazeDataset
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_transforms
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module

    if not dinov2_source.exists():
        raise FileNotFoundError(f"DINOv2 torch hub source does not exist: {dinov2_source}")
    original_hub_load = torch.hub.load

    def hub_load(repo_or_dir, model, *extra, **kwargs):
        if repo_or_dir == "facebookresearch/dinov2":
            return original_hub_load(str(dinov2_source), model, source="local", *extra, **kwargs)
        return original_hub_load(repo_or_dir, model, *extra, **kwargs)

    torch.hub.load = hub_load
    config_path = repo_root / "external/jepa-wms/configs/vjepa_wm/mz_sweep/mz_4f_fsk5_ask1_r224_vjtranoaug_predAdaLN_ftprop_depth6_repro_2roll_save_2n.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    aug = config["data_aug"]
    transform = make_transforms(
        img_size=224,
        normalize=aug["normalize"],
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    stats = PointMazeDataset(data_path=str(dataset_root), normalize_action=True)
    preprocessor = Preprocessor(
        stats.action_mean,
        stats.action_std,
        stats.state_mean,
        stats.state_std,
        stats.proprio_mean,
        stats.proprio_std,
        transform,
    )
    runtime_device = torch.device(device)
    model = init_module(
        str(checkpoint.parent), checkpoint.name, config["model"], runtime_device,
        action_dim=2, proprio_dim=4, preprocessor=preprocessor,
        cfgs_data=config["data"], wrapper_kwargs={"ctxt_window": 1},
    )
    model.eval()
    return model, transform


def _encode_sequences(model: Any, transform: Any, sequences: list[list[np.ndarray]], device: str) -> np.ndarray:
    # The released external export was produced through PointMazeDataset's
    # normalized transform followed by EncPredWM.encode, which applies its
    # preprocessor transform once more.  Reproduce that declared interface so
    # the successor comparison is not confounded by a preprocessing shift.
    raw = torch.from_numpy(np.stack(sequences)).permute(0, 1, 4, 2, 3).float() / 255.0
    batch = transform(raw) * 255.0
    with torch.no_grad():
        encoded = model.encode(batch.to(device))
    if encoded.ndim != 6:
        raise ValueError(f"expected temporal visual encoding [B,T,V,H,W,C], got {tuple(encoded.shape)}")
    rows = []
    for batch_item in encoded:
        rows.append(np.stack([
            canonical_pooled_latent(time_item, feature_dim=384).cpu().numpy()
            for time_item in batch_item
        ]))
    return np.stack(rows).astype(np.float64)


def verify(
    candidate_dataset: Path,
    arrays_path: Path,
    dataset_root: Path,
    encoder_checkpoint: Path,
    output: Path,
    *,
    max_roots: int,
    max_candidates: int,
    device: str,
    dinov2_source: Path,
    source_consistency_threshold: float,
    root_split: str | None = None,
) -> dict[str, Any]:
    dataset = _load_json(candidate_dataset)
    records = list(dataset["records"])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["instance_id"]), []).append(record)
    if root_split is not None:
        grouped = {
            key: value for key, value in grouped.items()
            if str(value[0]["split"]) == str(root_split)
        }
    roots = sorted(grouped, key=lambda key: (str(grouped[key][0]["split"]), key))[:max_roots]
    if not roots:
        raise ValueError("candidate dataset has no roots")
    arrays = np.load(arrays_path, allow_pickle=False)
    adapter_checkpoint = Path(dataset["metadata"]["checkpoint"])
    adapter_payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=False)
    pca_mean = np.asarray(adapter_payload["pca_mean"], dtype=np.float64)
    pca_basis = np.asarray(adapter_payload["pca_basis"], dtype=np.float64)
    encoder, transform = _load_visual_encoder(PROJECT_ROOT, encoder_checkpoint, dataset_root, device, dinov2_source)
    import mujoco

    physics = mujoco.MjModel.from_xml_string(_legacy_pointmaze_xml())
    all_rows: list[dict[str, Any]] = []
    root_reports: list[dict[str, Any]] = []
    for root_index, root in enumerate(roots):
        group = sorted(grouped[root], key=lambda item: int(item["candidate_id"]))[:max_candidates]
        episode = int(group[0]["start_episode_id"])
        images = torch.load(dataset_root / "obses" / f"episode_{episode}.pth", map_location="cpu", weights_only=True).numpy()
        states = torch.load(dataset_root / "states.pth", map_location="cpu", weights_only=True)[episode].numpy()
        source_frame = int(group[0].get(
            "source_frame",
            dataset["metadata"].get(
                "source_frame",
                int(dataset["metadata"].get("source_window_start", 0)) + 5,
            ),
        ))
        if source_frame < 0 or source_frame >= len(images):
            raise ValueError(f"source_frame {source_frame} outside episode RGB length {len(images)}")
        base, residual, offset = _static_scene_and_marker(images, images[source_frame], states[source_frame])
        renderer = _renderer_reconstruction(images, states, base, residual, offset)
        context_indices = [source_frame - 5, source_frame]
        if context_indices[0] < 0:
            raise ValueError(f"source frame {source_frame} has no preceding five-step context")
        candidate_sequences: list[list[np.ndarray]] = []
        terminal_states: list[np.ndarray] = []
        terminal_images: list[np.ndarray] = []
        for record in group:
            blocks = list(_simulate_blocks(physics, np.asarray(record["source_state_raw"], dtype=np.float64), np.asarray(record["action_sequence"], dtype=np.float64)))
            terminal = blocks[-1]
            terminal_states.append(terminal)
            rendered_blocks = [_render_dataset_matched(base, residual, state, offset) for state in blocks]
            terminal_images.append(rendered_blocks[-1])
            candidate_sequences.append([np.asarray(images[index], dtype=np.uint8) for index in context_indices] + rendered_blocks)
        encoded = _encode_sequences(encoder, transform, candidate_sequences, device)
        source_projection = (encoded[0, 1] - pca_mean) @ pca_basis.T
        source_error = float(np.mean(np.abs(source_projection - np.asarray(group[0]["source_latent_model"], dtype=np.float64))))
        source_consistency = {
            "renderer": renderer,
            "source_latent_projection_l1": source_error,
            "passed": bool(renderer["normalized_mean_l1"] <= source_consistency_threshold),
            "threshold_normalized_mean_l1": float(source_consistency_threshold),
        }
        for candidate_index, (record, terminal, realized) in enumerate(zip(group, terminal_images, terminal_states)):
            key = f"predicted::{root}/{int(record['candidate_id']):03d}"
            predicted = np.asarray(arrays[key], dtype=np.float64)
            # The adapter checkpoint stores an 8D PCA projection.  Recover the
            # same projection from the source export's persisted source vector
            # dimensionality via the checkpoint-side source model metadata.
            source_model = np.asarray(record["source_latent_model"], dtype=np.float64)
            if predicted.shape != source_model.shape:
                raise ValueError(f"predicted/source projection shape mismatch: {predicted.shape} vs {source_model.shape}")
            realized_raw = encoded[candidate_index, 1 + int(record["predicted_horizon"])]
            realized_projection = (realized_raw - pca_mean) @ pca_basis.T
            projection_error = realized_projection - predicted
            all_rows.append({
                "instance_id": root,
                "candidate_id": int(record["candidate_id"]),
                "split": str(record["split"]),
                "source_episode_id": episode,
                "renderer_consistency_pass": bool(source_consistency["passed"]),
                "source_latent_projection_l1": source_error,
                "source_projection": source_projection.tolist(),
                "predicted_terminal_projection": predicted.tolist(),
                "realized_terminal_projection": realized_projection.tolist(),
                "predicted_vs_realized_projection_l1": float(np.mean(np.abs(projection_error))),
                "predicted_vs_realized_projection_rmse": float(np.sqrt(np.mean(projection_error ** 2))),
                "predicted_vs_realized_projection_relative_l2": float(
                    np.linalg.norm(projection_error) / max(np.linalg.norm(realized_projection), 1e-8)
                ),
                "realized_terminal_state": realized.tolist(),
                "realized_terminal_rgb_mean": float(np.mean(terminal)),
                "realized_terminal_raw_latent": realized_raw.tolist(),
                "prediction_projection_status": "matched_adapter_pca_projection",
            })
        root_reports.append({
            "instance_id": root,
            "episode_id": episode,
            "split": str(group[0]["split"]),
            "candidate_count": len(group),
            "source_consistency": source_consistency,
        })
        print(f"[D1] root {root_index + 1}/{len(roots)} {root}: renderer_norm_l1={renderer['normalized_mean_l1']:.5f}, source_latent_l1={source_error:.5f}")
    output.mkdir(parents=True, exist_ok=True)
    def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        predicted = np.asarray([row["predicted_terminal_projection"] for row in rows], dtype=np.float64)
        realized = np.asarray([row["realized_terminal_projection"] for row in rows], dtype=np.float64)
        source = np.asarray([row["source_projection"] for row in rows], dtype=np.float64)
        pred_error = np.linalg.norm(predicted - realized, axis=1)
        baseline_error = np.linalg.norm(source - realized, axis=1)
        return {
            "rows": int(len(rows)),
            "predicted_vs_realized_l2_rmse": float(np.sqrt(np.mean((predicted - realized) ** 2))),
            "source_baseline_vs_realized_l2_rmse": float(np.sqrt(np.mean((source - realized) ** 2))),
            "mean_predicted_vs_realized_l2": float(np.mean(pred_error)),
            "mean_source_baseline_vs_realized_l2": float(np.mean(baseline_error)),
            "relative_l2_improvement_over_source_baseline": float(1.0 - np.mean(pred_error) / max(np.mean(baseline_error), 1e-12)),
            "fraction_predicted_better_than_source_baseline": float(np.mean(pred_error < baseline_error)),
        }

    split_summaries = {}
    record_split = {(str(item["instance_id"]), int(item["candidate_id"])): str(item["split"]) for item in records}
    for split in ("train", "calibration", "outcome"):
        split_rows = [
            row for row in all_rows
            if record_split[(str(row["instance_id"]), int(row["candidate_id"]))] == split
        ]
        split_summaries[split] = _summary(split_rows) if split_rows else {"rows": 0}
    report = {
        "schema": "pointmaze_rgb_successor_verification.v1",
        "protocol": {
            "candidate_dataset": str(candidate_dataset.resolve()),
            "encoder_checkpoint": str(encoder_checkpoint.resolve()),
            "dataset_root": str(dataset_root.resolve()),
            "renderer": "dataset_matched_static_scene_plus_mujoco_terminal_qpos",
            "runtime_encoder_input": "realized RGB only through the frozen export preprocessing contract",
            "preprocessing_contract": "PointMazeDataset transform followed by EncPredWM.encode preprocessing (reproduces frozen external export)",
            "temporal_context": "observed frame source_frame-5, observed source_frame, then realized candidate states after each five-step action block",
            "terminal_readout": "encoded temporal index 1 + predicted_horizon; no single-frame successor encoding",
            "simulator_state_role": "action execution and posthoc renderer construction only",
            "source_consistency_threshold_normalized_mean_l1": float(source_consistency_threshold),
            "max_roots": int(max_roots),
            "max_candidates": int(max_candidates),
            "root_split_filter": root_split,
        },
        "root_reports": root_reports,
        "rows": all_rows,
        "summary": {
            "all": _summary(all_rows),
            "by_split": split_summaries,
            "interpretation": "A positive relative_l2_improvement means the action-conditioned predictor is closer to the realized RGB successor than the no-action/source-latent baseline in the declared PCA space.",
        },
        "gates": {
            "renderer_consistency": bool(root_reports) and bool(all(item["source_consistency"]["passed"] for item in root_reports)),
            "realized_rgb_latents_available": bool(all_rows),
            "pca_projected_successor_comparison": bool(all_rows),
        },
        "claim_boundary": "RGB successor latent extraction with dataset-matched simulated renderer; physical causality, universal transfer, and closed-loop planning remain unproven.",
    }
    (output / "rgb_successor_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "realized_latents.npz",
        **{f"raw::{row['instance_id']}/{row['candidate_id']:03d}": np.asarray(row["realized_terminal_raw_latent"], dtype=np.float32) for row in all_rows},
        **{f"projection::{row['instance_id']}/{row['candidate_id']:03d}": np.asarray(row["realized_terminal_projection"], dtype=np.float32) for row in all_rows},
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-roots", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dinov2-source", type=Path, default=PROJECT_ROOT / "external/local_dinov2")
    parser.add_argument("--source-consistency-threshold", type=float, default=0.10)
    parser.add_argument("--root-split", choices=("train", "calibration", "outcome"), default=None)
    args = parser.parse_args()
    report = verify(
        args.candidate_dataset, args.arrays, args.dataset_root, args.encoder_checkpoint,
        args.output, max_roots=args.max_roots, max_candidates=args.max_candidates,
        device=args.device, dinov2_source=args.dinov2_source,
        source_consistency_threshold=args.source_consistency_threshold,
        root_split=args.root_split,
    )
    print(json.dumps(report["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
