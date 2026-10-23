"""Rebuild train/calibration effect-policy sources in the live RGB chart.

The original PointMaze counterfactual bank stores a latent exported through a
different preprocessing route than ``run_live_pointmaze_planning``.  This
utility replaces only train/calibration source features with the exact causal
two-frame RGB chart used at deployment.  Outcome rows and their labels are
copied untouched, so they cannot influence fitting or chart selection.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.run_live_pointmaze_planning import (
    FRAME_STRIDE,
    DatasetMatchedRenderer,
    _make_context,
    _make_gdd_source_contract_context,
    _load_jepa_model,
    _pool_visual,
    fit_visual_calibration,
    green_center,
    load_episode,
)


def _live_source_projection(
    model: Any,
    renderer: DatasetMatchedRenderer,
    states: np.ndarray,
    frame: int,
    calibration: Any,
    mean: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    if frame < FRAME_STRIDE or frame % FRAME_STRIDE:
        raise ValueError("source frame must be a positive FRAME_STRIDE multiple")
    images: list[np.ndarray] = []
    proprios: list[np.ndarray] = []
    previous_center: tuple[float, float] | None = None
    for index in range(0, frame + 1, FRAME_STRIDE):
        image = renderer.render(states[index])
        center = green_center(image)
        velocity = (0.0, 0.0) if previous_center is None else calibration.velocity(
            (center[0] - previous_center[0], center[1] - previous_center[1])
        )
        images.append(image)
        proprios.append(np.concatenate([calibration.position(center), velocity]))
        previous_center = center
    context = _make_context(images, proprios)
    transform = getattr(model, "_gdd_source_contract_transform", None)
    if callable(transform):
        context = _make_gdd_source_contract_context(context, transform)
    with torch.no_grad():
        encoded = model.encode(context)
        pooled = _pool_visual(encoded["visual"][:, -1])[0].detach().cpu().numpy()
    return ((np.asarray(pooled, dtype=np.float64) - mean) @ basis.T).astype(np.float64)


def build(
    dataset: Path,
    arrays: Path,
    source_model_checkpoint: Path,
    dataset_root: Path,
    jepa_checkpoint: Path,
    dinov2_source: Path,
    calibration_episodes: list[int],
    output: Path,
) -> dict[str, Any]:
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    rows = list(payload["records"])
    source_payload = torch.load(source_model_checkpoint, map_location="cpu", weights_only=False)
    mean = np.asarray(source_payload["pca_mean"], dtype=np.float64)
    basis = np.asarray(source_payload["pca_basis"], dtype=np.float64)
    device = torch.device("cuda")
    model, _transform, _stats, _action_mean, _action_std = _load_jepa_model(
        output.parents[2], dataset_root, jepa_checkpoint, dinov2_source, device,
    )
    calibration = fit_visual_calibration(dataset_root, calibration_episodes, 768)
    cache: dict[int, tuple[DatasetMatchedRenderer, np.ndarray]] = {}
    replaced_roots: set[str] = set()
    for row in rows:
        if str(row.get("split")) not in {"train", "calibration"}:
            continue
        root = str(row["instance_id"])
        if root not in replaced_roots:
            episode = int(row["start_episode_id"])
            images, states = load_episode(dataset_root, episode)
            cache[episode] = (DatasetMatchedRenderer.from_episode(images, calibration), states)
            projection = _live_source_projection(
                model, cache[episode][0], states, int(row["source_frame"]),
                calibration, mean, basis,
            )
            replaced_roots.add(root)
        else:
            episode = int(row["start_episode_id"])
            projection = _live_source_projection(
                model, cache[episode][0], cache[episode][1], int(row["source_frame"]),
                calibration, mean, basis,
            )
        row["source_latent_model"] = projection.tolist()
        row["source_history_model"] = projection.tolist()
        row["source_chart"] = "live_rgb_two_frame_encoder_pca_v1"
    metadata = dict(payload.get("metadata", {}))
    metadata.update({
        "source_chart": "live_rgb_two_frame_encoder_pca_v1",
        "source_chart_replaced_splits": ["train", "calibration"],
        "source_chart_outcome_rows_untouched": True,
        "source_chart_calibration_episodes": calibration_episodes,
        "source_chart_contract": "same_renderer_context_transform_and_encoder_as_live_pointmaze_policy",
    })
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(arrays, output / "arrays.npz")
    (output / "candidate_dataset.json").write_text(
        json.dumps({"metadata": metadata, "records": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema": "pointmaze_effect_rgb_source_chart.v1",
        "replaced_root_count": len(replaced_roots),
        "replaced_splits": ["train", "calibration"],
        "outcome_rows_untouched": True,
        "claim_boundary": "source-chart repair artifact; effect-policy and closed-loop validation remain separate",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--source-model-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--dinov2-source", type=Path, required=True)
    parser.add_argument("--calibration-episodes", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes = [int(value) for value in args.calibration_episodes.split(",") if value]
    print(json.dumps(build(args.dataset, args.arrays, args.source_model_checkpoint, args.dataset_root, args.jepa_checkpoint, args.dinov2_source, episodes, args.output), indent=2))


if __name__ == "__main__":
    main()
