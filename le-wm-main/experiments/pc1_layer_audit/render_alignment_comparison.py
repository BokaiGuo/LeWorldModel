"""Paired evaluation repair: cached vs state-synchronized render, fixed predictions."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc1_layer_audit import camera
from experiments.pc1_layer_audit.metrics import ranking_record
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.research_runtime.artifacts import atomic_write_json

SOURCE = camera.ROOT / "results/pc1_camera_layer_transfer_runtime_bound_20261107"
OUT = camera.ROOT / "results/pc1_render_alignment_comparison_20261107"


def main():
    if (OUT / "result.json").exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((SOURCE / "prepare.json").read_text())
    predicted = json.loads((SOURCE / "predict.json").read_text())
    with np.load(SOURCE / "prepare.npz", allow_pickle=False) as data:
        states = data["states"][:, 0].copy()
        current_images = data["frames"][:, 0].copy()
    old, render_seconds, _ = camera.render("rollouts", states)
    with np.load(SOURCE / "replay.npz", allow_pickle=False) as repaired:
        np.testing.assert_array_equal(old["traces"], repaired["traces"])
        np.testing.assert_array_equal(old["times"], repaired["times"])
        np.testing.assert_array_equal(old["images"][:, :, :, 0], np.broadcast_to(current_images[:, None], old["images"][:, :, :, 0].shape))
        pixel_mse = ((old["images"].astype(float)-repaired["images"].astype(float))**2).mean((4, 5, 6))
        costs = repaired["physical_costs"].copy()
    arrays_path = OUT / "cached_render.npz"
    with arrays_path.open("xb") as f:
        np.savez_compressed(f, images=old["images"], traces=old["traces"], times=old["times"],
                            pixel_mse=pixel_mse)
    model, _, _ = camera.load_model()
    before = model_fingerprint(model)
    if before != predicted["model_fingerprint"]:
        raise ValueError("different model")
    rows, cached_features = [], []
    with np.load(SOURCE / "predict.npz", allow_pickle=False) as pred, np.load(SOURCE / "analyze.npz", allow_pickle=False) as new, torch.inference_mode():
        for r, root in enumerate(camera.ROOTS):
            images = old["images"][r, :, :, 1:].transpose(1, 2, 0, 3, 4, 5).reshape(18, 224, 224, 3)
            features = np.concatenate([model.encode(camera.raw_rgb(image[None])).cpu().numpy()
                                       for image in images]).reshape(2, 3, 3, 256, 384)
            cached_features.append(features)
            entries = []
            for v in (0, 1):
                goal = pred[f"r{root}_v{v}_goal_visual"].reshape(256, 384)
                for h, steps in enumerate(camera.HORIZONS):
                    prediction = pred[f"r{root}_v{v}_pred_visual"][steps//5].reshape(3, 256, 384)
                    current = new["official_real_features"][r, v, 1+3*h:4+3*h]
                    old_cost = ((features[v, h]-goal)**2).mean((1, 2))
                    new_cost = ((current-goal)**2).mean((1, 2))
                    entries.append(dict(camera=v, steps=steps,
                        cached_prediction_mse=float(np.mean((prediction-features[v, h])**2)),
                        synchronized_prediction_mse=float(np.mean((prediction-current)**2)),
                        cached_real_ranking=ranking_record(old_cost, costs[r, h]),
                        synchronized_real_ranking=ranking_record(new_cost, costs[r, h])))
            rows.append(dict(root=root, entries=entries))
            print(f"paired render evaluation root={root}", flush=True)
    if before != model_fingerprint(model):
        raise ValueError("model changed")
    feature_path = OUT / "cached_features.npz"
    with feature_path.open("xb") as f:
        np.savez_compressed(f, features=np.stack(cached_features))
    sources = {str(p): file_hash(p) for p in (Path(__file__), camera.WORKER,
              SOURCE / "prepare.json", SOURCE / "predict.json", SOURCE / "replay.json", SOURCE / "analyze.json")}
    atomic_write_json(OUT / "result.json", dict(
        scope="post-hoc evaluation alignment repair; selected plans unchanged; not control improvement",
        track=prepared["track"], sources=sources, physics_max_error=0., render_seconds=render_seconds,
        pixel_mse_mean_by_snapshot=pixel_mse.mean((0, 1, 2)).tolist(),
        arrays_sha256=file_hash(arrays_path), features_sha256=file_hash(feature_path), rows=rows))


if __name__ == "__main__":
    main()
