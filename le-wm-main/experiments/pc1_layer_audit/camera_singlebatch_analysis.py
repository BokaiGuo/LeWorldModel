"""Evaluation-only batch parity repair; preserve model, predictions and thresholds."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pc1_layer_audit import camera, camera_runtime_bound
from experiments.pc1_layer_audit.metrics import ranking_record
from experiments.pc_input_permissions import file_hash, STATE
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.research_runtime.artifacts import atomic_write_json

OUT = camera_runtime_bound.OUT


def main():
    camera.OUT = OUT
    for stage in ("prepare", "predict", "replay"):
        camera.read_stage(stage)
    if (OUT / "analyze.npz").exists():
        raise FileExistsError(OUT / "analyze.npz")
    predicted = json.loads((OUT / "predict.json").read_text())
    model, _, _ = camera.load_model()
    fingerprint = model_fingerprint(model)
    if fingerprint != predicted["model_fingerprint"]:
        raise ValueError("model drift")
    backbone = model.model.encoder.base_model.model
    captured, buffers = {}, {i: [] for i in range(12)}
    def hook(i):
        def capture(_m, _in, value):
            if value.shape != (1, 257, 384):
                raise ValueError("single-frame full-token shape required")
            captured[i] = value.detach()
        return capture
    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(backbone.encoder.layer)]
    official, actual_prop = [], []
    max_error = 0.
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        with np.load(OUT / "prepare.npz", allow_pickle=False) as prep, np.load(OUT / "replay.npz", allow_pickle=False) as replay, torch.inference_mode():
            truth = replay["physical_costs"]
            for r, root in enumerate(camera.ROOTS):
                for v in range(2):
                    images = [prep["frames"][r, 0, v]] + [
                        replay["images"][r, a, v, k] for k in (1, 2, 3) for a in range(3)
                    ] + [prep["frames"][r, 1, v]]
                    states = [prep["states"][r, 0]] + [
                        replay["traces"][r, a, v, t] for t in camera.HORIZONS for a in range(3)
                    ] + [prep["states"][r, 1]]
                    for image, state in zip(images, states, strict=True):
                        encoded = model.encode({"visual": camera.raw_rgb(image[None]),
                            "proprio": torch.tensor(state, dtype=torch.float32, device="cuda")[None, None]})
                        visual = encoded["visual"].reshape(1, 256, 384)
                        final = backbone.layernorm(captured[11][:, 1:])
                        if model.normalize_reps:
                            final = torch.nn.functional.layer_norm(final, (384,))
                        torch.testing.assert_close(visual, final, atol=1e-5, rtol=1e-5)
                        max_error = max(max_error, float((visual-final).abs().max()))
                        official.append(visual.cpu().numpy())
                        actual_prop.append(encoded["proprio"].cpu().numpy().reshape(1, -1))
                        for i in range(12):
                            buffers[i].append(captured[i].cpu().numpy())
                        captured.clear()
                print(f"single-frame encoding root={root}", flush=True)
        torch.cuda.synchronize()
    finally:
        for h in handles:
            h.remove()
    encoding_seconds = time.perf_counter() - started
    official = np.concatenate(official).reshape(32, 2, 11, 256, 384)
    actual_prop = np.concatenate(actual_prop).reshape(32, 2, 11, -1)
    arrays = dict(physical_costs=truth, official_real_features=official, actual_proprio_features=actual_prop)
    result = dict(track=STATE, scope="offline diagnostics; no online calibration or gate promotion",
                  roots=camera.ROOTS, official_output_max_error=max_error, encoding_seconds=encoding_seconds,
                  encoding_batch_size=1, representations={}, predictor_transfer=[],
                  repair="batch-22 goal parity failed; match original batch-1 goal path without relaxing tolerance")
    # Check target parity before expensive all-layer analysis.
    goal_max_error = 0.
    with np.load(OUT / "predict.npz", allow_pickle=False) as prediction:
        for r, root in enumerate(camera.ROOTS):
            row = dict(root=root, horizons=[])
            for h, steps in enumerate(camera.HORIZONS):
                item = dict(environment_steps=steps, views=[])
                for v in (0, 1):
                    pv = prediction[f"r{root}_v{v}_pred_visual"][steps//5].reshape(3, 256, 384)
                    pp = prediction[f"r{root}_v{v}_pred_proprio"][steps//5].reshape(3, -1)
                    gv = prediction[f"r{root}_v{v}_goal_visual"].reshape(256, 384)
                    gp = prediction[f"r{root}_v{v}_goal_proprio"].reshape(-1)
                    np.testing.assert_allclose(gv, official[r, v, 10], atol=1e-4, rtol=1e-4)
                    np.testing.assert_allclose(gp, actual_prop[r, v, 10], atol=1e-5, rtol=1e-5)
                    goal_max_error = max(goal_max_error, float(np.max(np.abs(gv-official[r, v, 10]))))
                    actual = official[r, v, 1+h*3:4+h*3]
                    prop = actual_prop[r, v, 1+h*3:4+h*3]
                    predicted_visual = ((pv-gv)**2).mean((1, 2))
                    actual_visual = ((actual-gv)**2).mean((1, 2))
                    actual_official = actual_visual + .1*((prop-gp)**2).mean(-1)
                    pred_official = prediction[f"r{root}_v{v}_cost"][steps//5]
                    np.testing.assert_allclose(pred_official, predicted_visual + .1*((pp-gp)**2).mean(-1), atol=1e-5, rtol=1e-5)
                    item["views"].append(dict(camera=v,
                        visual_prediction_mse=float(np.mean((pv-actual)**2)),
                        visual_persistence_mse=float(np.mean((official[r, v, 0]-actual)**2)),
                        proprio_prediction_mse=float(np.mean((pp-prop)**2)),
                        proprio_persistence_mse=float(np.mean((actual_prop[r, v, 0]-prop)**2)),
                        predicted_visual=ranking_record(predicted_visual, truth[r, h]),
                        predicted_official=ranking_record(pred_official, truth[r, h]),
                        real_visual=ranking_record(actual_visual, truth[r, h]),
                        real_official=ranking_record(actual_official, truth[r, h]),
                        predicted_vs_encoded=ranking_record(pred_official, actual_official),
                        physical_costs=truth[r, h].tolist(), predicted_visual_costs=predicted_visual.tolist(),
                        real_visual_costs=actual_visual.tolist(), predicted_official_costs=pred_official.tolist(),
                        real_official_costs=actual_official.tolist()))
                row["horizons"].append(item)
            result["predictor_transfer"].append(row)
    result["goal_encoding_max_error"] = goal_max_error
    print(f"Goal encoding parity max error={goal_max_error}", flush=True)
    started = time.perf_counter()
    for i in range(13):
        x = np.concatenate(buffers.pop(i)).reshape(32, 2, 11, 257, 384) if i < 12 else official
        for mode in (("full", "cls") if i < 12 else ("official",)):
            name = f"block_{i+1:02d}_{mode}" if i < 12 else "official"
            values = x if mode != "cls" else x[..., :1, :]
            record, mapping = camera.representation_audit(values.astype(np.float64), truth)
            result["representations"][name] = record
            for key in ("rotation", "source_mean", "target_mean"):
                arrays[name+"_"+key] = getattr(mapping, key)
            if mode == "cls":
                arrays[name+"_features"] = values
            print(f"{name} complete", flush=True)
    result["layer_analysis_seconds"] = time.perf_counter()-started
    if fingerprint != model_fingerprint(model):
        raise ValueError("model changed")
    dependencies = (Path(__file__), Path(camera.__file__), Path(camera.__file__).with_name("metrics.py"),
                    *[OUT / f"{s}.json" for s in ("prepare", "predict", "replay")])
    result.update(model_fingerprint=fingerprint, arrays_sha256=camera.save_arrays("analyze", **arrays),
                  sources={str(p): file_hash(p) for p in dependencies})
    atomic_write_json(OUT / "analyze.json", result)
    print("All single-batch layer and predictor-transfer diagnostics complete.", flush=True)


if __name__ == "__main__":
    main()
