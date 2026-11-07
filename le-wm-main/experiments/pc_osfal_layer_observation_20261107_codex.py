"""Frozen, exposed-root PointMaze layer observation diagnostic."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_track_smoke_20261106_codex import ROOT, load_model

SOURCE = ROOT / "results/pc_real_camera_effect_dev_20261106_codex.npz"
PROTOCOL = ROOT.parent / "docs/PC_OSFAL_LAYER_OBSERVATION_DEV_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_osfal_layer_observation_20261107_codex.json"


def measure(features, positions, goal_positions):
    """features: root, view, (three final actions + goal), token, channel."""
    features = np.asarray(features, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    goal_positions = np.asarray(goal_positions, dtype=np.float64)
    if features.shape[:3] != (4, 2, 4) or positions.shape != (4, 3, 2):
        raise ValueError("unexpected input shapes")
    if goal_positions.shape != (4, 2) or not np.isfinite(features).all():
        raise ValueError("invalid goals or features")
    rows = []
    for root in range(4):
        gap = float(np.mean((features[root, 0, :3] - features[root, 1, :3]) ** 2))
        action_gap = float(np.mean([
            np.mean((features[root, v, i] - features[root, v, j]) ** 2)
            for v in range(2) for i, j in ((0, 1), (0, 2), (1, 2))
        ]))
        physical = np.linalg.norm(positions[root] - goal_positions[root], axis=-1)
        view_rows = []
        for view in range(2):
            scores = np.mean((features[root, view, :3] - features[root, view, 3]) ** 2,
                             axis=(-2, -1))
            choice = int(np.argmin(scores))
            view_rows.append({"camera": view, "encoded_costs": scores.tolist(),
                              "selected_action": choice,
                              "physical_regret": float(physical[choice] - physical.min())})
        rows.append({"root": 1880 + root, "view_gap_mse": gap, "action_gap_mse": action_gap,
                     "view_action_ratio": gap / action_gap if action_gap > 0 else None,
                     "physical_distance": physical.tolist(), "views": view_rows})
    return rows


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    print("Exposed-root frozen vision-layer observation diagnostic; no planner evaluation.", flush=True)
    with np.load(SOURCE, allow_pickle=False) as data:
        images, traces = data["images"], data["traces"]
        if images.shape != (8, 3, 2, 6, 224, 224, 3) or traces.shape != (8, 3, 2, 6, 4):
            raise ValueError("unexpected source shape")
        np.testing.assert_allclose(traces[:, :, 0], traces[:, :, 1], atol=1e-10, rtol=0)
        selected = np.stack([
            np.stack([np.stack([images[2*r, a, v, 5] for a in range(3)] +
                               [images[2*r+1, 0, v, 0]]) for v in range(2)])
            for r in range(4)])
        positions = traces[::2, :, 0, 5, :2].copy()
        goals = traces[1::2, 0, 0, 0, :2].copy()
    model, _, audit = load_model()
    before = model_fingerprint(model)
    backbone = model.model.encoder.base_model.model
    layers = backbone.encoder.layer
    if len(layers) != 12:
        raise ValueError("unexpected layer count")
    captured = {}
    handles = []
    def hook(index):
        def collect(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or output.shape[1:] != (257, 384):
                raise ValueError("unexpected token format")
            captured[index] = output[:, 1:].detach()
        return collect
    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(hook(index)))
    results = {f"block_{i+1:02d}": [] for i in range(12)}
    results["official_visual"] = []
    maximum_error = 0.0
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for root in range(4):
                images_tensor = torch.from_numpy(selected[root].copy()).reshape(8, 224, 224, 3)
                images_tensor = images_tensor.permute(0, 3, 1, 2)[:, None].cuda()
                official = model.encode(images_tensor).reshape(8, 256, 384)
                final = backbone.layernorm(captured[11])
                if model.normalize_reps:
                    final = torch.nn.functional.layer_norm(final, (final.size(-1),))
                maximum_error = max(maximum_error, float((official - final).abs().max()))
                torch.testing.assert_close(official, final, atol=1e-5, rtol=1e-5)
                for index in range(12):
                    results[f"block_{index+1:02d}"].append(
                        captured[index].cpu().numpy().reshape(2, 4, 256, 384))
                results["official_visual"].append(official.cpu().numpy().reshape(2, 4, 256, 384))
                captured.clear()
                print(f"root {1880 + root} encoded", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if before != model_fingerprint(model):
        raise ValueError("model weights changed")
    rows = {name: measure(np.stack(value), positions, goals) for name, value in results.items()}
    summary = {}
    for name, entries in rows.items():
        summary[name] = {
            "mean_view_gap_mse": float(np.mean([r["view_gap_mse"] for r in entries])),
            "mean_action_gap_mse": float(np.mean([r["action_gap_mse"] for r in entries])),
            "mean_root_view_action_ratio": float(np.mean([r["view_action_ratio"] for r in entries
                                                           if r["view_action_ratio"] is not None])),
            "mean_physical_regret": float(np.mean([v["physical_regret"] for r in entries for v in r["views"]])),
            "view_choice_disagreement_roots": sum(r["views"][0]["selected_action"] !=
                                                   r["views"][1]["selected_action"] for r in entries)}
    record = {"scope": "state-assisted checkpoint visual-branch observation only; no planner, no RGB-only claim",
              "roots": [1880, 1881, 1882, 1883], "source_sha256": file_hash(SOURCE),
              "protocol_sha256": file_hash(PROTOCOL), "code_sha256": file_hash(Path(__file__)),
              "frozen_model_fingerprint": before, "audit": audit.to_dict() if hasattr(audit, "to_dict") else str(audit),
              "final_layer_to_official_max_abs_error": maximum_error, "encoding_seconds_synchronized": elapsed,
              "summary": summary, "rows": rows}
    with OUTPUT.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(OUTPUT), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
