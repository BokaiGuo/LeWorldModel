"""Quantify same-state historical versus rerendered RGB encoder sensitivity."""
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model
from experiments import pc_pointmaze_heldout_amplitude_encoding_20261107_codex as audit

ROOT = audit.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_RENDER_DRIFT_DIAGNOSTIC_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_pointmaze_render_drift_diagnostic_20261107_codex.json"
FEATURES = ROOT / "results/pc_pointmaze_render_drift_diagnostic_features_20261107_codex.npz"


def metrics(old, new, goal, distances):
    old, new, goal, distances = (np.asarray(x, dtype=np.float64) for x in (old, new, goal, distances))
    if (old.shape != new.shape or old.ndim < 2 or old.shape[0] != 2
            or old.shape[1:] != goal.shape or distances.shape != (2,)
            or not all(np.isfinite(x).all() for x in (old, new, goal, distances))):
        raise ValueError("invalid two-action inputs")
    axes = tuple(range(1, old.ndim))
    old_scores, new_scores = np.square(old - goal).mean(axis=axes), np.square(new - goal).mean(axis=axes)
    old_choice, new_choice = int(np.argmin(old_scores)), int(np.argmin(new_scores))
    return {"feature_mse": np.square(old - new).mean(axis=axes).tolist(),
            "feature_max_abs": float(np.max(np.abs(old - new))),
            "old_scores": old_scores.tolist(), "new_scores": new_scores.tolist(),
            "old_choice": old_choice, "new_choice": new_choice,
            "old_physical_regret": float(distances[old_choice] - distances.min()),
            "new_physical_regret": float(distances[new_choice] - distances.min())}


def run():
    if OUTPUT.exists() or FEATURES.exists():
        raise FileExistsError("output exists")
    record = json.loads(audit.prior.source.PREFIX.with_suffix(".json").read_text())
    if record["features_sha256"] != file_hash(audit.prior.source.PREFIX.with_suffix(".npz")):
        raise ValueError("historical features drift")
    model, _, permission = load_model()
    if model_fingerprint(model) != record["frozen_fingerprint"] or permission.fingerprint != record["input_audit_hash"]:
        raise ValueError("model contract mismatch")
    rows, features = [], {}
    old_path = ROOT / "results/pc_real_camera_effect_dev_20261106_codex.npz"
    with (np.load(old_path, allow_pickle=False) as old_data,
          np.load(audit.FRAMES, allow_pickle=False) as new_data,
          np.load(audit.prior.PHYSICAL, allow_pickle=False) as physical,
          np.load(audit.prior.source.PREFIX.with_suffix(".npz"), allow_pickle=False) as saved,
          torch.inference_mode()):
        if str(new_data["physical_sha256"]) != file_hash(audit.prior.PHYSICAL):
            raise ValueError("render/physical mismatch")
        for root in range(4):
            for camera in range(2):
                key = f"r{root}_v{camera}"
                old_img = old_data["images"][2 * root, [2, 1], camera, -1]
                new_img = new_data["frames"][root, [0, 3], camera]
                old = np.stack([model.encode(rgb_tensor(image))[0].cpu().numpy() for image in old_img])
                new = np.stack([model.encode(rgb_tensor(image))[0].cpu().numpy() for image in new_img])
                expected = np.stack([saved[key + "_real_2"], saved[key + "_real_1"]])
                old_error = float(np.abs(old - expected).max())
                if old_error > 1e-5:
                    raise ValueError(f"historical encoding drift: {old_error}")
                goal = saved[key + "_goal"]
                target = physical["states"][2 * root + 1, :2]
                distances = np.linalg.norm(physical["traces"][2 * root, [0, 3], -1, :2] - target, axis=1)
                entry = metrics(old, new, goal, distances)
                difference = np.abs(old_img.astype(np.int16) - new_img.astype(np.int16))
                entry.update(root=1880 + root, camera=camera, historical_encoding_max_error=old_error,
                             pixel_mean=float(difference.mean()), pixel_max=int(difference.max()),
                             pixel_nonzero=int(np.count_nonzero(difference)), physical_distances=distances.tolist())
                rows.append(entry)
                features[key + "_old"] = old
                features[key + "_new"] = new
    if model_fingerprint(model) != record["frozen_fingerprint"]:
        raise ValueError("model changed")
    with FEATURES.open("xb") as stream:
        np.savez_compressed(stream, **features)
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "source_sha256": file_hash(old_path),
                   "render_sha256": file_hash(audit.FRAMES), "physical_sha256": file_hash(audit.prior.PHYSICAL),
                   "saved_features_sha256": file_hash(audit.prior.source.PREFIX.with_suffix(".npz")),
                   "features_sha256": file_hash(FEATURES), "model_sha256": record["frozen_fingerprint"],
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
                   "scope": "state-assisted exposed data; no renderer equivalence or closed-loop claim"},
                  stream, indent=2, allow_nan=False)
    print(json.dumps({"conditions": len(rows),
                      "choices_changed": sum(r["old_choice"] != r["new_choice"] for r in rows),
                      "max_old_feature_error": max(r["historical_encoding_max_error"] for r in rows),
                      "max_new_feature_difference": max(r["feature_max_abs"] for r in rows)}, indent=2))


if __name__ == "__main__":
    run()
