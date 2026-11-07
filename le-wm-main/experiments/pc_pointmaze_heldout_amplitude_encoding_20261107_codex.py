"""Align frozen predictions, rendered successor embeddings and physical outcomes."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_pointmaze_heldout_amplitude_physical_20261107_codex as prior

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_HELDOUT_AMPLITUDE_ENCODING_PROTOCOL_20261107_CODEX.md"
FRAMES = ROOT / "results/pc_pointmaze_heldout_amplitude_encoding_frames_v2_20261107_codex.npz"
FEATURES = ROOT / "results/pc_pointmaze_heldout_amplitude_encoding_features_20261107_codex.npz"
RESULT = ROOT / "results/pc_pointmaze_heldout_amplitude_encoding_result_20261107_codex.json"


def comparisons(predicted, actual, goal, distances):
    predicted, actual = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
    goal, distances = np.asarray(goal, dtype=float), np.asarray(distances, dtype=float)
    if (predicted.shape != actual.shape or predicted.ndim < 2 or predicted.shape[0] != 4
            or predicted.shape[1:] != goal.shape or distances.shape != (4,)
            or not all(np.isfinite(x).all() for x in (predicted, actual, goal, distances))):
        raise ValueError("inconsistent representations or distances")
    axes = tuple(range(1, predicted.ndim))
    pcost = np.square(predicted - goal).mean(axis=axes)
    acost = np.square(actual - goal).mean(axis=axes)
    selected = {"predicted": int(np.argmin(pcost)), "encoded": int(np.argmin(acost)),
                "physical": int(np.argmin(distances))}
    pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            def relation(left, right):
                a, b = left[i] - left[j], right[i] - right[j]
                return None if abs(a) <= 1e-12 or abs(b) <= 1e-12 else bool(a * b > 0)
            pairs.append({"i": i, "j": j, "prediction_encoding": relation(pcost, acost),
                          "encoding_physical": relation(acost, distances),
                          "prediction_physical": relation(pcost, distances)})
    return {"predicted_costs": pcost.tolist(), "encoded_costs": acost.tolist(),
            "distances": distances.tolist(), "selected": selected,
            "regret": {name: float(distances[index] - distances.min()) for name, index in selected.items()},
            "feature_mse": np.square(predicted - actual).mean(axis=axes).tolist(), "pairs": pairs}


def render():
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    if FRAMES.exists():
        raise FileExistsError(FRAMES)
    physical = np.load(prior.PHYSICAL, allow_pickle=False)
    selection = json.loads(prior.SELECT.read_text())
    if (str(physical["selection_sha256"]) != file_hash(prior.SELECT)
            or selection["runner_sha256"] != file_hash(prior.__file__)):
        raise ValueError("physical source drift")
    with tempfile.TemporaryDirectory(prefix="pc_amp_render_") as tmp:
        directory = Path(tmp)
        request, xml, result = directory / "request.npz", directory / "scene.xml", directory / "frames.npz"
        np.savez_compressed(request, states=physical["states"][::2])
        xml.write_text(official_xml())
        worker = Path(__file__).with_name("pc_pointmaze_heldout_amplitude_render_backend_20261107_codex.py")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(LEGACY / "bin/python"), str(worker), str(request), str(xml), str(result)],
                       env=env, check=True, timeout=180)
        with np.load(result, allow_pickle=False) as rendered, np.load(
                ROOT / "results/pc_real_camera_effect_dev_20261106_codex.npz", allow_pickle=False) as previous:
            traces, frames = rendered["traces"], rendered["frames"]
            if traces.shape != (4, 4, 2, 6, 4) or frames.shape != (4, 4, 2, 224, 224, 3):
                raise ValueError("render shape mismatch")
            np.testing.assert_allclose(traces, np.broadcast_to(physical["traces"][::2, :4, None],
                                                              traces.shape), atol=1e-9, rtol=0)
            ref = previous["images"][::2, :, :, -1]
            pixel_diffs = np.abs(frames[:, [3, 0]].astype(np.int16) - ref[:, [1, 2]].astype(np.int16))
            with FRAMES.open("xb") as stream:
                np.savez_compressed(stream, frames=frames, traces=traces,
                                    old_amplitude_pixel_max=np.asarray(pixel_diffs.max()),
                                    old_amplitude_pixel_mean=np.asarray(pixel_diffs.mean()),
                                    physical_sha256=np.asarray(file_hash(prior.PHYSICAL)),
                                    worker_sha256=np.asarray(file_hash(worker)))
    physical.close()
    print("Rendered physical successors; trajectories match sealed replay.")


def encode():
    import torch
    from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
    from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
    from experiments.pc_state_track_smoke_20261106_codex import load_model

    if FEATURES.exists():
        raise FileExistsError(FEATURES)
    record = json.loads(prior.source.PREFIX.with_suffix(".json").read_text())
    if record["features_sha256"] != file_hash(prior.source.PREFIX.with_suffix(".npz")):
        raise ValueError("source feature drift")
    model, _, audit = load_model()
    if model_fingerprint(model) != record["frozen_fingerprint"] or audit.fingerprint != record["input_audit_hash"]:
        raise ValueError("model contract mismatch")
    arrays, max_old_error = {}, 0.
    with np.load(FRAMES, allow_pickle=False) as images, np.load(
            prior.source.PREFIX.with_suffix(".npz"), allow_pickle=False) as old, torch.inference_mode():
        if str(images["physical_sha256"]) != file_hash(prior.PHYSICAL):
            raise ValueError("render binding mismatch")
        for root in range(4):
            for camera in range(2):
                key = f"r{root}_v{camera}"
                encoded = np.stack([model.encode(rgb_tensor(images["frames"][root, j, camera]))[0].cpu().numpy()
                                    for j in range(4)])
                for idx, oldidx in ((0, 2), (3, 1)):
                    max_old_error = max(max_old_error, float(np.max(np.abs(encoded[idx] - old[key + f"_real_{oldidx}"]))))
                arrays[key] = encoded
    if max_old_error > 1e-5 or model_fingerprint(model) != record["frozen_fingerprint"]:
        raise ValueError(f"old encoded reference or model mismatch: {max_old_error}")
    arrays["frames_sha256"] = np.asarray(file_hash(FRAMES))
    arrays["old_features_sha256"] = np.asarray(file_hash(prior.source.PREFIX.with_suffix(".npz")))
    arrays["model_sha256"] = np.asarray(record["frozen_fingerprint"])
    arrays["max_old_encoding_error"] = np.asarray(max_old_error)
    with FEATURES.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(f"Encoded 32 actual successors; max old embedding error {max_old_error}.")


def analyze():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    model_record = json.loads(prior.source.PREFIX.with_suffix(".json").read_text())
    physical = json.loads(prior.RESULT.read_text())
    rows = []
    with np.load(FEATURES, allow_pickle=False) as actual, np.load(
            prior.source.PREFIX.with_suffix(".npz"), allow_pickle=False) as predicted:
        if (str(actual["frames_sha256"]) != file_hash(FRAMES)
                or str(actual["old_features_sha256"]) != file_hash(prior.source.PREFIX.with_suffix(".npz"))
                or physical["physical_sha256"] != file_hash(prior.PHYSICAL)):
            raise ValueError("feature or physical source drift")
        for row in model_record["rows"]:
            root, camera = row["root"], row["camera"]
            key = f"r{root - 1880}_v{camera}"
            p = np.stack([predicted[key + f"_pred_{x}"] for x in prior.VALUES[:4]])
            d = next(x["distances"] for x in physical["rows"] if x["root"] == root and x["camera"] == camera)
            output = comparisons(p, actual[key], predicted[key + "_goal"], d)
            np.testing.assert_allclose(output["predicted_costs"], row["heldout_model_costs"], atol=1e-6, rtol=1e-5)
            output.update(root=root, camera=camera)
            rows.append(output)
    if len(rows) != 8 or len({(r["root"], r["camera"]) for r in rows}) != 8:
        raise ValueError("incomplete panel")
    means = {name: float(np.mean([np.mean([r["regret"][name] for r in rows if r["root"] == root])
                                 for root in range(1880, 1884)]))
             for name in ("predicted", "encoded", "physical")}
    pair_counts = {}
    for name in ("prediction_encoding", "encoding_physical", "prediction_physical"):
        observations = [p[name] for r in rows for p in r["pairs"] if p[name] is not None]
        pair_counts[name] = {"agree": sum(observations), "non_ties": len(observations)}
    with RESULT.open("x") as stream:
        json.dump({"rows": rows, "mean_regret": means, "pair_counts": pair_counts,
                   "features_sha256": file_hash(FEATURES), "physical_sha256": file_hash(prior.PHYSICAL),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "scope": "state-assisted exposed roots; actual encoding is evaluation only"},
                  stream, indent=2, allow_nan=False)
    print(json.dumps({"mean_regret": means, "pair_counts": pair_counts}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("render", "encode", "analyze"):
        raise SystemExit("expected render|encode|analyze")
    {"render": render, "encode": encode, "analyze": analyze}[sys.argv[1]]()
