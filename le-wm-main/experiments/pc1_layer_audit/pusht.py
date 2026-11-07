"""Frozen full-token/CLS audit with root-separated, offline Procrustes probes."""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from eval import img_transform
from experiments.checkpoint_compat import _resolve_local, load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc1_layer_audit.metrics import (
    effects, fit_orthogonal, ranking_record, root_summary, squared_distances, unit_rows,
)
from experiments.research_runtime.artifacts import atomic_write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/pc1_full_token_orthogonal_pusht_20261107"
PROTOCOL = ROOT.parent / "docs/PC1_FULL_TOKEN_ORTHOGONAL_PUSHT_PROTOCOL_20261107_CODEX.md"
PREFIXES = ("pc_geometric_contact_equivalence_dev_20261106_codex",
            "pc_osfal_pusht_heldout_geometry_20261107_codex")
ACTIONS, COLORS, SIDES = ("zero", "plus", "minus"), ("white", "gray"), (-1, 1)


def collect_inputs(transform):
    pixels, truth, predictions, predicted_latents, roots, source_hashes, saved_actual = [], [], [], [], [], {}, []
    for group, prefix in enumerate(PREFIXES):
        paths = {name: ROOT / f"results/{prefix}_{name}.{extension}" for name, extension in
                 (("physical", "json"), ("images", "npz"), ("model", "json"), ("features", "npz"))}
        source_hashes.update({str(p): file_hash(p) for p in paths.values()})
        physical = json.loads(paths["physical"].read_text())
        prior = json.loads(paths["model"].read_text())
        if physical["images_sha256"] != file_hash(paths["images"]):
            raise ValueError("image hash changed")
        if prior["physical_sha256"] != file_hash(paths["physical"]) or prior["features_sha256"] != file_hash(paths["features"]):
            raise ValueError("prediction source changed")
        with np.load(paths["images"], allow_pickle=False) as images, np.load(paths["features"], allow_pickle=False) as features:
            for row, old in zip(physical["rows"], prior["rows"], strict=True):
                if not row["valid"] or not old["valid"] or row["root"] != old["root"]:
                    raise ValueError("invalid or reordered root")
                costs, model_scores, stored_actual, latent_rows = [], [], [], []
                for color in COLORS:
                    for side in SIDES:
                        c = next(c for c in row["conditions"] if c["side"] == side and c["appearance"] == color)
                        m = next(c for c in old["conditions"] if c["side"] == side and c["appearance"] == color)
                        base = f"r{row['root']}_{color}_{side}"
                        keys = [f"{base}_zero_initial"] + [f"{base}_{a}_final" for a in ACTIONS] + [f"r{row['root']}_{color}_goal"]
                        pixels.extend(transform(images[k]) for k in keys)
                        values = []
                        for action in ACTIONS:
                            entry = c["actions"][action]
                            value = features[f"{base}_{action}"]
                            if value.shape != (4, 192):
                                raise ValueError("unexpected saved feature shape")
                            np.testing.assert_allclose(np.square(value[1] - value[3]).sum(),
                                                       m["actions"][action]["predicted_cost"], atol=1e-3, rtol=1e-4)
                            values.append(value)
                        costs.append([c["actions"][a]["full_state_distance"] for a in ACTIONS])
                        model_scores.append([m["actions"][a]["predicted_cost"] for a in ACTIONS])
                        stored_actual.append([m["actions"][a]["real_encoded_cost"] for a in ACTIONS])
                        latent_rows.append(values)
                roots.append({"id": f"{group}:{row['root']}", "group": group, "task": row["task"],
                              "native_final_successes": sum(
                                  c["actions"][a]["trace"][-1]["success"] for c in row["conditions"] for a in ACTIONS)})
                truth.append(np.asarray(costs).reshape(2, 2, 3))
                predictions.append(np.asarray(model_scores).reshape(2, 2, 3))
                saved_actual.append(np.asarray(stored_actual).reshape(2, 2, 3))
                predicted_latents.append(np.asarray(latent_rows).reshape(2, 2, 3, 4, 192))
    geometry = [{(tuple(r["task"]["position"]), r["task"]["angle"]) for r in roots if r["group"] == i} for i in (0, 1)]
    if len(roots) != 32 or geometry[0] & geometry[1]:
        raise ValueError("unexpected panel or geometry overlap")
    return pixels, np.asarray(truth), np.asarray(predictions), np.asarray(predicted_latents), roots, source_hashes, np.asarray(saved_actual)


def audit_representation(x, truth, roots):
    """x: root, appearance, side, frame, token, channel; no predictor replacement."""
    if x.shape[:4] != (32, 2, 2, 5):
        raise ValueError("feature layout mismatch")
    calibration = [r["id"] for r in roots[:16]]
    evaluation = [r["id"] for r in roots[16:]]
    fit_data = []
    for view in (0, 1):
        initial = x[:16, view, :, 0].reshape(-1, x.shape[-1])
        goals = x[:16, view, 0, 4].reshape(-1, x.shape[-1])
        fit_data.append(np.concatenate((initial, goals)))
    mapping = fit_orthogonal(fit_data[1], fit_data[0], calibration, evaluation)
    shifted = x[:, 1] - mapping.source_mean + mapping.target_mean
    aligned = mapping.apply(x[:, 1])
    own_cost = np.square(x[:, :, :, 1:4] - x[:, :, :, 4:5]).sum(axis=(-2, -1))
    aligned_own_cost = np.square(aligned[:, :, 1:4] - aligned[:, :, 4:5]).sum(axis=(-2, -1))
    np.testing.assert_allclose(aligned_own_cost, own_cost[:, 1], atol=1e-6, rtol=1e-8)
    cross_raw = np.square(x[:, 1, :, 1:4] - x[:, 0, :, 4:5]).sum(axis=(-2, -1))
    cross_aligned = np.square(aligned[:, :, 1:4] - x[:, 0, :, 4:5]).sum(axis=(-2, -1))
    flattened = x.reshape(32, 2, 2, 5, -1)
    normalized = unit_rows(flattened.reshape(-1, flattened.shape[-1])).reshape(flattened.shape)
    features = np.concatenate((normalized[:, :, :, 0], normalized[:, :, :, 4]), axis=-1)
    gallery = features[:16, 0].reshape(32, -1)
    query = features[16:, 1].reshape(32, -1)
    neighbors = np.argmin(squared_distances(query, gallery), axis=1)
    descriptors = effects(truth[:16, 0].reshape(32, 3))
    mean_descriptor = descriptors.mean(0)
    rows = []
    for root in range(32):
        d = {"root_id": roots[root]["id"],
             "normalized_background_distance": float(np.mean(np.sum((normalized[root, 0] - normalized[root, 1])**2, -1))),
             "normalized_initial_side_distance": float(np.mean(np.sum((normalized[root, :, 0, 0] - normalized[root, :, 1, 0])**2, -1))),
             "raw_alignment_mse": float(np.mean((x[root, 1] - x[root, 0])**2)),
             "mean_shift_alignment_mse": float(np.mean((shifted[root] - x[root, 0])**2)),
             "orthogonal_alignment_mse": float(np.mean((aligned[root] - x[root, 0])**2)),
             "real_successor_scoring": [ranking_record(own_cost[root, v, s], truth[root, v, s])
                                        for v in (0, 1) for s in (0, 1)],
             "cross_source_scoring_raw": [ranking_record(cross_raw[root, s], truth[root, 1, s]) for s in (0, 1)],
             "cross_source_scoring_aligned": [ranking_record(cross_aligned[root, s], truth[root, 1, s]) for s in (0, 1)]}
        if root >= 16:
            d["offline_initial_effect_retrieval"] = []
            d["offline_mean_effect_baseline"] = []
            for side in range(2):
                neighbor = int(neighbors[(root - 16)*2 + side])
                predicted = np.r_[0., -descriptors[neighbor]]
                d["offline_initial_effect_retrieval"].append(
                    {"calibration_root": calibration[neighbor // 2], "calibration_side": SIDES[neighbor % 2],
                     **ranking_record(predicted, truth[root, 1, side])})
                d["offline_mean_effect_baseline"].append(ranking_record(np.r_[0., -mean_descriptor], truth[root, 1, side]))
        rows.append(d)
    evaluated = rows[16:]
    summary = {}
    for name in ("normalized_background_distance", "normalized_initial_side_distance",
                 "raw_alignment_mse", "mean_shift_alignment_mse", "orthogonal_alignment_mse"):
        summary[name] = root_summary([r[name] for r in evaluated])
    for name in ("real_successor_scoring", "cross_source_scoring_raw", "cross_source_scoring_aligned",
                 "offline_initial_effect_retrieval", "offline_mean_effect_baseline"):
        summary[name + "_regret"] = root_summary([np.mean([c["regret"] for c in r[name]]) for r in evaluated])
    # Four disjoint calibration folds describe layer-selection variability downstream.
    summary["calibration_fold_regret"] = [float(np.mean([
        c["regret"] for r in rows[fold*4:(fold+1)*4] for c in r["real_successor_scoring"]])) for fold in range(4)]
    summary["within_source_distance_max_error"] = float(np.max(np.abs(aligned_own_cost-own_cost[:, 1])))
    return {"rows": rows, "summary": summary,
            "mapping": {"scope": "offline only, shared channel rotation for full tokens",
                        "fit_rows": mapping.fit_rows, "channels": x.shape[-1], "source_rank": mapping.source_rank,
                        "cross_rank": mapping.cross_rank, "calibration_roots": calibration,
                        "evaluation_roots": evaluation,
                        "orthogonality_max_error": float(np.max(np.abs(mapping.rotation.T @ mapping.rotation-np.eye(x.shape[-1]))))}}, mapping


def main():
    output = OUT / "result.json"
    if OUT.exists():
        raise FileExistsError(OUT)
    print("Post-hoc 32-geometry RGB audit: full tokens and CLS; no planner or camera claim.", flush=True)
    weights, config = _resolve_local("pusht/lewm")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    before = model_fingerprint(model)
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))
    pixels, truth, predictions, saved_features, roots, sources, saved_actual = collect_inputs(transform)
    captured, blocks, official = {}, {i: [] for i in range(12)}, []
    def hook(i):
        def store(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if value.shape[1:] != (257, 192):
                raise ValueError("unexpected complete hidden state")
            captured[i] = value.detach().cpu().numpy()
        return store
    handles = [block.register_forward_hook(hook(i)) for i, block in enumerate(model.encoder.layers)]
    if len(handles) != 12:
        raise ValueError("unexpected encoder layers")
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            for start in range(0, len(pixels), 32):
                values = model.encode({"pixels": torch.stack(pixels[start:start+32]).cuda()[:, None]})["emb"][:, 0]
                official.append(values.cpu().numpy())
                for i in range(12):
                    blocks[i].append(captured[i])
                captured.clear()
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    encoding_seconds = time.perf_counter() - started
    official = np.concatenate(official).reshape(32, 2, 2, 5, 1, 192)
    actual = np.square(official[:, :, :, 1:4, 0] - official[:, :, :, 4:5, 0]).sum(-1)
    np.testing.assert_allclose(actual, saved_actual, atol=1e-3, rtol=1e-4)
    np.testing.assert_allclose(official[:, :, :, 1:4, 0], saved_features[:, :, :, :, 2],
                               atol=1e-4, rtol=1e-4)
    result = {"scope": "post-hoc RGB observation/appearance audit; no online calibration, no gate promotion",
              "roots": roots, "encoding_seconds_synchronized": encoding_seconds,
              "official_cost_max_error": float(np.max(np.abs(actual-saved_actual))),
              "model_fingerprint": before, "representations": {}, "predictor_transfer": []}
    matrices = {}
    for i in range(13):
        token_values = np.concatenate(blocks.pop(i)).reshape(32, 2, 2, 5, 257, 192) if i < 12 else official
        modes = ("full", "cls") if i < 12 else ("official",)
        for mode in modes:
            name = f"block_{i+1:02d}_{mode}" if i < 12 else "official"
            values = token_values if mode != "cls" else token_values[..., :1, :]
            metrics, mapping = audit_representation(values.astype(np.float64), truth, roots)
            result["representations"][name] = metrics
            matrices[name+"_rotation"] = mapping.rotation
            matrices[name+"_source_mean"] = mapping.source_mean
            matrices[name+"_target_mean"] = mapping.target_mean
            if mode != "full":
                matrices[name+"_features"] = values
            print(f"{name}: heldout offline retrieval regret={metrics['summary']['offline_initial_effect_retrieval_regret']['mean']:.6f}", flush=True)
    for r, root in enumerate(roots):
        result["predictor_transfer"].append({"root_id": root["id"],
            "conditions": [{"appearance": COLORS[v], "side": SIDES[s],
                            "predicted": ranking_record(predictions[r, v, s], truth[r, v, s]),
                            "real_encoded": ranking_record(actual[r, v, s], truth[r, v, s]),
                            "feature_prediction_mse": float(np.mean((saved_features[r, v, s, :, 1] - saved_features[r, v, s, :, 2])**2))}
                           for v in (0, 1) for s in (0, 1)]})
    if before != model_fingerprint(model):
        raise ValueError("model changed")
    matrices.update(native_costs=truth, predicted_costs=predictions, real_encoded_costs=actual)
    OUT.mkdir(parents=True)
    array_path = OUT / "diagnostics.npz"
    with array_path.open("xb") as stream:
        np.savez_compressed(stream, **matrices)
    sources.update({str(p): file_hash(p) for p in (weights, config, PROTOCOL, Path(__file__), Path(__file__).with_name("metrics.py"))})
    result.update(sources=sources, arrays_sha256=file_hash(array_path))
    json.dumps(result, allow_nan=False)
    atomic_write_json(output, result)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
