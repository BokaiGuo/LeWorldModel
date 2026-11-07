"""Sealed 32-root state-assisted camera/layer/transfer diagnostic."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch

from experiments.pc_input_permissions import STATE, file_hash
from experiments import pc_input_permissions_v2_20261107_codex as permissions
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc1_layer_audit.metrics import (
    effects, fit_orthogonal, ranking_record, root_summary, squared_distances, unit_rows,
)
from experiments.research_runtime.artifacts import atomic_write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/pc1_camera_layer_transfer_20261107"
DATA = ROOT / "data/point_maze_tail1800_1899"
PROTOCOL = ROOT.parent / "docs/PC1_CAMERA_LAYER_TRANSFER_PROTOCOL_20261107_CODEX.md"
WORKER = Path(__file__).with_name("camera_worker.py")
ROOTS = tuple(range(1832, 1864))
HORIZONS = (5, 10, 25)
ACTIONS = np.array([[0., 0.], [.1, 0.], [-.1, 0.]], dtype=np.float32)


def source_hashes():
    paths = (Path(__file__), WORKER, Path(__file__).with_name("metrics.py"), PROTOCOL)
    return {str(p): file_hash(p) for p in paths}


def validate_sources(record):
    for path, digest in record["sources"].items():
        if file_hash(path) != digest:
            raise ValueError(f"source drift: {path}")


def read_stage(name):
    record = json.loads((OUT / f"{name}.json").read_text())
    validate_sources(record)
    if file_hash(OUT / f"{name}.npz") != record["arrays_sha256"]:
        raise ValueError(f"array drift: {name}")
    return record


def save_arrays(name, **arrays):
    path = OUT / f"{name}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return file_hash(path)


def render(mode, states):
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    with tempfile.TemporaryDirectory(prefix="pc1_camera_") as temp:
        folder = Path(temp)
        req, xml, res = folder / "request.npz", folder / "scene.xml", folder / "response.npz"
        with req.open("xb") as stream:
            np.savez_compressed(stream, states=states)
        text = official_xml()
        xml.write_text(text)
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        started = time.perf_counter()
        subprocess.run([str(LEGACY / "bin/python"), str(WORKER), mode, str(req), str(xml), str(res)],
                       env=env, check=True, timeout=600)
        elapsed = time.perf_counter() - started
        with np.load(res, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
    return arrays, elapsed, hashlib.sha256(text.encode()).hexdigest()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "prepare.npz").exists():
        raise FileExistsError("prepare output already exists")
    source = DATA / "states.pth"
    archive = torch.load(source, weights_only=True, map_location="cpu")
    states = np.stack([archive[root, [0, 5]].numpy() for root in ROOTS])
    sources = {str(source): file_hash(source), **source_hashes()}
    for name in ("maze_model.py", "dynamic_mjc.py"):
        p = ROOT / "external/jepa-wms/evals/simu_env_planning/envs/pointmaze_env" / name
        sources[str(p)] = file_hash(p)
    result, seconds, xml_hash = render("frames", states.reshape(-1, 4))
    frames = result["images"][:, 0, :, 0].reshape(32, 2, 2, 224, 224, 3)
    for i, root in enumerate(ROOTS):
        p = DATA / "obses" / f"episode_{root:03d}.pth"
        reference = torch.load(p, weights_only=True, map_location="cpu").numpy()[[0, 5]]
        np.testing.assert_array_equal(frames[i, :, 0], reference)
        sources[str(p)] = file_hash(p)
    record = dict(track=STATE, scope="development, archive provenance independence unknown",
                  roots=ROOTS, calibration_roots=ROOTS[:16], evaluation_roots=ROOTS[16:],
                  cameras=[[3., 3.], [3.5, 3.]], sources=sources,
                  xml_sha256=xml_hash, archive_pixel_max_error=0,
                  render_seconds=seconds, camera_default=result["camera_default"].tolist(),
                  arrays_sha256=save_arrays("prepare", states=states, frames=frames))
    atomic_write_json(OUT / "prepare.json", record)
    print("prepare complete: all 64 canonical images exactly match archive.", flush=True)


def load_model():
    from experiments.pc_state_track_smoke_20261106_codex import load_model as load
    model, dataset, audit = load()
    paths = dict(audit.dependencies)
    paths.update({str(p.resolve()): file_hash(p) for p in (Path(permissions.__file__), Path(__file__))})
    audit = replace(audit, dependencies=tuple(paths.items())).verify()
    return model, dataset, audit


def raw_rgb(images):
    return torch.from_numpy(np.array(images, copy=True)).permute(0, 3, 1, 2)[:, None].cuda()


def packet_actions(preprocessor):
    raw = torch.from_numpy(np.broadcast_to(ACTIONS[:, None], (3, 25, 2)).copy())
    packet = preprocessor.normalize_actions(raw).reshape(3, 5, 10).transpose(0, 1).contiguous().cuda()
    if packet.shape != (5, 3, 10):
        raise ValueError("packet layout")
    return raw.numpy(), packet


def predict():
    prepared = read_stage("prepare")
    if (OUT / "predict.npz").exists():
        raise FileExistsError("predict output already exists")
    model, _, audit = load_model()
    fingerprint = model_fingerprint(model)
    boundary = permissions.StateAssistedPointMazev2(model, audit)
    raw, packets = packet_actions(model.preprocessor)
    arrays = {"raw_actions": raw, "normalized_packets": packets.cpu().numpy()}
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    torch.cuda.synchronize()
    start = time.perf_counter()
    with np.load(OUT / "prepare.npz", allow_pickle=False) as data, torch.inference_mode():
        for r, root in enumerate(ROOTS):
            for v in range(2):
                inputs = {"visual": raw_rgb(np.repeat(data["frames"][r, 0, v][None], 3, axis=0)),
                          "proprio": torch.tensor(data["states"][r, 0], dtype=torch.float32, device="cuda")[None, None].expand(3, 1, 4).contiguous()}
                goal = model.encode({"visual": raw_rgb(data["frames"][r, 1, v][None]),
                                     "proprio": torch.tensor(data["states"][r, 1], dtype=torch.float32, device="cuda")[None, None]})
                pred = boundary.predict(inputs, packets)
                objective = ReprTargetDistMPCObjective({}, goal, alpha=.1)
                cost = objective(pred, packets, keepdims=True)
                components = {}
                for key in ("visual", "proprio"):
                    value = pred[key]
                    if value.shape[:2] != (6, 3):
                        raise ValueError("prediction clock/shape")
                    components[key] = ((value - goal[key])**2).flatten(2).mean(-1)
                    arrays[f"r{root}_v{v}_pred_{key}"] = value.cpu().numpy()
                    arrays[f"r{root}_v{v}_goal_{key}"] = goal[key].cpu().numpy()
                torch.testing.assert_close(cost, components["visual"] + .1*components["proprio"],
                                           atol=1e-6, rtol=1e-5)
                arrays[f"r{root}_v{v}_cost"] = cost.cpu().numpy()
            print(f"prediction sealed root={root}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if fingerprint != model_fingerprint(model):
        raise ValueError("model changed")
    sources = {**source_hashes(), str(OUT / "prepare.json"): file_hash(OUT / "prepare.json"), **dict(audit.dependencies)}
    record = dict(track=STATE, sources=sources, model_fingerprint=fingerprint,
                  input_audit=asdict(audit), synchronized_prediction_seconds=elapsed,
                  arrays_sha256=save_arrays("predict", **arrays),
                  receipts=permissions.call_manifest(audit, algorithm_version="pc1_camera_diagnostic",
                      protocol_path=PROTOCOL, task_manifest_path=OUT / "prepare.json", receipts=boundary.receipts))
    atomic_write_json(OUT / "predict.json", record)
    print("All predictions recorded before physical replay.", flush=True)


def replay():
    read_stage("prepare")
    read_stage("predict")
    if (OUT / "replay.npz").exists():
        raise FileExistsError("replay output already exists")
    with np.load(OUT / "prepare.npz", allow_pickle=False) as data:
        result, seconds, xml_hash = render("rollouts", data["states"][:, 0])
        traces, times = result["traces"], result["times"]
        if traces.shape != (32, 3, 2, 26, 4) or result["images"].shape != (32, 3, 2, 4, 224, 224, 3):
            raise ValueError("replay shape mismatch")
        np.testing.assert_allclose(traces[:, :, 0], traces[:, :, 1], atol=1e-10, rtol=0)
        np.testing.assert_allclose(times, np.broadcast_to(np.arange(26)*.05, times.shape), atol=1e-10, rtol=0)
        np.testing.assert_array_equal(result["images"][:, :, :, 0], np.broadcast_to(data["frames"][:, None, 0], result["images"][:, :, :, 0].shape))
        truth = np.linalg.norm(traces[:, :, 0][:, :, HORIZONS, :2] - data["states"][:, None, None, 1, :2], axis=-1).transpose(0, 2, 1)
    if xml_hash != read_stage("prepare")["xml_sha256"]:
        raise ValueError("renderer source drift")
    result["physical_costs"] = truth
    sources = {**source_hashes(), **{str(OUT / f"{n}.json"): file_hash(OUT / f"{n}.json") for n in ("prepare", "predict")}}
    atomic_write_json(OUT / "replay.json", dict(track=STATE, sources=sources, render_seconds=seconds,
        physical_camera_max_error=float(np.max(np.abs(traces[:, :, 0]-traces[:, :, 1]))),
        arrays_sha256=save_arrays("replay", **result), roots=32, trajectories=192,
        metric="endpoint Euclidean xy distance, not benchmark success"))
    print("Physical replay complete: same actions and state trajectories across cameras.", flush=True)


def representation_audit(x, truth):
    """[root,view,initial + horizon-major 3 actions + goal,token,channel]."""
    if x.shape[:3] != (32, 2, 11) or truth.shape != (32, 3, 3):
        raise ValueError("representation layout mismatch")
    fit = [x[:16, v][:, [0, 10]].reshape(-1, x.shape[-1]) for v in (0, 1)]
    mapping = fit_orthogonal(fit[1], fit[0], ROOTS[:16], ROOTS[16:])
    aligned = mapping.apply(x[:, 1])
    shifted = x[:, 1] - mapping.source_mean + mapping.target_mean
    cost = ((x[:, :, 1:10] - x[:, :, 10:11])**2).mean((-2, -1)).reshape(32, 2, 3, 3)
    aligned_cost = ((aligned[:, 1:10] - aligned[:, 10:11])**2).mean((-2, -1)).reshape(32, 3, 3)
    np.testing.assert_allclose(aligned_cost, cost[:, 1], atol=1e-7, rtol=1e-8)
    cross = ((x[:, 1, 1:10] - x[:, 0, 10:11])**2).mean((-2, -1)).reshape(32, 3, 3)
    cross_map = ((aligned[:, 1:10] - x[:, 0, 10:11])**2).mean((-2, -1)).reshape(32, 3, 3)
    normalized = unit_rows(x.reshape(32*2*11, -1)).reshape(32, 2, 11, -1)
    initial_goal = normalized[:, :, [0, 10]].reshape(32, 2, -1)
    neighbors = np.argmin(squared_distances(initial_goal[16:, 1], initial_goal[:16, 0]), axis=1)
    descriptor = effects(truth[:16])
    shuffled = descriptor.copy()
    rng = np.random.default_rng(71107)
    for r in range(16):
        for h in range(3):
            scores = np.r_[0., -descriptor[r, h]]
            shuffled[r, h] = effects(scores[rng.permutation(3)])
    rows = []
    for r, root in enumerate(ROOTS):
        row = dict(root=root, view_distance=float(np.mean(np.sum((normalized[r, 0]-normalized[r, 1])**2, -1))),
                   raw_alignment_mse=float(np.mean((x[r, 1]-x[r, 0])**2)),
                   shift_alignment_mse=float(np.mean((shifted[r]-x[r, 0])**2)),
                   orthogonal_alignment_mse=float(np.mean((aligned[r]-x[r, 0])**2)), horizons=[])
        for h, steps in enumerate(HORIZONS):
            end = normalized[r, :, 1+3*h:4+3*h]
            item = dict(environment_steps=steps,
                        action_distance=float(np.mean([np.sum((end[:, i]-end[:, j])**2, -1).mean()
                                                       for i, j in ((0, 1), (0, 2), (1, 2))])),
                        real_encoded=[ranking_record(cost[r, v, h], truth[r, h]) for v in (0, 1)],
                        cross_raw=ranking_record(cross[r, h], truth[r, h]),
                        cross_aligned=ranking_record(cross_map[r, h], truth[r, h]))
            if r >= 16:
                n = neighbors[r-16]
                item.update(offline_retrieval=ranking_record(np.r_[0., -descriptor[n, h]], truth[r, h]),
                            offline_mean=ranking_record(np.r_[0., -descriptor[:, h].mean(0)], truth[r, h]),
                            offline_shuffled=ranking_record(np.r_[0., -shuffled[n, h]], truth[r, h]),
                            neighbor_root=ROOTS[n])
            row["horizons"].append(item)
        rows.append(row)
    summary = {k: root_summary([row[k] for row in rows[16:]]) for k in
               ("view_distance", "raw_alignment_mse", "shift_alignment_mse", "orthogonal_alignment_mse")}
    summary["horizons"] = []
    for h, steps in enumerate(HORIZONS):
        item = {"environment_steps": steps}
        for key in ("cross_raw", "cross_aligned", "offline_retrieval", "offline_mean", "offline_shuffled"):
            item[key+"_regret"] = root_summary([row["horizons"][h][key]["regret"] for row in rows[16:]])
        item["real_encoded_regret"] = root_summary([np.mean([c["regret"] for c in row["horizons"][h]["real_encoded"]]) for row in rows[16:]])
        item["view_disagreement_roots"] = sum(row["horizons"][h]["real_encoded"][0]["choice"] != row["horizons"][h]["real_encoded"][1]["choice"] for row in rows[16:])
        item["calibration_fold_regret"] = [float(np.mean([c["regret"] for row in rows[f*4:(f+1)*4]
                                            for c in row["horizons"][h]["real_encoded"]])) for f in range(4)]
        summary["horizons"].append(item)
    return dict(rows=rows, summary=summary, mapping=dict(
        fit_rows=mapping.fit_rows, source_rank=mapping.source_rank, cross_rank=mapping.cross_rank,
        channels=x.shape[-1], fit_frames=["initial", "goal"], calibration_roots=ROOTS[:16],
        evaluation_roots=ROOTS[16:], full_token_mapping="shared channel rotation, not arbitrary flattened rotation",
        orthogonality_max_error=float(np.max(np.abs(mapping.rotation.T @ mapping.rotation-np.eye(x.shape[-1])))),
        within_source_score_max_error=float(np.max(np.abs(aligned_cost-cost[:, 1]))))), mapping


def analyze():
    prepared, predicted, physical = [read_stage(n) for n in ("prepare", "predict", "replay")]
    if (OUT / "analyze.npz").exists():
        raise FileExistsError("analyze output already exists")
    model, _, _ = load_model()
    fingerprint = model_fingerprint(model)
    if fingerprint != predicted["model_fingerprint"]:
        raise ValueError("different model")
    backbone = model.model.encoder.base_model.model
    buffers = {i: [] for i in range(12)}
    captured = {}
    def hook(i):
        def capture(_m, _in, value):
            if not isinstance(value, torch.Tensor) or value.shape[1:] != (257, 384):
                raise ValueError("full hidden state shape")
            captured[i] = value.detach()
        return capture
    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(backbone.encoder.layer)]
    if len(handles) != 12:
        raise ValueError("layer count")
    official, actual_prop = [], []
    max_error = 0.
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        with np.load(OUT / "prepare.npz", allow_pickle=False) as prep, np.load(OUT / "replay.npz", allow_pickle=False) as replay, torch.inference_mode():
            truth = replay["physical_costs"]
            for r, root in enumerate(ROOTS):
                images, states = [], []
                for v in range(2):
                    images.extend([prep["frames"][r, 0, v]] +
                                  [replay["images"][r, a, v, k] for k in (1, 2, 3) for a in range(3)] +
                                  [prep["frames"][r, 1, v]])
                    states.extend([prep["states"][r, 0]] +
                                  [replay["traces"][r, a, v, t] for t in HORIZONS for a in range(3)] +
                                  [prep["states"][r, 1]])
                encoded = model.encode({"visual": raw_rgb(np.stack(images)),
                    "proprio": torch.tensor(np.stack(states), dtype=torch.float32, device="cuda")[:, None]})
                visual = encoded["visual"].reshape(22, 256, 384)
                final = backbone.layernorm(captured[11][:, 1:])
                if model.normalize_reps:
                    final = torch.nn.functional.layer_norm(final, (384,))
                torch.testing.assert_close(visual, final, atol=1e-5, rtol=1e-5)
                max_error = max(max_error, float((visual-final).abs().max()))
                official.append(visual.cpu().numpy().reshape(2, 11, 256, 384))
                actual_prop.append(encoded["proprio"].cpu().numpy().reshape(2, 11, -1))
                for i in range(12):
                    buffers[i].append(captured[i].cpu().numpy().reshape(2, 11, 257, 384))
                captured.clear()
                print(f"true successor representations root={root}", flush=True)
        torch.cuda.synchronize()
    finally:
        for h in handles:
            h.remove()
    encoding_seconds = time.perf_counter() - started
    official, actual_prop = np.stack(official), np.stack(actual_prop)
    arrays = dict(physical_costs=truth, official_real_features=official, actual_proprio_features=actual_prop)
    result = dict(track=STATE, scope="offline diagnostics; no online calibration or gate promotion",
                  roots=ROOTS, official_output_max_error=max_error, encoding_seconds=encoding_seconds,
                  representations={}, predictor_transfer=[])
    started = time.perf_counter()
    for i in range(13):
        x = np.stack(buffers.pop(i)) if i < 12 else official
        for mode in (("full", "cls") if i < 12 else ("official",)):
            name = f"block_{i+1:02d}_{mode}" if i < 12 else "official"
            values = x if mode != "cls" else x[..., :1, :]
            record, mapping = representation_audit(values.astype(np.float64), truth)
            result["representations"][name] = record
            for key in ("rotation", "source_mean", "target_mean"):
                arrays[name+"_"+key] = getattr(mapping, key)
            if mode == "cls":
                arrays[name+"_features"] = values
            print(f"{name}: full heldout metrics complete", flush=True)
    result["layer_analysis_seconds"] = time.perf_counter() - started
    with np.load(OUT / "predict.npz", allow_pickle=False) as prediction:
        for r, root in enumerate(ROOTS):
            row = dict(root=root, horizons=[])
            for h, steps in enumerate(HORIZONS):
                item = dict(environment_steps=steps, views=[])
                for v in (0, 1):
                    pv = prediction[f"r{root}_v{v}_pred_visual"][steps//5].reshape(3, 256, 384)
                    pp = prediction[f"r{root}_v{v}_pred_proprio"][steps//5].reshape(3, -1)
                    gv = prediction[f"r{root}_v{v}_goal_visual"].reshape(256, 384)
                    gp = prediction[f"r{root}_v{v}_goal_proprio"].reshape(-1)
                    np.testing.assert_allclose(gv, official[r, v, 10], atol=1e-4, rtol=1e-4)
                    np.testing.assert_allclose(gp, actual_prop[r, v, 10], atol=1e-5, rtol=1e-5)
                    actual = official[r, v, 1+h*3:4+h*3]
                    prop = actual_prop[r, v, 1+h*3:4+h*3]
                    predicted_visual = ((pv-gv)**2).mean((1, 2))
                    actual_visual = ((actual-gv)**2).mean((1, 2))
                    actual_official = actual_visual + .1*((prop-gp)**2).mean(-1)
                    pred_official = prediction[f"r{root}_v{v}_cost"][steps//5]
                    np.testing.assert_allclose(pred_official, predicted_visual + .1*((pp-gp)**2).mean(-1), atol=1e-5, rtol=1e-5)
                    item["views"].append(dict(camera=v, visual_prediction_mse=float(np.mean((pv-actual)**2)),
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
    if fingerprint != model_fingerprint(model):
        raise ValueError("model changed")
    result.update(model_fingerprint=fingerprint, arrays_sha256=save_arrays("analyze", **arrays),
                  sources={**source_hashes(), **{str(OUT / f"{s}.json"): file_hash(OUT / f"{s}.json")
                                                for s in ("prepare", "predict", "replay")}})
    atomic_write_json(OUT / "analyze.json", result)
    print("All layer, root-separated calibration and predictor-transfer diagnostics complete.", flush=True)


def main(stage):
    print(f"PC1 camera {stage}: state-assisted diagnostic, fixed root/action support.", flush=True)
    if stage == "all":
        for name, fn in (("prepare", prepare), ("predict", predict), ("replay", replay), ("analyze", analyze)):
            if (OUT / f"{name}.json").exists():
                read_stage(name)
                print(f"verified existing {name}, resume next stage", flush=True)
            else:
                fn()
    else:
        {"prepare": prepare, "predict": predict, "replay": replay, "analyze": analyze}[stage]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "predict", "replay", "analyze", "all"))
    main(parser.parse_args().stage)
