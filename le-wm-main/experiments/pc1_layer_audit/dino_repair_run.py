"""Resumable source-parity repair, sealed prediction, then outcome evaluation."""
import argparse
from dataclasses import replace
import fcntl
import inspect
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from experiments.pc1_layer_audit import camera, dino_source_compatible as repair
from experiments.pc1_layer_audit.metrics import ranking_record
from experiments import pc_input_permissions_v2_20261107_codex as permissions
from experiments.pc_input_permissions import file_hash, STATE
from experiments.pc_pointmaze_dino_backend_dev_20261106_codex import setup, SOURCE, WEIGHTS
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.research_runtime.artifacts import atomic_write_json

ROOT = camera.ROOT
BASE = ROOT / "results/pc1_camera_layer_transfer_runtime_bound_20261107"
OUT = ROOT / "results/pc1_dino_source_compatible_20261107"
PROTOCOL = ROOT.parent / "docs/PC1_DINO_SOURCE_COMPATIBILITY_REPAIR_PROTOCOL_20261107_CODEX.md"
ATOL, RTOL, RELATIVE = 5e-4, 1e-4, 1e-4


def write_json(path, record):
    if path.exists():
        raise FileExistsError(path)
    json.dumps(record, allow_nan=False)
    atomic_write_json(path, record)


def check_sources(record):
    for name, digest in record["sources"].items():
        if file_hash(name) != digest:
            raise ValueError(f"sealed dependency changed: {name}")


def read_record(name):
    path = OUT / name
    record = json.loads(path.read_text())
    check_sources(record)
    if "arrays" in record and file_hash(OUT / record["arrays"]) != record["arrays_sha256"]:
        raise ValueError(f"array digest changed: {name}")
    return record


def sources(paths):
    return {str(Path(p).resolve()): file_hash(p) for p in paths}


def error_record(actual, reference):
    if actual.shape != reference.shape or not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError("nonfinite or unmatched tensors")
    delta = (actual-reference).float()
    relative = float(delta.norm()/reference.float().norm().clamp_min(1e-12))
    violations = int((delta.abs() > ATOL + RTOL * reference.abs()).sum())
    return dict(max_abs=float(delta.abs().max()), mse=float(delta.square().mean()),
                relative_l2=relative, violations=violations,
                passed=bool(violations == 0 and relative <= RELATIVE))


def require_parity(actual, reference, label):
    record = error_record(actual, reference)
    if not record["passed"]:
        raise ValueError(f"{label} outside frozen tolerance: {record}")
    return record


def initialize():
    if (OUT / "manifest.json").exists():
        return read_record("manifest.json")
    dependencies = [__file__, repair.__file__, camera.__file__, permissions.__file__,
                    ROOT / "experiments/pc_input_permissions.py", PROTOCOL, WEIGHTS,
                    WEIGHTS.parent / "config.json",
                    ROOT / "experiments/pc_pointmaze_dino_backend_dev_20261106_codex.py",
                    ROOT / "experiments/pc_dinov2_adapter_equivalence_dev_20261106_codex.py",
                    ROOT / "external/local_dinov2/hubconf.py",
                    ROOT / "experiments/pc_state_track_smoke_20261106_codex.py"]
    for stage in ("prepare", "predict", "replay", "analyze"):
        previous = json.loads((BASE / f"{stage}.json").read_text())
        check_sources(previous)
        if file_hash(BASE / f"{stage}.npz") != previous["arrays_sha256"]:
            raise ValueError(f"historical {stage} arrays changed")
        dependencies.extend((BASE / f"{stage}.json", BASE / f"{stage}.npz"))
    dependencies.extend(SOURCE.rglob("*.py"))
    record = dict(track=STATE, sources=sources(dependencies), roots=camera.ROOTS,
                  protocol="implementation repair; not a reopened performance hypothesis",
                  permissions="current/goal RGB and full proprio; evaluation truth excluded from prediction",
                  tolerances=dict(atol=ATOL, rtol=RTOL, relative_l2=RELATIVE),
                  created_unix=time.time(), scientific_gate_promotion=False)
    write_json(OUT / "manifest.json", record)
    return record


def models():
    model, dataset, audit, backends = setup()
    shim = backends["hf_shim"]
    before = model_fingerprint(model)
    repair.install_source_interpolation(shim.model)
    after = model_fingerprint(model)
    if before != after:
        raise ValueError("repair changed weights")
    paths = dict(audit.dependencies)
    paths.update(sources([repair.__file__, __file__, permissions.__file__,
                          inspect.getfile(type(shim.model)), PROTOCOL]))
    audit = replace(audit, dependencies=tuple(paths.items())).verify()
    return model, dataset, audit, shim, backends["source_snapshot"], before


def layer_parity():
    if (OUT / "parity.json").exists():
        return read_record("parity.json")
    model, _, _, shim, reference, fingerprint = models()
    hf, ref = {}, {}
    handles = []
    def capture(target, name):
        def hook(_module, _inputs, value):
            target[name] = value.detach()
        return hook
    handles.append(shim.model.embeddings.register_forward_hook(capture(hf, "embedding")))
    def ref_input(_module, inputs):
        ref["embedding"] = inputs[0].detach()
    handles.append(reference.blocks[0].register_forward_pre_hook(ref_input))
    for i in range(12):
        handles.append(shim.model.encoder.layer[i].register_forward_hook(capture(hf, f"block_{i+1:02d}")))
        handles.append(reference.blocks[i].register_forward_hook(capture(ref, f"block_{i+1:02d}")))
    rows = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        with np.load(BASE / "prepare.npz", allow_pickle=False) as prep, torch.inference_mode():
            for index in (0, 8, 16, 24):
                for view in (0, 1):
                    raw = camera.raw_rgb(prep["frames"][index, 0, view][None]).float()/255.
                    normalized = model.preprocessor.transform(raw).flatten(0, 1)
                    for shape in ((196, 196), (224, 224), (224, 280), (518, 518)):
                        x = F.interpolate(normalized, size=shape, mode="bilinear", align_corners=False, antialias=True)
                        for batch in (1, 3):
                            images = x.expand(batch, -1, -1, -1).contiguous()
                            expected = reference.forward_features(images)
                            actual = shim.forward_features(images)
                            checks = {k: require_parity(hf[k], ref[k], k) for k in hf}
                            for key in ("x_norm_clstoken", "x_norm_patchtokens"):
                                checks[key] = require_parity(actual[key], expected[key], key)
                            rows.append(dict(root=camera.ROOTS[index], view=view, shape=shape,
                                             batch=batch, checks=checks))
                            hf.clear(); ref.clear()
                    print(f"layer parity root={camera.ROOTS[index]}, view={view}", flush=True)
        torch.cuda.synchronize()
    finally:
        for h in handles:
            h.remove()
    if fingerprint != model_fingerprint(model):
        raise ValueError("parity changed weights")
    record = dict(sources=sources([OUT/"manifest.json"]), rows=rows, passed=True,
                  model_fingerprint=fingerprint, synchronized_seconds=time.perf_counter()-started,
                  gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  backend_sources=sources([inspect.getfile(type(shim.model)),
                      inspect.getfile(type(shim.model.embeddings)), inspect.getfile(type(reference))]))
    record["sources"].update(record["backend_sources"])
    write_json(OUT / "parity.json", record)
    return record


def predict():
    parity = read_record("parity.json")
    model, _, audit, shim, reference, fingerprint = models()
    if fingerprint != parity["model_fingerprint"]:
        raise ValueError("prediction model drift")
    raw, packet = camera.packet_actions(model.preprocessor)
    wrapper = permissions.StateAssistedPointMazev2(model, audit)
    with np.load(BASE / "prepare.npz", allow_pickle=False) as prep, torch.inference_mode():
        for index, root in enumerate(camera.ROOTS):
            name = f"predict_r{root}.json"
            if (OUT / name).exists():
                read_record(name)
                continue
            arrays, rows = {"raw_actions": raw, "packets": packet.cpu().numpy()}, []
            receipts_before = len(wrapper.receipts)
            torch.cuda.synchronize()
            started = time.perf_counter()
            for view in (0, 1):
                obs = dict(visual=camera.raw_rgb(np.repeat(prep["frames"][index, 0, view][None], 3, axis=0)),
                           proprio=torch.tensor(prep["states"][index, 0], dtype=torch.float32, device="cuda")[None, None].expand(3, 1, 4).contiguous())
                target = dict(visual=camera.raw_rgb(prep["frames"][index, 1, view][None]),
                              proprio=torch.tensor(prep["states"][index, 1], dtype=torch.float32, device="cuda")[None, None])
                outputs, costs = {}, {}
                for name_backend, encoder in (("repaired", shim), ("source", reference)):
                    model.model.encoder.base_model = encoder
                    goal = model.encode(target)
                    pred = wrapper.predict(obs, packet)
                    cost = sum(weight * ((pred[k]-goal[k])**2).flatten(2).mean(-1)
                               for k, weight in (("visual", 1.), ("proprio", .1)))
                    outputs[name_backend] = (goal, pred)
                    costs[name_backend] = cost
                checks = {}
                for key in ("visual", "proprio"):
                    for field, item in (("goal", 0), ("pred", 1)):
                        a, b = (outputs[n][item][key] for n in ("repaired", "source"))
                        checks[f"{field}_{key}"] = require_parity(a, b, f"{root}/{view}/{field}/{key}")
                        arrays[f"v{view}_{field}_{key}"] = a.cpu().numpy()
                checks["cost"] = require_parity(costs["repaired"], costs["source"], "cost")
                selections = {n: costs[n].argmin(-1).cpu().tolist() for n in costs}
                rows.append(dict(view=view, checks=checks, scores={n: c.cpu().tolist() for n, c in costs.items()},
                                 selections=selections, choices_equal=selections["repaired"] == selections["source"]))
            model.model.encoder.base_model = shim
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            if fingerprint != model_fingerprint(model):
                raise ValueError("prediction changed weights")
            arrays_name = f"predict_r{root}.npz"
            with (OUT / arrays_name).open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            record = dict(root=root, rows=rows, track=STATE, arrays=arrays_name,
                          arrays_sha256=file_hash(OUT/arrays_name), model_fingerprint=fingerprint,
                          sources=sources([OUT/"manifest.json", OUT/"parity.json"]),
                          synchronized_prediction_seconds=elapsed, new_goal_encodings=4,
                          new_current_images_encoded=12, new_candidate_rollouts=12,
                          model_packets=60, audit_sha256=audit.fingerprint,
                          receipts=wrapper.receipts[receipts_before:])
            write_json(OUT / f"predict_r{root}.json", record)
            print(f"sealed predictions root={root}, source choices equal={all(r['choices_equal'] for r in rows)}", flush=True)


def evaluate():
    # Enforce full-panel seal BEFORE opening any physical results.
    sealed = [read_record(f"predict_r{root}.json") for root in camera.ROOTS]
    model, _, _, shim, _, fingerprint = models()
    if any(r["model_fingerprint"] != fingerprint for r in sealed):
        raise ValueError("evaluation model drift")
    previous = json.loads((BASE / "analyze.json").read_text())
    with (np.load(BASE / "prepare.npz", allow_pickle=False) as prep,
          np.load(BASE / "replay.npz", allow_pickle=False) as replay, torch.inference_mode()):
        for index, root in enumerate(camera.ROOTS):
            name = f"evaluate_r{root}.json"
            if (OUT / name).exists():
                read_record(name)
                continue
            rows, arrays = [], {}
            torch.cuda.synchronize()
            started = time.perf_counter()
            with np.load(OUT/f"predict_r{root}.npz", allow_pickle=False) as prediction:
                for view in (0, 1):
                    real_v, real_p = [], []
                    images = [prep["frames"][index, 0, view]] + [
                        replay["images"][index, a, view, t] for t in (1, 2, 3) for a in range(3)
                    ] + [prep["frames"][index, 1, view]]
                    states = [prep["states"][index, 0]] + [
                        replay["traces"][index, a, view, t] for t in camera.HORIZONS for a in range(3)
                    ] + [prep["states"][index, 1]]
                    for image, state in zip(images, states, strict=True):
                        z = model.encode(dict(visual=camera.raw_rgb(image[None]),
                            proprio=torch.tensor(state, dtype=torch.float32, device="cuda")[None, None]))
                        real_v.append(z["visual"].cpu().numpy().reshape(256, 384))
                        real_p.append(z["proprio"].cpu().numpy().reshape(-1))
                    real_v, real_p = np.stack(real_v), np.stack(real_p)
                    arrays[f"v{view}_real_visual"], arrays[f"v{view}_real_proprio"] = real_v, real_p
                    gv = prediction[f"v{view}_goal_visual"].reshape(256, 384)
                    gp = prediction[f"v{view}_goal_proprio"].reshape(-1)
                    np.testing.assert_array_equal(gv, real_v[-1])
                    np.testing.assert_array_equal(gp, real_p[-1])
                    for hi, steps in enumerate(camera.HORIZONS):
                        pv = prediction[f"v{view}_pred_visual"][steps//5].reshape(3, 256, 384)
                        pp = prediction[f"v{view}_pred_proprio"][steps//5].reshape(3, -1)
                        av, ap = real_v[1+hi*3:4+hi*3], real_p[1+hi*3:4+hi*3]
                        pc = ((pv-gv)**2).mean((1, 2)) + .1*((pp-gp)**2).mean(-1)
                        ac = ((av-gv)**2).mean((1, 2)) + .1*((ap-gp)**2).mean(-1)
                        old = previous["predictor_transfer"][index]["horizons"][hi]["views"][view]
                        truth = replay["physical_costs"][index, hi]
                        np.testing.assert_array_equal(truth, old["physical_costs"])
                        np.testing.assert_allclose(pc, sealed[index]["rows"][view]["scores"]["repaired"][steps//5],
                                                   atol=1e-5, rtol=1e-5)
                        rows.append(dict(root=root, view=view, steps=steps, physical_costs=truth.tolist(),
                            predicted_costs=pc.tolist(), real_encoded_costs=ac.tolist(),
                            repaired_predicted=ranking_record(pc, truth), repaired_real=ranking_record(ac, truth),
                            raw_predicted=old["predicted_official"], raw_real=old["real_official"],
                            repaired_predicted_vs_real=ranking_record(pc, ac),
                            raw_visual_mse=old["visual_prediction_mse"],
                            repaired_visual_mse=float(np.mean((pv-av)**2)),
                            raw_proprio_mse=old["proprio_prediction_mse"],
                            repaired_proprio_mse=float(np.mean((pp-ap)**2)),
                            repaired_visual_persistence_mse=float(np.mean((real_v[0]-av)**2))))
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            if model_fingerprint(model) != fingerprint:
                raise ValueError("evaluation changed weights")
            array_name = f"evaluate_r{root}.npz"
            with (OUT/array_name).open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            write_json(OUT/name, dict(root=root, track=STATE, rows=rows, arrays=array_name,
                arrays_sha256=file_hash(OUT/array_name), synchronized_encoding_seconds=elapsed,
                new_actual_encodings=22,
                sources=sources([OUT/"manifest.json", OUT/f"predict_r{root}.json"])))
            print(f"evaluated root={root}", flush=True)


def summarize():
    if (OUT/"summary.json").exists():
        return read_record("summary.json")
    raw = [read_record(f"evaluate_r{r}.json") for r in camera.ROOTS]
    preds = [read_record(f"predict_r{r}.json") for r in camera.ROOTS]
    parity = read_record("parity.json")
    entries = [row for record in raw for row in record["rows"]]
    tables = []
    for group, roots in (("all_exposed", camera.ROOTS), ("prior_calibration", camera.ROOTS[:16]),
                         ("prior_evaluation_exposed", camera.ROOTS[16:])):
        for step in camera.HORIZONS:
            for view in (0, 1):
                rows = [r for r in entries if r["root"] in roots and r["steps"] == step and r["view"] == view]
                delta = np.array([r["repaired_predicted"]["regret"]-r["raw_predicted"]["regret"] for r in rows])
                rng = np.random.default_rng(711071)
                ci = np.quantile(rng.choice(delta, (10000, len(delta))).mean(1), [.025, .975])
                values = dict(group=group, steps=step, view=view, roots=len(rows),
                    mean_regret={k: float(np.mean([r[k]["regret"] for r in rows])) for k in
                        ("raw_predicted", "repaired_predicted", "raw_real", "repaired_real")},
                    mean_mse={k: float(np.mean([r[k] for r in rows])) for k in
                        ("raw_visual_mse", "repaired_visual_mse", "raw_proprio_mse",
                         "repaired_proprio_mse", "repaired_visual_persistence_mse")},
                    regret_delta_mean=float(delta.mean()), descriptive_root_bootstrap95=ci.tolist(),
                    improved=int((delta < -1e-9).sum()), worsened=int((delta > 1e-9).sum()),
                    action_changes=sum(r["raw_predicted"]["choice"] != r["repaired_predicted"]["choice"] for r in rows))
                tables.append(values)
    result = dict(track=STATE, scope="paired implementation repair on exposed roots; not control gain confirmation",
                  layer_case_count=len(parity["rows"]), parity_passed=parity["passed"],
                  maximum_layer_error=max(c["max_abs"] for r in parity["rows"] for c in r["checks"].values()),
                  source_choice_mismatches=sum(not r["choices_equal"] for p in preds for r in p["rows"]),
                  tables=tables, scientific_gate_promotion=False,
                  prediction_seconds=sum(p["synchronized_prediction_seconds"] for p in preds),
                  evaluation_encoding_seconds=sum(r["synchronized_encoding_seconds"] for r in raw),
                  new_candidate_rollouts=sum(p["new_candidate_rollouts"] for p in preds),
                  new_model_packets=sum(p["model_packets"] for p in preds),
                  new_actual_encodings=sum(r["new_actual_encodings"] for r in raw),
                  sources=sources([OUT/"manifest.json", OUT/"parity.json"] +
                      [OUT/f"{stage}_r{root}.json" for root in camera.ROOTS for stage in ("predict", "evaluate")]))
    write_json(OUT/"summary.json", result)
    print(json.dumps({k: result[k] for k in ("parity_passed", "maximum_layer_error", "source_choice_mismatches",
                                             "prediction_seconds", "evaluation_encoding_seconds")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("parity", "predict", "evaluate", "summarize", "all"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initialize()
        stages = dict(parity=layer_parity, predict=predict, evaluate=evaluate, summarize=summarize)
        try:
            for key in (stages if args.stage == "all" else (args.stage,)):
                print(f"Stage {key}: source fidelity and paired consequences; no training or gate promotion.", flush=True)
                stages[key]()
        except Exception:
            write_json(OUT/f"failure_{time.time_ns()}.json", dict(stage=args.stage, error=traceback.format_exc()))
            raise


if __name__ == "__main__":
    main()
