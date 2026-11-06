"""Fixed common suffix, one exposed v22 root; no planner feedback."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import traceback

import numpy as np

from experiments import pc_jepa_v22_fixed_bank_posthoc_20261105_codex as previous
from experiments import pc_jepa_v22_contact_feedback_dev_20261104_codex as feedback
from experiments import pc_jepa_v12_contact_score_bank_neural_posthoc_20261104_codex as neural
from experiments.spectral_reachability import p5_jepa_factual_nonzero_20261027_v2 as prior

ROOT, SOURCE = previous.ROOT, previous.SOURCE
OUT = ROOT / "results/pc_jepa_v22_unqueried_suffix_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_V22_UNQUERIED_SUFFIX_DEV_PROTOCOL_20261105_CODEX.md"
sha = feedback.sha


def make_bank(response, cem):
    prefixes = np.stack((np.zeros((5, 2), np.float32), response[:5], cem[:5],
                         np.random.default_rng(2026110501).uniform(-1, 1, (5, 2)).astype(np.float32)))
    probes = np.concatenate((np.zeros((1, 10, 2), np.float32),
                             np.random.default_rng(2026110502).uniform(-1, 1, (7, 10, 2)).astype(np.float32),
                             np.random.default_rng(2026110503).uniform(-1, 1, (8, 10, 2)).astype(np.float32)))
    if len({p.tobytes() for p in prefixes}) != 4 or len({p.tobytes() for p in probes}) != 16:
        raise ValueError("duplicate prefix or suffix")
    bank = np.stack([np.concatenate((p, u)) for p in prefixes for u in probes])
    return prefixes, probes, bank


def prepare():
    if OUT.exists():
        raise FileExistsError(OUT)
    previous.preflight()
    paths = [PROTOCOL, Path(__file__), SOURCE / "manifest.json", SOURCE / "summary.json",
             SOURCE / "audit.json", SOURCE / "chronology_audit.json", SOURCE / "physical_replay.json",
             SOURCE / "goal_rgb.npy", SOURCE / "goal_state_evaluation_only.npy"]
    decisions = []
    for arm in ("response", "cem"):
        directory = SOURCE / f"{arm}_0"
        dpath = directory / "output/decision.json"
        decision = json.loads(dpath.read_text())
        if decision["input_sha256"] != sha(directory / "input/context.npz"):
            raise ValueError("decision context mismatch")
        decisions.append(decision)
        paths.extend((dpath, directory / "input/context.npz", directory / "input/model_metadata.json"))
    for key in ("rgb", "state", "proprio"):
        paths.append(SOURCE / f"response_physical/frame_10/{key}.npy")
    chains = [np.asarray(d["decision"]["selected"]["chain"], np.float32) for d in decisions]
    prefixes, probes, bank = make_bank(*chains)
    old = {np.asarray(a, np.float32).tobytes()
           for d in decisions for batch in d["evaluations"] for a in batch["actions"]
           if np.asarray(a).shape == (15, 2)}
    overlaps = [i for i, action in enumerate(bank) if action.tobytes() in old]
    if any(i % 16 != 0 for i in overlaps):
        raise ValueError("nonzero suffix overlaps original first-decision query")
    bindings = {str(path): sha(path) for path in paths}
    OUT.mkdir()
    with (OUT / "bank.npz").open("xb") as handle:
        np.savez_compressed(handle, prefixes=prefixes, suffixes=probes, actions=bank)
    manifest = {"status": "prepared", "bindings": bindings, "bank_sha256": sha(OUT / "bank.npz"),
                "old_distinct_queries": len(old), "bank_count": len(bank),
                "exact_overlap_zero_tail_indices": overlaps, "nonzero_suffix_exact_overlap": 0}
    with (OUT / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(manifest), flush=True)


def verify():
    manifest = json.loads((OUT / "manifest.json").read_text())
    if manifest["status"] != "prepared" or sha(OUT / "bank.npz") != manifest["bank_sha256"]:
        raise ValueError("bank mismatch")
    if any(sha(Path(path)) != digest for path, digest in manifest["bindings"].items()):
        raise ValueError("source changed")
    return manifest


def physical():
    if (OUT / "physical.json").exists():
        raise FileExistsError("physical run already attempted")
    verify()
    if os.environ.get("PYTHONPATH") != str((ROOT / ".jepa-physics-site").resolve()):
        raise ValueError("physics overlay required")
    import gym
    import pymunk
    import skimage
    if (gym.__version__, pymunk.version, skimage.__version__) != ("0.23.1", "6.8.0", "0.25.2"):
        raise ValueError("physics versions")
    base = prior.helpers(ROOT)
    runtime, parent = base.load_runtime(ROOT)
    if runtime.probe(ROOT, parent, ROOT / f"results/{base.RUNTIME}/manifest_frozen.json")["status"] != "native_import_ready_physics_not_executed":
        raise ValueError("runtime preflight failed")
    spec = importlib.util.spec_from_file_location("v22_unqueried_env", ROOT / runtime.ENV_REL)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    goal = np.load(SOURCE / "goal_state_evaluation_only.npy", allow_pickle=False)
    boundary = {k: np.load(SOURCE / f"response_physical/frame_10/{k}.npy", allow_pickle=False)
                for k in ("rgb", "state", "proprio")}
    with np.load(OUT / "bank.npz", allow_pickle=False) as f:
        bank = f["actions"].copy()
    rows, frames, states = [], [], []
    try:
        for index, actions in enumerate(bank):
            env = native.PushTEnv(**base.CTOR)
            try:
                env.seed(manifest["seed"])
                obs, state = env.reset()
                for action in np.asarray(manifest["warmup"], np.float64):
                    obs, _, done, info = env.step(action)
                    if done:
                        raise ValueError("warmup terminated")
                    state = info["state"]
                for key, actual in (("rgb", obs["visual"]), ("state", state), ("proprio", obs["proprio"])):
                    np.testing.assert_array_equal(actual, boundary[key])
                for action in actions.astype(np.float64):
                    obs, _, done, info = env.step(action)
                    if done:
                        raise ValueError("chain terminated")
                    state = info["state"]
                frames.append(obs["visual"].copy())
                states.append(state.copy())
                rows.append({"index": index, "prefix": index // 16, "suffix": index % 16,
                             "status": "completed", "metrics": feedback.metrics(state, goal)})
            finally:
                env.close()
        with (OUT / "physical_arrays.npz").open("xb") as handle:
            np.savez_compressed(handle, terminal_rgb=np.stack(frames), states=np.stack(states))
        verify()
        result = {"status": "completed", "rows": rows, "arrays_sha256": sha(OUT / "physical_arrays.npz"),
                  "environment_sha256": sha(ROOT / runtime.ENV_REL)}
    except Exception:
        result = {"status": "failed", "rows": rows, "error": traceback.format_exc()}
    with (OUT / "physical.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"physical {result['status']} rows={len(rows)}", flush=True)
    if result["status"] != "completed":
        raise RuntimeError("physical failure retained")


def neural_score():
    verify()
    result = json.loads((OUT / "physical.json").read_text())
    if result["status"] != "completed" or result["arrays_sha256"] != sha(OUT / "physical_arrays.npz"):
        raise ValueError("physical result mismatch")
    base = OUT / "neural"
    base.mkdir()
    for group in ("predict", "encode"):
        (base / group / "input").mkdir(parents=True)
        (base / group / "output").mkdir()
    source_input = SOURCE / "response_0/input"
    for name in ("context.npz", "config.json", "model_metadata.json"):
        shutil.copyfile(source_input / name, base / "predict/input" / name)
    shutil.copyfile(source_input / "model_metadata.json", base / "encode/input/model_metadata.json")
    with np.load(OUT / "bank.npz", allow_pickle=False) as bank, np.load(OUT / "physical_arrays.npz", allow_pickle=False) as data:
        with (base / "predict/input/bank.npy").open("xb") as handle:
            np.save(handle, bank["actions"])
        with (base / "encode/input/images.npz").open("xb") as handle:
            np.savez_compressed(handle, goal_rgb=np.load(SOURCE / "goal_rgb.npy", allow_pickle=False),
                                terminal_rgb=data["terminal_rgb"])
    neural.launch(base / "predict", "experiments.pc_jepa_v12_contact_score_predict_worker_20261104_codex")
    neural.launch(base / "encode", "experiments.pc_jepa_v11_same_bank_actual_encoding_worker_20261104_codex")
    p = json.loads((base / "predict/output/decision.json").read_text())
    e = json.loads((base / "encode/output/encoded.json").read_text())
    if (p["bank_sha256"] != sha(base / "predict/input/bank.npy")
            or p["input_sha256"] != sha(source_input / "context.npz")
            or p["worker_sha256"] != sha(neural.PREDICT_WORKER)
            or e["image_sha256"] != sha(base / "encode/input/images.npz")
            or e["worker_sha256"] != sha(neural.ENCODE_WORKER)):
        raise ValueError("worker binding mismatch")
    predicted, encoded = np.asarray(p["decision"]["scores"]), np.asarray(e["scores"])
    if predicted.shape != (64,) or encoded.shape != (64,) or not np.isfinite(predicted).all() or not np.isfinite(encoded).all():
        raise ValueError("missing or invalid scores")
    order = previous.physical.order
    summaries = []
    for prefix in range(4):
        start = prefix * 16
        for name, sub in (("construction", slice(start + 1, start + 8)), ("validation", slice(start + 8, start + 16))):
            rows = result["rows"][sub]
            summaries.append({"prefix": prefix, "group": name,
                              "predicted_vs_encoded": order(predicted[sub], encoded[sub]),
                              "predicted_vs_object_error": order(predicted[sub], np.asarray([r["metrics"]["object_error"] for r in rows])),
                              "predicted_vs_angle_error": order(predicted[sub], np.asarray([r["metrics"]["angle_error_radians"] for r in rows])),
                              "mean_abs_relative_improvement_error": float(np.mean(np.abs((predicted[start]-predicted[sub])-(encoded[start]-encoded[sub]))))})
    with np.load(OUT / "physical_arrays.npz", allow_pickle=False) as data:
        poses = data["states"][:, 2:5]
        pose_counts = [len(np.unique(poses[i*16:(i+1)*16], axis=0)) for i in range(4)]
    output = {"status": "completed", "physical_sha256": sha(OUT / "physical.json"),
              "predict_sha256": sha(base / "predict/output/decision.json"),
              "encode_sha256": sha(base / "encode/output/encoded.json"), "summaries": summaries,
              "pose_counts": pose_counts, "prediction_meter": p["meter"]}
    verify()
    with (OUT / "report.json").open("x") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(output, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "physical", "neural"))
    phase = parser.parse_args().phase
    print(f"Fixed unqueried suffix diagnostic phase={phase}; evaluation only.", flush=True)
    {"prepare": prepare, "physical": physical, "neural": neural_score}[phase]()


if __name__ == "__main__":
    main()
