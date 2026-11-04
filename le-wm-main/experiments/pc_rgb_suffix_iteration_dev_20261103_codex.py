"""Fixed-candidate RGB suffix CEM batching diagnostic, with sealed physical replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch

from experiments import pc_goal_mode_matched_feedback_20261103_codex as prior
from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history

old, h, ROOT = prior.old, prior.h, prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_SUFFIX_ITERATION_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT / "results/pc_rgb_suffix_iteration_dev_20261103_codex"
MODEL = PREFIX.with_name(PREFIX.name + "_model.json")
PHYSICAL = PREFIX.with_name(PREFIX.name + "_physical.json")
CONFIGS = (("one12", 12, 12, 4), ("two6", 12, 6, 2), ("four12", 48, 12, 4))


def optimize(prefix, count, batch_size, elites, rng, predict):
    prefix = np.asarray(prefix, dtype=float)
    if prefix.shape != (5, 2) or count < 1 or batch_size < 1 or elites < 1 or elites > batch_size:
        raise ValueError("invalid fixed-prefix search settings")
    loc, scale = np.zeros((10, 2)), np.full((10, 2), 1 / 3)
    best, best_score, trace = None, float("inf"), []
    for offset in range(0, count, batch_size):
        size = min(batch_size, count - offset)
        samples = np.clip(loc + rng.normal(size=(size, 10, 2)) * scale, -1, 1)
        samples[0] = loc
        chains = np.concatenate((np.broadcast_to(prefix, (size, 5, 2)), samples), axis=1)
        values = np.asarray(predict(chains))
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError("invalid model scoring")
        order = np.argsort(values, kind="stable")
        if values[order[0]] < best_score:
            best, best_score = chains[order[0]].copy(), float(values[order[0]])
        if size > 1:
            selected = samples[order[:min(elites, size)]]
            loc, scale = selected.mean(0), np.maximum(selected.std(0, ddof=0), 1e-5)
        trace.append({"chains": chains.tolist(), "scores": values.tolist(), "post_mean": loc.tolist(),
                      "post_std": scale.tolist(), "elite_indices": order[:min(elites, size)].tolist()})
    return {"chain": best.tolist(), "terminal_score": best_score, "samples": count, "batches": trace}


def model_stage():
    if MODEL.exists() or PHYSICAL.exists():
        raise FileExistsError("diagnostic outputs already exist")
    sha = old.prior.sha
    source = json.loads(old.DECISIONS.read_text())
    if sha(old.HISTORY) != source["history_sha256"]:
        raise ValueError("history archive mismatch")
    camera, calibration = (json.loads(p.read_text()) for p in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    print("Fixed RGB candidates: one batch vs two batches vs 48 samples; development only.", flush=True)
    with np.load(old.HISTORY, allow_pickle=False) as histories, \
            np.load(h.OBSTACLE.with_suffix(".npz"), allow_pickle=False) as obstacle, \
            h.lock.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        device = h.require_cuda()
        args = argparse.Namespace(dataset_root=h.DEFAULT_DATASET, checkpoint=h.DEFAULT_CHECKPOINT,
                                  config=h.DEFAULT_CONFIG, dinov2_source=ROOT / "external/local_dinov2")
        model, dataset = h.load_model_and_dataset(args, device)
        model.eval()
        model_hash = h.parameter_hash(model)
        if model_hash != source["model_hash"] or sha(h.DEFAULT_CHECKPOINT) != source["checkpoint_sha256"]:
            raise ValueError("checkpoint mismatch")
        mean, std = dataset.action_mean.to(device), dataset.action_std.to(device)
        if not np.array_equal(mean.cpu().numpy(), source["action_mean"]) or not np.array_equal(std.cpu().numpy(), source["action_std"]):
            raise ValueError("normalizer mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        goal = obstacle["goal_rgb"]
        if old.digest_image(goal) != source["goal_sha256"]:
            raise ValueError("goal mismatch")
        with torch.no_grad():
            goal_state, _ = h.estimate_state(goal, metadata, offset)
            target = h.encode_with_state(model, goal, goal_state, device)["proprio"].reshape(256, 16)
        rows = []
        for root in range(4):
            frames = histories[f"root{root}_history_rgb"]
            if [old.digest_image(x) for x in frames] != source["rows"][root]["rgb_sha256"]:
                raise ValueError("history images differ")
            decision_path = prior.paths(root, "goal")[0][0]
            decision = json.loads(decision_path.read_text())
            if (decision["runner_sha256"] != sha(Path(prior.__file__))
                    or decision["protocol_sha256"] != sha(prior.PROTOCOL)
                    or decision["optimizer_sha256"] != sha(Path(prior.optimize.__code__.co_filename))
                    or decision["model_hash"] != model_hash
                    or decision["history_sha256"] != sha(old.HISTORY)
                    or decision["ticks"] != [5, 10, 15]):
                raise ValueError("prior decision binding mismatch")
            with torch.no_grad():
                entries = [h.encode_with_state(model, frames[t],
                           h.causal_state(frames[t - 1], frames[t], metadata, offset, .05), device)
                           for t in (5, 10, 15)]
            ctx = {key: torch.cat([e[key] for e in entries], dim=1) for key in ("visual", "proprio")}
            history = ((torch.as_tensor(old.WARMUP[5:], dtype=torch.float32, device=device) - mean) / std).reshape(2, 1, 10)

            def predict(chains):
                n = len(chains)
                packets = ((torch.as_tensor(chains, dtype=torch.float32, device=device) - mean) / std).reshape(n, 3, 10).transpose(0, 1)
                with torch.no_grad():
                    out = unroll_with_action_history(model, ctx, history.expand(2, n, 10), packets, 3)["proprio"][-1]
                    values = h.projected_cost(out.reshape(n, 256, 16), target, pinv, pos_std)
                    torch.cuda.synchronize()
                return values.cpu().numpy()

            programs = decision["result"]["programs"]
            if len(programs) != 3:
                raise ValueError("expected three fixed candidates")
            for program in programs:
                for name, count, batch, elites in CONFIGS:
                    started = time.monotonic()
                    result = optimize(program["prefix"], count, batch, elites,
                                      np.random.default_rng(2026110900 + 10 * root), predict)
                    result["seconds"] = time.monotonic() - started
                    if name == "one12":
                        old_chain = np.asarray(program["prefix"] + program["suffix"])
                        if (not np.allclose(result["chain"], old_chain, atol=1e-12, rtol=0)
                                or not np.isclose(result["terminal_score"], program["terminal_score"], atol=2e-6, rtol=0)):
                            raise ValueError("original one-batch outcome failed to reproduce")
                    rows.append({"root": root, "candidate": program["name"], "config": name,
                                 "decision_sha256": sha(decision_path), **result})
            print(f"root {root}: three fixed candidates x three search policies sealed", flush=True)
        if h.parameter_hash(model) != model_hash:
            raise ValueError("model mutated")
        old.write_json(MODEL, {"rows": rows, "scope": "fixed-candidate post-outcome development diagnostic",
                              "protocol_sha256": sha(PROTOCOL), "runner_sha256": sha(Path(__file__)),
                              "checkpoint_sha256": sha(h.DEFAULT_CHECKPOINT), "history_sha256": sha(old.HISTORY),
                              "model_hash": model_hash, "full_cost_certified": False})


def physical_stage():
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    sha = old.prior.sha
    record = json.loads(MODEL.read_text())
    if record["protocol_sha256"] != sha(PROTOCOL) or record["runner_sha256"] != sha(Path(__file__)):
        raise ValueError("model output seal mismatch")
    requests = [{"start": list(old.STARTS[row["root"]]), "policy": str(i),
                 "actions": old.WARMUP.tolist() + row["chain"]} for i, row in enumerate(record["rows"])]
    with tempfile.TemporaryDirectory(prefix="pc_rgb_suffix_iter_") as directory:
        tmp = Path(directory)
        req, xml, out = tmp / "req.json", tmp / "scene.xml", tmp / "out.json"
        req.write_text(json.dumps({"rows": requests}))
        xml.write_text(official_xml())
        if sha(xml) != "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d":
            raise ValueError("scene mismatch")
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(backend.RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(backend.RUNTIME / "lib"), str(backend.RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        subprocess.run([str(backend.LEGACY / "bin/python"), str(backend.WORKER), "native",
                        str(req), str(xml), str(out)], env=env, check=True, timeout=180)
        replay = json.loads(out.read_text())
    if replay["binding"] != "2.1.2.14" or len(replay["rows"]) != len(record["rows"]):
        raise ValueError("physical backend mismatch")
    rows = []
    for i, (source, item) in enumerate(zip(record["rows"], replay["rows"], strict=True)):
        if item["policy"] != str(i) or len(item["trace"]) != 31:
            raise ValueError("physical action clock mismatch")
        distance = np.linalg.norm(np.asarray([x["state"][:2] for x in item["trace"]]) - [1.8, 2.8], axis=1)
        rows.append({"root": source["root"], "candidate": source["candidate"], "config": source["config"],
                     "prediction": source["terminal_score"], "distances": [float(distance[t]) for t in (15, 20, 25, 30)],
                     "trace": item["trace"]})
    old.write_json(PHYSICAL, {"model_sha256": sha(MODEL), "protocol_sha256": sha(PROTOCOL),
                              "backend": replay["binding"], "rows": rows, "full_cost_certified": False})
    print(f"replayed {len(rows)} fixed chains; no feedback comparison", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("model", "physical"))
    stage = parser.parse_args().stage
    (model_stage if stage == "model" else physical_stage)()
