"""Frozen RGB model action derivatives and separately sealed physical replay."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
import torch

from experiments import pc_goal_mode_rgb_smoke_20261103_codex as prior
from experiments import pc_geometry_backend_replay_20261103_codex as backend
from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
from experiments.pointmaze_action_history_unroll_20261103 import unroll_with_action_history

old, h, ROOT = prior.old, prior.h, prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC1_MULTISTEP_RESPONSE_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT / "results/pc1_multistep_response_dev_20261103_codex"
MODEL = PREFIX.with_name(PREFIX.name + "_model.json")
PHYSICAL = PREFIX.with_name(PREFIX.name + "_physical.json")
EPS = (0.005, 0.05)


def direction(horizon, axis):
    result = np.zeros((horizon, 2), dtype=np.float32)
    result[:, axis] = 1
    return result


def metrics(reference, observed):
    a, b = np.asarray(reference, dtype=float), np.asarray(observed, dtype=float)
    error = float(np.linalg.norm(a - b))
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return {"absolute_error": error, "relative_to_reference": error / na if na > 1e-12 else None,
            "cosine": float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else None}


def model_stage():
    if MODEL.exists() or PHYSICAL.exists():
        raise FileExistsError("existing results must not be overwritten")
    sha = old.prior.sha
    source = json.loads(old.DECISIONS.read_text())
    if sha(old.HISTORY) != source["history_sha256"]:
        raise ValueError("history archive mismatch")
    camera, calibration = (json.loads(path.read_text()) for path in (h.CAMERA, h.OLD))
    metadata, offset = camera["metadata"], calibration["body_offset_from_static_scene"]
    print("PC1 development: AD versus FD; physical replay will be separate.", flush=True)
    with np.load(old.HISTORY, allow_pickle=False) as histories, \
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
            raise ValueError("action normalization mismatch")
        weights, _, _ = h.get_encoder()
        pinv = torch.as_tensor(np.linalg.pinv(weights), dtype=torch.float32, device=device)
        pos_std = torch.as_tensor(h.STD[:2], dtype=torch.float32, device=device)
        history = ((torch.as_tensor(old.WARMUP[5:], dtype=torch.float32, device=device) - mean) / std).reshape(2, 1, 10)
        rows = []
        started = time.monotonic()
        for root in range(4):
            frames = histories[f"root{root}_history_rgb"]
            if [old.digest_image(x) for x in frames] != source["rows"][root]["rgb_sha256"]:
                raise ValueError("history image mismatch")
            with torch.no_grad():
                entries = [h.encode_with_state(model, frames[t],
                           h.causal_state(frames[t - 1], frames[t], metadata, offset, .05), device)
                           for t in (5, 10, 15)]
            ctx = {key: torch.cat([entry[key] for entry in entries], dim=1) for key in ("visual", "proprio")}
            for horizon in (5, 10):
                def predict(actions):
                    packets = ((actions - mean) / std).reshape(1, horizon // 5, 10).transpose(0, 1)
                    output = unroll_with_action_history(model, ctx, history, packets, 3)["proprio"][-1]
                    return (output.reshape(1, 256, 16).mean(1) @ pinv.T)[0, :2] * pos_std

                origin = torch.zeros((horizon, 2), dtype=torch.float32, device=device, requires_grad=True)
                output = predict(origin)
                jac = torch.stack([torch.autograd.grad(output[i], origin, retain_graph=i == 0)[0] for i in range(2)])
                for axis in (0, 1):
                    vector = torch.as_tensor(direction(horizon, axis), device=device)
                    derivative = (jac * vector).sum(dim=(1, 2)).detach().cpu().numpy()
                    fd = []
                    with torch.no_grad():
                        for epsilon in EPS:
                            plus = predict(epsilon * vector).cpu().numpy()
                            minus = predict(-epsilon * vector).cpu().numpy()
                            value = (plus - minus) / (2 * epsilon)
                            fd.append({"epsilon": epsilon, "plus": plus.tolist(), "minus": minus.tolist(),
                                       "derivative": value.tolist(), "ad_comparison": metrics(derivative, value)})
                    rows.append({"root": root, "horizon": horizon, "axis": axis,
                                 "ad_derivative": derivative.tolist(), "fd": fd})
                print(f"model root={root} horizon={horizon} complete", flush=True)
        torch.cuda.synchronize()
        if h.parameter_hash(model) != model_hash:
            raise ValueError("model mutated")
        old.write_json(MODEL, {"scope": "reused development roots, not independent", "rows": rows,
                              "protocol_sha256": sha(PROTOCOL), "runner_sha256": sha(Path(__file__)),
                              "history_sha256": sha(old.HISTORY), "checkpoint_sha256": sha(h.DEFAULT_CHECKPOINT),
                              "model_hash": model_hash, "encoding_calls": 12,
                              "backward_calls": 16, "forward_calls": 72,
                              "elapsed_seconds": time.monotonic() - started})


def physical_stage():
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    sha = old.prior.sha
    model = json.loads(MODEL.read_text())
    if model["protocol_sha256"] != sha(PROTOCOL) or model["runner_sha256"] != sha(Path(__file__)):
        raise ValueError("sealed model protocol or runner mismatch")
    requests = []
    for row in model["rows"]:
        for epsilon in EPS:
            for sign in (-1, 1):
                actions = sign * epsilon * direction(row["horizon"], row["axis"])
                key = f'{row["root"]}-{row["horizon"]}-{row["axis"]}-{epsilon}-{sign}'
                requests.append({"start": list(old.STARTS[row["root"]]), "policy": key,
                                 "actions": old.WARMUP.tolist() + actions.tolist()})
    with tempfile.TemporaryDirectory(prefix="pc1_response_") as directory:
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
    if replay["binding"] != "2.1.2.14" or len(replay["rows"]) != len(requests):
        raise ValueError("backend mismatch")
    traces = {}
    warm = {}
    for request, row in zip(requests, replay["rows"], strict=True):
        if request["policy"] != row["policy"] or len(row["trace"]) != len(request["actions"]) + 1:
            raise ValueError("replay identity or clock mismatch")
        root = request["policy"].split("-")[0]
        state = np.asarray(row["trace"][15]["state"])
        if root in warm and not np.allclose(state, warm[root], atol=1e-10, rtol=0):
            raise ValueError("warmup state mismatch")
        warm[root] = state
        traces[row["policy"]] = row["trace"]
    rows = []
    for row in model["rows"]:
        for fd in row["fd"]:
            epsilon = fd["epsilon"]
            prefix = f'{row["root"]}-{row["horizon"]}-{row["axis"]}-{epsilon}'
            minus, plus = (np.asarray(traces[f"{prefix}-{sign}"][-1]["state"][:2]) for sign in (-1, 1))
            derivative = (plus - minus) / (2 * epsilon)
            rows.append({"root": row["root"], "horizon": row["horizon"], "axis": row["axis"],
                         "epsilon": epsilon, "physical_derivative": derivative.tolist(),
                         "model_fd_derivative": fd["derivative"],
                         "comparison": metrics(derivative, fd["derivative"])})
    old.write_json(PHYSICAL, {"scope": "physical diagnostic only", "model_sha256": sha(MODEL),
                              "protocol_sha256": sha(PROTOCOL), "worker_sha256": sha(backend.WORKER),
                              "backend": replay["binding"], "rows": rows, "traces": traces})
    print(f"saved {len(rows)} paired physical comparisons", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("model", "physical"))
    stage = parser.parse_args().stage
    (model_stage if stage == "model" else physical_stage)()
