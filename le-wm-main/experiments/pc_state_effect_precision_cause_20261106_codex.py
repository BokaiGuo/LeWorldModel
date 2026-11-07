"""Diagnose numerical precision vs scale dependence on already-exposed roots."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_track_smoke_20261106_codex import ROOT, load_model
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor

SOURCE = ROOT / "results/pc_state_effect_local_dev_20261106_codex.json"
DATA = ROOT / "results/pc_real_camera_effect_dev_20261106_codex.npz"
OUT = ROOT / "results/pc_state_effect_precision_cause_20261106_codex.json"
PROTOCOL = ROOT.parent / "docs/PC_STATE_EFFECT_PRECISION_CAUSE_PROTOCOL_20261106_CODEX.md"
STEPS = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)


def cost_at(model, encoded, goal, mean, std, value, dtype, *, gradient=False):
    x = torch.tensor(value, device="cuda", dtype=dtype, requires_grad=gradient)
    raw = torch.stack((x, x * 0)).expand(5, 2)
    packet = ((raw - mean) / std).reshape(1, 1, 10)
    prediction = model.unroll(encoded, act_suffix=packet)["visual"][-1]
    cost = (prediction - goal).square().mean()
    if gradient:
        derivative = float(torch.autograd.grad(cost, x)[0])
    else:
        derivative = None
    return float(cost.detach()), derivative


def scan(model, encoded, goal, mean, std, dtype):
    encoded, goal = encoded.to(dtype=dtype), goal.to(dtype=dtype)
    mean, std = mean.to(dtype=dtype), std.to(dtype=dtype)
    center, grad = cost_at(model, encoded, goal, mean, std, 0., dtype, gradient=True)
    slopes = {}
    for x in (-.001, .001):
        _, derivative = cost_at(model, encoded, goal, mean, std, x, dtype, gradient=True)
        slopes[str(x)] = derivative
    scans = []
    with torch.no_grad():
        for h in STEPS:
            low, _ = cost_at(model, encoded, goal, mean, std, -h, dtype)
            high, _ = cost_at(model, encoded, goal, mean, std, h, dtype)
            scans.append({"step": h, "minus": low, "plus": high,
                          "central_derivative": (high - low) / (2*h),
                          "curvature": (high + low - 2*center) / h**2})
    return {"center": center, "gradient": grad, "gradients_around_zero": slopes,
            "scan": scans, "predictor_calls": 3 + 2*len(STEPS), "backward_calls": 3}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    print("Post-hoc FP32/FP64 predictor precision and action-scale diagnostic.", flush=True)
    previous = json.loads(SOURCE.read_text())
    if previous["task_manifest_hash"] != file_hash(DATA):
        raise ValueError("input physical panel drift")
    model, dataset, audit = load_model()
    if not model.model.action_encoder_inpred:
        raise ValueError("diagnostic assumes action encoder belongs to predictor")
    if any(p.requires_grad for p in model.parameters()) or model.training:
        raise ValueError("requires frozen evaluation model")
    before = model_fingerprint(model)
    mean, std = dataset.action_mean.cuda(), dataset.action_std.cuda()
    contexts = []
    with np.load(DATA, allow_pickle=False) as data, torch.no_grad():
        for i in range(4):
            for camera in (0, 1):
                obs = {"visual": rgb_tensor(data["images"][2*i, 0, camera, 0]),
                       "proprio": torch.tensor(data["traces"][2*i, 0, camera, 0].copy(),
                                               device="cuda", dtype=torch.float32)[None, None]}
                encoded = model.encode(obs)
                goal = model.encode(rgb_tensor(data["images"][2*i+1, 0, camera, 0]))[0]
                contexts.append((1880+i, camera, encoded, goal))
    torch.cuda.synchronize()
    started = time.perf_counter()
    fp32 = []
    for root, camera, context, goal in contexts:
        result = scan(model, context, goal, mean, std, torch.float32)
        old = next(r for r in previous["rows"] if r["root"] == root and r["camera"] == camera)
        old_center = old["predicted_costs_zero_plus_minus"][0]
        effects = old["descriptor_effects"]
        original = [old_center, old_center - effects[0], old_center - effects[1]]
        current = [result["center"], result["scan"][3]["minus"], result["scan"][3]["plus"]]
        np.testing.assert_allclose(current, original, atol=1e-6, rtol=1e-5)
        fp32.append(result)
        print(f"root={root} camera={camera}: FP32 source reproduced", flush=True)
    model.model.predictor.double()
    fp64 = []
    for root, camera, context, goal in contexts:
        result = scan(model, context, goal, mean, std, torch.float64)
        fp64.append(result)
        print(f"root={root} camera={camera}: FP64 predictor scanned", flush=True)
    model.model.predictor.float()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if model_fingerprint(model) != before:
        raise ValueError("frozen model weights changed after float64 diagnostic")
    rows = []
    for (root, camera, _, _), single, double in zip(contexts, fp32, fp64, strict=True):
        rows.append({"root": root, "camera": camera, "fp32": single, "fp64": double,
                     "center_abs_diff": abs(single["center"] - double["center"]),
                     "gradient_abs_diff": abs(single["gradient"] - double["gradient"]),
                     "curvature_abs_differences": [abs(s["curvature"] - d["curvature"])
                                                   for s, d in zip(single["scan"], double["scan"], strict=True)]})
    audit.verify()
    record = {"track_id": audit.track_id, "input_audit": asdict(audit),
              "input_audit_hash": audit.fingerprint, "protocol_sha256": file_hash(PROTOCOL),
              "task_manifest_sha256": file_hash(DATA), "scientific_status": "not_evaluated",
              "execution_status": "completed"}
    record.update(rows=rows, checkpoint_fingerprint=before,
                  original_result_sha256=file_hash(SOURCE), code_sha256=file_hash(__file__),
                  scope="post-hoc state-assisted numerical diagnostic, cached FP32 encodings",
                  predict_calls=sum(r["predictor_calls"] for r in fp32 + fp64),
                  backward_calls=sum(r["backward_calls"] for r in fp32 + fp64),
                  synchronized_seconds=seconds)
    with OUT.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(OUT), "predict_calls": record["predict_calls"],
                      "backward_calls": record["backward_calls"], "seconds": seconds}, indent=2))


if __name__ == "__main__":
    main()
