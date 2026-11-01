"""Offline supplied-witness prediction diagnostic, never a planner candidate."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_reverse_target_cuda_20261101_main as target
from . import p1_reverse_control_20261101_main as control

ROOT = target.ROOT
NAME = "p1_reverse_witness_prediction_20261101_main"
DOC = ROOT.parent / "docs/D2P_P1_REVERSE_WITNESS_PREDICTION_PROTOCOL_20261101_MAIN.md"
PREP = ROOT / "results/p1_reverse_control_20261101_main_structured_PREP_MAIN01.json"
PREP_SHA = "95503f437bd1e69e753bc5ea2c5e1e5e72f3b176a4e21726bdb7c1897a9c1cb5"


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    base = target.encoder.base
    out = Path(output)
    if not out.is_absolute() or out.parent != ROOT / "results" or out.exists():
        raise ValueError("exclusive output")
    _, sources = target.collect()
    if base.sha(PREP) != PREP_SHA or base.sha(control.METRIC) != control.METRIC_SHA:
        raise ValueError("sealed diagnostic inputs")
    prepared = json.loads(PREP.read_bytes())
    task = prepared["source"]["tasks"][0]
    physical_data = target.witness.physical.p3.source_data(prepared["source"])
    ref = physical_data["references"][0]
    report = json.loads((target.DIRECTORY / "report.json").read_bytes())
    metric = json.loads(control.METRIC.read_bytes())
    sources.update(base.bindings())
    for path in (PREP, control.METRIC, DOC, Path(__file__).resolve()):
        sources[str(path)] = base.sha(path)
    frames = list(ref["history"])
    for step in (5, 10, 15, 20):
        with Image.open(report["rows"][step]["png"]) as image:
            frames.append(np.asarray(image).copy())
    actions = np.asarray([r["command"] for r in report["rows"][1:]], np.float32).reshape(4, 5, 2)
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device=device).eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint")
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        counts = dict(encoder_calls=0, predict_calls=0, action_encode_calls=0)
        rows = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            encoded = []
            for frame in frames:
                encoded.append(model.encode({"pixels": p0._image_tensor(
                    frame, transform=p0.image_transform(), device=device)})["emb"][0, 0])
                counts["encoder_calls"] += 1
            actual = torch.stack(encoded)
            old = tensor(task["root_context"]["latent_history"])
            root_error = float((actual[:3] - old).abs().max())
            goal = tensor(metric["goal_latent"])
            if root_error > 1e-5 or not torch.equal(actual[-1], goal):
                raise ValueError("root/target encode identity")
            matrix = torch.as_tensor(metric["sqrt_matrix"], dtype=torch.float64, device=device)
            score = lambda z: float(((z - goal).double() @ matrix.T).square().mean())
            mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
            initial_history = tensor(task["root_context"]["action_history"])[None]
            if not bool((std > 0).all()):
                raise ValueError("normalizer")
            for mode in ("witness_open", "witness_refreshed", "zero_open"):
                z, past = actual[:3][None].clone(), initial_history.clone()
                for i in range(4):
                    if mode == "witness_refreshed":
                        z = actual[i:i+3][None].clone()
                    command = torch.zeros((5, 2), device=device) if mode == "zero_open" else tensor(actions[i])
                    normalized = ((command - mean) / std).reshape(1, 1, 10)
                    context = torch.cat((past, normalized), dim=1)
                    embedded = model.action_encoder(context)
                    counts["action_encode_calls"] += 1
                    prediction = model.predict(z, embedded)
                    counts["predict_calls"] += 1
                    if prediction.shape != z.shape or not bool(torch.isfinite(prediction).all()):
                        raise ValueError("prediction")
                    final = prediction[0, -1]
                    row = dict(mode=mode, step=5*(i+1), predicted=final.cpu().tolist(),
                               predicted_goal_q=score(final))
                    if mode != "zero_open":
                        observed = actual[i+3]
                        row.update(observed=observed.cpu().tolist(), observed_goal_q=score(observed),
                                   prediction_mse=float((final-observed).square().mean()))
                    rows.append(row)
                    z = torch.cat((z, prediction[:, -1:]), dim=1)[:, -3:]
                    past = context[:, -2:]
        torch.cuda.synchronize()
        if base.model_hash(model) != before or any(base.sha(p) != h for p, h in sources.items()):
            raise ValueError("source/model drift")
        digest = base.write(out, dict(schema=NAME, status="complete", host=host, sources=sources,
            model_before=before, model_after=before, counts=counts, rows=rows,
            root_encode_max_abs_error=root_error, P1_admission=False,
            supplied_privileged_witness=True, planner_selection=False, independent_confirmation=False,
            warning="Zero path has model predictions only; no new physical counterfactual."))
    print(json.dumps(dict(status="complete", sha256=digest, rows=len(rows), counts=counts)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-host-cuda", action="store_true", required=True)
    args = parser.parse_args()
    run(args.output)
