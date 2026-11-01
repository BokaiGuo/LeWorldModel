"""Open versus actual-RGB-refreshed prediction of all fixed witness actions."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_centered_target_cuda_20261101_main as target
from . import p1_centered_metric_cuda_20261101_main as metric_source

ROOT = target.encoder.ROOT
METRIC = ROOT / "results/p1_centered_metric_cuda_20261101_main_MAIN01.json"
METRIC_SHA = "ef9ae31f143a18d73432fb45bf861db13f9f852baa776fe977962ec9cb2a4818"
DOC = ROOT.parent / "docs/D2P_P1_CENTERED_PREDICTION_PROTOCOL_20261101_MAIN.md"


def run(output):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    from . import preflight_pusht_d2p_p0 as p0

    output = Path(output)
    if not output.is_absolute() or output.parent != ROOT / "results" or output.exists():
        raise ValueError("exclusive output")
    base = target.encoder.base
    entry, _, sources = target.collect()
    if base.sha(metric_source.TARGET) != metric_source.TARGET_SHA or base.sha(METRIC) != METRIC_SHA:
        raise ValueError("fixed target/metric")
    encoded = json.loads(metric_source.TARGET.read_bytes())["rows"][0]
    metric = json.loads(METRIC.read_bytes())
    if encoded["goal_state"] != entry["goal_state"] or metric["goal_latent"] != encoded["latent"]:
        raise ValueError("goal binding")
    report = json.loads((target.DIRECTORY / "report.json").read_bytes())
    task = json.loads(target.witness.old.PREP.read_bytes())["source"]["tasks"][0]
    for p in (metric_source.TARGET, METRIC, DOC, Path(__file__).resolve()):
        sources[str(p)] = base.sha(p)
    sources.update(base.bindings())
    with base.locking.gpu_lock(base.LOCK):
        host = require_cuda_allocation("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(str(base.MODEL), device="cuda:0").eval().requires_grad_(False)
        before = base.model_hash(model)
        if before != base.MODEL_HASH:
            raise ValueError("checkpoint")
        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device="cuda:0")
        mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
        actual_history = list(encoded["history_latents"])
        raw_history = tensor(report["raw_history"])
        open_z = tensor(actual_history)[None]
        raw = tensor([r["command"] for r in report["rows"][1:]]).reshape(4, 5, 2)
        goal = np.asarray(encoded["latent"], np.float32)
        matrix = np.asarray(metric["sqrt_matrix"], float)

        def score(latent):
            residual = (np.asarray(latent, np.float32)-goal).astype(float)
            return float(np.mean((residual @ matrix.T)**2))

        rows = []
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            for i, packet in enumerate(raw):
                history = ((raw_history[-10:] - mean) / std).reshape(1, 2, 10)
                context = torch.cat((history, ((packet-mean)/std).reshape(1, 1, 10)), dim=1)
                predicted = model.predict(open_z, model.action_encoder(context))[:, -1:]
                fresh = model.predict(tensor(actual_history[-3:])[None], model.action_encoder(context))[:, -1:]
                open_z = torch.cat((open_z, predicted), dim=1)[:, -3:]
                cp = report["rows"][5*(i+1)]
                with Image.open(cp["png"]) as im:
                    pixels = np.asarray(im).copy()
                image = p0._image_tensor(pixels, transform=p0.image_transform(), device=torch.device("cuda:0"))
                actual = model.encode({"pixels": image})["emb"][0, 0].cpu().numpy()
                if not np.isfinite(actual).all():
                    raise ValueError("finite actual encoding")
                predicted, fresh = predicted[0, 0].cpu().numpy(), fresh[0, 0].cpu().numpy()
                if not np.isfinite(predicted).all() or not np.isfinite(fresh).all():
                    raise ValueError("finite predictions")
                rows.append(dict(step=cp["step"], open_latent=predicted.tolist(),
                                 refreshed_latent=fresh.tolist(), actual_latent=actual.tolist(),
                                 open_mse=float(np.mean((predicted-actual)**2)),
                                 refreshed_mse=float(np.mean((fresh-actual)**2)),
                                 open_Q=score(predicted), refreshed_Q=score(fresh), actual_Q=score(actual),
                                 native=target.witness.old.qualification.native(entry["goal_state"], cp["state"])))
                actual_history.append(actual.tolist())
                raw_history = torch.cat((raw_history, packet), dim=0)[-10:]
        torch.cuda.synchronize()
        if base.model_hash(model) != before or any(base.sha(p) != h for p, h in sources.items()):
            raise ValueError("model/source drift")
        digest = base.write(output, dict(status="complete", rows=rows, sources=sources, host=host,
                     model_before=before, model_after=before, encoder_calls=4, predict_calls=8,
                     action_encode_calls=8, P1_admission=False, independent_confirmation=False,
                     witness_actions_used_only_in_diagnostic=True))
    print(json.dumps(dict(sha256=digest, rows=[{k: r[k] for k in (
        "step", "open_mse", "refreshed_mse", "open_Q", "refreshed_Q", "actual_Q")} for r in rows])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-host-cuda", required=True, action="store_true")
    args = parser.parse_args()
    run(args.output)
