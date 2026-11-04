"""Encode all already-executed spectral candidate endpoints with the frozen model."""
import argparse
import os
from pathlib import Path

import numpy as np

from . import p1_dev304_qgoal_spectral_saved_20261102 as saved

physical, model_run, base = saved.physical, saved.model_run, saved.base
run = physical.run
ROOT = run.ROOT
NAME = "p1_dev304_qgoal_real_rgb_20261102"
AUTHORITY = "DEV304_QGOAL_REAL_RGB_CUDA"
need = run.require


def protocol():
    saved.analyze()
    source_spec = base.read(ROOT/f"results/{model_run.NAME}_PREP_DEV01.json", physical.CONTRACT_SHA)
    model = base.read(ROOT/f"results/{model_run.NAME}_RUN_DEV01/report.json", physical.MODEL_SHA)
    physical_report = base.read(saved.RUN/"report.json", saved.REPORT_SHA)
    candidates = []
    for row in physical_report["rows"]:
        frame = row["observations"][-1]
        need(frame["phase"] == "tail" and frame["step"] == 10, "terminal frame")
        candidates.append({"arm": row["arm"], "prefix": row["prefix"], "candidate": row["candidate"],
                           "frame": {"path": str((saved.RUN/frame["file"]).resolve()),
                                     "file_sha256": frame["file_sha256"], "pixel_sha256": frame["pixel_sha256"]}})
    need(len(candidates) == 18 and model["model_before"] == model["model_after"] == source_spec["model_hash"],
         "physical roster/model")
    return {"schema": NAME+".protocol.v1", "classification": "post_outcome_one_known_root",
            "code_sha256": base.sha(__file__), "model_report_sha256": physical.MODEL_SHA,
            "physical_report_sha256": saved.REPORT_SHA, "goal": source_spec["goal"],
            "policy": source_spec["policy"], "model_hash": source_spec["model_hash"],
            "latent_dim": source_spec["latent_dim"], "candidates": candidates,
            "p1_admission": False, "independent_confirmation": False}


def encode(contract_path, digest, output, authority):
    need(authority == AUTHORITY and base.read(contract_path, digest) == protocol()
         and not os.path.lexists(output), "contract/authority")
    spec = base.read(contract_path, digest)
    output = Path(output)
    output.mkdir()
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    device = torch.device("cuda:0")
    host = require_cuda_allocation("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    model = load_compatible_pretrained(spec["policy"], device=device).eval().requires_grad_(False)
    hasher = run.coherent.base.first.engine.base.blind.host.signal.model_hash
    before = hasher(model)
    need(before == spec["model_hash"], "model identity")
    counts, handles, rows = {}, [], []
    for name, module in (("encoder", model.encoder), ("projector", model.projector),
                         ("action_encoder", model.action_encoder), ("predictor", model.predictor)):
        def hook(_module, args, name=name):
            entry = counts.setdefault(name, {"calls": 0, "samples": 0})
            entry["calls"] += 1
            entry["samples"] += int(args[0].shape[0])
        handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            def frame_encode(ref):
                image = run.coherent.base.first.engine.base.p0._image_tensor(
                    run.coherent.base.frame(ref),
                    transform=run.coherent.base.first.engine.base.p0.image_transform(), device=device)
                z = model.encode({"pixels": image})["emb"]
                need(z.shape == (1, 1, spec["latent_dim"]) and bool(torch.isfinite(z).all()), "encoding")
                return z[0, 0].cpu().numpy().tolist()
            goal = frame_encode(spec["goal"])
            for candidate in spec["candidates"]:
                row = {"arm": candidate["arm"], "prefix": candidate["prefix"],
                       "candidate": candidate["candidate"], "terminal_latent": frame_encode(candidate["frame"])}
                rows.append(row)
                base.write_new(output/(candidate["arm"]+".json"), row)
        torch.cuda.synchronize(device)
        after = hasher(model)
        need(before == after and spec == protocol()
             and counts["encoder"] == counts["projector"] == {"calls": 19, "samples": 19}
             and "predictor" not in counts and "action_encoder" not in counts, "model or ledger")
        return base.write_new(output/"report.json", {"schema": NAME+".report.v1", "contract_sha256": digest,
            "host": host, "model_before": before, "model_after": after, "counts": counts,
            "goal_latent": goal, "rows": rows, "p1_admission": False, "independent_confirmation": False})
    except Exception as exc:
        base.write_new(output/"failure.json", {"type": type(exc).__name__, "message": str(exc)[:1000]})
        raise
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "run"))
    p.add_argument("--tag", required=True)
    p.add_argument("--contract-sha")
    p.add_argument("--authority")
    a = p.parse_args()
    need(a.tag.isascii() and a.tag.isalnum(), "tag")
    contract = ROOT/"results"/f"{NAME}_PREP_{a.tag}.json"
    if a.command == "prepare":
        print(base.write_new(contract, protocol()))
    else:
        with run.coherent.base.first.engine.base.locking.gpu_lock(ROOT/"results/e3_p2_host_cuda_cooperative.lock"):
            print(encode(contract, a.contract_sha, ROOT/"results"/f"{NAME}_RUN_{a.tag}", a.authority))
