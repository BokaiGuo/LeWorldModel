"""Encode the sealed real frame at every step against the same five-step forecast.

Earlier frames are temporal diagnostics, not predictions at those times.
"""
import argparse
import os

import numpy as np

from . import p1_dev305_contact_risk_projection_saved_20261102 as saved

base, need, ROOT = saved.base, saved.need, saved.ROOT
run = saved.source.run
NAME = "p1_dev305_contact_time_alignment_20261102"
AUTHORITY = "DEV305_CONTACT_TIME_ALIGNMENT_CUDA"
LABELS = ("toward_02", "away_02", "zero", "toward_05")


def protocol():
    # Verify the previous model/physical pairing and candidate byte identity.
    validated = saved.analyze()
    spec = base.read(ROOT / "results" / f"{saved.source.NAME}_PREP_DEV01.json", saved.PREP_SHA)
    plan = base.read(ROOT / "results" / f"{saved.source.source.NAME}_PLAN_DEV01/report.json",
                     saved.source.PLAN_SHA)
    physical = base.read(ROOT / "results" / f"{saved.real.source.NAME}_RUN_DEV01/report.json",
                         saved.real.REPORT_SHA)
    directory = ROOT / "results" / f"{saved.real.source.NAME}_RUN_DEV01"
    available = {row["arm"]: row for row in physical["rows"]}
    need(len(available) == len(validated["rows"]) == 12, "physical roster")
    rows = []
    for role, pred in zip(spec["roles"], plan["rows"], strict=True):
        need(role["arm"] == pred["arm"], "model role")
        for index, label in enumerate(LABELS):
            key = role["arm"] + "__" + label
            item = available[key]
            need(item["status"] == "fixed_horizon_completed"
                 and len(item["observations"]) == 6 and len(item["transitions"]) == 5,
                 "physical completion")
            frames = []
            for step, frame in enumerate(item["observations"][1:], 1):
                need(frame["step"] == item["transitions"][step - 1]["step"] == step,
                     "step alignment")
                ref = {"path": str((directory / frame["file"]).resolve()),
                       "file_sha256": frame["file_sha256"], "pixel_sha256": frame["pixel_sha256"]}
                run.coherent.base.frame(ref)
                frames.append(ref)
            rows.append({"arm": role["arm"], "candidate": label,
                         "predicted_goal_delta": pred["predicted_goal_delta"][index],
                         "frames": frames})
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "model_hash": spec["model_hash"], "policy": spec["policy"], "goal": spec["goal"],
            "latent_dim": spec["latent_dim"], "rows": rows,
            "physical_sha256": saved.real.REPORT_SHA, "model_prediction_sha256": saved.source.PLAN_SHA,
            "scope": "post-outcome fixed five-step model forecasts versus same-action real frames at steps 1-5; early comparisons are not early-horizon model predictions",
            "p1_admission": False, "independent_confirmation": False}


def execute(path, digest, output, authority):
    need(authority == AUTHORITY and base.read(path, digest) == protocol()
         and not os.path.lexists(output), "contract")
    spec = base.read(path, digest)
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
    need(before == spec["model_hash"], "model hash")
    counts, handles = {}, []
    for label, module in (("encoder", model.encoder), ("projector", model.projector)):
        def hook(_module, args, label=label):
            entry = counts.setdefault(label, {"calls": 0, "samples": 0})
            entry["calls"] += 1
            entry["samples"] += int(args[0].shape[0])
        handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            def encode(ref):
                image = run.coherent.base.first.engine.base.p0._image_tensor(
                    run.coherent.base.frame(ref),
                    transform=run.coherent.base.first.engine.base.p0.image_transform(), device=device)
                value = model.encode({"pixels": image})["emb"]
                need(tuple(value.shape) == (1, 1, spec["latent_dim"])
                     and bool(torch.isfinite(value).all()), "encoding")
                return value[0, 0]

            goal = encode(spec["goal"])
            rows = []
            for role in spec["rows"]:
                predicted = np.asarray(role["predicted_goal_delta"], dtype=np.float64)
                need(predicted.shape == (spec["latent_dim"],) and np.isfinite(predicted).all(),
                     "predicted latent")
                steps = []
                for step, ref in enumerate(role["frames"], 1):
                    delta = (encode(ref) - goal).cpu().numpy().astype(np.float64)
                    steps.append({"step": step, "real_goal_delta": delta.tolist(),
                                  "real_goal_mse": float(np.mean(delta ** 2)),
                                  "forecast_vs_real_mse": float(np.mean((predicted - delta) ** 2))})
                rows.append({"arm": role["arm"], "candidate": role["candidate"], "steps": steps})
        torch.cuda.synchronize(device)
        after = hasher(model)
        need(before == after and counts == {
            "encoder": {"calls": 61, "samples": 61}, "projector": {"calls": 61, "samples": 61}},
            "frozen model and counts")
        need(spec == protocol(), "source changed")
        return base.write_new(output / "report.json", {
            "schema": NAME + ".report.v1", "contract_sha256": digest, "host": host,
            "model_before": before, "model_after": after, "counts": counts, "rows": rows,
            "scope": spec["scope"], "p1_admission": False, "independent_confirmation": False})
    except Exception as exc:
        base.write_new(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)[:1000]})
        raise
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--contract-sha")
    parser.add_argument("--authority")
    args = parser.parse_args()
    need(args.tag.isascii() and args.tag.isalnum(), "tag")
    path = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    if args.command == "prepare":
        print(base.write_new(path, protocol()))
    else:
        with run.coherent.base.first.engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            print(execute(path, args.contract_sha, ROOT / "results" / f"{NAME}_RUN_{args.tag}", args.authority))
