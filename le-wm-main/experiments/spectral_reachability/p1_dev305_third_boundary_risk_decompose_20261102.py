"""Decompose terminal risk into model-predicted vs real-RGB latent projection."""
import argparse
import os

import numpy as np

from . import p1_dev305_third_boundary_saved_20261102 as audited

physical, base, need, ROOT = audited.physical, audited.base, audited.need, audited.ROOT
run = physical.planner.run
NAME = "p1_dev305_third_boundary_risk_decompose_20261102"
AUTHORITY = "DEV305_THIRD_RISK_DECOMPOSE_CUDA"


def protocol():
    checked = audited.analyze()
    spec = base.read(ROOT / "results" / f"{physical.NAME}_PREP_DEV01.json", audited.PREP_SHA)
    actual = base.read(ROOT / "results" / f"{physical.NAME}_RUN_DEV01/report.json", audited.REPORT_SHA)
    model_spec = base.read(ROOT / "results" / f"{physical.risk.NAME}_PREP_DEV01.json", physical.RISK_PREP_SHA)
    model = base.read(ROOT / "results" / f"{physical.risk.NAME}_RUN_DEV01/report.json", physical.RISK_REPORT_SHA)
    prediction = base.read(ROOT / "results" / f"{physical.planner.NAME}_PLAN_DEV01/report.json",
                           physical.risk.PLAN_SHA)
    need(len(checked["rows"]) == len(spec["roles"]) == len(actual["rows"]) == 6, "roster")
    directory = ROOT / "results" / f"{physical.NAME}_RUN_DEV01"
    rows = []
    for role, outcome in zip(spec["roles"], actual["rows"], strict=True):
        arm = role["arm"].rsplit("__", 1)[0]
        model_row = next(row for row in model["rows"] if row["arm"] == arm)
        pred_row = next(row for row in prediction["rows"] if row["arm"] == arm)
        index = ("toward_02", "away_02", "zero", "toward_05").index(role["candidate"])
        frame = outcome["observations"][-1]
        ref = {"path": str((directory / frame["file"]).resolve()), "file_sha256": frame["file_sha256"],
               "pixel_sha256": frame["pixel_sha256"]}
        run.coherent.base.frame(ref)
        rows.append({"arm": role["arm"], "candidate": role["candidate"], "selected": role["selected"],
                     "terminal": ref, "jacobian": model_row["jacobian"],
                     "current_goal_delta": model_row["current_goal_delta"],
                     "source_boxes": model_row["source_boxes"],
                     "model_goal_delta": pred_row["predicted_goal_delta"][index],
                     "model_projection": model_row["projections"][index]})
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "source_physical_sha256": audited.REPORT_SHA, "source_risk_sha256": physical.RISK_REPORT_SHA,
            "policy": model_spec["policy"], "model_hash": model_spec["model_hash"],
            "goal": model_spec["goal"], "latent_dim": model_spec["latent_dim"], "rows": rows,
            "scope": "post-outcome same actions, true terminal RGB used for diagnostic only, no selection",
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
    need(before == spec["model_hash"], "model")
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
                pixels = run.coherent.base.frame(ref)
                image = run.coherent.base.first.engine.base.p0._image_tensor(
                    pixels, transform=run.coherent.base.first.engine.base.p0.image_transform(), device=device)
                value = model.encode({"pixels": image})["emb"]
                need(tuple(value.shape) == (1, 1, spec["latent_dim"])
                     and bool(torch.isfinite(value).all()), "encoding")
                return value[0, 0]
            goal = encode(spec["goal"])
            rows = []
            for item in spec["rows"]:
                actual_delta = (encode(item["terminal"]) - goal).cpu().numpy()
                prediction = np.asarray(item["model_goal_delta"], dtype=np.float64)
                projection = physical.risk.engine.risk.predict_screen(
                    item["jacobian"], np.repeat(actual_delta[None, :], 4, axis=0),
                    item["current_goal_delta"], item["source_boxes"])[0]
                actual_screen = physical.risk.engine.risk.visibility.inspect(run.coherent.base.frame(item["terminal"]))
                rows.append({"arm": item["arm"], "candidate": item["candidate"], "selected": item["selected"],
                             "real_goal_delta": actual_delta.tolist(),
                             "prediction_error_mse": float(np.mean((prediction - actual_delta)**2)),
                             "real_latent_projection": projection,
                             "actual_boxes": {key: actual_screen[key]["bbox"] for key in ("object", "pusher")},
                             "actual_visible": actual_screen["eligible"]})
        torch.cuda.synchronize(device)
        after = hasher(model)
        need(before == after and spec == protocol(), "source and frozen model")
        need(counts == {"encoder": {"calls": 7, "samples": 7},
                        "projector": {"calls": 7, "samples": 7}}, "ledger")
        return base.write_new(output / "report.json", {
            "schema": NAME + ".report.v1", "contract_sha256": digest, "model_before": before,
            "model_after": after, "counts": counts, "host": host, "rows": rows,
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
