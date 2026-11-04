"""Same-action open versus observed-midpoint prediction at fixed dev305 endpoints."""
import argparse
import os
from pathlib import Path

import numpy as np

from . import p1_dev305_goalrank_real_rgb_saved_20261102 as prior

source, base = prior.source, prior.base
physical, run = source.physical, source.run
ROOT = source.ROOT
NAME = "p1_dev305_refresh_diagnostic_20261102"
AUTHORITY = "DEV305_REFRESH_DIAGNOSTIC_CUDA"
need = run.require


def protocol():
    prior.analyze()
    original = base.read(ROOT / f"results/{physical.model_run.NAME}_PREP_DEV01.json",
                         physical.CONTRACT_SHA)
    model = base.read(ROOT / f"results/{physical.model_run.NAME}_RUN_DEV01/report.json",
                      physical.MODEL_SHA)
    execution = base.read(source.PHYSICAL_RUN / "report.json", prior.saved.REPORT_SHA)
    encoded = base.read(prior.RUN / "report.json", prior.REPORT_SHA)
    rows = []
    for result, actual in zip(execution["rows"], encoded["rows"], strict=True):
        role = next(item for item in original["roles"] if item["arm"] == result["prefix"])
        model_row = next(item for item in model["rows"] if item["arm"] == result["prefix"])
        item = model_row["candidates"][result["candidate"]]
        need(result["arm"] == actual["arm"] and result["status"] == "fixed_horizon_completed",
             "paired endpoint")
        frame = next(x for x in result["observations"] if x["phase"] == "tail" and x["step"] == 5)
        reference = {"path": str((source.PHYSICAL_RUN / frame["file"]).resolve()),
                     "file_sha256": frame["file_sha256"], "pixel_sha256": frame["pixel_sha256"]}
        actions = np.asarray(item["actions"], dtype="<f4")
        prefix = np.asarray(role["actual_history"], dtype="<f4")[5:]
        need(actions.shape == (2, 5, 2) and prefix.shape == (5, 2)
             and all(np.array_equal(t["action"], a)
                     for t, a in zip(result["transitions"], np.concatenate((prefix, actions.reshape(10, 2))),
                                     strict=True)), "physical action history")
        rows.append({"arm": result["arm"], "prefix": result["prefix"], "candidate": result["candidate"],
                     "images": [role["images"][1], role["images"][2], reference],
                     "history": np.concatenate((prefix, actions[0])).tolist(), "action": actions[1].tolist(),
                     "open_terminal": item["terminal"], "true_terminal": actual["terminal_latent"]})
    need(len(rows) == 18, "complete roster")
    return {"schema": NAME+".protocol.v1", "code_sha256": base.sha(__file__),
            "source_report_sha256": prior.REPORT_SHA, "model_report_sha256": physical.MODEL_SHA,
            "physical_report_sha256": prior.saved.REPORT_SHA, "model_hash": original["model_hash"],
            "policy": original["policy"], "action_mean": original["action_mean"],
            "action_std": original["action_std"], "latent_dim": original["latent_dim"],
            "goal_latent": encoded["goal_latent"], "rows": rows,
            "comparison": "fixed open two-packet prediction versus same second action after real intermediate RGB",
            "classification": "development_after_outcome_not_independent",
            "independent_confirmation": False, "full_cost_certified": False}


def execute(path, digest, output, authority):
    need(authority == AUTHORITY and base.read(path, digest) == protocol()
         and not os.path.lexists(output), "authority and contract")
    spec = base.read(path, digest)
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
    need(before == spec["model_hash"], "frozen checkpoint")
    counts, handles = {}, []
    for label, module in (("encoder", model.encoder), ("projector", model.projector),
                          ("action_encoder", model.action_encoder), ("predictor", model.predictor)):
        def hook(_module, args, label=label):
            value = counts.setdefault(label, {"calls": 0, "samples": 0})
            value["calls"] += 1
            value["samples"] += int(args[0].shape[0])
        handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            def encode(ref):
                image = run.coherent.base.first.engine.base.p0._image_tensor(
                    run.coherent.base.frame(ref),
                    transform=run.coherent.base.first.engine.base.p0.image_transform(), device=device)
                value = model.encode({"pixels": image})["emb"]
                need(tuple(value.shape) == (1, 1, spec["latent_dim"]) and bool(torch.isfinite(value).all()),
                     "encoded frame")
                return value[0, 0]
            tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
            mean, std = tensor(spec["action_mean"]), tensor(spec["action_std"])
            need(bool(torch.isfinite(std).all()) and bool((std > 0).all()), "normalizer")
            observations = torch.stack([torch.stack([encode(frame) for frame in row["images"]])
                                        for row in spec["rows"]])
            histories = tensor([row["history"] for row in spec["rows"]]).reshape(18, 2, 5, 2)
            actions = tensor([row["action"] for row in spec["rows"]]).reshape(18, 1, 5, 2)
            tokens = ((torch.cat((histories, actions), dim=1)-mean)/std).reshape(18, 3, 10)
            prediction = model.predict(observations, model.action_encoder(tokens))
            need(prediction.shape == observations.shape and bool(torch.isfinite(prediction).all()), "prediction")
            refreshed = prediction[:, -1].cpu().numpy()
        torch.cuda.synchronize(device)
        after = hasher(model)
        need(before == after and spec == protocol(), "model and source")
        need(counts == {"encoder": {"calls": 54, "samples": 54},
                        "projector": {"calls": 54, "samples": 54},
                        "action_encoder": {"calls": 1, "samples": 18},
                        "predictor": {"calls": 1, "samples": 18}}, "call ledger")
        rows = [{"arm": row["arm"], "refreshed_terminal": z.tolist()}
                for row, z in zip(spec["rows"], refreshed, strict=True)]
        return base.write_new(output/"report.json", {"schema": NAME+".report.v1",
            "contract_sha256": digest, "host": host, "model_before": before, "model_after": after,
            "counts": counts, "rows": rows, "independent_confirmation": False, "full_cost_certified": False})
    except Exception as exc:
        base.write_new(output/"failure.json", {"type": type(exc).__name__, "message": str(exc)[:1000]})
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
    contract = ROOT/"results"/f"{NAME}_PREP_{args.tag}.json"
    if args.command == "prepare":
        print(base.write_new(contract, protocol()))
    else:
        with run.coherent.base.first.engine.base.locking.gpu_lock(ROOT/"results/e3_p2_host_cuda_cooperative.lock"):
            print(execute(contract, args.contract_sha, ROOT/"results"/f"{NAME}_RUN_{args.tag}", args.authority))
