"""Compare one-image and three-real-frame predictions for the same executed actions."""
import argparse
import os
from pathlib import Path

import numpy as np

from . import p1_dev305_compass_model_compare_20261102 as prior

base, need, run, ROOT = prior.base, prior.need, prior.run, prior.ROOT
NAME = "p1_dev305_real_history_contrast_20261102"
AUTHORITY = "DEV305_REAL_HISTORY_CONTRAST_CUDA"
PRIOR_CONTRACT_SHA = "cbb740dac63a72016b222e83b029def2d783e0ec6cd9183e2ecc237d88743339"
PRIOR_REPORT_SHA = "8386fdd538665362a0c476c757d45d70fbc779b091ac3a53128294400e755b96"


def protocol():
    original = base.read(ROOT / f"results/{prior.NAME}_PREP_DEV01.json", PRIOR_CONTRACT_SHA)
    report = base.read(ROOT / f"results/{prior.NAME}_RUN_DEV01/report.json", PRIOR_REPORT_SHA)
    need(original == prior.protocol() and report["contract_sha256"] == PRIOR_CONTRACT_SHA,
         "source contract")
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "source_contract_sha256": PRIOR_CONTRACT_SHA, "source_report_sha256": PRIOR_REPORT_SHA,
            "model_hash": original["model_hash"], "policy": original["policy"], "goal": original["goal"],
            "action_mean": original["action_mean"], "action_std": original["action_std"],
            "latent_dim": original["latent_dim"], "roles": original["roles"],
            "comparison": "three measured frames and two measured action packets versus last frame only; "
                          "same two action packets and same real endpoint",
            "classification": "post_outcome_development_no_independent_inference",
            "p1_admission": False}


def execute(path, digest, output, authority):
    need(authority == AUTHORITY and base.read(path, digest) == protocol() and not os.path.lexists(output),
         "contract or authority")
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
    need(before == spec["model_hash"], "model identity")
    counts, handles = {}, []
    for label, module in (("encoder", model.encoder), ("projector", model.projector),
                          ("action_encoder", model.action_encoder), ("predictor", model.predictor)):
        def hook(_module, args, label=label):
            entry = counts.setdefault(label, {"calls": 0, "samples": 0})
            entry["calls"] += 1
            entry["samples"] += int(args[0].shape[0])
        handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
            def encode(ref):
                image = run.coherent.base.first.engine.base.p0._image_tensor(
                    run.coherent.base.frame(ref),
                    transform=run.coherent.base.first.engine.base.p0.image_transform(), device=device)
                value = model.encode({"pixels": image})["emb"]
                need(tuple(value.shape) == (1, 1, spec["latent_dim"]) and bool(torch.isfinite(value).all()),
                     "encoding")
                return value[0, 0]
            goal = encode(spec["goal"])
            mean, std = tensor(spec["action_mean"]), tensor(spec["action_std"])
            need(bool(torch.isfinite(std).all()) and bool((std > 0).all()), "normalizer")
            rows = []
            for role in spec["roles"]:
                frames = torch.stack([encode(f) for f in role["images"]])[None].repeat(18, 1, 1)
                history = tensor(role["actual_history"]).reshape(1, 2, 5, 2).repeat(18, 1, 1, 1)
                past = ((history-mean)/std).reshape(18, 2, 10)
                actions = tensor([r["actions"] for r in role["rows"]]).reshape(18, 2, 5, 2)
                actual = torch.stack([encode(r["frame"]) for r in role["rows"]])
                predictions = {}
                for mode in ("real_history", "single_frame"):
                    z = frames if mode == "real_history" else frames[:, -1:]
                    action_history = past if mode == "real_history" else past[:, :0]
                    for k in range(2):
                        token = ((actions[:, k]-mean)/std).reshape(18, 1, 10)
                        context = torch.cat((action_history, token), dim=1)
                        predicted = model.predict(z, model.action_encoder(context))
                        need(predicted.shape == z.shape and bool(torch.isfinite(predicted).all()), "prediction")
                        z = torch.cat((z, predicted[:, -1:]), dim=1)
                        z = z[:, -3:] if mode == "real_history" else z[:, -1:]
                        action_history = context[:, -2:] if mode == "real_history" else context[:, :0]
                    predictions[mode] = z[:, -1]
                for i, item in enumerate(role["rows"]):
                    result = {"arm": role["arm"], "name": item["name"], "actions_sha256": item["actions_sha256"],
                              "true_goal_mse": float((actual[i]-goal).square().mean().cpu())}
                    for mode in predictions:
                        z = predictions[mode][i]
                        result[mode] = {"prediction_mse": float((z-actual[i]).square().mean().cpu()),
                                        "goal_mse": float((z-goal).square().mean().cpu())}
                    rows.append(result)
        torch.cuda.synchronize(device)
        after = hasher(model)
        need(before == after and spec == protocol(), "postrun source")
        need(counts == {"encoder": {"calls": 64, "samples": 64}, "projector": {"calls": 64, "samples": 64},
                        "action_encoder": {"calls": 12, "samples": 216},
                        "predictor": {"calls": 12, "samples": 216}}, "ledger")
        return base.write_new(output / "report.json", {"schema": NAME + ".report.v1",
            "contract_sha256": digest, "host": host, "model_before": before, "model_after": after,
            "counts": counts, "rows": rows, "p1_admission": False})
    except Exception as exc:
        base.write_new(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
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
    contract = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    if args.command == "prepare":
        print(base.write_new(contract, protocol()))
    else:
        with run.coherent.base.first.engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            print(execute(contract, args.contract_sha, ROOT / "results" / f"{NAME}_RUN_{args.tag}", args.authority))
