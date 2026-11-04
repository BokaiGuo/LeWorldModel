"""Replan equally from three real RGB midpoints of previously planned CEM tails."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

from . import p1_dev305_compass_midrgb_model_20261102 as prior

base, run, ROOT, need = prior.base, prior.run, prior.ROOT, prior.need
NAME = "p1_dev305_triprefix_midfeedback_20261102"
AUTHORITY = "DEV305_TRIPREFIX_MIDFEEDBACK_CUDA"
PREVIOUS_CONTRACT_SHA = "cc8fa593bdf176cda58edcca63861d9ff7c223b2809960c45a8fe04e053bcb91"


def protocol():
    sealed = base.read(ROOT / f"results/{prior.NAME}_PREP_DEV01.json", PREVIOUS_CONTRACT_SHA)
    need(sealed == prior.protocol(), "source contract")
    roles = []
    for role in sealed["roles"]:
        row = next(item for item in role["rows"] if item["name"] == "cem_selected")
        for image in row["images"]:
            run.coherent.base.frame(image)
        geometry = run.coherent.base.first.engine._geometry(run.coherent.base.frame(row["images"][-1]))
        history = np.asarray(row["history"], dtype="<f4")
        need(history.shape == (10, 2), "actual history")
        roles.append({"arm": role["arm"], "images": row["images"], "actual_history": history.tolist(),
                      "prefix_sha256": hashlib.sha256(history.tobytes()).hexdigest(),
                      "contact_direction_raw_xy": geometry["contact_direction_raw_xy"]})
    need(len(roles) == 3, "role roster")
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "source_contract_sha256": PREVIOUS_CONTRACT_SHA, "policy": sealed["policy"],
            "model_hash": sealed["model_hash"], "action_mean": sealed["action_mean"],
            "action_std": sealed["action_std"], "latent_dim": sealed["latent_dim"], "goal": sealed["goal"],
            "roles": roles, "optimizer": "same CEM seed, two-packet terminal score, execute first packet",
            "per_arm": {"predict_calls": 10, "predict_samples": 3200},
            "classification": "known_root_post_outcome_development",
            "p1_admission": False, "independent_confirmation": False, "full_cost_certified": False}


def plan(path, digest, output, authority):
    need(authority == AUTHORITY and base.read(path, digest) == protocol() and not os.path.lexists(output), "contract")
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
    need(before == spec["model_hash"], "model identity")
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    mean, std = tensor(spec["action_mean"]), tensor(spec["action_std"])
    need(bool(torch.isfinite(std).all()) and bool((std > 0).all()), "normalizer")
    counts, phase, handles, rows = {}, {"value": "encode"}, [], []
    for label, module in (("encoder", model.encoder), ("projector", model.projector),
                          ("action_encoder", model.action_encoder), ("predictor", model.predictor)):
        def hook(_module, args, label=label):
            entry = counts.setdefault(phase["value"] + ":" + label, {"calls": 0, "samples": 0})
            entry["calls"] += 1
            entry["samples"] += int(args[0].shape[0])
        handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            def encode(ref):
                image = run.coherent.base.first.engine.base.p0._image_tensor(
                    run.coherent.base.frame(ref), transform=run.coherent.base.first.engine.base.p0.image_transform(),
                    device=device)
                value = model.encode({"pixels": image})["emb"]
                need(tuple(value.shape) == (1, 1, spec["latent_dim"]) and bool(torch.isfinite(value).all()), "encoding")
                return value[0, 0]
            goal = encode(spec["goal"])[None]
            for role in spec["roles"]:
                phase["value"] = "encode"
                z0 = torch.stack([encode(ref) for ref in role["images"]])[None]
                history = ((tensor(role["actual_history"]).reshape(2, 5, 2) - mean) / std).reshape(1, 2, 10)
                phase["value"] = role["arm"]
                def score(variants):
                    flat = tensor(variants.reshape(320, 2, 5, 2))
                    z, past = z0.repeat(320, 1, 1), history.repeat(320, 1, 1)
                    for packet in range(2):
                        context = torch.cat((past, ((flat[:, packet] - mean) / std).reshape(320, 1, 10)), dim=1)
                        predicted = model.predict(z, model.action_encoder(context))
                        need(predicted.shape == z.shape and bool(torch.isfinite(predicted).all()), "predict")
                        z = torch.cat((z, predicted[:, -1:]), dim=1)[:, -3:]
                        past = context[:, -2:]
                    return (z[:, -1] - goal).square().mean(dim=1).reshape(64, 5).cpu().numpy()
                result = run.diverse.plan(role["contact_direction_raw_xy"], score, structured=False,
                                          common=run.coherent.base.first.engine.base.common,
                                          seed=run.coherent.base.first.planner.parent.SEED)
                actions = np.asarray(result["selected"]["actions"], dtype="<f4")
                need(actions.shape == (2, 5, 2) and np.isfinite(actions).all() and (np.abs(actions) <= 1).all(),
                     "selected plan")
                rows.append({"arm": role["arm"], "prefix_sha256": role["prefix_sha256"],
                             "selected_actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(), **result})
        torch.cuda.synchronize(device)
        after = hasher(model)
        need(before == after and spec == protocol(), "frozen source")
        for label in ("encoder", "projector"):
            need(counts["encode:"+label] == {"calls": 10, "samples": 10}, "encoding ledger")
        for role in spec["roles"]:
            for label in ("action_encoder", "predictor"):
                need(counts[role["arm"]+":"+label] == {"calls": 10, "samples": 3200}, "prediction ledger")
        return base.write_new(output / "report.json", {"schema": NAME + ".report.v1",
            "contract_sha256": digest, "host": host, "model_before": before, "model_after": after,
            "ledger": counts, "rows": rows, "physical_execution_completed": False,
            "p1_admission": False, "independent_confirmation": False, "full_cost_certified": False})
    except Exception as exc:
        base.write_new(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)[:1000]})
        raise
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "plan"))
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
            print(plan(path, args.contract_sha, ROOT / "results" / f"{NAME}_PLAN_{args.tag}", args.authority))
