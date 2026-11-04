"""Frozen-model selection among a fixed RGB-derived contact packet bank."""
import argparse
import os

import numpy as np

from . import p1_dev305_selected_bank_first_feedback_20261102 as original

base, need, run, ROOT = original.base, original.need, original.run, original.ROOT
NAME = "p1_dev305_contact_model_select_20261102"
AUTHORITY = "DEV305_CONTACT_MODEL_SELECT_CUDA"
SOURCE_SHA = "2e5463199b1924c89bf0347c9e9a60f936c80c51d6cff217d4516f3a2916cfb2"


def packets(direction):
    direction = np.asarray(direction, dtype=np.float64)
    if direction.shape != (2,) or not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction), 1, atol=1e-5):
        raise ValueError("unit RGB direction required")
    return np.stack([np.repeat(np.asarray(direction * scale, dtype="<f4")[None], 5, axis=0)
                     for scale in (.2, -.2, 0., .5)])


def protocol():
    spec = base.read(ROOT / "results" / f"{original.NAME}_PREP_DEV01.json", SOURCE_SHA)
    need(spec == original.protocol(), "source")
    roles = []
    for role in spec["roles"]:
        pixels = run.coherent.base.frame(role["images"][-1])
        need(original.visibility.inspect(pixels)["eligible"], "current visibility")
        geometry = run.coherent.base.first.engine._geometry(pixels)
        need(np.allclose(geometry["contact_direction_raw_xy"], role["contact_direction_raw_xy"], atol=1e-12), "direction")
        roles.append({"arm": role["arm"], "images": role["images"], "actual_history": role["actual_history"],
                      "actions": packets(role["contact_direction_raw_xy"]).tolist()})
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__), "source_sha256": SOURCE_SHA,
            "policy": spec["policy"], "model_hash": spec["model_hash"], "goal": spec["goal"],
            "action_mean": spec["action_mean"], "action_std": spec["action_std"],
            "latent_dim": spec["latent_dim"], "roles": roles,
            "bank_order": ["toward_02", "away_02", "zero", "toward_05"],
            "selection": "lowest one-packet predicted goal latent MSE; stable first index tie",
            "scope": "post-outcome known-root development; no simulator state in selector; goal visibility not certified",
            "p1_admission": False, "independent_confirmation": False}


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
    need(before == spec["model_hash"], "model")
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
            def encode(ref):
                image = run.coherent.base.first.engine.base.p0._image_tensor(
                    run.coherent.base.frame(ref), transform=run.coherent.base.first.engine.base.p0.image_transform(), device=device)
                val = model.encode({"pixels": image})["emb"]
                need(tuple(val.shape) == (1, 1, spec["latent_dim"]) and bool(torch.isfinite(val).all()), "encoding")
                return val[0, 0]
            tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
            mean, std = tensor(spec["action_mean"]), tensor(spec["action_std"])
            need(bool(torch.isfinite(std).all()) and bool((std > 0).all()), "normalizer")
            goal = encode(spec["goal"])
            rows = []
            for role in spec["roles"]:
                z = torch.stack([encode(x) for x in role["images"]])[None].repeat(4, 1, 1)
                history = ((tensor(role["actual_history"]).reshape(2, 5, 2) - mean) / std).reshape(1, 2, 10).repeat(4, 1, 1)
                actions = tensor(role["actions"])
                context = torch.cat((history, ((actions - mean) / std).reshape(4, 1, 10)), dim=1)
                pred = model.predict(z, model.action_encoder(context))
                need(pred.shape == z.shape and bool(torch.isfinite(pred).all()), "prediction")
                delta = pred[:, -1] - goal
                scores = delta.square().mean(dim=1).cpu().tolist()
                rows.append({"arm": role["arm"], "scores": scores, "selected_index": int(np.argmin(scores)),
                             "predicted_goal_delta": delta.cpu().tolist()})
        torch.cuda.synchronize(device)
        after = hasher(model)
        need(before == after and spec == protocol(), "source frozen")
        need(counts == {"encoder": {"calls": 10, "samples": 10}, "projector": {"calls": 10, "samples": 10},
                        "action_encoder": {"calls": 3, "samples": 12}, "predictor": {"calls": 3, "samples": 12}}, "ledger")
        return base.write_new(output / "report.json", {"schema": NAME + ".report.v1", "contract_sha256": digest,
             "model_before": before, "model_after": after, "host": host, "counts": counts, "rows": rows,
             "p1_admission": False, "independent_confirmation": False})
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
