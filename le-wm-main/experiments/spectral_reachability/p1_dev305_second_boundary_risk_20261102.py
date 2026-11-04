"""Project model-predicted terminal RGB risk at the second five-step boundary."""
import argparse
import os

import numpy as np

from . import p1_dev305_contact_second_boundary_20261102 as source
from . import p1_dev305_contact_risk_projection_20261102 as risk

base, need, run, ROOT = source.base, source.need, source.run, source.ROOT
NAME = "p1_dev305_second_boundary_risk_20261102"
AUTHORITY = "DEV305_SECOND_BOUNDARY_RISK_CUDA"
PREP_SHA = "92c482359a71619992f699b04f044e4e09fa4e46c19d62cc8c2e4545a7ca09d5"
PLAN_SHA = "756f8233062c24bdd5e8e3c82b97bb432ac314a4ec948c35621695bf24cfe256"


def protocol():
    spec = base.read(ROOT / "results" / f"{source.NAME}_PREP_DEV01.json", PREP_SHA)
    plan = base.read(ROOT / "results" / f"{source.NAME}_PLAN_DEV01/report.json", PLAN_SHA)
    need(spec == source.protocol() and plan["contract_sha256"] == PREP_SHA
         and plan["model_before"] == plan["model_after"] == spec["model_hash"], "source")
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "projection_sha256": base.sha(risk.__file__), "tangent_sha256": base.sha(risk.tangent.__file__),
            "plan_sha256": PLAN_SHA, "model_contract_sha256": PREP_SHA,
            "policy": spec["policy"], "model_hash": spec["model_hash"], "goal": spec["goal"],
            "latent_dim": spec["latent_dim"], "roles": spec["roles"],
            "rule": "unchanged ridge .01 smax squared; same RGB box translation and 8px margin; select lowest predicted latent cost among predicted-visible, otherwise abstain",
            "scope": "post-outcome known-root development, uncalibrated terminal heuristic, not safety certification",
            "p1_admission": False, "independent_confirmation": False}


def execute(path, digest, output, authority):
    need(authority == AUTHORITY and base.read(path, digest) == protocol()
         and not os.path.lexists(output), "contract")
    spec = base.read(path, digest)
    plan = base.read(ROOT / "results" / f"{source.NAME}_PLAN_DEV01/report.json", PLAN_SHA)
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
            def encode(pixels):
                image = run.coherent.base.first.engine.base.p0._image_tensor(
                    pixels, transform=run.coherent.base.first.engine.base.p0.image_transform(), device=device)
                val = model.encode({"pixels": image})["emb"]
                need(tuple(val.shape) == (1, 1, spec["latent_dim"]) and bool(torch.isfinite(val).all()), "encoding")
                return val[0, 0]

            goal = encode(run.coherent.base.frame(spec["goal"]))
            rows = []
            for role, predicted in zip(spec["roles"], plan["rows"], strict=True):
                need(role["arm"] == predicted["arm"], "role")
                pixels = run.coherent.base.frame(role["images"][-1])
                screen = risk.visibility.inspect(pixels)
                need(screen["eligible"], "visibility")
                jac = np.stack([((encode(risk.tangent.perturb(pixels, axis, 1)) -
                                  encode(risk.tangent.perturb(pixels, axis, -1))) / 2).cpu().numpy()
                                for axis in risk.tangent.AXES], axis=1)
                current = (encode(pixels) - goal).cpu().numpy()
                boxes = {key: screen[key]["bbox"] for key in ("object", "pusher")}
                projections = risk.predict_screen(jac, predicted["predicted_goal_delta"], current, boxes)
                safe = [i for i, value in enumerate(projections) if value["predicted_visible"]]
                selected = min(safe, key=lambda i: (predicted["scores"][i], i)) if safe else None
                rows.append({"arm": role["arm"], "jacobian": jac.tolist(), "current_goal_delta": current.tolist(),
                             "source_boxes": boxes, "projections": projections, "selected_index": selected})
        torch.cuda.synchronize(device)
        after = hasher(model)
        need(before == after and spec == protocol(), "frozen sources")
        need(counts == {"encoder": {"calls": 34, "samples": 34},
                        "projector": {"calls": 34, "samples": 34}}, "ledger")
        return base.write_new(output / "report.json", {
            "schema": NAME + ".report.v1", "contract_sha256": digest, "model_before": before,
            "model_after": after, "counts": counts, "host": host, "rows": rows,
            "p1_admission": False, "independent_confirmation": False})
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
