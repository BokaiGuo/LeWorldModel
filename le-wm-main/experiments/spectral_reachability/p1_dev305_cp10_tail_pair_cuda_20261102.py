"""Post-outcome same-context one-packet prediction of old and replanned tails."""
import argparse
import os
from pathlib import Path

import numpy as np

from . import p1_dev305_spectral_cp10_saved_20261102 as verified

physical, planner = verified.physical, verified.planner
base, run, need = physical.base, physical.run, physical.need
ROOT = physical.ROOT
NAME = "p1_dev305_cp10_tail_pair_cuda_20261102"
AUTHORITY = "DEV305_CP10_TAIL_PAIR_CUDA"


def protocol():
    verified.analyze()
    spec = base.read(ROOT/f"results/{planner.NAME}_PREP_DEV01.json", physical.PREP_SHA)
    physical_spec = base.read(ROOT/f"results/{physical.NAME}_PREP_DEV01.json", verified.CONTRACT_SHA)
    roles = []
    for role, dispatched in zip(spec["roles"], physical_spec["roles"], strict=True):
        need(role["arm"] == dispatched["arm"], "role binding")
        need(role["arm"].startswith("goal_"), "source panel")
        panel = planner.goal_saved
        prior = base.read(panel.ROOT/f"{panel.physical.NAME}_RUN_DEV01/report.json", panel.REPORT_SHA)
        row = next(r for r in prior["rows"] if r["arm"] == role["source_arm"])
        need(row["status"] == "fixed_horizon_completed" and len(row["transitions"]) == 15, "old tail")
        old_tail = np.asarray([t["action"] for t in row["transitions"][10:]], dtype="<f4")
        new_tail = np.asarray(dispatched["tail"], dtype="<f4")
        need(old_tail.shape == new_tail.shape == (5, 2) and np.isfinite(old_tail).all()
             and np.isfinite(new_tail).all(), "action shapes")
        roles.append({"arm": role["arm"], "images": role["images"],
                      "actual_history": role["actual_history"], "old": old_tail.tolist(),
                      "new": new_tail.tolist(), "source_arm": role["source_arm"],
                      "source_physical_sha256": panel.REPORT_SHA})
    return {"schema": NAME+".protocol.v1", "code_sha256": base.sha(__file__),
            "classification": "post_outcome_known_root_no_selection",
            "source_contract_sha256": verified.CONTRACT_SHA, "source_report_sha256": verified.REPORT_SHA,
            "policy": spec["policy"], "model_hash": spec["model_hash"], "goal": spec["goal"],
            "action_mean": spec["action_mean"], "action_std": spec["action_std"],
            "latent_dim": spec["latent_dim"], "roles": roles,
            "objective": "one executed 5-action packet predicted latent goal MSE",
            "p1_admission": False, "independent_confirmation": False}


def run_cuda(contract_path, digest, output, authority):
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
                    run.coherent.base.frame(ref), transform=run.coherent.base.first.engine.base.p0.image_transform(),
                    device=device)
                z = model.encode({"pixels": image})["emb"]
                need(tuple(z.shape) == (1, 1, spec["latent_dim"]) and bool(torch.isfinite(z).all()), "RGB")
                return z[0, 0]
            goal = encode(spec["goal"])
            mean = torch.as_tensor(spec["action_mean"], dtype=torch.float32, device=device)
            std = torch.as_tensor(spec["action_std"], dtype=torch.float32, device=device)
            need(bool(torch.isfinite(std).all()) and bool((std > 0).all()), "normalizer")
            for role in spec["roles"]:
                z = torch.stack([encode(ref) for ref in role["images"]])[None].repeat(2, 1, 1)
                history = torch.as_tensor(role["actual_history"], dtype=torch.float32, device=device).reshape(1, 2, 5, 2)
                history = ((history-mean)/std).reshape(1, 2, 10).repeat(2, 1, 1)
                packet = torch.as_tensor([role["old"], role["new"]], dtype=torch.float32, device=device)
                action = ((packet-mean)/std).reshape(2, 1, 10)
                predicted = model.predict(z, model.action_encoder(torch.cat((history, action), dim=1)))
                need(predicted.shape == z.shape and bool(torch.isfinite(predicted).all()), "prediction")
                latents = predicted[:, -1]
                score = (latents-goal).square().mean(dim=1)
                row = {"arm": role["arm"], "old_score": float(score[0].item()), "new_score": float(score[1].item()),
                       "old_latent": latents[0].cpu().tolist(), "new_latent": latents[1].cpu().tolist()}
                base.write_new(output/(role["arm"]+".json"), row)
                rows.append(row)
        torch.cuda.synchronize(device)
        after = hasher(model)
        need(before == after and protocol() == spec and counts["encoder"] == counts["projector"]
             == {"calls": 7, "samples": 7}
             and counts["predictor"] == counts["action_encoder"] == {"calls": 2, "samples": 4},
             "model, source, or ledger")
        return base.write_new(output/"report.json", {"schema": NAME+".report.v1", "contract_sha256": digest,
            "host": host, "model_before": before, "model_after": after, "counts": counts, "rows": rows,
            "p1_admission": False, "independent_confirmation": False})
    except Exception as exc:
        base.write_new(output/"failure.json", {"type": type(exc).__name__, "message": str(exc)[:1000]})
        raise
    finally:
        for handle in handles:
            handle.remove()


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
            print(run_cuda(contract, a.contract_sha, ROOT/"results"/f"{NAME}_RUN_{a.tag}", a.authority))
