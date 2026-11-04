"""Same-action open versus observed-RGB-refreshed prediction on fixed dev304 branches."""
import argparse
import os
from pathlib import Path

import numpy as np

from . import p1_dev304_common10_saved_20261102 as saved

source, base, run = saved.source, saved.base, saved.run
ROOT = run.ROOT
NAME = "p1_dev304_common10_prediction_20261102"
AUTHORITY = "DEV304_COMMON10_PREDICTION_CUDA"
need = source.need


def protocol():
    saved.analyze()
    common = base.read(saved.ROOT / f"{source.NAME}_PREP_DEV01.json", saved.CONTRACT_SHA)
    physical = base.read(saved.RUN / "report.json", saved.REPORT_SHA)
    feedback = base.read(ROOT / f"results/{source.f.NAME}_PREP_DEV01.json", source.physical.CONTRACT_SHA)
    entries = []
    for row in physical["rows"]:
        prefix = next(x for x in feedback["roles"] if x["arm"] == row["prefix"])
        tail = next(x for x in common["tails"] if x["name"] == row["tail"])
        def ref(o):
            return {"path": str((saved.RUN / o["file"]).resolve()),
                    "file_sha256": o["file_sha256"], "pixel_sha256": o["pixel_sha256"]}
        entries.append({"arm": row["arm"], "prefix": row["prefix"], "tail": row["tail"],
                        "context_rgb": prefix["images"], "actual_history": prefix["actual_history"],
                        "tail_actions": tail["actions"], "tail_sha256": tail["sha256"],
                        "real_5": ref(row["observations"][8]),
                        "real_10": ref(row["observations"][13])})
    return {"schema": NAME+".protocol.v1", "classification": "post_outcome_one_known_root",
            "source_code_sha256": base.sha(__file__), "common_report_sha256": saved.REPORT_SHA,
            "common_contract_sha256": saved.CONTRACT_SHA, "goal": feedback["goal"],
            "policy": feedback["policy"], "model_hash": feedback["model_hash"],
            "action_mean": feedback["action_mean"], "action_std": feedback["action_std"],
            "latent_dim": feedback["latent_dim"], "branches": entries,
            "comparison": "same actions at tail steps 5 and 10; open versus real step-5 refreshed",
            "uses_future_rgb_for_diagnostic_only": True, "p1_admission": False,
            "independent_confirmation": False}


def run_cuda(contract_path, contract_sha, output, authority):
    need(authority == AUTHORITY and base.read(contract_path, contract_sha) == protocol()
         and not os.path.lexists(output), "contract/authority")
    spec = base.read(contract_path, contract_sha)
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
    need(before == spec["model_hash"], "frozen model")
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    mean, std = tensor(spec["action_mean"]), tensor(spec["action_std"])
    need(bool(torch.isfinite(std).all()) and bool((std > 0).all()), "normalizer")
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
                pixels = run.coherent.base.frame(ref)
                image = run.coherent.base.first.engine.base.p0._image_tensor(
                    pixels, transform=run.coherent.base.first.engine.base.p0.image_transform(), device=device)
                z = model.encode({"pixels": image})["emb"]
                need(z.shape == (1, 1, spec["latent_dim"]) and bool(torch.isfinite(z).all()), "encoder")
                return z
            goal = encode(spec["goal"])
            for branch in spec["branches"]:
                context = torch.cat([encode(x) for x in branch["context_rgb"]], dim=1)
                real5, real10 = encode(branch["real_5"]), encode(branch["real_10"])
                history = tensor(branch["actual_history"]).reshape(1, 2, 5, 2)
                history = ((history - mean) / std).reshape(1, 2, 10)
                actions = tensor(branch["tail_actions"]).reshape(2, 5, 2)
                def predict(z, hist, a):
                    normalized = ((a-mean)/std).reshape(1, 1, 10)
                    context_actions = torch.cat((hist, normalized), dim=1)
                    prediction = model.predict(z, model.action_encoder(context_actions))
                    need(prediction.shape == z.shape and bool(torch.isfinite(prediction).all()), "predict")
                    return prediction[:, -1:], context_actions[:, -2:]
                pred5, hist5 = predict(context, history, actions[0])
                open_context = torch.cat((context[:, 1:], pred5), dim=1)
                refresh_context = torch.cat((context[:, 1:], real5), dim=1)
                opened, _ = predict(open_context, hist5, actions[1])
                refreshed, _ = predict(refresh_context, hist5, actions[1])
                def arr(x):
                    return x.reshape(-1).cpu().numpy().tolist()
                vectors = {"goal": arr(goal), "real5": arr(real5), "real10": arr(real10),
                           "pred5": arr(pred5), "open10": arr(opened), "refresh10": arr(refreshed)}
                row = {"arm": branch["arm"], "prefix": branch["prefix"], "tail": branch["tail"],
                       "tail_sha256": branch["tail_sha256"], "vectors": vectors}
                rows.append(row)
                base.write_new(output / (branch["arm"] + ".json"), row)
        torch.cuda.synchronize(device)
        after = hasher(model)
        need(before == after and spec == protocol(), "frozen sources postrun")
        need(counts["encoder"] == counts["projector"] == {"calls": 31, "samples": 31}
             and counts["action_encoder"] == counts["predictor"] == {"calls": 18, "samples": 18},
             "model ledger")
        return base.write_new(output/"report.json", {"schema": NAME+".report.v1",
                    "contract_sha256": contract_sha, "model_before": before, "model_after": after,
                    "host": host, "ledger": counts, "rows": rows, "p1_admission": False,
                    "independent_confirmation": False})
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
