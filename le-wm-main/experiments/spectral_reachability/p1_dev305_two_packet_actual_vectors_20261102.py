"""Encode actual RGB successors of the full fixed bank on the frozen model."""
import argparse
import os
from pathlib import Path

import numpy as np

from . import p1_dev305_two_packet_real_rgb_20261102 as physical
from . import p1_dev305_two_packet_model_bank_saved_20261102 as model_saved

base, need, run, ROOT = model_saved.base, model_saved.need, model_saved.model.run, model_saved.ROOT
NAME = "p1_dev305_two_packet_actual_vectors_20261102"
AUTHORITY = "DEV305_TWO_PACKET_ACTUAL_VECTORS_CUDA"
REPLAY_PREP_SHA = "df32b819ca55955a06877e1d6e39736a4a6266560d84fb0d8630202f12c14d80"
REPLAY_SHA = "ca90c9a22202148a7eeabfd45579f8e6f9fdb21ca442162c86614137786f835a"


def protocol():
    replay_spec = base.read(ROOT / "results" / f"{physical.NAME}_PREP_DEV01.json", REPLAY_PREP_SHA)
    replay = base.read(ROOT / "results" / f"{physical.NAME}_RUN_DEV01/report.json", REPLAY_SHA)
    model = base.read(ROOT / "results" / f"{model_saved.model.NAME}_PREP_DEV01.json", model_saved.CONTRACT_SHA)
    need(replay_spec == physical.protocol() and replay["contract_sha256"] == REPLAY_PREP_SHA
         and not replay["errors"] and len(replay["rows"]) == 867, "sealed physical replay")
    directory = ROOT / "results" / f"{physical.NAME}_RUN_DEV01"
    refs = []
    for row in replay["rows"]:
        need(row["status"] == "fixed_horizon_completed", "incomplete branch")
        frame = row["final"]
        refs.append({"arm": row["arm"], "first": row["first"], "second": row["second"],
                     "image": {"path": str((directory / frame["file"]).resolve()),
                               "file_sha256": frame["file_sha256"], "pixel_sha256": frame["pixel_sha256"]}})
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "physical_replay_sha256": REPLAY_SHA, "model_contract_sha256": model_saved.CONTRACT_SHA,
            "policy": model["policy"], "model_hash": model["model_hash"], "latent_dim": model["latent_dim"],
            "goal": model["goal"], "frames": refs,
            "classification": "post_outcome_privileged_real_successor_diagnostic_not_planner",
            "p1_admission": False, "independent_confirmation": False}


def execute(path, digest, output, authority):
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
                    run.coherent.base.frame(ref), transform=run.coherent.base.first.engine.base.p0.image_transform(),
                    device=device)
                value = model.encode({"pixels": image})["emb"]
                need(tuple(value.shape) == (1, 1, spec["latent_dim"]) and bool(torch.isfinite(value).all()), "encode")
                return value[0, 0]
            goal = encode(spec["goal"])
            rows = []
            for index, item in enumerate(spec["frames"]):
                actual = encode(item["image"])
                rows.append({"arm": item["arm"], "first": item["first"], "second": item["second"],
                             "actual_goal_mse": float((actual - goal).square().mean().cpu()),
                             "actual_goal_delta": (actual - goal).cpu().tolist()})
                if (index + 1) % 289 == 0:
                    print(f"encoded {index+1}/867 actual terminal frames", flush=True)
        torch.cuda.synchronize(device)
        after = hasher(model)
        need(before == after and spec == protocol(), "frozen source")
        need(counts == {"encoder": {"calls": 868, "samples": 868},
                        "projector": {"calls": 868, "samples": 868}}, "ledger")
        return base.write_new(output / "report.json", {"schema": NAME + ".report.v1",
            "contract_sha256": digest, "host": host, "model_before": before, "model_after": after,
            "counts": counts, "rows": rows, "p1_admission": False, "independent_confirmation": False})
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
