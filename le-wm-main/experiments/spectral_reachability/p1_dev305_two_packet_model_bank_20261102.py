"""Frozen RGB model scores the exact fixed 17x17 bank used by physical oracle."""
import argparse
import os
from pathlib import Path

import numpy as np

from . import p1_dev305_two_packet_physical_oracle_20261102 as physical
from . import p1_dev305_three_prefix_replan_20261102 as earlier

base, run, ROOT, need = earlier.base, earlier.run, earlier.ROOT, earlier.require
NAME = "p1_dev305_two_packet_model_bank_20261102"
AUTHORITY = "DEV305_TWO_PACKET_MODEL_BANK_CUDA"
PHYSICAL_CONTRACT_SHA = "f826ea69ac9a4afb4d207a8980b281eea7262a3e8c20285b00788670c909c921"


def protocol():
    states = earlier.protocol()
    oracle_contract = base.read(ROOT / "results" / f"{physical.NAME}_PREP_DEV01.json", PHYSICAL_CONTRACT_SHA)
    need(oracle_contract == physical.protocol(), "sealed physical bank")
    bank = oracle_contract["bank"]
    need(len(bank) == 17, "bank")
    # Explicitly exclude simulator state, previous outcomes, and oracle choices.
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "bank_contract_sha256": PHYSICAL_CONTRACT_SHA, "bank": bank, "packet_length": 5,
            "roles": states["roles"], "policy": states["policy"], "model_hash": states["model_hash"],
            "action_mean": states["action_mean"], "action_std": states["action_std"],
            "goal": states["goal"], "latent_dim": states["latent_dim"],
            "selection_inputs": "real RGB, goal RGB, actual action history; no physical outcome",
            "score": "predicted terminal latent MSE; no guard, perturbations or extra optimization",
            "classification": "known-root post-outcome development diagnostic",
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
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    mean, std = tensor(spec["action_mean"]), tensor(spec["action_std"])
    need(bool(torch.isfinite(std).all()) and bool((std > 0).all()), "normalizer")
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
            goal = encode(spec["goal"])[None]
            bank = tensor([x["action"] for x in spec["bank"]])
            packet = bank[:, None, :].repeat(1, 5, 1)
            rows = []
            for role in spec["roles"]:
                z0 = torch.stack([encode(x) for x in role["images"]])[None]
                history = ((tensor(role["actual_history"]).reshape(2, 5, 2) - mean) / std).reshape(1, 2, 10)
                terminal = []
                one_scores = []
                for i in range(17):
                    z = z0.repeat(17, 1, 1)
                    past = history.repeat(17, 1, 1)
                    for k in range(2):
                        raw = packet[i].unsqueeze(0).repeat(17, 1, 1) if k == 0 else packet
                        context = torch.cat((past, ((raw - mean) / std).reshape(17, 1, 10)), dim=1)
                        predicted = model.predict(z, model.action_encoder(context))
                        need(predicted.shape == z.shape and bool(torch.isfinite(predicted).all()), "prediction")
                        z = torch.cat((z, predicted[:, -1:]), dim=1)[:, -3:]
                        past = context[:, -2:]
                        if k == 0:
                            first = (z[:, -1] - goal).square().mean(dim=1)
                            need(bool(torch.allclose(first, first[:1].expand_as(first), atol=1e-6, rtol=0)),
                                 "duplicate first-packet predictions")
                            one_scores.append(float(first[0].cpu()))
                    terminal.append((z[:, -1] - goal).square().mean(dim=1).cpu().tolist())
                terminal = np.asarray(terminal)
                need(terminal.shape == (17, 17) and np.isfinite(terminal).all(), "scores")
                indices = np.unravel_index(int(np.argmin(terminal)), terminal.shape)
                rows.append({"arm": role["arm"], "first_scores": one_scores, "terminal_scores": terminal.tolist(),
                             "selected_pair": list(map(int, indices))})
        torch.cuda.synchronize(device)
        after = hasher(model)
        need(before == after and spec == protocol(), "frozen source")
        need(counts == {"encoder": {"calls": 10, "samples": 10},
                        "projector": {"calls": 10, "samples": 10},
                        "action_encoder": {"calls": 102, "samples": 1734},
                        "predictor": {"calls": 102, "samples": 1734}}, "ledger")
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
