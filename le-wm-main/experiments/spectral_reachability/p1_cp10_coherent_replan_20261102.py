"""Exploratory equal-call cp10 CEM with generic time-correlated RGB seeds."""
import argparse
import os
from pathlib import Path

import numpy as np

from . import p1_second_feedback_replan_20261102 as base
from . import p1_task_response_revision as revision

ROOT = base.ROOT
NAME = "p1_cp10_coherent_replan_20261102"
AUTHORITY = "DEV303_CP10_COHERENT_CUDA"
require = base.require


def protocol():
    source = base.protocol()
    return {"schema": NAME + ".protocol.v1",
            "classification": "known_development_root_post_oracle_not_holdout",
            "rows": source["rows"], "task": source["task"], "policy": source["policy"],
            "model_hash": source["model_hash"], "source_input_sha256": source["input_sha256"],
            "seed_rule": "generic RGB contact direction, fixed amplitudes and rotations; no oracle candidates",
            "optimizer": "five rounds of 64, five variants, same random seed, bound and feasible projection as V13",
            "scoring": "first-packet latent MSE; both model packets rolled out",
            "per_arm": {"encoder_calls": 2, "action_encoder_calls": 10,
                        "predictor_calls": 10, "predict_samples": 3200},
            "source_sha256": {str(Path(__file__).resolve()): base.first.file_sha(__file__),
                              str(Path(revision.__file__).resolve()): base.first.file_sha(revision.__file__)},
            "physical_execution": False, "independent_confirmation": False, "P1_admission": False,
            "full_cost_certified": False}


def run(contract, digest, output, authority):
    require(authority == AUTHORITY and base.first.fixed(contract, digest) == protocol(), "contract")
    require(not os.path.lexists(output), "exclusive output")
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
    spec = base.first.fixed(contract, digest)
    model = load_compatible_pretrained(spec["policy"], device=device).eval().requires_grad_(False)
    hasher = base.first.engine.base.blind.host.signal.model_hash
    before = hasher(model)
    require(before == spec["model_hash"], "checkpoint")
    task = spec["task"]
    tensor = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)
    mean, std, goal = tensor(task["action_mean"]), tensor(task["action_std"]), tensor(task["goal"])[None]
    ledger, handles, phase, rows = {}, [], {"name": ""}, []
    for name, module in (("encoder", model.encoder), ("projector", model.projector),
                         ("action_encoder", model.action_encoder), ("predictor", model.predictor)):
        def hook(_module, args, name=name):
            entry = ledger.setdefault(phase["name"] + ":" + name, {"calls": 0, "samples": 0})
            entry["calls"] += 1
            entry["samples"] += int(args[0].shape[0])
        handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            for row in spec["rows"]:
                phase["name"] = row["arm"] + ":encode"
                latents = []
                for key in ("rgb_cp5", "rgb_cp10"):
                    image = base.first.engine.base.p0._image_tensor(
                        base.frame(row[key]), transform=base.first.engine.base.p0.image_transform(), device=device)
                    z = model.encode({"pixels": image})["emb"]
                    require(z.shape == (1, 1, task["latent_dim"]) and bool(torch.isfinite(z).all()), "RGB latent")
                    latents.append(z[0, 0].cpu().numpy())
                states, history = base.context(task, *latents, row["executed_raw_action_packets"])
                z0, past0 = tensor(states)[None], tensor(history)[None]
                phase["name"] = row["arm"] + ":plan"
                def score(variants):
                    batch = base.first.engine.SAMPLES * base.first.engine.VARIANTS
                    flat = tensor(variants.reshape(batch, 2, 5, 2))
                    z, past = z0.repeat(batch, 1, 1), past0.repeat(batch, 1, 1)
                    endpoint = None
                    for packet in range(2):
                        actions = ((flat[:, packet] - mean) / std).reshape(batch, 1, 10)
                        ctx = torch.cat((past, actions), dim=1)
                        pred = model.predict(z, model.action_encoder(ctx))
                        require(pred.shape == z.shape and bool(torch.isfinite(pred).all()), "prediction")
                        z = torch.cat((z, pred[:, -1:]), dim=1)[:, -3:]
                        past = ctx[:, -2:]
                        if packet == 0:
                            endpoint = z[:, -1]
                    return ((endpoint-goal).square().mean(dim=1)
                            .reshape(base.first.engine.SAMPLES, base.first.engine.VARIANTS).cpu().numpy())
                result = revision.plan(row["contact_direction_raw_xy"], score, structured=True,
                                       common=base.first.engine.base.common, seed=base.first.planner.parent.SEED)
                actions = np.asarray(result["selected"]["actions"], dtype="<f4")
                require(actions.shape == (2, 5, 2) and np.isfinite(actions).all()
                        and (np.abs(actions) <= 1).all(), "bounded selected actions")
                out = {"arm": row["arm"], "cp10_pixel_sha256": row["rgb_cp10"]["pixel_sha256"],
                       "selected_actions_sha256": base.first.rgb.sha(actions.tobytes()), **result}
                rows.append(out)
                base.first.write_new(output / (row["arm"] + ".json"), out)
        torch.cuda.synchronize(device)
        after = hasher(model)
        require(before == after and len(rows) == 8, "frozen model and roster")
        for row in spec["rows"]:
            for name in ("encoder", "projector"):
                require(ledger[row["arm"]+":encode:"+name] == {"calls": 2, "samples": 2}, "encode ledger")
            for name in ("action_encoder", "predictor"):
                require(ledger[row["arm"]+":plan:"+name] == {"calls": 10, "samples": 3200}, "planning ledger")
        require(base.first.fixed(contract, digest) == protocol(), "postrun source closure")
        return base.first.write_new(output / "report.json", {
            "schema": NAME + ".report.v1", "classification": spec["classification"],
            "protocol_sha256": digest, "model_before": before, "model_after": after, "host": host,
            "rows": rows, "hook_ledger": ledger, "physical_execution": False,
            "independent_confirmation": False, "P1_admission": False, "full_cost_certified": False})
    except Exception as exc:
        base.first.write_new(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                       "phase": phase["name"], "completed_rows": len(rows)})
        raise
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--authority")
    args = parser.parse_args()
    require(args.tag.isascii() and args.tag.isalnum(), "tag")
    contract = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    if args.command == "prepare":
        require(not args.sha256 and not args.authority and not os.path.lexists(output), "prepare")
        print(base.first.write_new(contract, protocol()))
    else:
        require(args.sha256 and args.authority, "authorization")
        with base.first.engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            print(run(contract, args.sha256, output, args.authority))
