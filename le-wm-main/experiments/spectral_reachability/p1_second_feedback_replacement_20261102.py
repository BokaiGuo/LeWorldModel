"""Score sealed old/new action packets from RGB-only cp10 inputs.

Exploratory post-outcome application of a fixed robust rule; not confirmation.
Neither simulator state nor outcomes are inputs to this selector.
"""
import argparse
import os
from pathlib import Path

import numpy as np

from . import p1_second_feedback_replan_20261102 as base

ROOT = base.ROOT
NAME = "p1_second_feedback_replacement_20261102"
AUTHORITY = "DEV303_REPLACEMENT_RESCORING_CUDA"
FIRST_REPORT = ROOT / "results/p1_residual_cp5_timing_20261102_RUN_DEV01/report.json"
FIRST_SHA = "220102a98ec1be31c20132e863c24f81a8022fdafeb883ca179506cdaa24f830"
SECOND_REPORT = ROOT / "results/p1_second_feedback_replan_20261102_RUN_DEV01/report.json"
SECOND_SHA = "c7180ba2818c761c7a6cbd1e9b0fe702ce0badad5009be03230c3d26e88181c8"
require = base.require


def select(variant_scores):
    values = np.asarray(variant_scores, dtype=np.float64)
    require(values.shape == (2, 5) and np.isfinite(values).all(), "two finite five-variant scores")
    robust = base.first.engine.base.common.robust_scores(values)
    # Ties retain the pre-existing plan.
    choice = "new" if robust[1] < robust[0] else "old"
    return {"selected": choice, "nominal": values[:, 0].tolist(), "robust": robust.tolist(),
            "variant_scores": values.tolist(), "tie_rule": "retain_old"}


def protocol():
    inp = base.first.fixed(base.INPUT, base.INPUT_SHA)
    require(inp == base.inputs.build(), "verified RGB/action panel")
    first = base.first.fixed(FIRST_REPORT, FIRST_SHA)
    second = base.first.fixed(SECOND_REPORT, SECOND_SHA)
    spec = base.protocol()
    require(first["model_before"] == first["model_after"] == second["model_before"]
            == second["model_after"] == spec["model_hash"], "model identity")
    first_by = {f"candidate_{r['index']:02d}_{r['metric']}": r for r in first["rows"]}
    second_by = {r["arm"]: r for r in second["rows"]}
    require(len(first_by) == len(second_by) == len(inp["rows"]) == 8, "unique full arm roster")
    rows = []
    for r in inp["rows"]:
        old = np.asarray(first_by[r["arm"]]["selected"]["actions"], dtype="<f4").reshape(10, 2)
        new = np.asarray(second_by[r["arm"]]["selected"]["actions"], dtype="<f4").reshape(10, 2)
        history = np.asarray(r["executed_raw_action_packets"], dtype="<f4")
        require(np.array_equal(old[:5], history[1])
                and np.isfinite(old).all() and np.isfinite(new).all()
                and (np.abs(old) <= 1).all() and (np.abs(new) <= 1).all(), "executed history and bounds")
        rows.append({"arm": r["arm"], "rgb_cp5": r["rgb_cp5"], "rgb_cp10": r["rgb_cp10"],
                     "history_raw_packets": r["executed_raw_action_packets"],
                     "old_packet": old[5:].tolist(), "new_packet": new[:5].tolist()})
    return {"schema": NAME + ".protocol.v1", "classification": "post_outcome_development_only",
            "rows": rows, "policy": spec["policy"], "task": spec["task"], "model_hash": spec["model_hash"],
            "rule": "compare same cp10 first-packet mean-plus-range of five 0.05 perturbations; retain old on tie",
            "per_arm": {"rgb_encoder_calls": 2, "action_encoder_calls": 1,
                        "predictor_calls": 1, "predict_samples": 10},
            "no_outcomes_or_simulator_state_in_selection": True,
            "source_sha256": {str(Path(__file__).resolve()): base.first.file_sha(__file__)},
            "P1_admission": False, "independent_confirmation": False, "full_cost_certified": False}


def run(contract, sha, output, authority):
    require(authority == AUTHORITY and base.first.fixed(contract, sha) == protocol(), "contract")
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
    spec = base.first.fixed(contract, sha)
    model = load_compatible_pretrained(spec["policy"], device=device).eval().requires_grad_(False)
    hasher = base.first.engine.base.blind.host.signal.model_hash
    before = hasher(model)
    require(before == spec["model_hash"], "model identity")
    task = spec["task"]
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    mean, std, goal = tensor(task["action_mean"]), tensor(task["action_std"]), tensor(task["goal"])[None]
    rows, counts, hooks = [], {}, []
    phase = {"name": ""}
    for name, module in (("encoder", model.encoder), ("projector", model.projector),
                         ("action_encoder", model.action_encoder), ("predictor", model.predictor)):
        def hook(_module, args, name=name):
            k = phase["name"] + ":" + name
            v = counts.setdefault(k, {"calls": 0, "samples": 0})
            v["calls"] += 1
            v["samples"] += int(args[0].shape[0])
        hooks.append(module.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            for r in spec["rows"]:
                phase["name"] = r["arm"] + ":encode"
                observed = []
                for key in ("rgb_cp5", "rgb_cp10"):
                    image = base.first.engine.base.p0._image_tensor(
                        base.frame(r[key]), transform=base.first.engine.base.p0.image_transform(), device=device)
                    z = model.encode({"pixels": image})["emb"]
                    require(tuple(z.shape) == (1, 1, task["latent_dim"]) and bool(torch.isfinite(z).all()),
                            "encoded RGB")
                    observed.append(z[0, 0].cpu().numpy())
                z, past = base.context(task, *observed, r["history_raw_packets"])
                variants = []
                for label in ("old_packet", "new_packet"):
                    two = np.stack((np.asarray(r[label], np.float32), np.zeros((5, 2), np.float32)))
                    variants.append(base.first.engine.base.common.action_variants(two)[:, 0])
                batch = tensor(np.stack(variants).reshape(10, 5, 2))
                history = tensor(past)[None].repeat(10, 1, 1)
                states = tensor(z)[None].repeat(10, 1, 1)
                phase["name"] = r["arm"] + ":score"
                action = ((batch - mean) / std).reshape(10, 1, 10)
                pred = model.predict(states, model.action_encoder(torch.cat((history, action), dim=1)))
                require(pred.shape == states.shape and bool(torch.isfinite(pred).all()), "finite predictions")
                scores = (pred[:, -1] - goal).square().mean(dim=1).reshape(2, 5).cpu().numpy()
                result = {"arm": r["arm"], **select(scores)}
                rows.append(result)
                base.first.write_new(output / (r["arm"] + ".json"), result)
        torch.cuda.synchronize(device)
        after = hasher(model)
        require(before == after and len(rows) == 8, "checkpoint and roster")
        for r in spec["rows"]:
            for name in ("encoder", "projector"):
                require(counts[r["arm"] + ":encode:" + name] == {"calls": 2, "samples": 2}, "encode ledger")
            for name in ("action_encoder", "predictor"):
                require(counts[r["arm"] + ":score:" + name] == {"calls": 1, "samples": 10}, "score ledger")
        require(base.first.fixed(contract, sha) == protocol(), "postrun source closure")
        return base.first.write_new(output / "report.json", {
            "schema": NAME + ".report.v1", "classification": spec["classification"],
            "protocol_sha256": sha, "host": host, "model_before": before, "model_after": after,
            "rows": rows, "hook_ledger": counts, "P1_admission": False,
            "independent_confirmation": False, "full_cost_certified": False})
    except Exception as exc:
        base.first.write_new(output / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                                                       "phase": phase["name"], "completed_rows": len(rows)})
        raise
    finally:
        for hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--authority")
    args = parser.parse_args()
    require(args.tag.isascii() and args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    if args.command == "prepare":
        require(not args.sha256 and not args.authority and not os.path.lexists(output), "prepare only")
        print(base.first.write_new(prepared, protocol()))
    else:
        require(args.sha256 and args.authority, "missing binding")
        with base.first.engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            print(run(prepared, args.sha256, output, args.authority))


if __name__ == "__main__":
    main()
