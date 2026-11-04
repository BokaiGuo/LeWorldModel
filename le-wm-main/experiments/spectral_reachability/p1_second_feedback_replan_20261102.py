"""Development-only cp10 RGB feedback replanning for the fixed timing panel.

The preceding cp5 tail's first packet has actually been executed.  This module
does not consume the second packet's outcome, dispatch actions, or certify P1.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from . import p1_residual_cp5_timing_20261102 as first
from . import p1_timing_second_feedback_inputs_20261102 as inputs

ROOT = first.ROOT
NAME = "p1_second_feedback_replan_20261102"
INPUT = ROOT / "results/p1_timing_second_feedback_inputs_20261102_DEV01.json"
INPUT_SHA = "43fad9697ffc56e9dc2dd4c8b4a3ae8b1796b36f2804c45fcf004551cd3dfe7f"
AUTHORITY = "DEV303_SECOND_FEEDBACK_DEVELOPMENT_CUDA"
require = first.require


def context(task, cp5, cp10, packets):
    """Three actual observation latents and exactly two normalized packets."""
    latents = np.asarray([cp5, cp10], dtype=np.float32)
    raw = np.asarray(packets, dtype=np.float32)
    mean = np.asarray(task["action_mean"], dtype=np.float32)
    std = np.asarray(task["action_std"], dtype=np.float32)
    require(latents.shape == (2, task["latent_dim"]) and raw.shape == (2, 5, 2)
            and mean.shape == std.shape == (2,) and np.isfinite(raw).all()
            and np.isfinite(latents).all() and np.isfinite(mean).all()
            and np.isfinite(std).all() and (std > 0).all() and (np.abs(raw) <= 1).all(),
            "invalid actual feedback context")
    warmup = np.asarray(task["root_context"]["latent_history"][-1], dtype=np.float32)
    require(warmup.shape == (task["latent_dim"],) and np.isfinite(warmup).all(),
            "invalid warmup observation")
    return np.vstack((warmup, latents)), ((raw - mean) / std).reshape(2, 10)


def frame(meta):
    path = Path(meta["path"])
    return first.rgb._frame(path.parent, {"file": path.name, "file_sha256": meta["file_sha256"],
                                          "pixel_sha256": meta["pixel_sha256"]})


def protocol():
    source = first.fixed(INPUT, INPUT_SHA)
    require(source == inputs.build(), "saved input panel changed")
    prep, task, _goal, model, _roles = first.source_inputs()
    rows = []
    for row in source["rows"]:
        cp10 = frame(row["rgb_cp10"])
        geometry = first.engine._geometry(cp10)
        rows.append({**row, "contact_direction_raw_xy": geometry["contact_direction_raw_xy"]})
    paths = (Path(__file__), Path(inputs.__file__), Path(first.__file__),
             Path(first.planner.__file__), INPUT)
    return {"schema": NAME + ".protocol.v1", "classification": "development_only",
            "input_sha256": INPUT_SHA, "rows": rows, "task": task,
            "policy": prep["model_policy"], "model_hash": model["model_before"],
            "objective": "prefix_latent at first packet; two packets rolled out",
            "optimizer": "V13 common zero initialization, deterministic shared seed",
            "budget_per_arm": {"rgb_encodings": 2, "predict_calls": 10,
                               "action_encoder_calls": 10, "predict_samples": 3200},
            "execution": "model-only; candidate first five actions require separate simulation",
            "selection_inputs": "goal and observed RGB plus actually executed action history",
            "source_sha256": {str(p.resolve()): first.file_sha(p) for p in paths},
            "physical_execution": False, "P1_admission": False,
            "independent_confirmation": False, "full_cost_certified": False}


def run(contract, contract_sha, output, authority):
    require(authority == AUTHORITY and first.fixed(contract, contract_sha) == protocol(),
            "prepared contract or authority")
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
    spec = first.fixed(contract, contract_sha)
    model = load_compatible_pretrained(spec["policy"], device=device).eval().requires_grad_(False)
    before = first.engine.base.blind.host.signal.model_hash(model)
    require(before == spec["model_hash"], "checkpoint identity")
    task = spec["task"]
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    mean, std, goal = tensor(task["action_mean"]), tensor(task["action_std"]), tensor(task["goal"])[None]
    counts = {}
    rows = []
    phase = {"name": ""}
    handles = []
    for name, module in (("encoder", model.encoder), ("projector", model.projector),
                         ("action_encoder", model.action_encoder), ("predictor", model.predictor)):
        def hook(_module, args, name=name):
            key = phase["name"] + ":" + name
            item = counts.setdefault(key, {"calls": 0, "samples": 0})
            item["calls"] += 1
            item["samples"] += int(args[0].shape[0])
        handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            for row in spec["rows"]:
                phase["name"] = row["arm"] + ":encode"
                observed = []
                for key in ("rgb_cp5", "rgb_cp10"):
                    image = first.engine.base.p0._image_tensor(
                        frame(row[key]), transform=first.engine.base.p0.image_transform(), device=device)
                    latent = model.encode({"pixels": image})["emb"]
                    require(tuple(latent.shape) == (1, 1, task["latent_dim"])
                            and bool(torch.isfinite(latent).all()), "actual RGB encoding")
                    observed.append(latent[0, 0].cpu().numpy())
                states, actions = context(task, *observed, row["executed_raw_action_packets"])
                z0, history = tensor(states)[None], tensor(actions)[None]
                phase["name"] = row["arm"] + ":plan"

                def score(variants):
                    batch = first.engine.SAMPLES * first.engine.VARIANTS
                    flat = tensor(variants.reshape(batch, 2, 5, 2))
                    z, past = z0.repeat(batch, 1, 1), history.repeat(batch, 1, 1)
                    first_endpoint = None
                    for packet in range(first.engine.HORIZON):
                        action = ((flat[:, packet] - mean) / std).reshape(batch, 1, 10)
                        ctx = torch.cat((past, action), dim=1)
                        predicted = model.predict(z, model.action_encoder(ctx))
                        require(predicted.shape == z.shape and bool(torch.isfinite(predicted).all()),
                                "finite prediction")
                        z = torch.cat((z, predicted[:, -1:]), dim=1)[:, -3:]
                        past = ctx[:, -2:]
                        if packet == 0:
                            first_endpoint = z[:, -1]
                    return ((first_endpoint - goal).square().mean(dim=1)
                            .reshape(first.engine.SAMPLES, first.engine.VARIANTS).cpu().numpy())

                result = first.planner.plan_one(row["contact_direction_raw_xy"], score)
                selected = np.asarray(result["selected"]["actions"], dtype="<f4").reshape(10, 2)
                require(np.isfinite(selected).all() and (np.abs(selected) <= 1).all(), "bounded actions")
                out = {"arm": row["arm"], "cp5_pixel_sha256": row["rgb_cp5"]["pixel_sha256"],
                       "cp10_pixel_sha256": row["rgb_cp10"]["pixel_sha256"],
                       "actual_action_packets_sha256": row["action_packets_sha256"],
                       "selected_actions_sha256": first.rgb.sha(selected.tobytes()), **result}
                rows.append(out)
                first.write_new(output / (row["arm"] + ".json"), out)
        torch.cuda.synchronize(device)
        after = first.engine.base.blind.host.signal.model_hash(model)
        require(before == after and len(rows) == 8 and len({r["initial_bank_sha256"] for r in rows}) == 1,
                "model or initialization drift")
        for row in spec["rows"]:
            for name in ("encoder", "projector"):
                require(counts[row["arm"] + ":encode:" + name] == {"calls": 2, "samples": 2},
                        "RGB encoding ledger")
            for name in ("action_encoder", "predictor"):
                require(counts[row["arm"] + ":plan:" + name] == {"calls": 10, "samples": 3200},
                        "planning ledger")
        require(first.fixed(contract, contract_sha) == protocol(), "postrun source closure")
        report = {"schema": NAME + ".report.v1", "status": "model_only_complete",
                  "protocol_sha256": contract_sha, "host": host, "model_before": before,
                  "model_after": after, "rows": rows, "hook_ledger": counts,
                  "physical_execution": False, "P1_admission": False,
                  "independent_confirmation": False, "full_cost_certified": False}
        return first.write_new(output / "report.json", report)
    except Exception as exc:
        first.write_new(output / "failure.json", {"error": str(exc), "type": type(exc).__name__,
                                                  "phase": phase["name"], "hook_ledger": counts,
                                                  "completed_rows": len(rows)})
        raise
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--authority")
    args = parser.parse_args()
    require(args.tag.isascii() and args.tag.isalnum(), "invalid tag")
    contract = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    if args.command == "prepare":
        require(not args.sha256 and not args.authority and not os.path.lexists(output), "prepare only")
        print(first.write_new(contract, protocol()))
    else:
        require(args.sha256 and args.authority, "missing run binding")
        with first.engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            print(run(contract, args.sha256, output, args.authority))


if __name__ == "__main__":
    main()
