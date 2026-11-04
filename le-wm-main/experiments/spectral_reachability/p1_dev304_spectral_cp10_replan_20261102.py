"""Replan both dev304 prefixes from observed cp5 RGB and actual action history.

Post-outcome development only. This module does not dispatch physical actions.
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

from . import p1_dev304_diverse_saved_20261102 as saved
from . import p1_dev304_goalrank_spectral_saved_20261102 as goal_saved
from . import p1_dev304_qgoal_spectral_saved_20261102 as q_saved

run = saved.run
base = run.base
require = run.require
ROOT = run.ROOT
NAME = "p1_dev304_spectral_cp10_replan_20261102"
AUTHORITY = "DEV304_SPECTRAL_CP10_REPLAN_CUDA"


def protocol():
    source_spec = base.read(goal_saved.ROOT/f"{goal_saved.model_run.NAME}_PREP_DEV01.json",
                            goal_saved.physical.CONTRACT_SHA)
    roles = []
    for label, panel in (("goal", goal_saved), ("qgoal", q_saved)):
        panel.analyze()
        contract = base.read(panel.ROOT/f"{panel.physical.NAME}_PREP_DEV01.json", panel.PREP_SHA)
        execution = base.read(panel.RUN/"report.json", panel.REPORT_SHA)
        model_spec = base.read(panel.ROOT/f"{panel.model_run.NAME}_PREP_DEV01.json",
                               panel.physical.CONTRACT_SHA)
        for role, prior in zip(contract["roles"], model_spec["roles"], strict=True):
            index = role["model_selected_index"]
            name = f"{role['arm']}_{index:02d}"
            row = next(r for r in execution["rows"] if r["arm"] == name)
            require(row["status"] == "fixed_horizon_completed" and prior["arm"] == role["arm"],
                    "completed selected candidate")
            cp10 = next(o for o in row["observations"] if o["phase"] == "tail" and o["step"] == 5)
            ref = {"path": str((panel.RUN/cp10["file"]).resolve()),
                   "file_sha256": cp10["file_sha256"], "pixel_sha256": cp10["pixel_sha256"]}
            frames = [prior["images"][1], prior["images"][2], ref]
            prefix = np.asarray(role["prefix"], dtype="<f4")
            packet = np.asarray(role["candidates"][index]["actions"], dtype="<f4")[0]
            actual = np.concatenate((prefix, packet))
            require(actual.shape == (10, 2) and all(
                np.array_equal(t["action"], a) for t, a in zip(row["transitions"][:10], actual, strict=True)),
                "actual action history")
            for frame in frames:
                run.coherent.base.frame(frame)
            geometry = run.coherent.base.first.engine._geometry(run.coherent.base.frame(ref))
            roles.append({"arm": f"{label}_{role['arm']}", "images": frames,
                          "actual_history": actual.tolist(), "source_arm": name,
                          "source_physical_sha256": panel.REPORT_SHA,
                          "prefix_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                          "contact_direction_raw_xy": geometry["contact_direction_raw_xy"]})
    return {"schema": NAME + ".protocol.v1", "classification": "post_outcome_development",
            "code_sha256": base.sha(__file__), "policy": source_spec["policy"],
            "model_hash": source_spec["model_hash"], "action_mean": source_spec["action_mean"],
            "action_std": source_spec["action_std"], "latent_dim": source_spec["latent_dim"],
            "goal": source_spec["goal"], "roles": roles,
            "selection_inputs": "real cp10 RGB, earlier RGB, goal RGB and actual actions only",
            "optimizer": "same one-population CEM, same seed, two-packet terminal score",
            "per_arm": {"predict_calls": 10, "predict_samples": 3200},
            "physical_execution_completed": False, "p1_admission": False,
            "independent_confirmation": False, "full_cost_certified": False}


def plan(contract_path, contract_sha, output, authority):
    require(authority == AUTHORITY and base.read(contract_path, contract_sha) == protocol()
            and not os.path.lexists(output), "contract, authority or output")
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
    require(before == spec["model_hash"], "model identity")
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    mean, std = tensor(spec["action_mean"]), tensor(spec["action_std"])
    require(bool(torch.isfinite(std).all()) and bool((std > 0).all()), "action normalization")
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
                    run.coherent.base.frame(ref),
                    transform=run.coherent.base.first.engine.base.p0.image_transform(), device=device)
                value = model.encode({"pixels": image})["emb"]
                require(tuple(value.shape) == (1, 1, spec["latent_dim"])
                        and bool(torch.isfinite(value).all()), "RGB encoding")
                return value[0, 0]
            goal = encode(spec["goal"])[None]
            for role in spec["roles"]:
                phase["value"] = "encode"
                z0 = torch.stack([encode(ref) for ref in role["images"]])[None]
                actual = tensor(role["actual_history"]).reshape(2, 5, 2)
                history = ((actual-mean)/std).reshape(1, 2, 10)
                phase["value"] = role["arm"]
                def score(variants):
                    flat = tensor(variants.reshape(320, 2, 5, 2))
                    z, past = z0.repeat(320, 1, 1), history.repeat(320, 1, 1)
                    for packet in range(2):
                        context = torch.cat((past, ((flat[:, packet]-mean)/std).reshape(320, 1, 10)), dim=1)
                        predicted = model.predict(z, model.action_encoder(context))
                        require(predicted.shape == z.shape and bool(torch.isfinite(predicted).all()), "prediction")
                        z = torch.cat((z, predicted[:, -1:]), dim=1)[:, -3:]
                        past = context[:, -2:]
                    return (z[:, -1]-goal).square().mean(dim=1).reshape(64, 5).cpu().numpy()
                result = run.diverse.plan(role["contact_direction_raw_xy"], score, structured=False,
                                          common=run.coherent.base.first.engine.base.common,
                                          seed=run.coherent.base.first.planner.parent.SEED)
                actions = np.asarray(result["selected"]["actions"], dtype="<f4")
                row = {"arm": role["arm"], "prefix_sha256": role["prefix_sha256"],
                       "selected_actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(), **result}
                base.write_new(output / (role["arm"] + ".json"), row)
                rows.append(row)
        torch.cuda.synchronize(device)
        after = hasher(model)
        require(before == after and spec == protocol(), "postrun frozen source")
        for label in ("encoder", "projector"):
            require(counts["encode:"+label] == {"calls": 13, "samples": 13}, "encoding ledger")
        for role in spec["roles"]:
            for label in ("action_encoder", "predictor"):
                require(counts[role["arm"]+":"+label] == {"calls": 10, "samples": 3200}, "model ledger")
        return base.write_new(output / "report.json",
                              {"schema": NAME + ".report.v1", "contract_sha256": contract_sha,
                               "host": host, "model_before": before, "model_after": after,
                               "ledger": counts, "rows": rows, "physical_execution_completed": False,
                               "p1_admission": False, "independent_confirmation": False,
                               "full_cost_certified": False})
    except Exception as exc:
        base.write_new(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
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
    require(args.tag.isascii() and args.tag.isalnum(), "tag")
    contract = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    if args.command == "prepare":
        print(base.write_new(contract, protocol()))
    else:
        with run.coherent.base.first.engine.base.locking.gpu_lock(ROOT/"results/e3_p2_host_cuda_cooperative.lock"):
            print(plan(contract, args.contract_sha, ROOT/"results"/f"{NAME}_PLAN_{args.tag}", args.authority))
