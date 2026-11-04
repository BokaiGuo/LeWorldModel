"""Compare frozen-model and real RGB endpoints for a fixed, previously executed action bank."""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

from . import p1_dev305_three_prefix_replan_20261102 as planner
from . import p1_dev305_three_prefix_saved_20261102 as physical
from . import p1_dev305_compass_suffix_saved_20261102 as compass

base, need, run, ROOT = planner.base, planner.require, planner.run, planner.ROOT
NAME = "p1_dev305_compass_model_compare_20261102"
AUTHORITY = "DEV305_COMPASS_MODEL_COMPARE_CUDA"
PLANNER_SHA = physical.source.PLAN_CONTRACT_SHA


def protocol():
    physical.analyze()
    compass.analyze()
    spec = base.read(ROOT / f"results/{planner.NAME}_PREP_DEV01.json", PLANNER_SHA)
    tails = base.read(compass.ROOT / f"{compass.source.NAME}_PREP_DEV01.json", compass.CONTRACT_SHA)["tails"]
    data = base.read(compass.ROOT / f"{compass.source.NAME}_RUN_DEV01/report.json", compass.REPORT_SHA)
    physdir = physical.ROOT / f"{physical.source.NAME}_RUN_DEV01"
    physreport = base.read(physdir / "report.json", physical.REPORT_SHA)
    compassdir = compass.ROOT / f"{compass.source.NAME}_RUN_DEV01"
    roles = []
    for role in spec["roles"]:
        rows = []
        original = next(r for r in physreport["rows"] if r["arm"] == role["arm"])
        for tail in tails + [{"name": "cem_selected", "actions": [t["action"] for t in original["transitions"][5:]]}]:
            row = original if tail["name"] == "cem_selected" else next(
                r for r in data["rows"] if r["arm"] == role["arm"] + "__" + tail["name"])
            folder = physdir if tail["name"] == "cem_selected" else compassdir
            need(row["status"] == "fixed_horizon_completed" and len(row["transitions"]) == 15, "complete tail")
            actions = np.asarray(tail["actions"], dtype="<f4")
            need(actions.shape == (10, 2) and np.isfinite(actions).all()
                 and np.abs(actions).max() <= 1 and all(
                     np.array_equal(t["action"], a) for t, a in zip(row["transitions"][5:], actions, strict=True)),
                 "action binding")
            frame = next(o for o in row["observations"] if o["phase"] == "tail" and o["step"] == 10)
            ref = {"path": str((folder / frame["file"]).resolve()), "file_sha256": frame["file_sha256"],
                   "pixel_sha256": frame["pixel_sha256"]}
            direction = np.asarray(role["contact_direction_raw_xy"], dtype=np.float32)
            direction /= np.linalg.norm(direction)
            guard = float(np.dot(actions[:3].sum(axis=0), direction))
            rows.append({"name": tail["name"], "actions": actions.tolist(), "frame": ref,
                         "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                         "guard_dot": guard, "feasible": guard >= .15,
                         "reward_evaluation_only": row["transitions"][-1]["reward"]})
        need(len(rows) == 18, "bank roster")
        roles.append({"arm": role["arm"], "images": role["images"],
                      "actual_history": role["actual_history"], "rows": rows})
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "model_hash": spec["model_hash"], "policy": spec["policy"], "goal": spec["goal"],
            "action_mean": spec["action_mean"], "action_std": spec["action_std"],
            "latent_dim": spec["latent_dim"], "roles": roles, "planner_contract_sha256": PLANNER_SHA,
            "classification": "post_outcome_development_diagnostic_not_selector",
            "independent_confirmation": False, "p1_admission": False}


def execute(path, digest, output, authority):
    need(authority == AUTHORITY and base.read(path, digest) == protocol()
         and not os.path.lexists(output), "contract")
    spec = base.read(path, digest)
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
            tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
            def encode(ref):
                image = run.coherent.base.first.engine.base.p0._image_tensor(
                    run.coherent.base.frame(ref),
                    transform=run.coherent.base.first.engine.base.p0.image_transform(), device=device)
                value = model.encode({"pixels": image})["emb"]
                need(tuple(value.shape) == (1, 1, spec["latent_dim"]) and bool(torch.isfinite(value).all()),
                     "encoding")
                return value[0, 0]
            goal = encode(spec["goal"])
            mean, std = tensor(spec["action_mean"]), tensor(spec["action_std"])
            need(bool(torch.isfinite(std).all()) and bool((std > 0).all()), "normalizer")
            out = []
            for role in spec["roles"]:
                z = torch.stack([encode(f) for f in role["images"]])[None].repeat(18, 1, 1)
                history = tensor(role["actual_history"]).reshape(1, 2, 5, 2).repeat(18, 1, 1, 1)
                past = ((history - mean) / std).reshape(18, 2, 10)
                actions = tensor([r["actions"] for r in role["rows"]]).reshape(18, 2, 5, 2)
                for k in range(2):
                    context = torch.cat((past, ((actions[:, k] - mean) / std).reshape(18, 1, 10)), dim=1)
                    prediction = model.predict(z, model.action_encoder(context))
                    need(prediction.shape == z.shape and bool(torch.isfinite(prediction).all()), "predict")
                    z = torch.cat((z, prediction[:, -1:]), dim=1)[:, -3:]
                    past = context[:, -2:]
                actual = torch.stack([encode(r["frame"]) for r in role["rows"]])
                for row, predicted, observed in zip(role["rows"], z[:, -1], actual, strict=True):
                    out.append({"arm": role["arm"], "name": row["name"], "feasible": row["feasible"],
                                "guard_dot": row["guard_dot"], "actions_sha256": row["actions_sha256"],
                                "predicted_goal_mse": float((predicted-goal).square().mean().cpu()),
                                "actual_goal_mse": float((observed-goal).square().mean().cpu()),
                                "prediction_mse": float((predicted-observed).square().mean().cpu()),
                                "reward_evaluation_only": row["reward_evaluation_only"]})
        torch.cuda.synchronize(device)
        after = hasher(model)
        need(before == after and spec == protocol(), "frozen sources")
        need(counts == {"encoder": {"calls": 64, "samples": 64},
                        "projector": {"calls": 64, "samples": 64},
                        "action_encoder": {"calls": 6, "samples": 108},
                        "predictor": {"calls": 6, "samples": 108}}, "ledger")
        return base.write_new(output / "report.json", {"schema": NAME + ".report.v1",
            "contract_sha256": digest, "host": host, "model_before": before, "model_after": after,
            "counts": counts, "rows": out, "independent_confirmation": False, "p1_admission": False})
    except Exception as exc:
        base.write_new(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
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
    contract = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    if args.command == "prepare":
        print(base.write_new(contract, protocol()))
    else:
        with run.coherent.base.first.engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            print(execute(contract, args.contract_sha, ROOT / "results" / f"{NAME}_RUN_{args.tag}", args.authority))
