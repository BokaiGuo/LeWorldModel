"""Compare open predictions to actual-RGB-refreshed predictions for the same actions."""
import argparse
import os

import numpy as np

from . import p1_dev305_compass_midrgb_replay_20261102 as replay

base, need, run, ROOT = replay.base, replay.need, replay.run, replay.ROOT
NAME = "p1_dev305_compass_midrgb_model_20261102"
AUTHORITY = "DEV305_COMPASS_MIDRGB_MODEL_CUDA"
REPLAY_CONTRACT_SHA = "73d68ac059386c6e9e51b4a6ab7d4c7b45a13677b7c46c73df6abdd6ca99ea7e"
REPLAY_REPORT_SHA = "698ff89e4cdcfb6bf379a37b777dae5ae1902459d1c1b0350c33ece5d71180b8"


def protocol():
    contract = base.read(ROOT / f"results/{replay.NAME}_PREP_DEV01.json", REPLAY_CONTRACT_SHA)
    report = base.read(ROOT / f"results/{replay.NAME}_RUN_DEV01/report.json", REPLAY_REPORT_SHA)
    need(contract == replay.protocol() and not report["errors"] and len(report["rows"]) == 54
         and report["contract_sha256"] == REPLAY_CONTRACT_SHA, "replay source")
    old_contract = base.read(ROOT / f"results/{replay.compare.NAME}_PREP_DEV01.json",
                             replay.checked.CONTRACT_SHA)
    old_report = base.read(ROOT / f"results/{replay.compare.NAME}_RUN_DEV01/report.json",
                           replay.checked.REPORT_SHA)
    directory = ROOT / f"results/{replay.NAME}_RUN_DEV01"
    roles = []
    for role in old_contract["roles"]:
        rows = []
        prior_role = next(x for x in contract["roles"] if x["arm"] == role["arm"])
        for item in role["rows"]:
            stored = next(x for x in report["rows"] if x["arm"] == role["arm"] and x["name"] == item["name"])
            prior = next(x for x in old_report["rows"] if x["arm"] == role["arm"] and x["name"] == item["name"])
            need(stored["status"] == "fixed_horizon_completed" and prior["actions_sha256"] == item["actions_sha256"],
                 "paired branch")
            frame = stored["midpoint"]
            reference = {"path": str((directory / frame["file"]).resolve()),
                         "file_sha256": frame["file_sha256"], "pixel_sha256": frame["pixel_sha256"]}
            actions = np.asarray(item["actions"], dtype="<f4")
            history = np.asarray(role["actual_history"], dtype="<f4").reshape(2, 5, 2)
            need(actions.shape == (10, 2), "action shape")
            rows.append({"name": item["name"], "images": [role["images"][1], role["images"][2], reference],
                         "history": np.concatenate((history[-1], actions[:5])).tolist(),
                         "action": actions[5:].tolist(), "final": item["frame"],
                         "open_goal_mse": prior["predicted_goal_mse"],
                         "actual_goal_mse": prior["actual_goal_mse"], "open_prediction_mse": prior["prediction_mse"],
                         "reward_evaluation_only": prior["reward_evaluation_only"]})
        need(len(rows) == 18 and len(prior_role["branches"]) == 18, "roster")
        roles.append({"arm": role["arm"], "rows": rows})
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "source_report_sha256": REPLAY_REPORT_SHA, "model_hash": old_contract["model_hash"],
            "policy": old_contract["policy"], "goal": old_contract["goal"],
            "action_mean": old_contract["action_mean"], "action_std": old_contract["action_std"],
            "latent_dim": old_contract["latent_dim"], "roles": roles,
            "classification": "post_outcome_development_same_action_not_independent",
            "p1_admission": False, "full_cost_certified": False}


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
                    run.coherent.base.frame(ref), transform=run.coherent.base.first.engine.base.p0.image_transform(),
                    device=device)
                value = model.encode({"pixels": image})["emb"]
                need(tuple(value.shape) == (1, 1, spec["latent_dim"]) and bool(torch.isfinite(value).all()), "encoding")
                return value[0, 0]
            goal = encode(spec["goal"])
            mean, std = tensor(spec["action_mean"]), tensor(spec["action_std"])
            need(bool(torch.isfinite(std).all()) and bool((std > 0).all()), "normalizer")
            out = []
            for role in spec["roles"]:
                observations = torch.stack([torch.stack([encode(f) for f in row["images"]]) for row in role["rows"]])
                histories = tensor([row["history"] for row in role["rows"]]).reshape(18, 2, 5, 2)
                actions = tensor([row["action"] for row in role["rows"]]).reshape(18, 1, 5, 2)
                tokens = ((torch.cat((histories, actions), dim=1) - mean) / std).reshape(18, 3, 10)
                prediction = model.predict(observations, model.action_encoder(tokens))
                need(prediction.shape == observations.shape and bool(torch.isfinite(prediction).all()), "predict")
                actual = torch.stack([encode(row["final"]) for row in role["rows"]])
                for item, predicted, observed in zip(role["rows"], prediction[:, -1], actual, strict=True):
                    actual_goal = float((observed-goal).square().mean().cpu())
                    need(np.isclose(actual_goal, item["actual_goal_mse"], atol=2e-6, rtol=0), "actual encoding")
                    out.append({"arm": role["arm"], "name": item["name"],
                                "open_goal_mse": item["open_goal_mse"], "actual_goal_mse": actual_goal,
                                "refreshed_goal_mse": float((predicted-goal).square().mean().cpu()),
                                "open_prediction_mse": item["open_prediction_mse"],
                                "refreshed_prediction_mse": float((predicted-observed).square().mean().cpu()),
                                "reward_evaluation_only": item["reward_evaluation_only"]})
        torch.cuda.synchronize(device)
        need(before == hasher(model) and spec == protocol(), "model and source")
        need(counts == {"encoder": {"calls": 217, "samples": 217},
                        "projector": {"calls": 217, "samples": 217},
                        "action_encoder": {"calls": 3, "samples": 54},
                        "predictor": {"calls": 3, "samples": 54}}, "ledger")
        return base.write_new(output / "report.json", {"schema": NAME + ".report.v1",
            "contract_sha256": digest, "host": host, "model_before": before, "model_after": before,
            "counts": counts, "rows": out, "independent_confirmation": False, "p1_admission": False})
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
