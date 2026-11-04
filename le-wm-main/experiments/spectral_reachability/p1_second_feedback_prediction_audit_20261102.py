"""Post-outcome, same-cp10 old/new action prediction decomposition.

The physical outcome is used exclusively for analysis.  Nothing here is an
online selector, an independent confirmation, or a planning advantage result.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from . import p1_second_feedback_replan_20261102 as planner
from . import p1_second_feedback_saved_20261102 as saved
from . import p1_residual_cp5_timing_saved_20261102 as previous

ROOT = planner.ROOT
NAME = "p1_second_feedback_prediction_audit_20261102"
AUTHORITY = "DEV303_SECOND_FEEDBACK_POST_OUTCOME_CUDA"
require = planner.require


def classify(pred_old, pred_new, real_old, real_new):
    values = np.asarray([pred_old, pred_new, real_old, real_new], dtype=np.float64)
    require(np.isfinite(values).all(), "finite distance")
    pred_delta, real_delta = float(pred_new - pred_old), float(real_new - real_old)
    return {"predicted_new_minus_old": pred_delta, "actual_new_minus_old": real_delta,
            "ranking": ("tie" if pred_delta == 0 or real_delta == 0 else
                        "same" if np.sign(pred_delta) == np.sign(real_delta) else "opposite")}


def image_ref(row, phase, step, directory):
    found = [o for o in row["observations"] if o["phase"] == phase and o["step"] == step]
    require(len(found) == 1, "unique verified image")
    obs = found[0]
    path = directory / obs["file"]
    require(path.parent == directory and not path.is_symlink(), "local image")
    return {"path": str(path.resolve()), "file_sha256": obs["file_sha256"],
            "pixel_sha256": obs["pixel_sha256"]}


def protocol():
    summary = saved.analyze()
    old_contract = saved.physical.previous.read(previous.CONTRACT, previous.CONTRACT_SHA)
    old_report = saved.physical.previous.read(previous.RUN / "report.json", previous.REPORT_SHA)
    new_contract = saved.physical.previous.read(saved.CONTRACT, saved.CONTRACT_SHA)
    new_report = saved.physical.previous.read(saved.RUN / "report.json", saved.REPORT_SHA)
    model_contract = planner.first.fixed(
        ROOT / "results/p1_second_feedback_replan_20261102_PREP_DEV01.json",
        "d667aa06e82cdebb1e72abbedcd0d5ebe7110866a622621a5774170936ca74c0")
    require(len(summary["rows"]) == 8 and summary["frames_verified"] == 112,
            "physical verification")
    names = [[x["arm"] for x in items] for items in
             (old_contract["roles"], old_report["rows"], new_contract["roles"],
              new_report["rows"], model_contract["rows"], summary["rows"])]
    require(all(name == names[0] for name in names[1:]), "paired roster")
    rows = []
    for oc, old, nc, new, model_row, measured in zip(
            old_contract["roles"], old_report["rows"], new_contract["roles"], new_report["rows"],
            model_contract["rows"], summary["rows"], strict=True):
        old_packet = np.asarray(oc["tail"][5:10], dtype="<f4")
        new_packet = np.asarray(nc["new_first_packet"], dtype="<f4")
        require(old_packet.shape == new_packet.shape == (5, 2)
                and np.array_equal(np.asarray(oc["tail"][:5], dtype="<f4"),
                                   np.asarray(nc["previous_first_packet"], dtype="<f4")),
                "action packet binding")
        rows.append({"arm": oc["arm"], "rgb_cp5": model_row["rgb_cp5"],
                     "rgb_cp10": model_row["rgb_cp10"],
                     "history_raw_packets": model_row["executed_raw_action_packets"],
                     "old_packet": old_packet.tolist(), "new_packet": new_packet.tolist(),
                     "old_final_rgb": image_ref(old, "tail", 10, previous.RUN),
                     "new_final_rgb": image_ref(new, "new_tail", 5, saved.RUN),
                     "physical_evaluation_only": measured})
    return {"schema": NAME + ".protocol.v1", "classification": "post_outcome_development_only",
            "source_physical_sha256": saved.REPORT_SHA, "source_old_sha256": previous.REPORT_SHA,
            "model_hash": model_contract["model_hash"], "policy": model_contract["policy"],
            "task": model_contract["task"], "rows": rows,
            "per_arm": {"rgb_encoder_calls": 4, "action_encoder_calls": 2, "predictor_calls": 2},
            "prediction": "identical actual cp10 RGB/history, old or new five raw actions, one prediction packet",
            "no_selection_or_physical_execution": True,
            "source_sha256": {str(Path(__file__).resolve()): planner.first.file_sha(__file__),
                              str(Path(planner.__file__).resolve()): planner.first.file_sha(planner.__file__)},
            "P1_admission": False, "independent_confirmation": False}


def run(contract, sha, output, authority):
    require(authority == AUTHORITY and planner.first.fixed(contract, sha) == protocol(),
            "contract closure and authority")
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
    spec = planner.first.fixed(contract, sha)
    model = load_compatible_pretrained(spec["policy"], device=device).eval().requires_grad_(False)
    hasher = planner.first.engine.base.blind.host.signal.model_hash
    before = hasher(model)
    require(before == spec["model_hash"], "frozen model binding")
    task = spec["task"]
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    mean, std, goal = tensor(task["action_mean"]), tensor(task["action_std"]), tensor(task["goal"])[None]
    rows, counts, hooks = [], {}, []
    phase = {"name": ""}
    for name, module in (("encoder", model.encoder), ("projector", model.projector),
                         ("action_encoder", model.action_encoder), ("predictor", model.predictor)):
        def hook(_module, args, name=name):
            key = phase["name"] + ":" + name
            val = counts.setdefault(key, {"calls": 0, "samples": 0})
            val["calls"] += 1
            val["samples"] += int(args[0].shape[0])
        hooks.append(module.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            for row in spec["rows"]:
                phase["name"] = row["arm"] + ":encode"
                encoded = {}
                for key in ("rgb_cp5", "rgb_cp10", "old_final_rgb", "new_final_rgb"):
                    img = planner.first.engine.base.p0._image_tensor(
                        planner.frame(row[key]), transform=planner.first.engine.base.p0.image_transform(),
                        device=device)
                    value = model.encode({"pixels": img})["emb"]
                    require(tuple(value.shape) == (1, 1, task["latent_dim"])
                            and bool(torch.isfinite(value).all()), "finite RGB encoding")
                    encoded[key] = value[0, 0]
                z, past = planner.context(task, encoded["rgb_cp5"].cpu().numpy(),
                                          encoded["rgb_cp10"].cpu().numpy(), row["history_raw_packets"])
                z, past = tensor(z)[None], tensor(past)[None]
                phase["name"] = row["arm"] + ":predict"
                outputs = {}
                for label in ("old", "new"):
                    packet = tensor(row[label + "_packet"])
                    act = ((packet - mean) / std).reshape(1, 1, 10)
                    prediction = model.predict(z, model.action_encoder(torch.cat((past, act), dim=1)))
                    require(prediction.shape == z.shape and bool(torch.isfinite(prediction).all()),
                            "finite prediction")
                    outputs[label] = prediction[0, -1]
                mse = lambda a, b: float((a - b).square().mean().item())
                distances = {"pred_old": mse(outputs["old"], goal[0]),
                             "pred_new": mse(outputs["new"], goal[0]),
                             "actual_old": mse(encoded["old_final_rgb"], goal[0]),
                             "actual_new": mse(encoded["new_final_rgb"], goal[0])}
                result = {"arm": row["arm"], "goal_mse": distances,
                          "ranking": classify(*[distances[k] for k in
                                                ("pred_old", "pred_new", "actual_old", "actual_new")]),
                          "prediction_mse": {"old": mse(outputs["old"], encoded["old_final_rgb"]),
                                             "new": mse(outputs["new"], encoded["new_final_rgb"])},
                          "physical_evaluation_only": row["physical_evaluation_only"],
                          "vectors": {name: value.cpu().tolist() for name, value in
                                      (("pred_old", outputs["old"]), ("pred_new", outputs["new"]),
                                       ("actual_old", encoded["old_final_rgb"]),
                                       ("actual_new", encoded["new_final_rgb"]))}}
                rows.append(result)
                planner.first.write_new(output / (row["arm"] + ".json"), result)
        torch.cuda.synchronize(device)
        after = hasher(model)
        require(before == after and len(rows) == 8, "checkpoint or roster")
        for row in spec["rows"]:
            for name in ("encoder", "projector"):
                require(counts[row["arm"] + ":encode:" + name] == {"calls": 4, "samples": 4}, "encode ledger")
            for name in ("action_encoder", "predictor"):
                require(counts[row["arm"] + ":predict:" + name] == {"calls": 2, "samples": 2}, "predict ledger")
        require(planner.first.fixed(contract, sha) == protocol(), "postrun source closure")
        report = {"schema": NAME + ".report.v1", "protocol_sha256": sha,
                  "classification": "post_outcome_development_only", "host": host,
                  "model_before": before, "model_after": after, "rows": rows, "hook_ledger": counts,
                  "P1_admission": False, "independent_confirmation": False}
        return planner.first.write_new(output / "report.json", report)
    except Exception as exc:
        planner.first.write_new(output / "failure.json", {"type": type(exc).__name__, "error": str(exc),
                                                          "phase": phase["name"], "completed_rows": len(rows),
                                                          "hook_ledger": counts})
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
    contract = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    if args.command == "prepare":
        require(not args.sha256 and not args.authority and not os.path.lexists(output), "prepare only")
        print(planner.first.write_new(contract, protocol()))
    else:
        require(args.sha256 and args.authority, "missing authority or hash")
        with planner.first.engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            print(run(contract, args.sha256, output, args.authority))


if __name__ == "__main__":
    main()
