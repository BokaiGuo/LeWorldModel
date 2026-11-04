"""Post-outcome model ranking on the fixed privileged-oracle candidate bank."""
import argparse
import os
from pathlib import Path

import numpy as np

from . import p1_cp10_physics_oracle_saved_20261102 as saved
from . import p1_second_feedback_replan_20261102 as model_base

ROOT = saved.ROOT
NAME = "p1_cp10_oracle_model_ranking_20261102"
AUTHORITY = "DEV303_CP10_ORACLE_RANKING_CUDA"
require = saved.need


def protocol():
    check = saved.analyze()
    oracle = saved.oracle.saved.physical.previous.read(saved.CONTRACT, saved.CONTRACT_SHA)
    model = model_base.protocol()
    m_by = {r["arm"]: r for r in model["rows"]}
    require(len(check["rows"]) == len(oracle["arms"]) == 2, "verified oracle")
    rows = []
    for item in oracle["arms"]:
        r = m_by[item["arm"]]
        actions = np.asarray(item["candidate_actions"], dtype="<f4")
        require(actions.shape == (64, 5, 2), "full candidate bank")
        rows.append({"arm": item["arm"], "rgb_cp5": r["rgb_cp5"], "rgb_cp10": r["rgb_cp10"],
                     "history": r["executed_raw_action_packets"], "actions": item["candidate_actions"],
                     "candidate_sha256": item["candidate_sha256"]})
    return {"schema": NAME + ".protocol.v1", "classification": "post_outcome_development_diagnostic",
            "rows": rows, "model_hash": model["model_hash"], "policy": model["policy"],
            "task": model["task"], "oracle_source_sha256": saved.REPORT_SHA,
            "model_input": "actual RGB and action history plus complete fixed action bank",
            "oracle_state_used_in_model": False, "physical_outcomes_used_in_model": False,
            "per_arm": {"encode_calls": 2, "predict_calls": 1, "predict_samples": 64},
            "source_sha256": saved.oracle.saved.physical.previous.sha(__file__),
            "P1_admission": False, "independent_confirmation": False}


def run(contract, digest, output, authority):
    prev = saved.oracle.saved.physical.previous
    require(authority == AUTHORITY and prev.sha(contract) == digest and prev.read(contract) == protocol(),
            "contract and authority")
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
    spec = prev.read(contract)
    model = load_compatible_pretrained(spec["policy"], device=device).eval().requires_grad_(False)
    hasher = model_base.first.engine.base.blind.host.signal.model_hash
    before = hasher(model)
    require(before == spec["model_hash"], "model identity")
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
                    image = model_base.first.engine.base.p0._image_tensor(
                        model_base.frame(row[key]),
                        transform=model_base.first.engine.base.p0.image_transform(), device=device)
                    encoded = model.encode({"pixels": image})["emb"]
                    require(encoded.shape == (1, 1, task["latent_dim"]) and bool(torch.isfinite(encoded).all()),
                            "observation encoding")
                    latents.append(encoded[0, 0].cpu().numpy())
                z, past = model_base.context(task, *latents, row["history"])
                states, history = tensor(z)[None].repeat(64, 1, 1), tensor(past)[None].repeat(64, 1, 1)
                actions = tensor(row["actions"])
                phase["name"] = row["arm"] + ":predict"
                packet = ((actions - mean) / std).reshape(64, 1, 10)
                output_tensor = model.predict(states, model.action_encoder(torch.cat((history, packet), dim=1)))
                require(output_tensor.shape == states.shape and bool(torch.isfinite(output_tensor).all()), "prediction")
                scores = (output_tensor[:, -1] - goal).square().mean(dim=1).cpu().numpy()
                result = {"arm": row["arm"], "candidate_sha256": row["candidate_sha256"],
                          "predicted_latent_costs": scores.tolist(),
                          "model_argmin_index": int(np.argmin(scores))}
                rows.append(result)
                prev.write_new(output / (row["arm"] + ".json"), result)
        torch.cuda.synchronize(device)
        after = hasher(model)
        require(before == after and len(rows) == 2, "checkpoint and roster")
        for row in spec["rows"]:
            for name in ("encoder", "projector"):
                require(ledger[row["arm"] + ":encode:" + name] == {"calls": 2, "samples": 2}, "encode ledger")
            for name in ("action_encoder", "predictor"):
                require(ledger[row["arm"] + ":predict:" + name] == {"calls": 1, "samples": 64}, "predict ledger")
        require(prev.read(contract) == protocol(), "source closure")
        return prev.write_new(output / "report.json", {"schema": NAME + ".report.v1",
                 "classification": "post_outcome_development_diagnostic", "protocol_sha256": digest,
                 "host": host, "model_before": before, "model_after": after, "rows": rows, "hook_ledger": ledger,
                 "P1_admission": False, "independent_confirmation": False})
    except Exception as exc:
        prev.write_new(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                "phase": phase["name"], "completed_rows": len(rows)})
        raise
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "run"))
    p.add_argument("--tag", required=True)
    p.add_argument("--sha256")
    p.add_argument("--authority")
    a = p.parse_args()
    require(a.tag.isascii() and a.tag.isalnum(), "tag")
    contract = ROOT / "results" / f"{NAME}_PREP_{a.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{a.tag}"
    prev = saved.oracle.saved.physical.previous
    if a.command == "prepare":
        require(not a.sha256 and not a.authority and not os.path.lexists(output), "prepare")
        print(prev.write_new(contract, protocol()))
    else:
        require(a.sha256 and a.authority, "authorization")
        with model_base.first.engine.base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            print(run(contract, a.sha256, output, a.authority))
