"""Score fixed candidate bank using an RGB-observation-derived tangent metric."""
import argparse
import os

import numpy as np

from . import p1_dev305_two_packet_model_bank_20261102 as original
from . import p1_dev305_two_packet_model_bank_saved_20261102 as old
from . import p1_rgb_tangent_metric_20261101_main as tangent
from . import p1_rgb_visibility_20261101_main as visibility

base, need, run, ROOT = original.base, original.need, original.run, original.ROOT
NAME = "p1_dev305_component_tangent_bank_20261102"
AUTHORITY = "DEV305_COMPONENT_TANGENT_BANK_CUDA"


def component_scores(delta, jacobian):
    """Ridge-readout squared coordinates, in the fixed perturbation units."""
    delta, jacobian = np.asarray(delta, dtype=np.float64), np.asarray(jacobian, dtype=np.float64)
    need(jacobian.ndim == 2 and jacobian.shape[1] == 5, "jacobian shape")
    need(delta.shape[-1] == jacobian.shape[0], "latent shape")
    need(np.isfinite(delta).all() and np.isfinite(jacobian).all(), "nonfinite input")
    singular = np.linalg.svd(jacobian, compute_uv=False)
    need(singular[0] > 0 and np.isfinite(singular).all(), "zero jacobian")
    ridge = .01 * singular[0] ** 2
    readout = np.linalg.solve(jacobian.T @ jacobian + ridge * np.eye(5), jacobian.T)
    coordinates = delta @ readout.T
    squared = coordinates ** 2
    return {"object": squared[..., :3].sum(axis=-1),
            "pusher": squared[..., 3:].sum(axis=-1),
            "joint": squared.sum(axis=-1)}, ridge, singular


def protocol():
    source = base.read(ROOT / "results" / f"{original.NAME}_PREP_DEV01.json", old.CONTRACT_SHA)
    need(source == original.protocol(), "source")
    goal = run.coherent.base.frame(source["goal"])
    goal_visibility = visibility.inspect(goal)
    roles = []
    for role in source["roles"]:
        pixels = run.coherent.base.frame(role["images"][-1])
        screen = visibility.inspect(pixels)
        need(screen["eligible"], "current observation fails unchanged visibility guard")
        roles.append({"arm": role["arm"], "images": role["images"], "actual_history": role["actual_history"],
                      "visibility": screen})
    return {"schema": NAME + ".protocol.v1", "code_sha256": base.sha(__file__),
            "tangent_code_sha256": base.sha(tangent.__file__), "visibility_code_sha256": base.sha(visibility.__file__),
            "source_contract_sha256": old.CONTRACT_SHA, "old_report_sha256": old.REPORT_SHA,
            "policy": source["policy"], "model_hash": source["model_hash"],
            "action_mean": source["action_mean"], "action_std": source["action_std"],
            "goal": source["goal"], "goal_visibility": goal_visibility,
            "roles": roles, "bank": source["bank"], "latent_dim": source["latent_dim"],
            "metric": "five current RGB perturbations; ridge .01 times largest singular value squared; squared object, pusher, joint coordinate sums",
            "selection": "independent minimum of each fixed component score over the same two-packet bank",
            "classification": "post_outcome_known_root_development_not_confirmatory",
            "no_simulator_state_in_selection": True, "p1_admission": False, "independent_confirmation": False}


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
            entry = counts.setdefault(label, {"calls": 0, "samples": 0})
            entry["calls"] += 1
            entry["samples"] += int(args[0].shape[0])
        handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode(), torch.autocast("cuda", enabled=False), sdpa_kernel(SDPBackend.MATH):
            def encode(pixels):
                image = run.coherent.base.first.engine.base.p0._image_tensor(
                    pixels, transform=run.coherent.base.first.engine.base.p0.image_transform(), device=device)
                value = model.encode({"pixels": image})["emb"]
                need(tuple(value.shape) == (1, 1, spec["latent_dim"]) and bool(torch.isfinite(value).all()), "encode")
                return value[0, 0]
            goal = encode(run.coherent.base.frame(spec["goal"]))[None]
            bank = tensor([x["action"] for x in spec["bank"]])
            packet = bank[:, None, :].repeat(1, 5, 1)
            old_report = base.read(ROOT / "results" / f"{original.NAME}_RUN_DEV01/report.json", old.REPORT_SHA)
            rows = []
            for role in spec["roles"]:
                pixels = run.coherent.base.frame(role["images"][-1])
                derivatives = []
                for axis in tangent.AXES:
                    plus = encode(tangent.perturb(pixels, axis, 1))
                    minus = encode(tangent.perturb(pixels, axis, -1))
                    derivatives.append(((plus - minus) / 2).cpu().numpy())
                jacobian = np.stack(derivatives, axis=1)
                z0 = torch.stack([encode(run.coherent.base.frame(x)) for x in role["images"]])[None]
                history = ((tensor(role["actual_history"]).reshape(2, 5, 2) - mean) / std).reshape(1, 2, 10)
                deltas, baseline = [], []
                for i in range(17):
                    z, past = z0.repeat(17, 1, 1), history.repeat(17, 1, 1)
                    for k in range(2):
                        raw = packet[i].unsqueeze(0).repeat(17, 1, 1) if k == 0 else packet
                        context = torch.cat((past, ((raw - mean) / std).reshape(17, 1, 10)), dim=1)
                        predicted = model.predict(z, model.action_encoder(context))
                        need(predicted.shape == z.shape and bool(torch.isfinite(predicted).all()), "predict")
                        z = torch.cat((z, predicted[:, -1:]), dim=1)[:, -3:]
                        past = context[:, -2:]
                    delta = z[:, -1] - goal
                    deltas.append(delta.cpu().numpy())
                    baseline.append(delta.square().mean(dim=1).cpu().tolist())
                deltas, baseline = np.asarray(deltas), np.asarray(baseline)
                old_row = next(x for x in old_report["rows"] if x["arm"] == role["arm"])
                need(np.allclose(baseline, old_row["terminal_scores"], atol=2e-6, rtol=0),
                     "baseline rollout differs from sealed same-action model")
                scores, ridge, singular = component_scores(deltas, jacobian)
                selected = {key: list(map(int, np.unravel_index(int(np.argmin(value)), value.shape)))
                            for key, value in scores.items()}
                rows.append({"arm": role["arm"], "selected_pair": selected,
                             "scores": {key: value.tolist() for key, value in scores.items()},
                             "ridge": ridge, "singular_values": singular.tolist(),
                             "jacobian": jacobian.tolist(),
                             "baseline_max_abs_difference": float(np.max(np.abs(baseline - old_row["terminal_scores"])))})
        torch.cuda.synchronize(device)
        after = hasher(model)
        need(before == after and spec == protocol(), "frozen source")
        need(counts == {"encoder": {"calls": 40, "samples": 40}, "projector": {"calls": 40, "samples": 40},
                        "action_encoder": {"calls": 102, "samples": 1734},
                        "predictor": {"calls": 102, "samples": 1734}}, "ledger")
        return base.write_new(output / "report.json", {"schema": NAME + ".report.v1",
            "contract_sha256": digest, "model_before": before, "model_after": after, "host": host,
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
