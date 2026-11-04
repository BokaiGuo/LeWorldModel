"""Compile full two-packet action response at actual dev304 RGB feedback states.

Fixed comparison: original feedback plan, four +/- spectral perturbations,
and four random-direction controls. Model-only; no outcome-based selection.
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

from . import p1_dev305_diverse_saved_20261102 as previous

run = previous.run
base = run.base
ROOT = run.ROOT
NAME = "p1_dev305_goalrank_spectral_20261102"
AUTHORITY = "DEV305_GOALRANK_SPECTRAL_CUDA"
need = run.require


def protocol():
    previous.analyze()
    source = base.read(ROOT/f"results/{run.NAME}_PREP_DEV01.json", previous.SPEC_SHA)
    capture = base.read(ROOT/f"results/{run.NAME}_CAPTURE_DEV01/report.json", previous.CAPTURE_SHA)
    execution_dir = ROOT/f"results/{run.NAME}_EXEC_DEV01"
    execution = base.read(execution_dir/"report.json", previous.EXEC_SHA)
    plan = base.read(ROOT/f"results/{run.NAME}_PLAN_DEV01/report.json", previous.PLAN_SHA)
    blind = base.read(ROOT/f"results/{run.NAME}_CAPTURE_DEV01/blind.json", capture["blind_sha256"])
    goal = dict(blind["goal"])
    goal["path"] = str((ROOT/f"results/{run.NAME}_CAPTURE_DEV01"/goal.pop("file")).resolve())
    roles = []
    for physical_row, row in zip(execution["rows"], plan["rows"], strict=True):
        need(physical_row["arm"] == row["arm"] and physical_row["status"] == "fixed_horizon_completed",
             "source roster and complete prefix")
        original = np.asarray(row["selected"]["actions"], dtype="<f4")
        need(original.shape == (2, 5, 2)
             and hashlib.sha256(original.tobytes()).hexdigest() == row["selected_actions_sha256"], "anchor source")
        anchor = np.stack((original[1], original[1]))
        images = []
        for obs in (physical_row["observations"][1], physical_row["observations"][2],
                    physical_row["observations"][-1]):
            images.append({"path": str((execution_dir/obs["file"]).resolve()),
                           "file_sha256": obs["file_sha256"], "pixel_sha256": obs["pixel_sha256"]})
        roles.append({"arm": row["arm"], "images": images,
                      "actual_history": np.concatenate((np.zeros((5, 2), dtype="<f4"), original[0])).tolist(),
                      "prefix_sha256": hashlib.sha256(original[0].tobytes()).hexdigest(),
                      "anchor": anchor.tolist(),
                      "anchor_sha256": hashlib.sha256(anchor.tobytes()).hexdigest()})
    return {"schema": NAME+".protocol.v1", "classification": "development_history_and_training_membership_unknown",
            "code_sha256": base.sha(__file__), "input_sha256": previous.EXEC_SHA,
            "anchor_report_sha256": previous.PLAN_SHA, "model_hash": source["model_hash"],
            "policy": source["policy"], "goal": goal, "action_mean": source["action_mean"],
            "action_std": source["action_std"], "latent_dim": source["latent_dim"], "roles": roles,
            "proposal": "rank SVD modes by damped local objective contribution, both signs, radius 0.15",
            "random_control": "two Gaussian directions, both signs, radius 0.15, seed 91823",
            "radius": .15, "fd_epsilon": .001, "fd_is_unbounded_local_diagnostic": True,
            "selection": "minimum nonlinear terminal latent MSE among nine fixed candidates",
            "anchor_definition": "repeat the saved unexecuted second packet twice; not a newly optimized tail",
            "p1_admission": False, "independent_confirmation": False, "full_cost_certified": False}


def run_cuda(contract_path, digest, output, authority):
    need(authority == AUTHORITY and base.read(contract_path, digest) == protocol()
         and not os.path.lexists(output), "contract/authority")
    spec = base.read(contract_path, digest)
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
    need(before == spec["model_hash"], "model identity")
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    mean, std = tensor(spec["action_mean"]), tensor(spec["action_std"])
    need(bool(torch.isfinite(std).all()) and bool((std > 0).all()), "normalizer")
    counts, handles, rows = {"vjp_attempted": 0, "vjp_returned": 0, "svd_calls": 0}, [], []
    for name, module in (("encoder", model.encoder), ("projector", model.projector),
                         ("action_encoder", model.action_encoder), ("predictor", model.predictor)):
        def hook(_module, args, name=name):
            entry = counts.setdefault(name, {"calls": 0, "samples": 0})
            entry["calls"] += 1
            entry["samples"] += int(args[0].shape[0])
        handles.append(module.register_forward_pre_hook(hook))
    try:
        with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
            def encode(ref):
                image = run.coherent.base.first.engine.base.p0._image_tensor(
                    run.coherent.base.frame(ref),
                    transform=run.coherent.base.first.engine.base.p0.image_transform(), device=device)
                with torch.no_grad():
                    z = model.encode({"pixels": image})["emb"]
                need(z.shape == (1, 1, spec["latent_dim"]) and bool(torch.isfinite(z).all()), "encoding")
                return z.detach()
            goal = encode(spec["goal"])[0, 0]
            rng = np.random.default_rng(91823)
            for role in spec["roles"]:
                z0 = torch.cat([encode(x) for x in role["images"]], dim=1)
                history = ((tensor(role["actual_history"]).reshape(1, 2, 5, 2)-mean)/std).reshape(1, 2, 10)
                def rollout(actions):
                    batch = actions.shape[0]
                    z, past = z0.repeat(batch, 1, 1), history.repeat(batch, 1, 1)
                    for packet in range(2):
                        token = ((actions[:, packet]-mean)/std).reshape(batch, 1, 10)
                        history_new = torch.cat((past, token), dim=1)
                        pred = model.predict(z, model.action_encoder(history_new))
                        need(pred.shape == z.shape and bool(torch.isfinite(pred).all()), "prediction")
                        z = torch.cat((z, pred[:, -1:]), dim=1)[:, -3:]
                        past = history_new[:, -2:]
                    return z[:, -1]
                anchor = tensor(role["anchor"]).detach().clone().requires_grad_(True)
                terminal = rollout(anchor[None])[0]
                jacobian = []
                for k in range(spec["latent_dim"]):
                    counts["vjp_attempted"] += 1
                    grad = torch.autograd.grad(terminal[k], anchor, retain_graph=k < spec["latent_dim"]-1)[0]
                    counts["vjp_returned"] += 1
                    jacobian.append(grad.detach().reshape(-1))
                jac = torch.stack(jacobian).cpu().numpy()
                nominal = terminal.detach().cpu().numpy()
                goal_np = goal.detach().cpu().numpy()
                u, singular, vh = np.linalg.svd(jac.astype(np.float64), full_matrices=False)
                counts["svd_calls"] += 1
                projections = u.T @ (goal_np-nominal)
                contributions = singular**2/(singular**2+.01) * projections**2
                modes = np.argsort(-contributions, kind="stable")[:2]
                directions = []
                for mode in modes:
                    d = vh[mode].copy()
                    if d[np.argmax(np.abs(d))] < 0:
                        d *= -1
                    directions.append(("spectral", int(mode), d/np.max(np.abs(d))))
                for index in range(2):
                    d = rng.standard_normal(20)
                    directions.append(("random", index, d/np.max(np.abs(d))))
                anchor_np = np.asarray(role["anchor"], dtype=np.float32)
                candidates = [("anchor", -1, 0, anchor_np)]
                for kind, index, direction in directions:
                    for sign in (-1, 1):
                        action = np.clip(anchor_np+sign*spec["radius"]*direction.reshape(2, 5, 2), -1, 1)
                        candidates.append((kind, index, sign, action.astype("<f4")))
                with torch.no_grad():
                    outputs = rollout(tensor(np.stack([x[3] for x in candidates]))).cpu().numpy()
                    d = directions[0][2].reshape(2, 5, 2)
                    plus = rollout(tensor((anchor_np+spec["fd_epsilon"]*d)[None]))[0].cpu().numpy()
                    minus = rollout(tensor((anchor_np-spec["fd_epsilon"]*d)[None]))[0].cpu().numpy()
                finite_diff = (plus-minus)/(2*spec["fd_epsilon"])
                projected = jac @ d.reshape(-1)
                err = float(np.linalg.norm(finite_diff-projected)/(np.linalg.norm(projected)+1e-12))
                scores = ((outputs-goal_np)**2).mean(axis=1)
                candidate_rows = [{"kind": kind, "index": index, "sign": sign, "actions": action.tolist(),
                                   "actions_sha256": hashlib.sha256(action.astype("<f4").tobytes()).hexdigest(),
                                   "terminal": out.tolist(), "score": float(score)}
                                  for (kind, index, sign, action), out, score in zip(candidates, outputs, scores, strict=True)]
                row = {"arm": role["arm"], "prefix_sha256": role["prefix_sha256"],
                       "anchor_terminal": nominal.tolist(), "jacobian": jac.tolist(),
                       "singular_values": singular.tolist(), "mode_indices": modes.tolist(),
                       "directions": [{"kind": k, "index": i, "direction": d.tolist()} for k, i, d in directions],
                       "fd_direction": directions[0][2].tolist(), "fd_observed": finite_diff.tolist(),
                       "fd_jacobian": projected.tolist(), "fd_relative_error": err,
                       "candidates": candidate_rows, "selected_index": int(np.argmin(scores))}
                rows.append(row)
                base.write_new(output/(role["arm"]+".json"), row)
        torch.cuda.synchronize(device)
        after = hasher(model)
        need(before == after and spec == protocol() and counts["vjp_returned"] == 384, "model/source/VJP")
        return base.write_new(output/"report.json", {"schema": NAME+".report.v1", "contract_sha256": digest,
                "host": host, "model_before": before, "model_after": after, "counts": counts,
                "rows": rows, "p1_admission": False, "independent_confirmation": False,
                "full_cost_certified": False})
    except Exception as exc:
        base.write_new(output/"failure.json", {"type": type(exc).__name__, "message": str(exc)[:1000], "counts": counts})
        raise
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "run"))
    p.add_argument("--tag", required=True)
    p.add_argument("--contract-sha")
    p.add_argument("--authority")
    a = p.parse_args()
    need(a.tag.isascii() and a.tag.isalnum(), "tag")
    contract = ROOT/"results"/f"{NAME}_PREP_{a.tag}.json"
    if a.command == "prepare":
        print(base.write_new(contract, protocol()))
    else:
        with run.coherent.base.first.engine.base.locking.gpu_lock(ROOT/"results/e3_p2_host_cuda_cooperative.lock"):
            print(run_cuda(contract, a.contract_sha, ROOT/"results"/f"{NAME}_RUN_{a.tag}", a.authority))
