"""Compile full two-packet action response at actual dev304 RGB feedback states.

Fixed comparison: original feedback plan, four +/- spectral perturbations,
and four random-direction controls. Model-only; no outcome-based selection.
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

from . import p1_dev304_diverse_feedback_saved_20261102 as previous
from . import p1_dev303_goal_tangent_builder_20261102_exclusive as goal_metric

f, base, run = previous.f, previous.base, previous.run
ROOT = run.ROOT
NAME = "p1_dev304_qgoal_spectral_20261102"
AUTHORITY = "DEV304_QGOAL_SPECTRAL_CUDA"
need = run.require


def protocol():
    previous.analyze()
    source = base.read(ROOT/f"results/{f.NAME}_PREP_DEV01.json", previous.physical.CONTRACT_SHA)
    plan = base.read(ROOT/f"results/{f.NAME}_PLAN_DEV01/report.json", previous.physical.PLAN_SHA)
    roles = []
    for role, row in zip(source["roles"], plan["rows"], strict=True):
        need(role["arm"] == row["arm"], "source roster")
        anchor = np.asarray(row["selected"]["actions"], dtype="<f4")
        need(anchor.shape == (2, 5, 2)
             and hashlib.sha256(anchor.tobytes()).hexdigest() == row["selected_actions_sha256"], "anchor")
        roles.append({"arm": role["arm"], "images": role["images"], "actual_history": role["actual_history"],
                      "prefix_sha256": role["prefix_sha256"], "anchor": anchor.tolist(),
                      "anchor_sha256": row["selected_actions_sha256"]})
    return {"schema": NAME+".protocol.v1", "classification": "known_root_post_outcome_development",
            "code_sha256": base.sha(__file__), "input_sha256": previous.physical.CONTRACT_SHA,
            "anchor_report_sha256": previous.physical.PLAN_SHA, "model_hash": source["model_hash"],
            "policy": source["policy"], "goal": source["goal"], "action_mean": source["action_mean"],
            "action_std": source["action_std"], "latent_dim": source["latent_dim"], "roles": roles,
            "proposal": "goal-RGB tangent metric used consistently for spectral modes and nonlinear score",
            "metric_source_sha256": base.sha(goal_metric.__file__),
            "random_control": "two Gaussian directions, both signs, radius 0.15, seed 91823",
            "radius": .15, "fd_epsilon": .001, "fd_is_unbounded_local_diagnostic": True,
            "selection": "minimum nonlinear goal-tangent weighted terminal MSE among nine fixed candidates",
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
            goal_rgb = run.coherent.base.frame(spec["goal"])
            metric_np, metric_info = goal_metric.build(model, goal_rgb, device)
            need(metric_np.shape == (spec["latent_dim"], spec["latent_dim"])
                 and np.isfinite(metric_np).all(), "goal metric")
            goal_np = np.asarray(metric_info["goal_latent"], dtype=np.float64)
            need(goal_np.shape == (spec["latent_dim"],) and np.isfinite(goal_np).all(), "goal")
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
                weighted_jac = metric_np @ jac.astype(np.float64)
                u, singular, vh = np.linalg.svd(weighted_jac, full_matrices=False)
                counts["svd_calls"] += 1
                projections = u.T @ (metric_np @ (goal_np-nominal))
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
                scores = (((outputs-goal_np) @ metric_np.T)**2).mean(axis=1)
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
                "metric": metric_np.tolist(), "metric_info": metric_info,
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
