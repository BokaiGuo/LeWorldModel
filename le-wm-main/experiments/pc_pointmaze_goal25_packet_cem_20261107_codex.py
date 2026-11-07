"""Matched action parameterization CEM on sealed state-assisted goal-25 panel."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import yaml

from experiments import pc_pointmaze_goal25_retention_cem_20261107_codex as prior
from experiments.pc_input_permissions import STATE, StateAssistedPointMaze, file_hash
from experiments.pc_pointmaze_context2_loader_20261107_codex import configure_context
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_GOAL25_PACKET_CEM_PROTOCOL_20261107_CODEX.md"
BASE = ROOT / "results/pc_pointmaze_goal25_packet_cem_20261107_codex"
CANDIDATES = Path(str(BASE) + "_candidates.npz")
PLAN = Path(str(BASE) + "_plan.json")
PHYSICAL = Path(str(BASE) + "_physical.json")
RESULT = Path(str(BASE) + "_result.json")


def packet_samples(rng, mean, std):
    mean, std = np.asarray(mean), np.asarray(std)
    if (mean.shape != (5, 2) or std.shape != (5, 2) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std < 0).any()):
        raise ValueError("invalid packet distribution")
    samples = np.clip(rng.normal(size=(16, 5, 2)).astype(np.float32) * std + mean, -1, 1)
    samples[0] = np.clip(mean, -1, 1)
    return samples


def select_scored(costs, limit):
    costs = np.asarray(costs)
    if costs.shape != (160,) or not np.isfinite(costs).all() or limit not in (48, 160):
        raise ValueError("invalid scored budget")
    return int(np.argmin(costs[:limit]))


def plan():
    if CANDIDATES.exists() or PLAN.exists():
        raise FileExistsError("sealed plan exists")
    previous = prior.check_plan()
    model, dataset, audit1 = load_model()
    fingerprint = model_fingerprint(model)
    if fingerprint != previous["model"]:
        raise ValueError("model drift")
    audit2 = configure_context(model, audit1)
    model.ctxt_window = 2
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    cfg = yaml.safe_load(prior.retention.corrected.prior.prior.old.inputs.CONFIG.read_text())
    action_mean, action_std = dataset.action_mean.cuda(), dataset.action_std.cuda()
    rows, arrays = [], {}
    with np.load(prior.PUBLIC, allow_pickle=False) as inputs, torch.inference_mode():
        for index in range(8):
            root = 1884 + index
            current = {"visual": rgb_tensor(inputs["current"][index]),
                       "proprio": torch.as_tensor(inputs["proprio"][index], dtype=torch.float32, device="cuda")[None, None]}
            goal = {"visual": rgb_tensor(inputs["goal"][index]),
                    "proprio": torch.as_tensor(inputs["goal_proprio"][index], dtype=torch.float32, device="cuda")[None, None]}
            torch.cuda.synchronize()
            start = time.perf_counter()
            target = model.encode(dict(goal))
            torch.cuda.synchronize()
            goal_seconds = time.perf_counter() - start
            objective = ReprTargetDistMPCObjective(cfg, target, alpha=.1)
            boundary = StateAssistedPointMaze(model, audit2)
            boundary._record("encode_goal", goal)
            rng = np.random.default_rng(7102500 + root)
            mean, std = np.zeros((5, 2), np.float32), np.full((5, 2), .5, np.float32)
            all_actions, all_costs, rounds, batch_lengths = [], [], [], []
            torch.cuda.synchronize()
            started = time.perf_counter()
            seconds48 = None
            for step in range(10):
                packets_raw = packet_samples(rng, mean, std)
                chains = np.repeat(packets_raw, 5, axis=1)
                costs = []
                for begin in range(0, 16, 4):
                    raw = torch.as_tensor(chains[begin:begin+4], dtype=torch.float32, device="cuda")
                    packets = ((raw - action_mean) / action_std).reshape(4, 5, 10).transpose(0, 1).contiguous()
                    expanded = {k: v.expand(4, *v.shape[1:]).contiguous() for k, v in current.items()}
                    values = objective(boundary.predict(expanded, packets), packets)
                    if values.shape != (4,) or not torch.isfinite(values).all():
                        raise ValueError("invalid model scores")
                    costs.extend(values.cpu().tolist())
                    batch_lengths.append(4)
                costs = np.asarray(costs)
                elite = packets_raw[np.argsort(costs, kind="stable")[:4]]
                rounds.append({"mean": mean.tolist(), "std": std.tolist(), "costs": costs.tolist()})
                mean, std = elite.mean(axis=0), elite.std(axis=0)
                all_actions.append(chains)
                all_costs.append(costs)
                if step == 2:
                    torch.cuda.synchronize()
                    seconds48 = time.perf_counter() - started
            torch.cuda.synchronize()
            seconds160 = time.perf_counter() - started
            actions = np.concatenate(all_actions)
            costs = np.concatenate(all_costs)
            if actions.shape != (160, 25, 2) or sum(batch_lengths) != 160:
                raise ValueError("bad budget")
            chosen = {str(limit): select_scored(costs, limit) for limit in (48, 160)}
            arrays[f"r{root}_actions"] = actions
            arrays[f"r{root}_costs"] = costs
            rows.append({"root": root, "seed": 7102500 + root, "chosen": chosen,
                         "selected": {str(limit): {"actions": actions[chosen[str(limit)]].tolist(),
                                                    "cost": float(costs[chosen[str(limit)]])}
                                      for limit in (48, 160)},
                         "rounds": rounds, "forward_batches": batch_lengths,
                         "goal_encode_seconds": goal_seconds, "seconds48": seconds48, "seconds160": seconds160})
            print(f"root={root} packet CEM selections={chosen}", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with CANDIDATES.open("xb") as f:
        np.savez_compressed(f, **arrays)
    with PLAN.open("x") as f:
        json.dump({"track": STATE, "protocol": file_hash(PROTOCOL), "runner": file_hash(__file__),
                   "old_plan": file_hash(prior.PLAN), "public": file_hash(prior.PUBLIC),
                   "candidates": file_hash(CANDIDATES), "model": fingerprint, "rows": rows},
                  f, indent=2, allow_nan=False)


def check_plan():
    plan_record = json.loads(PLAN.read_text())
    for key, path in (("protocol", PROTOCOL), ("runner", __file__), ("old_plan", prior.PLAN),
                      ("public", prior.PUBLIC), ("candidates", CANDIDATES)):
        if plan_record[key] != file_hash(path):
            raise ValueError(f"plan drift: {key}")
    return plan_record


def physical():
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plans, old = check_plan(), prior.check_plan()
    old_real = json.loads(prior.PHYSICAL.read_text())
    if old_real["plan"] != file_hash(prior.PLAN) or old_real["witness"] != file_hash(prior.PRIVATE):
        raise ValueError("prior physical drift")
    rows = []
    with np.load(prior.PRIVATE, allow_pickle=False) as witness, tempfile.TemporaryDirectory(prefix="pc_packet_cem_") as tmp:
        if str(witness["public_sha256"]) != file_hash(prior.PUBLIC):
            raise ValueError("witness mismatch")
        folder = Path(tmp)
        xml = folder / "scene.xml"
        xml.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        for index, row in enumerate(plans["rows"]):
            if row["root"] != 1884 + index or old["rows"][index]["root"] != row["root"]:
                raise ValueError("root mismatch")
            chains = np.stack([row["selected"]["48"]["actions"], row["selected"]["160"]["actions"],
                               old["rows"][index]["arms"]["cem48"]["actions"]]).astype(np.float32)
            if chains.shape != (3, 25, 2) or not np.isfinite(chains).all() or np.abs(chains).max() > 1:
                raise ValueError("illegal actions")
            req, res = folder / f"req_{index}.npz", folder / f"res_{index}.npz"
            with req.open("xb") as f:
                np.savez_compressed(f, state=witness["initial"][index], actions=chains)
            subprocess.run([str(LEGACY / "bin/python"), str(Path(prior.retention.archive.physics.__file__).resolve()),
                            "--backend", str(req), str(xml), str(res)], env=env, check=True, timeout=120)
            with np.load(res, allow_pickle=False) as f:
                trace = f["trace"].copy()
            if trace.shape != (3, 26, 5):
                raise ValueError("invalid physical clock")
            np.testing.assert_allclose(trace[:, :, 4], np.broadcast_to(np.arange(26) * .05, (3, 26)), atol=1e-10, rtol=0)
            old_trace = np.asarray(old_real["rows"][index]["trace"])[len(prior.retention.ARMS)]
            delta = float(np.max(np.abs(trace[2] - old_trace)))
            if delta > 1e-7:
                raise ValueError(f"previous CEM replay differs: {delta}")
            distances = np.linalg.norm(trace[:, :, :2] - witness["goal"][index, :2], axis=2)
            rows.append({"root": row["root"], "initial_success": bool(distances[0, 0] < .5),
                         "prior_cem_trace_max_abs_error": delta, "distance": distances.tolist(),
                         "trace": trace.tolist()})
            print(f"root={row['root']} prior CEM replay max={delta:.3g}", flush=True)
    with PHYSICAL.open("x") as f:
        json.dump({"track": STATE, "plan": file_hash(PLAN), "old_physical": file_hash(prior.PHYSICAL),
                   "witness": file_hash(prior.PRIVATE), "rows": rows}, f, indent=2, allow_nan=False)


def evaluate():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan_record, real = check_plan(), json.loads(PHYSICAL.read_text())
    if real["plan"] != file_hash(PLAN) or real["old_physical"] != file_hash(prior.PHYSICAL):
        raise ValueError("physical drift")
    names = ("packet_cem48", "packet_cem160", "full_cem48")
    rows = []
    with np.load(CANDIDATES, allow_pickle=False) as candidates:
        for p, r in zip(plan_record["rows"], real["rows"], strict=True):
            if p["root"] != r["root"]:
                raise ValueError("root mismatch")
            costs, actions = candidates[f"r{p['root']}_costs"], candidates[f"r{p['root']}_actions"]
            for limit in (48, 160):
                index = select_scored(costs, limit)
                if index != p["chosen"][str(limit)]:
                    raise ValueError("selected index mismatch")
                np.testing.assert_array_equal(actions[index], np.asarray(p["selected"][str(limit)]["actions"], np.float32))
            dist = np.asarray(r["distance"])
            if dist.shape != (3, 26):
                raise ValueError("distance mismatch")
            rows.append({"root": p["root"], "initial_success": r["initial_success"],
                         "outcomes": {name: {"final_distance": float(dist[i, -1]),
                                             "final_success": bool(dist[i, -1] < .5),
                                             "ever_success": bool((dist[i] < .5).any())}
                                      for i, name in enumerate(names)}})
    summary = {}
    for group, subset in (("all", rows), ("initial_not_success", [r for r in rows if not r["initial_success"]])):
        summary[group] = {"tasks": len(subset), "arms": {}}
        for name in names:
            vals = [r["outcomes"][name] for r in subset]
            summary[group]["arms"][name] = {"final_success": sum(v["final_success"] for v in vals),
                                            "mean_final_distance": float(np.mean([v["final_distance"] for v in vals]))}
    with RESULT.open("x") as f:
        json.dump({"track": STATE, "plan": file_hash(PLAN), "physical": file_hash(PHYSICAL),
                   "summary": summary, "rows": rows,
                   "scope": "exposed state-assisted open-loop development; not a tuned baseline"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical", "evaluate"):
        raise SystemExit("expected plan|physical|evaluate")
    {"plan": plan, "physical": physical, "evaluate": evaluate}[sys.argv[1]]()
