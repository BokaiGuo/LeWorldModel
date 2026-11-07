"""Sealed fixed-prefix and small CEM development comparison at archive frame 25."""
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

from experiments import pc_pointmaze_corrected_camera_retention_20261107_codex as retention
from experiments.pc_input_permissions import STATE, StateAssistedPointMaze, file_hash
from experiments.pc_pointmaze_context2_loader_20261107_codex import configure_context
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

ROOT = retention.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_GOAL25_RETENTION_CEM_PROTOCOL_20261107_CODEX.md"
BASE = ROOT / "results/pc_pointmaze_goal25_retention_cem_20261107_codex"
PUBLIC = Path(str(BASE) + "_input.npz")
PRIVATE = Path(str(BASE) + "_witness.npz")
PLAN = Path(str(BASE) + "_plan.json")
PHYSICAL = Path(str(BASE) + "_physical.json")
RESULT = Path(str(BASE) + "_result.json")
ARMS = (*retention.ARMS, "cem48")


def prepare():
    if PUBLIC.exists() or PRIVATE.exists():
        raise FileExistsError("panel exists")
    folder = retention.archive.FOLDER
    states = torch.load(folder / "states.pth", map_location="cpu", weights_only=True).numpy()
    actions = torch.load(folder / "actions.pth", map_location="cpu", weights_only=True).numpy()
    images, digests = [], []
    for root in range(1884, 1892):
        path = folder / "obses" / f"episode_{root:03d}.pth"
        frames = torch.load(path, map_location="cpu", weights_only=True).numpy()
        images.append(frames[[0, 25]])
        digests.append(file_hash(path))
    images = np.asarray(images)
    if images.shape != (8, 2, 224, 224, 3) or np.abs(actions[1884:1892, :25]).max() > 1:
        raise ValueError("invalid archive panel or witness action")
    with PUBLIC.open("xb") as f:
        np.savez_compressed(f, current=images[:, 0], goal=images[:, 1],
                            proprio=states[1884:1892, 0].astype(np.float32),
                            goal_proprio=states[1884:1892, 25].astype(np.float32),
                            images_sha256=np.asarray(digests), states_sha256=np.asarray(file_hash(folder / "states.pth")))
    with PRIVATE.open("xb") as f:
        np.savez_compressed(f, initial=states[1884:1892, 0], goal=states[1884:1892, 25],
                            actions=actions[1884:1892, :25], actions_sha256=np.asarray(file_hash(folder / "actions.pth")),
                            public_sha256=np.asarray(file_hash(PUBLIC)))
    print("Public goal observations and separate evaluation-only witness prepared.", flush=True)


def cem_candidates(rng, mean, std, count=16):
    if mean.shape != (25, 2) or std.shape != (25, 2) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("invalid CEM distribution")
    samples = np.clip(rng.normal(size=(count, 25, 2)).astype(np.float32) * std + mean, -1, 1)
    samples[0] = np.clip(mean, -1, 1)
    return samples


def plan():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    model, dataset, audit1 = load_model()
    original = json.loads(retention.corrected.PLAN.read_text())
    fingerprint = model_fingerprint(model)
    if fingerprint != original["model"]:
        raise ValueError("model drift")
    audit2 = configure_context(model, audit1)
    model.ctxt_window = 2
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    cfg = yaml.safe_load(retention.corrected.prior.prior.old.inputs.CONFIG.read_text())
    parents, probes, tails = retention.bank()
    bank_a, bank_b = retention.matrix_chains(parents, probes), retention.matrix_chains(parents, tails)
    mean, std = dataset.action_mean.cuda(), dataset.action_std.cuda()
    rows = []
    with np.load(PUBLIC, allow_pickle=False) as inputs, torch.inference_mode():
        for i in range(8):
            root = 1884 + i
            current = {"visual": rgb_tensor(inputs["current"][i]),
                       "proprio": torch.as_tensor(inputs["proprio"][i], dtype=torch.float32, device="cuda")[None, None]}
            goal = {"visual": rgb_tensor(inputs["goal"][i]),
                    "proprio": torch.as_tensor(inputs["goal_proprio"][i], dtype=torch.float32, device="cuda")[None, None]}
            torch.cuda.synchronize()
            started = time.perf_counter()
            target = model.encode(dict(goal))
            torch.cuda.synchronize()
            goal_seconds = time.perf_counter() - started
            objective = ReprTargetDistMPCObjective(cfg, target, alpha=.1)
            arms = {}
            for arm in ARMS:
                boundary = StateAssistedPointMaze(model, audit2)
                boundary._record("encode_goal", goal)
                batches = []
                torch.cuda.synchronize()
                started = time.perf_counter()

                def score(chains):
                    result = []
                    for begin in range(0, len(chains), 4):
                        raw = torch.as_tensor(chains[begin:begin+4], dtype=torch.float32, device="cuda")
                        count = len(raw)
                        packets = ((raw - mean) / std).reshape(count, 5, 10).transpose(0, 1).contiguous()
                        expanded = {k: v.expand(count, *v.shape[1:]).contiguous() for k, v in current.items()}
                        costs = objective(boundary.predict(expanded, packets), packets)
                        if costs.shape != (count,) or not torch.isfinite(costs).all():
                            raise ValueError("bad model score")
                        result.extend(costs.cpu().tolist())
                        batches.append(count)
                    return np.asarray(result)

                if arm != "cem48":
                    pc = score(parents)
                    ac = score(bank_a.reshape(32, 25, 2)).reshape(8, 4)
                    kept = retention.choose_retained(arm, pc, ac, 80310 + root)
                    chosen_b = bank_b[kept].reshape(8, 25, 2)
                    bc = score(chosen_b)
                    candidates = np.concatenate((parents, bank_a.reshape(32, 25, 2), chosen_b))
                    costs = np.concatenate((pc, ac.ravel(), bc))
                    details = {"retained": kept, "parent_cost": pc.tolist(), "probe_cost": ac.tolist(),
                               "tail_cost": bc.tolist()}
                else:
                    rng = np.random.default_rng(7102500 + root)
                    dist_mean, dist_std = np.zeros((25, 2), np.float32), np.full((25, 2), .5, np.float32)
                    pool, values, rounds = [], [], []
                    for _ in range(3):
                        samples = cem_candidates(rng, dist_mean, dist_std)
                        costs_iteration = score(samples)
                        elite = samples[np.argsort(costs_iteration, kind="stable")[:4]]
                        dist_mean, dist_std = elite.mean(axis=0), elite.std(axis=0)
                        pool.append(samples)
                        values.append(costs_iteration)
                        rounds.append(costs_iteration.tolist())
                    candidates, costs = np.concatenate(pool), np.concatenate(values)
                    details = {"round_costs": rounds, "elite_count": 4, "seed": 7102500 + root}
                if len(costs) != 48 or sum(batches) != 48 or np.abs(candidates).max() > 1:
                    raise ValueError("budget or action violation")
                chosen = int(np.argmin(costs))
                torch.cuda.synchronize()
                details.update({"actions": candidates[chosen].tolist(), "chosen_index": chosen,
                                "chosen_cost": float(costs[chosen]), "logical_rollouts": len(costs),
                                "forward_batches": batches, "seconds": time.perf_counter() - started})
                arms[arm] = details
                print(f"root={root} arm={arm} choice={chosen}", flush=True)
            rows.append({"root": root, "goal_encode_seconds": goal_seconds, "arms": arms})
    if model_fingerprint(model) != fingerprint:
        raise ValueError("weights changed")
    with PLAN.open("x") as f:
        json.dump({"track": STATE, "protocol": file_hash(PROTOCOL), "runner": file_hash(__file__),
                   "public": file_hash(PUBLIC), "retention_runner": file_hash(retention.__file__),
                   "config": file_hash(retention.corrected.prior.prior.old.inputs.CONFIG),
                   "model": fingerprint, "rows": rows}, f, indent=2, allow_nan=False)


def check_plan():
    plan_record = json.loads(PLAN.read_text())
    for key, path in (("protocol", PROTOCOL), ("runner", __file__), ("public", PUBLIC),
                      ("retention_runner", retention.__file__),
                      ("config", retention.corrected.prior.prior.old.inputs.CONFIG)):
        if plan_record[key] != file_hash(path):
            raise ValueError(f"plan drift: {key}")
    return plan_record


def physical():
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plans = check_plan()
    rows = []
    with np.load(PRIVATE, allow_pickle=False) as witness, tempfile.TemporaryDirectory(prefix="pc_goal25_") as tmp:
        if str(witness["public_sha256"]) != file_hash(PUBLIC):
            raise ValueError("witness mismatch")
        folder = Path(tmp)
        xml = folder / "scene.xml"
        xml.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        for i, row in enumerate(plans["rows"]):
            if row["root"] != 1884 + i:
                raise ValueError("root mismatch")
            selected = [np.asarray(row["arms"][arm]["actions"], dtype=np.float32) for arm in ARMS]
            candidates = np.stack([*selected, witness["actions"][i], np.zeros((25, 2))])
            if candidates.shape != (8, 25, 2) or not np.isfinite(candidates).all() or np.abs(candidates).max() > 1:
                raise ValueError("invalid actions")
            req, res = folder / f"req_{i}.npz", folder / f"res_{i}.npz"
            with req.open("xb") as f:
                np.savez_compressed(f, state=witness["initial"][i], actions=candidates)
            subprocess.run([str(LEGACY / "bin/python"), str(Path(retention.archive.physics.__file__).resolve()),
                            "--backend", str(req), str(xml), str(res)], env=env, check=True, timeout=120)
            with np.load(res, allow_pickle=False) as f:
                trace = f["trace"].copy()
            if trace.shape != (8, 26, 5):
                raise ValueError("clock mismatch")
            np.testing.assert_allclose(trace[:, :, 4], np.broadcast_to(np.arange(26) * .05, (8, 26)), atol=1e-10, rtol=0)
            goal = witness["goal"][i]
            distance = np.linalg.norm(trace[:, :, :2] - goal[:2], axis=2)
            witness_error = float(np.max(np.abs(trace[len(ARMS), -1, :4] - goal)))
            rows.append({"root": row["root"], "initial_success": bool(distance[0, 0] < .5),
                         "witness_state_max_error": witness_error, "distance": distance.tolist(), "trace": trace.tolist()})
            print(f"root={row['root']} witness_state_error={witness_error:.6g}", flush=True)
    with PHYSICAL.open("x") as f:
        json.dump({"track": STATE, "plan": file_hash(PLAN), "witness": file_hash(PRIVATE),
                   "rows": rows}, f, indent=2, allow_nan=False)


def evaluate():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plans, physical_record = check_plan(), json.loads(PHYSICAL.read_text())
    if physical_record["plan"] != file_hash(PLAN) or physical_record["witness"] != file_hash(PRIVATE):
        raise ValueError("physical drift")
    names = (*ARMS, "archive_witness", "zero")
    rows = []
    for p, real in zip(plans["rows"], physical_record["rows"], strict=True):
        if p["root"] != real["root"]:
            raise ValueError("root mismatch")
        distances = np.asarray(real["distance"])
        if distances.shape != (8, 26):
            raise ValueError("distance clock")
        outcomes = {name: {"final_distance": float(distances[i, -1]),
                           "final_success": bool(distances[i, -1] < .5),
                           "ever_success": bool((distances[i] < .5).any())} for i, name in enumerate(names)}
        rows.append({"root": p["root"], "initial_success": real["initial_success"],
                     "witness_state_max_error": real["witness_state_max_error"], "outcomes": outcomes})
    summary = {}
    for group, selected in (("all", rows), ("initial_not_success", [r for r in rows if not r["initial_success"]])):
        summary[group] = {"tasks": len(selected), "arms": {}}
        for name in names:
            values = [r["outcomes"][name] for r in selected]
            summary[group]["arms"][name] = {"final_success": sum(v["final_success"] for v in values),
                                            "ever_success": sum(v["ever_success"] for v in values),
                                            "mean_final_distance": float(np.mean([v["final_distance"] for v in values]))}
    with RESULT.open("x") as f:
        json.dump({"track": STATE, "plan": file_hash(PLAN), "physical": file_hash(PHYSICAL),
                   "summary": summary, "rows": rows,
                   "scope": "exposed archive goal-25 state-assisted open-loop development"},
                  f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "plan", "physical", "evaluate"):
        raise SystemExit("expected prepare|plan|physical|evaluate")
    {"prepare": prepare, "plan": plan, "physical": physical, "evaluate": evaluate}[sys.argv[1]]()
