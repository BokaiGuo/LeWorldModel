"""Equal-opportunity fixed-prefix retention on a corrected state-assisted camera."""
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

from experiments import pc_pointmaze_camera_corrected_effect_20261107_codex as corrected
from experiments import pc_pointmaze_archive_context_multistep_20261107_codex as archive
from experiments.pc_input_permissions import STATE, StateAssistedPointMaze, file_hash
from experiments.pc_pointmaze_context2_loader_20261107_codex import configure_context
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

ROOT = corrected.ROOT
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_CORRECTED_CAMERA_RETENTION_PROTOCOL_20261107_CODEX.md"
BASE = ROOT / "results/pc_pointmaze_corrected_camera_retention_20261107_codex"
BANK = Path(str(BASE) + "_bank.npz")
PLAN = Path(str(BASE) + "_plan.json")
PHYSICAL = Path(str(BASE) + "_physical.json")
RESULT = Path(str(BASE) + "_result.json")
ARMS = ("uniform", "parent", "lookahead", "vector", "shuffled")


def bank():
    parents = corrected.prior.prior.old.action_bank()
    rng_a = np.random.default_rng(4176501)
    rng_b = np.random.default_rng(4176502)
    probes = np.repeat(rng_a.uniform(-.4, .4, (4, 3, 2)).astype(np.float32), 5, axis=1)
    tails = np.repeat(rng_b.uniform(-.4, .4, (4, 3, 2)).astype(np.float32), 5, axis=1)
    return parents, probes, tails


def choose_retained(arm, parent_cost, probe_cost, seed):
    parent_cost, probe_cost = np.asarray(parent_cost), np.asarray(probe_cost)
    if (arm not in ARMS or parent_cost.shape != (8,) or probe_cost.shape != (8, 4)
            or not np.isfinite(parent_cost).all() or not np.isfinite(probe_cost).all()):
        raise ValueError("invalid retention inputs")
    if arm == "uniform":
        return [0, 1]
    if arm == "parent":
        return np.argsort(parent_cost, kind="stable")[:2].tolist()
    order = np.argsort(probe_cost.min(axis=1), kind="stable")
    if arm == "lookahead":
        return order[:2].tolist()
    vectors = probe_cost - probe_cost.mean(axis=1, keepdims=True)
    if arm == "shuffled":
        rng = np.random.default_rng(seed)
        vectors = np.stack([v[rng.permutation(4)] for v in vectors])
    first = int(order[0])
    distance = np.square(vectors - vectors[first]).sum(axis=1)
    distance[first] = -np.inf
    return [first, int(np.argmax(distance))]


def matrix_chains(parents, suffix):
    parents, suffix = np.asarray(parents), np.asarray(suffix)
    if parents.shape != (8, 25, 2) or suffix.shape != (4, 15, 2):
        raise ValueError("invalid program bank")
    prefixes = np.broadcast_to(parents[:, None, :10], (8, 4, 10, 2))
    suffixes = np.broadcast_to(suffix[None], (8, 4, 15, 2))
    return np.concatenate((prefixes, suffixes), axis=2)


def plan():
    if BANK.exists() or PLAN.exists():
        raise FileExistsError("selection already exists")
    corrected.check_plan()
    model, dataset, audit1 = load_model()
    source = json.loads(corrected.PLAN.read_text())
    fingerprint = model_fingerprint(model)
    if fingerprint != source["model"] or audit1.fingerprint != json.loads(corrected.prior.prior.PLAN.read_text())["audit1"]:
        raise ValueError("model drift")
    audit2 = configure_context(model, audit1)
    model.ctxt_window = 2
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    cfg = yaml.safe_load(corrected.prior.prior.old.inputs.CONFIG.read_text())
    parents, probes, tails = bank()
    bank_a, bank_b = matrix_chains(parents, probes), matrix_chains(parents, tails)
    mean, std = dataset.action_mean.cuda(), dataset.action_std.cuda()
    rows = []
    with np.load(corrected.INPUT, allow_pickle=False) as inputs, torch.inference_mode():
        for index in range(8):
            root = 1884 + index
            current = {"visual": rgb_tensor(inputs["current"][index]),
                       "proprio": torch.as_tensor(inputs["proprio"][index], dtype=torch.float32, device="cuda")[None, None]}
            goal = {"visual": rgb_tensor(inputs["goal"][index]),
                    "proprio": torch.as_tensor(inputs["goal_proprio"][index], dtype=torch.float32, device="cuda")[None, None]}
            torch.cuda.synchronize()
            goal_started = time.perf_counter()
            target = model.encode(dict(goal))
            torch.cuda.synchronize()
            goal_encode_seconds = time.perf_counter() - goal_started
            objective = ReprTargetDistMPCObjective(cfg, target, alpha=.1)
            arms = {}
            for arm in ARMS:
                boundary = StateAssistedPointMaze(model, audit2)
                boundary._record("encode_goal", goal)
                batch_lengths = []
                torch.cuda.synchronize()
                start = time.perf_counter()

                def score(chains):
                    scores = []
                    for pos in range(0, len(chains), 4):
                        raw = torch.as_tensor(chains[pos:pos + 4], dtype=torch.float32, device="cuda")
                        count = len(raw)
                        packets = ((raw - mean) / std).reshape(count, 5, 10).transpose(0, 1).contiguous()
                        expanded = {k: v.expand(count, *v.shape[1:]).contiguous() for k, v in current.items()}
                        values = objective(boundary.predict(expanded, packets), packets)
                        if values.shape != (count,) or not torch.isfinite(values).all():
                            raise ValueError("bad scores")
                        scores.extend(values.cpu().tolist())
                        batch_lengths.append(count)
                    return np.asarray(scores)

                parent_cost = score(parents)
                probe_cost = score(bank_a.reshape(32, 25, 2)).reshape(8, 4)
                retained = choose_retained(arm, parent_cost, probe_cost, 80310 + root)
                selected_b = bank_b[retained].reshape(8, 25, 2)
                tail_cost = score(selected_b)
                combined = np.concatenate((parent_cost, probe_cost.ravel(), tail_cost))
                actions = np.concatenate((parents, bank_a.reshape(32, 25, 2), selected_b), axis=0)
                choice = int(np.argmin(combined))
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                arms[arm] = {"retained": retained, "parent_cost": parent_cost.tolist(),
                             "probe_cost": probe_cost.tolist(), "tail_cost": tail_cost.tolist(),
                             "chosen_index": choice, "chosen_cost": float(combined[choice]),
                             "actions": actions[choice].tolist(), "logical_rollouts": len(combined),
                             "forward_batches": batch_lengths, "synchronized_seconds": elapsed,
                             "goal_encoding_count": 1, "goal_encode_shared_measured_separately": True}
                print(f"root={root} {arm} kept={retained} choice={choice}", flush=True)
            rows.append({"root": root, "goal_encode_synchronized_seconds": goal_encode_seconds, "arms": arms})
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with BANK.open("xb") as f:
        np.savez_compressed(f, parents=parents, probes=probes, tails=tails)
    with PLAN.open("x") as f:
        json.dump({"track": STATE, "protocol": file_hash(PROTOCOL), "runner": file_hash(__file__),
                   "input": file_hash(corrected.INPUT), "source_plan": file_hash(corrected.PLAN),
                   "bank": file_hash(BANK), "model": fingerprint, "rows": rows,
                   "scope": "state-assisted fixed-bank predictions, before physical outcomes"},
                  f, indent=2, allow_nan=False)


def check_plan():
    plan = json.loads(PLAN.read_text())
    for key, path in (("protocol", PROTOCOL), ("runner", __file__), ("input", corrected.INPUT),
                      ("source_plan", corrected.PLAN), ("bank", BANK)):
        if plan[key] != file_hash(path):
            raise ValueError(f"sealed drift: {key}")
    return plan


def physical():
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    plan_record = check_plan()
    states = torch.load(archive.FOLDER / "states.pth", map_location="cpu", weights_only=True)
    rows = []
    with np.load(BANK, allow_pickle=False) as pool, np.load(corrected.INPUT, allow_pickle=False) as inputs, tempfile.TemporaryDirectory(prefix="pc_retention_physical_") as temp:
        bank_b = matrix_chains(pool["parents"], pool["tails"]).reshape(32, 25, 2)
        folder = Path(temp)
        xml = folder / "scene.xml"
        xml.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        for index, entry in enumerate(plan_record["rows"]):
            root = 1884 + index
            if entry["root"] != root:
                raise ValueError("root mismatch")
            initial = states[root, 0].numpy()
            np.testing.assert_allclose(initial, inputs["proprio"][index], atol=1e-7, rtol=0)
            chosen = np.stack([np.asarray(entry["arms"][arm]["actions"], dtype=np.float32) for arm in ARMS])
            chains = np.concatenate((chosen, bank_b))
            if chains.shape != (37, 25, 2) or not np.isfinite(chains).all() or np.abs(chains).max() > 1:
                raise ValueError("illegal selected chain")
            req, res = folder / f"req_{root}.npz", folder / f"res_{root}.npz"
            with req.open("xb") as f:
                np.savez_compressed(f, state=initial, actions=chains)
            subprocess.run([str(LEGACY / "bin/python"), str(Path(archive.physics.__file__).resolve()),
                            "--backend", str(req), str(xml), str(res)], env=env, check=True, timeout=120)
            with np.load(res, allow_pickle=False) as f:
                trace = f["trace"].copy()
            if trace.shape != (37, 26, 5):
                raise ValueError("physical clock")
            np.testing.assert_allclose(trace[:, :, 4], np.broadcast_to(np.arange(26) * .05, (37, 26)), atol=1e-10, rtol=0)
            goal = inputs["goal_proprio"][index, :2]
            distance = np.linalg.norm(trace[:, :, :2] - goal, axis=2)
            rows.append({"root": root, "initial_success": bool(distance[0, 0] < .5),
                         "distances": distance.tolist(), "traces": trace.tolist()})
            print(f"root={root} selected chains and evaluation-only B bank replayed", flush=True)
    with PHYSICAL.open("x") as f:
        json.dump({"track": STATE, "plan": file_hash(PLAN), "bank": file_hash(BANK),
                   "state_sha256": file_hash(archive.FOLDER / "states.pth"),
                   "rows": rows}, f, indent=2, allow_nan=False)


def evaluate():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan_record, real = check_plan(), json.loads(PHYSICAL.read_text())
    if real["plan"] != file_hash(PLAN) or real["bank"] != file_hash(BANK):
        raise ValueError("physical drift")
    rows = []
    for p, r in zip(plan_record["rows"], real["rows"], strict=True):
        if p["root"] != r["root"]:
            raise ValueError("root mismatch")
        dist = np.asarray(r["distances"])
        if dist.shape != (37, 26):
            raise ValueError("distance shape")
        oracle_b = dist[len(ARMS):, -1].reshape(8, 4)
        oracle_value = float(oracle_b.min())
        arms = {}
        for i, arm in enumerate(ARMS):
            retained = p["arms"][arm]["retained"]
            best = float(oracle_b[retained].min())
            arms[arm] = {"final_distance": float(dist[i, -1]), "final_success": bool(dist[i, -1] < .5),
                         "ever_success": bool((dist[i] < .5).any()),
                         "retained_b_physical_regret": best - oracle_value,
                         "retained_b_oracle": bool(best <= oracle_value + 1e-10)}
        rows.append({"root": p["root"], "initial_success": r["initial_success"],
                     "physical_b_oracle_distance": oracle_value, "arms": arms})
    summary = {}
    for group, selected in (("all", rows), ("initial_not_success", [r for r in rows if not r["initial_success"]])):
        summary[group] = {"tasks": len(selected), "arms": {}}
        for arm in ARMS:
            vals = [r["arms"][arm] for r in selected]
            summary[group]["arms"][arm] = {
                "mean_final_distance": float(np.mean([v["final_distance"] for v in vals])),
                "final_success": sum(v["final_success"] for v in vals),
                "ever_success": sum(v["ever_success"] for v in vals),
                "mean_retained_b_physical_regret": float(np.mean([v["retained_b_physical_regret"] for v in vals])),
                "retained_b_oracle": sum(v["retained_b_oracle"] for v in vals)}
    with RESULT.open("x") as f:
        json.dump({"track": STATE, "plan": file_hash(PLAN), "physical": file_hash(PHYSICAL),
                   "scope": "exposed fixed-bank state-assisted open-loop development, not a CEM comparison",
                   "summary": summary, "rows": rows}, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical", "evaluate"):
        raise SystemExit("expected plan|physical|evaluate")
    {"plan": plan, "physical": physical, "evaluate": evaluate}[sys.argv[1]]()
