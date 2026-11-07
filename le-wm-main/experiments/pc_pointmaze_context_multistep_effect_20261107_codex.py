"""Predict fixed multi-step state-assisted candidates, then replay them physically."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
import yaml

from experiments import pc_pointmaze_archive_context_multistep_20261107_codex as archive
from experiments import pc_pointmaze_multimodal_objective_20261107_codex as inputs
from experiments.pc_input_permissions import STATE, StateAssistedPointMaze, file_hash
from experiments.pc_pointmaze_context2_loader_20261107_codex import configure_context
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_state_effect_local_dev_20261106_codex import rgb_tensor
from experiments.pc_state_track_smoke_20261106_codex import load_model

PROTOCOL = inputs.ROOT.parent / "docs/PC_POINTMAZE_CONTEXT_MULTISTEP_EFFECT_PROTOCOL_20261107_CODEX.md"
BASE = inputs.ROOT / "results/pc_pointmaze_context_multistep_effect_20261107_codex"
PLAN = Path(str(BASE) + "_plan.json")
POOL = Path(str(BASE) + "_pool.npz")
PHYSICAL = Path(str(BASE) + "_physical.json")
RESULT = Path(str(BASE) + "_result.json")


def action_bank():
    rng = np.random.default_rng(27155000)
    a = rng.uniform(-.4, .4, (8, 5, 2)).astype(np.float32)
    a[0] = 0
    return np.repeat(a, 5, axis=1)


def agreement(scores, truth):
    scores, truth = np.asarray(scores), np.asarray(truth)
    if scores.shape != (8,) or truth.shape != (8,) or not np.isfinite(scores).all() or not np.isfinite(truth).all():
        raise ValueError("invalid rank inputs")
    correct = count = 0
    for i in range(8):
        for j in range(i + 1, 8):
            a, b = scores[i] - scores[j], truth[i] - truth[j]
            if abs(a) <= 1e-10 or abs(b) <= 1e-10:
                continue
            count += 1
            correct += int(a * b > 0)
    return correct, count


def plan():
    if PLAN.exists() or POOL.exists():
        raise FileExistsError("sealed plan exists")
    print("Score fixed actions before any physical outcomes are read.", flush=True)
    model, dataset, audit1 = load_model()
    from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
    cfg = yaml.safe_load(inputs.CONFIG.read_text())
    if model.ctxt_window != 1 or cfg["planner"]["planning_objective"]["alpha"] != .1:
        raise ValueError("unexpected source configuration")
    fingerprint = model_fingerprint(model)
    audit2 = configure_context(model, audit1)
    actions = action_bank()
    rows = []
    with np.load(inputs.PUBLIC, allow_pickle=False) as public, torch.inference_mode():
        for index in range(8):
            current = {"visual": rgb_tensor(public["current"][index, 0]),
                       "proprio": torch.as_tensor(public["proprio"][index, 0], dtype=torch.float32, device="cuda")[None, None]}
            goal = {"visual": rgb_tensor(public["goal"][index, 0]),
                    "proprio": torch.as_tensor(public["goal_proprio"][index, 0], dtype=torch.float32, device="cuda")[None, None]}
            context_scores = {}
            for window, audit in ((1, audit1), (2, audit2)):
                model.ctxt_window = window
                boundary = StateAssistedPointMaze(model, audit)
                boundary._record("encode_goal", goal)
                objective = ReprTargetDistMPCObjective(cfg, model.encode(dict(goal)), alpha=.1)
                values = []
                for start in (0, 4):
                    raw = torch.as_tensor(actions[start:start + 4], dtype=torch.float32)
                    packets = model.preprocessor.normalize_actions(raw).reshape(4, 5, 10).transpose(0, 1).contiguous().cuda()
                    expanded = {k: v.expand(4, *v.shape[1:]).contiguous() for k, v in current.items()}
                    scores = objective(boundary.predict(expanded, packets), packets)
                    if scores.shape != (4,) or not torch.isfinite(scores).all():
                        raise ValueError("invalid model scores")
                    values.extend(scores.cpu().tolist())
                context_scores[str(window)] = {"cost": values, "choice": int(np.argmin(values)),
                                               "queries": 8, "receipts": boundary.receipts}
            rows.append({"root": 1884 + index, "windows": context_scores})
            print(f"root={1884+index} scored", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model drift")
    with POOL.open("xb") as f:
        np.savez_compressed(f, actions=actions)
    with PLAN.open("x") as f:
        json.dump({"track": STATE, "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "pool_sha256": file_hash(POOL), "public_sha256": file_hash(inputs.PUBLIC),
                   "config_sha256": file_hash(inputs.CONFIG), "model_sha256": fingerprint,
                   "rows": rows, "scope": "fixed action predictions before physical evaluation"}, f, indent=2)


def physical():
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    if PHYSICAL.exists():
        raise FileExistsError(PHYSICAL)
    record = json.loads(PLAN.read_text())
    for key, path in (("protocol_sha256", PROTOCOL), ("runner_sha256", __file__), ("pool_sha256", POOL),
                      ("public_sha256", inputs.PUBLIC), ("config_sha256", inputs.CONFIG)):
        if record[key] != file_hash(path):
            raise ValueError(f"plan drift: {key}")
    states = torch.load(archive.FOLDER / "states.pth", map_location="cpu", weights_only=True)
    rows = []
    with (np.load(inputs.PUBLIC, allow_pickle=False) as public, np.load(POOL, allow_pickle=False) as pool,
          tempfile.TemporaryDirectory(prefix="pc_context_effect_") as temporary):
        actions = pool["actions"]
        np.testing.assert_array_equal(actions, action_bank())
        folder = Path(temporary)
        xml_path = folder / "scene.xml"
        xml_path.write_text(official_xml())
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        for index, row in enumerate(record["rows"]):
            root = row["root"]
            if root != 1884 + index:
                raise ValueError("root mismatch")
            initial = states[root, 0].numpy()
            np.testing.assert_allclose(initial, public["proprio"][index, 0], atol=1e-7, rtol=0)
            goal = public["goal_proprio"][index, 0, :2]
            request, response = folder / f"req_{root}.npz", folder / f"res_{root}.npz"
            with request.open("xb") as f:
                np.savez_compressed(f, state=initial, actions=actions)
            subprocess.run([str(LEGACY / "bin/python"), str(Path(archive.physics.__file__).resolve()),
                            "--backend", str(request), str(xml_path), str(response)], env=env, check=True, timeout=120)
            with np.load(response, allow_pickle=False) as replay:
                trace = replay["trace"].copy()
            if trace.shape != (8, 26, 5) or not np.isfinite(trace).all():
                raise ValueError("invalid replay")
            np.testing.assert_allclose(trace[:, :, 4], np.broadcast_to(np.arange(26) * .05, (8, 26)), atol=1e-10, rtol=0)
            distance = np.linalg.norm(trace[:, :, :2] - goal, axis=2)
            rows.append({"root": root, "initial_success": bool(distance[0, 0] < .5),
                         "final_distance": distance[:, -1].tolist(), "ever": (distance < .5).any(axis=1).tolist(),
                         "traces": trace.tolist()})
            print(f"root={root} replayed", flush=True)
    with PHYSICAL.open("x") as f:
        json.dump({"track": STATE, "plan_sha256": file_hash(PLAN), "rows": rows}, f, indent=2)


def evaluate():
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan_record, real = json.loads(PLAN.read_text()), json.loads(PHYSICAL.read_text())
    if real["plan_sha256"] != file_hash(PLAN) or plan_record["runner_sha256"] != file_hash(__file__):
        raise ValueError("sealed record drift")
    rows = []
    for p, r in zip(plan_record["rows"], real["rows"], strict=True):
        if p["root"] != r["root"]:
            raise ValueError("root mismatch")
        dist = np.asarray(r["final_distance"])
        cases = {}
        for window in ("1", "2"):
            score = p["windows"][window]["cost"]
            choice = p["windows"][window]["choice"]
            if choice != int(np.argmin(score)):
                raise ValueError("unsealed choice")
            correct, count = agreement(score, dist)
            cases[window] = {"choice": choice, "regret": float(dist[choice] - dist.min()),
                             "ever": r["ever"][choice], "correct_pairs": correct, "pair_count": count,
                             "rank_accuracy": correct / count if count else None}
        rows.append({"root": p["root"], "initial_success": r["initial_success"], "windows": cases})
    summaries = {}
    for group, selected in (("all", rows), ("initial_not_success", [r for r in rows if not r["initial_success"]])):
        summaries[group] = {"tasks": len(selected)}
        for window in ("1", "2"):
            vals = [r["windows"][window] for r in selected]
            ranks = [v["rank_accuracy"] for v in vals if v["rank_accuracy"] is not None]
            summaries[group][window] = {"mean_rank_accuracy": float(np.mean(ranks)) if ranks else None,
                                         "mean_regret": float(np.mean([v["regret"] for v in vals])),
                                         "ever": sum(v["ever"] for v in vals)}
    a, b = summaries["all"]["1"], summaries["all"]["2"]
    summaries["development_signal"] = bool(b["mean_rank_accuracy"] > a["mean_rank_accuracy"] and b["mean_regret"] < a["mean_regret"])
    with RESULT.open("x") as f:
        json.dump({"plan_sha256": file_hash(PLAN), "physical_sha256": file_hash(PHYSICAL), "rows": rows,
                   "summary": summaries, "scope": "exposed state-assisted fixed-bank evaluation"}, f, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical", "evaluate"):
        raise SystemExit("expected plan|physical|evaluate")
    {"plan": plan, "physical": physical, "evaluate": evaluate}[sys.argv[1]]()
