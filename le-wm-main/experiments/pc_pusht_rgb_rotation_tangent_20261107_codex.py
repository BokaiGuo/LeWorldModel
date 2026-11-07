"""Sealed RGB-goal tangent scores followed by isolated physical-label evaluation."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from experiments import pc_pusht_rgb_goal_rotation_score_20261107_codex as old
from experiments.checkpoint_compat import load_compatible_pretrained
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_pusht_shared_pool_real_endpoint_20261107_codex import OUT as PHYSICAL, summarize

OUT = old.ROOT / "results/pc_pusht_rgb_rotation_tangent_20261107_codex"
PROTOCOL = old.ROOT.parent / "docs/PC_PUSHT_RGB_ROTATION_TANGENT_PROTOCOL_20261107_CODEX.md"
METHODS = ("original", "rotation", "random")


def projected_cost(predictions, goal, direction):
    pred = np.asarray(predictions, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if (pred.ndim != 2 or goal.shape != (pred.shape[1],) or direction.shape != goal.shape
            or not np.isfinite(pred).all() or not np.isfinite(goal).all()):
        raise ValueError("invalid input")
    diff = pred - goal
    if not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-6:
        return np.sum(diff * diff, axis=1), True
    unit = direction / np.linalg.norm(direction)
    residual = diff - (diff @ unit)[:, None] * unit[None, :]
    return np.sum(residual * residual, axis=1), False


def score():
    if OUT.with_suffix(".json").exists() or OUT.with_suffix(".npz").exists():
        raise FileExistsError(OUT)
    rotation = json.loads(old.OUT.with_suffix(".json").read_text())
    prior = json.loads(old.PRIOR.with_suffix(".json").read_text())
    source = json.loads(old.SOURCE.with_suffix(".json").read_text())
    if (file_hash(old.OUT.with_suffix(".npz")) != rotation["arrays_sha256"]
            or file_hash(old.PRIOR.with_suffix(".npz")) != prior["arrays_sha256"]
            or file_hash(old.SOURCE.with_suffix(".npz")) != source["arrays_sha256"]):
        raise ValueError("source arrays drift")
    model = load_compatible_pretrained("pusht/lewm", device="cuda").eval().requires_grad_(False)
    fingerprint = model_fingerprint(model)
    if any(fingerprint != x["model_sha256"] for x in (rotation, prior, source)):
        raise ValueError("checkpoint mismatch")
    rows, arrays = [], {}
    with (np.load(old.OUT.with_suffix(".npz"), allow_pickle=False) as goal_data,
          np.load(old.SOURCE.with_suffix(".npz"), allow_pickle=False) as data,
          torch.inference_mode()):
        for task in range(4):
            record = prior["plans"][task]
            index = next(i for i, row in enumerate(source["rows"]) if row["task"] == task and row["selected"])
            goals = goal_data[f"goals_{task}"]
            if goals.shape != (3, 192) or rotation["rows"][task]["detection"]["abstained"]:
                raise ValueError("sealed goal variants absent")
            actions = []
            batches = [torch.load(old.BASE / "capture" / f"call_{task * 5 + i:03}.pt",
                                  map_location="cpu", weights_only=True) for i in range(5)]
            for candidate in record["candidates"]:
                action = batches[candidate["call"]]["actions"][0, candidate["slot"]].numpy()
                if hashlib.sha256(action.tobytes()).hexdigest() != candidate["sha256"]:
                    raise ValueError("candidate identity drift")
                actions.append(action)
            actions = torch.as_tensor(np.stack(actions), device="cuda")
            observed = torch.as_tensor(data["actual"][index, 1:2], device="cuda")
            pred = old.conditional_rollout(model, observed, actions[:, 2:], 0)[:, -1].cpu().numpy()
            if pred.shape != (301, 192):
                raise ValueError("unexpected model prediction shape")
            original = np.sum(np.square(pred.astype(np.float64) - goals[0]), axis=1)
            if np.max(np.abs(original - goal_data[f"original_{task}"])) > 0.002:
                raise ValueError("original scoring parity failure")
            tangent = goals[2].astype(np.float64) - goals[1].astype(np.float64)
            random_direction = np.random.default_rng(20261107100 + task).normal(size=192)
            rotation_score, abstain = projected_cost(pred, goals[0], tangent)
            random_score, random_abstain = projected_cost(pred, goals[0], random_direction)
            if random_abstain:
                raise ValueError("random direction invalid")
            values = {"original": original, "rotation": rotation_score, "random": random_score}
            winners = {key: int(np.argmin(v)) for key, v in values.items()}
            arrays[f"pred_{task}"] = pred
            arrays[f"tangent_{task}"] = tangent
            arrays[f"random_direction_{task}"] = random_direction
            for name, value in values.items():
                arrays[f"{name}_{task}"] = value
            rows.append({"task": task, "abstained": abstain, "tangent_norm": float(np.linalg.norm(tangent)),
                         "winners": winners,
                         "winner_sha256": {k: record["candidates"][v]["sha256"] for k, v in winners.items()},
                         "prediction_candidates": len(actions), "goal_encoding_images_reused": 3})
            print(f"task={task} winners={winners}", flush=True)
    if model_fingerprint(model) != fingerprint:
        raise ValueError("model changed")
    with OUT.with_suffix(".npz").open("xb") as f:
        np.savez_compressed(f, **arrays)
    sources = (PROTOCOL, Path(__file__), old.OUT.with_suffix(".json"), old.OUT.with_suffix(".npz"),
               old.PRIOR.with_suffix(".json"), old.PRIOR.with_suffix(".npz"),
               old.SOURCE.with_suffix(".json"), old.SOURCE.with_suffix(".npz"))
    with OUT.with_suffix(".json").open("x") as f:
        json.dump({"track": "primary_rgb", "rows": rows, "model_sha256": fingerprint,
                   "arrays_sha256": file_hash(OUT.with_suffix(".npz")),
                   "sources": {str(p): file_hash(p) for p in sources},
                   "scope": "exposed fixed-pool development"}, f, indent=2, allow_nan=False)


def evaluate():
    result_path = Path(str(OUT) + "_evaluation.json")
    if result_path.exists():
        raise FileExistsError(result_path)
    plan = json.loads(OUT.with_suffix(".json").read_text())
    if plan["arrays_sha256"] != file_hash(OUT.with_suffix(".npz")):
        raise ValueError("score binding drift")
    for path, value in plan["sources"].items():
        if file_hash(Path(path)) != value:
            raise ValueError("source drift")
    prior = json.loads(old.PRIOR.with_suffix(".json").read_text())
    rows, hashes = [], {}
    with np.load(OUT.with_suffix(".npz"), allow_pickle=False) as scores:
        for row in plan["rows"]:
            task = row["task"]
            stem = PHYSICAL.parent / f"{PHYSICAL.name}_task{task}"
            meta = json.loads(stem.with_suffix(".json").read_text())
            if (file_hash(stem.with_suffix(".npz")) != meta["arrays_sha256"]
                    or meta["model_sha256"] != plan["model_sha256"]
                    or meta["hashes"] != [c["sha256"] for c in prior["plans"][task]["candidates"]]):
                raise ValueError("physical identity mismatch")
            hashes[str(stem.with_suffix(".json"))] = file_hash(stem.with_suffix(".json"))
            hashes[str(stem.with_suffix(".npz"))] = file_hash(stem.with_suffix(".npz"))
            with np.load(stem.with_suffix(".npz"), allow_pickle=False) as data:
                outcomes = {}
                for method in METHODS:
                    value = scores[f"{method}_{task}"]
                    if int(np.argmin(value)) != row["winners"][method]:
                        raise ValueError("selection drift")
                    outcomes[method] = summarize(value, data["ever"], data["final"])
            rows.append({"task": task, "outcomes": outcomes})
    by_task = {r["task"]: r["outcomes"] for r in rows}
    exit_pass = (by_task[1]["rotation"]["winner_final"]
                 and by_task[0]["rotation"]["winner_final"]
                 and not by_task[1]["random"]["winner_final"])
    with result_path.open("x") as f:
        json.dump({"rows": rows, "development_exit": bool(exit_pass),
                   "plan_sha256": file_hash(OUT.with_suffix(".json")),
                   "arrays_sha256": file_hash(OUT.with_suffix(".npz")),
                   "physical_sources": hashes, "evaluator_sha256": file_hash(Path(__file__)),
                   "scope": "exposed fixed-pool evaluation; no gate claim"},
                  f, indent=2, allow_nan=False)
    print(json.dumps({"development_exit": bool(exit_pass), "rows": rows}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("score", "evaluate"):
        raise SystemExit("expected score|evaluate")
    {"score": score, "evaluate": evaluate}[sys.argv[1]]()
