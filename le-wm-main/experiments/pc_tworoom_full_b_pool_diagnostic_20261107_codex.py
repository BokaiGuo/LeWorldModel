"""Evaluator-only exhaustive replay of a sealed, previously exposed B bank."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments import pc_tworoom_quality_gated_retention_20261107_codex as q
from experiments.pc_input_permissions import RGB, file_hash, freeze_audit
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint
from experiments.pc_rgb_observation_cache_20261107_codex import RGBObservationScoringSession

PREFIX = q.ROOT / "results/pc_tworoom_full_b_pool_diagnostic_20261107_codex"
SCORES = Path(str(PREFIX) + "_scores.npz")
SEAL = Path(str(PREFIX) + "_seal.json")
PHYS = Path(str(PREFIX) + "_physical.npz")
RESULT = Path(str(PREFIX) + "_result.json")
PROTOCOL = q.ROOT.parent / "docs/PC_TWOROOM_FULL_B_POOL_DIAGNOSTIC_PROTOCOL_20261107_CODEX.md"


def summarize_root(scores, success, queried, incumbent):
    scores = np.asarray(scores)
    success = np.asarray(success, dtype=bool)
    queried = np.asarray(queried, dtype=int)
    if scores.shape != (64,) or success.shape != (64,) or not np.isfinite(scores).all():
        raise ValueError("invalid fixed bank")
    if np.any(queried < 0) or np.any(queried >= 64) or len(np.unique(queried)) != len(queried):
        raise ValueError("invalid queried indices")
    order = np.argsort(scores, kind="stable")
    winning = order[success[order]]
    return {"pool_success_count": int(success.sum()),
            "queried_success_count": int(success[queried].sum()),
            "model_best_id": int(order[0]),
            "model_best_success": bool(success[order[0]]),
            "best_success_rank_1based": int(np.flatnonzero(success[order])[0] + 1) if len(winning) else None,
            "best_success_score_minus_incumbent": float(scores[winning[0]] - incumbent) if len(winning) else None}


def score():
    if SCORES.exists() or SEAL.exists():
        raise FileExistsError("score output exists")
    prior = json.loads(q.PLAN.read_text())
    if prior["pool_sha256"] != file_hash(q.POOL) or prior["old_pool_sha256"] != file_hash(q.old.POOL):
        raise ValueError("source pool drift")
    common = q.old.base.source.block.prior.common
    inp = q.old.base.source.block.prior.inputs.INPUT
    if file_hash(inp) != prior["input_sha256"]:
        raise ValueError("input drift")
    model, scaler, transform, normalizer = common.model_setup()
    if model_fingerprint(model) != prior["model_sha256"] or normalizer != prior["normalizer_sha256"]:
        raise ValueError("model drift")
    audit = freeze_audit(RGB, "lewm", [__file__, PROTOCOL, q.PLAN, q.POOL, q.old.POOL, inp,
                         common.CHECKPOINT / "weights.pt"], preprocessing="ImageNet 224, TwoRoom action scaler",
                         clock="five packets of five steps")
    arrays, diffs = {}, []
    with np.load(q.old.POOL, allow_pickle=False) as old, np.load(q.POOL, allow_pickle=False) as previous, \
            np.load(inp, allow_pickle=False) as data, torch.inference_mode():
        for row in prior["rows"]:
            root = row["root"]
            actions = old[f"r{root}_b"].reshape(64, 25, 2)
            if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
                raise ValueError("illegal actions")
            context = {"pixels": transform(data["current"][root])[None, None, None].cuda(),
                       "goal": transform(data["goal"][root])[None, None, None].cuda(),
                       "action": torch.empty((1, 1, 0, 10), device="cuda")}
            session = RGBObservationScoringSession(model, audit)
            normalized = scaler.transform(actions.reshape(-1, 2)).reshape(64, 5, 10)
            vals = []
            for start in (0, 32):
                tensor = torch.as_tensor(normalized[start:start + 32][None], dtype=torch.float32, device="cuda")
                v, _ = session.score(context, tensor)
                vals.extend(v[0].cpu().tolist())
            vals = np.asarray(vals, dtype=np.float64)
            ids = previous[f"r{root}_b_ids"]
            delta = np.abs(vals[ids] - previous[f"r{root}_b_cost"])
            diffs.append(float(delta.max()))
            arrays[f"r{root}_scores"] = vals
            print(f"scored root={root} prior_max_diff={delta.max():.8g}", flush=True)
    if model_fingerprint(model) != prior["model_sha256"]:
        raise ValueError("model changed")
    with SCORES.open("xb") as f:
        np.savez_compressed(f, **arrays)
    bindings = {"scores": SCORES, "previous_plan": q.PLAN, "previous_pool": q.POOL,
                "old_pool": q.old.POOL, "protocol": PROTOCOL, "runner": Path(__file__),
                "input": inp, "data": common.DATA}
    with SEAL.open("x") as f:
        json.dump({"sha256": {k: file_hash(v) for k, v in bindings.items()},
                   "model_sha256": prior["model_sha256"], "normalizer_sha256": normalizer,
                   "max_prior_score_diff": max(diffs), "roots": [r["root"] for r in prior["rows"]]},
                  f, indent=2, allow_nan=False)


def verify():
    seal = json.loads(SEAL.read_text())
    common = q.old.base.source.block.prior.common
    paths = {"scores": SCORES, "previous_plan": q.PLAN, "previous_pool": q.POOL,
             "old_pool": q.old.POOL, "protocol": PROTOCOL, "runner": Path(__file__),
             "input": q.old.base.source.block.prior.inputs.INPUT, "data": common.DATA}
    if any(seal["sha256"][k] != file_hash(v) for k, v in paths.items()):
        raise ValueError("sealed binding drift")
    if seal["roots"] != list(range(10, 50)):
        raise ValueError("incomplete panel")


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    verify()
    if PHYS.exists():
        raise FileExistsError(PHYS)
    plan = json.loads(q.PLAN.read_text())
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    arrays = {}
    try:
        with np.load(q.old.POOL, allow_pickle=False) as pool, \
                h5py.File(q.old.base.source.block.prior.common.DATA, "r") as data:
            for row in plan["rows"]:
                root = row["root"]
                actions = pool[f"r{root}_b"].reshape(64, 25, 2)
                dist = np.empty((64, 25), dtype=np.float64)
                term = np.empty((64, 25), dtype=bool)
                trunc = np.empty((64, 25), dtype=bool)
                for i, chain in enumerate(actions):
                    env.reset(seed=42)
                    actor = env.unwrapped
                    actor._set_state(data["proprio"][row["row"]])
                    actor._set_goal_state(data["proprio"][row["row"] + 25])
                    for t, action in enumerate(chain):
                        _, _, done, cut, info = actor.step(action.copy())
                        dist[i, t], term[i, t], trunc[i, t] = info["distance_to_target"], done, cut
                arrays[f"r{root}_distance"], arrays[f"r{root}_term"], arrays[f"r{root}_trunc"] = dist, term, trunc
                print(f"replayed root={root} success={term.any(axis=1).sum()}/64", flush=True)
    finally:
        env.close()
    with PHYS.open("xb") as f:
        np.savez_compressed(f, **arrays)


def report():
    verify()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    plan = json.loads(q.PLAN.read_text())
    old_phys = json.loads(q.PHYSICAL.read_text())
    if old_phys["plan_sha256"] != file_hash(q.PLAN) or old_phys["pool_sha256"] != file_hash(q.POOL):
        raise ValueError("prior physical drift")
    previous_rows = {r["root"]: r for r in old_phys["rows"]}
    rows = []
    with np.load(SCORES, allow_pickle=False) as scores, np.load(PHYS, allow_pickle=False) as phys, \
            np.load(q.POOL, allow_pickle=False) as prev:
        for row in plan["rows"]:
            root = row["root"]
            term = phys[f"r{root}_term"]
            if term.shape != (64, 25):
                raise ValueError("incomplete physical replay")
            cost, ids = scores[f"r{root}_scores"], prev[f"r{root}_b_ids"]
            incumbent = float(prev[f"r{root}_inc_cost"].min())
            outcomes = previous_rows[root]["outcomes"]
            metrics = {}
            for name, success in (("ever", term.any(axis=1)), ("final", term[:, -1]),
                                  ("post_prefix", ~term[:, :10].any(axis=1) & term[:, 10:].any(axis=1))):
                metrics[name] = summarize_root(cost, success, ids, incumbent)
            arm_queries = {m: int(term[(np.asarray(row["decisions"][m]["retained"])[:, None]
                                        * 8 + np.arange(8)).ravel()].any(axis=1).sum()) for m in q.METHODS}
            rows.append({"root": root, "metrics": metrics, "arm_queried_ever_success_count": arm_queries,
                         "previous_selected_ever": {m: outcomes[m]["ever"] for m in q.METHODS},
                         "previous_selected_final": {m: outcomes[m]["final"] for m in q.METHODS}})
    summary = {}
    for key in ("ever", "final", "post_prefix"):
        summary[key] = {"roots_with_pool_success": sum(r["metrics"][key]["pool_success_count"] > 0 for r in rows),
                        "roots_with_queried_success": sum(r["metrics"][key]["queried_success_count"] > 0 for r in rows),
                        "roots_model_best_success": sum(r["metrics"][key]["model_best_success"] for r in rows),
                        "roots_success_scored_better_than_incumbent": sum(
                            r["metrics"][key]["best_success_score_minus_incumbent"] is not None and
                            r["metrics"][key]["best_success_score_minus_incumbent"] < 0 for r in rows)}
    summary["previous_failure_roots_with_b_success"] = {
        m: [r["root"] for r in rows if not r["previous_selected_ever"][m]
            and r["metrics"]["ever"]["pool_success_count"]] for m in q.METHODS}
    with RESULT.open("x") as f:
        json.dump({"summary": summary, "rows": rows, "seal_sha256": file_hash(SEAL),
                   "physical_sha256": file_hash(PHYS), "previous_physical_sha256": file_hash(q.PHYSICAL),
                   "scope": "exposed development, evaluator only"}, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("score", "physical", "report"):
        raise SystemExit("expected score|physical|report")
    {"score": score, "physical": physical, "report": report}[sys.argv[1]]()
