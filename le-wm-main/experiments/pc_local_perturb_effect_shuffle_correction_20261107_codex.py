"""Post-hoc correction: permute action effects, not absolute parent scores."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from experiments import pc_tworoom_local_perturb_stability_20261107_codex as source
from experiments.pc_input_permissions import file_hash
from experiments.research_runtime.artifacts import atomic_write_json

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_LOCAL_PERTURB_EFFECT_SHUFFLE_CORRECTION_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_local_perturb_effect_shuffle_correction_20261107_codex_plan.json"
OUT = ROOT / "results/pc_local_perturb_effect_shuffle_correction_20261107_codex_physical.json"


def effect_scores(parents, children, mapping):
    parents = np.asarray(parents, np.float64)
    children = np.asarray(children, np.float64)
    mapping = np.asarray(mapping)
    if (parents.shape != (8,) or children.shape != (8, 4) or mapping.shape != (8,)
            or not np.isfinite(parents).all() or not np.isfinite(children).all()
            or sorted(mapping.tolist()) != list(range(8))
            or np.any(mapping == np.arange(8))):
        raise ValueError("invalid effect shuffle input")
    delta = children - parents[:, None]
    return parents + delta.sum(axis=1) / 5, parents + delta[mapping].sum(axis=1) / 5


def plan():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    prior = json.loads(source.PLAN.read_text())
    for key, path in (("source_pool_sha256", source.source.POOL), ("pool_sha256", source.POOL),
                      ("runner_sha256", source.__file__), ("protocol_sha256", source.PROTOCOL)):
        if prior[key] != file_hash(path):
            raise ValueError(f"source drift: {key}")
    rows = []
    with np.load(source.source.POOL, allow_pickle=False) as original, np.load(source.POOL, allow_pickle=False) as pool:
        for row in prior["rows"]:
            root = row["root"]
            all_costs = original[f"r{root}_b_cost"].ravel()
            indices = np.asarray(row["parents"])
            children = pool[f"r{root}_child_cost"]
            matched, shuffled = effect_scores(all_costs[indices], children, row["mapping"])
            if not np.allclose(matched, row["robust_cost"], atol=1e-10, rtol=1e-10):
                raise ValueError("matched score identity mismatch")
            rows.append({"root": root, "row": row["row"], "corrected_index": int(indices[np.argmin(shuffled)]),
                         "old_shuffled_index": row["shuffled"], "robust_index": row["robust"],
                         "scores": shuffled.tolist(), "mapping": row["mapping"]})
    atomic_write_json(PLAN, {"rows": rows, "source_plan_sha256": file_hash(source.PLAN),
                            "source_pool_sha256": file_hash(source.POOL),
                            "original_pool_sha256": file_hash(source.source.POOL),
                            "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                            "scope": "post-hoc corrected control; exposed tasks"})


def physical():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    if OUT.exists():
        raise FileExistsError(OUT)
    plan_record = json.loads(PLAN.read_text())
    for key, path in (("source_plan_sha256", source.PLAN), ("source_pool_sha256", source.POOL),
                      ("original_pool_sha256", source.source.POOL), ("protocol_sha256", PROTOCOL),
                      ("runner_sha256", __file__)):
        if plan_record[key] != file_hash(path):
            raise ValueError(f"dependency drift: {key}")
    old_physical = json.loads(source.source.PHYSICAL.read_text())
    original = json.loads(source.OUT.read_text())
    if original["plan_sha256"] != file_hash(source.PLAN) or old_physical["pool_sha256"] != file_hash(source.source.POOL):
        raise ValueError("physical provenance mismatch")
    common = source.source.source.block.prior.common
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with np.load(source.source.POOL, allow_pickle=False) as pool, h5py.File(common.DATA, "r") as data:
            for row, old, old_result in zip(plan_record["rows"], old_physical["rows"], original["rows"], strict=True):
                if row["root"] != old["root"] or row["root"] != old_result["root"]:
                    raise ValueError("root mismatch")
                actions = pool[f"r{row['root']}_b_actions"].reshape(64, 25, 2)[row["corrected_index"]]
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(data["proprio"][row["row"]])
                actor._set_goal_state(data["proprio"][row["row"] + 25])
                trace = []
                for action in actions:
                    _, _, term, trunc, info = actor.step(action.copy())
                    if trunc:
                        raise ValueError("unexpected truncation")
                    trace.append({"distance": float(info["distance_to_target"]), "success": bool(term)})
                previous = old["outcomes"][row["corrected_index"]]["trace"]
                max_diff = max(abs(a["distance"] - b["distance"]) for a, b in zip(trace, previous, strict=True))
                mismatch = sum(a["success"] != b["terminated"] for a, b in zip(trace, previous, strict=True))
                if max_diff > 1e-5 or mismatch:
                    raise ValueError("prior physical replay mismatch")
                rows.append({"root": row["root"], "trace": trace, "ever": any(t["success"] for t in trace),
                             "prefix_success": any(t["success"] for t in trace[:10]), "max_replay_diff": max_diff,
                             "direct_ever": old_result["arms"]["direct"]["ever"],
                             "robust_ever": old_result["arms"]["robust"]["ever"],
                             "old_shuffle_ever": old_result["arms"]["shuffled"]["ever"]})
                print(f"root={row['root']} corrected control replayed", flush=True)
    finally:
        env.close()
    output = {"rows": rows, "corrected_success": sum(r["ever"] for r in rows),
              "robust_success": sum(r["robust_ever"] for r in rows),
              "old_shuffled_success": sum(r["old_shuffle_ever"] for r in rows),
              "corrected_vs_robust_wins": sum(r["ever"] and not r["robust_ever"] for r in rows),
              "corrected_vs_robust_losses": sum(r["robust_ever"] and not r["ever"] for r in rows),
              "changed_from_old": sum(r["corrected_index"] != r["old_shuffled_index"] for r in plan_record["rows"]),
              "plan_sha256": file_hash(PLAN), "original_physical_sha256": file_hash(source.OUT),
              "scope": "post-hoc fixed pool; not independent confirmation"}
    atomic_write_json(OUT, output)
    print(json.dumps({k: v for k, v in output.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "physical"):
        raise SystemExit("expected plan|physical")
    {"plan": plan, "physical": physical}[sys.argv[1]]()
