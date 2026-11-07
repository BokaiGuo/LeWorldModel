"""Post-exposure fixed-pool neighborhood stability screen."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_tworoom_block_heldout_effect_20261107_codex as source
from experiments.pc_input_permissions import file_hash
from experiments.research_runtime.artifacts import atomic_write_json

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_NEIGHBOR_STABILITY_SCREEN_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_neighbor_stability_screen_20261107_codex_plan.json"
OUT = ROOT / "results/pc_tworoom_neighbor_stability_screen_20261107_codex_result.json"


def select(tails, scores, seed):
    tails, scores = np.asarray(tails), np.asarray(scores)
    if tails.shape != (8, 3, 2) or scores.shape != (8, 8):
        raise ValueError("unexpected tail or score shape")
    if not np.isfinite(tails).all() or not np.isfinite(scores).all() or np.max(np.abs(tails)) > 1:
        raise ValueError("illegal input")
    distances = np.linalg.norm((tails[:, None] - tails[None, :]).reshape(8, 8, -1), axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :2]
    rng = np.random.default_rng(seed)
    random = np.stack([rng.choice(np.delete(np.arange(8), i), size=2, replace=False)
                       for i in range(8)])

    def aggregate(neighbors):
        return (scores + scores[:, neighbors].sum(axis=2)) / 3

    smoothed, control = aggregate(nearest), aggregate(random)
    return {
        "baseline": int(np.argmin(scores)),
        "neighbor": int(np.argmin(smoothed)),
        "random": int(np.argmin(control)),
        "nearest": nearest.tolist(),
        "random_neighbors": random.tolist(),
        "nearest_distances": np.take_along_axis(distances, nearest, axis=1).tolist(),
        "random_distances": np.take_along_axis(distances, random, axis=1).tolist(),
        "neighbor_scores": smoothed.tolist(),
        "random_scores": control.tolist(),
    }


def plan():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    original = json.loads(source.PLAN.read_text())
    if original["pool_sha256"] != file_hash(source.POOL):
        raise ValueError("source pool drift")
    rows = []
    with np.load(source.POOL, allow_pickle=False) as pool:
        for root in range(10):
            tails = pool[f"r{root}_b"]
            actions = pool[f"r{root}_b_actions"]
            if not np.array_equal(actions[:, :, 10:, :], np.broadcast_to(
                    np.repeat(tails, 5, axis=1)[None], (8, 8, 15, 2))):
                raise ValueError("action identity mismatch")
            rows.append({"root": root, **select(tails, pool[f"r{root}_b_cost"], 29141000 + root)})
    atomic_write_json(PLAN, {"rows": rows, "source_plan_sha256": file_hash(source.PLAN),
                            "pool_sha256": file_hash(source.POOL), "protocol_sha256": file_hash(PROTOCOL),
                            "runner_sha256": file_hash(Path(__file__))})


def evaluate():
    if OUT.exists():
        raise FileExistsError(OUT)
    plan_record = json.loads(PLAN.read_text())
    for key, path in (("source_plan_sha256", source.PLAN), ("pool_sha256", source.POOL),
                      ("protocol_sha256", PROTOCOL), ("runner_sha256", __file__)):
        if plan_record[key] != file_hash(path):
            raise ValueError(f"plan dependency drift: {key}")
    physical = json.loads(source.PHYSICAL.read_text())
    if physical["plan_sha256"] != file_hash(source.PLAN) or physical["pool_sha256"] != file_hash(source.POOL):
        raise ValueError("physical source mismatch")
    rows = []
    for chosen, observed in zip(plan_record["rows"], physical["rows"], strict=True):
        if chosen["root"] != observed["root"] or len(observed["outcomes"]) != 64:
            raise ValueError("outcome alignment mismatch")
        outcomes = observed["outcomes"]
        arms = {arm: {"index": chosen[arm], **{key: outcomes[chosen[arm]][key]
                    for key in ("ever", "prefix_success", "final_distance")}}
                for arm in ("baseline", "neighbor", "random")}
        rows.append({"root": chosen["root"], "arms": arms,
                     "oracle": any(item["ever"] for item in outcomes)})
    totals = {arm: sum(row["arms"][arm]["ever"] for row in rows)
              for arm in ("baseline", "neighbor", "random")}
    wins = sum(r["arms"]["neighbor"]["ever"] and not r["arms"]["baseline"]["ever"] for r in rows)
    losses = sum(r["arms"]["baseline"]["ever"] and not r["arms"]["neighbor"]["ever"] for r in rows)
    output = {"rows": rows, "success": totals, "wins": wins, "losses": losses,
              "changed": sum(r["arms"]["neighbor"]["index"] != r["arms"]["baseline"]["index"] for r in rows),
              "oracle": sum(r["oracle"] for r in rows),
              "development_exit": wins >= 1 and losses == 0 and totals["neighbor"] > totals["random"],
              "mean_nearest_distance": float(np.mean([r["nearest_distances"] for r in plan_record["rows"]])),
              "mean_random_distance": float(np.mean([r["random_distances"] for r in plan_record["rows"]])),
              "plan_sha256": file_hash(PLAN), "physical_sha256": file_hash(source.PHYSICAL),
              "scope": "post-exposure fixed pool; not independent confirmation"}
    atomic_write_json(OUT, output)
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "evaluate"):
        raise SystemExit("expected plan|evaluate")
    {"plan": plan, "evaluate": evaluate}[sys.argv[1]]()
