"""Reconstruct CEM candidate banks from fixed RNG and recorded model scores.

This is deterministic algorithm replay, not a fresh model or physical run.
"""
import numpy as np

from . import p1_dev304_coherent_saved_20261102 as first
from . import p1_dev304_terminal_saved_20261102 as terminal


def replay(saved):
    run = saved.run
    plan_dir = saved.ROOT / f"{saved.NAME}_PLAN_DEV01"
    report = run.base.read(plan_dir / "report.json", saved.PLAN_SHA)
    spec = run.base.read(plan_dir / "plan_contract.json", report["plan_contract_sha256"])
    direction = np.asarray(spec["contact_direction_raw_xy"], np.float32)
    direction /= np.linalg.norm(direction)
    common = run.coherent.base.first.engine.base.common
    seed = run.coherent.base.first.planner.parent.SEED
    outputs = {}
    for row in report["rows"]:
        rng = np.random.default_rng(seed)
        mean = np.zeros((2, 5, 2), np.float32)
        std = np.full_like(mean, .25)
        seeds = (run.coherent.revision.coherent_seeds(direction) if row["arm"] == "coherent"
                 else np.stack([mean, mean]))
        candidates, records = [], []
        for iteration, event in enumerate(row["iterations"]):
            bank = common.make_bank(rng, mean, std, seeds, iteration)
            scores = np.asarray(event["all_scores"], dtype=float)
            robust = common.robust_scores(scores)
            feasible = np.einsum("sij,j->s", bank[:, 0, :3], direction) >= .15
            order = np.flatnonzero(feasible)
            order = order[np.argsort(robust[order], kind="stable")]
            run.require(len(order) > 0 and int(feasible.sum()) == event["feasible"], "feasible replay")
            elite = np.pad(order, (0, max(0, 8-len(order))), mode="edge")[:8]
            candidates.extend((float(robust[i]), iteration, int(i), bank[i].copy()) for i in order)
            records.append({"elite_indices": elite.tolist(), "elite_actions": bank[elite].tolist()})
            mean = bank[elite].mean(axis=0, dtype=np.float32)
            std = np.clip(bank[elite].std(axis=0, dtype=np.float32), .03, 1)
        winner = min(candidates, key=lambda x: x[:3])
        run.require(winner[:3] == (row["selected"]["robust_score"], row["selected"]["iteration"],
                                  row["selected"]["candidate"])
                    and np.array_equal(winner[3], row["selected"]["actions"]), "selected replay")
        outputs[row["arm"]] = records
    comparisons = []
    for a, b in zip(outputs["original"], outputs["coherent"], strict=True):
        comparisons.append({"elite_indices_equal": a["elite_indices"] == b["elite_indices"],
                            "elite_actions_equal": np.array_equal(a["elite_actions"], b["elite_actions"]),
                            "original_elite_indices": a["elite_indices"],
                            "coherent_elite_indices": b["elite_indices"]})
    return {"comparisons": comparisons, "classification": "deterministic_saved_score_replay",
            "model_or_physical_rerun": False, "p1_admission": False}


def analyze():
    first.analyze()
    terminal.analyze()
    return {"first_packet": replay(first), "terminal_packet": replay(terminal)}


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(first.run.base.write_new(parser.parse_args().output, analyze()))
