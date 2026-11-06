"""Equal-opportunity suffix diagnostic; not a full-method resource comparison."""
import numpy as np

from experiments.spectral_reachability import pc_shared_multistart_cem


def evaluate_bank(predict, prefixes, selections, *, tail_steps, samples, restarts,
                  seed, batch_size=12, elites=4):
    """Optimize every fixed prefix with identical seed schedule and sample counts.

    The caller owns metering and durable recording of every predictor call.
    Selections must be established before invoking this diagnostic.
    """
    bank = np.asarray(prefixes, dtype=np.float32)
    if (bank.ndim != 3 or not len(bank) or not bank.shape[1] or not bank.shape[2]
            or not np.isfinite(bank).all() or np.any(np.abs(bank) > 1)
            or not isinstance(tail_steps, int) or tail_steps < 1
            or not isinstance(restarts, int) or restarts < 1):
        raise ValueError("invalid bank or diagnostic configuration")
    if not selections:
        raise ValueError("missing fixed retention selections")
    fixed = {}
    for name, indices in selections.items():
        if not isinstance(name, str) or not name:
            raise ValueError("invalid selection name")
        if not indices or any(not isinstance(i, (int, np.integer)) or i < 0 or i >= len(bank)
                              for i in indices) or len(set(indices)) != len(indices):
            raise ValueError("invalid selection indices")
        fixed[name] = [int(i) for i in indices]
    if len({len(indices) for indices in fixed.values()}) != 1:
        raise ValueError("retention rules must have equal capacity")
    initial = np.zeros((1, tail_steps, bank.shape[2]), dtype=np.float32)
    candidates = []
    for index, prefix in enumerate(bank):
        runs = []
        for restart in range(restarts):
            def score(chains):
                _, values = predict(chains)
                return values

            result = pc_shared_multistart_cem.optimize(
                score, prefix, initial, samples, seed=seed + restart,
                batch_size=batch_size, elites=elites)
            runs.append({"restart": restart, "score": result["score"],
                         "actions": result["actions"].tolist(), "samples": result["samples"]})
        winner = min(range(restarts), key=lambda i: runs[i]["score"])
        candidates.append({"index": index, "runs": runs, "best_score": runs[winner]["score"],
                           "best_restart": winner})
    all_best = min(item["best_score"] for item in candidates)
    rules = {}
    for name, indices in fixed.items():
        winner = min(indices, key=lambda i: (candidates[i]["best_score"], i))
        best = candidates[winner]["best_score"]
        rules[name] = {"indices": indices, "winner": winner, "best_score": best,
                       "gap_to_bank_best": best - all_best}
    return {"candidates": candidates, "rules": rules, "all_bank_best": all_best,
            "samples_per_candidate": samples * restarts,
            "scope": "fixed-bank diagnostic; caller accounts for generation and selection separately"}
