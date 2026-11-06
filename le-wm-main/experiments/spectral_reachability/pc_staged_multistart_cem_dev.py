"""Two-stage budget allocation using the existing shared CEM implementation."""
import numpy as np

from experiments.spectral_reachability import pc_shared_multistart_cem as shared


def optimize(evaluate, prefix, initial_tails, sample_budget, *, seed=0,
             batch_size=12, elites=4, initial_scale=0.35, min_scale=0.07):
    seeds = np.asarray(initial_tails, np.float32)
    prefix = np.asarray(prefix, np.float32)
    if seeds.ndim != 3 or len(seeds) == 0:
        raise ValueError("invalid initial tails")
    minimum = 1 + 2 * batch_size
    survivors = (len(seeds) + 1) // 2
    # Fall back deterministically when the available budget cannot fund two
    # iterations for each initial start and each continued survivor.
    if sample_budget < minimum * (len(seeds) + survivors):
        return shared.optimize(evaluate, prefix, seeds, sample_budget, seed=seed,
            batch_size=batch_size, elites=elites, initial_scale=initial_scale, min_scale=min_scale)
    first_total = sample_budget // 2
    first_total = max(first_total, minimum * len(seeds))
    first_total = min(first_total, sample_budget - minimum * survivors)
    first_alloc = [first_total // len(seeds) + (i < first_total % len(seeds))
                   for i in range(len(seeds))]
    remaining = sample_budget - first_total
    runs = []

    def run(tail, budget, index, phase):
        derived_seed = int(np.random.SeedSequence([int(seed), phase, index]).generate_state(1)[0])
        return shared.optimize(evaluate, prefix, tail[None], budget, seed=derived_seed,
            batch_size=batch_size, elites=elites, initial_scale=initial_scale, min_scale=min_scale)

    for index, tail in enumerate(seeds):
        runs.append(run(tail, first_alloc[index], index, 0))
    order = sorted(range(len(runs)), key=lambda i: (runs[i]["score"], i))
    selected = order[:survivors]
    extras = [remaining // survivors + (i < remaining % survivors) for i in range(survivors)]
    final = list(runs)
    starts = [dict(item["starts"][0], allocation_policy="staged",
                   promoted=False, continuation_samples=0) for item in runs]
    for rank, index in enumerate(selected):
        tail = np.asarray(runs[index]["actions"], np.float32)[len(prefix):]
        continued = run(tail, extras[rank], index, 1)
        starts[index]["samples"] += extras[rank]
        starts[index]["promoted"] = True
        starts[index]["continuation_samples"] = extras[rank]
        starts[index]["generations"] += continued["starts"][0]["generations"]
        starts[index]["best_score"] = min(starts[index]["best_score"], continued["score"])
        if continued["score"] < final[index]["score"]:
            final[index] = continued
    winner = min(range(len(final)), key=lambda i: (final[i]["score"], i))
    return {"actions": final[winner]["actions"], "score": final[winner]["score"],
            "samples": int(sample_budget), "starts": starts}
