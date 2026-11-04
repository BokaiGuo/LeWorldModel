"""Budget-matched single versus two-population CEM development ablation.

Both algorithms score 64 candidates and five perturbations per iteration.
The structured arm allocates half the samples to a seeded population, but
the returned plan is always the best feasible scored candidate overall.
The seed is RGB geometry, not a dynamics-derived spectral program.
"""
import numpy as np

from . import p1_task_response_revision as previous


def plan(direction, score, *, structured, common, seed):
    unit = np.asarray(direction, dtype=np.float32).copy()
    if unit.shape != (2,) or not np.isfinite(unit).all() or np.linalg.norm(unit) < 1e-8:
        raise ValueError("finite nonzero RGB direction required")
    unit /= np.linalg.norm(unit)
    rng = np.random.default_rng(seed)
    size = 32 if structured else 64
    groups = 2 if structured else 1
    means = np.zeros((groups, 2, 5, 2), dtype=np.float32)
    stds = np.full_like(means, .25)
    structured_seeds = previous.coherent_seeds(unit)
    events, candidates = [], []
    for iteration in range(5):
        banks = []
        for group in range(groups):
            bank = np.clip(rng.normal(means[group], stds[group],
                                      (size, 2, 5, 2)), -1, 1).astype(np.float32)
            bank[0] = means[group]
            if not structured and iteration == 0:
                bank[1] = means[group]
            if structured and group == 1 and iteration == 0:
                bank[:len(structured_seeds)] = structured_seeds
            banks.append(bank)
        bank = np.concatenate(banks)
        variants = np.stack([common.action_variants(candidate) for candidate in bank])
        values = np.asarray(score(variants), dtype=float)
        if values.shape != (64, 5) or not np.isfinite(values).all():
            raise ValueError("complete finite scores required")
        robust = common.robust_scores(values)
        feasible = np.einsum("sij,j->s", bank[:, 0, :3], unit) >= .15
        elite_groups = []
        for group in range(groups):
            indices = np.arange(group * size, (group + 1) * size)
            order = indices[feasible[indices]]
            order = order[np.argsort(robust[order], kind="stable")]
            if not len(order):
                raise ValueError(f"no feasible candidate for population {group}")
            count = 4 if structured else 8
            elites = np.pad(order, (0, max(0, count - len(order))), mode="edge")[:count]
            means[group] = bank[elites].mean(axis=0, dtype=np.float32)
            stds[group] = np.clip(bank[elites].std(axis=0, dtype=np.float32), .03, 1)
            elite_groups.append(elites.tolist())
        candidates.extend(dict(iteration=iteration, candidate=int(i),
                               robust_score=float(robust[i]), actions=bank[i].tolist())
                          for i in np.flatnonzero(feasible))
        events.append({"iteration": iteration, "bank": bank.tolist(),
                       "all_scores": values.tolist(), "feasible": feasible.tolist(),
                       "elite_indices_by_population": elite_groups})
    return {"selected": min(candidates, key=lambda x: (x["robust_score"], x["iteration"], x["candidate"])),
            "iterations": events, "scored_candidates": 320, "perturbation_samples": 1600}
