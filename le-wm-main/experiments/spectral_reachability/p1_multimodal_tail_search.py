"""Fixed-budget, mode-preserving CEM for development-only tail search."""
import numpy as np

from .p1_task_response_revision import coherent_seeds


def plan(direction, score, *, common, seed):
    """Evaluate 4 basins x 16 candidates x 5 iterations x 5 variants."""
    unit = np.asarray(direction, dtype=np.float64)
    if unit.shape != (2,) or not np.isfinite(unit).all() or np.linalg.norm(unit) < 1e-8:
        raise ValueError("finite nonzero RGB direction required")
    unit = (unit / np.linalg.norm(unit)).astype(np.float32)
    seeds = coherent_seeds(unit)
    # amplitude .5 with angles -90, 0, +90, 180 degrees
    means = seeds[[1, 3, 5, 6]].copy()
    stds = np.full_like(means, .25)
    rng = np.random.default_rng(seed)
    records, events = [], []
    for iteration in range(5):
        bank = np.clip(rng.normal(means[:, None], stds[:, None],
                                 size=(4, 16, 2, 5, 2)), -1, 1).astype(np.float32)
        bank[:, 0] = means
        flat = bank.reshape(64, 2, 5, 2)
        variants = np.stack([common.action_variants(candidate) for candidate in flat])
        scores = np.asarray(score(variants), dtype=np.float64)
        if scores.shape != (64, 5) or not np.isfinite(scores).all():
            raise ValueError("complete finite candidate and perturbation scores required")
        robust = common.robust_scores(scores)
        feasible = np.einsum("sij,j->s", flat[:, 0, :3], unit) >= .15
        modes = []
        for mode in range(4):
            indices = np.flatnonzero(feasible[mode*16:(mode+1)*16]) + mode*16
            indices = indices[np.argsort(robust[indices], kind="stable")]
            if len(indices):
                elite_indices = np.pad(indices, (0, max(0, 2-len(indices))), mode="edge")[:2]
                elites = flat[elite_indices]
                means[mode] = elites.mean(axis=0, dtype=np.float32)
                stds[mode] = np.clip(elites.std(axis=0, dtype=np.float32), .03, 1)
                records.extend(dict(iteration=iteration, candidate=int(i),
                                    mode=mode, robust_score=float(robust[i]),
                                    actions=flat[i].tolist()) for i in indices)
            modes.append(dict(mode=mode, feasible=int(len(indices)),
                              updated=bool(len(indices))))
        events.append(dict(iteration=iteration, modes=modes, all_scores=scores.tolist()))
    if not records:
        raise ValueError("no feasible model-evaluated candidate")
    return dict(selected=min(records, key=lambda r: (r["robust_score"], r["iteration"], r["candidate"])),
                iterations=events, scored_candidates=320, perturbation_samples=1600,
                independent_distributions=4)
