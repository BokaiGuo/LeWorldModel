"""Development ablation: coordinate-separated score and coherent tail seeds.

No physical outcomes, simulator state, learned adapter or gate changes.
The metric is a local RGB-tangent approximation, not a pose estimator.
"""
import numpy as np


def coordinate_metric(jacobian, object_weight=1., pusher_weight=.05):
    j = np.asarray(jacobian, dtype=np.float64)
    if j.shape != (192, 5) or not np.isfinite(j).all():
        raise ValueError("finite RGB tangent [192,5] required")
    weights = np.array([object_weight] * 3 + [pusher_weight] * 2, dtype=float)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("positive coordinate weights required")
    u, s, vh = np.linalg.svd(j, full_matrices=False)
    if s[-1] <= 0 or s[0] / s[-1] > 1e6:
        raise ValueError("unresolved RGB tangent")
    inverse = (vh.T * (s / (s*s + .01*s[0]**2))) @ u.T
    # Keep coordinate labels, unlike simply weighting the left singular modes.
    q = inverse.T @ np.diag(weights) @ inverse
    q += .01 / s[0]**2 * (np.eye(192) - u @ u.T)
    q = (q + q.T) / 2
    q *= 192 / np.trace(q)
    values, vectors = np.linalg.eigh(q)
    if values.min() <= 0:
        raise ValueError("positive metric required")
    return (vectors * np.sqrt(values)) @ vectors.T


def coherent_seeds(direction):
    """Fixed correlated packets spanning approach then continue/turn/reverse.

    This expands candidate coverage, not a guarantee of contact or usefulness.
    """
    d = np.asarray(direction, dtype=float)
    if d.shape != (2,) or not np.isfinite(d).all() or np.linalg.norm(d) < 1e-8:
        raise ValueError("finite nonzero RGB direction required")
    d = d / np.linalg.norm(d)
    seeds = [np.zeros((2, 5, 2), np.float32)]
    for amplitude in (.5, 1.):
        for angle in (-90, -45, 0, 45, 90, 180):
            a = np.deg2rad(angle)
            rotated = np.array([[np.cos(a), -np.sin(a)],
                                [np.sin(a), np.cos(a)]]) @ d
            seeds.append(np.stack([np.tile(amplitude*d, (5, 1)),
                                   np.tile(amplitude*rotated, (5, 1))]).astype(np.float32))
    return np.clip(np.stack(seeds), -1, 1)


def plan(direction, score, *, structured, common, seed):
    """Exactly 5x64 candidates x5 perturbations; no extra scoring of seeds."""
    rng = np.random.default_rng(seed)
    mean = np.zeros((2, 5, 2), np.float32)
    std = np.full_like(mean, .25)
    seeds = coherent_seeds(direction) if structured else np.stack([mean, mean])
    unit = np.asarray(direction, np.float32).copy()
    unit /= np.linalg.norm(unit)
    records, events = [], []
    for iteration in range(5):
        bank = common.make_bank(rng, mean, std, seeds, iteration)
        variants = np.stack([common.action_variants(candidate) for candidate in bank])
        values = np.asarray(score(variants), dtype=float)
        if values.shape != (64, 5) or not np.isfinite(values).all():
            raise ValueError("complete finite candidate/perturbation scores required")
        robust = common.robust_scores(values)
        feasible = np.einsum("sij,j->s", bank[:, 0, :3], unit) >= .15
        order = np.flatnonzero(feasible)
        order = order[np.argsort(robust[order], kind="stable")]
        if not len(order):
            raise ValueError("no feasible candidate; no retries or replacement")
        elite_indices = np.pad(order, (0, max(0, 8-len(order))), mode="edge")[:8]
        elite = bank[elite_indices]
        mean = elite.mean(axis=0, dtype=np.float32)
        std = np.clip(elite.std(axis=0, dtype=np.float32), .03, 1)
        records.extend(dict(iteration=iteration, candidate=int(i),
                            robust_score=float(robust[i]), actions=bank[i].tolist())
                       for i in order)
        events.append(dict(iteration=iteration, feasible=int(feasible.sum()),
                           seed_count=len(seeds) if iteration == 0 else 0,
                           all_scores=values.tolist()))
    return dict(selected=min(records, key=lambda x: (x["robust_score"], x["iteration"], x["candidate"])),
                iterations=events, scored_candidates=320, perturbation_samples=1600)
