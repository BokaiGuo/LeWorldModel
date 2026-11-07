"""Executable-prefix / equal-duration action-effect bank, with explicit reference."""
import hashlib

import numpy as np


def effect_bank(prefixes, probes, reference, score):
    """Score full executable chains; score receives (N, horizon, action_dim).

    The caller supplies the reference action: zero action is not assumed to be
    a physically static hold. The callback must return one cost per chain.
    No environment state or evaluation labels enter this function.
    """
    prefixes, probes, reference = [np.asarray(x) for x in (prefixes, probes, reference)]
    if (prefixes.ndim != 3 or probes.ndim != 3 or reference.ndim != 2
            or not len(prefixes) or not len(probes) or not prefixes.shape[1]
            or not probes.shape[1] or not prefixes.shape[2]
            or probes.shape[1:] != reference.shape
            or prefixes.shape[2] != probes.shape[2]):
        raise ValueError("prefix, probe and reference shapes incompatible")
    if any(not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all()
           or np.any(np.abs(x) > 1) for x in (prefixes, probes, reference)):
        raise ValueError("finite normalized raw actions in [-1,1] required")
    n, m = len(prefixes), len(probes)
    tails = np.concatenate((reference[None], probes), axis=0)
    chains = np.concatenate((np.broadcast_to(prefixes[:, None], (n, m + 1, *prefixes.shape[1:])),
                             np.broadcast_to(tails[None], (n, m + 1, *tails.shape[1:]))), axis=2)
    flat = np.ascontiguousarray(chains.reshape(n * (m + 1), *chains.shape[2:]))
    cost = np.asarray(score(flat), dtype=np.float64)
    if cost.shape != (len(flat),) or not np.isfinite(cost).all():
        raise ValueError("scorer must return one finite cost per executable chain")
    matrix = cost.reshape(n, m + 1)
    reference_cost, probe_costs = matrix[:, 0], matrix[:, 1:]
    return {"reference_cost": reference_cost, "probe_costs": probe_costs,
            "effect": reference_cost[:, None] - probe_costs,
            "opportunity": probe_costs.min(axis=1),
            "num_full_chain_scores": len(flat),
            "chains_sha256": hashlib.sha256(flat.tobytes()).hexdigest()}
