"""Hold-relative effects for common, executable prefixes and tails.

The scorer receives complete *raw* action chains. The caller is responsible
for the RGB-only boundary and for establishing that the declared reference
tail is a valid hold at the states under study.
"""
import numpy as np


def score_prefix_effects(score, prefixes, hold, probes, *, remaining_steps, packet_steps=5):
    prefixes = np.asarray(prefixes, dtype=np.float32)
    hold = np.asarray(hold, dtype=np.float32)
    probes = np.asarray(probes, dtype=np.float32)
    if prefixes.ndim != 3 or hold.ndim != 2 or probes.ndim != 3:
        raise ValueError("prefixes, hold and probes must be action arrays")
    n, prefix_steps, width = prefixes.shape
    if (type(packet_steps) is not int or packet_steps <= 0 or type(remaining_steps) is not int
            or n < 1 or width < 1 or len(probes) < 1 or prefix_steps < 1
            or hold.shape[0] < 1 or hold.shape[1] != width
            or probes.shape[1:] != hold.shape
            or prefix_steps % packet_steps or hold.shape[0] % packet_steps
            or prefix_steps + hold.shape[0] != remaining_steps):
        raise ValueError("incompatible action shape or packet clock")
    if any(not np.isfinite(x).all() or np.any(np.abs(x) > 1) for x in (prefixes, hold, probes)):
        raise ValueError("actions must be finite and bounded")
    tails = np.concatenate((hold[None], probes), axis=0)
    chains = np.concatenate((np.broadcast_to(prefixes[:, None], (n, len(tails), prefix_steps, width)),
                             np.broadcast_to(tails[None], (n, *tails.shape))), axis=2).copy()
    costs = np.asarray(score(chains.reshape(-1, remaining_steps, width)))
    if costs.shape != (n * len(tails),) or not np.isfinite(costs).all():
        raise ValueError("scorer must return one finite cost per chain")
    costs = costs.reshape(n, len(tails))
    return {"hold_cost": costs[:, 0].copy(), "probe_cost": costs[:, 1:].copy(),
            "effects": costs[:, :1] - costs[:, 1:],
            "queries": int(n * len(tails)), "model_packet_steps": int(n * len(tails) * remaining_steps // packet_steps),
            "chains": chains}
