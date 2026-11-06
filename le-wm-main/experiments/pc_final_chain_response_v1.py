"""Full-tail, actual-action response measurement for a sealed final chain.

This is a measurement interface, not a mechanism gate or planner.
The caller owns durable pre-request logging, task binding, and budget accounting.
"""
import hashlib
import numpy as np

from experiments.pc_boundary_response_posthoc import probes, differences


def prepare(final_chain, prefix, *, prefix_steps=5, epsilon=.01):
    """Build feasible float32 requests without altering the selected tail."""
    chain = np.asarray(final_chain, np.float32)
    prefix = np.asarray(prefix, np.float32)
    if chain.ndim != 2 or chain.shape[1] != 2 or not 0 < prefix_steps < len(chain):
        raise ValueError("invalid final chain")
    if prefix.shape != (prefix_steps, 2):
        raise ValueError("invalid intervention prefix")
    if not np.isfinite(prefix).all() or np.max(np.abs(prefix)) > 1:
        raise ValueError("invalid intervention action")
    tails, specs = probes(chain[prefix_steps:], epsilon, dimensions=chain[prefix_steps:].size)
    actions = np.concatenate((np.broadcast_to(prefix, (len(tails), *prefix.shape)), tails), axis=1)
    return {"actions": actions, "specs": specs, "epsilon": float(epsilon),
            "selected_chain_sha256_float32": hashlib.sha256(chain.tobytes()).hexdigest(),
            "tail_steps": len(chain) - prefix_steps, "coordinates": chain[prefix_steps:].size}


def evaluate(request, endpoints, branches, goal):
    """Evaluate endpoint secants, preserving direction and branch classifications."""
    actions = request["actions"]
    endpoints = np.asarray(endpoints)
    branches = np.asarray(branches)
    if len(endpoints) != len(actions) or len(branches) != len(actions):
        raise ValueError("response batch mismatch")
    matrix, stable = differences(endpoints, branches, request["specs"])
    distance_vector = endpoints[0] - np.asarray(goal)
    norm = float(np.linalg.norm(distance_vector))
    gradient = None if norm <= 1e-10 else (matrix.T @ (distance_vector / norm)).tolist()
    return {"matrix": matrix.tolist(), "goal_gradient": gradient,
            "coordinate_branch_stable": stable,
            "coordinate_class": [
                spec["kind"] if same else "cross_branch_" + spec["kind"]
                for spec, same in zip(request["specs"], stable, strict=True)],
            "all_branch_stable": all(stable)}
