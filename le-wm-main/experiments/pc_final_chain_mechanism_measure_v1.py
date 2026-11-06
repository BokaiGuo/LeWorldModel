"""Request-journaled final-chain measurement for prospective intervention callers.

This does not seal plans or decide whether a mechanism gate passes. The caller
must seal all plans, dependencies, protocol, controls, and budgets in advance.
"""
import numpy as np

from experiments import pc_final_chain_response_v1 as response


def measure(final_chain, controls, goal, predict, branch_rollout, emit,
            *, scales=(.01, .05), prefix_steps=5):
    """Measure every control at the actual selected tail, persisting before calls.

    ``emit`` must durably persist its argument or raise. ``predict`` must itself
    account for prediction cost; branch rollouts are a separate cost category.
    """
    if not controls or "selected" not in controls:
        raise ValueError("selected control required")
    selected = np.asarray(final_chain, np.float32)
    if not np.array_equal(np.asarray(controls["selected"], np.float32), selected[:prefix_steps]):
        raise ValueError("selected control must be actual final-chain prefix")
    results = {}
    for name, prefix in controls.items():
        results[name] = {}
        for epsilon in scales:
            request = response.prepare(selected, prefix, prefix_steps=prefix_steps, epsilon=epsilon)
            actions = request["actions"]
            identity = {"control": name, "epsilon": float(epsilon),
                        "chain_sha256_float32": request["selected_chain_sha256_float32"]}
            emit({"event": "request", **identity, "actions": actions.tolist(),
                  "specs": request["specs"], "coordinates": request["coordinates"],
                  "endpoint_requested_chains": len(actions)})
            positions, scores = predict(actions.copy())
            positions, scores = np.asarray(positions), np.asarray(scores)
            if scores.shape != (len(actions),):
                raise ValueError("score batch mismatch")
            emit({"event": "branch_request", **identity,
                  "branch_requested_chains": len(actions)})
            branch_positions, branches = branch_rollout(actions.copy())
            np.testing.assert_allclose(branch_positions, positions, rtol=0, atol=1e-12)
            result = response.evaluate(request, positions, branches, goal)
            result.update({"endpoint_nominal": positions[0].tolist(),
                           "absolute_nominal_score": float(scores[0]),
                           "endpoint_positions": positions.tolist(),
                           "scores": scores.tolist(), "branches": np.asarray(branches).tolist()})
            emit({"event": "result", **identity, "result": result})
            results[name][str(epsilon)] = result
    return results
