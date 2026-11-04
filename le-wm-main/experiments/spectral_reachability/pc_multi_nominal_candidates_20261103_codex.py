"""Outcome-free, finite-horizon action-response candidates."""
import numpy as np


def build_candidates(predict, lengths=(5, 10), epsilon=.05, perturbation=.25, keep_per_length=4,
                     svd=np.linalg.svd):
    """Use only predicted response features; caller instruments predict and SVD.

    predict accepts a single bounded H-by-2 raw action sequence and returns
    finite fixed-dimensional model features. The generated sequences are not
    complete programs and cannot be classified as successful mechanisms.
    """
    if (not lengths or len(set(lengths)) != len(lengths)
            or any(type(h) is not int or h <= 0 or h % 5 for h in lengths)
            or not 0 < epsilon <= .1 or not 0 < perturbation <= .5
            or type(keep_per_length) is not int or keep_per_length < 1):
        raise ValueError("invalid candidate-generation policy")
    retained, evidence = [], []
    for horizon in lengths:
        group = []
        for nominal_index, x in enumerate((0., .25, -.25)):
            nominal = np.tile(np.array([x, 0.]), (horizon, 1))
            columns = []
            for unit in np.eye(2*horizon).reshape(2*horizon, horizon, 2):
                plus = np.asarray(predict(nominal+epsilon*unit), dtype=float).ravel()
                minus = np.asarray(predict(nominal-epsilon*unit), dtype=float).ravel()
                if plus.shape != minus.shape or not plus.size or not np.isfinite(plus).all() or not np.isfinite(minus).all():
                    raise ValueError("invalid model feature response")
                columns.append((plus-minus)/(2*epsilon))
            jacobian = np.stack(columns, axis=1)
            _, singular, vh = svd(jacobian, full_matrices=False)
            entry = {"horizon": horizon, "nominal_index": nominal_index, "nominal": nominal.tolist(),
                     "jacobian": jacobian.tolist(), "singular_values": singular.tolist()}
            evidence.append(entry)
            if not len(singular) or singular[0] <= 1e-10:
                continue
            direction = vh[0].reshape(horizon, 2)
            direction = direction/np.max(np.abs(direction))
            for sign in (1, -1):
                action = np.clip(nominal+sign*perturbation*direction, -1, 1)
                feature = np.asarray(predict(action), dtype=float).ravel()
                if feature.shape != plus.shape or not np.isfinite(feature).all():
                    raise ValueError("invalid candidate feature response")
                group.append({"name": f"h{horizon}_n{nominal_index}_s{sign:+d}", "horizon": horizon,
                              "nominal_index": nominal_index, "prefix": action.tolist(),
                              "predicted_response": feature.tolist()})
        # Stable initial anchor and farthest-point selection, no task/outcome rank.
        if group:
            picked, remaining = [group[0]], group[1:]
            while remaining and len(picked) < keep_per_length:
                i = max(range(len(remaining)), key=lambda k: min(
                    np.linalg.norm(np.asarray(remaining[k]["predicted_response"])-x["predicted_response"])
                    for x in picked))
                picked.append(remaining.pop(i))
            retained.extend(picked)
    return {"candidates": retained, "evidence": evidence}
