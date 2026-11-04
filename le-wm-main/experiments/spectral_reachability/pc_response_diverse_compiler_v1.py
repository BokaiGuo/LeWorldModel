"""Two-mode response proposals with model-based successor diversity."""
import numpy as np

from experiments import pc_successor_rgb_execution_dev_20261103_codex as prior
from experiments.pc_budget_allocation_20261103_codex import allocate_samples
from experiments.spectral_reachability.pc_cem_scale_floor_v1 import optimize
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter

ARMS = ("response_diverse", "response_greedy", "random_diverse", "full_cem")


def directions_from_response(target, predict, meter, nominal):
    def measure(action):
        return np.asarray(meter.call("prediction_samples", 1, lambda x: predict(x[None])[0][0], action))

    center = measure(nominal)
    epsilon = .05
    jacobian = np.stack([(measure(nominal + epsilon * axis) - measure(nominal - epsilon * axis))
                         / (2 * epsilon) for axis in np.eye(10).reshape(10, 5, 2)], axis=1)
    if jacobian.shape != (len(target), 10) or min(jacobian.shape) < 2:
        raise ValueError("response requires at least two output coordinates")
    _, singular, vt = meter.call("svd", 1, np.linalg.svd, jacobian, False)
    return vt[:2], {"nominal": nominal.tolist(), "center": center.tolist(),
                    "jacobian": jacobian.tolist(), "singular": singular.tolist(),
                    "right_directions": vt[:2].tolist()}


def first_bank(nominal, directions, source):
    bank = [{"name": "zero", "prefix": np.zeros((5, 2)).tolist(), "source": "zero"}]
    for index, vector in enumerate(directions):
        delta = np.asarray(vector).reshape(5, 2)
        magnitude = float(np.max(np.abs(delta)))
        if magnitude < 1e-12:
            raise ValueError("zero candidate direction")
        delta = .5 * delta / magnitude
        for sign in (-1, 1):
            bank.append({"name": f"{source}_{index}_{sign:+d}", "source": source,
                         "prefix": np.clip(nominal + sign * delta, -1, 1).tolist()})
    return bank


def decide(arm, root, target, predict, token_meter):
    if arm not in ARMS or token_meter.token_samples != 0:
        raise ValueError("invalid arm or nonempty meter")
    target = np.asarray(target, dtype=float)
    cap, reserve, seed = prior.policy.CAP, 3, 2026121600 + root
    meter = Meter({"prediction_samples": 63, "svd": 3, "linear_solve": 0})
    first, bank, retained, programs = [], [], [], []
    positions, prefix_scores, response, successors = [], [], None, []
    if arm == "full_cem":
        count = (cap - reserve) // 3
        chain, score, scores = optimize([], 15, count, np.random.default_rng(seed), lambda x: predict(x)[1])
        selected = {"name": "full_cem", "chain": chain.tolist(), "terminal_score": score,
                    "samples": count, "scores": scores}
    else:
        nominal = prior.NOMINALS[0].copy()
        random_rng = np.random.default_rng(2026121700 + root)
        if arm == "random_diverse":
            directions = np.linalg.qr(random_rng.normal(size=(10, 2)), mode="reduced")[0].T
        else:
            directions, response = directions_from_response(target, predict, meter, nominal)
        first = first_bank(nominal, directions, "random" if arm == "random_diverse" else "response")
        for item in first:
            pos, score = predict(np.asarray(item["prefix"])[None])
            positions.append(np.asarray(pos[0]))
            prefix_scores.append(float(score[0]))
        best = int(np.argmin(prefix_scores))
        if arm == "response_greedy":
            retained = np.argsort(prefix_scores, kind="stable")[:2].tolist()
        else:
            distance = np.linalg.norm(np.asarray(positions) - positions[best], axis=1)
            distance[best] = -np.inf
            retained = [best, int(np.argmax(distance))]
        for index in retained:
            parent = first[index]
            prefix = np.asarray(parent["prefix"])
            if arm == "random_diverse":
                delta = random_rng.normal(size=(5, 2))
                delta *= .5 / np.max(np.abs(delta))
                second = [("zero", np.zeros((5, 2))),
                          ("random_plus", np.clip(nominal + delta, -1, 1)),
                          ("random_minus", np.clip(nominal - delta, -1, 1))]
            else:
                _, evidence = prior.policy.generator.generate(
                    lambda a: predict(np.concatenate((prefix, np.asarray(a)))[None])[0][0],
                    target, prior.NOMINALS, meter, mode="svd")
                successors.append(evidence)
                delta = np.asarray(evidence[0]["bounded_delta"]).reshape(5, 2)
                second = [("zero", np.zeros((5, 2))),
                          ("response_plus", np.clip(nominal + delta, -1, 1)),
                          ("response_minus", np.clip(nominal - delta, -1, 1))]
            for name, second_prefix in second:
                bank.append({"name": parent["name"] + "/" + name,
                             "prefix": np.concatenate((prefix, second_prefix)).tolist(),
                             "parent_index": index})
        affordable = (cap - reserve - token_meter.token_samples) // 3
        counts = allocate_samples(3 * affordable // 4, 0, len(bank), 1)
        if min(counts) < 13:
            raise ValueError("insufficient iterative search budget")
        for item, count in zip(bank, counts, strict=True):
            chain, score, scores = optimize(item["prefix"], 15, count, np.random.default_rng(seed),
                                            lambda x: predict(x)[1])
            programs.append({"name": item["name"], "chain": chain.tolist(), "terminal_score": score,
                             "samples": count, "scores": scores})
        selected = min(programs, key=lambda x: x["terminal_score"])
        count = (cap - reserve - token_meter.token_samples) // 3
        chain, score = prior.policy.refine(selected["chain"], selected["terminal_score"], count,
                                           np.random.default_rng(seed + 1), lambda x: predict(x)[1])
        if score < selected["terminal_score"]:
            selected = {"name": "joint_refined", "chain": chain.tolist(), "terminal_score": score,
                        "refined_from": selected["name"], "samples": count}
    before = selected.copy()
    comparison = np.asarray(selected["chain"]).copy()
    if arm != "full_cem":
        comparison[:5] = 0
    comparison_score = float(predict(comparison[None])[1][0])
    abstained = arm != "full_cem" and comparison_score <= before["terminal_score"]
    if abstained:
        selected = {"name": "zero_prefix_counterfactual", "chain": comparison.tolist(),
                    "terminal_score": comparison_score}
    return {"arm": arm, "first_bank": first, "first_positions": [x.tolist() for x in positions],
            "prefix_scores": prefix_scores, "retained": retained, "bank": bank, "response": response,
            "successor_response": successors, "programs": programs, "precomparison": before,
            "comparison_chain": comparison.tolist(), "comparison_score": comparison_score,
            "abstained": bool(abstained) if arm != "full_cem" else None, "selected": selected,
            "structure_counts": meter.counts, "structure_seconds": meter.seconds}
