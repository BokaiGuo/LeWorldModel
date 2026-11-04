"""Development RGB policy with actual successor-conditioned composition."""
import hashlib
from pathlib import Path

import numpy as np

from experiments import pc_final_chain_abstention_rgb_dev_20261103_codex as fallback
from experiments import pc_fixed_terminal_fair_feedback_dev_20261103_codex as runner
from experiments import pc_multi_nominal_rgb_dev_20261103_codex as policy
from experiments.pc_budget_allocation_20261103_codex import allocate_samples, optimize
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter

ROOT = runner.ROOT
PROTOCOL = ROOT.parent / "docs/PC_SUCCESSOR_RGB_EXECUTION_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT / "results/pc_successor_rgb_execution_dev_20261103_codex"
ARMS = ("successor", "cem")
NOMINALS = policy.NOMINALS[:1]


def successor_first(root, target, predict, token_meter):
    """Generate and optimize four two-segment prefixes, all on the same model."""
    seed = 2026120800 + root
    meter = Meter({"prediction_samples": 63, "svd": 3, "linear_solve": 0})
    first, evidence = policy.generator.generate(
        lambda x: predict(np.asarray(x)[None])[0][0], target, NOMINALS, meter, mode="svd")
    bank, successors = [], []
    for item in first:
        prefix = np.asarray(item["prefix"])
        second, ev = policy.generator.generate(
            lambda x: predict(np.concatenate((prefix, np.asarray(x)))[None])[0][0],
            target, NOMINALS, meter, mode="svd")
        successors.append(ev)
        for item2 in second:
            bank.append({"name": item["name"] + "/" + item2["name"],
                         "prefix": np.concatenate((prefix, item2["prefix"])).tolist()})
    affordable = (policy.CAP - token_meter.token_samples) // 3
    counts = allocate_samples(3 * affordable // 4, 0, len(bank), 1)
    if min(counts) < 13:
        raise ValueError("not enough budget for two CEM batches per prefix")
    programs = []
    for item, count in zip(bank, counts, strict=True):
        chain, score, scores = optimize(item["prefix"], 15, count,
                                        np.random.default_rng(seed), lambda x: predict(x)[1])
        programs.append({"name": item["name"], "chain": chain.tolist(),
                         "terminal_score": score, "samples": count, "scores": scores})
    selected = min(programs, key=lambda item: item["terminal_score"])
    count = (policy.CAP - token_meter.token_samples) // 3
    chain, score = policy.refine(selected["chain"], selected["terminal_score"], count,
                                 np.random.default_rng(seed + 1), lambda x: predict(x)[1])
    if score < selected["terminal_score"]:
        selected = {"name": "joint_refined", "chain": chain.tolist(), "terminal_score": score,
                    "refined_from": selected["name"], "samples": count}
    return {"horizon": 15, "programs": programs, "first_bank": first, "bank": bank,
            "generator": evidence, "successor_generator": successors, "selected": selected,
            "selection_source": selected["name"], "carry": None, "carry_score": None,
            "spectral_counts": meter.counts, "spectral_seconds": meter.seconds}


def decide(arm, root, step, target, predict, token_meter, carry=None):
    if arm not in ARMS or step not in (0, 1, 2):
        raise ValueError("invalid decision")
    horizon, trace = 15 - 5 * step, []
    reserve = horizon // 5
    budget = fallback.ReservedMeter(token_meter, reserve)

    def traced(actions):
        positions, scores = predict(actions)
        actions = np.asarray(actions)
        trace.append({"requested_actions": actions.tolist(), "model_actions": actions.astype(np.float32).tolist(),
                      "positions": np.asarray(positions).tolist(), "scores": np.asarray(scores).tolist(),
                      "tokens": len(actions) * actions.shape[1] // 5})
        return positions, scores

    if arm == "successor" and step == 0:
        if carry is not None:
            raise ValueError("unexpected carry")
        result = successor_first(root, target, traced, budget)
    else:
        result = policy.decide("svd" if arm == "successor" else "cem",
                               root, step, target, traced, budget, carry)
    selected = result["selected"]
    chain = np.asarray(selected["chain"], dtype=float)
    comparison = chain.copy()
    if arm == "successor":
        comparison[:5] = 0.
    before = token_meter.token_samples
    _, value = traced(comparison[None])
    if token_meter.token_samples - before != reserve:
        raise ValueError("comparison not metered")
    score = float(np.asarray(value)[0])
    if not np.isfinite(score):
        raise ValueError("invalid comparison score")
    abstained = arm == "successor" and score <= selected["terminal_score"]
    if abstained:
        result["selected"] = {"name": "zero_prefix_counterfactual", "chain": comparison.tolist(),
                              "terminal_score": score}
    result["selection_source"] = result["selected"]["name"]
    result["abstained_reconfiguration"] = bool(abstained) if arm == "successor" else None
    result["final_prefix_comparison"] = {"precomparison_source": selected["name"],
                                         "precomparison_chain": chain.tolist(),
                                         "precomparison_score": selected["terminal_score"],
                                         "comparison_chain": comparison.tolist(), "comparison_score": score}
    result["prediction_trace"] = trace
    result["policy_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main():
    runner.PROTOCOL, runner.PREFIX, runner.ARMS, runner.decide = PROTOCOL, PREFIX, ARMS, decide
    runner.main()


if __name__ == "__main__":
    main()
