"""Multi-nominal development policy on the existing fixed-terminal RGB runner."""
import hashlib
from pathlib import Path

import numpy as np

from experiments import pc_fixed_terminal_fair_feedback_dev_20261103_codex as runner
from experiments.pc_budget_allocation_20261103_codex import allocate_samples, optimize
from experiments.spectral_reachability import pc_multi_nominal_response_dev as generator
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter

ROOT = runner.ROOT
PROTOCOL = ROOT.parent / "docs/PC_MULTI_NOMINAL_RGB_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT / "results/pc_multi_nominal_rgb_dev_20261103_codex"
ARMS = ("svd", "ridge", "nominal", "random", "greedy", "cem")
CAP = 480
NOMINALS = np.stack([np.tile([value, 0.], (5, 1)) for value in (0.25, -0.25)])


def refine(chain, initial_score, count, rng, predict):
    """Joint CEM retains the input incumbent, even if every sample is worse."""
    best = np.asarray(chain, dtype=float).copy()
    score = float(initial_score)
    loc, scale = best.copy(), np.full(best.shape, 1 / 3)
    for start in range(0, count, 12):
        size = min(12, count - start)
        samples = np.clip(loc + rng.normal(size=(size, *best.shape)) * scale, -1, 1)
        samples[0] = loc
        values = np.asarray(predict(samples), dtype=float)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError("invalid refinement scores")
        index = int(np.argmin(values))
        if values[index] < score:
            best, score = samples[index].copy(), float(values[index])
        if size > 1:
            elite = samples[np.argsort(values, kind="stable")[:min(4, size)]]
            loc = elite.mean(0)
            scale = np.maximum(elite.std(0), 1e-5)
    return best, score


def decide(arm, root, step, target, predict, token_meter, carry=None):
    if arm not in ARMS or step not in (0, 1, 2):
        raise ValueError("invalid decision")
    horizon = 15 - 5 * step
    per = horizon // 5
    if (carry is None) != (step == 0):
        raise ValueError("missing or unexpected previous tail")
    prior_score = None
    if carry is not None:
        carry = np.asarray(carry, dtype=float)
        if carry.shape != (horizon, 2):
            raise ValueError("carry horizon mismatch")
        prior_score = float(predict(carry[None])[1][0])
    seed = 2026113000 + 10 * root + step
    meter = Meter({"prediction_samples": 42, "svd": 2, "linear_solve": 2})
    evidence, prefix_scores, programs = [], [], []
    if arm == "cem":
        count = (CAP - token_meter.token_samples) // per
        chain, score, scores = optimize([], horizon, count, np.random.default_rng(seed),
                                        lambda x: predict(x)[1])
        selected = {"name": "direct_cem", "chain": chain.tolist(), "terminal_score": score,
                    "samples": count, "scores": scores}
        programs.append(selected)
    else:
        if arm in ("ridge", "svd", "greedy"):
            mode = "ridge" if arm == "ridge" else "svd"
            bank, evidence = generator.generate(
                lambda prefix: predict(np.asarray(prefix)[None])[0][0],
                target, NOMINALS, meter, mode=mode)
        else:
            bank = [{"name": "zero", "prefix": np.zeros((5, 2)).tolist()}]
            rng = np.random.default_rng(seed)
            for index, nominal in enumerate(NOMINALS):
                action = nominal.copy()
                if arm == "random":
                    direction = rng.normal(size=(5, 2))
                    action = np.clip(action + .5 * direction / np.max(np.abs(direction)), -1, 1)
                bank.append({"name": f"{arm}_{index}", "prefix": action.tolist()})
        for item in bank:
            score = float(predict(np.asarray(item["prefix"])[None])[1][0])
            prefix_scores.append({"name": item["name"], "score": score})
        if horizon > 5:
            retained = ([bank[int(np.argmin([x["score"] for x in prefix_scores]))]]
                        if arm == "greedy" else bank)
            affordable = (CAP - token_meter.token_samples) // per
            suffix_total = 3 * affordable // 4
            counts = allocate_samples(suffix_total, 0, len(retained), 1)
            for item, count in zip(retained, counts, strict=True):
                chain, score, scores = optimize(item["prefix"], horizon, count,
                                                np.random.default_rng(seed), lambda x: predict(x)[1])
                programs.append({"name": item["name"], "chain": chain.tolist(),
                                 "terminal_score": score, "samples": count, "scores": scores})
        else:
            programs = [{"name": item["name"], "chain": item["prefix"],
                         "terminal_score": entry["score"], "samples": 0, "scores": []}
                        for item, entry in zip(bank, prefix_scores, strict=True)]
        selected = min(programs, key=lambda x: x["terminal_score"])
        count = (CAP - token_meter.token_samples) // per
        chain, score = refine(selected["chain"], selected["terminal_score"], count,
                              np.random.default_rng(seed + 1), lambda x: predict(x)[1])
        if score < selected["terminal_score"]:
            selected = {"name": "joint_refined", "chain": chain.tolist(),
                        "terminal_score": score, "refined_from": selected["name"],
                        "samples": count}
    if carry is not None and prior_score < selected["terminal_score"]:
        selected = {"name": "carry", "chain": carry.tolist(), "terminal_score": prior_score}
    return {"horizon": horizon, "programs": programs, "prefix_scores": prefix_scores,
            "generator": evidence, "selected": selected, "selection_source": selected["name"],
            "abstained_reconfiguration": selected["name"] == "zero" if arm != "cem" else None,
            "carry": None if carry is None else carry.tolist(), "carry_score": prior_score,
            "spectral_counts": meter.counts, "spectral_seconds": meter.seconds,
            "policy_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "multi_generator_sha256": hashlib.sha256(Path(generator.__file__).read_bytes()).hexdigest()}


def main():
    # Keep the original runner and outputs untouched while binding its execution
    # loop to this pre-registered policy and a distinct result namespace.
    runner.PROTOCOL = PROTOCOL
    runner.PREFIX = PREFIX
    runner.ARMS = ARMS
    runner.decide = decide
    runner.main()


if __name__ == "__main__":
    main()
