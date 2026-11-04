"""Outcome-free spectral prefix bank and shared suffix planning primitives.

The caller owns real observation encoding, model prediction, and execution.
Counters here cover only calls made through this module; they do not certify a
whole experiment or convert heterogeneous costs into a single resource unit.
"""
from dataclasses import dataclass, field
import time

import numpy as np

from experiments.spectral_reachability.pc_rgb_candidate_policy_20261103_codex import build_fixed_bank


@dataclass
class Meter:
    limits: dict
    counts: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)

    def call(self, kind, amount, function, *args):
        if kind not in self.limits or amount < 1:
            raise ValueError("missing budget category or invalid amount")
        if self.counts.get(kind, 0) + amount > self.limits[kind]:
            raise RuntimeError(f"{kind} budget exhausted")
        self.counts[kind] = self.counts.get(kind, 0) + amount
        start = time.monotonic()
        try:
            return function(*args)
        finally:
            self.seconds[kind] = self.seconds.get(kind, 0.0) + time.monotonic() - start


def spectral_bank(response, meter, seed, epsilon=0.05, amplitude=0.5):
    """Add both signs of the top two response singular vectors.

    response maps an actual (5,2) action prefix to a model-predicted feature
    vector. Central differences are with respect to raw actions at zero.
    """
    if not 0 < epsilon <= 1 or not 0 < amplitude <= 1:
        raise ValueError("invalid spectral scale")
    basis = np.eye(10).reshape(10, 5, 2)
    columns = []
    for direction in basis:
        plus = np.asarray(meter.call("prediction_samples", 1, response, epsilon * direction), dtype=float).ravel()
        minus = np.asarray(meter.call("prediction_samples", 1, response, -epsilon * direction), dtype=float).ravel()
        if plus.shape != minus.shape or not plus.size or not np.isfinite(plus).all() or not np.isfinite(minus).all():
            raise ValueError("invalid model response")
        columns.append((plus - minus) / (2 * epsilon))
    jacobian = np.stack(columns, axis=1)
    _, singular, vh = meter.call("svd", 1, np.linalg.svd, jacobian, False)
    bank = build_fixed_bank(seed)
    for i, vector in enumerate(vh[:2]):
        # A null response has no identified action direction.
        if singular[i] <= 1e-10:
            continue
        vector = vector.reshape(5, 2)
        scale = np.max(np.abs(vector))
        if scale <= 1e-12:
            continue
        for sign in (1, -1):
            bank.append({"name": f"spectral_{i}_{sign:+d}", "source": "rgb_model",
                         "prefix": (sign * amplitude * vector / scale).tolist()})
    return bank, {"jacobian": jacobian.tolist(), "singular_values": singular.tolist(),
                  "epsilon": epsilon, "amplitude": amplitude, "seed": seed}


def plan_bank(bank, initial_score, prefix_score, optimize_suffix, meter, suffix_samples,
              retained_per_group=None):
    """Compare complete model programs; greedy uses the same initial bank.

    optimize_suffix(prefix, sample_budget, meter) returns (suffix, terminal_score)
    and must route prediction samples through meter.call.
    If a group cap is specified, select within predicted progress/nonprogress
    by prefix score first, but never force the final winner to be nonprogress.
    """
    if not bank or not np.isfinite(initial_score) or suffix_samples < 1:
        raise ValueError("invalid planning inputs")
    names = [x["name"] for x in bank]
    if len(set(names)) != len(names) or names.count("zero") != 1:
        raise ValueError("bank must contain one zero and unique names")
    scored = []
    for item in bank:
        prefix = np.asarray(item["prefix"], dtype=float)
        if prefix.shape != (5, 2) or not np.isfinite(prefix).all() or np.max(np.abs(prefix)) > 1:
            raise ValueError("invalid prefix")
        score = float(meter.call("prediction_samples", 1, prefix_score, prefix))
        if not np.isfinite(score):
            raise ValueError("nonfinite prefix score")
        scored.append({**item, "prefix_score": score})
    greedy = min(scored, key=lambda x: x["prefix_score"])
    if retained_per_group is None:
        retained = scored
    else:
        if retained_per_group < 1:
            raise ValueError("invalid retained group size")
        groups = [[x for x in scored if (x["prefix_score"] < initial_score) == progress]
                  for progress in (True, False)]
        retained = [x for group in groups for x in sorted(group, key=lambda x: x["prefix_score"])[:retained_per_group]]
        # Always keep the explicit zero control, regardless of its group.
        zero = next(x for x in scored if x["name"] == "zero")
        if zero not in retained:
            retained.append(zero)
    programs = []
    for item in retained:
        before = meter.counts.get("prediction_samples", 0)
        suffix, terminal = meter.call("suffix_optimizations", 1, optimize_suffix,
                                      np.asarray(item["prefix"]), suffix_samples, meter)
        consumed = meter.counts.get("prediction_samples", 0) - before
        if consumed < 1 or consumed > suffix_samples:
            raise RuntimeError("suffix optimizer did not respect metered sample cap")
        suffix = np.asarray(suffix, dtype=float)
        if suffix.shape != (10, 2) or not np.isfinite(suffix).all() or np.max(np.abs(suffix)) > 1 or not np.isfinite(terminal):
            raise ValueError("invalid suffix optimization result")
        programs.append({**item, "suffix": suffix.tolist(), "terminal_score": float(terminal)})
    selected = min(programs, key=lambda x: x["terminal_score"])
    zero = next(x for x in programs if x["name"] == "zero")
    return {"selected": selected, "same_bank_greedy_name": greedy["name"],
            "prefix_scores": scored, "programs": programs,
            "qualified_against_zero_only": selected["prefix_score"] >= initial_score
            and selected["terminal_score"] < zero["terminal_score"]}
