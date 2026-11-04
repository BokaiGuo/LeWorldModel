"""Development adapter: test the final executed prefix against a zero prefix."""
import hashlib
from pathlib import Path

import numpy as np

from experiments import pc_fixed_terminal_fair_feedback_dev_20261103_codex as runner
from experiments import pc_traced_adaptive_rgb_dev_20261103_codex as adaptive

ROOT = runner.ROOT
PROTOCOL = ROOT.parent / "docs/PC_FINAL_CHAIN_ABSTENTION_RGB_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT / "results/pc_final_chain_abstention_rgb_dev_20261103_codex"
ARMS = adaptive.ARMS


class ReservedMeter:
    """Expose a reserved token budget to a policy without changing its cap."""

    def __init__(self, meter, reserve):
        self.meter = meter
        self.reserve = reserve

    @property
    def token_samples(self):
        return self.meter.token_samples + self.reserve


def decide(arm, root, step, target, predict, token_meter, carry=None):
    if arm not in ARMS or step not in (0, 1, 2):
        raise ValueError("invalid arm or step")
    # One complete rollout is held back from every arm, including CEM.
    horizon = 15 - 5 * step
    reserve = horizon // 5
    start = token_meter.token_samples
    result = adaptive.decide(arm, root, step, target, predict,
                             ReservedMeter(token_meter, reserve), carry)
    selected = result["selected"]
    chain = np.asarray(selected["chain"], dtype=float)
    if chain.shape != (horizon, 2) or not np.isfinite(chain).all():
        raise ValueError("invalid final chain")
    if arm == "cem":
        # A direct full-horizon control uses its reserved rollout to rescore
        # its own selected chain; it is not assigned abstention semantics.
        comparison = chain.copy()
    else:
        comparison = chain.copy()
        comparison[:5] = 0.
    before = token_meter.token_samples
    positions, scores = predict(comparison[None])
    scores = np.asarray(scores)
    if scores.shape != (1,) or not np.isfinite(scores).all():
        raise ValueError("invalid final counterfactual score")
    if token_meter.token_samples - before != reserve:
        raise ValueError("counterfactual was not fully metered")
    old_score = float(selected["terminal_score"])
    if not np.isfinite(old_score):
        raise ValueError("invalid selected score")
    if arm != "cem":
        abstained = float(scores[0]) <= old_score
        if abstained:
            result["selected"] = {"name": "zero_prefix_counterfactual",
                                  "chain": comparison.tolist(), "terminal_score": float(scores[0])}
        result["selection_source"] = result["selected"]["name"]
        result["abstained_reconfiguration"] = bool(abstained)
    else:
        result["abstained_reconfiguration"] = None
    result["final_prefix_comparison"] = {
        "precomparison_source": selected["name"], "precomparison_chain": chain.tolist(),
        "precomparison_score": old_score, "comparison_chain": comparison.tolist(),
        "comparison_score": float(scores[0]), "comparison_position": np.asarray(positions).tolist(),
        "additional_prediction_tokens": token_meter.token_samples - before,
        "tokens_used_this_decision": token_meter.token_samples - start,
        "rule": "zero prefix with unchanged final suffix; ties abstain" if arm != "cem" else "rescore",
    }
    result["final_abstention_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main():
    runner.PROTOCOL, runner.PREFIX, runner.ARMS, runner.decide = PROTOCOL, PREFIX, ARMS, decide
    runner.main()


if __name__ == "__main__":
    main()
