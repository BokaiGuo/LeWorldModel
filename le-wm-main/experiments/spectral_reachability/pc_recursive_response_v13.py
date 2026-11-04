"""Budgeted global action proposals feeding the v12 response compiler.

The proposal bank contains no physical-state or witness inputs.  Its
predictions and the nested compiler's predictions share one token cap.
"""
from dataclasses import asdict, dataclass
import time

import numpy as np

from experiments.spectral_reachability import pc_recursive_response_v12 as core


@dataclass(frozen=True)
class Config(core.Config):
    global_samples: int = 64


def proposal_bank(config, carry=None):
    """Construct the same unprivileged global bank for method and controls."""
    c = config
    horizon, dim = c.segment_length * c.segments, c.action_dim
    if (min(horizon, dim, c.model_packet_length, c.global_samples) < 1
            or horizon % c.model_packet_length or c.global_samples % 2):
        raise ValueError("invalid global proposal configuration")
    old_carry = None
    if carry is not None:
        old_carry = np.asarray(carry, np.float32)
        if (old_carry.shape != (horizon, dim) or not np.isfinite(old_carry).all()
                or np.any(np.abs(old_carry) > 1)):
            raise ValueError("invalid carry")
    rng = np.random.default_rng(c.seed + 9137)
    half = c.global_samples // 2
    constant = np.broadcast_to(rng.uniform(-1, 1, (half, 1, dim)),
                               (half, horizon, dim))
    pieces = rng.uniform(-1, 1, (half, 2, dim))
    selector = (np.arange(horizon) >= horizon // 2).astype(int)
    two_phase = pieces[:, selector, :]
    zero = np.zeros((1, horizon, dim))
    parts = [zero]
    if old_carry is not None:
        parts.append(old_carry[None])
    parts.extend((constant, two_phase))
    return np.concatenate(parts).astype(np.float32)


def compile_program(predict, target, config, carry=None):
    c = config
    bank = proposal_bank(c, carry)
    horizon = c.segment_length * c.segments
    packets = horizon // c.model_packet_length
    bank_count = len(bank)
    bank_cost = bank_count * packets
    if bank_cost >= c.token_cap:
        raise ValueError("insufficient budget for global proposals and compiler")
    started = time.monotonic()
    features, scores = predict(bank.copy())
    features, scores = np.asarray(features, np.float64), np.asarray(scores, np.float64)
    target_array = np.asarray(target, np.float64)
    if (features.shape != (bank_count, target_array.size) or scores.shape != (bank_count,)
            or not np.isfinite(features).all() or not np.isfinite(scores).all()):
        raise ValueError("invalid global predictor output")
    choice = int(np.argmin(scores))
    kwargs = asdict(c)
    del kwargs["global_samples"]
    kwargs["token_cap"] -= bank_cost
    child = core.compile_program(predict, target, core.Config(**kwargs), carry=bank[choice])
    resource = dict(child["resource"])
    resource["prediction_tokens"] += bank_cost
    resource["prediction_calls"] += 1
    resource["full_samples"] += bank_count
    resource["token_cap"] = c.token_cap
    resource["wall_seconds"] = time.monotonic() - started
    if resource["prediction_tokens"] > c.token_cap:
        raise ValueError("budget exceeded")
    return {"method": "recursive_response_v13_global_development", "config": asdict(c),
            "global_proposals": {"actions": bank.tolist(), "scores": scores.tolist(),
                                 "choice": choice, "samples": bank_count, "tokens": bank_cost},
            "child": child, "selected": child["selected"], "selected_score": child["selected_score"],
            "incumbent": child["incumbent"], "comparison": child["comparison"],
            "paired_scores": child["paired_scores"], "abstained": child["abstained"],
            "resource": resource}
