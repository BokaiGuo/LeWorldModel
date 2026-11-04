"""Global full-chain proposals with carry-conditioned recursive nominals."""
from dataclasses import asdict, dataclass

import numpy as np

from experiments.spectral_reachability import pc_recursive_response_v13 as global_core
from experiments.spectral_reachability import pc_recursive_response_v14_child as core


@dataclass(frozen=True)
class Config(core.Config):
    global_samples: int = 64


def compile_program(predict, target, config, carry=None):
    bank = global_core.proposal_bank(config, carry)
    packets = (config.segment_length * config.segments) // config.model_packet_length
    bank_cost = len(bank) * packets
    if bank_cost >= config.token_cap:
        raise ValueError("insufficient budget for global proposals and compiler")
    features, scores = predict(bank.copy())
    features, scores = np.asarray(features, np.float64), np.asarray(scores, np.float64)
    target = np.asarray(target, np.float64)
    if (features.shape != (len(bank), target.size) or scores.shape != (len(bank),)
            or not np.isfinite(features).all() or not np.isfinite(scores).all()):
        raise ValueError("invalid global predictor output")
    choice = int(np.argmin(scores))
    kwargs = asdict(config)
    del kwargs["global_samples"]
    kwargs["token_cap"] -= bank_cost
    child = core.compile_program(predict, target, core.Config(**kwargs), carry=bank[choice])
    resource = dict(child["resource"])
    resource["prediction_tokens"] += bank_cost
    resource["prediction_calls"] += 1
    resource["full_samples"] += len(bank)
    resource["token_cap"] = config.token_cap
    if resource["prediction_tokens"] > config.token_cap:
        raise ValueError("budget exceeded")
    return {"method": "recursive_response_v14_carry_nominal_development", "config": asdict(config),
            "global_proposals": {"actions": bank.tolist(), "scores": scores.tolist(),
                                 "choice": choice, "samples": len(bank), "tokens": bank_cost},
            "child": child, "selected": child["selected"], "selected_score": child["selected_score"],
            "incumbent": child["incumbent"], "comparison": child["comparison"],
            "paired_scores": child["paired_scores"], "abstained": child["abstained"],
            "resource": resource}
