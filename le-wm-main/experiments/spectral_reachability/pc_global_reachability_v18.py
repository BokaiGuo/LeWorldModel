"""Global proposals plus score-augmented successor-reachability search."""
from dataclasses import asdict, dataclass
import time

import numpy as np

from experiments.spectral_reachability import pc_recursive_reachability_v18 as core
from experiments.spectral_reachability import pc_recursive_response_v13 as v13


@dataclass(frozen=True)
class Config(core.Config):
    global_samples: int = 64


def compile_program(predict, target, config, *, score_scale, score_target, carry=None):
    target = np.asarray(target, np.float64)
    if (target.ndim != 1 or not target.size or not np.isfinite(target).all()
            or not np.isfinite(score_scale) or score_scale <= 0 or not np.isfinite(score_target)):
        raise ValueError("invalid target or score geometry")
    bank = v13.proposal_bank(config, carry)
    horizon = config.segment_length * config.segments
    packets = horizon // config.model_packet_length
    cost = len(bank) * packets
    if cost >= config.token_cap:
        raise ValueError("insufficient global bank budget")
    started = time.monotonic()
    features, scores = predict(bank.copy())
    features, scores = np.asarray(features, np.float64), np.asarray(scores, np.float64)
    if (features.shape != (len(bank), len(target)) or scores.shape != (len(bank),)
            or not np.isfinite(features).all() or not np.isfinite(scores).all()):
        raise ValueError("invalid global predictor output")
    choice = int(np.argmin(scores))

    def augmented(actions):
        features, scores = predict(actions)
        features, scores = np.asarray(features, np.float64), np.asarray(scores, np.float64)
        if features.shape != (len(actions), len(target)) or scores.shape != (len(actions),):
            raise ValueError("invalid predictor output")
        return np.concatenate((features, score_scale * scores[:, None]), axis=1), scores

    kwargs = asdict(config)
    del kwargs["global_samples"]
    kwargs["token_cap"] -= cost
    extended = np.concatenate((target, [score_scale * score_target]))
    child = core.compile_program(augmented, extended, core.Config(**kwargs), carry=bank[choice])
    resource = dict(child["resource"])
    resource["prediction_tokens"] += cost
    resource["prediction_calls"] += 1
    resource["full_samples"] += len(bank)
    resource["token_cap"] = config.token_cap
    resource["wall_seconds"] = time.monotonic() - started
    if resource["prediction_tokens"] > config.token_cap:
        raise ValueError("budget exceeded")
    return {"method": "global_reachability_v18_development", "config": asdict(config),
            "score_geometry": {"scale": float(score_scale), "target": float(score_target)},
            "global_proposals": {"actions": bank.tolist(), "scores": scores.tolist(),
                                 "choice": choice, "samples": len(bank), "tokens": cost},
            "child": child, "selected": child["selected"], "selected_score": child["selected_score"],
            "incumbent": child["incumbent"], "comparison": child["comparison"],
            "paired_scores": child["paired_scores"], "abstained": child["abstained"],
            "resource": resource}
