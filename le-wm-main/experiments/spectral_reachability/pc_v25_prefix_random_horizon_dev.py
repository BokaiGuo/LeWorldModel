"""Horizon-compatible development policy; not a response-reliability fallback."""
from experiments.spectral_reachability import pc_v25_prefix_random_refinement_dev as variant

Config = variant.Config


def compile_program(predict, target, config, *, score_scale, score_target, carry=None):
    if config.segments < 1:
        raise ValueError("segments must be positive")
    random_tail = config.segments > 1
    result = variant.compile_program(
        predict, target, config, score_scale=score_scale, score_target=score_target,
        carry=carry, random_refinement=random_tail,
    )
    result["horizon_policy"] = {
        "name": "prefix_random_when_tail_exists_else_v25",
        "segments": config.segments,
        "random_tail_enabled": random_tail,
        "fallback_reason": None if random_tail else "no_tail_after_first_segment",
    }
    return result
