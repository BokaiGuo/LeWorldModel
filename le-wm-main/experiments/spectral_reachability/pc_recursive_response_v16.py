"""Task-independent multi-nominal development variant of the v13 compiler."""
from dataclasses import replace

import numpy as np

from experiments.spectral_reachability import pc_recursive_response_v13 as v13


def extra_nominals(segment_length, action_dim, seed):
    if min(segment_length, action_dim) < 1:
        raise ValueError("invalid action dimensions")
    points = np.random.default_rng(seed).uniform(-.8, .8, (4, action_dim))
    return tuple(tuple(tuple(float(x) for x in point) for _ in range(segment_length))
                 for point in points)


def compile_program(predict, target, config, nominal_seed, carry=None):
    extended = tuple(config.nominals) + extra_nominals(config.segment_length, config.action_dim, nominal_seed)
    result = v13.compile_program(predict, target, replace(config, nominals=extended), carry=carry)
    return {**result, "method": "recursive_response_v16_multinominal_development",
            "nominal_seed": int(nominal_seed), "extra_nominals": extended[len(config.nominals):]}
