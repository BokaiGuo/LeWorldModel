"""Experimental v25 refinement-seed ablation without modifying frozen modules.

This is an isolated development harness, not a replacement formal compiler.
Only already-budgeted refinement seed slots after the incumbent are changed.
"""
from types import FunctionType, SimpleNamespace

import numpy as np

from experiments.spectral_reachability import pc_global_reachability_v25 as frozen

Config = frozen.Config


def compile_program(predict, target, config, *, score_scale, score_target, carry=None,
                    random_refinement=False):
    if not isinstance(random_refinement, bool):
        raise ValueError("random_refinement must be boolean")
    if not random_refinement:
        return frozen.compile_program(predict, target, config, score_scale=score_scale,
                                      score_target=score_target, carry=carry)

    metadata = []
    shared = frozen.core.pc_shared_multistart_cem

    def optimize(evaluate, prefix, initial_tails, sample_budget, **kwargs):
        seeds = np.asarray(initial_tails, np.float32).copy()
        if np.asarray(prefix).shape[0] != 0:
            raise ValueError("random seed policy supports complete-chain refinement only")
        rng = np.random.default_rng(np.random.SeedSequence([20310000, int(kwargs["seed"])]))
        if len(seeds) > 1:
            seeds[1:] = rng.uniform(-1, 1, seeds[1:].shape).astype(np.float32)
        metadata.append({"replaced_seed_indices": list(range(1, len(seeds))),
                         "policy": "uniform_full_chain", "seed": int(kwargs["seed"])})
        return shared.optimize(evaluate, prefix, seeds, sample_budget, **kwargs)

    # Clone function globals instead of mutating shared module attributes:
    # independent calls and other experiments retain the original optimizer.
    core_globals = dict(frozen.core.compile_program.__globals__)
    core_globals["pc_shared_multistart_cem"] = SimpleNamespace(optimize=optimize)
    isolated_core = FunctionType(frozen.core.compile_program.__code__, core_globals,
                                 frozen.core.compile_program.__name__,
                                 frozen.core.compile_program.__defaults__,
                                 frozen.core.compile_program.__closure__)
    global_globals = dict(frozen.compile_program.__globals__)
    global_globals["core"] = SimpleNamespace(Config=frozen.core.Config, compile_program=isolated_core)
    isolated_global = FunctionType(frozen.compile_program.__code__, global_globals,
                                   frozen.compile_program.__name__,
                                   frozen.compile_program.__defaults__,
                                   frozen.compile_program.__closure__)
    isolated_global.__kwdefaults__ = frozen.compile_program.__kwdefaults__
    result = isolated_global(predict, target, config, score_scale=score_scale,
                             score_target=score_target, carry=carry)
    if len(metadata) != 1:
        raise RuntimeError("expected exactly one refinement invocation")
    for index in metadata[0]["replaced_seed_indices"]:
        start = result["child"]["refinement_starts"][index]
        start["replaced_source"] = start["source"]
        start["source"] = "uniform_full_chain"
    result["method"] = "v25_random_refinement_seed_development"
    result["refinement_seed_policy"] = metadata[0]
    return result
