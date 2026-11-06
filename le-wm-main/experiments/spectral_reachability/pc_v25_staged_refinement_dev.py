"""Development-only staged optimizer adapter without mutating sealed modules."""
from types import FunctionType, SimpleNamespace

from experiments.spectral_reachability import pc_global_reachability_v25 as frozen
from experiments.spectral_reachability import pc_global_shared_cem_v23 as baseline
from experiments.spectral_reachability import pc_staged_multistart_cem_dev as staged

Config = frozen.Config


def clone(function, replacements):
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def compile_program(predict, target, config, *, score_scale, score_target, carry=None):
    core = clone(frozen.core.compile_program, {"pc_shared_multistart_cem": staged})
    compile_global = clone(frozen.compile_program, {
        "core": SimpleNamespace(Config=frozen.core.Config, compile_program=core)})
    result = compile_global(predict, target, config, score_scale=score_scale,
                            score_target=score_target, carry=carry)
    result["method"] = "v25_staged_refinement_development"
    return result


def plan_cem(predict, config, *, seed, carry=None):
    plan = clone(baseline.plan, {"cem": staged})
    return plan(predict, config, seed=seed, carry=carry)
