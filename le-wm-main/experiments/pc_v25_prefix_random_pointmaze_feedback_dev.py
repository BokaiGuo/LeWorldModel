"""PointMaze development adapter using one horizon-dependent candidate policy.

The existing timed executor is isolated per call: historical experiment
dependencies and module globals are not modified.
"""
from types import FunctionType, SimpleNamespace

from experiments import pc_v25_pointmaze_feedback_adapter as base
from experiments.spectral_reachability import pc_v25_prefix_random_horizon_dev as policy

CAP = base.CAP
METHOD = "v25_prefix_random_horizon_pointmaze_development"


def _clone(function, replacements):
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


def decide(arm, root, step, target, predict, resource, carry=None):
    executor = _clone(base.timed.execute, {"pc_global_reachability_v25": policy})
    decision = _clone(base.decide, {"compiler": policy, "timed": SimpleNamespace(execute=executor)})
    result = decision(arm, root, step, target, predict, resource, carry)
    if arm == "response":
        result["selected"]["name"] = METHOD
    result["adapter_method"] = METHOD
    return result
