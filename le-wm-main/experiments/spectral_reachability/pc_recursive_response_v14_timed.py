"""v14-compatible development adapter with correctly scoped outer timing.

The original v14 and its sealed runs remain untouched. This wrapper does not
change planning or establish comparable GPU timing; callers must synchronize
accelerator work inside their predictor if they report wall time.
"""
import time

from experiments.spectral_reachability import pc_recursive_response_v14 as v14


def compile_program(predict, target, config, carry=None):
    started = time.monotonic()
    result = v14.compile_program(predict, target, config, carry=carry)
    elapsed = time.monotonic() - started
    resource = dict(result["resource"])
    resource["child_wall_seconds"] = resource.get("wall_seconds")
    resource["wall_seconds"] = elapsed
    return {**result, "method": "recursive_response_v14_timed_development", "resource": resource}
