"""Control-flow kernel for fixed 15-step native plans; no environment construction.

Callers must bind authentic source/reset/render/dispatch and a durable cp5
guard. This kernel is not a production runner or physical-release authority.
"""
import numpy as np


def sequence(raw15, dispatch, capture, guard):
    """Run literal actions in order; stop on native termination and failed guard.

    dispatch(step, raw2) returns a dict with boolean `terminated` and
    `truncated`; capture(step, returned) must durably save the observation and
    return its identity; guard(step, identity) must durably check or establish
    the exact prefix identity and return literal True before action 6.
    Exceptions intentionally propagate; callers retain partial records.
    """
    arr = np.asarray(raw15)
    if (arr.shape != (15, 2) or arr.dtype.kind not in "fi" or not np.isfinite(arr).all()
            or not np.array_equal(arr, arr.astype(np.float32)) or (np.abs(arr) > 1).any()):
        raise ValueError("expected literal bounded float32-compatible raw15")
    arr = arr.astype(np.float32)
    checkpoints = {0: None, 5: None, 10: None, 15: None}
    checkpoints[0] = capture(0, None)
    if checkpoints[0] is None:
        raise ValueError("missing initial observation")
    returned = 0
    for i, command in enumerate(arr, 1):
        transition = dispatch(i, command.copy())
        if (not isinstance(transition, dict) or type(transition.get("terminated")) is not bool
                or type(transition.get("truncated")) is not bool):
            raise ValueError("native termination identity missing")
        returned += 1
        observation = capture(i, transition)
        if observation is None:
            raise ValueError("missing returned observation")
        if i in checkpoints:
            checkpoints[i] = observation
        if transition["terminated"] or transition["truncated"]:
            return dict(status="native_stopped", returned_steps=returned, checkpoints=checkpoints)
        if i == 5 and guard(i, observation) is not True:
            raise ValueError("prefix guard rejected; no suffix dispatch")
    return dict(status="horizon_completed", returned_steps=returned, checkpoints=checkpoints)
