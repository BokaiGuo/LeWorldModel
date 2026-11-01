"""Separate per-prefix resource axes and native outcome fields; no execution authority."""
import math
import time

import numpy as np

AXES = ('encoder', 'projector', 'predict', 'action_encoder', 'jacobian',
        'vjp_directions', 'svd', 'gradient', 'model_replay', 'physical_step', 'render')


def need(ok, message):
    if not ok:
        raise ValueError(message)


class Budget:
    def __init__(self, caps, sink, *, clock=time.monotonic, wall_seconds):
        need(type(caps) is dict and set(caps) == set(AXES) and
             all(type(v) is int and v >= 0 for v in caps.values()), 'all independent integer caps')
        need(type(wall_seconds) in (int, float) and math.isfinite(wall_seconds) and wall_seconds > 0,
             'wall cap')
        self.caps, self.sink, self.clock = dict(caps), sink, clock
        self.start, self.wall_seconds = clock(), wall_seconds
        self.counts = {k: dict(attempted=0, returned=0, failed=0) for k in AXES}
        self.pending, self.serial = {}, 0

    def call(self, axis, units, fn):
        need(axis in AXES and type(units) is int and units > 0, 'resource axis/units')
        need(self.clock()-self.start < self.wall_seconds, 'wall cap')
        count = self.counts[axis]
        need(count['attempted']+units <= self.caps[axis], 'resource cap: '+axis)
        ident = self.serial
        self.serial += 1
        self.sink(dict(kind='attempt', id=ident, axis=axis, units=units))
        count['attempted'] += units
        self.pending[ident] = (axis, units)
        try:
            result = fn()
        except BaseException as exc:
            count['failed'] += units
            self.sink(dict(kind='failure', id=ident, axis=axis, units=units,
                           error_type=type(exc).__name__, message=str(exc)))
            del self.pending[ident]
            raise
        self.sink(dict(kind='return', id=ident, axis=axis, units=units))
        count['returned'] += units
        del self.pending[ident]
        return result


def outcome(goal, state, *, native_success, terminated, truncated, censored):
    """Separate final native7 deviations; no success inferred from completion."""
    g, s = np.asarray(goal, np.float64), np.asarray(state, np.float64)
    need(g.shape == s.shape == (7,) and np.isfinite(g).all() and np.isfinite(s).all(),
         'finite native state')
    need(type(native_success) is type(terminated) is type(truncated) is type(censored) is bool, 'native flags')
    d = s-g
    angular = float(abs(np.arctan2(np.sin(d[4]), np.cos(d[4]))))
    return dict(native_success=native_success,
                object_position_error=float(np.linalg.norm(d[2:4])),
                angle_error_radians=angular, pusher_position_error=float(np.linalg.norm(d[:2])),
                terminated=terminated, truncated=truncated, censored=censored)
