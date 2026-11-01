"""Opt-in fail-closed guard for the DEFAULT RoyalBlue PushT RGB contract.

Not a general vision detector, collision predictor, recovery controller, or P1
admission. It cannot prevent the first out-of-view transition. No state/goal
labels enter the guard. A new feedback schedule must be matched in baselines.
"""
import numpy as np


class ObservationUnavailable(RuntimeError):
    pass


class RgbObservabilityGuard:
    def __init__(self, *, root_id, agent_rgb):
        if not isinstance(root_id, str) or not root_id:
            raise ValueError('bound root identity required')
        if tuple(agent_rgb) != (65, 105, 225):
            raise ValueError('only explicitly configured canonical RoyalBlue contract supported')
        self.root_id = root_id
        self._next_step = 0
        self._ready = False
        self._failed = False
        self._records = []

    def observe(self, rgb, *, primitive_step):
        if type(primitive_step) is not int or primitive_step < 0 or self._failed or primitive_step != self._next_step or self._ready:
            raise ObservationUnavailable('missing, reused, out-of-order or invalidated observation')
        rgb = np.asarray(rgb)
        if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
            self._failed = True
            raise ValueError('official uint8 RGB required')
        count = int(np.count_nonzero(np.all(rgb == [65, 105, 225], axis=-1)))
        self._records.append({'primitive_step': primitive_step, 'canonical_agent_pixels': count})
        if count == 0:
            self._failed = True
            raise ObservationUnavailable('canonical agent appearance absent; no blind continuation')
        self._ready = True

    def dispatch(self, command):
        if self._failed or not self._ready:
            raise ObservationUnavailable('fresh visible RGB required before each primitive command')
        value = np.asarray(command)
        if value.dtype.kind not in 'fi' or value.shape != (2,) or not np.isfinite(value).all() or np.any(np.abs(value) > 1):
            self._failed = True
            raise ValueError('legal raw two-coordinate command required')
        self._ready = False
        self._next_step += 1
        return value.copy()

    def receipt(self):
        return {'root_id': self.root_id, 'observations': [dict(r) for r in self._records],
                'primitive_commands_authorized': self._next_step, 'abstained': self._failed,
                'scope': 'canonical appearance presence only; not full observability or prevention guarantee',
                'P1_admission': False}
