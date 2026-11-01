"""Fail-closed two-packet dispatch primitive for live replanning."""
import hashlib

import numpy as np


def packet_bytes(value):
    raw = np.asarray(value, dtype="<f4")
    if raw.shape != (5, 2) or not np.isfinite(raw).all() or (np.abs(raw) > 1).any():
        raise ValueError("expected finite bounded five-action packet")
    return raw.copy()


class LivePacketDispatch:
    """Second packet cannot be read before a single live replan succeeds."""

    def __init__(self, first):
        self.first = packet_bytes(first)
        self._second = None
        self._issued = 0

    def next_action(self):
        if self._issued >= 10:
            raise ValueError("packet exhausted")
        if self._issued < 5:
            action = self.first[self._issued]
        else:
            if self._second is None:
                raise ValueError("live second packet unavailable")
            action = self._second[self._issued - 5]
        self._issued += 1
        return action.copy()

    def bind_second(self, second):
        if self._issued != 5 or self._second is not None:
            raise ValueError("second packet can only bind once after five actions")
        self._second = packet_bytes(second)
        return hashlib.sha256(self._second.tobytes()).hexdigest()

    @property
    def issued(self):
        return self._issued
