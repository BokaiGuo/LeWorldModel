"""Bind one-action commands to copied RGB-only feedback and a monotone clock."""
import numpy as np


def rgb_copy(value):
    if not isinstance(value, np.ndarray) or value.shape != (224, 224, 3) or value.dtype != np.uint8:
        raise RuntimeError("invalid RGB observation")
    return value.copy()


class RGBClock:
    def __init__(self, bridge):
        if bridge.step_count != 0:
            raise RuntimeError("bridge did not start at action zero")
        self._bridge = bridge
        self._frames = [rgb_copy(bridge.initial_rgb)]
        self.clock = 0
        self._failed = False

    def advance(self, action):
        if self._failed:
            raise RuntimeError("RGB feedback stream failed")
        packet = np.asarray(action, dtype=np.float64)
        if packet.shape != (2,) or not np.isfinite(packet).all() or np.any(np.abs(packet) > 1):
            raise ValueError("invalid action")
        try:
            image = rgb_copy(self._bridge.step(packet.copy()[None]))
            if self._bridge.step_count != self.clock + 1:
                raise RuntimeError("action clock mismatch")
        except BaseException:
            self._failed = True
            raise
        self._frames.append(image)
        self.clock += 1
        return image.copy()

    def pair(self):
        if self._failed or len(self._frames) < 2:
            raise RuntimeError("no valid consecutive RGB feedback")
        return self._frames[-2].copy(), self._frames[-1].copy()
