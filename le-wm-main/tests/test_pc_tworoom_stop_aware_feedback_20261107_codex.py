import numpy as np

from experiments.pc_tworoom_stop_aware_feedback_20261107_codex import execute_until_done


class FakeEnv:
    def __init__(self, stop_at, truncation=False):
        self.count = 0
        self.stop_at = stop_at
        self.truncation = truncation

    def step(self, action):
        self.count += 1
        stop = self.count == self.stop_at
        return None, 0, stop and not self.truncation, stop and self.truncation, {"distance_to_target": float(self.count)}


def test_stops_at_termination_and_truncation():
    for truncation in (False, True):
        env = FakeEnv(3, truncation)
        trace, dispatched = [], []
        assert execute_until_done(env, np.zeros((10, 2)), trace, dispatched)
        assert env.count == len(trace) == len(dispatched) == 3
        assert trace[-1][2 if truncation else 1]


def test_exhausts_actions_if_not_done():
    env = FakeEnv(20)
    trace, dispatched = [], []
    assert not execute_until_done(env, np.zeros((10, 2)), trace, dispatched)
    assert env.count == len(trace) == len(dispatched) == 10
