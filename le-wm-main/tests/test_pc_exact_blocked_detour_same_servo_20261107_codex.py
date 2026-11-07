import numpy as np

from experiments.pc_exact_blocked_detour_20261107_codex import generate
from experiments.pc_exact_blocked_detour_same_servo_20261107_codex import no_detour


def test_actions_identical_through_first_trigger():
    start, goal, offsets = [-.5, -.45, 0.], [.5, -.35], [-1., -.5, 0., .5, 1.]
    control, _ = no_detour(start, goal, offsets)
    active, _, triggers, _ = generate(start, goal, offsets)
    for i in range(len(offsets)):
        first = np.flatnonzero(triggers[i])
        stop = int(first[0] + 1) if len(first) else 15
        np.testing.assert_allclose(control[i, :stop], active[i, :stop], atol=1e-12, rtol=0)
