"""RAM-only checks for MAIN's independent arithmetic, not research evidence."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import audit_moving_capability_MAIN_20261027_v1 as m


def fixture():
    amplitude = float(np.float32(.1))
    directions = [(0, 0), (amplitude, 0), (-amplitude, 0), (0, amplitude), (0, -amplitude)]
    start = np.array([10., 20., 30., 40., 0., 0., 0.])
    rows = []
    for name, (x, y) in zip(('zero', '+x', '-x', '+y', '-y'), directions):
        state = start + np.array([x, y, 2*x, 3*y, x+y, 0, 0])
        snapshot = dict(state=state.tolist(), state_float64_bits=state.astype('<f8').tobytes().hex(),
                        bodies=dict(block=dict(unwrapped_angle=float(x+y))))
        observations = [dict(phase='probe', step=i, state=state.tolist(), snapshot=deepcopy(snapshot)) for i in range(1, 6)]
        rows.append(dict(probe=name, status='probe_completed', start_verified=True, start_state=start.tolist(),
                         root_slot=0, condition='off', arm='GN', trajectory=[dict(phase='warmup')]*17+observations,
                         responses=[dict(state=state.tolist(), delta_native7=(state-start).tolist()) for _ in range(5)]))
    goal = start + np.array([1, 0, 0, 1, 0, 0, 0])
    analyzed = dict(spaces={})
    for space, indices, matrix in [('object_xy', slice(2, 4), [[2, 0], [0, 3]]),
                                    ('agent_xy', slice(0, 2), [[1, 0], [0, 1]]),
                                    ('joint_xy', slice(0, 4), [[1, 0], [0, 1], [2, 0], [0, 3]]),
                                    ('object_angle_unwrapped', None, [[1, 1]])]:
        mat = np.asarray(matrix, dtype=float)
        direction = None if indices is None else (goal-start)[indices]
        if direction is not None: direction /= np.linalg.norm(direction)
        finite = []
        for d in directions:
            delta = mat @ d
            finite.append(dict(delta=delta.tolist(), task_projection=None if direction is None else float(direction @ delta)))
        analyzed['spaces'][space] = dict(matrix=matrix, spectrum=dict(singular_values=np.linalg.svd(mat, compute_uv=False).tolist()),
                                         task_direction=None if direction is None else direction.tolist(), finite_responses=finite)
    return rows, analyzed, dict(goal=goal.tolist(), root_state=start.tolist())


def test_independent_signed_units():
    result = m.compare_group(*fixture())
    assert result['matrices_checked'] == 4


@pytest.mark.parametrize('fault', ['matrix', 'start', 'response', 'order'])
def test_rejects_wrong_saved_evidence(fault):
    rows, analyzed, upstream = fixture()
    if fault == 'matrix': analyzed['spaces']['object_xy']['matrix'][0][0] = 3
    if fault == 'start': rows[1]['start_state'][0] += 1
    if fault == 'response': rows[1]['responses'][0]['delta_native7'][0] += 1
    if fault == 'order': rows.reverse()
    with pytest.raises((ValueError, AssertionError)):
        m.compare_group(rows, analyzed, upstream)
