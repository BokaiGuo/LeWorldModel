"""Pure saved data and memory fixtures; no neural computation or simulator."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import p1_same_prefix_physical_main_20261027_v1 as m


def test_all_coordinates_signs_radii_and_zero():
    probes = m.suffixes()
    assert len(probes) == 41 and probes[0]['logits'] == [0.]*10
    assert {(p['coordinate'], p['radius'], p['sign']) for p in probes[1:]} == {
        (i, r, s) for i in range(10) for r in (.25, .5) for s in (-1, 1)}
    for p in probes[1:]:
        x = np.asarray(p['logits'])
        assert np.count_nonzero(x) == 1 and x[p['coordinate']] == p['radius']*p['sign']
        assert np.array_equal(x, x.astype(np.float32))


def fixture():
    plans, rows, goals = [], [], {}
    for root in ('root0', 'root1'):
        goals[root] = [10., 10., 10., 10., 1., 0., 0.]
        for role in m.ROLES:
            for probe in m.suffixes():
                i = probe['index']; end = [float(i)/100, 0., 0., 0., 0., 0., 0.]
                identity = dict(root_sha256=root, arm=role+str(i), selection_sha256=root+role+str(i))
                plans.append(dict(**identity, role=role, probe=probe))
                rows.append(dict(**identity, status='completed', full_horizon=True, final_state=end,
                    checkpoints=[dict(step=0), dict(step=5, state=[0.]*7, rgb_sha256='same')],
                    body_dynamics={'5':{'mock_only':True, 'block_velocity':[0., 0.]}},
                    official_evaluation=dict(success=False, state_distance=20.-i/100),
                    pose_diagnostics=dict(independent=dict(wrapped_angle_error=1.))))
    return plans, rows, goals


def test_full_denominator_and_centered_direction():
    p, r, g = fixture(); result = m.analyze(p, r, g)
    assert len(result['groups']) == 6 and not result['P1_admission']
    for group in result['groups']:
        assert group['planned'] == group['completed'] == 41
        assert len(group['centered_responses']) == 41 and group['native_successes'] == 0
        assert group['centered_responses'][0]['endpoint_delta_state7'] == [0.]*7
        assert group['centered_responses'][1]['common_position_direction_response'] == pytest.approx(.005)


def test_failure_retained_not_replaced():
    p, r, g = fixture(); r[7]['status'] = 'environment_terminated'; r[7]['full_horizon'] = False
    group = m.analyze(p, r, g)['groups'][0]
    assert group['planned'] == 41 and group['completed'] == 40
    assert group['unavailable_probe_indices'] == [7]


@pytest.mark.parametrize('defect', ['root', 'selection', 'order', 'prefix_state', 'prefix_rgb', 'block_velocity'])
def test_identity_census_prefix_tamper_rejected(defect):
    p, r, g = fixture()
    if defect == 'root': r[2]['root_sha256'] = 'other'
    elif defect == 'selection': r[2]['selection_sha256'] = 'other'
    elif defect == 'order': p[2]['probe']['index'] = 1
    elif defect == 'prefix_state': r[2]['checkpoints'][1]['state'][0] = .1
    elif defect == 'prefix_rgb': r[2]['checkpoints'][1]['rgb_sha256'] = 'other'
    else: r[2]['body_dynamics']['5']['block_velocity'][0] = .1
    with pytest.raises(ValueError): m.analyze(p, r, g)


def test_angle_difference_wrap_and_units_not_silently_normalized():
    a = [1., 2., 3., 4., 2*np.pi-.1, 5., 6.]
    d = m.geometric_delta(a, [0.]*7)
    assert d[4] == pytest.approx(-.1)
    assert d[[0, 1, 2, 3, 5, 6]].tolist() == [1., 2., 3., 4., 5., 6.]


def test_body_dynamic_snapshot_keeps_unwrapped_angle_and_block_velocity():
    b = SimpleNamespace(position=[1., 2.], velocity=[3., 4.], angle=7., angular_velocity=8.,
                        force=[9., 10.], torque=11., is_sleeping=False)
    snapshot = m.body_dynamics(SimpleNamespace(agent=b, block=b))
    assert snapshot['block']['unwrapped_angle'] == 7.
    assert snapshot['block']['velocity'] == [3., 4.]
    assert snapshot['agent']['angular_velocity'] == 8.


def test_no_authority_no_verification_or_environment(monkeypatch):
    def forbidden(*a): raise AssertionError('must not enter')
    monkeypatch.setattr(m, 'verify', forbidden)
    with pytest.raises(ValueError, match='authorization'):
        m.run('missing', 'bad', 'missing', forbidden)


def test_storage_cap_and_floor(monkeypatch):
    s = object.__new__(m.Store); s.used = m.CAP-1; s.path = Path('/memory_only')
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda p:SimpleNamespace(free=m.FLOOR+m.CAP))
    with pytest.raises(ValueError, match='cap'): s.check(2)
    s.used = 0
    monkeypatch.setattr(m.shutil, 'disk_usage', lambda p:SimpleNamespace(free=m.FLOOR))
    with pytest.raises(ValueError, match='floor'): s.check(1)


def test_actual_roles_and_full_plan_saved_only():
    data = m.collect()
    assert len(data['plans']) == 246 and len(data['roles']) == 6
    assert [r['role'] for r in data['roles']] == list(m.ROLES)*2
    for k in range(0, 246, 41):
        group = data['plans'][k:k+41]
        assert all(p['commands'][:5] == group[0]['commands'][:5] for p in group)
        assert np.asarray(group[0]['commands'])[5:].tolist() == np.zeros((5, 2)).tolist()
        for p in group:
            assert np.array_equal(np.asarray(p['commands'][5:]), np.tanh(np.asarray(p['probe']['logits'], np.float32)).reshape(5, 2))
