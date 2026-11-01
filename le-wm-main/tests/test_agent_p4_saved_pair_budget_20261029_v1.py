import copy
import hashlib
import math
import sys

import pytest
from experiments.spectral_reachability import agent_p4_saved_pair_budget_20261029_v1 as m


@pytest.fixture(scope='module')
def inputs():
    return m.load_inputs()


def test_saved_fixed_cohort_and_missing(inputs):
    a = m.analyze(inputs)
    assert a['denominator'] == dict(episodes=8, pairs=4, independent_known_roots=2, compile_slots=24)
    assert [sum(q['R_minus_I'] is not None for q in p['checkpoints']) for p in a['pairs']] == [5,3,5,5]
    p = a['pairs'][1]
    assert p['arms'][1]['control_returned'] == 11
    assert p['checkpoints'][3]['R'] is None and p['checkpoints'][4]['R_minus_I'] is None
    assert a['actual_counts']['predict']['attempted'] == 115
    assert a['unknowns']['FLOPs'] is None and not a['P1_admission']


def test_math_and_wrapped_angle():
    x = m.metrics(dict(state=[3,4,0,0,2*math.pi+0.1,3,4], goal=[0]*7))
    assert x['agent_xy'] == 5 and x['joint_xy'] == 5 and x['velocity_xy'] == 5
    assert x['wrapped_angle_abs'] == pytest.approx(0.1)
    assert x['state_minus_goal'][4] > 6


@pytest.mark.parametrize('mutate', [
    lambda x: x['report']['rows'].pop(),
    lambda x: x['report']['rows'][0].update(mode='R'),
    lambda x: x['report']['rows'][0]['executed_raw'][0].__setitem__(0, 999),
    lambda x: x['report']['rows'][0]['checkpoints'].pop(),
    lambda x: x['report']['rows'][0]['checkpoints'][1]['goal'].__setitem__(0, 999),
    lambda x: x['report']['rows'][0]['checkpoints'][1]['state'].__setitem__(0, float('nan')),
    lambda x: x['strict']['compiled'].pop(),
    lambda x: x['strict']['compiled'][0].update(arrays_verified=False),
    lambda x: x['strict']['counts']['predict'].update(attempted=0),
    lambda x: x['strict'].update(report_sha256='0'*64),
    lambda x: x['native'].update(report_sha256='0'*64),
    lambda x: x['rolling'].update(reached_boundaries=24),
    lambda x: x['rolling']['inputs'].__setitem__(str(m.ROOT / m.INPUTS['prepare'][0]), '0'*64),
])
def test_refuses_mutated_evidence(inputs, mutate):
    value = copy.deepcopy(inputs); mutate(value)
    with pytest.raises((ValueError, KeyError)):
        m.analyze(value)


def test_json_duplicate_and_nonfinite_refused():
    for s in ('{"x":1,"x":2}', '{"x":NaN}'):
        with pytest.raises(ValueError): m.decode(s)


def test_deterministic_and_no_selection(inputs):
    a = m.analyze(inputs)
    assert m.canonical(a) == m.canonical(m.analyze(inputs))
    assert [p['pair'] for p in a['pairs']] == list(range(4))
    assert all(len(p['checkpoints']) == 5 for p in a['pairs'])


def test_verify_complete_saved_analysis(inputs, tmp_path, monkeypatch, capsys):
    raw = m.canonical(m.analyze(inputs)).encode()
    path = tmp_path / 'analysis.json'; path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(sys, 'argv', ['p4-analysis', 'verify', '--analysis', str(path), '--sha256', sha])
    m.main()
    assert m.decode(capsys.readouterr().out)['status'] == 'exact_analysis_verified'
    path.write_bytes(raw.replace(b'"episodes":8', b'"episodes":9'))
    with pytest.raises(ValueError): m.main()
