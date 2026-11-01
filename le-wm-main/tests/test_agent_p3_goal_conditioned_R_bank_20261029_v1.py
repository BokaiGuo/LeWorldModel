"""RAM numerical/temporary-file fixtures only, no producer, neural or physics imports."""
from copy import deepcopy
import itertools

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_goal_conditioned_R_bank_20261029_v1 as m


def fixture(h=1, kind='mixed'):
    task = dict(root_context=dict(layout='states_with_between_state_actions_v1', context_length=3,
                latent_dim=192, action_dim=10, latent_history=np.zeros((3, 192)).tolist(),
                action_history=np.zeros((2, 10)).tolist()), goal=np.zeros(192).tolist(),
                projection=np.eye(192).tolist(), action_mean=[0., 0.], action_std=[1., 1.])
    task['goal'][2:4] = [1., 1.]
    j = np.zeros((192, 10*h), np.float32)
    j[:4, :4] = np.diag([4., 3., 2., 1.])
    if kind == 'zeroJ': j[:] = 0
    elif kind == 'zero_residual': task['goal'] = [0.]*192
    elif kind == 'collinear':
        j[:] = 0
        j[2, 2] = 1
        task['goal'][3] = 0.
    b = {k: m.digest(['fixture', k]) for k in m.BINDINGS}
    task['binding'] = b
    signal = dict(horizon=h, nominal=[0.]*192, jacobian=j.tolist(), producer_sha256='a'*64)
    provenance = dict(kind='RAM_FIXTURE', artifact_path='/RAM/not_a_real_native_artifact.json', artifact_sha256='b'*64,
                      call_id=0, origin_index=1, nominal_index=2, jacobian_index=3, paid_source_resources=None)
    rebind(task, signal, provenance)
    return task, signal, provenance


def rebind(task, signal, provenance):
    task['binding'].update(context_sha256=m.digest(task['root_context']), goal_sha256=m.digest(task['goal']),
        projection_sha256=m.digest(task['projection']), normalizer_sha256=m.digest([task['action_mean'], task['action_std']]),
        action_domain_sha256=m.digest(m.domain()))
    signal.update(binding_sha256=m.digest(task['binding']), parent_context_sha256=m.digest(task['root_context']))
    provenance.update(task_sha256=m.digest(task), signal_sha256=m.digest(signal))


def run(args, **kwargs):
    return m.build_bank(*args, clock=itertools.count().__next__, **kwargs)


@pytest.mark.parametrize('h', [1, 3])
def test_full_basis_goal_relevance_formula_order_rawbits_and_diagnostics(h):
    task, signal, source = fixture(h)
    before = deepcopy((task, signal, source))
    out = run((task, signal, source))
    assert (task, signal, source) == before
    assert out['status'] == 'complete'
    v = out['intermediates']
    assert np.asarray(v['U']).shape == (192, 10*h)
    assert np.asarray(v['Vt']).shape == (10*h, 10*h)
    assert len(v['singular_values']) == 10*h  # zero singular values retained
    expected_p = np.zeros(10*h)
    expected_p[2:4] = [2/(4+m.LAMBDA), 1/(1+m.LAMBDA)]
    np.testing.assert_allclose(v['p'], expected_p, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(v['b'], np.r_[0., 0., 2., 1., np.zeros(10*h-4)], rtol=0, atol=0)
    assert out['directions']['d0'][:2] == [0., 0.]  # no goal mass in top-two singular modes
    expected_t = expected_p-np.asarray(v['b'])*(np.dot(v['b'], expected_p)/np.dot(v['b'], v['b']))
    np.testing.assert_allclose(v['t'], expected_t, rtol=1e-14, atol=1e-15)
    order = [(radius, d, sign) for radius in m.RADII for d in ('d0', 'd1') for sign in (-1, 1)]
    assert len(out['candidates']) == 8
    for row, (radius, d, sign) in zip(out['candidates'], order):
        assert (row['radius'], row['direction'], row['sign']) == (radius, d, sign)
        params = sign*radius*np.asarray(out['directions'][d], np.float64)
        actions = np.tanh(params).astype(np.float32)
        assert row['params'] == params.tolist()
        assert row['raw_float32_bits'] == actions.astype('<f4').tobytes().hex()
        assert np.array_equal(np.asarray(row['actions'], np.float32).reshape(-1), actions)
        assert row['b_dot_param'] == float(np.asarray(v['b'])@params)
        assert row['b_dot_cast32_action'] == float(np.asarray(v['b'])@actions.astype(np.float64))
    tangent = out['candidates'][2]
    assert abs(tangent['b_dot_param']) < 1e-14
    assert abs(tangent['b_dot_cast32_action']) > 1e-3  # tanh is NOT a first-order tangent guarantee
    assert out['ledger']['attempted'] == out['ledger']['returned'] == m.COST
    assert m.audit_ledger(out['ledger'])
    assert out['ledger']['entries'][0]['ended_ns'] > out['ledger']['entries'][2]['ended_ns']
    assert out['new_resource_vector']['predict_API'] == out['new_resource_vector']['jacobian_API'] == 0
    assert out['inherited_source_resources'] is None
    assert not out['zero']['evaluation_paid'] and len(out['zero']['actions']) == 5*h
    assert not any(out[k] for k in m.FLAGS)
    assert not out['source_artifact_authenticated_here'] and not out['source_artifact_bytes_read_here']


@pytest.mark.parametrize('kind,generated', [('zeroJ', 0), ('zero_residual', 0), ('collinear', 4)])
def test_degenerate_slots_are_null_no_zero_replacement_but_fixed_math_cost(kind, generated):
    out = run(fixture(1, kind))
    assert out['status'] == 'complete_with_blocked_directions'
    assert sum(r['status'] == 'generated_unexecuted' for r in out['candidates']) == generated
    for row in out['candidates']:
        if row['status'] == 'blocked_degenerate_direction':
            assert row['params'] is row['actions'] is row['raw_float32_bits'] is None
            assert row['b_dot_param'] is row['b_dot_cast32_action'] is None
    assert out['ledger']['attempted'] == out['ledger']['returned'] == m.COST
    assert m.audit_ledger(out['ledger'])


def test_exact_64eps_inclusive_boundary_without_tuning():
    b = np.array([1., 0.])
    at = np.array([1., 64*m.EPS])
    above = np.array([1., np.nextafter(64*m.EPS, np.inf)])
    directions, reasons, diagnostic = m.direction_data(at, b)
    assert diagnostic['tnorm'] == diagnostic['tangent_cutoff']
    assert directions['d1'] is None and reasons['d1'] == 'tangent_at_or_below_64eps_pnorm'
    directions, _, _ = m.direction_data(above, b)
    assert directions['d1'] == [0., 1.]
    for p, b, reason in ((np.zeros(2), np.ones(2), 'p_zero'), (np.ones(2), np.zeros(2), 'b_zero')):
        directions, reasons, _ = m.direction_data(p, b)
        assert directions == dict(d0=None, d1=None) and reasons == dict(d0=reason, d1=reason)


def test_no_second_normalizer_rescaling_and_no_source_authentication():
    first = fixture()
    second = deepcopy(first)
    second[0]['action_mean'] = [.25, -.25]
    second[0]['action_std'] = [2., 4.]
    second[2]['kind'] = 'SAVED_REFERENCE'
    second[2]['paid_source_resources'] = dict.fromkeys(m.AXES, 0)
    second[2]['paid_source_resources']['jacobian_API'] = 1
    second[2]['paid_source_resources']['VJP_directions'] = 192
    rebind(*second)
    one, two = run(first), run(second)
    assert one['candidates'] == two['candidates']
    assert two['inherited_source_resources']['VJP_directions'] == 192
    assert two['new_resource_vector']['VJP_directions'] == 0
    assert not two['source_artifact_authenticated_here']


def test_single_actual_SVD_and_solve_even_for_zero_residual(monkeypatch):
    calls = []
    svd, solve = np.linalg.svd, np.linalg.solve
    def wrapped_svd(a, **kw):
        calls.append(('svd', a.shape, kw))
        return svd(a, **kw)
    def wrapped_solve(a, b):
        calls.append(('solve', a.shape, b.shape))
        return solve(a, b)
    monkeypatch.setattr(m.np.linalg, 'svd', wrapped_svd)
    monkeypatch.setattr(m.np.linalg, 'solve', wrapped_solve)
    out = run(fixture(3, 'zero_residual'))
    assert calls == [('svd', (192, 30), dict(full_matrices=False)), ('solve', (30, 30), (30,))]
    assert out['ledger']['returned'] == m.COST


@pytest.mark.parametrize('resource', tuple(m.COST))
def test_atomic_budget_prevents_any_dispatch(resource, monkeypatch):
    caps = dict(m.COST, **{resource: 0})
    def forbidden(*args, **kwargs):
        pytest.fail('atomic reserve must precede dispatch')
    monkeypatch.setattr(m.np.linalg, 'svd', forbidden)
    monkeypatch.setattr(m.np.linalg, 'solve', forbidden)
    out = run(fixture(), caps=caps)
    assert out['status'] == 'blocked_before_bank_budget'
    assert out['ledger']['events'] == []
    assert out['ledger']['attempted'] == out['ledger']['returned'] == dict.fromkeys(m.COST, 0)
    assert m.audit_ledger(out['ledger'])


@pytest.mark.parametrize('operation', ['svd', 'solve'])
def test_nested_failure_actual_attempts_no_refund(operation, monkeypatch):
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError('injected RAM failure')
    monkeypatch.setattr(m.np.linalg, operation, fail)
    out = run(fixture())
    assert out['status'] == 'partial_failure' and out['error']['type'] == 'LinAlgError'
    assert out['ledger']['failed']
    assert out['ledger']['attempted']['program_compile_calls'] == 1
    assert out['ledger']['returned']['program_compile_calls'] == 0
    assert out['ledger']['attempted']['array_SVD_calls'] == 1
    assert out['ledger']['attempted']['solve_calls'] == int(operation == 'solve')
    assert out['ledger']['returned']['array_SVD_calls'] == int(operation == 'solve')
    assert [e['kind'] for e in out['ledger']['events']][-2:] == ['failure', 'failure']
    assert m.audit_ledger(out['ledger'])  # child failure followed by parent failed is legal


@pytest.mark.parametrize('kind', ['Q', 'boolgoal', 'nanJ', 'f64J', 'context', 'normalizer', 'binding',
                                'signalhash', 'rawhbool', 'future_label', 'unknown_paid', 'path', 'paidbool'])
def test_input_guards(kind):
    task, signal, source = fixture()
    if kind == 'Q': task['projection'][0][0] = 2.
    elif kind == 'boolgoal': task['goal'][0] = False
    elif kind == 'nanJ': signal['jacobian'][0][0] = float('nan')
    elif kind == 'f64J': signal['jacobian'][0][0] = .1
    elif kind == 'context': task['root_context']['latent_history'][0][0] = 1.
    elif kind == 'normalizer': task['action_std'][0] = 0.
    elif kind == 'binding': signal['binding_sha256'] = '0'*64
    elif kind == 'signalhash': source['signal_sha256'] = '0'*64
    elif kind == 'rawhbool': signal['horizon'] = True
    elif kind == 'future_label': signal['native_success'] = True
    elif kind == 'unknown_paid': source['paid_source_resources'] = {}
    elif kind == 'path': source['artifact_path'] = 'guessed_relative.json'
    elif kind == 'paidbool':
        source['paid_source_resources'] = dict.fromkeys(m.AXES, 0)
        source['paid_source_resources']['grad_API'] = False
    # Some malformed inputs cannot be serialized to a valid source digest at all; still reject before math.
    if kind in ('Q', 'boolgoal', 'normalizer', 'rawhbool', 'future_label', 'f64J'):
        rebind(task, signal, source)
    with pytest.raises(ValueError):
        run((task, signal, source))


@pytest.mark.parametrize('kind', ['unknown', 'event_reorder', 'counter', 'time', 'boolparent'])
def test_entire_ledger_mutation_rejected(kind):
    out = run(fixture())
    ledger = deepcopy(out['ledger'])
    if kind == 'unknown': ledger['events'][1]['kind'] = 'skip'
    elif kind == 'event_reorder': ledger['events'][1], ledger['events'][2] = ledger['events'][2], ledger['events'][1]
    elif kind == 'counter': ledger['attempted']['solve_calls'] = 0
    elif kind == 'time': ledger['events'][-1]['entry']['ended_ns'] = -1
    elif kind == 'boolparent': ledger['events'][1]['entry']['parent'] = False
    with pytest.raises(ValueError):
        m.audit_ledger(ledger)


def disk_input(tmp_path):
    task, signal, provenance = fixture()
    path = tmp_path/'input.json'
    path.write_bytes(m.canonical(dict(schema=m.NAME+'.input', task=task, signal=signal, provenance=provenance)))
    return path


def test_real_CLI_writer_and_numeric_verify(tmp_path, capsys):
    path = disk_input(tmp_path)
    output = tmp_path/'bank.json'
    m.main(['contract'])
    m.main(['build', '--input', str(path), '--input-sha', m.sha(path), '--output', str(output)])
    out = m.read_pin(output, m.sha(output))
    assert out['status'] == 'complete' and out['provenance']['kind'] == 'RAM_FIXTURE'
    m.main(['verify', '--report', str(output), '--report-sha', m.sha(output)])
    assert 'verified_numeric_only' in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        m.write_new(output, out)
    out['candidates'][2]['b_dot_cast32_action'] = 0.
    changed = tmp_path/'coherent_rehashed_corruption.json'
    m.write_new(changed, out)
    with pytest.raises(ValueError, match='payload'):
        m.verify_file(changed, m.sha(changed))


def test_disk_schema_hash_caps_and_json_rejection(tmp_path):
    path = disk_input(tmp_path)
    with pytest.raises(ValueError, match='SHA'):
        m.read_pin(path, '0'*64)
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
        with pytest.raises(ValueError):
            m.decode(raw)
    link = tmp_path/'symlink'
    link.symlink_to(path)
    with pytest.raises(ValueError):
        m.read_pin(link, m.sha(path))


def test_fsync_open_file_before_dir(tmp_path, monkeypatch):
    import os
    import stat
    calls, real = [], os.fsync
    def sync(fd):
        calls.append('dir' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
        real(fd)
    monkeypatch.setattr(m.os, 'fsync', sync)
    m.write_new(tmp_path/'fixture.json', {'fixture': True})
    assert calls == ['file', 'dir']
