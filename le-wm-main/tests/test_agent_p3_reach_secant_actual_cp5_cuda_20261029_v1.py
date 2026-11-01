"""Synthetic real-shape schemas/NumPy callbacks only; NO production upstream consume."""
from copy import deepcopy
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_reach_secant_actual_cp5_cuda_20261029_v1 as m

f = runpy.run_path(str(m.ROOT/'tests/test_agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1.py'))
RAM, Memory = f['RAM'], f['Memory']


def small_source(h=1, *, mode='normal'):
    n = 10*h; d0 = np.eye(n)[0]; d1 = .6*np.eye(n)[0]+.8*np.eye(n)[1]
    if mode == 'gram': d1 = d0.copy()
    directions = dict(d0=d0.tolist(), d1=d1.tolist()); rows = []
    for i in range(4):
        direction, sign = 'd'+str(i//2), (-1, 1)[i % 2]
        d = np.asarray(directions[direction]); param = sign*m.kernel.R0*d
        si, sr = (1., .5)[i//2], (-1., -2.)[i//2]
        if mode == 'zero': si = 0.
        if mode == 'opposed': sr = si
        rows.append(dict(index=i, direction=direction, sign=sign, radius=m.kernel.R0,
            status='complete', params=param.tolist(), actions=np.tanh(param).astype(np.float32).reshape(5*h, 2).tolist(),
            immediate=2+sign*m.kernel.R0*si, reach=2e6+sign*m.kernel.R0*sr))
    if mode == 'null':
        directions['d1'] = None
        for row in rows[2:]: row.update(status='blocked_degenerate_direction', params=None, actions=None, immediate=None, reach=None)
    return directions, rows


def math_run(h=1, mode='normal', dispatch=None):
    directions, small = small_source(h, mode=mode)
    ledger = m.core.Ledger(m.caps(h))
    ledger.scope = dict(phase='RAM_secant_math')
    return m.kernel.build(h, directions, small, ledger.call if dispatch is None else dispatch), ledger


def source_fixture():
    """Literal own-schema fixture, NOT an authenticated real model run.

    The production loader must separately rebuild the entire frozen consume.
    Scores here are fixed RAM literals; no producer execution is performed.
    """
    material = f['material'](); p = dict(material, schema=m.source.NAME+'.prepare', roster=m.roster(),
        expected_model_hash='RAM_model', **m.source.FLAGS)
    p['sources'] = {'/RAM/class.py': 'a'*64}
    classes = [dict(path='/RAM/class.py', sha256='a'*64)]
    r = dict(schema=m.source.NAME+'.report', prepared=m.AUTH['prepared']['path'], prepare_sha256=m.AUTH['prepared']['sha256'],
        status='four_slots_settled_not_admitted', errors=[], rows=[], sources_verified=True,
        source_manifest_sha256=m.digest(p['sources']), model_before=dict(original_state_dict='RAM_model'),
        model_after=dict(original_state_dict='RAM_model'), classes_before=deepcopy(classes), classes_after=deepcopy(classes), **m.source.FLAGS)
    receipt = dict(schema=m.source.NAME+'.consume', status='actual_cp5_goal_R_model_deliveries_verified_NOT_physical',
        prepared=m.AUTH['prepared']['path'], prepare_sha256=m.AUTH['prepared']['sha256'], report_sha256=m.AUTH['report']['sha256'],
        denominator=4, candidate_denominator=32, controls=4, arms=[], **m.source.FLAGS)
    for identity in m.roster():
        root, h = identity['root'], identity['horizon']; task = p['tasks'][root]; pro = p['provenance'][root]
        local = m.parent.parent_task(task, task['root_context'])
        native = dict(binding_sha256=m.digest(local['binding']), parent_context_sha256=m.digest(local['root_context']),
            horizon=h, nominal=[0.]*192, jacobian=np.zeros((192, 10*h), np.float32).tolist(), producer_sha256=m.sha(m.signal.__file__))
        mapping = dict(schema=m.parent.NAME+'.signal_map', root_task_sha256=m.digest(task), root_binding=deepcopy(task['binding']),
            parent_task=local, native_request=m.core.request(local, dict(context=local['root_context']), operation='compile_finite_horizon_J', horizon=h),
            outer_request=m.core.request(task, dict(context=task['root_context']), operation='compile_finite_horizon_J', horizon=h),
            native_response=native, outward_response=dict(native, binding_sha256=m.digest(task['binding']), producer_sha256=m.sha(m.parent.__file__)))
        mapping['mapping_sha256'] = m.digest(mapping)
        directions, small = small_source(h); candidates = []; items = []
        for i in range(8):
            item = deepcopy(small[i % 4]); item['index'] = i
            if i >= 4:
                item['radius'] = m.source.numeric.RADII[1]
                param = item['sign']*item['radius']*np.asarray(directions[item['direction']])
                item.update(params=param.tolist(), actions=np.tanh(param).astype(np.float32).reshape(5*h, 2).tolist())
            candidates.append({k: deepcopy(item[k]) for k in ('index', 'direction', 'sign', 'radius', 'params', 'actions')})
            items.append(dict(index=i, status='complete', immediate=item['immediate'], reach=item['reach'],
                              actions=deepcopy(item['actions']), eligible=False))
        bank = dict(candidates=candidates, directions=directions, mapping=mapping, local_task=local, native_response=native,
            provenance=dict(task_sha256=m.digest(local), signal_sha256=m.digest(native)), source_capsule_bytes_verified_here=True,
            kernel_source_artifact_authenticated_here=False, kernel_flags=deepcopy(m.source.numeric.FLAGS))
        g = dict(bank=bank, rows=items, control=dict(index='zero', status='complete', immediate=0., reach=1e6,
                                                   actions=np.zeros((5*h, 2), np.float32).tolist()))
        counters = m.source.expected_usage(h, 8, False)
        resources = dict(attempted=deepcopy(counters), completed=deepcopy(counters))
        result = dict(schema=m.source.NAME+'.group', backend_kind='HOST_CUDA', task_sha256=m.digest(task),
            provenance_sha256=m.digest(pro), source_closure_sha256=m.digest(p['sources']), root_context_role=m.ROLE,
            actual_cp5_start=True, errors=[], status='no_witness', delivery=None, R_groups=[g], resource_receipt=resources, **m.source.FLAGS)
        r['rows'].append(dict(**identity, status='no_witness', result=result))
        receipt['arms'].append(dict(**identity, status='no_witness', complete=True, delivery=None,
                                   no_witness=True, resources=deepcopy(resources), **m.source.FLAGS))
    return p, r, receipt


def material(mode='normal'):
    p, r, receipt = source_fixture(); groups, inherited = m.source_material(p, r, receipt, deepcopy(receipt))
    if mode != 'normal':
        for g in groups:
            g['directions'], g['small_rows'] = small_source(g['horizon'], mode=mode)
    p.update(profile=m.profile(), source_groups=groups, inherited_resources=inherited)
    return p


@pytest.fixture(scope='module')
def saved():
    p = material(); store = Memory(); rows = m.execute(p, lambda t, s: RAM(t, s), store, permit_ram=True)
    return p, rows, store


def events_for(store, slot):
    return [e['payload'] for e in store.events if e['kind'] == 'group_event' and e['slot'] == slot]


@pytest.mark.parametrize('h', [1, 3])
def test_nonorthogonal_minimum_norm_against_independent_active_set_QP(h):
    out, ledger = math_run(h); v = out['intermediates']
    assert out['status'] == 'complete' and v['G'] != np.eye(2).tolist()
    g, a, b = np.asarray(v['G']), np.asarray(v['a']), np.asarray(v['b']); A = np.stack((a, b))
    candidates = []
    for mask in ((0,), (1,), (0, 1)):
        active = A[list(mask)]
        block = np.block([[g, -active.T], [active, np.zeros((len(mask), len(mask)))]])
        solved = np.linalg.solve(block, np.r_[np.zeros(2), np.ones(len(mask))]); c, lam = solved[:2], solved[2:]
        if np.all(A@c >= 1-1e-12) and np.all(lam >= -1e-12): candidates.append(c)
    ref = min(candidates, key=lambda c: c@g@c)
    np.testing.assert_allclose(v['c_star'], ref, rtol=1e-12, atol=1e-12)  # RAM math only, never a model tolerance.
    assert v['sI_dot_c'] > 0 and v['sR_dot_c'] < 0
    assert len(out['direction']) == h*10 and len(out['candidates']) == 4
    for i, row in enumerate(out['candidates']):
        assert row['radius'] == m.kernel.R0/2**(i+1)
        raw = np.tanh(np.asarray(row['params'], np.float64)).astype(np.float32).reshape(5*h, 2)
        assert raw.tobytes().hex() == row['raw_float32_bits']
        assert np.array_equal(raw, row['actions'])
    assert ledger.attempted == ledger.completed == m.surrogate_cost()


@pytest.mark.parametrize('mode,reason,svd,solve,compile_', [
    ('null', 'source_direction_null', 0, 0, 0), ('gram', 'gram_rank_unresolved', 1, 0, 1),
    ('zero', 'secant_zero_or_unresolved', 1, 1, 1), ('opposed', 'secant_cone_unresolved', 1, 1, 1)])
def test_degenerate_reason_null_and_exact_actual_math_charges(mode, reason, svd, solve, compile_):
    out, ledger = math_run(mode=mode)
    assert out['status'] == 'surrogate_unavailable' and out['reason'] == reason
    assert len(out['candidates']) == 4 and all(x['actions'] is None for x in out['candidates'])
    assert ledger.attempted['array_SVD_calls'] == svd and ledger.attempted['solve_calls'] == solve
    assert ledger.attempted['program_compile_calls'] == compile_ and ledger.attempted['predict_API'] == 0


@pytest.mark.parametrize('damage', ['index', 'radius', 'sign', 'params', 'raw', 'score', 'nullstatus', 'missing'])
def test_kernel_bad_source_shape_indices_and_null_refuse(damage):
    directions, rows = small_source()
    if damage == 'index': rows[0]['index'] = 1
    elif damage == 'radius': rows[0]['radius'] *= 2
    elif damage == 'sign': rows[0]['sign'] = 1
    elif damage == 'params': rows[0]['params'][0] += .125
    elif damage == 'raw': rows[0]['actions'][0][0] += .125
    elif damage == 'score': rows[0]['reach'] = True
    elif damage == 'nullstatus': directions['d0'] = None
    else: rows.pop()
    with pytest.raises(ValueError):
        m.kernel.build(1, directions, rows, lambda *args: pytest.fail('invalid source dispatched'))


def test_same_normalized_covectors_closed_form_is_valid():
    directions, rows = small_source()
    directions['d1'] = np.eye(10)[1].tolist()
    for row in rows:
        param = row['sign']*m.kernel.R0*np.asarray(directions[row['direction']])
        row.update(params=param.tolist(), actions=np.tanh(param).astype(np.float32).reshape(5, 2).tolist(),
                   immediate=2+row['sign']*m.kernel.R0*(row['direction'] == 'd0'))
    # Set reach secant to exact negative immediate secant, leaving feasible
    # same-way normalized halfspaces (kappa=+1), not an infeasible cone.
    for row in rows: row['reach'] = 4-row['immediate']
    ledger = m.core.Ledger(m.caps(1))
    ledger.scope = dict(phase='RAM_secant_math')
    out = m.kernel.build(1, directions, rows, ledger.call)
    assert out['status'] == 'complete'
    np.testing.assert_allclose(out['intermediates']['constraint_residuals'], [0, 0], atol=1e-12)


@pytest.mark.parametrize('damage', ['receipt', 'schema', 'C3', 'missingC3', 'order', 'sourcepin', 'local', 'native',
    'capsule', 'index', 'raw', 'score', 'resource', 'model', 'flag'])
def test_whole_receipt_and_real_source_ABI_refuses_mutations(damage):
    p, r, receipt = source_fixture(); rebuilt = deepcopy(receipt)
    g = r['rows'][0]['result']['R_groups'][0]; bank = g['bank']
    if damage == 'receipt': receipt['unverified_extra'] = 1
    elif damage == 'schema': receipt['schema'] = 'same_name_but_old.consume'; rebuilt = deepcopy(receipt)
    elif damage == 'C3': receipt['full_C3_demonstrated'] = True; rebuilt = deepcopy(receipt)
    elif damage == 'missingC3': del receipt['full_C3_demonstrated']; receipt['C3'] = False; rebuilt = deepcopy(receipt)
    elif damage == 'order': r['rows'].reverse()
    elif damage == 'sourcepin': receipt['report_sha256'] = '0'*64; rebuilt = deepcopy(receipt)
    elif damage == 'local': bank['local_task']['root_context']['latent_history'][0][0] += .125
    elif damage == 'native': bank['native_response']['binding_sha256'] = '0'*64
    elif damage == 'capsule': bank['source_capsule_bytes_verified_here'] = False
    elif damage == 'index': bank['candidates'][0]['direction'] = 'd1'
    elif damage == 'raw': g['rows'][0]['actions'][0][0] += .125
    elif damage == 'score': g['rows'][0]['eligible'] = True
    elif damage == 'resource': receipt['arms'][0]['resources']['attempted']['predict_API'] = 0; rebuilt = deepcopy(receipt)
    elif damage == 'model': r['model_after'] = {}
    elif damage == 'flag': r['rows'][0]['result']['physical_release'] = True
    with pytest.raises((ValueError, KeyError)):
        m.source_material(p, r, receipt, rebuilt)


def test_whole_source_extraction_preserves_original0plus8_and_paid196_40():
    p, r, receipt = source_fixture(); groups, paid = m.source_material(p, r, receipt, deepcopy(receipt))
    assert len(groups) == 4 and all(len(g['original_items']) == 9 for g in groups)
    assert paid['predict_API'] == 196 and paid['jacobian_API'] == 40 and paid['replay_calls'] == 72
    assert all(len(g['small_rows']) == 4 for g in groups)


def test_four_groups_all16_paid_even_when_first_eligible_no_delivery(saved):
    p, rows, store = saved
    assert [r['status'] for r in rows] == ['all_four_scored']*4, [(r['status'], r.get('result', {}).get('errors')) for r in rows]
    for row in rows:
        r = row['result']; assert len(r['rows']) == 4 and len(r['native_calls']) == 12
        assert r['rows'][0]['eligible'] is True  # Fixed literal RAM source zero, not a production outcome.
        assert all(x['status'] == 'complete' for x in r['rows'])
        assert r['resource_receipt']['attempted'] == r['resource_receipt']['completed'] == m.caps(row['horizon'])
        assert r['delivery'] is None and r['model_delivery_verified'] is False and r['full_C3_demonstrated'] is False
        for item in r['rows']:
            assert len(item['actions']) == 5*row['horizon'] and len(item['suffix_actions']) == 5
            assert item['signal']['parent_context_sha256'] == m.digest(item['postcontext'])
            assert len(item['postcontext']['latent_history']) == 3 and len(item['postcontext']['action_history']) == 2
    summary = m.accounting(p, rows)
    assert summary['inherited']['predict_API'] == 196 and summary['inherited']['jacobian_API'] == 40
    assert summary['new_attempted']['predict_API'] == 80 and summary['new_attempted']['jacobian_API'] == 16
    assert summary['cumulative_attempted']['predict_API'] == 276 and summary['cumulative_attempted']['jacobian_API'] == 56
    assert summary['new_attempted']['solve_calls'] == summary['new_attempted']['array_SVD_calls'] == 20
    assert summary['new_AD_primal_predict_API'] == 16 and summary['inherited_AD_primal_predict_API'] == 44


def test_full_saved_four_group_reconstruction(saved):
    p, rows, store = saved
    summaries = m.validate_rows(p, rows, store.events, permit_fixture=True)
    assert len(summaries) == 4 and all(x['complete'] and x['scored'] == 4 for x in summaries)


@pytest.mark.parametrize('damage', ['params', 'gram', 'radius', 'I', 'R', 'gate', 'source', 'flags', 'delivery', 'cost'])
def test_saved_math_native_scores_and_identity_mutations_refuse(saved, damage):
    p, rows, store = saved; r = deepcopy(rows[0]['result'])
    if damage == 'params': r['rows'][0]['params'][0] += .125
    elif damage == 'gram': r['surrogate']['intermediates']['G'][0][0] += .125
    elif damage == 'radius': r['rows'][0]['radius'] *= 2
    elif damage == 'I': r['rows'][0]['immediate'] += .125
    elif damage == 'R': r['rows'][0]['reach'] += .125
    elif damage == 'gate': r['rows'][0]['eligible'] = False
    elif damage == 'source': r['source_group_sha256'] = '0'*64
    elif damage == 'flags': r['full_C3_demonstrated'] = True
    elif damage == 'delivery': r['delivery'] = {'not_allowed': True}
    else: r['resource_receipt']['caps']['solve_calls'] += 1
    with pytest.raises(ValueError):
        m.validate_group(p['tasks'][0], p['provenance'][0], 1, p['source_groups'][0], r, events_for(store, 0),
                         source_sha=m.digest(p['sources']), permit_fixture=True)


@pytest.mark.parametrize('damage', ['unknown', 'orphan', 'missing', 'duplicate'])
def test_complete_event_census_mutations(saved, damage):
    p, rows, store = saved; es = deepcopy(events_for(store, 0))
    if damage == 'unknown': es.append(dict(kind='io', call=None, payload=dict(kind='outcome')))
    elif damage == 'orphan': es[0]['call'] = 999
    elif damage == 'missing': es.pop(next(i for i, e in enumerate(es) if e['payload']['kind'] == 'secant_return'))
    else: es.append(deepcopy(next(e for e in es if e['payload']['kind'] == 'secant_item_return')))
    with pytest.raises(ValueError):
        m.validate_group(p['tasks'][0], p['provenance'][0], 1, p['source_groups'][0], rows[0]['result'], es,
                         source_sha=m.digest(p['sources']), permit_fixture=True)


@pytest.mark.parametrize('mode', ['null', 'gram', 'zero', 'opposed'])
def test_all_four_degenerate_groups_settle_null_without_NN(mode):
    p = material(mode); store = Memory()
    rows = m.execute(p, lambda *args: pytest.fail('degenerate source ran a model'), store, permit_ram=True)
    assert [r['status'] for r in rows] == ['surrogate_unavailable']*4
    assert all(len(r['result']['rows']) == 4 and not r['result']['native_calls'] for r in rows)
    for row in rows:
        assert all(x['eligible'] is None and x['actions'] is None for x in row['result']['rows'])
    m.validate_rows(p, rows, store.events, permit_fixture=True)


def test_native_failure_retains_paid_prefix_stops_later_groups():
    class Fail(RAM):
        def replay(self, *args): raise RuntimeError('RAM edge callback failure')
    p = material(); store = Memory(); rows = m.execute(p, lambda t, s: Fail(t, s), store, permit_ram=True)
    r = rows[0]['result']
    assert r['status'] == 'partial_runtime_failure' and len(r['rows']) == 4
    assert r['resource_receipt']['attempted']['replay_calls'] == 1 and r['resource_receipt']['completed']['replay_calls'] == 0
    assert all(x['status'] == 'blocked_after_runtime_failure' for x in rows[1:])
    out = m.validate_rows(p, rows, store.events, permit_fixture=True)
    assert len(out) == 1 and out[0]['complete'] is False


def test_solve_failure_not_padded_and_later_groups_stop(monkeypatch):
    p = material(); store = Memory(); original = m.kernel.np.linalg.solve
    def fail(*args, **kwargs): raise np.linalg.LinAlgError('RAM surrogate solve failure')
    monkeypatch.setattr(m.kernel.np.linalg, 'solve', fail)
    rows = m.execute(p, lambda *args: pytest.fail('failed solve ran NN'), store, permit_ram=True)
    r = rows[0]['result']; counts = r['resource_receipt']
    assert r['status'] == 'partial_runtime_failure' and counts['attempted']['solve_calls'] == 1
    assert counts['completed']['solve_calls'] == 0 and counts['completed']['array_SVD_calls'] == 1
    assert counts['attempted']['predict_API'] == 0 and all(x['status'] == 'blocked_after_runtime_failure' for x in rows[1:])
    m.validate_rows(p, rows, store.events, permit_fixture=True)


def test_cold_import_cli_pins_and_worker_guard():
    assert all(m.sha(p) == h for p, h in m.PINS.items())
    with pytest.raises(ValueError, match='MAIN source audit'): m.run('/not/read', 'bad', '/not/write', authorization='worker')
    code = f'import sys; import {m.__name__}; assert "torch" not in sys.modules and "gymnasium" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], cwd=m.ROOT, check=True)
    out = subprocess.run([sys.executable, '-m', m.__name__, '--help'], cwd=m.ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and all(x in out.stdout for x in ('prepare', 'verify', 'run', 'consume'))


@pytest.mark.parametrize('damage,match', [
    ('intact', 'fixed four groups'), ('extra', 'exact file census'), ('missing', 'exact file census'),
    ('bytes', 'all artifact bytes'), ('traversal', 'exact file census'), ('binding', 'persisted binding'),
    ('hashchain', 'actual journal hashchain'), ('sourceflag', 'full_C3_demonstrated'), ('oldC3alias', None)])
def test_own_file_consumer_rejects_incomplete_or_mutated_RAM_artifacts(tmp_path, monkeypatch, damage, match):
    # Incomplete literal fixture never receives a receipt. Only NEW own verify
    # is stubbed: no production upstream consume and no shared globals patched.
    output = tmp_path/'RAM_saved'; output.mkdir(); prepared = tmp_path/'RAM_prepare.json'; ps = 'a'*64
    p = dict(sources={'/RAM/class': 'b'*64}, expected_model_hash='RAM_model')
    monkeypatch.setattr(m, 'verify', lambda path, h: p)
    binding = dict(prepared=str(prepared), prepare_sha256=ps)
    (output/'binding.json').write_bytes(m.canonical(binding))
    classes = [dict(path='/RAM/class', sha256='b'*64)]; model = dict(original_state_dict='RAM_model')
    events = []
    for phase in ('before', 'after'):
        events += [dict(kind='model_'+phase, hashes=model, classes=classes),
                   dict(kind='source_'+phase, sha256=m.digest(p['sources'])),
                   dict(kind='precision_'+phase, value=m.old.precision_contract())]
    tail = '0'*64; lines = []
    for i, payload in enumerate(events):
        entry = dict(index=i, previous=tail, payload=payload); tail = m.digest(entry)
        lines.append(m.canonical(dict(entry, sha256=tail))+b'\n')
    (output/'events.jsonl').write_bytes(b''.join(lines))
    r = dict(schema=m.NAME+'.report', prepared=str(prepared), prepare_sha256=ps,
        errors=[], sources_verified=True, delivery=None, source_manifest_sha256=m.digest(p['sources']),
        model_before=model, model_after=model, classes_before=classes, classes_after=classes,
        precision=[m.old.precision_contract()]*2,
        host=dict(device='cuda:0', allocation_sum=4., visible_nvidia_nodes=['RAM_NOT_A_DEVICE']),
        artifact_sha256={n: m.sha(output/n) for n in ('binding.json', 'events.jsonl')},
        journal_tail=tail, rows=[], **m.FLAGS)
    if damage == 'extra': (output/'extra.json').write_bytes(b'{}')
    elif damage == 'missing': (output/'binding.json').unlink()
    elif damage == 'bytes': (output/'binding.json').write_bytes(b'{}')
    elif damage == 'traversal': r['artifact_sha256']['../bad.json'] = 'b'*64
    elif damage == 'binding':
        (output/'binding.json').write_bytes(m.canonical(dict(binding, prepare_sha256='b'*64)))
        r['artifact_sha256']['binding.json'] = m.sha(output/'binding.json')
    elif damage == 'hashchain':
        data = (output/'events.jsonl').read_bytes().replace(b'model_before', b'model_beforX')
        (output/'events.jsonl').write_bytes(data); r['artifact_sha256']['events.jsonl'] = m.sha(output/'events.jsonl')
    elif damage == 'sourceflag': r['full_C3_demonstrated'] = True
    elif damage == 'oldC3alias': del r['full_C3_demonstrated']; r['C3'] = False
    (output/'report.json').write_bytes(m.canonical(r))
    with pytest.raises((ValueError, KeyError), match=match):
        m.consume(output, m.sha(output/'report.json'), prepared, ps)
