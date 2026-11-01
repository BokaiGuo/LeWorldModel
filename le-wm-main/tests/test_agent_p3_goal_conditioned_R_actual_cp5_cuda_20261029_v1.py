"""RAM/NumPy native-buffer fixtures only; no production prepare/consume/model."""
from copy import deepcopy
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_goal_conditioned_R_actual_cp5_cuda_20261029_v1 as m

f = runpy.run_path(str(m.ROOT/'tests/test_agent_p3_actual_cp5_r_delivery_20261029_v1.py'))
native = f['f']['fixtures']['NumpyNative']
Memory = f['f']['Memory']


def material():
    p = f['material'](); p['profile'] = m.profile()
    # A fixed RAM-only controllable two-channel goal exercises all eight slots
    # at BOTH synthetic roots. Orthogonal/zero-J degeneracy has its own fixture.
    goal = [.5, .25, 0., 1.]+[0.]*188
    t, pro = p['tasks'][1], p['provenance'][1]
    for task in (t, pro['original_task']):
        task['goal'] = goal.copy(); task['binding']['goal_sha256'] = m.digest(goal)
    pro['original_task_sha256'] = m.digest(pro['original_task']); pro['local_task_sha256'] = m.digest(t)
    return p


class RAM(native):
    """Actual NumPy two-channel nonlinear dynamics, analytic raw-zero J per context."""
    zero_dynamics = False

    def call(self, name, fn, inputs, shape=None):
        if name == 'action_encoder': self.outgoing = inputs['0'].copy()
        if name == 'predictor':
            def transition():
                z = inputs['0'].copy(); v = z[0, -1]
                raw = self.outgoing[0, -1].reshape(5, 2)*np.asarray(self.task['action_std'], np.float32)+np.asarray(self.task['action_mean'], np.float32)
                if not self.zero_dynamics:
                    u, w = raw.mean(axis=0, dtype=np.float32)
                    v[0] -= np.float32(1+10000*v[1])*u
                    v[1] += np.float32(.01)*u-np.float32(.5)*w
                return z
            fn = transition
        return super().call(name, fn, inputs, shape)

    def signal(self, req, ledger):
        self.ledger = ledger; self.step = None; h = req['horizon']; zero = np.zeros((5*h, 2), np.float32)
        self.phase = 'nominal'
        n = np.asarray(self.trajectory(req['parent_context'], zero, False)[-1]['latent_history'][-1], np.float32)
        self.arrays('nominal', value=n); self.phase = 'AD'
        def jacobian():
            ad = np.asarray(self.trajectory(req['parent_context'], zero, False)[-1]['latent_history'][-1], np.float32)
            self.step = None; self.arrays('AD_terminal', value=ad)
            j = np.zeros((192, 10*h), np.float32)
            if not self.zero_dynamics:
                y = np.float32(req['parent_context']['latent_history'][-1][1])
                j[0, ::2] = -np.float32(1+10000*y)/np.float32(5)
                j[1, ::2] = np.float32(.002); j[1, 1::2] = np.float32(-.1)
            for i, row in enumerate(j):
                self.sink.event(dict(kind='direction', scope=deepcopy(ledger.scope), direction=i))
                basis = np.eye(192, dtype=np.float32)[i]
                def raw_return():
                    self.arrays('raw_backward', len(ledger.events)-1, value=row); return row.copy()
                ledger.call('vjp', lambda: self.call('grad', lambda: ledger.call('raw_backward', raw_return),
                    {'output': ad, 'raw': zero.reshape(-1), 'basis': basis}), directions=1)
            return j
        j = ledger.call('jacobian', jacobian); self.arrays('jacobian', value=j)
        return dict(binding_sha256=req['binding_sha256'], parent_context_sha256=req['parent_context_sha256'], horizon=h,
                    nominal=n.tolist(), jacobian=j.tolist(), producer_sha256=m.sha(m.signal.__file__))


class ZeroRAM(RAM):
    zero_dynamics = True


def execute(cls=RAM):
    p = material(); s = Memory(); rows = m.execute(p, lambda t, sink: cls(t, sink), s, permit_ram=True)
    return p, rows, s


@pytest.fixture(scope='module')
def saved(): return execute()


@pytest.fixture(scope='module')
def zero_saved(): return execute(ZeroRAM)


def events_for(s, slot):
    return [e['payload'] for e in s.events if e['kind'] == 'group_event' and e['slot'] == slot]


@pytest.mark.parametrize('h,predicts', [(1, 40), (3, 66)])
def test_caps_only_solve_plus_one_all27(h, predicts):
    cap, oldcap = m.caps(h), m.prior.caps(h)
    assert set(cap) == set(m.core.AXES) and len(cap) == 27
    assert all(cap[k] == oldcap[k]+(k == 'solve_calls') for k in cap)
    assert cap['predict_API'] == predicts and cap['jacobian_API'] == 10 and cap['replay_calls'] == 20
    assert cap['solve_calls'] == cap['array_SVD_calls'] == 10
    assert m.expected_usage(h, 8, True) == cap
    assert m.bank_cost()['solve_calls'] == m.bank_cost()['array_SVD_calls'] == 1
    assert m.profile()['wall_seconds'] == 3600 and m.profile()['group_wall_seconds'] == 1800
    assert m.profile()['final_reserve_bytes'] == 512*1024**2


def test_all_four_actual_inputs_native_buffers_and_delivery(saved):
    p, rows, s = saved
    assert len(rows) == 4
    assert all(r['status'] in m.SETTLED for r in rows), [(r['status'], r.get('result', {}).get('errors')) for r in rows]
    for row in rows:
        r = row['result']; g = r['R_groups'][0]; bank = g['bank']
        assert bank['status'] == 'complete'
        assert len(bank['candidates']) == len(g['rows']) == 8
        assert g['control']['status'] == 'complete'
        assert bank['kernel_flags']['C3'] is False and 'C3' not in r
        assert bank['rule']['source_artifact_authenticated_here'] is False
        assert r['full_C3_demonstrated'] is False and bank['source_capsule_bytes_verified_here'] is False
        assert bank['local_task'] == bank['mapping']['parent_task']
        assert bank['native_response'] == bank['mapping']['native_response']
        assert bank['provenance']['task_sha256'] == m.digest(bank['local_task'])
        assert bank['provenance']['signal_sha256'] == m.digest(bank['native_response'])
        assert r['eligible_order'] == sorted(g['eligible_indices'], key=lambda i: (g['rows'][i]['immediate'], i))
        assert r['resource_receipt']['attempted'] == m.expected_usage(row['horizon'], 8, bool(r['eligible_order']))
        for item in [g['control']]+g['rows']:
            assert len(item['postcontext']['latent_history']) == 3 and len(item['postcontext']['action_history']) == 2
            assert item['signal']['parent_context_sha256'] == m.digest(item['postcontext'])
        if r['delivery'] is not None:
            d = r['delivery']; assert d['schema'] == m.NAME+'.delivery'
            assert d['bank_sha256'] == m.digest(bank) and d['candidate_index'] == r['eligible_order'][0]
            assert d['commit_raw5'] == d['raw_edge'][:5] and d['lookahead_raw'] == d['raw_edge'][5:]
            assert len(d['raw_edge']) == row['horizon']*5 and len(d['commit_raw5']) == 5
            assert d['suffix_in_plan'] is False and d['physical_release'] is False
            assert r['verification'][1]['segments'] == ([5] if row['horizon'] == 1 else [5, 10])
            assert all(v['path'] == d['path'] for v in r['verification'])
        else:
            assert g['eligible_indices'] == [] and all(v['status'] == 'not_requested' for v in r['verification'])
    assert len(s.capsules) == 4


@pytest.mark.parametrize('slot', range(4))
def test_literal_math_bitwise_matches_frozen_numeric_kernel(saved, slot):
    p, rows, s = saved; bank = rows[slot]['result']['R_groups'][0]['bank']
    reference = m.numeric.build_bank(bank['local_task'], bank['native_response'], bank['provenance'])
    assert reference['status'] == bank['status']
    for k in ('intermediates', 'directions', 'direction_reasons', 'candidates'):
        assert m.canonical(bank[k]) == m.canonical(reference[k]), k
    assert m.canonical(reference['zero']['actions']) == m.canonical(bank['zero'])
    assert bank['resources'] == bank['completed'] == m.bank_cost()
    assert [e['resource'] for e in bank['ledger_events']] == ['program_compile', 'array_svd', 'solve']
    assert all(e['status'] == 'returned' for e in bank['ledger_events'])


def test_full_saved_four_slot_reconstruction(saved):
    p, rows, s = saved
    summaries = m.validate_rows(p, rows, s.events, permit_fixture=True, capsule_reader=s.reader)
    assert len(summaries) == 4 and all(x['complete'] for x in summaries)
    assert all(x['valid_candidates'] == 8 and x['blocked_candidates'] == 0 for x in summaries)


def test_zero_gradient_all8_null_and_separately_paid_zero(zero_saved):
    p, rows, s = zero_saved
    assert [r['status'] for r in rows] == ['no_witness']*4
    for row in rows:
        r = row['result']; g = r['R_groups'][0]
        assert len(g['rows']) == 8 and g['control']['status'] == 'complete'
        assert all(x['status'] == 'blocked_degenerate_direction' and x['actions'] is None and x['eligible'] is None for x in g['rows'])
        assert all(x['reason'] == 'b_zero' for x in g['rows'])
        assert r['delivery'] is None and len(r['native_calls']) == 4
        assert r['resource_receipt']['attempted'] == m.expected_usage(row['horizon'], 0, False)
        assert r['resource_receipt']['attempted']['replay_calls'] == 2
        assert r['resource_receipt']['attempted']['jacobian_API'] == 2
        bank = g['bank']; reference = m.numeric.build_bank(bank['local_task'], bank['native_response'], bank['provenance'])
        for key in ('intermediates', 'directions', 'direction_reasons', 'candidates'):
            assert m.canonical(bank[key]) == m.canonical(reference[key])
    summaries = m.validate_rows(p, rows, s.events, permit_fixture=True, capsule_reader=s.reader)
    assert len(summaries) == 4 and all(x['blocked_candidates'] == 8 for x in summaries)


def test_rank_one_tangent_null_without_replacement():
    p = material(); store = Memory()
    sink = type('Sink', (), {'event': lambda self, e: store.event(e)})()
    r = m.group(p['tasks'][0], p['provenance'][0], 1, lambda t, s: f['RAM'](t, s), sink,
        source_sha=m.digest(p['sources']), permit_ram=True, capsule=lambda cid, call: store.capsule(0, cid, call))
    assert r['status'] in m.SETTLED, r['errors']
    bank = r['R_groups'][0]['bank']
    assert bank['status'] == 'complete_with_blocked_directions'
    assert [c['index'] for c in bank['candidates'] if c['actions'] is None] == [2, 3, 6, 7]
    assert len(r['native_calls']) == 16+2*bool(r['delivery'])
    assert r['resource_receipt']['attempted'] == m.expected_usage(1, 4, bool(r['delivery']))
    reference = m.numeric.build_bank(bank['local_task'], bank['native_response'], bank['provenance'])
    for key in ('intermediates', 'directions', 'direction_reasons', 'candidates'):
        assert m.canonical(bank[key]) == m.canonical(reference[key])
    m.validate_group(p['tasks'][0], p['provenance'][0], 1, r, store.events, source_sha=m.digest(p['sources']),
                     permit_fixture=True, capsule_reader=lambda cid, call: store.reader(0, cid, call))


@pytest.mark.parametrize('damage', ['caps', 'eligibility', 'kernelC3', 'productionC3', 'bankbinding', 'source',
    'localtask', 'raw', 'direction', 'intermediate', 'zero', 'order', 'items', 'resources', 'normalizer', 'role'])
def test_saved_mutations_reject_without_model(saved, damage):
    p, rows, s = saved; r = deepcopy(rows[0]['result']); g = r['R_groups'][0]; bank = g['bank']
    if damage == 'caps': r['resource_receipt']['caps']['solve_calls'] -= 1
    elif damage == 'eligibility': g['rows'][0]['eligible'] = not g['rows'][0]['eligible']
    elif damage == 'kernelC3': bank['kernel_flags']['C3'] = True
    elif damage == 'productionC3': r['full_C3_demonstrated'] = True
    elif damage == 'bankbinding': bank['native_response']['binding_sha256'] = m.digest(p['tasks'][0]['binding'])+'bad'
    elif damage == 'source': bank['source_capsule_bytes_verified_here'] = True
    elif damage == 'localtask': bank['local_task']['root_context']['latent_history'][0][0] += .125
    elif damage == 'raw': g['rows'][0]['actions'][0][0] += .125
    elif damage == 'direction': bank['directions']['d0'][0] += .125
    elif damage == 'intermediate': bank['intermediates']['b'][0] += .125
    elif damage == 'zero': g['control']['status'] = 'not_requested'
    elif damage == 'order': r['eligible_order'] = [999]
    elif damage == 'items': g['rows'].pop()
    elif damage == 'resources': g['resources']['solve_calls'] += 1
    elif damage == 'normalizer': bank['local_task']['action_std'][0] *= 2
    elif damage == 'role': r['actual_cp5_start'] = False
    with pytest.raises((ValueError, AssertionError)):
        m.validate_group(p['tasks'][0], p['provenance'][0], 1, r, events_for(s, 0),
            source_sha=m.digest(p['sources']), permit_fixture=True, capsule_reader=lambda cid, call: s.reader(0, cid, call))


@pytest.mark.parametrize('damage', ['padding', 'replacement', 'eligible', 'missing'])
def test_null_mutations_reject(zero_saved, damage):
    p, rows, s = zero_saved; r = deepcopy(rows[0]['result']); row = r['R_groups'][0]['rows'][0]
    if damage == 'padding': row['status'] = 'complete'
    elif damage == 'replacement': row['actions'] = [[0., 0.]]*5
    elif damage == 'eligible': row['eligible'] = False
    elif damage == 'missing': r['R_groups'][0]['rows'].pop()
    with pytest.raises(ValueError):
        m.validate_group(p['tasks'][0], p['provenance'][0], 1, r, events_for(s, 0),
            source_sha=m.digest(p['sources']), permit_fixture=True, capsule_reader=lambda cid, call: s.reader(0, cid, call))


@pytest.mark.parametrize('damage', ['unknown', 'orphan', 'missing', 'duplicate'])
def test_event_census_rejects(saved, damage):
    p, rows, s = saved; es = deepcopy(events_for(s, 0))
    if damage == 'unknown': es.append(dict(kind='io', call=None, payload=dict(kind='reward')))
    elif damage == 'orphan': es[0]['call'] = 999
    elif damage == 'missing': es.pop(next(i for i, e in enumerate(es) if e['payload']['kind'] == 'goal_bank_return'))
    elif damage == 'duplicate': es.append(deepcopy(next(e for e in es if e['payload']['kind'] == 'delivery_selection')))
    with pytest.raises(ValueError):
        m.validate_group(p['tasks'][0], p['provenance'][0], 1, rows[0]['result'], es,
            source_sha=m.digest(p['sources']), permit_fixture=True, capsule_reader=lambda cid, call: s.reader(0, cid, call))


def test_native_failure_keeps_paid_prefix_and_blocks_later_dispatch():
    class Fail(RAM):
        def replay(self, *args): raise RuntimeError('RAM paid edge failure')
    p, rows, s = execute(Fail)
    r = rows[0]['result']
    assert r['status'] == 'partial_runtime_failure' and r['R_groups'][0]['control']['status'] == 'failed'
    assert len(r['R_groups'][0]['rows']) == 8
    assert r['resource_receipt']['attempted']['replay_calls'] == 1
    assert r['resource_receipt']['completed']['replay_calls'] == 0
    assert all(x['status'] == 'blocked_after_runtime_failure' for x in rows[1:])
    summaries = m.validate_rows(p, rows, s.events, permit_fixture=True, capsule_reader=s.reader)
    assert len(summaries) == 1 and summaries[0]['complete'] is False


def test_bank_atomic_reserve_preflight_happens_before_any_bank_math():
    p = material(); t = p['tasks'][0]
    ledger = m.core.Ledger(m.prior.caps(1))  # Deliberately lacks newly authorized solve+1.
    ledger.attempted = m.base.signal_cost(1)
    session = type('Session', (), {'ledger': ledger})()
    with pytest.raises(ValueError, match='atomic maximum'):
        m.evaluate_R(t, dict(native_response=dict(horizon=1)), session, on_start=lambda _: pytest.fail('bank started'))


@pytest.mark.parametrize('damage', ['local', 'binding', 'normalizer', 'sourceproducer'])
def test_make_bank_refuses_coherent_bad_local_pairing(saved, damage):
    p, rows, s = saved; bank = rows[0]['result']['R_groups'][0]['bank']; mapping = deepcopy(bank['mapping'])
    if damage == 'local': mapping['parent_task']['root_context']['latent_history'][0][0] += .125
    elif damage == 'binding': mapping['native_response']['binding_sha256'] = m.digest(p['provenance'][0]['original_task']['binding'])
    elif damage == 'normalizer': mapping['parent_task']['action_std'][0] *= 2
    else: mapping['native_response']['producer_sha256'] = 'f'*64
    mapping['mapping_sha256'] = m.digest({k: v for k, v in mapping.items() if k != 'mapping_sha256'})
    session = type('S', (), {'ledger': None})()
    with pytest.raises(ValueError):
        m.make_bank(p['tasks'][0], mapping, session, on_start=lambda _: pytest.fail('invalid local source admitted'))


def test_p_zero_and_b_zero_reasons_preserve_all_slots():
    # p_zero is the frozen defensive branch (not claimed observed in a finite
    # positive-ridge HOST solve). Exercise it without altering the solver/J.
    for p, b, reason in ((np.zeros(10), np.ones(10), 'p_zero'), (np.ones(10), np.zeros(10), 'b_zero')):
        directions, reasons, data = m.numeric.direction_data(p, b)
        assert directions == dict(d0=None, d1=None) and reasons == dict(d0=reason, d1=reason)
        slots = m.numeric.slots()
        assert len(slots) == 8 and all(x['actions'] is None for x in slots)


@pytest.mark.parametrize('mode', ['whole', 'segmented'])
def test_independent_verification_wrong_path_never_delivers(mode):
    class Wrong(RAM):
        def replay(self, context, actions, segmented, ledger):
            path = super().replay(context, actions, segmented, ledger)
            if ledger.scope['phase'] == 'delivery_verification' and ledger.scope['mode'] == mode:
                path[-1]['latent_history'][-1][0] += .125
            return path
    p, rows, store = execute(Wrong)
    # These predetermined RAM dynamics have a witness at root0. No model
    # outcome is consulted or used to pick a production root/parameter.
    r = rows[0]['result']
    assert r['eligible_order'] and r['status'] == 'partial_runtime_failure'
    assert r['delivery'] is None and not r['model_delivery_verified']
    assert all(x['status'] == 'blocked_after_runtime_failure' for x in rows[1:])
    assert r['resource_receipt']['attempted']['replay_calls'] == (19 if mode == 'whole' else 20)


def test_exact_flags_pins_cold_help_and_authorization():
    assert all(m.sha(p) == h for p, h in m.PINS.items())
    assert m.FLAGS['full_C3_demonstrated'] is False and 'C3' not in m.FLAGS
    assert m.rules()['bank']['C3'] is False
    with pytest.raises(ValueError, match='MAIN source audit'):
        m.run('/not/read', 'bad', '/not/write', authorization='worker')
    code = f'import sys; import {m.__name__}; assert "torch" not in sys.modules; assert "gymnasium" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], cwd=m.ROOT, check=True)
    run = subprocess.run([sys.executable, '-m', m.__name__, '--help'], cwd=m.ROOT, capture_output=True, text=True)
    assert run.returncode == 0 and all(k in run.stdout for k in ('prepare', 'verify', 'run', 'consume'))


@pytest.mark.parametrize('damage,match', [
    ('intact_reaches_denominator', 'all four fixed slots'),
    ('extra', 'exact all file census'), ('missing', 'exact all file census'),
    ('bytes', 'all saved bytes'), ('traversal', 'explicit artifacts only'),
    ('binding', 'persisted binding'), ('hashchain', 'actual journal hashchain'),
    ('sourceflag', 'full_C3_demonstrated'), ('oldC3alias', None),
])
def test_consumer_own_file_census_sha_binding_and_hashchain(tmp_path, monkeypatch, damage, match):
    # Deliberately incomplete RAM file fixture: it MUST NEVER receive a receipt.
    # verify is replaced only on this NEW module to avoid production upstream
    # build. No shared module/NN callbacks or producer globals are patched.
    output = tmp_path/'RAM_saved'; output.mkdir(); prepared = tmp_path/'RAM_prepare.json'
    prepare_sha = 'a'*64
    p = dict(sources={'/RAM/class': 'b'*64}, expected_model_hash='RAM_model')
    monkeypatch.setattr(m, 'verify', lambda path, h: p)
    binding = dict(prepared=str(prepared), prepare_sha256=prepare_sha)
    (output/'binding.json').write_bytes(m.canonical(binding))
    classes = [dict(path='/RAM/class', sha256='b'*64)]
    model = dict(original_state_dict='RAM_model')
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
    r = dict(schema=m.NAME+'.report', prepared=str(prepared), prepare_sha256=prepare_sha,
        status='four_slots_settled_not_admitted', errors=[], sources_verified=True,
        source_manifest_sha256=m.digest(p['sources']), model_before=model, model_after=model,
        classes_before=classes, classes_after=classes, precision=[m.old.precision_contract()]*2,
        host=dict(device='cuda:0', allocation_sum=4., visible_nvidia_nodes=['RAM_NOT_A_DEVICE']),
        artifact_sha256={n: m.sha(output/n) for n in ('binding.json', 'events.jsonl')},
        journal_tail=tail, rows=[], **m.FLAGS)
    if damage == 'extra': (output/'extra.json').write_bytes(b'{}')
    elif damage == 'missing': (output/'binding.json').unlink()
    elif damage == 'bytes': (output/'binding.json').write_bytes(b'{}')
    elif damage == 'traversal': r['artifact_sha256']['../signal_s0_0000.json'] = 'b'*64
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
        m.consume(output, m.sha(output/'report.json'), prepared, prepare_sha)
