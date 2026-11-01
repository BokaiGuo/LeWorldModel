"""RAM dynamics and callback mutations only; no model or production artifacts."""
from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_parent_r_bank_evaluator_20261029_v1 as m


def task():
    c = dict(layout='states_with_between_state_actions_v1', context_length=3,
        latent_dim=192, action_dim=10, latent_history=np.zeros((3,192)).tolist(),
        action_history=np.zeros((2,10)).tolist())
    t = dict(root_context=c, goal=[1.]+[0.]*191, projection=np.eye(192).tolist(),
        action_mean=[.125, -.25], action_std=[2., 4.])
    t['binding'] = {k:m.digest(k) for k in m.core.BINDINGS}
    t['binding'].update(goal_sha256=m.digest(t['goal']), projection_sha256=m.digest(t['projection']),
        context_sha256=m.digest(c), normalizer_sha256=m.digest([t['action_mean'], t['action_std']]),
        action_domain_sha256=m.digest(m.core.action_domain()))
    return m.core.validate_task(t)


class RAM:
    """x' = x-(1+100y)*mean(raw); y'=y+mean(raw), one packet.

    Prefix changes actual postcontext response; nominal and J are analytic
    outputs of this explicit nonneural fixture, not a frozen-model experiment.
    """
    kind = 'RAM_FIXTURE'

    def __init__(self, mutate=None):
        self.calls = []
        self.mutate = mutate

    @staticmethod
    def roll(t, context, actions):
        c = deepcopy(context)
        path = []
        for raw in np.asarray(actions, np.float32).reshape(-1,5,2):
            v = np.asarray(c['latent_history'][-1], np.float32).copy()
            step = raw.mean(dtype=np.float32)
            v[0] -= np.float32(1+100*v[1])*step
            v[1] += step
            token = ((raw-np.asarray(t['action_mean'],np.float32))/np.asarray(t['action_std'],np.float32)).reshape(10)
            c = dict(c, latent_history=c['latent_history'][1:]+[v.tolist()],
                action_history=[c['action_history'][-1], token.tolist()])
            path.append(deepcopy(c))
        return path

    def __call__(self, req):
        self.calls.append(deepcopy(req))
        if req['operation'] == 'signal':
            j = np.zeros((192,10), np.float32)
            j[0] = -np.float32(1+100*req['context']['latent_history'][-1][1])/np.float32(10)
            j[1] = np.float32(.1)
            payload = dict(nominal_path=self.roll(req['task'], req['context'], [[0.,0.]]*5),
                jacobian=j.tolist(), input_context_sha256=m.digest(req['context']),
                jacobian_chart=m.contract()['jacobian_chart'])
        else:
            payload = dict(path=self.roll(req['task'], req['context'], req['inputs']['actions']))
        out = dict(request_sha256=m.digest(req), resources=req['resources'], payload=payload)
        if self.mutate:
            self.mutate(req, out)
        return out


def inputs(h=1, equal=False):
    t = task()
    bank = [np.full((5*h,2), 0. if equal else .125+i*.015625, np.float32).tolist() for i in range(8)]
    z = np.zeros((5*h,2), np.float32).tolist()
    a = dict(source_sha256=m.digest('caller bank fixture'), bank_sha256=m.digest(bank),
        zero_sha256=m.digest(z), root_task_sha256=m.digest(t), parent_context_sha256=m.digest(t['root_context']),
        goal_sha256=m.digest(t['goal']), Q_sha256=m.digest(t['projection']),
        normalizer_sha256=m.digest([t['action_mean'],t['action_std']]),
        action_domain_sha256=t['binding']['action_domain_sha256'],
        bank_authenticated=False, authentication_sha256=None)
    return dict(task=t, parent=deepcopy(t['root_context']), prefix_actions=[], prefix_path=[], bank=bank,
        zero=z, horizon=h, authority=a, caps=m.minimum(h), used=m.zeros(),
        verification_reserve=m.zeros(), sink=lambda e:None)


@pytest.mark.parametrize('h', [1,3])
def test_full_eight_zero_all_contexts_real_child_j_and_complete_charges(h):
    k = inputs(h)
    events = []
    k['sink'] = events.append
    cb = RAM()
    r = m.evaluate(**k, callback=cb)
    assert r['status'] == 'witness_found'
    assert len(cb.calls) == 27 and len(r['rows']) == 8
    assert r['control']['status'] == 'complete'
    assert r['attempted'] == r['completed'] == m.minimum(h)
    assert r['attempted']['predict_API'] == 9*(h+3)
    assert r['attempted']['jacobian_API'] == 9
    assert r['attempted']['J_elements'] == 9*1920
    assert r['attempted']['VJP_directions'] == 9*192
    assert r['attempted']['solve_calls'] == r['attempted']['array_SVD_calls'] == 9
    assert r['attempted']['encoder_calls'] == 0
    assert events == r['events']
    assert r['selected_index'] == min(r['eligible_indices'], key=lambda i:(r['rows'][i]['reach'],i))
    assert all(len(v['edge_actions']) == 5*h and len(v['suffix_actions']) == 5 for v in r['rows'])
    for row in r['rows']:
        post = row['edge']['context']
        assert row['signal']['input_context_sha256'] == m.digest(post)
        expected = m.original.bounded_ridge(row['signal']['jacobian'],
            (np.asarray(k['task']['goal'])-np.asarray(row['signal']['nominal'])).tolist())
        assert row['solve'] == expected
        assert row['reach'] == expected['own_reach_value']
        assert row['signal']['jacobian'] != r['control']['signal']['jacobian']
    # Real primitive command and normalization chain is retained, not terminal192 only.
    for req in cb.calls:
        assert len(req['context']['latent_history']) == 3
        assert len(req['context']['action_history']) == 2
    assert not r['production_capable'] and not r['full_policy_implemented']
    json.dumps(r, allow_nan=False)


def test_no_witness_keeps_all_eight_no_fallback_or_copied_winner():
    r = m.evaluate(**inputs(equal=True), callback=RAM())
    assert r['status'] == 'no_witness'
    assert r['selected_index'] is None and r['eligible_indices'] == []
    assert [v['eligible'] for v in r['rows']] == [False]*8
    assert [v['index'] for v in r['rows']] == list(range(8))
    assert r['attempted'] == m.minimum(1)


def test_stable_tie_and_strict_equalities():
    k = inputs()
    k['bank'] = [deepcopy(k['bank'][0]) for _ in range(8)]
    k['authority']['bank_sha256'] = m.digest(k['bank'])
    r = m.evaluate(**k, callback=RAM())
    assert r['eligible_indices'] == list(range(8)) and r['selected_index'] == 0
    assert len([e for e in r['events'] if e['kind']=='dispatch']) == 36


def test_equal_immediate_is_ineligible_even_with_strictly_better_reach():
    k = inputs()
    k['task']['goal'] = [0., .25]+[0.]*190
    k['task']['binding']['goal_sha256'] = m.digest(k['task']['goal'])
    k['bank'] = [np.full((5,2), .25, np.float32).tolist() for _ in range(8)]
    k['authority'].update(goal_sha256=m.digest(k['task']['goal']),
        root_task_sha256=m.digest(k['task']), bank_sha256=m.digest(k['bank']))
    r = m.evaluate(**k, callback=RAM())
    assert r['status'] == 'no_witness'
    assert all(v['immediate'] == r['control']['immediate'] and
        v['reach'] < r['control']['reach'] for v in r['rows'])
    assert all(v['eligible'] is False for v in r['rows'])


@pytest.mark.parametrize('axis', m.AXES)
def test_atomic_reserve_all_axes_no_callback_before_refusal(axis):
    k = inputs()
    k['verification_reserve'][axis] = 1
    cb = RAM()
    r = m.evaluate(**k, callback=cb)
    assert r['status'] == 'resource_cap_before_bank' and cb.calls == []
    assert r['attempted'] == r['completed'] == m.zeros()
    assert all(v['status']=='not_requested' for v in r['rows'])


def test_used_plus_reserve_exact_fit_and_unspent_reserve():
    k = inputs()
    k['used']['predict_API'] = 5
    k['verification_reserve']['predict_API'] = 7
    k['caps']['predict_API'] += 12
    r = m.evaluate(**k, callback=RAM())
    assert r['status'] == 'witness_found'
    assert r['used_after']['predict_API']+7 == k['caps']['predict_API']


@pytest.mark.parametrize('field', ['bank_sha256','zero_sha256','root_task_sha256',
    'parent_context_sha256','normalizer_sha256','Q_sha256','goal_sha256','action_domain_sha256'])
def test_identity_mutations_before_dispatch(field):
    k = inputs()
    k['authority'][field] = m.digest('wrong')
    cb = RAM()
    with pytest.raises(ValueError):
        m.evaluate(**k, callback=cb)
    assert cb.calls == []


@pytest.mark.parametrize('mutation', ['width','h2','length','raw','zero','Q','nan','float32', 'unknown_axis'])
def test_input_contract_failclosed(mutation):
    k = inputs()
    if mutation == 'width': k['bank'].pop()
    if mutation == 'h2': k['horizon'] = 2
    if mutation == 'length': k['bank'][0].pop()
    if mutation == 'raw': k['bank'][0][0][0] = 2.
    if mutation == 'zero': k['zero'][0][0] = .25
    if mutation == 'Q': k['task']['projection'][0][0] = 2.
    if mutation == 'nan': k['bank'][0][0][0] = float('nan')
    if mutation == 'float32': k['bank'][0][0][0] = .1
    if mutation == 'unknown_axis': k['caps']['FLOPs'] = 0
    with pytest.raises(ValueError):
        m.evaluate(**k, callback=RAM())


@pytest.mark.parametrize('mutation', ['request','resources','path','J_parent','J_chart','J_shape','J_nan','exception'])
def test_partial_failure_preserves_attempted_and_callback_return(mutation):
    def mutate(req, out):
        if mutation == 'exception': raise RuntimeError('explicit fixture error')
        if mutation == 'request': out['request_sha256'] = m.digest('stale')
        if mutation == 'resources': out['resources']['predict_API'] = 0
        if mutation == 'path': out['payload']['path'][0]['action_history'][1][0] = 0.
        if req['operation'] == 'signal':
            if mutation == 'J_parent': out['payload']['input_context_sha256'] = m.digest('root not post')
            if mutation == 'J_chart': out['payload']['jacobian_chart'] = 'unbound chart'
            if mutation == 'J_shape': out['payload']['jacobian'].pop()
            if mutation == 'J_nan': out['payload']['jacobian'][0][0] = float('nan')
    cb = RAM(mutate)
    r = m.evaluate(**inputs(), callback=cb)
    assert r['status'] == 'partial_failure' and r['control']['status'] == 'failed'
    assert all(v['status']=='not_requested' for v in r['rows'])
    assert sum(r['attempted'].values()) > sum(r['completed'].values())
    assert r['events'][-1]['kind'] == 'failed'
    assert r['selected_index'] is None


def test_nonroot_full_prefix_and_old_history_identity():
    k = inputs()
    prefix = np.full((5,2), .0625, np.float32).tolist()
    path = RAM.roll(k['task'], k['parent'], prefix)
    k.update(prefix_actions=prefix, prefix_path=path, parent=deepcopy(path[-1]))
    k['authority']['parent_context_sha256'] = m.digest(k['parent'])
    r = m.evaluate(**k, callback=RAM())
    assert r['status'] in ('no_witness','witness_found')
    k['parent']['latent_history'][0][5] = 1.
    k['authority']['parent_context_sha256'] = m.digest(k['parent'])
    with pytest.raises(ValueError, match='root-to-parent'):
        m.evaluate(**k, callback=RAM())


def test_input_mutating_callback_cannot_mutate_authority():
    k = inputs()
    before = m.digest(k['task'])
    def mutate(req, out):
        req['task']['goal'][0] = 22.
    r = m.evaluate(**k, callback=RAM(mutate))
    assert r['status'] == 'witness_found' and m.digest(k['task']) == before


def test_late_failure_keeps_eight_slots_no_refund_or_retry():
    def mutate(req, out):
        if req['index'] == 3 and req['operation'] == 'signal':
            raise RuntimeError('late fixture error')
    cb = RAM(mutate)
    r = m.evaluate(**inputs(), callback=cb)
    assert r['status'] == 'partial_failure'
    assert [v['status'] for v in r['rows']] == ['complete']*3+['failed']+['not_requested']*4
    assert r['control']['status'] == 'complete' and r['selected_index'] is None
    assert r['attempted']['jacobian_API'] == 5 and r['completed']['jacobian_API'] == 4
    assert r['attempted']['solve_calls'] == 4
    assert r['used_after'] == r['attempted']
    assert len([q for q in cb.calls if q['index'] == 3 and q['operation']=='signal']) == 1


@pytest.mark.parametrize('field', ['caps','used','verification_reserve'])
def test_missing_unknown_resource_not_filled_zero(field):
    k = inputs()
    k[field].pop('AD_primal_calls')
    with pytest.raises(ValueError):
        m.evaluate(**k, callback=RAM())


@pytest.mark.parametrize('kind', ['HOST_CUDA',None])
def test_no_production_even_with_claimed_authenticated_bank(kind):
    k = inputs()
    k['authority'].update(bank_authenticated=True, authentication_sha256=m.digest('pointer only'))
    cb = RAM()
    cb.kind = kind
    with pytest.raises(ValueError, match='RAM fixtures only'):
        m.evaluate(**k, callback=cb)


def test_cold_import_no_torch_or_foreign_execution_and_cli():
    code = """
import sys
from experiments.spectral_reachability import agent_p3_parent_r_bank_evaluator_20261029_v1 as m
assert 'torch' not in sys.modules
assert 'gymnasium' not in sys.modules
assert not m.contract()['production_capable']
"""
    subprocess.run([sys.executable, '-c', code], check=True)
    p = subprocess.run([sys.executable, '-m', m.__name__, '--contract'], check=True, capture_output=True, text=True)
    assert json.loads(p.stdout)['candidates'] == 8
