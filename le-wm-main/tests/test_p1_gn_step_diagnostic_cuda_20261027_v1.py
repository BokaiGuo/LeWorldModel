"""NumPy/in-memory mocks and saved inputs only; no tmp directories/models."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import p1_gn_step_diagnostic_cuda_20261027_v1 as m


class MemoryStore:
    def __init__(self): self.events = []; self.files = {}
    def event(self, e): self.events.append(deepcopy(e))
    def write(self, name, value, **kwargs):
        assert name not in self.files
        self.files[name] = deepcopy(value)


def task():
    return dict(environment='fixture', task_contract_sha256='1'*64, model_sha256='2'*64,
        model_config_sha256='3'*64, metric_source_sha256='4'*64, goal_source_sha256='5'*64,
        representation='native_projected_cls', latent_dim=192, horizon=2, packet_steps=5, control_dim=2,
        action_mean=[0., 0.], action_std=[1., 1.], packet_logit_cap=1.7, feedback='open_loop_model_only',
        goal=np.zeros(192).tolist(), projection=np.eye(192).tolist(), root_raw_action_history=np.zeros((10, 2)).tolist(),
        root_context=dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192, action_dim=10,
            latent_history=np.zeros((3, 192)).tolist(), action_history=np.zeros((2, 10)).tolist()))


def path_for(t, raw):
    before = deepcopy(t['root_context']); result = []
    tokens = m.core.normalized_tokens(t, raw)
    for k in range(2):
        last = (np.asarray(before['latent_history'][-1], np.float32)+np.float32(np.mean(raw[5*k:5*k+5]))).tolist()
        after = dict(before, latent_history=before['latent_history'][1:]+[last],
                     action_history=[before['action_history'][-1], tokens[k].tolist()])
        result.append(after); before = after
    return result


class Backend:
    def rollout(self, t, raw, meter, mode):
        contexts = path_for(t, raw)
        for k, c in enumerate(contexts):
            for resource in ('action_encoder', 'predict'):
                token = meter.begin(resource); meter.end(token)
            meter.emit('native_packet', mode=mode, packet=k, context=c)
        return contexts


def fixture_root(index):
    t = task(); x = np.full((2, 10), .5); y = -x
    r = dict(root_index=index, root_sha256=str(index)*64, task_id=m.core.task_id(t), incumbent_logits=x.tolist(),
             candidates=[m.segment(x, y, s, t['packet_logit_cap']) for s in m.SCALES])
    raw = np.tanh(x.astype(np.float32)).reshape(10, 2); contexts = path_for(t, raw)
    return r, t, dict(contexts=contexts, cost=m.core.model_cost(t, contexts[-1]['latent_history'][-1]))


def test_fixed_full_two_root_budget_even_first_scale_improves():
    store = MemoryStore(); meter = m.Meter(store)
    rows = [m.diagnostic_root(*fixture_root(i), Backend(), meter, store) for i in range(2)]
    assert meter.counts() == {k:dict(attempted=32, completed=32) for k in ('predict', 'action_encoder')}
    assert len(meter.events) == 64 and meter.ordinal == 15
    assert len([e for e in store.events if e['kind'] == 'native_packet']) == 32
    assert all(r['strict_improvement'] and r['selected_scale'] == .5 and len(r['scale_costs']) == 4 for r in rows)
    assert [store.files[f'eval_{i:02d}.json']['mode'] for i in range(5, 8)] == ['nominal', 'whole_replay', 'segmented_replay']


def test_segment_same_convex_logit_line_then_native_tanh():
    x = np.full((2, 10), .5); y = -x
    for s in m.SCALES:
        r = m.segment(x, y, s, 1.7)
        z = (x+s*(y-x)).astype(np.float32)
        assert np.array_equal(r['logits'], z)
        assert np.array_equal(r['commands'], np.tanh(z).reshape(10, 2))


@pytest.mark.parametrize('scale', [1., 0., .75, -.5, float('nan')])
def test_no_added_scale_or_endpoint_candidate(scale):
    with pytest.raises(ValueError): m.segment(np.zeros((2, 10)), np.zeros((2, 10)), scale, 1.)


def test_no_projection_or_clipping_of_illegal_endpoints():
    with pytest.raises(ValueError): m.segment(np.ones((2, 10)), np.zeros((2, 10)), .5, 1.)


def test_select_requires_all_scales_and_keeps_incumbent_on_tie():
    es = [dict(label=l, cost=1.) for l in ['incumbent', 'scale_0', 'scale_1', 'scale_2', 'scale_3']]
    assert m.select(es) == 0
    es[-1]['cost'] = .25
    assert m.select(es) == 4
    with pytest.raises(ValueError): m.select(es[:-1])


def test_meter_rejects_scale_shortcut_before_dispatch():
    meter = m.Meter(MemoryStore())
    with pytest.raises(ValueError): meter.start(0, 'scale_0', 'whole_replay')
    assert not meter.events


def test_failed_attempt_no_refund_or_resume():
    meter = m.Meter(MemoryStore()); meter.start(0, 'incumbent', 'whole_replay')
    meter.begin('action_encoder')
    assert meter.counts()['action_encoder'] == dict(attempted=1, completed=0)
    with pytest.raises(ValueError): meter.start(0, 'incumbent', 'whole_replay')
    with pytest.raises(ValueError): meter.begin('predict')


def test_grad_enabled_or_AD_resource_never_dispatches():
    meter = m.Meter(MemoryStore(), no_grad=lambda: False); meter.start(0, 'incumbent', 'whole_replay')
    with pytest.raises(ValueError): meter.begin('action_encoder')
    meter.no_grad = lambda: True
    with pytest.raises(ValueError): meter.begin('jacobian')
    assert meter.events == []


def test_bad_normalized_history_not_accepted():
    r, t, _ = fixture_root(0); raw = np.tanh(np.asarray(r['incumbent_logits'], np.float32)).reshape(10, 2)
    path = path_for(t, raw); path[0]['action_history'][-1][0] += .1
    with pytest.raises(ValueError): m.check_path(t, raw, path)


def test_selected_replay_mismatch_preserves_already_paid_work():
    class Bad(Backend):
        def rollout(self, t, raw, meter, mode):
            contexts = super().rollout(t, raw, meter, mode)
            if mode == 'segmented_replay': contexts[-1]['latent_history'][-1][0] += 1
            return contexts
    store = MemoryStore(); meter = m.Meter(store)
    with pytest.raises(ValueError): m.diagnostic_root(*fixture_root(0), Bad(), meter, store)
    assert meter.counts()['predict'] == dict(attempted=16, completed=16)
    assert 'root_0.result.json' not in store.files


def test_journal_failure_before_packet_return_prevents_continuation():
    class Broken(MemoryStore):
        def event(self, e):
            if e['kind'] == 'native_packet': raise OSError('mock durable write failure')
            super().event(e)
    meter = m.Meter(Broken())
    with pytest.raises(OSError): m.diagnostic_root(*fixture_root(0), Backend(), meter, meter.storage)
    assert meter.counts()['predict'] == dict(attempted=1, completed=1) and meter.packet == 0


def test_real_saved_preconditions_are_small_and_no_stagnation_report_dependency():
    p = m.material()
    assert len(m.canonical(p)) < 512*1024
    assert p['scales'] == [.5, .25, .125, .0625]
    assert p['predict_cap'] == p['action_encoder_cap'] == 32
    assert p['new_AD_cap'] == p['new_image_cap'] == p['new_physics_cap'] == 0
    assert all(r['saved_gradient_dot_direction'] < 0 and len(r['candidates']) == 4 for r in p['roots'])
    assert not any('stagnation_saved' in path for path in p['bindings_sha256'])


def test_saved_iter8_not_iter8_incumbent_or_iter7_proposal():
    encoded = m.read(m.B8/'task_0.json.gz'); c = m.read(m.B8/'root_0_full_H2_GN/report.json.gz')
    import gzip, json
    events = [json.loads(line) for line in gzip.decompress((m.B8/'root_0_full_H2_GN/direct_journal.jsonl.gz').read_bytes()).splitlines()]
    result = m.extract(encoded['task'], c, events)
    assert result['incumbent_logits'] == c['direct_runs'][0][7]['incumbent']['logits']
    assert result['rejected_logits'] == c['direct_runs'][0][8]['proposal']['logits']
    c['direct_runs'][0][8]['accepted'] = True
    with pytest.raises(ValueError): m.extract(encoded['task'], c, events)
