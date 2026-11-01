"""Non-neural orchestration/algebra fixtures; no CUDA, physics or efficacy claim."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import p4_multiround_feedback_runtime_20261027_v1 as m


def rgb(step=0, visible=True):
    value = np.zeros((224,224,3), np.uint8)
    if visible:
        value[0,0] = [65,105,225]
    value[1,1] = step
    return value


def context(value=0):
    return dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192,
        action_dim=10, latent_history=np.full((3,192), value, dtype=float).tolist(),
        action_history=np.zeros((2,10)).tolist())


class Env:
    def __init__(self, absent=None, throws=None, terminates=None, render_throws=None):
        self.step_i = 0
        self.absent, self.throws, self.terminates, self.render_throws = absent, throws, terminates, render_throws

    def step(self, command):
        self.step_i += 1
        if self.step_i == self.throws:
            raise RuntimeError('injected step failure')
        return None, 0., self.step_i==self.terminates, False, {}

    def render(self):
        if self.step_i == self.render_throws:
            raise RuntimeError('injected render failure')
        return rgb(self.step_i, self.step_i!=self.absent)


class Backend:
    """Deterministic algebra witness generator, NOT a neural or physics substitute."""
    def __init__(self, encode_throws=None, compile_throws=None, tamper=None):
        self.contexts, self.encodes = [], []
        self.encode_throws, self.compile_throws, self.tamper = encode_throws, compile_throws, tamper

    def replay(self, c, commands, method):
        after = deepcopy(c)
        after['latent_history'] = c['latent_history'][1:]+[np.full(192,5.).tolist()]
        after['action_history'] = c['action_history'][1:]+[np.zeros(10).tolist()]
        return after

    def encode(self, frames, between, method):
        if self.encode_throws and method.endswith(str(self.encode_throws)):
            raise RuntimeError('injected encode failure')
        self.encodes.append((np.array(frames), np.array(between)))
        c = context()
        c['latent_history'] = [np.full(192, float(f[1,1,0])+100).tolist() for f in frames]
        return c

    def compile(self, c, p, arm, boundary):
        if self.compile_throws == boundary:
            raise RuntimeError('injected compiler failure')
        self.contexts.append(deepcopy(c))
        nominal, jac = np.zeros(192), np.zeros((192,10))
        bank = m.compiler.packet.spectral_bank(jac, nominal, p['goal'], p['q'])
        candidates = []
        for i, params in enumerate(bank['params']):
            u = np.asarray(params, np.float32).astype(float)
            candidates.append(dict(candidate_id=f'suffix_{i}', kind=bank['kinds'][i], suffix_params=u.tolist(),
                commands=p['prefix']+np.tanh(u).reshape(5,2).tolist(), predicted_terminal=nominal.tolist(),
                nonlinear_terminal_cost=0.))
        plan = dict(root_sha256=m.ROOT, root_index=1, role='selected', arm=arm, prefix=p['prefix'],
            context=deepcopy(c), goal=p['goal'], q=p['q'], nominal_terminal=nominal.tolist(),
            jacobian=jac.tolist(), candidates=candidates, selected_candidate_id='suffix_0', independent_replay_exact=True)
        after = deepcopy(c)
        after['latent_history'] = c['latent_history'][1:]+[nominal.tolist()]
        after['action_history'] = c['action_history'][1:]+[np.zeros(10).tolist()]
        if self.tamper == 'winner':
            plan['selected_candidate_id'] = 'suffix_1'
        elif self.tamper == 'metric':
            plan['q'][0][0] = 2.
        elif self.tamper == 'history':
            after['latent_history'][0][0] += 1
        return plan, after


@pytest.fixture
def protocol(tmp_path):
    path = tmp_path/'prefix.png'
    Image.fromarray(rgb(5)).save(path)
    return dict(prefix=np.zeros((5,2)).tolist(), prefix_rgb=str(path), initial_context=context(),
        saved_predicted_prefix=np.full(192,5.).tolist(), goal=np.zeros(192).tolist(), q=np.eye(192).tolist())


def run_fixture(protocol, env, backend, arm='actual', emit=None):
    slots, cost, events = m.roster(), m.new_cost(), []
    def event(value, image=None):
        events.append(deepcopy(value))
        if emit:
            emit(value, image)
    row = m.arm_loop(protocol, arm, env=env,
        reset=lambda: (np.stack([rgb(),rgb(),rgb()]), np.zeros((2,5,2))),
        state_of=lambda e: np.zeros(7), backend=backend, emit=event, slots=slots, cost=cost)
    return row, slots, cost, events


@pytest.mark.parametrize('arm', m.ARMS)
def test_exact_three_packets_and_pre_dispatch_selection(protocol, arm):
    env, backend = Env(), Backend()
    row, slots, cost, events = run_fixture(protocol, env, backend, arm)
    assert row['status']=='completed' and row['completed_packets']==4 and env.step_i==20
    assert cost['selected_attempts']==cost['selected_returns']==20
    assert [e['boundary'] for e in events if e['kind']=='selection']==[5,10,15]
    assert [s['status'] for s in slots if s['arm']==arm]==['completed']*3
    assert [s['status'] for s in slots if s['arm']!=arm]==['blocked_not_reached']*3
    for b in m.BOUNDARIES:
        selection = next(i for i,e in enumerate(events) if e['kind']=='selection' and e['boundary']==b)
        dispatch = next(i for i,e in enumerate(events) if e['kind']=='step_attempt' and e['primitive_step']==b+1)
        assert selection < dispatch
        event = events[selection]
        value = {k:v for k,v in event.items() if k not in ('kind','arm','selection_sha256')}
        value['arm'] = arm
        assert m.content_sha(value)==event['selection_sha256']
    for i,(frames,between) in enumerate(backend.encodes):
        assert [int(f[1,1,0]) for f in frames]==[[0,0,5],[0,5,10],[5,10,15]][i]
        assert between.shape==(2,5,2)


def test_imagined_control_never_uses_observed_latents(protocol):
    control, actual = Backend(), Backend()
    run_fixture(protocol, Env(), control, 'imagined')
    run_fixture(protocol, Env(), actual, 'actual')
    assert control.contexts[0]['latent_history'][-1][0]==5
    assert actual.contexts[0]['latent_history'][-1][0]==105
    assert control.contexts[1]['latent_history'][-1][0]==0
    assert actual.contexts[1]['latent_history'][-1][0]==110
    assert len(control.encodes)==len(actual.encodes)==3


@pytest.mark.parametrize('absent,attempts,packets', [(0,0,0),(3,3,0),(6,6,1),(11,11,2),(16,16,3),(20,20,3)])
def test_guard_no_blind_or_shortened_endpoint(protocol, absent, attempts, packets):
    backend, env = Backend(), Env(absent=absent)
    row, slots, cost, events = run_fixture(protocol, env, backend)
    assert row['status']=='aborted' and row['guard']['abstained']
    assert cost['selected_returns']==attempts and env.step_i==attempts
    assert row['completed_packets']==packets
    assert not any(e['kind']=='packet_complete' and e['next_boundary']>attempts for e in events)
    m.census(slots)


@pytest.mark.parametrize('failure', ['step','render','termination','encode','compile','prefix'])
def test_partial_failures_durable_without_retry(protocol, tmp_path, failure):
    env = Env(throws=7 if failure=='step' else None, render_throws=7 if failure=='render' else None,
        terminates=7 if failure=='termination' else None)
    backend = Backend(encode_throws=10 if failure=='encode' else None, compile_throws=10 if failure=='compile' else None)
    if failure=='prefix':
        Image.fromarray(rgb(99)).save(protocol['prefix_rgb'])
    output = tmp_path/'journal'; output.mkdir()
    journal = m.Journal(output)
    row, slots, cost, events = run_fixture(protocol, env, backend, emit=journal.emit)
    journal.close()
    assert row['status']=='aborted' and env.step_i<20
    saved = [json.loads(line) for line in (output/'runtime.events.jsonl').read_text().splitlines()]
    assert saved[-1]['kind']=='arm_complete'
    assert saved[-1]['row']['status']=='aborted'
    assert [v['event_index'] for v in saved]==list(range(len(saved)))
    for v in saved:
        if 'rgb_path' in v:
            assert m.sha(v['rgb_path'])==v['rgb_sha256']
    if failure=='step':
        assert cost['selected_attempts']==7 and cost['selected_returns']==6
    if failure=='render':
        assert cost['selected_attempts']==cost['selected_returns']==7
    if failure=='prefix':
        assert not backend.encodes and env.step_i==5
    m.census(slots)


@pytest.mark.parametrize('tamper', ['winner','metric','history'])
def test_bad_witness_never_dispatched(protocol, tamper):
    env = Env()
    row, slots, cost, events = run_fixture(protocol, env, Backend(tamper=tamper))
    assert row['status']=='aborted' and env.step_i==5
    assert not any(e['kind']=='selection' for e in events)


@pytest.mark.parametrize('mutation', ['missing','duplicate','order','root'])
def test_census_retains_exact_six_slots(mutation):
    slots = m.roster()
    if mutation=='missing': slots.pop()
    if mutation=='duplicate': slots[-1]=deepcopy(slots[0])
    if mutation=='order': slots.reverse()
    if mutation=='root': slots[0]['root_sha256']='other'
    with pytest.raises(ValueError): m.census(slots)


def test_fixed_schedule_and_cost_contract():
    c = m.contract()
    assert c['scheduled_predict_samples']==2+6*(2+10+1)==80
    assert c['scheduled_selected_steps']==2*(5+3*5)==40
    assert c['scheduled_encode_frames']==2*3*3==18
    assert not c['outcome_gating'] and not c['P1_admission'] and not c['P4_admission']
    assert c['boundaries']==[5,10,15] and not c['total_compute_matched']


def test_journal_exclusive_and_fsync(tmp_path, monkeypatch):
    calls=[]
    real=m.os.fsync
    monkeypatch.setattr(m.os, 'fsync', lambda fd: (calls.append(fd), real(fd))[1])
    j=m.Journal(tmp_path)
    j.emit(dict(kind='test'),rgb())
    j.close()
    assert len(calls)>=4
    with pytest.raises(FileExistsError): m.Journal(tmp_path)


def test_refuse_existing_prepare_before_import_or_runtime(tmp_path):
    with pytest.raises(ValueError): m.prepare(tmp_path)


@pytest.mark.parametrize('mutation', ['latent','action','terminal'])
def test_explicit_shift_tampering(mutation):
    before=context()
    after=deepcopy(before)
    terminal=np.zeros(192).tolist()
    if mutation=='latent': after['latent_history'][0][0]=1
    if mutation=='action': after['action_history'][0][0]=1
    if mutation=='terminal': terminal[0]=1
    with pytest.raises(ValueError): m.validate_transition(before,after,terminal)
