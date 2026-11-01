"""Offline arithmetic and injected metadata fixtures only; no neural/physics calls."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import p4_multiround_outcomes_20261027_v1 as m


def rgb(step=0, visible=True):
    a = np.zeros((224,224,3),np.uint8)
    if visible:
        a[0,0] = [65,105,225]
    a[1,1] = step
    return a


def context():
    return dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=np.zeros((3,192)).tolist(),action_history=m.saved.normalize(np.zeros((2,5,2))).tolist())


class Env:
    def __init__(self, absent=None, throws=None, render_throws=None, terminate=None):
        self.i=0
        self.absent,self.throws,self.render_throws,self.terminate=absent,throws,render_throws,terminate

    def render(self):
        if self.i==self.render_throws:
            raise RuntimeError('fixture render failure')
        return rgb(self.i,self.i!=self.absent)

    def step(self, command):
        self.i+=1
        if self.i==self.throws:
            raise RuntimeError('fixture step failure')
        return None,0.,self.i==self.terminate,False,{}


class Backend:
    """Algebra witnesses, explicitly not a world model or real environment."""
    def __init__(self,emit):
        self.emit=emit

    def replay(self, c, commands, method):
        return dict(c,latent_history=c['latent_history'][1:]+[np.zeros(192).tolist()],
            action_history=c['action_history'][1:]+[m.saved.normalize(np.asarray(commands)).tolist()])

    def encode(self, frames, between, method):
        return dict(context(),action_history=m.saved.normalize(between).tolist())

    def compile(self, c, p, arm, boundary):
        nominal=np.zeros(192);jac=np.zeros((192,10))
        bank=m.runtime.compiler.packet.spectral_bank(jac,nominal,p['goal'],p['q'])
        candidates=[]
        for i,params in enumerate(bank['params']):
            u=np.asarray(params,np.float32).astype(float)
            candidates.append(dict(candidate_id=f'suffix_{i}',kind=bank['kinds'][i],suffix_params=u.tolist(),
                commands=p['prefix']+np.tanh(u).reshape(5,2).tolist(),predicted_terminal=nominal.tolist(),
                nonlinear_terminal_cost=0.))
        plan=dict(root_sha256=m.runtime.ROOT,root_index=1,role='selected',arm=arm,prefix=p['prefix'],
            context=c,goal=p['goal'],q=p['q'],nominal_terminal=nominal.tolist(),jacobian=jac.tolist(),
            candidates=candidates,selected_candidate_id='suffix_0',independent_replay_exact=True)
        self.emit(dict(kind='derivative_receipt',method=f'{arm}_{boundary}',arm=arm,
            nominal_terminal=nominal.tolist(),jacobian=jac.tolist()))
        for candidate in candidates:
            self.emit(dict(kind='candidate',method=f'{arm}_{boundary}',arm=arm,candidate=candidate))
        return plan,self.replay(c,candidates[0]['commands'][5:],'fixture')


@pytest.fixture
def make_branch(tmp_path):
    def make(arm='actual',**env_args):
        folder=tmp_path/str(len(list(tmp_path.iterdir())))
        folder.mkdir()
        prefix=folder/'prefix.png';Image.fromarray(rgb(5)).save(prefix)
        p=dict(prefix=np.zeros((5,2)).tolist(),prefix_rgb=str(prefix),between_raw=np.zeros((2,5,2)).tolist(),
            initial_context=context(),saved_predicted_prefix=np.zeros(192).tolist(),
            goal=np.zeros(192).tolist(),q=np.eye(192).tolist(),physical_goal=np.zeros(7).tolist())
        frames=np.stack([rgb(),rgb(),rgb()]);slots=m.runtime.roster();cost=m.runtime.new_cost()
        env=Env(**env_args);j=m.runtime.Journal(folder)
        def reset():
            j.emit(dict(kind='reset_identity',arm=arm,goal=p['physical_goal'],history=np.zeros((3,2)).tolist(),
                identity=dict(history_rgb_sha256=m.saved.rgb_sha(frames))))
            return frames,np.zeros((2,5,2))
        row=m.runtime.arm_loop(p,arm,env=env,reset=reset,state_of=lambda e:np.zeros(7),backend=Backend(j.emit),
            emit=j.emit,slots=slots,cost=cost)
        j.close()
        events=[m.saved.strict_json(line) for line in (folder/'runtime.events.jsonl').read_bytes().splitlines()]
        return p,row,events,slots,frames
    return make


@pytest.mark.parametrize('arm',m.runtime.ARMS)
def test_complete20_saved_arithmetic_and_history(make_branch,arm):
    p,row,events,slots,frames=make_branch(arm=arm)
    value=m.branch(m.saved.Inputs(),p,row,events,slots,frames)
    assert value['classification']=='complete20' and value['exact20']['primitive_step']==20
    assert len(value['steps'])==21 and len(value['boundaries'])==3
    assert value['exact20']['official']['official_success']
    assert value['exact20']['pose']['pose_cost']==0
    assert value['selected_attempts']==value['selected_returns']==20


@pytest.mark.parametrize('step',[3,5,6,11,16,20])
def test_guard_partial_not_integrity_or_exact20(make_branch,step):
    p,row,events,slots,frames=make_branch(absent=step)
    value=m.branch(m.saved.Inputs(),p,row,events,slots,frames)
    assert value['classification']=='guard_aborted'
    assert value['last_observed']['primitive_step']==step and value['exact20'] is None
    assert value['observed_at_step20']==(step==20)
    assert not value['full_horizon_completed']


def test_absent_initial_rgb_cannot_override_sealed_root_identity(make_branch):
    p,row,events,slots,frames=make_branch(absent=0)
    with pytest.raises(ValueError,match='initial RGB differs'):
        m.branch(m.saved.Inputs(),p,row,events,slots,frames)


@pytest.mark.parametrize('kwargs,classification,last,unknown',[
    (dict(throws=7),'execution_or_integrity_failure',6,True),
    (dict(render_throws=7),'execution_or_integrity_failure',6,True),
    (dict(terminate=7),'environment_terminated',7,False)])
def test_other_partial_endpoint_classification(make_branch,kwargs,classification,last,unknown):
    p,row,events,slots,frames=make_branch(**kwargs)
    value=m.branch(m.saved.Inputs(),p,row,events,slots,frames)
    assert value['classification']==classification and value['exact20'] is None
    assert value['last_observed']['primitive_step']==last
    assert value['endpoint_unobserved_after_attempt']==unknown


@pytest.mark.parametrize('mutation',['pixel_sha','image_sha','root','count','command','history_rgb','raw_between',
    'normalization','chosen','predicted_shift','metric','selection_sha','successor','candidate','dispatch_order','guard_count','final_row',
    'derivative','candidate_log'])
def test_tampered_evidence_fails_closed(make_branch,mutation):
    p,row,events,slots,frames=make_branch()
    observation=next(e for e in events if e['kind']=='observation')
    boundary=next(e for e in events if e['kind']=='boundary_begin')
    c=next(e for e in events if e['kind']=='planning_context')
    s=next(e for e in events if e['kind']=='selection')
    if mutation=='pixel_sha': observation['pixel_sha256']='0'*64
    if mutation=='image_sha': observation['rgb_sha256']='0'*64
    if mutation=='root': row['root_sha256']='other'
    if mutation=='count': row['selected_returns']-=1
    if mutation=='command': next(e for e in events if e['kind']=='step_attempt')['command'][0]=.2
    if mutation=='history_rgb': boundary['frames_pixel_sha256'].reverse()
    if mutation=='raw_between': boundary['between_raw'][0][0][0]=.2
    if mutation=='normalization': c['chosen_context']['action_history'][0][0]=1.
    if mutation=='chosen': c['chosen_context']=context();c['chosen_context']['latent_history'][0][0]=10.
    if mutation=='predicted_shift': c['predicted_context']['latent_history'][0][0]=10.
    if mutation=='metric': s['compiler_plan']['q'][0][0]=2.
    if mutation=='selection_sha': s['selection_sha256']='0'*64
    if mutation=='successor': s['successor']['latent_history'][0][0]=1.
    if mutation=='candidate': s['compiler_plan']['selected_candidate_id']='suffix_1'
    if mutation=='dispatch_order': s['event_index']=100000
    if mutation=='guard_count': row['guard']['observations'][-1]['canonical_agent_pixels']+=1
    if mutation=='final_row': next(e for e in events if e['kind']=='arm_complete')['row']['selected_returns']=0
    if mutation=='derivative': next(e for e in events if e['kind']=='derivative_receipt')['jacobian'][0][0]=1.
    if mutation=='candidate_log': next(e for e in events if e['kind']=='candidate')['candidate']['kind']='other'
    with pytest.raises(ValueError): m.branch(m.saved.Inputs(),p,row,events,slots,frames)


def test_official_and_pose_are_distinct_unchanged_metrics():
    goal=np.zeros(7);state=np.zeros(7);state[0]=30;state[5]=40;state[4]=2*np.pi-.1
    v=m.label(state,goal)
    assert v['pose']['xy_distance']==0 and v['pose']['angle_distance']==pytest.approx(.1)
    assert v['pose']['pose_cost']==pytest.approx(2.)
    assert v['official']['official_position_error']==30
    assert v['official']['official_state_distance']>50
    assert not v['official']['official_success'] and v['official']['object_only_success']


@pytest.mark.parametrize('value',[None,[],[float('nan')]*7,[True]*7])
def test_nonfinite_or_wrong_state_rejected(value):
    with pytest.raises(ValueError):m.label(value,np.zeros(7))


def test_partial_never_enters_exact20_pair(make_branch):
    rows=[]
    for arm,kwargs in [('actual',dict(absent=20)),('imagined',{})]:
        p,row,events,slots,frames=make_branch(arm=arm,**kwargs)
        rows.append(m.branch(m.saved.Inputs(),p,row,events,slots,frames))
    result=m.compare(rows)
    assert not result['exact20_comparable'] and result['actual_minus_imagined'] is None
    assert not result['partial_endpoints_pooled']


@pytest.mark.parametrize('terminated,truncated',[(True,False),(False,True)])
def test_environment_termination_with_failure_none(terminated,truncated):
    row=dict(status='aborted',failure=None,guard=dict(abstained=False))
    observed=[dict(primitive_step=7,terminated=terminated,truncated=truncated)]
    assert m.classify(row,observed,[],[])=='environment_terminated'


def test_nonfull_without_failure_object_is_not_an_exception():
    row=dict(status='aborted',failure=None,guard=dict(abstained=False))
    assert m.classify(row,[],[],[])=='execution_or_integrity_failure'


def test_blocked_slot_retained(make_branch):
    p,row,events,slots,frames=make_branch()
    row=dict(arm='actual',status='blocked',root_sha256=m.runtime.ROOT)
    value=m.branch(m.saved.Inputs(),p,row,[],m.runtime.roster(),frames)
    assert value['classification']=='blocked' and value['exact20'] is None and value['last_observed'] is None


def test_original_metric_sources_are_read_only():
    inputs=m.saved.Inputs()
    result=m.metric_sources(inputs)
    assert result['official_source_sha256']==m.ENV_SHA and not result['retuned']
    assert result['neural_calls']==result['simulation_calls']==0
    assert inputs.stable()


def test_actual_saved_report_arithmetic_no_run_calls(monkeypatch):
    if not m.DEFAULT_REPORT.exists():
        pytest.skip('actual runtime report not yet present')
    def forbidden(*args,**kwargs):
        pytest.fail('runtime prepare/run must never be called by arithmetic consumer')
    monkeypatch.setattr(m.runtime,'run',forbidden)
    monkeypatch.setattr(m.runtime,'prepare',forbidden)
    value=m.audit(m.DEFAULT_REPORT)
    assert value['status']=='saved_arithmetic_audit_passed' and value['inputs_unchanged']
    assert value['neural_calls']==value['physics_calls']==0
    assert len(value['rows'])==2 and not value['P4_admission']


def test_output_exclusive_before_analysis(tmp_path):
    with pytest.raises(ValueError):m.run(tmp_path)
