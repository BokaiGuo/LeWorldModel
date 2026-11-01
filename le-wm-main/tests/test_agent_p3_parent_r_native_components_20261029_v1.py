"""RAM-only callback binding, normalization, ledger and failure tests."""
from copy import deepcopy
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_parent_r_native_components_20261029_v1 as m
from experiments.spectral_reachability import agent_p3_local_signal_cuda_20261028_v1 as signal


def task():
    c = dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,
        action_dim=10,latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())
    t = dict(root_context=c,goal=[1.]+[0.]*191,projection=np.eye(192).tolist(),action_mean=[.125,-.25],action_std=[2.,4.])
    t['binding']={k:m.digest(k) for k in m.core.BINDINGS}
    t['binding'].update(goal_sha256=m.digest(t['goal']),projection_sha256=m.digest(t['projection']),
        normalizer_sha256=m.digest([t['action_mean'],t['action_std']]),context_sha256=m.digest(c),
        action_domain_sha256=m.digest(m.core.action_domain()))
    return t


class RAM:
    kind='RAM_FIXTURE'
    def __init__(self,t,sink,mode=None):
        self.t,self.sink,self.mode=deepcopy(t),sink,mode

    def forward(self,c,raw,ledger):
        raw=np.asarray(raw,np.float32)
        token=((raw-np.asarray(self.t['action_mean'],np.float32))/np.asarray(self.t['action_std'],np.float32)).reshape(10)
        ledger.call('action_encoder',lambda:None,shape=[1,3,10])
        def pred():
            ledger.call('predictor',lambda:None,shape=[1,3,192])
            ledger.call('pred_proj',lambda:None,shape=[3,192])
        ledger.call('predict',pred,shape=[1,3,192])
        v=np.asarray(c['latent_history'][-1],np.float32).copy()
        v[0]+=np.float32(1)+raw.sum(dtype=np.float32)
        return dict(c,latent_history=c['latent_history'][1:]+[v.tolist()],
            action_history=[c['action_history'][-1],token.tolist()])

    def replay(self,c,raw,segmented,ledger):
        if self.mode=='raise':raise RuntimeError('RAM failure')
        out=self.forward(c,raw,ledger)
        if self.mode=='wrongpath':out['action_history'][-1][0]=22.
        return [out]

    def signal(self,req,ledger):
        assert req==m.core.request(self.t,dict(context=self.t['root_context']),operation='compile_finite_horizon_J',horizon=1)
        c=self.forward(self.t['root_context'],np.zeros((5,2),np.float32),ledger)
        b=dict(kind='boundary',phase='nominal',step=0,arrays=dict(
            latent=signal.blob(np.asarray(c['latent_history'],np.float32)[None]),
            history=signal.blob(np.asarray(c['action_history'],np.float32)[None])))
        if self.mode=='phase':b['phase']='AD'
        if self.mode=='boundary':b['arrays']['history']=signal.blob(np.zeros((1,2,10),np.float32))
        self.sink.event(b)
        if self.mode=='duplicate':self.sink.event(b)
        def jac():
            self.forward(self.t['root_context'],np.zeros((5,2),np.float32),ledger)
            for _ in range(192):
                def grad():
                    return ledger.call('grad',lambda:ledger.call('raw_backward',lambda:None))
                ledger.call('vjp',grad,directions=1)
        if self.mode!='missingJ':ledger.call('jacobian',jac)
        j=np.zeros((192,10),np.float32)
        j[0]=np.float32(self.t['root_context']['latent_history'][-1][0]+1)
        out=dict(binding_sha256=req['binding_sha256'],parent_context_sha256=req['parent_context_sha256'],
            horizon=1,nominal=c['latent_history'][-1],jacobian=j.tolist(),producer_sha256=m.digest('RAM_fixture'))
        if self.mode=='identity':out['parent_context_sha256']=m.digest('wrong')
        if self.mode=='nominal':out['nominal']=deepcopy(out['nominal']);out['nominal'][0]+=1.
        return out


def component(mode=None,caps=None,reserve=None):
    cap={k:m.expected('postcontext_H1')[k]+m.expected('bounded_suffix_H1')[k] for k in m.core.AXES}
    made=[];events=[]
    def factory(t,s):
        made.append(deepcopy(t))
        return RAM(t,s,mode)
    obj=m.Components.ram(root=task(),caps=cap if caps is None else caps,
        verification_reserve=dict.fromkeys(m.core.AXES,0) if reserve is None else reserve,
        factory=factory,sink=events.append)
    return obj,made,events


def test_real_postcontext_request_nominal_boundary_without_extra_predict_and_suffix():
    obj,made,events=component()
    # Explicit prefix fixture; no prior neural model invocation.
    t=task();c=deepcopy(t['root_context']);raw=np.full((5,2),.25,np.float32).tolist()
    after=dict(c,latent_history=c['latent_history'][1:]+[[7.]+[0.]*191],
        action_history=[c['action_history'][-1],((np.asarray(raw,np.float32)-np.asarray(t['action_mean'],np.float32))/
            np.asarray(t['action_std'],np.float32)).reshape(10).tolist()])
    h=obj.h1(prefix_actions=raw,prefix_path=[after])
    assert h['status']=='complete' and h['native_closure_verified'] is False
    assert made[0]['root_context']==after
    assert h['result']['input_context_sha256']==m.digest(after)
    assert h['result']['nominal_path'][0]['latent_history'][-1][0]==8.
    assert h['result']['jacobian'][0]==[8.]*10
    assert h['resources']==m.expected('postcontext_H1')
    assert h['resources']['predict_API']==2
    ptr=h['nominal_boundary_pointer']
    assert m.digest(h['native_transcript'][ptr['transcript_index']])==ptr['boundary_sha256']
    params=np.linspace(-3,3,10).tolist()
    s=obj.suffix(prefix_actions=raw,prefix_path=[after],params=params)
    expected=np.tanh(np.asarray(params,np.float64)).astype(np.float32).reshape(5,2)
    assert s['raw_float32']==expected.tolist()
    actual_token=((expected-np.asarray(t['action_mean'],np.float32))/np.asarray(t['action_std'],np.float32)).reshape(10)
    assert s['result']['path'][0]['action_history'][-1]==actual_token.tolist()
    assert s['resources']==m.expected('bounded_suffix_H1')
    assert len(obj.calls)==2 and all(e['call_id'] in (0,1) for e in events)
    json.dumps(obj.calls,allow_nan=False)


@pytest.mark.parametrize('mode',['phase','boundary','duplicate','missingJ','identity','nominal'])
def test_saved_boundary_identity_and_resource_mutations_fail_closed(mode):
    obj,_,_=component(mode)
    with pytest.raises(m.ComponentFailure) as err:
        obj.h1(prefix_actions=[],prefix_path=[])
    r=err.value.receipt
    assert r['status']=='failed' and r['result'] is None
    assert r['resources']['signal_compile_calls']==1 and r['events']
    with pytest.raises(ValueError,match='fail-stop'):
        obj.h1(prefix_actions=[],prefix_path=[])


@pytest.mark.parametrize('operation',['postcontext_H1','bounded_suffix_H1'])
@pytest.mark.parametrize('axis',m.core.AXES)
def test_each_resource_reserve_axis_atomic_before_factory(operation,axis):
    caps=m.expected(operation);reserve=dict.fromkeys(m.core.AXES,0);reserve[axis]=1
    obj,made,_=component(caps=caps,reserve=reserve)
    with pytest.raises(m.ComponentFailure) as err:
        if operation=='postcontext_H1':obj.h1(prefix_actions=[],prefix_path=[])
        else:obj.suffix(prefix_actions=[],prefix_path=[],params=[0.]*10)
    assert made==[] and not any(err.value.receipt['resources'].values())


def test_failure_prefix_nonrefundable_native_events():
    obj,_,_=component('raise')
    with pytest.raises(m.ComponentFailure) as err:
        obj.suffix(prefix_actions=[],prefix_path=[],params=[0.]*10)
    r=err.value.receipt
    assert r['resources']['replay_calls']==1
    assert any(e['payload']['kind']=='native_call_failure' for e in r['events'])


@pytest.mark.parametrize('params',[[0.]*9,[float('nan')]*10,[True]*10])
def test_bad_suffix_before_dispatch(params):
    obj,made,_=component()
    with pytest.raises(ValueError):obj.suffix(prefix_actions=[],prefix_path=[],params=params)
    assert not made and not obj.events


def test_float64_tanh_not_float32_tanh_and_no_normalization_in_raw_helper():
    p=np.linspace(-.9,.9,10)
    assert m.suffix_raw(p.tolist())==np.tanh(p).astype(np.float32).reshape(5,2).tolist()
    assert m.suffix_raw([0.]*10)==[[0.,0.]]*5


def test_host_requires_main_authorization_and_cuda_no_model_touch():
    kwargs=dict(root=task(),caps=m.expected('postcontext_H1'),verification_reserve=dict.fromkeys(m.core.AXES,0),
        model=object(),device=SimpleNamespace(type='cpu'),sink=lambda e:None)
    with pytest.raises(ValueError,match='authorization'):m.Components.host(**kwargs)
    with pytest.raises(ValueError,match='CUDA'):m.Components.host(**kwargs,main_authorized=True,guard=lambda:None)
    with pytest.raises(TypeError):m.Components()


def test_ram_cannot_impersonate_host_backend():
    obj,_,_=component()
    obj.factory=lambda t,s:SimpleNamespace(kind='HOST_CUDA')
    with pytest.raises(m.ComponentFailure,match='kind mismatch'):
        obj.h1(prefix_actions=[],prefix_path=[])


def test_root_history_mutation_refused_before_callback():
    obj,made,_=component()
    with pytest.raises(ValueError):obj.h1(prefix_actions=[],prefix_path=[obj.root['root_context']])
    assert not made


def test_immutable_root_and_no_borrowed_parent_context():
    obj,made,_=component()
    obj.root['goal'][0]=2.
    with pytest.raises(ValueError,match='immutable'):
        obj.h1(prefix_actions=[],prefix_path=[])
    assert made==[]


def test_host_construction_is_lazy_guard_stops_before_native_model():
    touched=[]
    class NoModel:
        def __getattr__(self,name):
            touched.append(name)
            raise AssertionError('model must not be touched')
    def guard():raise ValueError('MAIN protocol not released')
    obj=m.Components.host(root=task(),caps=m.expected('postcontext_H1'),
        verification_reserve=dict.fromkeys(m.core.AXES,0),model=NoModel(),
        device=SimpleNamespace(type='cuda'),sink=lambda e:None,main_authorized=True,guard=guard)
    assert obj.kind=='HOST_CUDA' and touched==[]
    with pytest.raises(m.ComponentFailure,match='protocol not released') as err:
        obj.h1(prefix_actions=[],prefix_path=[])
    assert touched==[] and not any(err.value.receipt['resources'].values())


def test_cold_import_and_contract_no_neural_foreign_load():
    code='''
import sys
from experiments.spectral_reachability import agent_p3_parent_r_native_components_20261029_v1 as m
assert 'torch' not in sys.modules and 'gymnasium' not in sys.modules
assert not m.contract()['production_runner']
m.check_sources()
'''
    subprocess.run([sys.executable,'-c',code],check=True)
    r=subprocess.run([sys.executable,'-m',m.__name__,'--contract'],check=True,capture_output=True,text=True)
    assert not json.loads(r.stdout)['bank_selected']
