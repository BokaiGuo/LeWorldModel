"""RAM-only raw forward/model-interface doubles; no torch or physics."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_midfeedback_raw_jacobian_fd_20261028_v1 as m
import test_agent_p2_contextual_midfeedback_recompile_20261028_v2 as prior
import test_agent_p2_midfeedback_terminal_rgb_20261028_v1 as tensor


Memory=prior.Memory


def point():
    b=prior.branch();j=np.zeros((192,10),np.float64);j[:10]=np.eye(10)*.5
    raw=np.asarray(b['old_suffix_commands'],np.float32)
    z=(np.asarray(b['observed_context']['latent_history'][-1])+j@raw.reshape(10)).astype(np.float32)
    incoming=m.F.planner.incoming(b,b['observed_context'],raw)
    after=dict(b['observed_context'],latent_history=b['observed_context']['latent_history'][1:]+[z.astype(float).tolist()],action_history=incoming['action_history'][-2:])
    return dict(index=0,root_slot=0,condition='off',arm='joint80',task_id='a'*64,branch_id=b['branch_id'],
        context=b['observed_context'],context_sha256=m.F.core.digest(b['observed_context']),raw5=b['old_suffix_commands'],
        raw_history10=b['observed_raw_history'],goal=b['goal'],action_mean=b['action_mean'],action_std=b['action_std'],
        anchor_context=after,anchor_context_sha256=m.F.core.digest(after),jacobian=j.tolist(),jacobian_sha256='b'*64)


class Backend:
    """Actual new API wrapper + actual module hooks; pure NumPy stand-in."""
    def __init__(self,p,modules,meter,curvature=0):
        self.p,self.modules,self.meter,self.curvature=p,modules,meter,curvature
        self.model=NS(predict=self.predict)
        self.j=np.zeros((192,10),np.float64);self.j[:10]=np.eye(10)*.5
    def module(self,kind,inputs,out):
        args=tuple(prior.core.Tensor(a) for a in inputs)
        self.modules.pre(kind)(kind,args);self.modules.post(kind,args,prior.core.Tensor(out))
    def predict(self,lat,ae,z):
        out=lat.copy();out[0,-1]=z
        self.module('predict',[lat,ae],out);self.module('pred_proj',[out.reshape(3,192)],out.reshape(3,192));return z
    def step(self,c,commands):
        incoming=m.F.planner.incoming(m.task(self.p),c,commands)
        d=np.asarray(commands,np.float64).reshape(10)-np.asarray(self.p['raw5']).reshape(10)
        z=np.asarray(self.p['anchor_context']['latent_history'][-1])+self.j@d
        z[:10]+=self.curvature*d*d;z=z.astype(np.float32)
        ae=np.asarray(incoming['action_history'],np.float32)[None];lat=np.asarray(incoming['latent_history'],np.float32)[None]
        self.module('action_encoder',[ae],ae);self.model.predict(lat,ae,z)
        return dict(c,latent_history=deepcopy(c['latent_history'][1:])+[z.astype(float).tolist()],action_history=incoming['action_history'][-2:])


def execute(p=None,curvature=0):
    p=p or point();s=Memory();schedule=m.requests(p);meter=NS(stage=None,expected_input=None,last_native=None)
    modules=m.Modules(NS(),s,'point0',meter,schedule);backend=Backend(p,modules,meter,curvature)
    api=m.PredictAPI(backend.model,meter,s,lambda:None,schedule);r=dict(index=p['index'])
    with api:m.evaluate(p,schedule,backend,meter,s,r)
    r.update(module_ledger=deepcopy(modules.entries),predict_API_ledger=deepcopy(api.entries),AD_API_ledger=[],resources=modules.counts(),
        predict_API_resources=api.counts(),AD_API_resources={k:dict(attempted=0,completed=0) for k in ('grad','jacobian')})
    return p,schedule,r,s


@pytest.fixture(scope='module')
def replay():return execute()


def test_all_raw_coordinates_steps192_outputs_and_exact_native_ABI(replay):
    p,req,r,s=replay;result=m.check_point(p,req,r,s.events,s.load)
    assert len(req)==60 and len(r['module_ledger'])==180
    assert r['resources']['predict']==dict(attempted=60,completed=60)
    assert [v['h'] for v in result['steps']]==[.01,.001,.0001]
    for group in result['steps']:
        assert len(group['columns'])==10 and group['full_matrix_available']
        assert np.asarray(group['nominal_FD_matrix']).shape==(192,10)
        assert all(len(col['AD'])==len(col['nominal_fd'])==192 for col in group['columns'])
        assert group['nominal_matrix_metrics']['direction_cosine']>.999


def test_float32_raw_perturbation_not_tanh_and_actual_spacing():
    p=point();schedule=m.requests(p);anchor=np.asarray(p['raw5'],np.float32).reshape(10)
    for request in schedule:
        a=anchor.copy();c=request['coordinate'];a[c]=np.float32(float(a[c])+request['sign']*request['h'])
        assert request['raw_bits']==a.astype('<f4').tobytes().hex()
        assert request['actual_delta']==float(a[c])-float(anchor[c])
        assert request['commands']!=np.tanh(a).reshape(5,2).astype(float).tolist()


def test_out_of_domain_probes_null_no_clamp_or_subset_matrix():
    p=point();p['raw5'][0][0]=float(np.float32(.9999))
    # Set a consistent RAM anchor, not any real sourcepoint or scientific result.
    p,req,r,s=execute(p);result=m.check_point(p,req,r,s.events,s.load)
    assert len(r['evaluations'])==60
    assert any(not x['legal'] and abs(x['commands'][0][0])>1 for x in req)
    assert r['resources']['predict']['completed']<60
    for group in result['steps'][:2]:
        assert group['columns'][0]['missing'] and group['columns'][0]['nominal_fd'] is None
        assert group['nominal_matrix_metrics'] is group['nominal_FD_matrix'] is None
        assert group['columns'][1]['nominal_fd'] is not None


def test_correct_local_AD_with_quadratic_response_distinguished_from_wrong_sign():
    p,req,r,s=execute(curvature=2.)
    good=m.check_point(p,req,r,s.events,s.load)
    assert good['steps'][0]['nominal_matrix_metrics']['direction_cosine']>.999
    assert good['steps'][0]['columns'][0]['curvature']['second_difference_l2']>3.9
    p=deepcopy(p);p['jacobian']=(-np.asarray(p['jacobian'])).tolist()
    bad=m.check_point(p,req,r,s.events,s.load)
    assert bad['steps'][0]['nominal_matrix_metrics']['direction_cosine']<-.999
    assert bad['steps'][0]['nominal_matrix_metrics']['relative_l2_to_AD']>1.99
    assert not good['AD_correctness_certified'] and good['scientific_threshold'] is None


def test_full_zero_norm_and_relative_element_unknown_not_epsilon_filled():
    z=m.measures(np.zeros(192),np.zeros(192))
    assert z['absolute_l2']==0 and z['direction_cosine'] is z['relative_l2_to_AD'] is z['symmetric_relative_l2'] is None
    z=m.measures(np.zeros(192),np.ones(192))
    assert z['relative_l2_to_AD'] is z['direction_cosine'] is None and z['symmetric_relative_l2']==1


@pytest.mark.parametrize('fault',['unknown','drop_attempt','drop_input','drop_return','order','API','intent','reserve_missing'])
def test_full_events_no_filtering_or_pending(replay,fault):
    p,req,r,s=replay;events=deepcopy(s.events)
    if fault=='unknown':events.insert(0,dict(kind='UNKNOWN'))
    elif fault in ('drop_attempt','drop_input','drop_return'):
        kind={'drop_attempt':'module_attempt','drop_input':'module_inputs_saved','drop_return':'module_return'}[fault]
        events.remove(next(e for e in events if e['kind']==kind))
    elif fault=='order':events[1],events[2]=events[2],events[1]
    elif fault=='API':next(e for e in events if e['kind']=='predict_API_return')['entry']['completed']=False
    elif fault=='intent':next(e for e in events if e['kind']=='evaluation_intent')['expected_input']['latent_history'][0][0]+=1
    else:events.remove(next(e for e in events if e['kind']=='evaluation_intent'))
    with pytest.raises(ValueError):m.check_point(p,req,r,events,s.load)


@pytest.mark.parametrize('fault',['count','raw','AD','tokens','context','module_input','endpoint','missing','drop_eval','anchor'])
def test_strict_original_arrays_raw_and_resources(replay,fault):
    p,req,original,store=replay;p=deepcopy(p);r=deepcopy(original);s=Memory();s.files=dict(store.files);events=deepcopy(store.events)
    if fault=='count':r['resources']['predict']['completed']-=1
    elif fault=='raw':r['evaluations'][1]['request']['commands'][0][0]+=.01
    elif fault=='AD':r['AD_API_ledger']=[dict(resource='jacobian',completed=True)]
    elif fault=='tokens':
        a=s.load(r['evaluations'][1]['token_array']);a[0,0]+=1
        r['evaluations'][1]['token_array']=m.F.save_array(s,'badtoken.npy',a)
    elif fault=='context':p['context']['latent_history'][0][0]+=1
    elif fault=='module_input':r['module_ledger'][4]['inputs']=[]
    elif fault=='endpoint':
        a=s.load(r['evaluations'][1]['latent_array']);a[-1,0]+=1
        r['evaluations'][1]['latent_array']=m.F.save_array(s,'badlatent.npy',a)
    elif fault=='missing':r['evaluations'][1]['status']='missing_domain'
    elif fault=='drop_eval':r['evaluations'].pop()
    else:p['anchor_context']['latent_history'][-1][0]+=1
    with pytest.raises((ValueError,KeyError,IndexError)):m.check_point(p,req,r,events,s.load)


@pytest.mark.parametrize('kind',['grad','jacobian','hessian','backward'])
def test_no_AD_engine_calls_and_restoration(kind):
    t=tensor.fake_torch();events=[];old=t.autograd.grad
    with pytest.raises(ValueError):
        with m.NoAD(t,m.F.blind.ARMS[1],lambda e:events.append(e),lambda:None):
            fn=t.autograd.grad if kind=='grad' else t.autograd.backward if kind=='backward' else getattr(t.autograd.functional,kind)
            fn(None)
    assert t.autograd.grad is old and events[-1]['kind']=='forbidden_AD_API'


def test_AUTH_before_IO_and_fixed_budget():
    with pytest.raises(ValueError):m.run('/NONE','0'*64,'/NONE','wrong')
    c=m.contract();assert c['max_predict']==480 and c['anchor_replays']==0 and c['signed_probe_denominator']==480 and c['central_pair_denominator']==240
    assert c['scientific_threshold'] is None and c['new_AD']==c['new_physics']==c['new_visual']==0
    assert not c['AD_correctness_certified'] and c['output_cap_bytes']==128*1024**2


def test_compile_binding_required_before_reconstruction(monkeypatch):
    monkeypatch.setattr(m.source,'consume_compile',lambda *a:pytest.fail('wrong source SHA must stop'))
    with pytest.raises(ValueError):m.build('/NONE','0'*64)
