"""RAM exact schema/tensor/hook doubles. No torch, neural model or environment."""
from copy import deepcopy
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_midfeedback_terminal_rgb_20261028_v1 as m
import test_agent_p2_contextual_physical_rgb_encode_20261028_v1 as old


def fixture(missing=()):
    prep,_,_=old.fixture();plans=[];pairs=[];previous=[]
    for i,original in enumerate(prep['plans']):
        context=deepcopy(original['root_context']);context.update(layout='states_with_between_state_actions_v1',
            context_length=3,latent_dim=192,action_dim=10)
        context['latent_history'][-1]=old.z(.5)
        raw=np.full((5,2),.125,np.float32).tolist()
        b=dict(prefix_commands=original['commands'][:5],observed_context=context,goal=original['goal'],branch_id=old.h(i))
        row=dict(new_suffix_commands=raw,evaluations=[{},dict(context=context,cost=m.bridge.mse(old.z(.5),original['goal']))],equal_new_replay=True)
        plan=m.intervention.seal_plan(b,row,original,'a'*64);plans.append(plan)
        cp=dict(step=10,png=f'b{i}_plan_10_{1 if plan["condition"]=="on" else 0}.png',
            file_sha256=old.h(('PNG',i)),pixel_sha256=old.h(('pixels',i)),state=[0.]*7,snapshot=dict(RAM_ONLY=True),
            condition=plan['condition'],compiler_selection_digest=plan['compiler_selection_digest'],task_id=plan['task_id'],
            original_task_file=plan['original_task_file'],original_task_sha256=plan['original_task_sha256'],
            original_blind_path=plan['blind_path'],original_blind_sha256=plan['blind_sha256'],encoder_identity=plan['encoder_identity'],
            encoding_performed=False,model_bound_by_prepare=True,raw_action_history10=plan['commands'],raw_action_history_bits=plan['raw_float32_bits'])
        identity=dict(index=i,root_slot=plan['root_slot'],condition=plan['condition'],arm=plan['arm'])
        actual=dict(**identity,selection_digest=plan['selection_digest'],compiler_selection_digest=plan['compiler_selection_digest'],
            rgb_encoding_inputs=[] if i in missing else [cp],missing_checkpoints=[10] if i in missing else [],
            own_returned=9 if i in missing else 10,full_horizon=i not in missing,final_metrics=dict(native_success=False,native_coverage=None))
        control=dict(**identity,selection_digest=original['selection_digest'],final_metrics=dict(native_success=False,native_coverage=None))
        pairs.append(dict(index=i,intervention=actual,control=control,new_minus_control=dict(native_coverage=None)))
        previous.append(dict(**identity,selection_digest=original['selection_digest'],checkpoints=[
            dict(step=s,latent_Q=.5 if s==10 else .25) for s in (0,5,10)],real_mid_same_suffix_predicted_Q=.375))
    strict=dict(schema=m.intervention.NAME+'.physical_consume',status='paired_known_intervention_recorded_NOT_science',
        fixed_denominator=8,known_roots=2,P1_admission=False,P2_admission=False,physical_release=False,rows=pairs)
    prior=dict(schema=m.bridge.NAME+'.consume',status='actual_RGB_and_same_suffix_saved_verified_NOT_science',rows=previous)
    return plans,strict,prior


def prepared(missing=()):
    rows=m.fixed_rows(*fixture(missing),'/RAM_ONLY/physical');n=8-len(missing)
    return dict(rows=rows,contract=m.contract(),expected_calls={k:n if k in ('encoder','projector') else 0 for k in m.bridge.KINDS})


class Tensor(old.Tensor):
    def backward(self,*a,**kw):pytest.fail('no neural backward')


class Module(old.Module):
    def forward(self,*args):return self.fn(*args)
    def __call__(self,*args):
        for fn in self.pre:fn(self,args)
        out=self.forward(*args)
        for fn in self.post:fn(self,args,out)
        return out


def fake_torch():
    def noop(*a,**kw):pytest.fail('no AD execution')
    functional=NS(**{k:noop for k in ('jacobian','hessian','jvp','vjp','hvp','vhp')})
    func=NS(**{k:noop for k in ('jvp','vjp','jacrev','jacfwd','hessian','grad','grad_and_value')})
    return NS(float32=np.dtype('float32'),is_grad_enabled=lambda:False,cuda=NS(synchronize=lambda:None),
        autograd=NS(grad=noop,backward=noop,functional=functional),Tensor=Tensor,func=func)


class Model:
    def __init__(self,tokens=257):
        self.encoder=Module(lambda x:Tensor(np.full((1,tokens,192),x.a.mean(),np.float32)))
        self.projector=Module(lambda x:Tensor(x.a+.125))
        self.action_encoder=Module(lambda *a:pytest.fail('forbidden action encoder'))
        self.predictor=Module(lambda *a:pytest.fail('forbidden predictor'))
        self.pred_proj=Module(lambda *a:pytest.fail('forbidden pred proj'))
    def predict(self,*a):pytest.fail('forbidden predict')


def pixels(f):return np.full((224,224,3),64,np.uint8)


def execute(monkeypatch,missing=(),tokens=257,forbidden=None):
    p=prepared(missing);rows=deepcopy(p['rows']);s=old.Memory();model=Model(tokens);torch=fake_torch();errors=[]
    monkeypatch.setattr(m.bridge,'load_pixels',pixels)
    trace=m.bridge.Trace(model,torch,s,lambda:None,p['expected_calls'])
    guard=m.bridge.native.Ledger(2,lambda e:s.event(dict(kind='guard_event',event=e)),no_grad=lambda:True)
    original=model.predict
    def encode(pix,row,scope,ordinal):
        guard.image=ordinal
        with m.physical.saved.overlay.traced_scope(trace,scope,row):
            if forbidden=='predict':model.predict(None)
            if forbidden=='AD':torch.autograd.functional.jacobian(None)
            if forbidden=='action_encoder':model.action_encoder(None)
            pre=m.bridge.pixels.preprocess_pixels(pix)
            row['preprocessed']=trace.save(Tensor(pre),f'r{row["index"]}_preprocessed')
            return model.projector(model.encoder(Tensor(pre.reshape(1,3,224,224)))[:,0]).a[0].copy()
    with trace,m.bridge.native.guard(model,torch,guard,errors):m.process(p,rows,s,encode)
    assert model.predict==original and not errors and not guard.blocked
    r=dict(rows=rows,contract=m.contract(),errors=errors,blocked=guard.blocked,modules=trace.entries,guards=guard.entries,
        counts=m.bridge.counts(trace.entries))
    return p,r,s


@pytest.fixture(scope='module')
def replay():
    with pytest.MonkeyPatch.context() as patch:return execute(patch)


def test_production_tensor_hooks_AND_encode_only_guard_all8(replay,monkeypatch):
    p,r,s=replay;monkeypatch.setattr(m.bridge,'load_pixels',pixels)
    result=m.validate(p,r,s.events,s.load)
    assert len(result)==8 and len(r['modules'])==16 and len(r['guards'])==16
    assert r['counts']['encoder']==r['counts']['projector']==dict(attempted=8,completed=8)
    assert all(v['new_suffix_predicted_Q']==.25 and v['old_fixed_actual_Q']==.5 for v in result)
    assert all(v['new_minus_old_predicted_Q']==-.125 and not v['physical_benefit_claim'] for v in result)


@pytest.mark.parametrize('missing',[(0,),(7,),(0,3,7),tuple(range(8))])
def test_available_subset_keeps_full8_without_padding(monkeypatch,missing):
    p,r,s=execute(monkeypatch,missing);result=m.validate(p,r,s.events,s.load)
    assert len(result)==8 and p['expected_calls']['encoder']==8-len(missing)
    for i in missing:assert result[i]['new_suffix_actual_Q'] is result[i]['new_endpoint_MSE'] is None
    assert sum(x['latent_missing'] for x in result)==len(missing)


@pytest.mark.parametrize('bad',['predict','AD','action_encoder'])
def test_production_guard_blocks_new_predict_and_AD(monkeypatch,bad):
    with pytest.raises(ValueError):execute(monkeypatch,forbidden=bad)


def test_hidden197_refused_not_shape_relaxed(monkeypatch):
    with pytest.raises(ValueError,match='hidden257'):execute(monkeypatch,tokens=197)


@pytest.mark.parametrize('fault',['parent_as_new','cost','context','witness_raw','witness_goal','control','condition',
    'rawhistory','cp_model','cp_unknown','cp_early','drop_slot'])
def test_true_new_plan_schema_binding_mutations(fault):
    plans,s,oldrgb=fixture();plan=plans[0];actual=s['rows'][0]['intervention']
    if fault=='parent_as_new':plan['predicted_Q']=plan['parent_predicted_Q']
    elif fault=='cost':plan['new_suffix_Q']+=1
    elif fault=='context':plan['new_suffix_context']['latent_history'][-1][0]+=1
    elif fault=='witness_raw':plan['new_suffix_witness']['raw5'][0][0]=.5
    elif fault=='witness_goal':plan['new_suffix_witness']['goal'][0]=1.
    elif fault=='control':oldrgb['rows'][0]['selection_digest']='0'*64
    elif fault=='condition':actual['rgb_encoding_inputs'][0]['condition']='on'
    elif fault=='rawhistory':actual['rgb_encoding_inputs'][0]['raw_action_history_bits']='00'*80
    elif fault=='cp_model':actual['rgb_encoding_inputs'][0]['encoder_identity']={}
    elif fault=='cp_unknown':actual['missing_checkpoints']=[10]
    elif fault=='cp_early':actual['own_returned']=9
    else:s['rows'].pop()
    if fault in ('cost','context','witness_raw','witness_goal'):
        plan['selection_digest']=m.physical.saved.digest({k:v for k,v in plan.items() if k!='selection_digest'})
        actual['selection_digest']=plan['selection_digest']
    with pytest.raises((ValueError,KeyError)):m.fixed_rows(plans,s,oldrgb,'/RAM_ONLY/physical')


@pytest.mark.parametrize('fault',['event','drop_attempt','order','shape','preprocessed','pixel','native_output','plan','guard','missing','count','blocked'])
def test_saved_arrays_and_dual_ledger_complete_not_hash_only(replay,monkeypatch,fault):
    p,original,store=replay;r=deepcopy(original);events=deepcopy(store.events);s=old.Memory();s.files=dict(store.files)
    monkeypatch.setattr(m.bridge,'load_pixels',pixels);row=r['rows'][0]
    if fault=='event':events.insert(0,dict(kind='BAD_EVENT'))
    elif fault=='drop_attempt':events.remove(next(e for e in events if e['kind']=='module_attempt'))
    elif fault=='order':events[2],events[3]=events[3],events[2]
    elif fault=='shape':row['modules'][0]['output']=s.array('bad197',np.zeros((1,197,192),np.float32))
    elif fault=='preprocessed':row['preprocessed']=s.array('badpre',np.zeros((1,1,3,224,224),np.float32))
    elif fault=='pixel':row['pixels']=s.array('badpix',np.zeros((224,224,3),np.uint8))
    elif fault=='native_output':row['latent']=s.array('badlatent',np.zeros(192,np.float32))
    elif fault=='plan':row['new_suffix_Q']+=1
    elif fault=='guard':r['guards'].pop()
    elif fault=='missing':row['source']=None
    elif fault=='count':r['counts']['encoder']['completed']-=1
    else:r['blocked']=['predict']
    with pytest.raises((ValueError,KeyError)):m.validate(p,r,events,s.load)


def test_AUTH_before_prepare_or_model_IO(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('no IO'))
    with pytest.raises(ValueError,match='MAIN explicit CUDA'):m.run('/NONE','0'*64,'/NONE','wrong')


def test_authority_reconstruction_not_just_SHA_and_status(monkeypatch):
    monkeypatch.setattr(m,'sha',lambda path:m.INTERVENTION_SHA if str(path)==m.intervention.__file__ else '1'*64)
    calls=[];monkeypatch.setattr(m.intervention,'consume_physical',lambda *a:(calls.append(a) or dict(status='one')))
    monkeypatch.setattr(m.physical,'read',lambda path:dict(status='two'))
    with pytest.raises(ValueError,match='whole actual physical strict reconstruction'):
        m.build('/RAM_ONLY/physical','2'*64,'/RAM_ONLY/receipt','1'*64)
    assert len(calls)==1
