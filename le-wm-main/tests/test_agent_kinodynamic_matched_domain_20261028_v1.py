"""RAM fixtures only: no Torch, model loading, physics, prepare or disk fixtures."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_kinodynamic_matched_domain_20261028_v1 as m


def task(slot=0):
    moving=m.sampling.old.moving
    packet=dict(rgb_history=np.stack([np.full((224,224,3),i+slot,np.uint8) for i in range(3)]),
        raw_action_history=moving.fixed_raw(),goal_rgb=np.full((224,224,3),9+slot,np.uint8))
    s=dict(source_slot=slot,blind_sha256=str(slot+2)*64,
        pixel_sha256=[moving.digest(x.tobytes()) for x in [*packet['rgb_history'],packet['goal_rgb']]],
        raw_action_sha256=moving.digest(packet['raw_action_history'].tobytes()))
    p=dict(normalization=moving.normalization(),model_bindings=moving.MODELS,arguments=dict(manifest_sha='a'*64))
    z=np.random.default_rng(17+slot).normal(size=(4,192)).astype(np.float32)
    export=moving.task_export(p,'b'*64,s,packet,list(z))
    return m.sampling.old.interface(moving.task_interfaces(export))['gn_task']


def receipt(a):
    a=np.asarray(a,np.float32)
    return dict(shape=list(a.shape),dtype='float32',values=a.astype(float).tolist())


class Backend:
    def __init__(self,fail=None):self.calls=0;self.fail=fail
    def advance(self,t,before,raw,meter):
        self.calls+=1;previous=deepcopy(before);raw=m.commands(raw);path=[]
        for k in range(len(raw)//5):
            meter.packet=k;meter.available()
            z=np.asarray(previous['latent_history'],np.float32)[None]
            outgoing=np.concatenate((np.asarray(previous['action_history'],np.float32),
                m.core.normalized_tokens(t,raw[k*5:k*5+5])))[None]
            a=meter.call('action_encoder',[receipt(outgoing)],lambda:outgoing.copy(),receipt)
            def api():
                if self.calls==self.fail:raise RuntimeError('fixture API fault')
                def predict():
                    out=z.copy();out[:,-1]+=(a.sum()*.0001+raw[k*5:k*5+5].sum()*.01)
                    return out
                pred=meter.call('predictor',[receipt(z),receipt(a)],predict,receipt)
                return meter.call('pred_proj',[receipt(pred.reshape(3,192))],
                    lambda:pred.reshape(3,192).copy(),receipt).reshape(1,3,192)
            out=meter.call('predict_API',[receipt(z),receipt(a)],api,receipt)
            previous=m.rolled(t,previous,raw[k*5:k*5+5],out[0,-1])
            path.append(previous);meter.emit('native_packet',packet=k,context=previous)
        return path


@pytest.fixture(scope='module')
def full():
    result=[]
    for slot in (0,1):
        t=task(slot);events=[];meter=m.Meter(events.append);b=Backend()
        d=m.Dispatch(t,b,meter,events.append);r=m.search(t,slot,d,events.append)
        result.append((t,r,events,meter,d,b))
    return result


@pytest.mark.parametrize('slot',[0,1])
def test_full_counts_saved_reconstruction(full,slot):
    t,r,e,meter,d,b=full[slot]
    assert meter.counts()==m.budget() and meter.predictor_samples()==dict(attempted=144,completed=144)
    assert d.index==b.calls==143 and len(r['first'])==20 and len(r['second'])==120
    assert len(r['retained_paths'])==6 and len(r['commands'])==10
    assert m.journal_check(t,slot,r,e)==r
    assert sum(x['kind']=='native_packet' for x in e)==144
    assert e[0]['kind']=='bank_sealed_before_evaluation'
    assert r['whole_contexts']==r['segmented_contexts']


@pytest.mark.parametrize('slot',[0,1])
def test_bank_full_time_domain(slot):
    b=m.bank(slot);assert b==m.bank(slot)
    logits=np.asarray(b['logits']);raw=np.asarray(b['commands'])
    assert logits.shape==(20,10) and raw.shape==(20,5,2)
    assert np.all(np.linalg.norm(logits,axis=1)<=m.CAP)
    assert np.array_equal(raw,np.tanh(logits.astype(np.float32)).reshape(20,5,2))
    assert not raw[0].any() and all(not np.all(x==x[0]) for x in raw[1:])
    assert len({x.tobytes() for x in raw})==20


def test_witnesses_and_no_short_winner(full):
    for t,r,e,_,_,_ in full:
        bank=np.asarray(m.bank(r['source_slot'])['commands'])
        for n in r['first']+r['second']:
            assert n['commands']==np.concatenate([bank[i] for i in n['path']]).tolist()
            assert n['edge_commands']==bank[n['path'][-1]].tolist()
            assert n['context']['action_history'][-1]==m.core.normalized_tokens(t,n['edge_commands'])[0].tolist()
        best=min(r['second'],key=lambda n:(n['cost'],n['path']))
        assert r['selected_path']==best['path'] and len(best['path'])==2
        assert r['retained_paths']==[n['path'] for n in sorted(r['first'],key=lambda n:(n['cost'],n['path']))[:6]]
        assert next(i for i,x in enumerate(e) if x['kind']=='selection_sealed_before_replays') < next(i for i,x in enumerate(e) if x['kind']=='edge_attempt' and x['edge']['phase']=='fresh_whole')


def test_ties_stable_and_all_depth2():
    t=task();t['goal']=t['root_context']['latent_history'][-1]
    calls=[]
    def dispatch(c,raw,phase,path):
        calls.append(phase);out=[]
        for k in range(len(raw)//5):
            c=m.rolled(t,c,raw[k*5:k*5+5],c['latent_history'][-1]);out.append(c)
        return out
    r=m.search(t,0,dispatch,lambda e:None)
    assert r['retained_paths']==[[i] for i in range(6)] and r['selected_path']==[0,0]
    assert calls.count('second')==120 and len(calls)==143


@pytest.mark.parametrize('change',['missing','extra','batch','pending','history','output','position','selection','parent','budget'])
def test_tamper_failclosed(full,change):
    t,r,events,_,_,_=full[0];ev=deepcopy(events)
    i=next(i for i,x in enumerate(ev) if x['kind']=='dispatch_attempted')
    if change=='missing':ev.pop(i)
    elif change=='extra':ev.insert(i,deepcopy(ev[i]))
    elif change=='batch':ev[i]['receipt']['batch_size']=2
    elif change=='pending':
        j=next(j for j,x in enumerate(ev) if x['kind']=='dispatch_returned');ev[j]['receipt']['status']='pending'
    elif change=='history':ev[i]['receipt']['inputs'][0]['values'][0][0][0]+=.1
    elif change=='output':
        j=next(j for j,x in enumerate(ev) if x['kind']=='dispatch_returned');ev[j]['receipt']['output']['values'][0][0][0]+=.1
    elif change=='position':ev.insert(0,ev.pop(i))
    elif change=='selection':next(x for x in ev if x['kind']=='selection_sealed_before_replays')['path']=[19,19]
    elif change=='parent':next(x for x in ev if x['kind']=='node_saved')['node']['parent_id']='invented'
    else:ev.append(deepcopy(ev[-1]))
    with pytest.raises(ValueError):m.journal_check(t,0,r,ev)


def test_failure_attempts_no_retry():
    t=task();ev=[];meter=m.Meter(ev.append);b=Backend(fail=1);d=m.Dispatch(t,b,meter,ev.append)
    with pytest.raises(RuntimeError,match='fixture API fault'):m.search(t,0,d,ev.append)
    assert meter.counts()['predict_API']==dict(attempted=1,completed=0)
    assert meter.counts()['action_encoder']==dict(attempted=1,completed=1)
    assert meter.counts()['predictor']==dict(attempted=0,completed=0)
    with pytest.raises(ValueError):d(t['root_context'],m.bank(0)['commands'][0],'first',[0])
    assert b.calls==1 and ev[-1]['kind']=='edge_failed'


def test_no_budget_padding(full):
    t,r,e,meter,d,b=full[0]
    with pytest.raises(ValueError):d(t['root_context'],m.bank(0)['commands'][0],'first',[0])
    assert b.calls==143
    calls=[]
    with pytest.raises(ValueError):meter.call('predict_API',[],lambda:calls.append(1))
    assert calls==[]


def test_plan_identity_and_denominator(full):
    t,r,*_=full[0]
    slots=[dict(source_slot=s,task_id='id'+str(s),blind_sha256='b'*64,task_export='task',task_sha256='a'*64) for s in (0,1)]
    p=m.model_plan(slots[0],r)
    assert p['schema']==m.NAME+'.model_plan' and p['arm']==m.ARM and p['commands']==r['commands']
    assert not p['selection_uses_physics'] and not p['P1_admission']
    rows=m.initial_rows(dict(slots=slots));assert len(rows)==2 and all(x['status']=='not_started' for x in rows)


def test_partial_keeps_second_root(monkeypatch):
    slots=[dict(source_slot=s,task_id='id'+str(s),blind_sha256='b'*64,task_export='task',task_sha256='a'*64) for s in (0,1)]
    rows=m.initial_rows(dict(slots=slots));monkeypatch.setattr(m.sampling.old,'load_export',lambda *a:dict(gn_task=task()))
    class Store:
        def event(self,e):return None
    def fail(*a):raise RuntimeError('stop')
    with pytest.raises(RuntimeError):m.plan_all(dict(slots=slots),Store(),fail,rows)
    assert len(rows)==2 and rows[0]['status']=='attempted' and rows[1]['status']=='not_started'


@pytest.mark.parametrize('value',[np.zeros((4,2)),np.full((5,2),np.nan),np.ones((5,2)),np.full((5,2),.123456789)])
def test_illegal_raw(value):
    with pytest.raises(ValueError):m.commands(value)


def test_contract_and_inherited_authority():
    c=m.contract()
    assert c['total']==m.budget(288) and c['schedule']['total_predict']==144
    assert c['bank']['seeds']==[20281028,20281129]
    assert not any(c[k] for k in ('P1_admission','equal_FLOPs','equal_total_compute','independent_confirmation','official_reproduction'))
    assert c['output_cap_bytes']==64*1024**2 and c['floor_bytes']==200*1024**2
    assert c['wall_seconds']==900 and c['AD']==c['new_physics']==c['new_encoder']==0
    assert m.sha(m.sampling.__file__)==m.SAMPLING_SHA and m.sha(m.contexts.__file__)==m.CONTEXT_SHA
