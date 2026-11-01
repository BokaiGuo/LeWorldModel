"""RAM/temp-file fixtures only: no torch, models, physics, production prepare."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.spectral_reachability import agent_lewm_bounded_native_cem_h3_20261029_v1 as m


@pytest.fixture
def saved(tmp_path,monkeypatch):
    # Only NEW module parameters are reduced; frozen solver/hooks remain unchanged.
    monkeypatch.setattr(m,'N',4);monkeypatch.setattr(m,'I',2);monkeypatch.setattr(m,'K',2)
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:type('Free',(),{'free':10*1024**3})())
    store=m.Store(tmp_path/'results'/(m.NAME+'_fixture'));rows=[];slots=[];bases={}
    def mapping(logits,final=False):
        commands=np.tanh(logits.astype(np.float64)).astype(np.float32)
        commands=commands.reshape(15,2) if final else commands.reshape(1,m.N,3,5,2)
        return commands,dict(returned=True,logits=store.array(logits),raw=store.array(commands))
    def normalize(commands,slot):
        tokens=(commands.reshape(*commands.shape[:-2],10)-np.tile(np.asarray(slot['mean'],np.float32),5))/np.tile(np.asarray(slot['std'],np.float32),5)
        return tokens,dict(returned=True,raw=store.array(commands),tokens=store.array(tokens))
    for root in (0,1):
        slot=dict(source_slot=root,task_id=str(root)*64,task_sha256=str(root+2)*64,mean=[-.0087,.0068],std=[.2019,.2002]);slots.append(slot)
        base=dict(emb=np.full((1,3,192),root,np.float32),pixels=np.zeros((1,3,3,224,224),np.float32),
            goal=np.zeros((1,1,3,224,224),np.float32),goal_emb=np.full((1,1,192),.5,np.float32),
            warm=np.arange(20,dtype=np.float32).reshape(1,1,2,10)/8)
        bases[root]=base
        plan=dict(root=root,status='logit_mean_sealed',cost_attempted=m.I,cost_returned=m.I,
            base_inputs={k:store.array(v) for k,v in base.items()},costs=[],callbacks=[],modules=[])
        def module(target,kind,args,out,**fields):
            target.append(dict(index=len(target),kind=kind,returned=True,inputs=[store.array(a) for a in args],output=store.array(out),**fields))
        mean=np.zeros((1,3,10),np.float32);std=np.ones_like(mean)
        for j in range(m.I):
            logits=np.arange(120,dtype=np.float32).reshape(1,4,3,10)/64-.5;logits[:,0]=mean
            commands,mp=mapping(logits);tokens,norm=normalize(commands,slot)
            actions=np.concatenate((np.broadcast_to(base['warm'],(1,4,2,10)),tokens),axis=2)
            pred=np.zeros((1,4,6,192),np.float32);pred[:,:,:3]=base['emb'][:,None]
            pred[:,:,3]=np.arange(4,dtype=np.float32)[None,:,None]/8
            pred[:,:,4]=np.arange(4,dtype=np.float32)[None,:,None]/4
            pred[:,:,5]=np.arange(4,dtype=np.float32)[None,:,None]/2
            ae=np.zeros((4,5,192),np.float32);module(plan['modules'],'action_encoder',[actions.reshape(4,5,10)],ae,iteration=j)
            for step in range(3):
                pi=pred[:,:,step:step+3].reshape(4,3,192);po=pi.copy();po[:,-1]=pred[:,:,3+step].reshape(4,192)
                module(plan['modules'],'predictor',[pi,ae[:,step:step+3]],po,iteration=j)
                module(plan['modules'],'pred_proj',[po.reshape(12,192)],po.reshape(12,192).copy(),iteration=j)
            cost=((pred[:,:,-1]-base['goal_emb'][:,0,None])**2).sum(axis=-1,dtype=np.float32)
            inds=np.argsort(cost,axis=1,kind='stable')[:,:m.K].astype(np.int64);elite=logits[np.arange(1)[:,None],inds]
            newmean=elite.mean(axis=1);newstd=elite.std(axis=1,ddof=1)
            cb=dict(candidates=logits,costs=cost,topk_vals=np.take_along_axis(cost,inds,1),topk_inds=inds,topk_candidates=elite,
                mean=newmean,var=newstd,prev_mean=mean,prev_var=std)
            plan['callbacks'].append({k:store.array(v) for k,v in cb.items()})
            plan['costs'].append(dict(iteration=j,returned=True,rollout_returned=True,mapping=mp,normalization=norm,
                actions=store.array(actions),predicted_emb=store.array(pred),action_prefix=store.array(actions[:,:,:3]),cost=store.array(cost)))
            mean,std=newmean,newstd
        plan.update(returned_logits=store.array(mean),mean=store.array(mean),std=store.array(std),elite_average_cost=cb['topk_vals'].mean(axis=1).tolist())
        commands,final=mapping(mean,True)
        # Genuine Store -> strict NPY reader -> new deployment inputs.
        reread=m.old.load_array(store.path,final['raw'],store.inventory)
        tokens,norm=normalize(reread.reshape(1,3,5,2),slot)
        delivery=dict(root=root,status='paid_finalmean_evaluated',raw15=store.array(reread),normalization=norm,
            inputs={k:store.array(v) for k,v in dict(base,actions=tokens).items()},modules=[],layouts={},scores={})
        actions=np.concatenate((base['warm'],tokens[:,None]),axis=2)
        for layout in m.paid.LAYOUTS:
            pred=np.zeros((1,1,6,192),np.float32);pred[:,0,:3]=base['emb'];pred[:,:,3]=.125;pred[:,:,4]=.25;pred[:,:,5]=.5
            if layout=='segmented':pred[:,:,3:]+=np.float32(1/4096)
            e=dict(attempted=True,returned=True)
            if layout=='whole':
                ae=np.zeros((1,5,192),np.float32);module(delivery['modules'],'action_encoder',[actions[:,0]],ae,layout=layout)
                e['action_prefix']=store.array(actions[:,:,:3])
                for step in range(3):
                    po=pred[:,0,step:step+3].copy();po[:,-1]=pred[:,0,3+step]
                    module(delivery['modules'],'predictor',[pred[:,0,step:step+3],ae[:,step:step+3]],po,layout=layout)
                    module(delivery['modules'],'pred_proj',[po.reshape(3,192)],po.reshape(3,192).copy(),layout=layout)
            else:
                e['full3']=[]
                for step in range(3):
                    ae=np.zeros((1,3,192),np.float32);module(delivery['modules'],'action_encoder',[actions[:,0,step:step+3]],ae,layout=layout)
                    po=pred[:,0,step:step+3].copy();po[:,-1]=pred[:,0,3+step]
                    module(delivery['modules'],'predictor',[pred[:,0,step:step+3],ae],po,layout=layout)
                    module(delivery['modules'],'pred_proj',[po.reshape(3,192)],po.reshape(3,192).copy(),layout=layout);e['full3'].append(store.array(po))
            e['predicted_emb']=store.array(pred);delivery['layouts'][layout]=e
            cost=((pred[:,:,-1]-base['goal_emb'][:,0,None])**2).sum(axis=-1,dtype=np.float32)
            delivery['scores'][layout]=dict(attempted=True,returned=True,cost=store.array(cost))
        rows.append(dict(root=root,task_id=slot['task_id'],task_sha256=slot['task_sha256'],planning=plan,final_mapping=final,delivery=delivery))
    p=dict(slots=slots,inherited={'encoder':8,'projector':8},expected_model_hash='c'*64)
    status='known2_H3_bounded_CEM_actual_raw_evaluated'
    for e in m.expected_events('a'*64,status):store.event(**e)
    journal=store.path/'journal.jsonl';store.inventory['journal.jsonl']=dict(sha256=m.sha(journal),file_bytes=journal.stat().st_size)
    r=dict(schema=m.NAME+'.report',status=status,prepared='fixture',prepare_sha256='a'*64,contract=m.contract(),rows=rows,
        all_final_predictions_sealed=True,source_verified=True,errors=[],model_before='c'*64,model_after='c'*64,
        inherited=p['inherited'],physical_release=False,scientific_admission=False,elapsed_seconds=1.,GPU_peak_allocated=1,GPU_peak_reserved=1,
        actual_modules=m.module_totals(rows),
        journal_tail=store.tail,artifacts=copy.deepcopy(store.inventory),
        numerics=dict(requested_SDP='MATH',kernel_certificate=False,observed_inference=True,observed_autocast=False,observed_TF32=False))
    store.put('report.json',m.raw(r),final=True)
    monkeypatch.setattr(m,'verify',lambda *args:p);monkeypatch.setattr(m,'source_arrays',lambda slot:bases[slot['source_slot']])
    return store,r,p


def test_real_store_complete_reader(saved):
    store,r,p=saved;receipt=m.consume(store.path,m.sha(store.path/'report.json'))
    assert receipt['status']=='known2_H3_bounded_CEM_actual_raw_saved_verified'
    assert len(receipt['rows'])==2 and not receipt['physical_release'] and not receipt['parity_upgraded']
    for row in receipt['rows']:
        assert row['candidate_count']==8 and row['elite_average_is_not_delivery_cost']
        assert row['delivery']['comparisons']['all_future']['bitwise'] is False
        assert row['delivery']['comparisons']['frame25']['max_ULP']>0
    lines=(store.path/'journal.jsonl').read_bytes().splitlines(keepends=True)
    assert len(lines)==len(m.expected_events('a'*64,r['status']))
    assert all(line.endswith(b'\n') and not line.endswith(b'\n\n') for line in lines)
    assert m.old.SAMPLES==300 and m.old.STEPS==30 and m.old.SEED==1234


@pytest.mark.parametrize('mutation',['logits','rawdomain','normalizer','warm','context','goal','mean','std','elite','cost','causal',
    'missingcandidate','iteration','count','finalraw','finaltokens','reload','segmented','dtype'])
def test_all_root_mutations(saved,mutation):
    store,r,p=saved;row=r['rows'][0];plan=row['planning'];d=row['delivery'];e=plan['costs'][0];c=plan['callbacks'][0]
    def change(rec,fn):return store.array(fn(m.old.load_array(store.path,rec,store.inventory).copy()))
    if mutation=='logits':e['mapping']['logits']=change(e['mapping']['logits'],lambda a:a+1)
    elif mutation=='rawdomain':e['mapping']['raw']=change(e['mapping']['raw'],lambda a:a+2)
    elif mutation=='normalizer':e['normalization']['tokens']=change(e['normalization']['tokens'],lambda a:a+1)
    elif mutation=='warm':plan['base_inputs']['warm']=change(plan['base_inputs']['warm'],lambda a:a+1)
    elif mutation=='context':plan['base_inputs']['emb']=change(plan['base_inputs']['emb'],lambda a:a+1)
    elif mutation=='goal':plan['base_inputs']['goal_emb']=change(plan['base_inputs']['goal_emb'],lambda a:a+1)
    elif mutation=='mean':c['mean']=change(c['mean'],lambda a:a+1)
    elif mutation=='std':c['var']=change(c['var'],lambda a:a+1)
    elif mutation=='elite':c['topk_inds']=change(c['topk_inds'],lambda a:a*0)
    elif mutation=='cost':e['cost']=change(e['cost'],lambda a:a+1)
    elif mutation=='causal':plan['modules'][3]['inputs'][0]=change(plan['modules'][3]['inputs'][0],lambda a:a+1)
    elif mutation=='missingcandidate':e['mapping']['raw']=change(e['mapping']['raw'],lambda a:a[:,:3])
    elif mutation=='iteration':plan['cost_returned']-=1
    elif mutation=='count':plan['modules'].pop()
    elif mutation=='finalraw':row['final_mapping']['logits']=change(row['final_mapping']['logits'],lambda a:a+1)
    elif mutation=='finaltokens':d['inputs']['actions']=change(d['inputs']['actions'],lambda a:a+1)
    elif mutation=='reload':d['raw15']=change(d['raw15'],lambda a:a+1)
    elif mutation=='segmented':d['modules'][9]['inputs'][0]=change(d['modules'][9]['inputs'][0],lambda a:a+1)
    else:plan['modules'][1]['output']=change(plan['modules'][1]['output'],lambda a:a.astype(np.int64))
    with pytest.raises(ValueError):m.validate_root(p['slots'][0],row,lambda rec:m.old.load_array(store.path,rec,store.inventory))


@pytest.mark.parametrize('mutation',['partial','root','source','model','seal','memory','inherited','untracked','blank','hash','earlyscore','missingroot'])
def test_full_consume_rejects(saved,mutation):
    store,r,p=saved
    if mutation=='partial':r['status']='partial_or_failed'
    elif mutation=='root':r['rows'][0]['root']=1
    elif mutation=='source':r['source_verified']=False
    elif mutation=='model':r['model_after']='d'*64
    elif mutation=='seal':r['all_final_predictions_sealed']=False
    elif mutation=='memory':r['GPU_peak_reserved']=5*1024**3
    elif mutation=='inherited':r['inherited']={}
    elif mutation=='untracked':(store.path/'extra').write_bytes(b'x')
    elif mutation=='hash':r['journal_tail']='e'*64
    elif mutation=='missingroot':r['rows'].pop()
    else:
        path=store.path/'journal.jsonl';data=path.read_bytes()
        data=data+b'\n' if mutation=='blank' else data.replace(b'planning_predictions_sealed',b'criterion_before_sealing')
        path.write_bytes(data);r['artifacts']['journal.jsonl']=dict(sha256=m.sha(path),file_bytes=path.stat().st_size)
    (store.path/'report.json').write_bytes(m.raw(r))
    with pytest.raises((ValueError,json.JSONDecodeError)):m.consume(store.path,m.sha(store.path/'report.json'))


def test_fixed_budget_and_approved_protocol():
    r=m.resources()
    for k,v in dict(cost_oracle_API=60,model_get_cost_API=0,rollout_API=62,criterion_API=64,
        action_encoder=68,AE_samples=18008,AE_tokens=90028,predictor=192,predictor_samples=54012,
        predictor_context_tokens=162036,pred_proj=192,pred_proj_rows=162036,tanh_API=62,normalize_API=62).items():assert r[k]==v
    assert all(m.sha(p)==h for p,h in m.PINS.items())
    assert 'MAIN' in m.contract()['seed_authority'] and m.old.SEED==1234
    assert m.contract()['same_horizon_as_P2_common_H3'] is True
    events=m.expected_events('a','complete')
    for root in (0,1):
        for j in range(m.I):
            seal=next(i for i,e in enumerate(events) if e==dict(kind='planning_predictions_sealed',root=root,iteration=j))
            score=next(i for i,e in enumerate(events) if e==dict(kind='criterion_intent',root=root,layout='planning',iteration=j))
            assert seal<score
    seal=next(i for i,e in enumerate(events) if e['kind']=='all_final_predictions_sealed')
    assert sum(e['kind']=='criterion_intent' and e.get('layout')!='planning' for e in events)==4
    assert all(i>seal for i,e in enumerate(events) if e['kind']=='criterion_intent' and e.get('layout')!='planning')


def test_mapping_is_one_tanh_and_no_clip_source():
    source=Path(m.__file__).read_text();tree=ast.parse(source)
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)]
    assert sum(n.func.value.id=='torch' and n.func.attr=='tanh' for n in calls)==1
    assert not any(n.func.attr in ('clip','clamp') for n in calls)
    assert 'CEMSolver(Cost(),device=device,seed=SEED' in source
    assert 'all_final_predictions_sealed' in source


def test_mapping_cpu_reference_is_only_diagnostic():
    logits=np.zeros((1,3,10),np.float32);commands=np.ones((15,2),np.float32)/2
    a={'l':logits,'r':commands}
    _,diag=m.mapping_check(dict(returned=True,logits='l',raw='r'),a.__getitem__,(1,3,10))
    assert not diag['NumPy_tanh_diagnostic']['bitwise']  # source-bound observed call, not a libm equality certificate


class FixtureTensor:
    """NumPy arithmetic fixture for wiring; NOT a CUDA arithmetic certificate."""
    def __init__(self,a,device='cuda'):
        self.a=np.asarray(a);self.device=SimpleNamespace(type=device);self.dtype=self.a.dtype
        self.shape=self.a.shape;self.ndim=self.a.ndim;self.requires_grad=False
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.a
    def reshape(self,*shape):return FixtureTensor(self.a.reshape(*shape))
    def repeat(self,n):return FixtureTensor(np.tile(self.a,n))
    def is_floating_point(self):return self.dtype==np.float32
    def abs(self):return FixtureTensor(np.abs(self.a))
    def all(self):return bool(self.a.all())
    def any(self):return bool(self.a.any())
    def __le__(self,v):return FixtureTensor(self.a<=v)
    def __gt__(self,v):return FixtureTensor(self.a>v)
    def __sub__(self,v):return FixtureTensor(self.a-v.a)
    def __truediv__(self,v):return FixtureTensor(self.a/v.a)


@pytest.mark.parametrize('batch',[4,300])
def test_real_mapping_and_normalize_wiring_with_fixture_tensor(saved,monkeypatch,batch):
    store,_,p=saved;calls=[]
    monkeypatch.setattr(m,'N',batch)
    fake=SimpleNamespace(Tensor=FixtureTensor,float32=np.float32,
        isfinite=lambda a:FixtureTensor(np.isfinite(a.a)),
        as_tensor=lambda value,**kw:FixtureTensor(np.asarray(value,dtype=kw['dtype'])))
    def tanh(a):calls.append(a.a.copy());return FixtureTensor(np.tanh(a.a))
    fake.tanh=tanh;monkeypatch.setitem(sys.modules,'torch',fake)
    for final in (False,True):
        logits=FixtureTensor(np.ones((1,3,10) if final else (1,m.N,3,10),np.float32))
        actual,mapping=m.native_map(logits,store,0,None if final else 0,final)
        commands=actual.reshape(1,3,5,2) if final else actual
        tokens,normalization=m.native_normalize(commands,p['slots'][0],store,0,None if final else 0)
        load=lambda rec:m.old.load_array(store.path,rec,store.inventory)
        assert m.exact(load(mapping['raw']),actual.a)
        assert m.exact(m.normalization_check(normalization,commands.a,p['slots'][0],load),tokens.a)
    assert len(calls)==2
    mappingevents=[e for e in store.events if e['kind']=='mapping_return']
    assert [e['final_mapping'] for e in mappingevents[-2:]]==[False,True]
    with pytest.raises(ValueError):m.native_map(FixtureTensor(np.zeros((1,3,10),np.float32),'cpu'),store,0,None,True)


@pytest.mark.parametrize('mutation',['counter','task_id','task_sha'])
def test_complete_saved_identity_and_counter_rejection(saved,mutation):
    store,r,_=saved
    if mutation=='counter':r['actual_modules']['predictor']['samples_returned']+=1
    else:r['rows'][0]['task_id' if mutation=='task_id' else 'task_sha256']='e'*64
    (store.path/'report.json').write_bytes(m.raw(r))
    with pytest.raises(ValueError):m.consume(store.path,m.sha(store.path/'report.json'))


@pytest.mark.parametrize('kind',['wall','cap','floor'])
def test_limits(saved,monkeypatch,kind):
    store=saved[0]
    if kind=='wall':store.started-=m.WALL+1
    elif kind=='cap':store.total=m.CAP-m.RESERVE
    else:monkeypatch.setattr(m.shutil,'disk_usage',lambda p:type('Free',(),{'free':m.FLOOR})())
    with pytest.raises(ValueError):store.put('not_written',b'x')


def test_cold_cli_and_auth():
    with pytest.raises(ValueError):m.run('absent','no','absent','wrong')
    result=subprocess.run([sys.executable,'-c',
        'import sys;from experiments.spectral_reachability import agent_lewm_bounded_native_cem_h3_20261029_v1 as m;'
        'assert "torch" not in sys.modules;assert "pymunk" not in sys.modules;m.main()','--help'],capture_output=True,text=True)
    assert result.returncode==0 and all(x in result.stdout for x in ('prepare','verify','run','consume'))


def test_conservative_storage_bound():
    census=m.shape_census()
    assert census['array_count']==2170 and census['payload_bytes']==804556336
    assert census['working_upper_bytes']==950741040
    assert census['working_upper_bytes']<m.CAP-m.RESERVE
    assert census['largest_array_bytes']<16*1024**2
    assert len(m.whole_schedule(300))==7 and len(m.delivery_schedule())==16


def test_fixture_shape_census_exact(saved):
    store,report,_=saved
    inventory=report['artifacts']
    npys=[m.old.load_array(store.path,dict(file=f,**rec,
        shape=list(np.load(store.path/f,allow_pickle=False).shape),
        dtype=np.load(store.path/f,allow_pickle=False).dtype.str,
        nbytes=np.load(store.path/f,allow_pickle=False).nbytes),inventory)
        for f,rec in inventory.items() if f.endswith('.npy')]
    census=m.shape_census(m.N,m.I,m.K)
    assert len(npys)==census['array_count']
    assert sum(a.nbytes for a in npys)==census['payload_bytes']
    assert all(rec['file_bytes']-np.load(store.path/f,allow_pickle=False).nbytes<=256
        for f,rec in inventory.items() if f.endswith('.npy'))
    assert max(map(len,(store.path/'journal.jsonl').read_bytes().splitlines(keepends=True)))<=2048


def reseal(store,report,events):
    import hashlib
    tail=None;lines=[]
    for i,event in enumerate(events):
        e=dict(sequence=i,previous_sha256=tail,**event);tail=hashlib.sha256(m.raw(e)).hexdigest()
        e['event_sha256']=tail;lines.append(m.raw(e))
    path=store.path/'journal.jsonl';path.write_bytes(b''.join(lines));report['journal_tail']=tail
    report['artifacts']['journal.jsonl']=dict(sha256=m.sha(path),file_bytes=path.stat().st_size)
    (store.path/'report.json').write_bytes(m.raw(report));return m.sha(store.path/'report.json')


@pytest.mark.parametrize('root',[0,1])
@pytest.mark.parametrize('phase',['iteration0','iteration1','whole','segmented'])
@pytest.mark.parametrize('mutation',['score_before_seal','missing_seal'])
def test_valid_rehashed_barriers(saved,root,phase,mutation):
    store,r,_=saved;events=m.expected_events(r['prepare_sha256'],r['status'])
    if phase.startswith('iteration'):
        j=int(phase[-1]);seal=events.index(dict(kind='planning_predictions_sealed',root=root,iteration=j))
        score=events.index(dict(kind='criterion_intent',root=root,layout='planning',iteration=j))
    else:
        seal=events.index(dict(kind='all_final_predictions_sealed'))
        score=events.index(dict(kind='criterion_intent',root=root,layout=phase,iteration=None))
    if mutation=='missing_seal':events.pop(seal)
    else:events[seal],events[score]=events[score],events[seal]
    with pytest.raises(ValueError,match='seal-before-criterion orders'):m.consume(store.path,reseal(store,r,events))


@pytest.mark.parametrize('root',[0,1])
def test_raw_reload_no_second_tanh(saved,root):
    store,r,p=saved;d=r['rows'][root]['delivery'];load=lambda rec:m.old.load_array(store.path,rec,store.inventory)
    commands=load(d['normalization']['raw']);newraw=np.tanh(commands).astype(np.float32)
    assert not m.exact(commands,newraw)
    d['normalization']['raw']=store.array(newraw)
    with pytest.raises(ValueError,match='actual supplied raw normalizer input'):m.validate_root(p['slots'][root],r['rows'][root],load)


@pytest.mark.parametrize('root',[0,1])
def test_full_consume_rejects_extra_reload_mapping(saved,root):
    store,r,_=saved;events=m.expected_events(r['prepare_sha256'],r['status'])
    i=events.index(dict(kind='raw_reload_return',root=root))
    events[i:i]=[dict(kind=k,root=root,iteration=None,final_mapping=True) for k in ('mapping_intent','mapping_return')]
    with pytest.raises(ValueError,match='seal-before-criterion orders'):
        m.consume(store.path,reseal(store,r,events))


class FakeModule:
    def __init__(self):self.pre=[];self.post=[]
    def register_forward_pre_hook(self,fn):
        self.pre.append(fn);return SimpleNamespace(remove=lambda:self.pre.remove(fn))
    def register_forward_hook(self,fn):
        self.post.append(fn);return SimpleNamespace(remove=lambda:self.post.remove(fn))


@pytest.mark.parametrize('planning',[False,True])
def test_new_trace_shapes_and_cleanup(saved,monkeypatch,planning):
    store=saved[0]
    fake=SimpleNamespace(float32=np.float32,is_inference_mode_enabled=lambda:True,is_grad_enabled=lambda:False,
        is_autocast_enabled=lambda device:False,backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
        cudnn=SimpleNamespace(allow_tf32=False)))
    monkeypatch.setitem(sys.modules,'torch',fake)
    model=SimpleNamespace(**{k:FakeModule() for k in (*m.KINDS,'encoder','projector')})
    row=dict(root=0,modules=[],cost_attempted=1,active_layout='whole')
    tensor=lambda shape:FixtureTensor(np.zeros(shape,np.float32))
    with m.Trace(model,store,row,planning=planning):
        with pytest.raises(ValueError):model.encoder.pre[0](model.encoder,())
        wrong=(m.N if planning else 1,4,10)
        with pytest.raises(ValueError):model.action_encoder.pre[0](model.action_encoder,(tensor(wrong),))
        schedules=[m.whole_schedule(m.N)]*m.I if planning else [m.delivery_schedule()]
        for j,schedule in enumerate(schedules):
            row['cost_attempted']=j+1
            for i,(kind,shapes,outshape) in enumerate(schedule):
                row['active_layout']='whole' if i<7 else 'segmented'
                module=getattr(model,kind);args=tuple(tensor(shape) for shape in shapes)
                module.pre[0](module,args);module.post[0](module,args,tensor(outshape))
    assert len(row['modules'])==(m.I*7 if planning else 16)
    assert all(not getattr(model,k).pre and not getattr(model,k).post for k in (*m.KINDS,'encoder','projector'))
    with pytest.raises(RuntimeError):
        with m.Trace(model,store,dict(root=0,modules=[],cost_attempted=1),planning=True):raise RuntimeError('interrupted')
    assert all(not getattr(model,k).pre for k in (*m.KINDS,'encoder','projector'))


@pytest.mark.parametrize('mutation',['none','task_id','root_order','condition','context','goal','warm','normal','tasksha','encoding','source_override'])
def test_build_P_identity_fixture(tmp_path,monkeypatch,mutation):
    monkeypatch.setattr(m,'ROOT',tmp_path);monkeypatch.setattr(m,'PINS',{})
    slots=[];tasks={};stats=dict(mean=np.asarray([-.0087,.0068],np.float32).tolist(),std=np.asarray([.2019,.2002],np.float32).tolist())
    for root in (0,1):
        context=dict(latent_history=np.zeros((3,192),np.float32).tolist(),action_history=np.zeros((2,10),np.float32).tolist())
        task=dict(fixture_id=str(root)*64,root_context=copy.deepcopy(context),goal=np.zeros(192).tolist(),projection=np.eye(192).tolist(),
            root_raw_action_history=np.zeros((10,2)).tolist(),action_mean=stats['mean'],action_std=stats['std'])
        tasks[str(root)]=dict(task_id=task['fixture_id'],task=task)
        slots.append(dict(root_slot=root,condition='off',task_file=str(root),task_sha256=str(root+2)*64,task_id=task['fixture_id'],
            q=np.eye(192).tolist(),warm_raw10=task['root_raw_action_history'],packet=dict(root_context=context,goal=task['goal']),
            blind_path=str(root),blind_sha256='d'*64))
    prepared=dict(schema='agent_p2_common_resource_grasp_cuda_20261029_v1.prepare',slots=slots,
        policy='fixture_policy',expected_model_hash='c'*64,bindings_sha256={})
    enc=dict(module_ledger=[dict(resource=k,completed=True) for k in ('encoder','projector') for _ in range(16)])
    fake_domain=SimpleNamespace(parse=lambda obj:obj,read_bound=lambda path,*a,**kw:tasks[path],normalization=lambda:stats,
        core=SimpleNamespace(task_id=lambda t:t['fixture_id']),TASK_SHAS=tuple(s['task_sha256'] for s in slots),__file__='fixture_domain.py')
    monkeypatch.setattr(m,'domain',fake_domain)
    fakeold=SimpleNamespace(**vars(m.old));fakeold.inputs=SimpleNamespace(moving=SimpleNamespace(load_blind=lambda *args:
        dict(rgb_history=np.zeros((3,224,224,3),np.uint8),goal_rgb=np.zeros((224,224,3),np.uint8))))
    monkeypatch.setattr(m,'old',fakeold)
    monkeypatch.setattr(m,'read',lambda path,h:prepared if path==m.P_PREPARE else enc)
    monkeypatch.setattr(m,'sha',lambda path:{str(m.P_PREPARE):m.P_SHA,
        str(m.ENCODING):m.ENCODING_SHA}.get(str(path),'f'*64))
    if mutation=='task_id':slots[0]['task_id']='e'*64
    elif mutation=='root_order':slots.reverse()
    elif mutation=='condition':slots[0]['condition']='on'
    elif mutation=='context':tasks['0']['task']['root_context']['latent_history'][0][0]=1
    elif mutation=='goal':tasks['0']['task']['goal']=[1.]*192
    elif mutation=='warm':slots[0]['warm_raw10']=[[1.,1.]]*10
    elif mutation=='normal':tasks['0']['task']['action_std']=[1.,1.]
    elif mutation=='tasksha':slots[0]['task_sha256']='a'*64
    elif mutation=='encoding':enc['module_ledger'].pop()
    elif mutation=='source_override':prepared['bindings_sha256']['fixture_domain.py']='e'*64
    if mutation=='none':
        built=m.build();assert [s['task_id'] for s in built['slots']]==['0'*64,'1'*64]
        assert built['inherited']['encoder']==dict(attempted=16,returned=16)
        assert not built['contract']['fair_ranking']
    else:
        with pytest.raises(ValueError):m.build()

