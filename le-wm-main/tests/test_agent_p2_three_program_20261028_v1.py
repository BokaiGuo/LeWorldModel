"""RAM numerical callbacks and complete saved lifecycle mutations; no torch."""
from copy import deepcopy
import io
import json
import shutil
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_three_program_20261028_v1 as m


def slot(root=0):
    raw=np.zeros((10,2),np.float32)
    context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=np.zeros((3,192)).tolist(),action_history=m.kernel.normalized_history(raw).tolist())
    return dict(index=2*root,root_slot=root,condition='off',task_id=str(root+1)*64,task_file='/RAM_ONLY/task.json.gz',
        task_sha256='3'*64,blind_sha256='4'*64,blind_path='/RAM_ONLY/blind.npz',warm_raw10=raw.tolist(),
        packet=dict(root_context=context,goal=np.full(192,.125).tolist()),q=np.eye(192).tolist())


class Numeric:
    """No learned ops/weights. Shape-faithful additive RAM response only."""
    def __init__(self,index=0,sink=lambda e:None):
        self.row=dict(index=index,forwards=[],AD=[],status='attempted');self.events=[];self.arrays={};self.sink=sink
    def event(self,e):self.events.append(deepcopy(e));self.sink(deepcopy(e))
    def rollout(self,c,commands,tag):
        paths=[]
        for k,actions in enumerate(commands):
            a=np.asarray(actions,np.float32);n=len(self.row['forwards']);request=m.packet_request(n,c,a,tag+'/'+str(k))
            self.event(dict(kind='forward_intent',request=request));input_context=deepcopy(c)
            self.event(dict(kind='forward_inputs',index=n,context=c,raw=m.blob(a)))
            incoming=np.concatenate((np.asarray(c['action_history'],np.float32),m.kernel.normalized_history(a)))[None]
            emb=np.zeros((1,3,192),np.float32);emb[:,:,:10]=incoming
            z=np.asarray(c['latent_history'],np.float32)[None]
            prediction=(z+np.float32(a.sum()/64)).astype(np.float32);projected=prediction.reshape(3,192)
            modules=[m.module_receipt(3*n,'action_encoder',[m.blob(incoming)],m.blob(emb)),
                m.module_receipt(3*n+1,'predictor',[m.blob(z),m.blob(emb)],m.blob(prediction)),
                m.module_receipt(3*n+2,'pred_proj',[m.blob(projected)],m.blob(projected.copy()))]
            c=dict(c,latent_history=np.concatenate((z[0,1:],projected[-1:])).tolist(),action_history=incoming[0,1:].tolist())
            f=dict(index=n,tag=tag+'/'+str(k),request=request,input=input_context,raw=m.blob(a),modules=modules,output=c,completed=True)
            self.row['forwards'].append(f)
            for e in m.module_events(f):self.event(e)
            m.validate_packet(f);self.event(dict(kind='forward_return',index=n,context=c,completed=True));paths.append(deepcopy(c))
        return paths
    def jacobian(self,c,u,tag,logit):
        n=len(self.row['AD']);request=dict(index=n,tag=tag,input_context_sha256=m.digest(c),at_logits=np.asarray(u).tolist(),
            raw=m.chart(u).tolist(),coordinates='logit_zero' if logit else 'raw_incumbent',shape=[192,len(u)],
            mathematical_VJP=192,vectorize=True,completed=False)
        self.event(dict(kind='AD_intent',request=request));path=self.rollout(c,m.chart(u),tag+'/forward')
        j=np.full((192,len(u)),1/64,np.float32);data=io.BytesIO();np.save(data,j,allow_pickle=False)
        name=f'root{self.row["index"]}_J{n:03}.npy';desc=dict(file=name,sha256=m.hashlib.sha256(data.getvalue()).hexdigest(),shape=list(j.shape),dtype='float32')
        self.arrays[name]=data.getvalue();self.last_ad_contexts=deepcopy(path)
        self.row['AD'].append(dict(request=request,array=desc,contexts=path,completed=True))
        self.event(dict(kind='AD_return',index=n,array=desc,contexts=path,completed=True));return j.astype(float)


@pytest.fixture(scope='module')
def numerical():
    backend=Numeric();result=m.plan(slot(),backend);backend.row['result']=result
    return backend,result


def test_fixed_full15_denominators_and_paid_axes():
    c=m.contract();assert c['candidate_denominator']==168 and c['plan_denominator']==4
    assert m.FORWARDS==6+42*3+42*(1+14+7)+12==1068
    assert m.JACS==1+42*2==85 and c['total']['mathematical_VJP']==32640
    assert c['joint_per_root']['standalone_predict']==138 and c['composed_per_root']['standalone_predict']==936
    assert c['total_FLOPs'] is c['engine_internal_backward_calls'] is None
    assert not any(c[k] for k in ('equal_FLOPs','equal_total_compute','P1_admission','P2_mechanism','P2_admission','physical_release','fresh'))


def test_full_algorithm_three_independent_compiler_nodes_and_guard(numerical):
    backend,r=numerical;assert 'torch' not in sys.modules
    assert len(backend.row['forwards'])==1068 and len(backend.row['AD'])==85
    assert [e['request']['shape'] for e in backend.row['AD']]==[[192,30]]+[[192,d] for _ in range(42) for d in (20,10)]
    assert len(r['bank']['params'])==42 and len(r['candidates'][m.ARMS[0]])==len(r['candidates'][m.ARMS[1]])==42
    for i,c in enumerate(r['candidates'][m.ARMS[1]]):
        u=np.asarray(r['bank']['params'][i]);assert np.array_equal(np.asarray(c['logits'])[:10],u[:10])
        assert c['edge2']['input_context']==c['contexts'][0]
        assert c['edge3']['input_context']==c['contexts'][1]
        assert c['edge3']['input_context']!=slot()['packet']['root_context'] or i==0
        assert c['cost']<=r['candidates'][m.ARMS[0]][i]['cost']
        for e in (c['edge2'],c['edge3']):
            assert e['selected_cost']<=e['incumbent_cost'] and len(e['evaluations'])==5
            assert e['selected_index']==min(range(5),key=lambda k:(e['evaluations'][k]['cost'],k))
    assert r['bank']['params'][0]==[0.]*30


def test_complete_saved_backend_reconstructs_every_event_and_all_scores(numerical):
    backend,result=numerical;events=[]
    def load(d):return np.load(io.BytesIO(backend.arrays[d['file']]),allow_pickle=False)
    saved=m.SavedBackend(backend.row,events.append,load);rebuilt=m.plan(slot(),saved);saved.complete()
    assert m.same(result,rebuilt) and m.same(events,backend.events)


@pytest.mark.parametrize('change',['raw','context','module_input','native_output','index','AD_anchor','AD_shape','AD_context','AD_completed'])
def test_saved_inputs_modules_and_AD_mutation_rejection(numerical,change):
    source,_=numerical;row=deepcopy(source.row);f=row['forwards'][0]
    if change=='raw':f['raw']=m.blob(np.ones((5,2),np.float32)*.1)
    elif change=='context':f['input']['latent_history'][0][0]=1
    elif change=='module_input':f['modules'][1]['inputs'][0]=m.blob(np.ones((1,3,192),np.float32))
    elif change=='native_output':f['output']['latent_history'][-1][0]=1
    elif change=='index':f['modules'][1]['index']=77
    elif change=='AD_anchor':row['AD'][0]['request']['at_logits'][0]=.1
    elif change=='AD_shape':row['AD'][0]['request']['shape']=[192,20]
    elif change=='AD_context':row['AD'][0]['contexts'][0]['latent_history'][-1][0]=1
    else:row['AD'][0]['completed']=False
    events=[];saved=m.SavedBackend(row,events.append,lambda d:np.load(io.BytesIO(source.arrays[d['file']]),allow_pickle=False))
    with pytest.raises(ValueError):
        m.plan(slot(),saved);saved.complete()


def test_stable_tie_incumbent_and_exact_chart_rounding_no_inverse():
    prefix=np.zeros(10);u=np.linspace(-.2,.2,20);z=np.ones(192);j=np.zeros((192,20))
    p=m.proposals(prefix,u,z,j,z)
    assert p['logits']==[u.tolist()]*5 and p['remaining_radius']==m.CAP_LOGIT
    assert m.chart(u).reshape(-1).tobytes()==np.tanh(u).astype('<f4').tobytes()
    assert not np.array_equal(np.tanh(u).astype(np.float32),np.tanh(u.astype(np.float32)))
    for value in ([float('nan')]*20,[100.]*20):
        with pytest.raises(ValueError):m.proposals(prefix,value,z,j,z)


def test_new_bank_generation_rounding_never_repairs_incumbent():
    u=np.zeros(30);u[0]=np.nextafter(m.CAP_LOGIT,np.inf)
    assert not m.legal([],u)
    x=m.inward(u,m.CAP_LOGIT);assert m.legal([],x) and not np.array_equal(x,u)
    with pytest.raises(ValueError,match='incumbent'):m.proposals(u[:10],u[10:],np.zeros(192),np.zeros((192,20)),np.zeros(192))


def test_failure_stops_at_first_cause_no_candidate_refill():
    class Broken(Numeric):
        def rollout(self,c,a,tag):
            if tag=='compose/0/edge2/candidate1':raise ValueError('RAM first failure')
            return super().rollout(c,a,tag)
    b=Broken()
    with pytest.raises(ValueError,match='RAM first failure'):m.plan(slot(),b)
    assert len(b.row['AD'])==2 and len(b.row['forwards'])<m.FORWARDS
    assert not any(e.get('kind')=='selection_frozen' for e in b.events)


def test_auth_rejection_precedes_prepare_and_output(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('verify must not run'))
    with pytest.raises(ValueError,match='authorization'):m.run_cuda('none','0'*64,tmp_path/m.NAME,'no')
    assert not (tmp_path/m.NAME).exists() and 'torch' not in sys.modules


def test_real_saved_original_off_tasks_schema_no_prepare_no_neural():
    for i in (0,2):
        path=m.assets.PARENT/f'task_{i:02}.json.gz';e=m.parent.overlay.read(path,m.PREP_CAP)
        t=e['task'];assert e['task_contract']['condition']=='off'
        assert np.asarray(t['root_raw_action_history']).shape==(10,2)
        assert Path(e['task_contract']['blind_path']).is_absolute()
        assert np.array_equal(t['projection'],np.eye(192))
        m.kernel.validate_context(t['root_context'])
    assert 'torch' not in sys.modules


@pytest.fixture(scope='module')
def saved_bundle(numerical,tmp_path_factory):
    # Isolated new-module authority seam. No frozen module/global mutations.
    source,result=numerical;slots=[slot(0),slot(1)]
    p=dict(slots=slots,expected_model_hash='a'*64,source_report_sha256='b'*64,bindings_sha256={},contract=m.contract())
    output=tmp_path_factory.mktemp('three_program')/(m.NAME+'_RAM');store=m.Store(output,time.monotonic());summaries=[]
    store.put('binding.json',dict(prepared='/RAM_ONLY/prepared.json',prepare_sha256='c'*64));store.put('loader.json',dict(files=[]))
    for s in slots:
        root=s['root_slot'];row=deepcopy(source.row);row['index']=root
        for k,entry in enumerate(row['AD']):
            name=f'root{root}_J{k:03}.npy';data=source.arrays[f'root0_J{k:03}.npy']
            store._write(name,data);entry['array']['file']=name
        store.event(dict(kind='root_started',root_slot=root,task_id=s['task_id']))
        events=[];saved=m.SavedBackend(row,events.append,lambda d:np.load(output/d['file'],allow_pickle=False))
        rebuilt=m.plan(s,saved);saved.complete()
        for e in events:store.event(dict(root_slot=root,event=e))
        row.update(status='sealed',result=rebuilt,plans=[m.model_plan(p,s,rebuilt,a) for a in m.ARMS])
        store.put(f'root{root}.json.gz',row)
        store.event(dict(kind='root_sealed',root_slot=root,selections=[x['selection_digest'] for x in row['plans']]))
        summaries.append(dict(root_slot=root,condition='off',status='sealed',candidate_denominator=[42,42],plan_denominator=2,
            file=f'root{root}.json.gz',predict=m.FORWARDS,jacobian=m.JACS))
    store.close();report=dict(schema=m.NAME+'.report',status='all4_full15_plans_sealed_not_admitted',prepared='/RAM_ONLY/prepared.json',prepare_sha256='c'*64,
        contract=m.contract(),rows=summaries,errors=[],first_error=None,model_before='a'*64,model_after='a'*64,model_unchanged=True,source_verified=True,
        P1_admission=False,P2_mechanism=False,physical_release=False,
        actual_resources=dict(predict_attempted=2136,predict_completed=2136,jacobian_attempted=170,jacobian_completed=170),
        artifact_sha256={f.name:m.sha(f) for f in output.iterdir()})
    store.put('report.json',report,final=True);return output,report,p


@pytest.fixture
def bundle(saved_bundle,tmp_path,monkeypatch):
    source,report,p=saved_bundle;output=tmp_path/(m.NAME+'_RAM')
    shutil.copytree(source,output)  # Copy only this module's synthetic temp evidence.
    monkeypatch.setattr(m,'verify',lambda *args:deepcopy(p))
    proxy=NS(**vars(m.loader));proxy.validate_loader=lambda receipt,p,h: m.require(receipt=={'files':[]},'RAM fixture loader')
    monkeypatch.setattr(m,'loader',proxy);return output,deepcopy(report)


def test_full_actual_consumer_synthetic_callbacks_and_physical_handoff(bundle):
    output,report=bundle;value=m.consume(output,m.sha(output/'report.json'))
    assert len(value['plans'])==4 and not value['P2_mechanism']
    for i,plan in enumerate(value['plans']):
        assert plan['root_slot']==i//2 and plan['arm']==m.ARMS[i%2] and len(plan['commands'])==15
        assert len(plan['edges'])==3 and plan['compiled_nodes']==([0] if i%2==0 else [0,5,10])
        assert plan['raw_float32_bits']==np.asarray(plan['commands'],'<f4').tobytes().hex()
        assert plan['root_warm_raw10']==slot()['warm_raw10'] and plan['physical_release'] is False


@pytest.mark.parametrize('change',['failure','denominator','model','claim','extra','rowcost','rowskip','ADfile'])
def test_full_saved_consume_mutations(bundle,change):
    output,report=bundle
    if change=='failure':report['errors']=[{'message':'failure'}]
    elif change=='denominator':report['rows'][0]['candidate_denominator']=[41,42]
    elif change=='model':report['model_after']='f'*64
    elif change=='claim':report['P2_mechanism']=True
    elif change=='extra':(output/'extra').write_bytes(b'unknown')
    elif change=='ADfile':(output/'root0_J000.npy').write_bytes(b'bad')
    else:
        row=m.parent.overlay.read(output/'root0.json.gz',m.CAP)
        if change=='rowcost':row['result']['candidates'][m.ARMS[1]][0]['cost']=100
        else:row['forwards'].pop()
        c=m.zlib.compressobj(6,m.zlib.DEFLATED,31);(output/'root0.json.gz').write_bytes(c.compress(m.raw(row))+c.flush())
        report['artifact_sha256']['root0.json.gz']=m.sha(output/'root0.json.gz')
    (output/'report.json').write_bytes(m.raw(report))
    with pytest.raises((ValueError,IndexError)):m.consume(output,m.sha(output/'report.json'))


@pytest.mark.parametrize('change',['splice','unknown','blank','chain'])
def test_full_event_rehash_does_not_hide_lifecycle_splice(bundle,change):
    output,report=bundle;events=m.journal(output)
    if change=='splice':events[5],events[6]=events[6],events[5]
    elif change=='unknown':events.insert(5,dict(kind='unknown'))
    previous='0'*64;rows=[]
    for k,payload in enumerate(events):
        e=dict(index=k,previous_sha256=previous,payload=payload);previous=m.digest(e)
        rows.append(m.raw(dict(e,event_sha256=previous)))
    data=b''.join(rows)
    if change=='blank':data=data.replace(b'\n',b'\n\n',1)
    elif change=='chain':data=data.replace(b'"index":0',b'"index":7',1)
    c=m.zlib.compressobj(6,m.zlib.DEFLATED,31);(output/'events.jsonl.gz').write_bytes(c.compress(data)+c.flush())
    report['artifact_sha256']['events.jsonl.gz']=m.sha(output/'events.jsonl.gz');(output/'report.json').write_bytes(m.raw(report))
    with pytest.raises(ValueError):m.consume(output,m.sha(output/'report.json'))


def test_timeout_final_path_writes_failure_without_normal_time_gate(tmp_path):
    store=m.Store(tmp_path/(m.NAME+'_timeout'),time.monotonic());store.event(dict(kind='before_timeout'));store.started-=m.WALL+1
    with pytest.raises(ValueError,match='wall'):store.event(dict(kind='not_executed'))
    store.close();store.put('report.json',dict(status='failed_partial',completed=False),final=True)
    assert json.loads((store.output/'report.json').read_bytes())['completed'] is False
    assert m.journal(store.output)==[dict(kind='before_timeout')]

