"""RAM/native-shaped synthetic files only; never torch or production consume.

Only this NEW module's N/source boundary changes in fixtures. Frozen modules
are not monkeypatched. Real inherited hooks/Store/journal run on fake tensors.
"""
from copy import deepcopy
import io
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_early_history_factorial_20261029_v1 as m
from test_agent_p5_short_start_cuda_probe_20261029_v1 import FixtureBackend,Tensor

ORIGINAL_BUILD=m.build


def npy(directory,name,a):
    b=io.BytesIO();np.save(b,a,allow_pickle=False);p=directory/name;p.write_bytes(b.getvalue())
    return dict(file=name,sha256=m.sha(p),file_bytes=p.stat().st_size,dtype=str(a.dtype),shape=list(a.shape),nbytes=a.nbytes)


def reference(directory,name,a):return dict(directory=str(directory),array=npy(directory,name,a))


@pytest.fixture
def case(tmp_path,monkeypatch):
    root=tmp_path/'inner';(root/'results').mkdir(parents=True);(root/'innerdocs').mkdir();(root/'tests').mkdir()
    source=root/'saved';source.mkdir()
    monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'N',1);monkeypatch.setattr(m,'pair_schedule',lambda:[])
    doc=root/'innerdocs/agent_p5_next_identifiability_protocol_20261029_v1.md';doc.write_text('synthetic protocol marker')
    monkeypatch.setattr(m,'PROTOCOL_SHA',m.sha(doc))
    (root/'innerdocs'/(m.NAME+'.md')).write_text('synthetic innerdoc')
    (root/'tests'/('test_'+m.NAME+'.py')).write_text('synthetic test marker')
    z=lambda shape,v:np.full(shape,v,np.float32)
    actual=(z(m.SHAPES[0],1),np.arange(30,dtype=np.float32).reshape(1,3,10),z(m.SHAPES[2],2))
    rolled=(z(m.SHAPES[0],3),actual[1].copy(),z(m.SHAPES[2],4))
    aa=[reference(source,f'actual{k}.npy',a) for k,a in enumerate(actual)]
    rr=[reference(source,f'rolled{k}.npy',a) for k,a in enumerate(rolled)]
    causes=[]
    for j,t in enumerate((1,2,3)):
        causes.append(dict(visual=reference(source,f'causalv{j}.npy',z((1,t,256,384),3)),
            proprio=reference(source,f'causalp{j}.npy',z((1,t,256,16),4))))
    outputs=dict(visual=reference(source,'oldoutv.npy',rolled[0].reshape(1,3,256,384)+np.float32(.25)),
        proprio=reference(source,'oldoutp.npy',rolled[2]+np.float32(.5)))
    label=reference(source,'labels.npy',z((2,256,384),6))
    identity=dict(index=0,root_slot=0,condition='off',task_id='task-fixture',selection_digest='select-fixture')
    req=dict(identity,actual_inputs=aa,rolled_inputs=rr,causal_outputs=causes,RR_outputs=outputs,label=label,label_png={'fixture':'20'})
    req['input_census']=m.causal_check(req)
    manifest=dict(schema=m.NAME+'.prepare',status='prepared_no_neural_execution',protocol=m.protocol(),requests=[req],
        record=dict(expected_model={'fixture':'immutable weights'},expected_strict_loads={'fixture':'strict'}),
        bridge_prepared='/fixture/never_loaded',bridge_prepare_sha256='a'*64,
        bindings={str(p):m.sha(p) for p in source.iterdir()},inherited={'fixture':'complete prior receipts, not real science'})
    monkeypatch.setattr(m,'build',lambda:deepcopy(manifest))
    prepared=root/'results'/(m.NAME+'_prepare_fixture.json');meta=m.write_new(prepared,manifest)
    return NS(root=root,source=source,m=manifest,req=req,actual=actual,rolled=rolled,prepared=prepared,expected=meta['sha256'],
        out=root/'results'/(m.NAME+'_fixture'))


def perform(case,backend=FixtureBackend):
    v=m.execute(case.m,case.prepared,case.expected,case.out,backend_factory=backend)
    return v,m.evidence.read(case.out/'report.json',v['sha256'],{})


def test_fixed656_1968_caps_and_all80_prior_protocol_unchanged():
    assert m.N==164 and len(m.pair_schedule())==80
    assert m.units()==dict(api=656,batch_samples=656,frame_samples=1968)
    p=m.protocol()
    assert p['native_IO_bytes']==1612264320 and p['label_tensor_bytes']==128974848 and p['total_tensor_bytes']==1741239168
    assert p['total_tensor_bytes']<m.TENSOR_CAP and m.CAP==3*1024**3 and m.WALL==900
    assert p['cells']==['AA','AR','RA','RR'] and p['new_encoder']==p['new_physics']==p['new_training']==0
    assert m.short.units()==dict(api=656,batch_samples=656,frame_samples=1476)  # old guard unchanged
    assert not p['prior_fullstrict_rebuilt']


def test_656_actual_T3_guard_no_padding_retry():
    g=m.Meter()
    for i in range(656):
        w=g.before('predictor',list(m.SHAPES),['cuda:0']*3,['float32']*3,[False]*3,True,True,True)
        assert w==dict(stage=3,units=dict(api=1,batch_samples=1,frame_samples=3))
        g.after([(1,3,256,384),None,(1,3,256,16)],['cuda:0']*2,['float32']*2)
    assert g.attempted==g.returned==m.units()
    with pytest.raises(ValueError):g.before('predictor',list(m.SHAPES),['cuda:0']*3,['float32']*3,[False]*3,True,True,True)


@pytest.mark.parametrize('cell',m.CELLS)
def test_splice_both_early_slots_latest_and_all_actions_bitwise(case,cell):
    a,r=case.actual,case.rolled;before=[x.tobytes() for block in (a,r) for x in block]
    args=m.splice(a,r,cell);w=m.witness(case.req,cell,args)
    assert w['early_sources']==list(cell) and w['frames']==[5,10,15]
    assert m.exact(args[0][:,:2],(a if cell[0]=='A' else r)[0][:,:2])
    assert m.exact(args[2][:,:2],(a if cell[1]=='A' else r)[2][:,:2])
    assert m.exact(args[0][:,2:],r[0][:,2:]) and m.exact(args[2][:,2:],r[2][:,2:]) and m.exact(args[1],r[1])
    assert [x.tobytes() for block in (a,r) for x in block]==before


@pytest.mark.parametrize('fault',('action','shape','dtype','nan','cell','latest'))
def test_wrong_input_contract_rejected(case,fault):
    a=[x.copy() for x in case.actual];r=[x.copy() for x in case.rolled]
    if fault=='action':a[1][0,0,0]+=1
    elif fault=='shape':a[0]=a[0][:,:2]
    elif fault=='dtype':a[0]=a[0].astype(np.float64)
    elif fault=='nan':a[0][0,0,0,0,0,0]=np.nan
    if fault=='latest':
        args=list(m.splice(a,r,'AA'));args[0][0,2,0,0,0,0]+=1
        with pytest.raises(ValueError):m.witness(case.req,'AA',args)
    else:
        with pytest.raises(ValueError):m.splice(a,r,'unknown' if fault=='cell' else 'AA')


def test_real_hook_store_complete_run_seal_scoring_and_own_strict(case):
    v,r=perform(case)
    assert r['status']=='all164_early_history_factorial_recorded' and len(r['calls'])==4
    assert r['tensor_bytes']==10617312 and len(list(case.out.glob('tensor_*.npy')))==21
    assert r['rows'][0]['RR_verified']==dict(inputs=[True]*3,full3_outputs=[True]*3)
    assert r['counts']==dict(attempted=m.units(),returned=m.units(),validated=m.units())
    events=list(m.codec.journal(case.out/'journal.jsonl'));kinds=[e['kind'] for e in events]
    assert kinds.count('outputs_saved')==4 and kinds.index('label_read_intent')>kinds.index('all_predictions_sealed')
    assert r['summaries'][0]['branches']==1 and sum(x['branches'] for x in r['summaries'][1:])==1
    c=m.consume(case.out,v['sha256'])
    assert c['status']=='all164_early_history_four_NEW_cells_and80_pairs_verified' and c['RR_all_native_IO_bitwise_verified']


@pytest.mark.parametrize('fault',('dispatch','return','RR_byte'))
def test_failed_fourth_call_preserves_denominators_no_label_decode(case,fault):
    class Broken(FixtureBackend):
        fail_stage=4 if fault=='dispatch' else None
        bad_stage=4 if fault=='return' else None
    if fault=='RR_byte':
        old=case.req['RR_outputs']['visual'];a=m.load_ref(old);a[0,0,0,0]=np.nextafter(a[0,0,0,0],np.float32(np.inf))
        case.req['RR_outputs']['visual']=reference(case.source,'wrongRR.npy',a)
        case.m['bindings'][str(case.source/'wrongRR.npy')]=m.sha(case.source/'wrongRR.npy')
    v,r=perform(case,Broken)
    assert r['status']=='partial_or_failed' and set(r['rows'][0]['predictions'])==set(m.CELLS)
    assert r['rows'][0]['label'] is r['rows'][0]['score'] is r['prediction_seal'] is None
    assert r['counts']['attempted']['api']==4 and r['counts']['returned']['api']==(3 if fault=='dispatch' else 4)
    assert r['counts']['validated']['api']==(3 if fault in ('dispatch','return') else 4)
    assert 'label_read_intent' not in [e['kind'] for e in m.codec.journal(case.out/'journal.jsonl')]
    assert m.consume(case.out,v['sha256'])['status']=='blocked_partial_factorial'


def test_seal_missing_656th_fails_before_opening_label_or_seal(monkeypatch):
    rows=[dict(index=i,predictions={c:{'fixture':'fulltree'} for c in m.CELLS},RR_verified={'fixture':'true'}) for i in range(164)]
    rows[-1]['predictions']['RR']=None
    with pytest.raises(ValueError):next(m.labels_after_seal({},rows,NS(path='/absent'),{},'0'*64))


@pytest.mark.parametrize('fault',('CPU','encoder','dtype','grad','shape','crossdevice','TF32'))
def test_all_model_hooks_block_encoder_and_bad_actual_ABI(case,fault):
    s=m.Store(case.out);trace=m.Trace(s);b=FixtureBackend(case.m,trace);args=[Tensor(x) for x in case.rolled];model=b.models['predictor']
    if fault=='CPU':args[0].device='cpu'
    elif fault=='encoder':model=b.models['official_encoder']
    elif fault=='dtype':args[0].dtype='float64'
    elif fault=='grad':args[0].requires_grad=True
    elif fault=='shape':args[0].shape=(1,2,1,16,16,384)
    elif fault=='crossdevice':args[1].device='cuda:1'
    else:
        with pytest.raises(ValueError):trace.guard.before('predictor',list(m.SHAPES),['cuda:0']*3,['float32']*3,[False]*3,True,False,True)
        assert trace.guard.attempted['api']==0;trace.close();return
    with pytest.raises(ValueError):model(*args)
    assert trace.guard.attempted['api']==0 and not trace.close()


def test_real_Store_two_event_exactLF_chain_and_aggregate_caps(case):
    s=m.Store(case.out);s.emit('first',n=1);s.emit('second',n=2)
    body=(case.out/'journal.jsonl').read_bytes()
    assert body.count(b'\n')==2 and b'\n\n' not in body
    assert [e['kind'] for e in m.codec.journal(case.out/'journal.jsonl')]==['first','second']
    assert s.artifacts['journal.jsonl']['sha256']==m.sha(case.out/'journal.jsonl')
    with pytest.raises(ValueError):s.check(m.CAP)
    with pytest.raises(ValueError):s.put('../escape',b'x')
    with pytest.raises(ValueError):s.array(np.zeros(1,np.float64))
    with pytest.raises(ValueError):m.Store(case.out)
    (case.out/'journal.jsonl').write_bytes(body+b'\n')
    with pytest.raises(ValueError):list(m.codec.journal(case.out/'journal.jsonl'))


def test_factorial_exact_scalar_and_vector_arithmetic():
    x=dict(AA=1.,AR=3.,RA=4.,RR=10.)
    assert m.effects(x)==dict(visual=5.,proprio=4.,interaction=4.,joint=9.)
    with pytest.raises(ValueError):m.effects(dict(AA=1,RR=2))
    a={k:np.full((1,256,384),v,np.float32) for k,v in dict(AA=1,AR=2,RA=3,RR=4).items()}
    s=m.row_score(a,np.zeros((1,256,384),np.float32))
    assert s['MSE']==dict(AA=1.,AR=4.,RA=9.,RR=16.)
    assert s['MSE_effects']==dict(visual=10.,proprio=5.,interaction=4.,joint=15.)


def test_all80_zero_null_and_pair_central_difference_no_epsilon():
    z=np.zeros((1,256,384),np.float32);one=z.copy();one[0,0,0]=1
    a={c:z for c in m.CELLS};b={c:one if c!='AA' else z for c in m.CELLS}
    for s in m.pair_schedule():
        p=m.pair(s,(a,z),(b,z))
        assert p['missing']==[] and all(v['cosine'] is None and v['cosine_sign']=='null' for v in p['cells'].values())
        assert p['cells']['RR']['central_predicted_L2']==1/(2*s['h'])
        p=m.pair(s,(a,z),(b,one))
        assert p['cells']['AA']['cosine'] is None and p['cells']['RR']['cosine']==1.
    neg={c:-one for c in m.CELLS}
    assert m.pair(m.pair_schedule()[0],(a,z),(neg,one))['cells']['RR']['cosine']==-1.


def test_all164_fourstrata_no_empty_stratum_pruning():
    rows=[]
    for i in range(164):
        root,cond=((0,'off'),(0,'on'),(1,'off'),(1,'on'))[i//41]
        rows.append(dict(root_slot=root,condition=cond,score=dict(MSE={c:1. for c in m.CELLS},MSE_effects={k:0. for k in ('visual','proprio','interaction','joint')})))
    z=np.zeros((1,256,384),np.float32);cs={c:z for c in m.CELLS}
    ps=[m.pair(s,(cs,z),(cs,z)) for s in m.pair_schedule()];ss=m.summaries(rows,ps)
    assert [(g['branches'],g['pairs']) for g in ss]==[(164,80)]+[(41,20)]*4
    assert ss[0]['responses']['AA']['signs']==dict(positive=0,negative=0,zero=0,null=80)
    assert ss[0]['sign_transitions']['AA__to__RR']['null__to__null']==80


@pytest.mark.parametrize('fault',('artifact','extra','counts','row','seal','event','model','input','RR','score','summary','source','fused'))
def test_consumer_rejects_integrity_and_semantic_tampering(case,fault):
    v,r=perform(case)
    if fault=='artifact':(case.out/'tensor_00000.npy').write_bytes(b'bad')
    elif fault=='extra':(case.out/'unexpected').write_bytes(b'x')
    elif fault=='counts':r['counts']['returned']['api']=3
    elif fault=='row':r['rows'][0]['task_id']='wrong'
    elif fault=='seal':r['prediction_seal']['sha256']='0'*64
    elif fault=='event':r['calls'][0]['requested_MATH']=False
    elif fault=='model':r['model_after']={'wrong':'weights'}
    elif fault=='input':r['calls'][0]['inputs']['$tuple'][0]['$tuple'][0]=r['calls'][1]['inputs']['$tuple'][0]['$tuple'][0]
    elif fault=='RR':r['rows'][0]['RR_verified']['full3_outputs'][0]=False
    elif fault=='score':r['rows'][0]['score']['MSE']['AA']+=1
    elif fault=='summary':r['summaries'][0]['mean_MSE']['RR']+=1
    elif fault=='source':r['imported_sources']={'wrong':'/missing/source.py'}
    else:r['attention_ops']={'aten._scaled_dot_product_flash_attention.default':1}
    (case.out/'report.json').write_bytes(m.raw(r));h=m.sha(case.out/'report.json')
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):m.consume(case.out,h)


def test_prepare_verify_checks_entire_own_bytes_and_exclusive_output(case):
    assert m.verify(case.prepared,case.expected)==case.m
    with pytest.raises(ValueError):m.write_new(case.prepared,case.m)
    with pytest.raises(ValueError):m.run(case.prepared,case.expected,case.out,'not_authorized')
    assert not case.out.exists()
    (case.prepared).write_bytes(m.raw(dict(case.m,status='wrong')))
    with pytest.raises(ValueError):m.verify(case.prepared,m.sha(case.prepared))


def synthetic_bundles(case):
    """Complete synthetic boundary metadata for the REAL new build function."""
    req=case.req;identity={k:req[k] for k in ('index','root_slot','condition','task_id','selection_digest')}
    desc=lambda ref:deepcopy(ref['array'])
    tree=lambda refs:{'$tuple':[{'$tuple':[{'$array':desc(x)} for x in refs]},{'$dict':{}}]}
    out=lambda v,p:{'$tuple':[{'$array':desc(v)},None,{'$array':desc(p)}]}
    packets=np.concatenate([np.zeros((1,10),np.float32),case.rolled[1][0]],axis=0).tolist()
    norm=dict(commands=[[0.,0.]]*20,packets=packets,raw_proprio=[[1.,2.,3.,4.]]*3,stats={'fixture':'fixed'})
    q=dict(identity,context_global_frames=[0,5,10],successor_global_frames=[15,20],commands=norm['commands'],
        packets=packets,raw_proprio=norm['raw_proprio'],labels=[{'fixture':'15'},req['label_png']])
    record=dict(identity,parent_frames=[0,5,10],source_normalization=norm)
    sr=dict(index=0,identity={k:identity[k] for k in identity if k!='index'},labels=[None,desc(req['label'])],stages=[])
    sc=[]
    for j,c in enumerate(req['causal_outputs']):
        outs=out(c['visual'],c['proprio']);sc.append(dict(outputs=outs))
        sr['stages'].append(dict(target=(5,10,15)[j],frames=list(m.short.FRAMES[j]),call=j,prediction=outs))
    last=dict(index=3,status='returned',units=dict(api=1,batch_samples=1,frame_samples=3),inputs=tree(req['rolled_inputs']),
        outputs=out(req['RR_outputs']['visual'],req['RR_outputs']['proprio']))
    sc.append(last);sr['stages'].append(dict(target=20,frames=[5,10,15],call=3,prediction=last['outputs']))
    ac=deepcopy(last);ac.update(branch=0,module='predictor',phase='prediction',inputs=tree(req['actual_inputs']))
    artifacts={p.name:dict(sha256=m.sha(p),file_bytes=p.stat().st_size) for p in case.source.iterdir()}
    common=dict(model_before={'fixture':'model'},strict_loads={'fixture':'schema'},protocol={'pairs':[]},artifacts=artifacts)
    bs=dict(directory=case.source,report=dict(common,rows=[sr],calls=sc),
        prepared=dict(authority=dict(rows=[dict(record=record,labels=[None,dict(array=desc(req['label']))])]),
            record=case.m['record'],bridge_prepared='fixture',bridge_prepare_sha256='a'*64),resources={'fixture':'source'})
    bh=dict(directory=case.source,report=dict(common,rows=[dict(identity,label_patch=desc(req['label']))],
        calls=[{}, {}, {},ac,dict(outputs={'$dict':{'x_norm_patchtokens':{'$array':desc(req['label'])}}})]),
        prepared=dict(bridge=dict(requests=[q],stats=norm['stats'])),resources={'fixture':'source'})
    return bs,bh


@pytest.mark.parametrize('fault',(None,'time','branch','action','normalizer','lasttoken','all_equal_visual','all_equal_proprio','label_ref'))
def test_real_build_join_and_identifiability_census_no_label_decode(case,monkeypatch,fault):
    bs,bh=synthetic_bundles(case)
    if fault=='time':bh['prepared']['bridge']['requests'][0]['context_global_frames']=[5,10,15]
    elif fault=='branch':bh['report']['rows'][0]['task_id']='different'
    elif fault=='action':bh['prepared']['bridge']['requests'][0]['packets'][1][0]+=1
    elif fault=='normalizer':bh['prepared']['bridge']['stats']={'fixture':'wrong'}
    elif fault=='label_ref':bs['report']['rows'][0]['labels'][1]['sha256']='0'*64
    elif fault=='lasttoken':
        ref=bs['report']['calls'][0]['outputs']['$tuple'][0]['$array']
        wrong=np.zeros((1,1,256,384),np.float32);ref.update(npy(case.source,'wrongstage.npy',wrong))
        bs['report']['artifacts'][ref['file']]={k:ref[k] for k in ('sha256','file_bytes')}
    elif fault in ('all_equal_visual','all_equal_proprio'):
        j=0 if fault=='all_equal_visual' else 2
        bh['report']['calls'][3]['inputs']['$tuple'][0]['$tuple'][j]=deepcopy(bs['report']['calls'][3]['inputs']['$tuple'][0]['$tuple'][j])
    monkeypatch.setattr(m,'bundles',lambda used:(deepcopy(bs),deepcopy(bh)))
    orig=m.load_ref
    def no_future(ref):
        assert ref['array']['file']!='labels.npy','future numerical label access during prepare'
        return orig(ref)
    monkeypatch.setattr(m,'load_ref',no_future)
    if fault:
        with pytest.raises((ValueError,KeyError)):ORIGINAL_BUILD()
    else:
        p=ORIGINAL_BUILD();assert p['requests'][0]['input_census']==dict(index=0,visual_early_equal=False,proprio_early_equal=False)
        assert p['inherited']['prior_fullstrict_rebuilt'] is False and p['protocol']['budget_mode']=='all_four_NEW_656'


def test_source_has_no_nested_fullstrict_or_production_outcome_read():
    import ast
    tree=ast.parse(__import__('pathlib').Path(m.__file__).read_text())
    called=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert 'consume' not in called and 'analyze' not in called and 'verify_analysis' not in called
    assert 'load_host_cuda' not in called  # only inherited frozen backend, never prepare/build
    assert len(m.CELLS)==4


def test_cli_help_no_torch_import():
    r=subprocess.run([sys.executable,'-m',m.__name__,'--help'],capture_output=True,text=True)
    assert r.returncode==0 and all(x in r.stdout for x in ('prepare','verify','run','consume'))
    r=subprocess.run([sys.executable,'-c',f'import sys; import {m.__name__}; assert "torch" not in sys.modules'],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
