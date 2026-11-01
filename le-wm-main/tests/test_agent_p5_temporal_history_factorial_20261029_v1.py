"""Synthetic saved/native-shaped RAM fixtures; no model/physics production.

Only NEW module N/ROOT/prior boundary is varied. No frozen globals patched.
"""
import ast
from copy import deepcopy
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_temporal_history_factorial_20261029_v1 as m
from test_agent_p5_short_start_cuda_probe_20261029_v1 import FixtureBackend,Tensor

REAL_BUILD=m.build
REAL_PRIOR=m.prior


def array(d,name,a):
    b=io.BytesIO();np.save(b,a,allow_pickle=False);p=d/name;p.write_bytes(b.getvalue())
    return dict(file=name,sha256=m.sha(p),file_bytes=p.stat().st_size,shape=list(a.shape),dtype=str(a.dtype),nbytes=a.nbytes)


def ref(d,name,a):return dict(directory=str(d),array=array(d,name,a))


@pytest.fixture
def case(tmp_path,monkeypatch):
    root=tmp_path/'inner';(root/'results').mkdir(parents=True);(root/'tests').mkdir();(root/'innerdocs').mkdir()
    source=root/'saved';source.mkdir();monkeypatch.setattr(m,'ROOT',root);monkeypatch.setattr(m,'N',1);monkeypatch.setattr(m,'pair_schedule',lambda:[])
    for path in (root/'tests'/('test_'+m.NAME+'.py'),root/'innerdocs'/(m.NAME+'.md')):path.write_text('synthetic own binding')
    doc=root/'innerdocs/agent_p5_factorial_next_identifiability_20261029_v1.md';doc.write_text('synthetic prospective contract')
    monkeypatch.setattr(m,'PROTOCOL_SHA',m.sha(doc))
    full=lambda s,v:np.full(s,v,np.float32)
    rr=(full(m.SHAPES[0],3),np.arange(30,dtype=np.float32).reshape(1,3,10),full(m.SHAPES[2],4))
    aa=tuple(x.copy() for x in rr);aa[0][:,0]=1;aa[0][:,1]=2;aa[2][:,0]=5;aa[2][:,1]=6
    a_refs=[ref(source,f'a{k}.npy',x) for k,x in enumerate(aa)];r_refs=[ref(source,f'r{k}.npy',x) for k,x in enumerate(rr)]
    outputs=lambda args,prefix:dict(visual=ref(source,prefix+'v.npy',args[0].reshape(1,3,256,384)+np.float32(.25)),
        proprio=ref(source,prefix+'p.npy',args[2]+np.float32(.5)))
    req=dict(index=0,root_slot=0,condition='off',task_id='fixture-task',selection_digest='fixture-selection',AA_inputs=a_refs,RR_inputs=r_refs,
        AA_outputs=outputs(aa,'aa'),RR_outputs=outputs(rr,'rr'),label=ref(source,'label.npy',full((2,256,384),7)))
    req['input_census']=m.census(req)
    manifest=dict(schema=m.NAME+'.prepare',status='prepared_no_neural_execution',protocol=m.protocol(),requests=[req],
        record=dict(expected_model={'fixture':'frozenweights'},expected_strict_loads={'fixture':'strict'}),
        bridge_prepared='synthetic-no-loader',bridge_prepare_sha256='a'*64,bindings={str(p):m.sha(p) for p in source.iterdir()},inherited={'fixture':'not real authority'})
    monkeypatch.setattr(m,'build',lambda:deepcopy(manifest))
    prepared=root/'results'/(m.NAME+'_prepare_fixture.json');v=m.write_new(prepared,manifest)
    return NS(root=root,source=source,req=req,m=manifest,aa=aa,rr=rr,prepared=prepared,expected=v['sha256'],out=root/'results'/(m.NAME+'_fixture'))


def perform(case,backend=FixtureBackend):
    v=m.execute(case.m,case.prepared,case.expected,case.out,backend_factory=backend)
    return v,m.read(case.out/'report.json',v['sha256'])


def test_fixed_492_1476_80_and_caps_no_old_globals_change():
    assert m.N==164 and len(m.pair_schedule())==80 and m.units()==dict(api=492,batch_samples=492,frame_samples=1476)
    p=m.protocol();assert p['budget_mode']=='predeclared_328_plus164_RR_492'
    assert p['native_IO_bytes']==1209198240 and p['label_tensor_bytes']==128974848 and p['total_tensor_bytes']==1338173088
    assert p['total_tensor_bytes']<m.TENSOR_CAP and m.CAP==3*1024**3 and m.WALL==900
    assert m.early.N==164 and m.early.units()==dict(api=656,batch_samples=656,frame_samples=1968)
    assert p['new_encoder']==p['new_physics']==p['new_training']==0


def test_guard_exact492_T3_calls_and_extra_call_rejected():
    g=m.Meter()
    for i in range(492):
        assert g.before('predictor',list(m.SHAPES),['cuda:0']*3,['float32']*3,[False]*3,True,True,True)==dict(stage=3,units=dict(api=1,batch_samples=1,frame_samples=3))
        g.after([(1,3,256,384),None,(1,3,256,16)],['cuda:0']*2,['float32']*2)
    assert g.attempted==g.returned==m.units()
    with pytest.raises(ValueError):g.before('predictor',list(m.SHAPES),['cuda:0']*3,['float32']*3,[False]*3,True,True,True)


@pytest.mark.parametrize('cell',m.CELLS)
def test_temporal_slots_pair_visual_proprio_latest_and_actions(case,cell):
    a,r=case.aa,case.rr;args=m.splice(a,r,cell);w=m.witness(case.req,cell,args)
    assert w['slot5_source']==cell[2] and w['slot10_source']==cell[3]
    for j in (0,1):
        src=a if cell[2+j]=='A' else r
        assert m.exact(args[0][:,j:j+1],src[0][:,j:j+1]) and m.exact(args[2][:,j:j+1],src[2][:,j:j+1])
    assert m.exact(args[0][:,2:],r[0][:,2:]) and m.exact(args[2][:,2:],r[2][:,2:]) and m.exact(args[1],r[1])
    assert not any(m.exact(args[0],x[0]) and m.exact(args[2],x[2]) for x in (a,r)) if cell in m.ARMS[:2] else True
    # Temporal mixed corners are not old modality mixed corners.
    if cell in m.ARMS[:2]:
        for old in ('AR','RA'):
            other=m.early.splice(a,r,old)
            assert not all(m.exact(x,y) for x,y in zip(args,other,strict=True))


@pytest.mark.parametrize('fault',('action','latest_v','latest_p','shape','dtype','nan','old_cell','witness'))
def test_wrong_history_contract_fail_closed(case,fault):
    a=[x.copy() for x in case.aa];r=[x.copy() for x in case.rr]
    if fault=='action':a[1][0,0,0]+=1
    elif fault=='latest_v':a[0][:,2]+=1
    elif fault=='latest_p':a[2][:,2]+=1
    elif fault=='shape':a[0]=a[0][:,:2]
    elif fault=='dtype':r[0]=r[0].astype(np.float64)
    elif fault=='nan':a[0][0,0,0,0,0,0]=np.nan
    if fault=='witness':
        args=list(m.splice(a,r,'T_AR'));args[2][0,0,0,0]+=1
        with pytest.raises(ValueError):m.witness(case.req,'T_AR',args)
    else:
        with pytest.raises(ValueError):m.splice(a,r,'AR' if fault=='old_cell' else 'T_AR')


def test_real_store_hooks_3calls_fullseal_scores_strict(case):
    v,r=perform(case)
    assert r['status']=='all164_temporal_history_492_recorded' and len(r['calls'])==3
    assert r['tensor_bytes']==8159592 and len(list(case.out.glob('tensor_*.npy')))==16
    assert set(r['rows'][0]['predictions'])==set(m.ARMS) and 'T_AA' not in r['rows'][0]['predictions']
    assert r['rows'][0]['RR_verified']==dict(inputs=[True]*3,full3_outputs=[True]*3)
    assert r['counts']==dict(attempted=m.units(),returned=m.units(),validated=m.units())
    events=list(m.codec.journal(case.out/'journal.jsonl'));kinds=[e['kind'] for e in events]
    assert [x['cell'] for x in events if x['kind']=='history_ready']==list(m.ARMS)
    assert kinds.index('label_read_intent')>kinds.index('all_predictions_sealed')>max(i for i,k in enumerate(kinds) if k=='outputs_saved')
    c=m.consume(case.out,v['sha256']);assert c['status']=='all164_temporal_two_corners_RR_bridge_and80_pairs_verified'
    assert c['predictor']==m.units() and c['inherited_AA']


@pytest.mark.parametrize('fault',('dispatch','bad_return','RR_byte'))
def test_failure_at_last_call_no_label_keeps_denominators(case,fault):
    class Broken(FixtureBackend):
        fail_stage=3 if fault=='dispatch' else None
        bad_stage=3 if fault=='bad_return' else None
    if fault=='RR_byte':
        a=m.load_ref(case.req['RR_outputs']['visual']);a[0,0,0,0]=np.nextafter(a[0,0,0,0],np.float32(np.inf))
        case.req['RR_outputs']['visual']=ref(case.source,'badRR.npy',a)
    v,r=perform(case,Broken)
    assert r['status']=='partial_or_failed' and set(r['rows'][0]['predictions'])==set(m.ARMS)
    assert r['rows'][0]['label'] is r['rows'][0]['score'] is r['prediction_seal'] is None
    assert r['counts']['attempted']['api']==3 and r['counts']['returned']['api']==(2 if fault=='dispatch' else 3)
    assert r['counts']['validated']['api']==(2 if fault in ('dispatch','bad_return') else 3)
    assert 'label_read_intent' not in [e['kind'] for e in m.codec.journal(case.out/'journal.jsonl')]
    assert m.consume(case.out,v['sha256'])['status']=='blocked_partial_temporal'


def test_missing_492nd_blocks_label_and_seal_open():
    rows=[dict(index=i,predictions={c:{'fixture':'complete'} for c in m.ARMS},inherited_T_AA={},RR_verified={}) for i in range(164)]
    rows[-1]['predictions']['T_RR']=None
    with pytest.raises(ValueError):next(m.labels_after_seal({},rows,NS(path='/unreachable'),{},'0'*64))


@pytest.mark.parametrize('fault',('CPU','encoder','dtype','grad','shape','device','TF32','inference'))
def test_real_all_model_hooks_native_dispatch_guards(case,fault):
    store=m.Store(case.out);trace=m.Trace(store);b=FixtureBackend(case.m,trace);args=[Tensor(x) for x in case.rr];model=b.models['predictor']
    if fault=='CPU':args[0].device='cpu'
    elif fault=='encoder':model=b.models['proprio_encoder']
    elif fault=='dtype':args[0].dtype='float64'
    elif fault=='grad':args[0].requires_grad=True
    elif fault=='shape':args[0].shape=(1,2,1,16,16,384)
    elif fault=='device':args[2].device='cuda:1'
    else:
        with pytest.raises(ValueError):trace.guard.before('predictor',list(m.SHAPES),['cuda:0']*3,['float32']*3,[False]*3,fault!='inference',fault!='TF32',True)
        assert not trace.close();return
    with pytest.raises(ValueError):model(*args)
    assert trace.guard.attempted['api']==0 and not trace.close()


def test_real_Store_newline_hashchain_caps_floor_and_exclusive(case,monkeypatch):
    s=m.Store(case.out);s.emit('one',n=1);s.emit('two',n=2);body=(case.out/'journal.jsonl').read_bytes()
    assert body.count(b'\n')==2 and b'\n\n' not in body
    assert [e['kind'] for e in m.codec.journal(case.out/'journal.jsonl')]==['one','two']
    assert s.artifacts['journal.jsonl']['sha256']==m.sha(case.out/'journal.jsonl')
    for op in (lambda:s.check(m.CAP),lambda:s.array(np.zeros(1,np.float64)),lambda:s.put('../bad',b'x'),lambda:m.Store(case.out)):
        with pytest.raises(ValueError):op()
    monkeypatch.setattr(m,'FLOOR',10**18)
    with pytest.raises(ValueError):s.check()
    (case.out/'journal.jsonl').write_bytes(body+b'\n')
    with pytest.raises(ValueError):list(m.codec.journal(case.out/'journal.jsonl'))


def test_temporal_scalar_names_and_constant_hand_arithmetic():
    v=dict(zip(m.CELLS,(1.,3.,4.,10.),strict=True))
    assert m.effects(v)==dict(slot5=5.,slot10=4.,interaction=4.,joint=9.)
    cs={c:np.full((1,256,384),j+1,np.float32) for j,c in enumerate(m.CELLS)}
    s=m.row_score(cs,np.zeros((1,256,384),np.float32))
    assert s['MSE']==dict(zip(m.CELLS,(1.,4.,9.,16.),strict=True))
    assert s['MSE_effects']==dict(slot5=10.,slot10=5.,interaction=4.,joint=15.)


def test_all80_pairs_zero_null_positive_negative_and_fourstrata():
    z=np.zeros((1,256,384),np.float32);one=z.copy();one[0,0,0]=1
    a={c:z for c in m.CELLS};b={c:one for c in m.CELLS};neg={c:-one for c in m.CELLS}
    for s in m.pair_schedule():
        p=m.pair(s,(a,z),(b,one));assert all(x['cosine']==1 for x in p['cells'].values())
        assert p['cells']['T_RR']['central_predicted_L2']==1/(2*s['h'])
        assert all(x['cosine']==-1 for x in m.pair(s,(a,z),(neg,one))['cells'].values())
        assert all(x['cosine'] is None for x in m.pair(s,(a,z),(b,z))['cells'].values())
    rows=[dict(root_slot=((0,'off'),(0,'on'),(1,'off'),(1,'on'))[i//41][0],condition=((0,'off'),(0,'on'),(1,'off'),(1,'on'))[i//41][1],score=m.row_score(a,z)) for i in range(164)]
    ps=[m.pair(s,(a,z),(a,z)) for s in m.pair_schedule()];ss=m.summaries(rows,ps)
    assert [(x['branches'],x['pairs']) for x in ss]==[(164,80)]+[(41,20)]*4
    assert ss[0]['sign_transitions']['T_AA__to__T_RR']['null__to__null']==80


@pytest.mark.parametrize('fault',('artifact','extra','counts','row','AA','seal','event','model','RR','score','pair','summary','source','fused'))
def test_strict_rejects_tampered_new_native_history_and_scalars(case,fault):
    v,r=perform(case)
    if fault=='artifact':(case.out/'tensor_00000.npy').write_bytes(b'bad')
    elif fault=='extra':(case.out/'extra').write_bytes(b'x')
    elif fault=='counts':r['counts']['returned']['api']=2
    elif fault=='row':r['rows'][0]['task_id']='wrong'
    elif fault=='AA':r['rows'][0]['inherited_T_AA']['visual']['array']['sha256']='0'*64
    elif fault=='seal':r['prediction_seal']['sha256']='0'*64
    elif fault=='event':r['calls'][0]['requested_MATH']=False
    elif fault=='model':r['model_after']={'wrong':'weights'}
    elif fault=='RR':r['rows'][0]['RR_verified']['full3_outputs'][0]=False
    elif fault=='score':r['rows'][0]['score']['MSE_effects']['slot5']+=1
    elif fault=='pair':r['pairs']=[dict(fixture='extra')]
    elif fault=='summary':r['summaries'][0]['mean_MSE']['T_AR']+=1
    elif fault=='source':r['imported_sources']={'bad':'/absent.py'}
    else:r['attention_ops']={'aten._scaled_dot_product_flash_attention.default':1}
    (case.out/'report.json').write_bytes(m.raw(r))
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):m.consume(case.out,m.sha(case.out/'report.json'))


def old_metadata(case):
    identity={k:case.req[k] for k in ('index','root_slot','condition','task_id','selection_digest')}
    tree=lambda refs:{'$tuple':[{'$tuple':[{'$array':deepcopy(x['array'])} for x in refs]},{'$dict':{}}]}
    out=lambda refs:{'$tuple':[{'$array':deepcopy(refs['visual']['array'])},None,{'$array':deepcopy(refs['proprio']['array'])}]}
    calls=[];predictions={}
    for j,c in enumerate(m.early.CELLS):
        inputs=case.req['AA_inputs'] if j==0 else case.req['RR_inputs'];outputs=out(case.req['AA_outputs'] if j==0 else case.req['RR_outputs'])
        calls.append(dict(index=j,T=3,status='returned',units=dict(api=1,batch_samples=1,frame_samples=3),inputs=tree(inputs),outputs=outputs));predictions[c]=outputs
    row=dict(identity,predictions=predictions,RR_verified=dict(inputs=[True]*3,full3_outputs=[True]*3),label=deepcopy(case.req['label']['array']))
    q=dict(identity,actual_inputs=case.req['AA_inputs'],rolled_inputs=case.req['RR_inputs'],label=case.req['label'])
    p=dict(record=case.m['record'],requests=[q],bridge_prepared=case.m['bridge_prepared'],bridge_prepare_sha256=case.m['bridge_prepare_sha256'])
    artifacts={x.name:dict(sha256=m.sha(x),file_bytes=x.stat().st_size) for x in case.source.iterdir()}
    return case.source,dict(rows=[row],calls=calls,artifacts=artifacts,counts={'fixture':'old656'},elapsed_seconds=1.,inherited={'fixture':'upstream'}),p


@pytest.mark.parametrize('fault',(None,'identity','time_latest','action','label','call','all_duplicates'))
def test_real_cold_build_original_metadata_ABI_and_no_label_decode(case,monkeypatch,fault):
    d,r,p=old_metadata(case)
    if fault=='identity':r['rows'][0]['task_id']='wrong'
    elif fault=='time_latest':
        bad=case.aa[0].copy();bad[:,2]+=1;rec=array(case.source,'badtime.npy',bad)
        r['calls'][0]['inputs']['$tuple'][0]['$tuple'][0]['$array']=rec;r['artifacts'][rec['file']]={k:rec[k] for k in ('sha256','file_bytes')}
    elif fault=='action':
        bad=case.aa[1].copy();bad[:,0]+=1;rec=array(case.source,'badaction.npy',bad)
        r['calls'][0]['inputs']['$tuple'][0]['$tuple'][1]['$array']=rec;r['artifacts'][rec['file']]={k:rec[k] for k in ('sha256','file_bytes')}
    elif fault=='label':r['rows'][0]['label']['sha256']='0'*64
    elif fault=='call':r['calls'][3]['index']=0
    elif fault=='all_duplicates':r['calls'][0]['inputs']=deepcopy(r['calls'][3]['inputs'])
    monkeypatch.setattr(m,'prior',lambda used:(d,deepcopy(r),deepcopy(p)))
    orig=m.load_ref
    def guarded(ref):
        assert ref['array']['file']!='label.npy','future label decode during build'
        return orig(ref)
    monkeypatch.setattr(m,'load_ref',guarded)
    if fault:
        with pytest.raises((ValueError,KeyError)):REAL_BUILD()
    else:
        v=REAL_BUILD();assert len(v['requests'])==1 and v['protocol']['budget_mode']=='predeclared_328_plus164_RR_492'
        assert all(not x for x in v['requests'][0]['input_census']['new_corner_equals_old']['T_AR'].values())
        assert v['inherited']['independent_scalar_sha256']==m.SCALAR_SHA


def test_prepare_verify_exclusive_SHA_authority(case):
    assert m.verify(case.prepared,case.expected)==case.m
    with pytest.raises(ValueError):m.write_new(case.prepared,case.m)
    with pytest.raises(ValueError):m.run(case.prepared,case.expected,case.out,'not_authorized')
    assert not case.out.exists()
    case.prepared.write_bytes(m.raw(dict(case.m,status='tampered')))
    with pytest.raises(ValueError):m.verify(case.prepared,m.sha(case.prepared))


@pytest.mark.parametrize('fault',(None,'receipt','prepare','scalar','binding','journal','inventory'))
def test_prior_real_persisted_receipt_SHA_source_inventory_and_scalar_boundary(case,monkeypatch,fault):
    # Synthetic boundary files only. Keep frozen early.protocol/units untouched;
    # the NEW module denominator alone is1 for this metadata integration test.
    _,r,p=old_metadata(case)
    d=case.root/'results'/(m.early.NAME+'_cuda_MAIN01');d.mkdir()
    prep=case.root/'results'/(m.early.NAME+'_prepare_MAIN01.json')
    marker=case.root/'source-marker';marker.write_text('synthetic frozen source')
    p.update(schema=m.early.NAME+'.prepare',status='prepared_no_neural_execution',protocol=m.early.protocol(),
        bindings={str(marker):m.sha(marker)},inherited={'fixture':'older costs'})
    prep.write_bytes(m.raw(p));monkeypatch.setattr(m,'PREPARE_SHA',m.sha(prep))
    e=dict(sequence=0,previous_sha256=None,kind='run_finished',fixture=True);e['event_sha256']=m.digest(e)
    journal=d/'journal.jsonl';journal.write_bytes(m.raw(e))
    r.update(schema=m.early.NAME+'.report',status='all164_early_history_factorial_recorded',protocol=m.early.protocol(),pairs=[],
        errors=[],source_errors=[],counts=dict(attempted=m.early.units(),returned=m.early.units(),validated=m.early.units()),
        model_before=p['record']['expected_model'],model_after_predictions=p['record']['expected_model'],model_after=p['record']['expected_model'],
        strict_loads=p['record']['expected_strict_loads'],P1_admission=False,P5_admission=False,
        execution_binding=dict(prepared=str(prep),prepare_sha256=m.sha(prep)),inherited=p['inherited'],
        artifacts={'journal.jsonl':dict(sha256=m.sha(journal),file_bytes=journal.stat().st_size)},journal_chain_tail=e['event_sha256'])
    rp=d/'report.json';rp.write_bytes(m.raw(r));monkeypatch.setattr(m,'REPORT_SHA',m.sha(rp))
    cp=case.root/'results'/(m.early.NAME+'_consume_MAIN01.json')
    c=dict(schema=m.early.NAME+'.consume',status='all164_early_history_four_NEW_cells_and80_pairs_verified',report_sha256=m.sha(rp),
        branches=1,pairs=0,predictor=m.early.units(),RR_all_native_IO_bitwise_verified=True,prior_fullstrict_rebuilt=False,P1_admission=False,P5_admission=False)
    cp.write_bytes(m.raw(c));monkeypatch.setattr(m,'RECEIPT_SHA',m.sha(cp))
    ap=case.root/'results/agent_p5_factorial_independent_scalars_20261029_v1_MAIN01.json'
    a=dict(schema='agent_p5_factorial_independent_scalars_20261029_v1.analysis',status='all164_fourcell_and80_pair_independent_scalars_verified',branches=1,pair_count=0,
        arguments=dict(report=str(rp),report_sha=m.sha(rp),receipt=str(cp),receipt_sha=m.sha(cp)),
        verification_boundary=dict(scalar_arithmetic_independently_recomputed=True,prior_fullstrict_rebuilt=False),P1_admission=False,P5_admission=False,
        bindings={str(marker):m.sha(marker)})
    ap.write_bytes(m.raw(a));monkeypatch.setattr(m,'SCALAR_SHA',m.sha(ap))
    if fault=='receipt':cp.write_bytes(m.raw(dict(c,status='blocked_partial_factorial')))
    elif fault=='prepare':prep.write_bytes(m.raw(dict(p,status='wrong')))
    elif fault=='scalar':ap.write_bytes(m.raw(dict(a,status='wrong')))
    elif fault=='binding':marker.write_text('source drift')
    elif fault=='journal':journal.write_bytes(journal.read_bytes()+b'\n')
    elif fault=='inventory':(d/'unreported').write_text('extra')
    if fault:
        with pytest.raises(ValueError):REAL_PRIOR({})
    else:
        used={};dd,rr,pp=REAL_PRIOR(used)
        assert dd==d and rr['status']=='all164_early_history_factorial_recorded' and pp==p
        assert used[str(cp)]==m.RECEIPT_SHA and used[str(ap)]==m.SCALAR_SHA


def test_no_nested_old_fullstrict_and_import_has_no_torch():
    tree=ast.parse(Path(m.__file__).read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):assert n.func.attr not in ('consume','analyze','verify_analysis','build','verify')
    r=subprocess.run([sys.executable,'-m',m.__name__,'--help'],capture_output=True,text=True)
    assert r.returncode==0 and all(x in r.stdout for x in ('prepare','verify','run','consume'))
    r=subprocess.run([sys.executable,'-c',f'import sys; import {m.__name__}; assert "torch" not in sys.modules'],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
