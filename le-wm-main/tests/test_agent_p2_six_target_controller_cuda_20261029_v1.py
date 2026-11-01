"""RAM callbacks/metadata ABI; no weights/model/production full strict calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_six_target_controller_cuda_20261029_v1 as m


def data(i=0):
    return dict(z=np.arange(576,dtype=np.float32).reshape(1,3,192)/np.float32(1024)+np.float32(i//3),
        a=np.arange(20,dtype=np.float32).reshape(1,2,10)/np.float32(1024),
        target=np.full((1,192),.125+i*.01,np.float32))


def kernels(fail=None):
    def check(x):m.need(type(x) is np.ndarray and x.dtype==np.float32 and np.isfinite(x).all(),'RAM f32')
    def ae(a):
        if fail=='action_encoder':raise RuntimeError('fixture AE failure')
        return np.repeat(a.sum(-1,keepdims=True,dtype=np.float32),192,axis=-1)
    def predict(z,a):
        if fail=='low_predictor':raise RuntimeError('fixture predictor failure')
        return z+a/np.float32(16)
    return m.core.low.Kernels(lambda xs,dim:np.concatenate(xs,axis=dim),lambda xs,dim:np.stack(xs,axis=dim),check,ae,predict,lambda x:x.copy(),'ram_fixture')


def trace(i=0,fail=None):
    v=data(i)
    return m.core.track_fixture(v['z'],v['a'],v['target'],spec=m.tracker.spec(),kernels=kernels(fail))


def fixture_manifest():
    slots=[]
    for i in range(6):
        v=data(i);slots.append(dict(slot=i,root=i//3,arm=m.ARMS[i%3],condition='off',
            controller_payload={k:a.tolist() for k,a in v.items()},input_sha256={k:m.core.digest(a.tobytes()) for k,a in v.items()}))
    return dict(schema=m.TAG+'.prepare',slots=slots)


def assembly_inputs():
    rows=[];oldinputs={};rrows=[]
    for i in range(6):
        v=data(i);row=dict(root=i//3,condition='off',arm=m.ARMS[i%3],task_id='task'+str(i//3),task_sha256='t'*64,
            blind_sha256='b'*64,history_sha256=m.core.digest(v['z'].tobytes()),action_history_sha256=m.core.digest(v['a'].tobytes()),
            status='missing' if i%3==2 else 'assembled_not_dispatched',controller_payload=None if i%3==2 else {k:a.tolist() for k,a in v.items()},
            target_sha256=m.core.digest(v['target'].tobytes()))
        rows.append(row)
        rrows.append(dict(slot=i,root=i//3,arm=m.ARMS[i%3],status='missing' if i%3==2 else 'returned',input_files={k:dict(file=f'slot{i}_input_{k}.npy') for k in v}))
        oldinputs[i]=v
    d=dict(schema=m.delivery.TAG+'.prepare',rows=rows,denominator=dict(roots=2,arms=3,total=6,assembled=4,missing=2,dispatched=0))
    parent=m.refs()['delivery'];plans=[];slots=[]
    for root in (0,1):
        i=3*root+2;v=data(i)
        plans.append(dict(root=root,condition='off',status='delivered',failure=None,source_time=None,
            task_id=rows[i]['task_id'],controller_payload={k:a.tolist() for k,a in v.items()},
            target32_sha256=m.core.digest(v['target'].tobytes()),history_sha256=rows[i]['history_sha256'],action_history_sha256=rows[i]['action_history_sha256']))
        slots.append(dict(root=root,condition='off',task_id=rows[i]['task_id'],task_sha256=rows[i]['task_sha256'],blind_sha256=rows[i]['blind_sha256'],z=v['z'].tolist(),a=v['a'].tolist()))
    ac=dict(schema=m.arbitrary.NAME+'.receipt',complete=True,denominator=2,status='all2_arbitrary_FIRST_numerically_verified_NOT_tracked',plans=plans,parent_delivery=parent,
        prepare_sha256=m.refs()['arbitrary_prepare']['sha256'],report_sha256=m.refs()['arbitrary_report']['sha256'],completion_sha256=m.refs()['arbitrary_completion']['sha256'])
    return d,dict(delivery_artifact=d),dict(rows=rrows),ac,dict(slots=slots,parent_delivery=parent),lambda i,k:oldinputs[i][k]


def test_fixed_protocol_all27_budget():
    p=m.protocol();s=m.tracker.spec().planned()
    assert len(set(m.AXES))==len(m.AXES)==27 and p['active_slots']==list(range(6))
    assert (p['planned_predict_API'],p['planned_predictor_samples'])==(72,324)
    assert p['spec']['segments']==[1,2] and p['spec']['seed']==0
    assert (s['action_encoder_tokens']*6,s['low_predictor_context_tokens']*6,s['low_projection_rows']*6)==(1296,972,972)
    assert not any(p[k] for k in ('raw_inverse','raw_deployment_authorized','scientific_admission','fair_baseline','equal_generation_budget'))
    assert json.loads(m.encode(p))==p and p['prior_physical_time'] is None


def test_actual_saved_input_abi_only():
    rr=m.refs();docs={k:m.read(v['path'],v['sha256']) for k,v in rr.items()}
    def get(i,k):
        desc=docs['old_report']['rows'][i]['input_files'][k]
        return m.old.load_array(Path(rr['old_report']['path']).parent,desc)
    slots=m.assemble(docs['delivery'],docs['old_prepare'],docs['old_report'],docs['arbitrary_receipt'],docs['arbitrary_prepare'],get)
    out=m.select_payloads(dict(schema=m.TAG+'.prepare',slots=slots))
    assert tuple(out)==tuple(range(6)) and all(set(v)=={'z','a','target'} for v in out.values())
    for i in m.old.ACTIVE:assert slots[i]['source']['row']==docs['delivery']['rows'][i]
    assert [slots[i]['source']['plan']['target32_sha256'] for i in (2,5)]==[x['target32_sha256'] for x in docs['arbitrary_receipt']['plans']]


def test_assembly_fixed_six_without_mutating_parent():
    args=assembly_inputs();before=deepcopy(args[0]);slots=m.assemble(*args)
    assert args[0]==before and len(slots)==6 and slots[2]['source']['kind']=='arbitrary_receipt_plan'
    for i in m.old.ACTIVE:assert slots[i]['source']['row']==before['rows'][i]


@pytest.mark.parametrize('fault',['old_byte','signedzero','witness','target','arbitrary_context','order','complete','time','receipt','task','missing'])
def test_assembly_mutations(fault):
    d,cm,r,ac,ap,get=assembly_inputs()
    if fault=='old_byte':
        def bad(i,k):
            a=get(i,k).copy()
            if i==0 and k=='target':a[0,0]=np.nextafter(a[0,0],np.float32(1))
            return a
        get0=bad
    elif fault=='signedzero':
        def bad(i,k):
            a=get(i,k).copy()
            if i==0 and k=='a':a[0,0,0]=-0.
            return a
        get0=bad
    else:get0=get
    if fault=='witness':ac['plans'][0]['controller_payload']['witness']=[]
    elif fault=='target':ac['plans'][0]['target32_sha256']='0'*64
    elif fault=='arbitrary_context':ac['plans'][0]['controller_payload']['z'][0][0][0]+=1
    elif fault=='order':ac['plans'].reverse()
    elif fault=='complete':ac['complete']=False
    elif fault=='time':ac['plans'][0]['source_time']=5
    elif fault=='receipt':ac['completion_sha256']='0'*64
    elif fault=='task':ap['slots'][0]['task_sha256']='0'*64
    elif fault=='missing':d['rows'][2]['status']='assembled_not_dispatched'
    with pytest.raises(ValueError):m.assemble(d,cm,r,ac,ap,get0)


@pytest.mark.parametrize('fault',['keys','order','sha','shape','context'])
def test_six_payload_gate(fault):
    a=fixture_manifest()
    if fault=='keys':a['slots'][2]['controller_payload']['raw']=[]
    elif fault=='order':a['slots'][2]['slot']=0
    elif fault=='sha':a['slots'][2]['input_sha256']['target']='0'*64
    elif fault=='shape':a['slots'][2]['controller_payload']['z']=[[]]
    else:
        a['slots'][2]['controller_payload']['a'][0][0][0]=.5
        a['slots'][2]['input_sha256']['a']=m.core.digest(np.asarray(a['slots'][2]['controller_payload']['a'],np.float32).tobytes())
    with pytest.raises(ValueError):m.select_payloads(a)


def test_counter_axes_derive_full_callback_trace():
    tr,a=trace();m.core.validate_saved(tr,a);c=m.accounting27(tr,a)
    assert c['predict_API']==12 and c['predictor_samples']==54
    assert (c['AE_rows'],c['AE_tokens'],c['predictor_context_tokens'],c['lowproj_rows'])==(54,216,162,162)
    assert (c['candidate_cost_calls'],c['candidate_trajectories'],c['mean_paid_replays'],c['replay_trajectories'])==(2,16,2,2)
    assert all(c[k]==12 for k in ('AE_attempts','AE_dispatched','AE_returned','AE_captured','predictor_attempts','lowproj_returned'))
    assert c['FLOPs'] is None


@pytest.mark.parametrize('fail,api',[('action_encoder',0),('low_predictor',1)])
def test_counter_partial_failed_work_kept(fail,api):
    tr,a=trace(fail=fail);c=m.accounting27(tr,a)
    assert not tr['completed'] and tr['errors'] and c['predict_API']==api
    assert c['AE_attempts']==c['AE_dispatched']==1
    assert c['candidate_cost_calls']==1 and c['candidate_trajectories']==8
    if fail=='low_predictor':assert c['predictor_attempts']==c['predictor_dispatched']==1 and c['predictor_returned']==0


def test_aggregate_unknown_is_not_refunded_or_padded():
    rows=m.initial_rows();rows[0]['axes'].update(predict_API=12,predictor_samples=54)
    rows[1]['axes']={k:None for k in m.AXES}
    a=m.aggregate(rows)
    assert a['known_subtotals']['predict_API']==12 and a['actual']['predict_API'] is None
    assert a['unknown_slots']['predict_API']==[1] and a['actual']['FLOPs'] is None


@pytest.fixture
def store(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    return m.Store(tmp_path/'results'/(m.TAG+'_fixture'))


def test_all18_inputs_before_any_invoke_and_failure_unknown(store):
    rows=m.initial_rows();values=m.select_payloads(fixture_manifest())
    with pytest.raises(ValueError):m.execute_slots(store,rows,values,None,None,{})
    m.persist_inputs(store,rows,values);assert len(store.files)==19
    seen=[]
    def invoke(p):
        assert len(store.files)==19 and set(p)=={'z','a','target'}
        p['z'][:]=99;seen.append(True);raise RuntimeError('RAM unexpected interruption')
    with pytest.raises(RuntimeError):m.execute_slots(store,rows,values,invoke,lambda:{},{})
    assert len(seen)==1 and rows[0]['axes']['predict_API'] is None and rows[0]['status']=='attempted'
    assert rows[1]['status']=='not_started' and not np.all(values[0]['z']==99)


def test_fixture_never_promoted_and_trace_saved(store):
    rows=m.initial_rows();v=m.select_payloads(fixture_manifest());m.persist_inputs(store,rows,v)
    def invoke(p):return m.core.track_fixture(p['z'],p['a'],p['target'],spec=m.tracker.spec(),kernels=kernels())
    with pytest.raises(ValueError,match='HOST only'):m.execute_slots(store,rows,v,invoke,lambda:{},{})
    assert rows[0]['status']=='partial' and rows[0]['axes']['predict_API']==12
    d=rows[0]['trace'];tr=m.read(store.path/d['file'],d['file_sha256'])
    assert tr['backend']=='ram_fixture_not_production'
    assert rows[1]['status']=='not_started' and rows[2]['status']=='not_started'


@pytest.mark.parametrize('slot',[0,1,2,3,4,5])
@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.int64])
def test_all_six_array_readers(store,slot,dtype):
    a=np.zeros((1,3,10),dtype);d=store.array(f'slot{slot}_array_0.npy',a,slot//3)
    assert m.delivery.same(m.load_array(store.path,d),a)


@pytest.mark.parametrize('fault',['path','slot6','trailer','sha','nan','shape','symlink'])
def test_array_mutations(store,fault):
    d=store.array('slot2_array_0.npy',np.zeros(3,np.float32),0);p=store.path/d['file']
    if fault=='path':d['file']='../slot2_array_0.npy'
    elif fault=='slot6':d['file']='slot6_array_0.npy'
    elif fault=='sha':d['data_sha256']='0'*64
    elif fault=='shape':d['shape']=[2]
    elif fault=='symlink':q=store.path/'original';p.rename(q);p.symlink_to(q)
    else:
        b=p.read_bytes()+b'x' if fault=='trailer' else p.read_bytes()[:-4]+np.asarray(np.nan,np.float32).tobytes()
        p.write_bytes(b);d.update(file_bytes=len(b),file_sha256=hashlib.sha256(b).hexdigest())
    with pytest.raises((ValueError,OSError)):m.load_array(store.path,d)


def test_output_caps_root_wall_and_final_footer(store):
    store.root_used[0]=m.ROOT_CAP-m.ROOT_RESERVE
    with pytest.raises(ValueError):store.put('blocked',b'x',0)
    store.put('trace.json',b'{}',0,footer=True)
    store.used=m.CAP-m.RESERVE
    with pytest.raises(ValueError):store.guard(1)
    store.guard(1,final=True)
    store.used=0;store.start-=m.WALL+1
    with pytest.raises(ValueError):store.guard()
    store.guard(1,footer=True)


def saved_fixture(store,monkeypatch):
    # Source guard relaxed ONLY for this explicit RAM saved-orchestration fixture.
    monkeypatch.setattr(m,'require_host',lambda tr:None)
    monkeypatch.setattr(m,'check_authentication',lambda ref,expected:None)
    manifest=fixture_manifest();state={str(i):dict(shape=[1],dtype='torch.float32',sha256='f'*64) for i in range(425)}
    construction=dict(own_sha256={},tensor_copy_attempts=425,tensor_copy_returns=425,state_key_count=425,forward_calls=0,physics=0)
    manifest.update(expected_state=state,expected_construction=construction,sources={},full_receipts=dict(arbitrary={}))
    store.put('dispatch_manifest.json',m.encode(manifest));rows=m.initial_rows();v=m.select_payloads(manifest)
    seal=m.persist_inputs(store,rows,v);cf=store.put('construction.json',m.encode(construction))
    calls=[]
    def invoke(p):
        i=len(calls);calls.append(i)
        assert all(m.delivery.same(p[k],v[i][k]) for k in p)
        tr,a=trace(i);tr.update(model_before=state,model_after=state,sources_before={},sources_after={},host_entry_through_postguard_wall_seconds=tr['whole_core_wall_seconds']+.001)
        return tr,a
    m.execute_slots(store,rows,v,invoke,lambda:state,state)
    assert calls==list(range(6))
    # The fixed +.001 synthetic host-entry gap may exceed measured fixture wall.
    for row in rows:row['wall_seconds']=max(row['wall_seconds'],row['capture']['host_entry_wall_seconds']+.001)
    r=dict(schema=m.TAG+'.report',protocol=m.protocol(),completed=True,errors=[],rows=rows,input_seal=seal,construction=construction,construction_file=cf,
        manifest='/fixture/manifest.json',manifest_sha256='a'*64,
        model_before=state,model_after=state,sources_after={},runtime_authentication={},files=store.files,write_journal=store.journal,
        accounting27=m.aggregate(rows),side_axes=dict(encoder=0,high=0,physics=0,SVD=0,model_state_copies=425,original_Hi_scaler_certified=False,prior_physical_time=None,independent_gradient_recompute=False),
        cuda_resources=dict(dispatch_guard_calls=1,max_allocated=1,max_reserved=2,allocated_cap=m.ALLOCATED,reserved_cap=m.RESERVED),
        whole_wall_seconds_before_report=5.,file_bytes_before_report=store.used,root_file_bytes=store.root_used,raw_inverse=False,scientific_admission=False)
    r.update(m.old.totals(rows));store.put('report.json',m.encode(r),final=True)
    store.put('completion.json',m.encode(dict(fixture=True)),final=True)
    return manifest,m.read(store.path/'report.json',m.sha(store.path/'report.json'))


def test_full_six_saved_fixture_consumer_27axes(store,monkeypatch):
    manifest,r=saved_fixture(store,monkeypatch);c=m.analyze_saved(manifest,r,store.path)
    assert c['denominator']==dict(total=6,returned=6,missing=0) and c['predict_API']==72 and c['predictor_samples']==324
    a=c['accounting27']['actual'];assert a['AE_tokens']==1296 and a['lowproj_rows']==a['predictor_context_tokens']==972
    assert a['candidate_trajectories']==96 and a['mean_paid_replays']==12 and a['replay_trajectories']==12
    assert not c['scientific_admission'] and c['FLOPs'] is None


@pytest.mark.parametrize('fault',[None,'completion_false','completion_overwall','completion_sha'])
def test_full_consume_fixture_inclusive_completion_and_exclusive_receipt(store,monkeypatch,fault):
    manifest,r=saved_fixture(store,monkeypatch)
    manifest['authorities']=dict(arbitrary_receipt=dict(path='fixture_arbitrary_receipt',sha256='b'*64))
    # Manifest is changed only inside this RAM fixture: re-seal its file and report inventory.
    p=store.path/'dispatch_manifest.json';body=m.encode(manifest);p.write_bytes(body)
    d=r['files'][p.name];oldsize=d['file_bytes'];d.update(file_bytes=len(body),file_sha256=hashlib.sha256(body).hexdigest())
    r['file_bytes_before_report']+=len(body)-oldsize
    for e in r['write_journal']:
        if e['name']==p.name:e['bytes']=len(body)
    rp=store.path/'report.json';rp.write_bytes(m.encode(r));rh=m.sha(rp)
    proof=dict(path='explicit_RAM_authentication',sha256='c'*64)
    monkeypatch.setattr(m,'verify',lambda path,h:(manifest,proof))
    done=dict(schema=m.TAG+'.completion',report_sha256=rh,complete=True,whole_wall_through_report_fsync_hash=6.,
        boundary='Store start before verify through report fsync/hash; completion acknowledgement excluded')
    if fault=='completion_false':done['complete']=False
    elif fault=='completion_overwall':done['whole_wall_through_report_fsync_hash']=1801
    elif fault=='completion_sha':done['report_sha256']='0'*64
    (store.path/'completion.json').write_bytes(m.encode(done))
    out=store.path.parent/(m.TAG+'_consume_fixture.json')
    if fault:
        with pytest.raises(ValueError):m.consume(rp,rh,out)
        assert not out.exists()
    else:
        result=m.consume(rp,rh,out);receipt=m.read(out,result['sha256'])
        assert receipt['predict_API']==72 and receipt['predictor_samples']==324 and receipt['saved_consume_authentication']==proof
        with pytest.raises(FileExistsError):m.consume(rp,rh,out)


@pytest.mark.parametrize('fault',['context','old_result','cross_slot','axis','capture','partial','model','extra_file','journal','whole','gpu','missing','spec','source','aggregate'])
def test_full_six_saved_mutations(store,monkeypatch,fault):
    manifest,r=saved_fixture(store,monkeypatch)
    if fault=='context':manifest['slots'][2]['controller_payload']['a'][0][0][0]+=1
    elif fault=='old_result':r['rows'][0]['trace']['file_sha256']='0'*64
    elif fault=='cross_slot':r['rows'][2]['arrays']=r['rows'][0]['arrays']
    elif fault=='axis':r['rows'][2]['axes']['AE_tokens']-=1
    elif fault=='capture':r['rows'][2]['capture']['saved_array_count']-=1
    elif fault=='partial':r['rows'][5]['status']='partial'
    elif fault=='model':r['rows'][2]['dispatch_model_after']={}
    elif fault=='extra_file':(store.path/'extra').write_bytes(b'x')
    elif fault=='journal':r['write_journal'].reverse()
    elif fault=='whole':r['whole_wall_seconds_before_report']=1801
    elif fault=='gpu':r['cuda_resources']['max_allocated']=m.ALLOCATED+1
    elif fault=='missing':r['rows'].pop()
    elif fault=='spec':r['protocol']['spec']['iterations']=1
    elif fault=='source':r['sources_after']={'changed':'0'*64}
    else:r['accounting27']['actual']['predict_API']=71
    with pytest.raises((ValueError,KeyError,IndexError)):m.analyze_saved(manifest,r,store.path)


def test_authentication_reconstructs_exact_whole_receipt_retained(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);results=tmp_path/'results';results.mkdir()
    src=tmp_path/'source.py';src.write_bytes(b'fixture source');monkeypatch.setattr(m,'PINS',{src:m.sha(src)})
    rr={}
    expected=dict(verification_cpu_calls=dict(loss=130,analytic_gradient=128,update=128),entire_field=['retained','not_status_only'])
    for key in m.REFS:
        p=results/(key+'.json');m.tracker.durable(p,m.encode(expected if key=='arbitrary_receipt' else dict(fixture=key)))
        rr[key]=dict(path=str(p),sha256=m.sha(p))
    monkeypatch.setattr(m,'refs',lambda:deepcopy(rr));seen=[]
    def consume(prepared,ph,folder,rh,output):
        assert Path(folder)==Path(rr['arbitrary_report']['path']).parent
        m.tracker.durable(output,m.encode(expected));seen.append(output)
        return dict(path=str(Path(output).absolute()),sha256=m.sha(output))
    monkeypatch.setattr(m.arbitrary,'consume',consume)
    ref=m.authenticate_arbitrary();proof=m.check_authentication(ref,expected)
    assert len(seen)==1 and Path(seen[0]).is_file() and proof['verification_cpu_calls']==expected['verification_cpu_calls']
    # Non-status extra field mutation in newly returned receipt is rejected, failed proof retained.
    def bad(*args):
        output=args[-1];changed=deepcopy(expected);changed['entire_field']=['changed'];m.tracker.durable(output,m.encode(changed))
        return dict(path=str(Path(output).absolute()),sha256=m.sha(output))
    monkeypatch.setattr(m.arbitrary,'consume',bad)
    with pytest.raises(ValueError,match='whole persisted receipt'):m.authenticate_arbitrary()
    assert len(list(results.glob(m.TAG+'_auth_*')))==2


def test_prepare_verify_own_fixture_cli(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir();proof=dict(path='fixtureproof',sha256='f'*64)
    expected=dict(schema='explicit_RAM_prepare_fixture',upstream_authentication=proof,full_receipts=dict(arbitrary={}))
    monkeypatch.setattr(m,'authenticate_arbitrary',lambda:proof)
    monkeypatch.setattr(m,'build',lambda auth:deepcopy(expected))
    monkeypatch.setattr(m,'check_authentication',lambda auth,receipt:None)
    p=tmp_path/'results'/(m.TAG+'_prepare_fixture.json');out=m.prepare(p)
    actual,auth=m.verify(p,out['sha256']);assert actual==expected and auth==proof
    m.main(['verify','--manifest',str(p),'--manifest-sha256',out['sha256']]);assert json.loads(capsys.readouterr().out)['verified']
    with pytest.raises(ValueError):m.prepare(p)
    expected['changed']=True
    with pytest.raises(ValueError):m.verify(p,out['sha256'])


def test_file_level_builder_prepare_verify_fixture_no_production_consumers(tmp_path,monkeypatch):
    """Real tiny JSON/NPY files through new build/prepare/verify, owner heavy checks stubbed."""
    monkeypatch.setattr(m,'ROOT',tmp_path);results=tmp_path/'results';results.mkdir()
    oldfolder=results/'old_job';oldfolder.mkdir();oldstore=object.__new__(m.Store)
    oldstore.path=oldfolder;oldstore.start=m.time.monotonic();oldstore.used=0;oldstore.root_used={0:0,1:0};oldstore.files={};oldstore.journal=[]
    d,cm,report,ac,ap,get=assembly_inputs();rr={}
    def put(role,obj,where=None):
        p=where or results/(role+'.json');m.tracker.durable(p,m.encode(obj));rr[role]=dict(path=str(p),sha256=m.sha(p));return rr[role]
    put('delivery',d)
    src=results/'fixture_model_state.npy';desc=oldstore.array('slot0_array_9000.npy',np.zeros(1,np.float32),0)
    src=oldfolder/desc['file'];sources={str(src):m.sha(src)}
    state={str(i):dict(shape=[1],dtype='torch.float32',sha256='f'*64) for i in range(425)}
    cm.update(sources=sources,expected_state=state,expected_construction=dict(fixture='constructor'),export=dict(fixture='no_loader'))
    put('old_prepare',cm)
    for i in m.old.ACTIVE:
        report['rows'][i]['input_files']={k:oldstore.array(f'slot{i}_input_{k}.npy',get(i,k),i//3) for k in ('z','a','target')}
    report.update(manifest=rr['old_prepare']['path'],manifest_sha256=rr['old_prepare']['sha256'])
    put('old_report',report,oldfolder/'report.json')
    oldcheck=dict(schema='explicit_RAM_old_saved_check',whole_non_status_field=[1,2,3])
    oldreceipt=dict(**oldcheck,report_sha256=rr['old_report']['sha256'],manifest_sha256=rr['old_prepare']['sha256'])
    put('old_receipt',oldreceipt)
    ap.update(parent_delivery=rr['delivery'],bindings=sources,inherited_expected_state_sha256=m.arbitrary.digest(state))
    put('arbitrary_prepare',ap)
    ar=dict(rows=ac['plans'],prepare_sha256=rr['arbitrary_prepare']['sha256'],prepared=rr['arbitrary_prepare']['path'])
    put('arbitrary_report',ar)
    put('arbitrary_completion',dict(complete=True,report_sha256=rr['arbitrary_report']['sha256']))
    ac.update(parent_delivery=rr['delivery'],prepare_sha256=rr['arbitrary_prepare']['sha256'],report_sha256=rr['arbitrary_report']['sha256'],completion_sha256=rr['arbitrary_completion']['sha256'])
    put('arbitrary_receipt',ac)
    authdir=results/(m.TAG+'_auth_fixture');authdir.mkdir();receipt_ref=dict(path=str(authdir/'receipt.json'),sha256=rr['arbitrary_receipt']['sha256'])
    m.tracker.durable(receipt_ref['path'],m.encode(ac));authpath=authdir/'proof.json'
    m.tracker.durable(authpath,m.encode(dict(receipt=receipt_ref)));authref=dict(path=str(authpath),sha256=m.sha(authpath))
    monkeypatch.setattr(m,'refs',lambda:deepcopy(rr))
    monkeypatch.setattr(m.delivery,'build',lambda:deepcopy(d))
    monkeypatch.setattr(m.old,'build',lambda path,h:deepcopy(cm))
    monkeypatch.setattr(m.old,'analyze_saved',lambda *args:deepcopy(oldcheck))
    monkeypatch.setattr(m,'check_authentication',lambda ref,receipt:dict(receipt=receipt_ref))
    monkeypatch.setattr(m,'authenticate_arbitrary',lambda:deepcopy(authref))
    monkeypatch.setattr(m,'closure',lambda:{})
    output=results/(m.TAG+'_prepare_file_fixture.json');result=m.prepare(output);actual,proof=m.verify(output,result['sha256'])
    assert len(actual['slots'])==6 and actual['full_receipts']['old']==oldreceipt and actual['full_receipts']['arbitrary']==ac
    assert actual['expected_state']==state and proof==authref
    oldcheck['whole_non_status_field']=[9]
    with pytest.raises(ValueError,match='entire original strict receipt'):m.verify(output,result['sha256'])


def test_unauthorized_before_io_cold_import_and_no_torch(tmp_path):
    p=tmp_path/'no_output'
    with pytest.raises(ValueError,match='MAIN authorization'):m.run_cuda('missing','0'*64,p)
    assert not p.exists()
    code='import sys; from experiments.spectral_reachability import '+m.TAG+' as m; m.main(["protocol"]); assert "torch" not in sys.modules'
    result=subprocess.run([sys.executable,'-B','-c',code],text=True,capture_output=True,check=True)
    assert json.loads(result.stdout)['planned_predict_API']==72


def test_parent_source_pins_and_approved_protocol_unchanged():
    out=m.closure();assert all(out[str(p)]==h for p,h in m.PINS.items())
    assert out[str(m.PROTOCOL_PATH)]==m.PROTOCOL_SHA
