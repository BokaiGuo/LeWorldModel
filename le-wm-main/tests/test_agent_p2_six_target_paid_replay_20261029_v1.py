"""Model-free RAM/Store fixtures only. No real prepare or upstream full consume."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_six_target_paid_replay_20261029_v1 as m


class Ops:
    mul=staticmethod(lambda a,b:np.multiply(a,b,dtype=np.float32))
    add=staticmethod(lambda a,b:np.add(a,b,dtype=np.float32))
    sub=staticmethod(lambda a,b:np.subtract(a,b,dtype=np.float32))
    div=staticmethod(lambda a,b:np.divide(a,b,dtype=np.float32))
    def capture(self,x):return x.copy()
    def layout(self,x):return dict(shape=list(x.shape),stride=[s//x.itemsize for s in x.strides],dtype='numpy.float32',device='ram_fixture',requires_grad=False)


def kernels(fail=None):
    def check(x):m.need(type(x) is np.ndarray and x.dtype==np.float32 and np.isfinite(x).all(),'RAM f32 only')
    def ae(x):
        if fail=='action_encoder':raise RuntimeError('RAM injected AE failure')
        return np.repeat(x.sum(axis=-1,keepdims=True,dtype=np.float32),192,axis=-1)
    def pred(z,a):
        if fail=='low_predictor':raise TimeoutError('RAM injected mid predict timeout')
        return z+a/np.float32(16)
    def project(x):
        if fail=='low_pred_proj':raise RuntimeError('RAM injected projection failure')
        return x.copy()
    return m.low.Kernels(lambda xs,dim:np.concatenate(xs,axis=dim),lambda xs,dim:np.stack(xs,axis=dim),check,ae,pred,project,'ram_fixture')


def fixture_domain():
    mu=np.asarray([-.0087,.0068],np.float32);std=np.asarray([.2019,.2002],np.float32)
    cm=dict(schema=m.six.TAG+'.prepare',protocol=dict(spec={'RAM_fixture':True}),slots=[])
    rows=[]
    for i in range(6):
        root=i//3
        values=dict(z=np.full((1,3,192),root/16,np.float32),a=np.full((1,2,10),root/32,np.float32),
                    target=np.full((1,192),(i+1)/32,np.float32))
        identity=dict(task_id=f'RAM-root{root}',task_sha256=str(root)*64,blind_sha256=str(root+2)*64)
        shas={k:m.core.digest(v.tobytes()) for k,v in values.items()}
        cm['slots'].append(dict(slot=i,root=root,condition='off',arm=m.six.ARMS[i%3],**identity,
            controller_payload={k:v.tolist() for k,v in values.items()},input_sha256=shas))
        norm=(np.linspace(-1,1,30,dtype=np.float32)+np.float32(i/32)).reshape(1,3,10)
        checks=m.raw.check_mean(norm,mu,std)
        rows.append(dict(slot=i,root=root,condition='off',arm=m.six.ARMS[i%3],task_identity=identity,
                         input_sha256=shas,checks=checks,mean_source=dict(data_sha256=m.core.digest(norm.tobytes()))))
    return dict(schema=m.TAG+'.domain',controller_prepare=cm,rows=rows,blocked_slot_ids=[],
                denominator=dict(total=6,checked=6,missing=0),normalizer=dict(mean_bits=mu.tobytes().hex(),std_bits=std.tobytes().hex()))


def test_fixed_protocol_no_spliced_four():
    p=m.protocol()
    assert p['active']==list(range(6)) and p['missing']==[] and p['denominator']==6
    assert p['planned']==dict(predict_API=36,predictor_samples=36,action_encoder=36,low_projection=36,
                             AE_tokens=144,predictor_context_tokens=108,projection_rows=108)
    assert [p[k] for k in ('output_cap','reserve','per_root_cap','floor')]==[x*1024**2 for x in (768,96,384,200)]
    assert (p['whole_wall'],p['per_slot_wall'])==(1350,180)
    assert all(p[k] is False for k in ('search','reselection','clip','tanh','raw_deployment_authorized','scientific_admission','original_Hi_scaler_certified'))
    assert p['FLOPs'] is None and p['physics']==p['encoder']==p['AD_primal']==0


def test_all_six_payloads_full_inverse_equation():
    d=fixture_domain();v=m.payloads(d)
    assert list(v)==list(range(6))
    for i,x in v.items():
        rr=np.add(np.multiply(x['mean'].reshape(15,2),x['std'],dtype=np.float32),x['mean_xy'],dtype=np.float32)
        assert rr.tobytes()==x['sealed_raw'].tobytes()
        assert x['a'].shape==(1,2,10) and x['z'].shape==(1,3,192)
        assert x['mean'].shape==(1,3,10)
    assert v[2]['mean'].tobytes()!=v[0]['mean'].tobytes()


@pytest.mark.parametrize('fault',['missing','slot','task','payload_sha','mean_sha','raw','normalizer','blocked','domainflag','zero_sign'])
def test_domain_authentication_mutations(fault):
    d=fixture_domain()
    if fault=='missing':d['rows'].pop(2)
    elif fault=='slot':d['rows'][2]['slot']=1
    elif fault=='task':d['rows'][2]['task_identity']['task_id']='forged'
    elif fault=='payload_sha':d['rows'][2]['input_sha256']={'z':'bad'}
    elif fault=='mean_sha':d['rows'][2]['mean_source']['data_sha256']='bad'
    elif fault=='raw':d['rows'][2]['checks']['array_bits']['raw']='00'*120
    elif fault=='normalizer':d['normalizer']['mean_bits']=np.zeros(2,np.float32).tobytes().hex()
    elif fault=='blocked':d['blocked_slot_ids']=[2]
    elif fault=='domainflag':d['rows'][2]['checks']['domain_ok']=False
    else:d['rows'][2]['checks']['array_bits']['normalized']='00000080'+d['rows'][2]['checks']['array_bits']['normalized'][8:]
    with pytest.raises((ValueError,KeyError)):m.payloads(d)


def test_domain_rows_actual_schema_mean_mapping():
    d=fixture_domain();cm=d['controller_prepare'];values=m.payloads(d);r=dict(rows=[]);arrays={};traces={}
    def desc(i,key):return dict(file=f'slot{i}_{key}.npy',root=i//3)
    for i in range(6):
        ids={k:desc(i,'input_'+k) for k in ('z','a','target')}
        for k in ids:arrays[ids[k]['file']]=values[i][k]
        ad={k:desc(i,k) for k in ('array_0','array_1','array_2')}
        for k in ad:arrays[ad[k]['file']]=values[i]['mean'].copy()
        td=dict(file=f'slot{i}_trace.json',root=i//3)
        traces[td['file']]=dict(completed=True,errors=[],backend='host_cuda_installed_CEM',host_tensor_seam_checked=True,
            spec=cm['protocol']['spec'],selected=dict(mean='array_0'),native_return=dict(actions='array_1',mean='array_2'))
        r['rows'].append(dict(slot=i,root=i//3,condition='off',arm=m.six.ARMS[i%3],status='returned',input_files=ids,arrays=ad,trace=td))
    get=lambda i,desc:arrays[desc['file']]
    trace=lambda i,desc:traces[desc['file']]
    mu=values[0]['mean_xy'];std=values[0]['std']
    out=m.domain_rows(cm,r,mu,std,get,trace);assert len(out)==6
    assert all(out[i]['checks']==d['rows'][i]['checks'] for i in range(6))
    arrays['slot2_array_1.npy'][0,0,0]=np.float32(99)
    with pytest.raises(ValueError,match='native returned'):m.domain_rows(cm,r,mu,std,get,trace)


@pytest.mark.parametrize('i',range(6))
def test_independent_paid_ram_replay_including_arbitrary(i):
    v=m.payloads(fixture_domain())[i];tr,a=m.paid.execute(v,kernels(),Ops(),'ram_fixture')
    c=m.paid.validate_trace(tr,a,v,host=False)
    assert c['predict_API']==c['predictor_samples']==6 and c['AE_tokens']==24 and c['predictor_context_tokens']==18
    assert tr['events'] and len(tr['replays'])==2
    assert m.scores(tr,a,v['target'])[0]==m.scores(tr,a,v['target'])[1]
    with pytest.raises(ValueError,match='backend'):m.paid.validate_trace(tr,a,v)


@pytest.mark.parametrize('kind',['action_encoder','low_predictor','low_pred_proj'])
def test_mid_call_failure_keeps_debits_and_arrays(kind):
    v=m.payloads(fixture_domain())[2];tr,arrays=m.paid.execute(v,kernels(kind),Ops(),'ram_fixture')
    assert tr['completed'] is False and tr['errors'] and arrays
    assert tr['events'][-1]['dispatched'] and not tr['events'][-1]['module_returned']
    assert tr['predict_API_debited']==(0 if kind=='action_encoder' else 1)
    with pytest.raises(ValueError):m.paid.validate_trace(tr,arrays,v,host=False)


def test_whole_domain_block_before_prepare_or_model(monkeypatch,tmp_path):
    d=fixture_domain();d['blocked_slot_ids']=[5]
    monkeypatch.setattr(m,'verify_domain',lambda *a:d)
    monkeypatch.setattr(m,'save_metadata',lambda *a:pytest.fail('blocked batch must not produce prepare'))
    with pytest.raises(ValueError,match='whole batch BLOCKED'):m.prepare('RAM','0'*64,tmp_path/'never.json')


def test_extreme_mean_no_clip_no_fallback():
    mu=np.zeros(2,np.float32);std=np.ones(2,np.float32)
    ch=m.raw.check_mean(np.full((1,3,10),2,np.float32),mu,std)
    assert ch['blocked_native_domain'] and len(ch['out_of_domain'])==30
    assert np.frombuffer(bytes.fromhex(ch['array_bits']['raw']),'<f4').tolist()==[2]*30


def make_saved(folder):
    """Real Store→disk→reader complete six lifecycle; explicitly synthetic, never evidence."""
    d=fixture_domain();values=m.payloads(d)
    construction=dict(own_sha256={},tensor_copy_attempts=425,tensor_copy_returns=425,state_key_count=425,forward_calls=0,physics=0)
    manifest=dict(expected_state={f'RAMparam{i}':str(i) for i in range(425)},expected_construction=construction,sources={})
    store=m.Store(folder)
    r=dict(schema=m.TAG+'.report',protocol=m.protocol(),completed=True,errors=[],rows=m.rows_for(),files=store.files,write_journal=store.journal,
        construction=construction,model_before=manifest['expected_state'],model_after=manifest['expected_state'],sources_after={},
        raw_deployment_authorized=False,scientific_admission=False)
    store.put('dispatch_manifest.json',m.encode(manifest))
    for i in range(6):r['rows'][i]['inputs']={k:store.array(f'slot{i}_array_{9000+j}.npy',v,i//3) for j,(k,v) in enumerate(values[i].items())}
    r['input_seal']=store.put('input_seal.json',m.encode([dict(slot=x['slot'],inputs=x['inputs']) for x in r['rows']]))
    r['construction_file']=store.put('construction.json',m.encode(construction));pending={}
    for i in range(6):
        row=r['rows'][i];tr,arrays=m.paid.execute(values[i],kernels(),Ops(),'ram_fixture')
        m.paid.persist_trace(store,row,tr,arrays)
        row.update(status='predictions_sealed',local_check=m.paid.validate_trace(tr,arrays,values[i],host=False),
            debits=dict(known=True,predict_API=6,predictor_samples=6),model_before=manifest['expected_state'],model_after=manifest['expected_state'],wall_seconds=.1)
        pending[i]=(tr,arrays)
    r['prediction_seal']=m.seal_predictions(store,r['rows'])
    for i,row in enumerate(r['rows']):
        row['target_MSE_fsum']=m.scores(*pending[i],values[i]['target'])
        row['score_file']=store.put(f'slot{i}_scores.json',m.encode(row['target_MSE_fsum']),i//3);row['status']='returned'
    r['debits']=m.six.old.totals(r['rows'])
    r['cuda_resources']=dict(dispatch_guard_calls=1,max_allocated=1,max_reserved=1,allocated_cap=3*1024**3,reserved_cap=4*1024**3)
    result=m.finalize(store,r)
    saved=m.read(result['report'],result['sha256']);m.validate_files(saved,folder)
    return manifest,d,saved,folder


@pytest.fixture(scope='module')
def saved(tmp_path_factory):
    # Patch only own output policy in this RAM fixture; no production module globals modified.
    base=tmp_path_factory.mktemp('six_paid_RAM');old=m.own_output
    m.own_output=lambda x:Path(x).absolute()
    try:return make_saved(base/'fixture')
    finally:m.own_output=old


def test_complete_real_store_reader_saved_consumer(saved):
    pm,d,r,folder=saved;c=m.validate_saved(pm,d,r,folder,host=False)
    assert c['evidence_backend']=='RAM_fixture_not_evidence'
    assert c['denominator']==dict(total=6,returned=6,blocked=0,missing=0)
    assert c['predict_API']==36 and c['module_resources']['AE_tokens']==144
    assert c['module_resources']['predictor_context_tokens']==108 and c['module_resources']['projection_rows']==108
    assert len(r['write_journal'])>100 and all(x['returned'] for x in r['write_journal'])
    with pytest.raises(ValueError):m.validate_saved(pm,d,r,folder)


@pytest.mark.parametrize('fault',['missing2','slot','task','status','model','source','input','crossslot','debit','score','barrier','extra_file','wall','GPU','partial','journal'])
def test_complete_consumer_tampering(saved,fault):
    pm,d,r,folder=saved;pm,d,r=deepcopy((pm,d,r))
    if fault=='missing2':r['rows'].pop(2)
    elif fault=='slot':r['rows'][5]['slot']=2
    elif fault=='task':d['rows'][5]['task_identity']['task_id']='forged'
    elif fault=='status':r['rows'][2]['status']='blocked'
    elif fault=='model':r['rows'][2]['model_after']={}
    elif fault=='source':r['sources_after']={'fake':'hash'}
    elif fault=='input':r['rows'][2]['inputs']['a']=r['rows'][2]['inputs']['z']
    elif fault=='crossslot':r['rows'][2]['trace']=r['rows'][0]['trace']
    elif fault=='debit':r['rows'][2]['debits']['predict_API']=0
    elif fault=='score':r['rows'][2]['target_MSE_fsum'][0]+=.01
    elif fault=='barrier':
        j=next(i for i,x in enumerate(r['write_journal']) if x['name']=='prediction_seal.json');r['write_journal'].insert(0,r['write_journal'].pop(j))
    elif fault=='extra_file':r['files']['extra.json']={}
    elif fault=='wall':r['rows'][2]['wall_seconds']=181
    elif fault=='GPU':r['cuda_resources']['max_allocated']=4*1024**3
    elif fault=='partial':r['completed']=False;r['errors']=[dict(type='RAM_timeout')]
    else:r['write_journal'].append(r['write_journal'][0])
    with pytest.raises((ValueError,KeyError)):m.validate_saved(pm,d,r,folder,host=False)


def test_final_fsync_boundary_and_report_binding(saved):
    _,_,r,folder=saved;c=m.read(folder/'completion.json',m.sha(folder/'completion.json'))
    assert c['boundary']==m.BOUNDARY and c['complete'] is True
    assert r['whole_wall_seconds_before_report']<=c['whole_wall_through_report_fsync_hash']<=m.WALL
    assert c['report_sha256']==m.sha(folder/'report.json')
    m.check_completion(r,c,m.sha(folder/'report.json'))


@pytest.mark.parametrize('fault',['schema','report','complete','boundary','elapsed','before_report'])
def test_completion_tampering(saved,fault):
    _,_,r,folder=saved;r=deepcopy(r);c=m.read(folder/'completion.json',m.sha(folder/'completion.json'))
    if fault=='schema':c['schema']=m.paid.TAG+'.completion'
    elif fault=='report':c['report_sha256']='0'*64
    elif fault=='complete':c['complete']=False
    elif fault=='boundary':c['boundary']='kernel only'
    elif fault=='elapsed':c['whole_wall_through_report_fsync_hash']=m.WALL+1
    else:r['whole_wall_seconds_before_report']=c['whole_wall_through_report_fsync_hash']+1
    with pytest.raises(ValueError):m.check_completion(r,c,m.sha(folder/'report.json'))


def test_store_failure_preserves_failed_write_and_reserve(monkeypatch,tmp_path):
    monkeypatch.setattr(m,'own_output',lambda p:Path(p))
    st=m.Store(tmp_path/'reserve');monkeypatch.setattr(m,'CAP',128);monkeypatch.setattr(m,'RESERVE',64)
    st.put('first',b'x'*60)
    with pytest.raises(ValueError,match='reserve'):st.put('failed',b'x'*5)
    assert st.journal[-1]['returned'] is False and 'failed' not in st.files and not (st.path/'failed').exists()
    st.put('report.json',b'partial',final=True)
    assert (st.path/'report.json').read_bytes()==b'partial'


def test_actual_saved_authority_shape_only():
    # No upstream(), domain(), build_domain(), verify(), consume() or actual objective math.
    rr=m.refs();docs={k:m.read(v['path'],v['sha256']) for k,v in rr.items()}
    c,r,pm=(docs[k] for k in ('receipt','report','prepare'))
    assert c['schema']==m.six.TAG+'.receipt' and c['manifest_sha256']==rr['prepare']['sha256'] and c['report_sha256']==rr['report']['sha256']
    assert c['denominator']==dict(total=6,returned=6,missing=0)
    assert [x['slot'] for x in c['local_checks']]==list(range(6))
    assert len(pm['slots'])==len(r['rows'])==6
    assert all({'trace','arrays','input_files','slot','root','arm','condition','status'}<=set(x) for x in r['rows'])


def test_actual_mean_NPY_and_task_mapping_ABI_only():
    # Read frozen bytes/shape/identity ONLY. No domain/inverse, objective or fullstrict.
    rr=m.refs();cm=m.read(rr['prepare']['path'],rr['prepare']['sha256']);report=m.read(rr['report']['path'],rr['report']['sha256'])
    folder=Path(rr['report']['path']).parent
    for i,row in enumerate(report['rows']):
        td=row['trace'];tr=m.read(folder/td['file'],td['file_sha256'])
        mean=row['arrays'][tr['selected']['mean']];a=m.six.load_array(folder,mean)
        assert a.dtype==np.float32 and a.shape==(1,3,10)
        assert all(a.tobytes()==m.six.load_array(folder,row['arrays'][tr['native_return'][k]]).tobytes() for k in ('actions','mean'))
        assert mean['file'].startswith(f'slot{i}_') and mean['root']==i//3
        assert tr['backend']=='host_cuda_installed_CEM' and tr['host_tensor_seam_checked'] is True
    dr=m.six.refs()['delivery'];delivery=m.read(dr['path'],dr['sha256'])
    pr=delivery['authorities']['program_receipt'];program=m.read(pr['path'],pr['sha256'])
    pp,ph=m.raw.PINS['physical_prepare'];physical=m.read(m.ROOT/pp,ph)
    assert len(program['plans'])==4
    for root in (0,1):
        plan=program['plans'][2*root+1]
        assert physical['plans'][2*root+1]['planner_plan']==plan
        assert np.asarray(plan['root_warm_raw10']).shape==(10,2)
        for i in range(3*root,3*root+3):
            assert all(cm['slots'][i][k]==delivery['rows'][3*root][k]==plan[k] for k in ('task_id','task_sha256','blind_sha256'))


def test_import_and_protocol_no_neural_or_environment():
    code=f'import sys; from experiments.spectral_reachability import {m.TAG} as m; m.protocol(); assert not any(x in sys.modules for x in ("torch","gymnasium","pymunk","stable_worldmodel"))'
    result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_run_requires_authority_before_io(tmp_path):
    with pytest.raises(ValueError,match='MAIN authorization'):m.run_cuda('bad','bad',tmp_path/'none')
    assert not (tmp_path/'none').exists()
