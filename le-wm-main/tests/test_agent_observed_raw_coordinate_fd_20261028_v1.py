"""All164 ambient raw FD: NumPy native ABI and saved-layout tests ONLY."""
from copy import deepcopy
from pathlib import Path
import ast
import time
import numpy as np
import pytest
from experiments.spectral_reachability import agent_observed_raw_coordinate_fd_20261028_v1 as m
import test_agent_nonlinear_suffix_observed_recompile_20261028_v2 as prior

inputs=prior.inputs
complete=prior.complete


@pytest.fixture(scope='module')
def envelopes(complete):
    specs=[];rows=[]
    for i in range(4):
        spec=prior.spec_for(complete,i);j=prior.n.parent.Events();l=m.core.Ledger(m.F.caps(18,8),sink=j.event)
        result=m.source.compile_row(spec,prior.RAM(spec['request']['task'],j,l),j)
        specs.append(spec);rows.append(dict(index=i,status='sealed',result=result))
    authority=dict(prepared='/RAM/prepared.json',prepare_sha256='a'*64,output='/RAM/output',report_sha256='b'*64,
        receipt='/RAM/receipt.json',receipt_sha256='c'*64)
    p=dict(schema=m.source.NAME+'.prepare',contract=m.source.contract(),production_ready=True,rows=specs)
    r=dict(schema=m.source.NAME+'.report',contract=m.source.contract(),status='observed_suffix4_recompiled_not_admitted',errors=[],
        prepared=authority['prepared'],prepare_sha256=authority['prepare_sha256'],known_roots=2,rows=rows)
    a=dict(schema=m.source.NAME+'.consume',status='observed_suffix4_verified_science_blocked',complete=True,
        report_sha256=authority['report_sha256'],fixed_denominator=4,known_roots=2,rows=deepcopy(rows),
        plans=[v['result']['plan'] for v in rows],actual_new_compute=dict(attempted=m.F.caps(72,32),completed=m.F.caps(72,32)),
        P1_admission=False,scientific_admission=False,freshness=False,physical_release=False)
    return p,r,a,authority


@pytest.fixture(scope='module')
def jobs(envelopes):return m.plan_jobs(*envelopes)


def test_pure_manifest_fixed164_all10_raw_not_old2packetdims(jobs):
    assert len(jobs)==164 and sum(j['coord'] is None for j in jobs)==4
    assert all(j['legal'] and not j['deployment_allowed'] and not j['physical_release'] for j in jobs)
    assert len({j['job_sha256'] for j in jobs})==len({j['id'] for j in jobs})==164
    for slot in range(4):
        local=jobs[41*slot:41*(slot+1)];base=local[0]
        assert base['h'] is base['coord'] is None and base['sign']==0
        assert [(j['h'],j['coord'],j['sign']) for j in local[1:]]==[(h,c,s) for h in m.STEPS for c in range(10) for s in (-1,1)]
        for j in local:
            assert j['slot']==slot and j['full10'][:5]==base['prefix5']
            assert j['full10'][5:]==j['raw_suffix5']
            assert j['job_sha256']==m.digest({k:v for k,v in j.items() if k!='job_sha256'})
            assert j['task']==base['task'] and j['observed_context']==base['observed_context']
            if j['coord'] is not None:
                x=np.asarray(base['raw_suffix5'],np.float32).reshape(10);y=np.asarray(j['raw_suffix5'],np.float32).reshape(10)
                assert [k for k in range(10) if x[k].tobytes()!=y[k].tobytes()]==[j['coord']]
                assert float(y[j['coord']])-float(x[j['coord']])==j['actual_delta']


@pytest.mark.parametrize('bad',['missing','rows','selected','Q','receipt','budget','goal','bool'])
def test_upstream_envelope_tamper_rejected(envelopes,bad):
    p,r,a,h=deepcopy(envelopes)
    if bad=='missing':p['rows'].pop()
    elif bad=='rows':a['rows'][0]['status']='not_started'
    elif bad=='selected':r['rows'][0]['result']['numeric']['iterates'][0]['commands'][0][0]=.9
    elif bad=='Q':r['rows'][0]['result']['numeric']['iterates'][0]['Q']+=1
    elif bad=='receipt':a['complete']=False
    elif bad=='budget':a['actual_new_compute']['completed']['predict_API']=1
    elif bad=='goal':p['rows'][0]['request']['task']['goal'][0]+=.1
    else:p['rows'][0]['request']['observed_raw10'][0][0]=True
    with pytest.raises((ValueError,IndexError)):m.plan_jobs(p,r,a,h)


@pytest.mark.parametrize('coord,h,sign',[(True,.01,1),(0,True,1),(0,.01,True),(10,.01,1),(0,.0001,1)])
def test_fixed_only_literal_coordinates_h_sign(coord,h,sign):
    with pytest.raises(ValueError):m.perturb(np.zeros((5,2)).tolist(),coord,h,sign)


def test_rawbox_not_L2_no_clamp_and_exact_float32_span():
    x=np.zeros((5,2),np.float32);x[0,0]=1.
    y,d,reason=m.perturb(x.tolist(),0,.01,1)
    assert y[0,0]>1 and reason=='outside_rawbox'
    x[0,0]=np.float32(.2);minus=m.perturb(x.tolist(),0,.01,-1)[0];plus=m.perturb(x.tolist(),0,.01,1)[0]
    assert float(plus[0,0])-float(minus[0,0])!=.02
    x[4,1]=-0.
    for c in range(9):assert np.signbit(m.perturb(x.tolist(),c,.001,1)[0][4,1])
    with pytest.raises(ValueError):m.packet([[.1,0.]]*5)


class RAM(prior.n.parent.RAM):
    def objective(self,after):
        y=np.asarray(after['latent_history'][-1],np.float32);g=np.asarray(self.task['goal'],np.float32)
        d=y-g;s=d*d
        return {k:m.signal.blob(v) for k,v in dict(endpoint=y,goal=g,residual=d,square=s,loss=np.asarray([s.mean()],np.float32)).items()}


@pytest.fixture(scope='module')
def recorded(jobs):
    point=m.points_from_jobs(jobs)[0];j=prior.n.parent.Events();l=m.core.Ledger(m.F.caps(41,0),sink=j.event)
    row=m.new_row(0);m.evaluate(point,jobs[:41],RAM(point['task'],j,l),row,j);row['resources']=l.receipt()
    return point,jobs[:41],row,j.events


def test_real_format41_native_forwards_no_AD(recorded):
    p,js,row,es=recorded;summary=m.audit_row(p,js,row,es)
    assert row['resources']['completed']==m.F.caps(41,0)
    assert len(summary['steps'])==2 and all(len(s['pairs'])==10 for s in summary['steps'])
    assert summary['baseline']['exact_original_iterate0'] and summary['baseline']['exact_original_AD']
    for s in summary['steps']:
        for pair in s['pairs']:
            d=pair['native_Q_plus']-pair['native_Q_minus']
            assert pair['FD_native_Q_actual_spacing']==d/pair['actual_float32_spacing']
            assert pair['FD_native_Q_nominal_spacing']==d/pair['nominal_spacing']
            assert len(pair['endpoint192_derivative_actual_spacing'])==192


@pytest.mark.parametrize('bad',['unknown','native_splice','intent_splice','AD','module','raw','Q','missing','context','baseline','index'])
def test_native_full_lifecycle_mutations(recorded,bad):
    p,js,row,es=deepcopy(recorded)
    if bad=='unknown':es.insert(1,dict(kind='unexpected_native'))
    elif bad=='native_splice':
        e=next(e for e in es if e['kind']=='program_packet');es.remove(e);es.insert(-1,e)
    elif bad=='intent_splice':
        e=es.pop(0);es.insert(4,e)
    elif bad=='AD':row['resources']['completed']['grad_API']=1
    elif bad=='module':
        e=next(e for e in es if 'predict:output' in e.get('arrays',{}));x=m.signal.unblob(e['arrays']['predict:output']).copy();x.flat[0]+=.1;e['arrays']['predict:output']=m.signal.blob(x)
    elif bad=='raw':next(e for e in es if e['kind']=='forward_started')['commands'][0][0]+=.1
    elif bad=='Q':row['values'][1]['objective']['diagnostic']['native_float32_Q']+=1
    elif bad=='missing':row['values'].pop()
    elif bad=='context':row['values'][1]['context']['latent_history'][0][0]+=.1
    elif bad=='baseline':js[0]['original_iterate0_context']['latent_history'][-1][0]+=.1
    else:row['values'][0]['local_index']=False
    with pytest.raises((ValueError,KeyError)):m.audit_row(p,js,row,es)


def test_actual_baseline_runtime_mismatch_stops_before_second_probe(jobs):
    js=deepcopy(jobs[:41]);js[0]['original_iterate0_context']['latent_history'][-1][0]+=.1
    p=m.points_from_jobs(jobs)[0];j=prior.n.parent.Events();l=m.core.Ledger(m.F.caps(41,0),sink=j.event);row=m.new_row(0)
    with pytest.raises(ValueError,match='baseline MUST'):m.evaluate(p,js,RAM(p['task'],j,l),row,j)
    assert l.completed['predict_API']==1 and row['values'][0]['status']=='entered'
    assert all(v['status']=='not_started' for v in row['values'][1:])


def test_zero_response_and_zero_AD_cosine_remain_null(recorded):
    p,js,row,_=deepcopy(recorded);p['raw_gradient']=np.zeros((5,2)).tolist()
    for v in row['values'][1:]:v['objective']=deepcopy(row['values'][0]['objective']);v['context']=deepcopy(row['values'][0]['context'])
    s=m.summarize(p,js,row['values'])
    for step in s['steps']:
        assert step['metrics_actual_spacing']['cosine'] is step['metrics_actual_spacing']['relative_L2'] is None
        assert all(x['FD_native_Q_actual_spacing']==0 for x in step['pairs'])


@pytest.mark.parametrize('fail',[False,True])
def test_full164_durable_dispatch_or_partial_denominator(jobs,tmp_path,monkeypatch,fail):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    p=dict(points=m.points_from_jobs(jobs),jobs=jobs,production_ready=True);rows=[m.new_row(i) for i in range(4)]
    store=m.Store(tmp_path/'results'/(m.NAME+'_RAM'),time.monotonic());count=0
    def factory(task,j,l):
        nonlocal count
        count+=1
        if fail and count==2:raise RuntimeError('RAM failure')
        return RAM(task,j,l)
    if fail:
        with pytest.raises(RuntimeError):m.dispatch(p,store,rows,lambda:None,factory)
        assert [r['status'] for r in rows]==['recorded','failed_partial','not_started','not_started']
    else:
        m.dispatch(p,store,rows,lambda:None,factory)
        assert sum(r['resources']['completed']['predict_API'] for r in rows)==164
    for i,r in enumerate(rows):
        if r['resources'] is None:continue
        es=m.F.load_journal(store.path,r);m.transport.audit_resources(r['resources'],es)
        if r['status']=='recorded':m.audit_row(p['points'][i],jobs[41*i:41*(i+1)],r,es)
    # Saved full consumer under explicitly RAM-only replacement of upstream verify.
    h='d'*64;p.update(bindings={},original_model_before=dict(original_state_dict=h),original_model_after=dict(original_state_dict=h),
        authority={},manifest_sha256=m.digest(jobs),inherited_compute={},inherited_resources={})
    prepared=tmp_path/'fixture.json';prepared.write_text('RAM only');ph=m.sha(prepared)
    monkeypatch.setattr(m,'verify',lambda f,x:p if str(f)==str(prepared) and x==ph else m.require(False,'RAM binding'))
    store.put('binding.json',dict(prepared=str(prepared),prepare_sha256=ph))
    precision=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)
    report=dict(schema=m.NAME+'.report',contract=m.contract(),rows=rows,prepared=str(prepared),prepare_sha256=ph,
        status='partial_failclosed' if fail else 'observed164_recorded_diagnostic_only',errors=['RAM failure'] if fail else [],
        fixed_denominator=164,source_verified=True,source_manifest_sha256=m.digest({}),model_before=p['original_model_before'],
        model_after=p['original_model_after'],precision_before=precision,precision_after=precision,CUDA_peak_bytes=[0,0],wall_seconds=1.,
        artifact_sha256={str(f.relative_to(store.path)):m.sha(f) for f in store.path.rglob('*') if f.is_file()},
        AD_correctness_certified=False,P1_admission=False,freshness=False,scientific_admission=False,physical_release=False)
    store.put('report.json',report,final=True);receipt=m.consume(store.path,m.sha(store.path/'report.json'))
    assert receipt['complete'] is (not fail) and receipt['fixed_denominator']==164
    assert len(receipt['panels'])==(1 if fail else 4)
    (store.path/'unexpected').write_text('extra')
    with pytest.raises(ValueError):m.consume(store.path,m.sha(store.path/'report.json'))


def test_no_worker_neural_and_no_extra_operations():
    c=m.contract();assert c['forwards']==164 and c['max_new_compute']==m.F.caps(164,0)
    assert c['output_bytes']==128*1024**2 and c['wall_seconds']==1800
    tree=ast.parse(Path(m.__file__).read_text())
    forbidden={'gradient','grad','backward','jacobian','jacrev','jacfwd','setattr'}
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden for n in ast.walk(tree))
    with pytest.raises(ValueError):m.run('/not-read','0'*64,'/not-created','NO')


def test_binding_and_prepare_hash_mutation(tmp_path):
    f=tmp_path/'binding';f.write_text('original');bindings={str(f):m.sha(f)};m.F.check_bindings(bindings)
    f.write_text('tampered')
    with pytest.raises(ValueError):m.F.check_bindings(bindings)
    with pytest.raises(ValueError):m.verify(f,'0'*64)


def test_actual_saved_schema_manifest_only_not_reconsume():
    root=m.ROOT/'results';stem=m.source.NAME;prep=root/(stem+'_prepare_MAIN01.json');output=root/(stem+'_cuda_MAIN01');receipt=root/(stem+'_consume_MAIN01.json')
    if not receipt.exists():pytest.skip('optional exact MAIN saved metadata unavailable')
    p=m.signal.read(prep,32*1024**2);r=m.signal.read(output/'report.json',32*1024**2);a=m.signal.read(receipt,32*1024**2)
    authority=dict(prepared=str(prep),prepare_sha256=m.sha(prep),output=str(output),report_sha256=m.sha(output/'report.json'),receipt=str(receipt),receipt_sha256=m.sha(receipt))
    assert authority['report_sha256']=='3de58e380acbef66269e69150cab7bca312604b7df88a0f1ae5663764d691b45'
    jobs=m.plan_jobs(p,r,a,authority)
    assert len(jobs)==164 and all(j['legal'] for j in jobs)
    # External metadata cap is unchanged; full task in every job must actually fit.
    assert len(m.raw(jobs))+2*1024**2<32*1024**2
