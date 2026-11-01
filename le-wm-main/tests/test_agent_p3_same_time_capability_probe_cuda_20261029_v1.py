"""Explicit RAM-only native transcript and temporary build/publication fixtures."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_same_time_capability_probe_cuda_20261029_v1 as m

REAL_ROOT=m.ROOT
# Reuse the already reviewed complete NumPy native-IO fixture, not its producer.
# Loading this test module defines callbacks only; no old tests/production run.
OLD=runpy.run_path(str(REAL_ROOT/'tests/test_agent_p3_parent_native_cuda_20261029_v1.py'))
NumpyNative=OLD['NumpyNative']


def cp5(root):
    t=OLD['original']['task']()
    t['binding'].update(model_sha256=m.digest('RAM model'),task_id=m.digest(['RAM cp5',root]),root_sha256=m.digest(['root',root]))
    return m.core.validate_task(t)


def bridge_fixture():
    """Literal final Jason schema; no imaginary report fields or native evidence."""
    fs={k:False for k in ('P1_admission','P3_admission','physical_release','scientific_admission',
         'full_C3_demonstrated','equal_FLOPs','equal_total_compute','execution_authorized','freshness','independent_confirmation','training')}
    authority={k:dict(path='/fixture/'+k+'.json',sha256=m.digest(k)) for k in ('prepare','report','completion','receipt','source')}
    counts={k:dict(attempted=5 if k in ('encoder','projector') else 0,completed=5 if k in ('encoder','projector') else 0)
            for k in ('encoder','projector','action_encoder','predict_API','predictor','pred_proj')}
    bp=dict(schema=m.BRIDGE+'.prepare',rows=[],expected_model_hash=m.digest('RAM model'),contract={'fixture':'explicit RAM'},
            policy='/fixture/model',bindings={})
    br=dict(schema=m.BRIDGE+'.report',status='all5_cp10_RGB_and_actual_history_tasks_sealed_NOT_admitted',
            prepared=authority['prepare']['path'],prepare_sha256=authority['prepare']['sha256'],
            rows=[],errors=[],blocked=[],fixed_denominator=5,model_unchanged=True,source_verified=True,
            model_hash_before=bp['expected_model_hash'],model_hash_after=bp['expected_model_hash'],
            contract=deepcopy(bp['contract']),counts=deepcopy(counts),**fs)
    bc=dict(schema=m.BRIDGE+'.completion',status='published_within_cap',report_sha256=authority['report']['sha256'],
            prepared=br['prepared'],prepare_sha256=br['prepare_sha256'],**fs)
    tasks=[]
    for i,(root,role) in enumerate(m.ORDER):
        old=cp5(root)
        incoming=np.zeros((10,2),np.float32)
        if role!='Z':incoming[5:]=np.float32(.125)
        tokens=((incoming-np.asarray(old['action_mean'],np.float32))/np.asarray(old['action_std'],np.float32)).reshape(2,10)
        source=dict(png=f'cp10_{i}.png',state=[float(i)]*7,snapshot=dict(state=[float(i)]*7),
                    file_sha256=m.digest(['png',i]),pixel_sha256=m.digest(['pixel',i]))
        hist=[dict(absolute_step=s,file_sha256=m.digest([i,s]),pixel_sha256=m.digest(['pixel',i,s])) for s in (10,15,20)]
        descriptor=dict(file=f'latent_{i}.npy.gz',sha256=m.digest(['npy',i]),shape=[192],dtype='float32')
        spec=dict(index=i,root_slot=root,role=role,cp5_task=old,base_task_id=m.digest(['base',root]),
                  source=source,history_sources=hist,source_report=dict(path='/fixture/physical_report',sha256=m.digest('physical')),
                  source_receipt=dict(path='/fixture/physical_receipt',sha256=m.digest('physical receipt')),
                  causal_raw10=incoming.tolist(),causal_raw10_bits=incoming.tobytes().hex(),
                  normalized_action_history=tokens.tolist(),absolute_steps=[10,15,20])
        identity=dict(namespace=m.BRIDGE,index=i,root_slot=root,role=role,absolute_steps=[10,15,20],history_sources=hist,
                      raw10_float32_sha256=hashlib.sha256(incoming.tobytes()).hexdigest(),old_cp5_task_sha256=m.digest(old),
                      source_report=spec['source_report'],source_receipt=spec['source_receipt'],encoded_latent=descriptor)
        c=deepcopy(old['root_context']); z=np.zeros(192,np.float32);z[0]=np.float32(i/8)
        c['latent_history']=c['latent_history'][1:]+[z.tolist()];c['action_history']=tokens.tolist()
        t=deepcopy(old);t['root_context']=c
        t['binding'].update(context_sha256=m.digest(c),task_id=m.digest([old['binding']['task_id'],identity]),
            task_sha256=m.digest([old,c,identity]),task_contract_sha256=m.digest([m.BRIDGE,identity]))
        item=dict(index=i,root_slot=root,role=role,task=t,task_sha256=m.digest(t),identity=identity,
                  old_cp5_task_id=old['binding']['task_id'],base_task_id=spec['base_task_id'],
                  absolute_steps=[10,15,20],source=source,causal_raw10=incoming.tolist(),**fs)
        tasks.append(item);bp['rows'].append(spec);br['rows'].append(dict(spec,status='encoded',latent=descriptor))
    bs=dict(schema=m.BRIDGE+'.consume',status=br['status'],report_sha256=authority['report']['sha256'],
            completion_sha256=authority['completion']['sha256'],completion=bc,tasks=tasks,
            fixed_denominator=5,known_roots=2,fresh_roots=0,consume_new_neural_calls=0,consume_new_physics=0,
            counts=counts,bindings={},inherited_physical_counts={'C':{'public':20},'ZP':{'public':80}},**fs)
    return json.loads(m.raw([bp,br,bc,bs,authority,tasks]))


def panel():
    bp,br,bc,bs,a,tasks=bridge_fixture()
    p=m.assemble(bp,br,bc,bs,a,tasks)
    p.update(contract=m.contract(),expected_model_hash=bp['expected_model_hash'])
    return p


@pytest.fixture(scope='module')
def saved():
    p=panel();sink=m.Memory()
    result=m.execute(p,lambda t,s:NumpyNative(t,s),sink,permit_ram=True)
    assert result['errors']==[]
    return p,result,sink.events,sink.files['native_seal.json']


def test_full7_signals_5U_22Q_all_native_arrays_and_resources(saved):
    p,r,events,seal=saved
    v=m.validate(p,r,events,seal,permit_fixture=True)
    assert v['full_native_closure'] and v['saved_SVD_calls']==7 and v['native_calls']==17
    assert r['resource_receipt']['attempted']==r['resource_receipt']['completed']==m.caps()
    assert m.caps()['predict_API']==24 and m.caps()['VJP_directions']==1344 and m.caps()['metric_calls']==22
    assert len(r['signals'])==7 and len(r['probes'])==5 and len(r['metrics'])==22
    assert [s['time'] for s in p['signals']]==[5,5,10,10,10,10,10]
    assert [(s['root'],s['role']) for s in p['endpoints']]==m.ORDER
    assert all(len(s['response']['jacobian'])==192 and len(s['response']['jacobian'][0])==10 and
               len(s['solve']['singular_values'])==10 for s in r['signals'])
    assert all(q['whole']==q['segmented'] for q in r['probes'])
    i=next(i for i,e in enumerate(events) if e['kind']=='native_sealed')
    assert all(e['entry']['resource']!='metric' for e in events[:i] if e['kind']=='resource_attempt')
    assert sum(m.signal.unblob(b).nbytes for e in events if 'arrays' in e for b in e['arrays'].values())==m.shape_census()['all_repeated_native_array_bytes']
    assert len(events)<16000 and len(r['resource_receipt']['events'])==4193


@pytest.mark.parametrize('damage',['goal','normalizer','root','raw','order','missing','source','history','namespace','falseflag','counts'])
def test_final_bridge_ABI_mutations(damage):
    bp,br,bc,bs,a,t=bridge_fixture()
    if damage=='goal':t[0]['task']['goal'][0]+=1.
    elif damage=='normalizer':bp['rows'][0]['cp5_task']['action_mean'][0]+=.1
    elif damage=='root':t[0]['root_slot']=1
    elif damage=='raw':t[0]['causal_raw10'][0][0]=.25
    elif damage=='order':t.reverse()
    elif damage=='missing':t.pop()
    elif damage=='source':t[0]['source']['file_sha256']='f'*64
    elif damage=='history':bp['rows'][0]['history_sources'][2]['absolute_step']=15
    elif damage=='namespace':t[0]['identity']['namespace']='old cp5'
    elif damage=='falseflag':bs['full_C3_demonstrated']=True
    else:bs['counts']['encoder']['completed']=4
    with pytest.raises((ValueError,KeyError,IndexError)):
        m.assemble(bp,br,bc,bs,a,t)


@pytest.mark.parametrize('damage',['J','VJP','raw','older','extra','double','metric','sourcegoal'])
def test_full_native_mutations(saved,damage):
    p,r,events,seal=deepcopy(saved)
    if damage=='J':r['signals'][0]['response']['jacobian'][0][0]+=.25
    elif damage=='VJP':events.remove(next(e for e in events if e['kind']=='direction'))
    elif damage in ('raw','older'):
        e=next(e for e in events if e['kind']=='origin');key='raw_flat' if damage=='raw' else 'latent'
        x=m.signal.unblob(e['arrays'][key]).copy();x.flat[0]+=.25;e['arrays'][key]=m.signal.blob(x)
    elif damage=='extra':events.append(dict(kind='outcome',success=False))
    elif damage=='double':r['probes'][0]['segmented'][0]['latent_history'][0][0]+=.25
    elif damage=='metric':r['metrics'][0]['Q']+=.25
    else:p['signals'][0]['task']['goal'][0]+=.25
    with pytest.raises((ValueError,KeyError)):
        m.validate(p,r,events,seal,permit_fixture=True)


class ADInterrupted(NumpyNative):
    def arrays(self,kind,index=None,**values):
        super().arrays(kind,index,**values)
        if kind=='AD_terminal':raise RuntimeError('RAM timeout after actual AD forward, before J rows')


class ThirdReturned(NumpyNative):
    def call(self,name,fn,inputs,shape=None):
        out=super().call(name,fn,inputs,shape)
        if name=='pred_proj':raise RuntimeError('RAM third module returned, outer predict has not returned')
        return out


@pytest.mark.parametrize('backend',[ADInterrupted,ThirdReturned])
def test_mid_AD_or_module_failure_transport_no_fabricated_return(backend):
    p=panel();sink=m.Memory();r=m.execute(p,lambda t,s:backend(t,s),sink,permit_ram=True)
    assert r['status']=='partial_or_failed' and r['signals'][0]['status']=='failed'
    assert all(s['status']=='blocked_after_failure' for s in r['signals'][1:]+r['probes'])
    assert not r['metrics'] and 'native_seal.json' not in sink.files
    assert r['resource_receipt']['attempted']['predict_API']>0
    assert r['resource_receipt']['completed']['metric_calls']==0
    assert not m.validate(p,r,sink.events,None,permit_fixture=True)['full_native_closure']
    if backend is ThirdReturned:
        assert r['resource_receipt']['completed']['pred_proj_calls']==1
        assert r['resource_receipt']['completed']['predict_API']==0


def test_public_consumer_rejects_implicit_RAM(saved):
    with pytest.raises(ValueError,match='RAM vs actual HOST'):
        m.validate(*saved)


def test_shape_and_publication_proof(saved):
    c=m.shape_census()
    assert c['fits'] and c['all_repeated_native_array_bytes']==2910896
    assert c['whole_upper_bytes']==1390542848 < 2*1024**3
    assert c['final_upper_including_recovery_bytes']<=256*m.MiB
    assert max(len(m.raw(e))+512 for e in saved[2] if e['kind'] not in m.SPECIAL)<64*1024
    assert len(m.raw(saved[1]['resource_receipt']))<96*m.MiB
    assert m.contract()['damping']==.002 and m.contract()['cap']==float(np.sqrt(5)*.75)
    params=np.linspace(-3,3,10)
    assert np.asarray(m.components.suffix_raw(params.tolist())).astype('<f4').tobytes()==np.tanh(params).astype('<f4').reshape(5,2).tobytes()


def temp_build(tmp_path,monkeypatch):
    """Full build/verify, actual source bytes; fake authorities are clearly RAM."""
    root=tmp_path/'work';(root/'experiments/spectral_reachability').mkdir(parents=True)
    (root/'tests').mkdir();(root/'innerdocs').mkdir();(root/'results').mkdir()
    for rel in ['experiments/checkpoint_compat.py','experiments/gpu_preflight.py',
                'experiments/spectral_reachability/'+m.BRIDGE+'.py','tests/test_'+m.NAME+'.py','innerdocs/'+m.NAME+'.md']:
        (root/rel).write_bytes((REAL_ROOT/rel).read_bytes())
    bp,br,bc,bs,a,tasks=bridge_fixture();o=root/'results/bridge_cuda';o.mkdir()
    policy=root/'fixture_model';policy.mkdir();(policy/'weights.pt').write_bytes(b'RAM fixture bytes, NEVER loaded')
    (policy/'config.json').write_bytes(b'{"fixture":true}')
    bp.update(policy=str(policy),bindings={str(f):m.sha(f) for f in policy.iterdir()})
    refs={k:root/'results'/('bridge_'+k+'.json') for k in ('prepare','receipt')}
    refs.update(report=o/'report.json',completion=o/'completion.json',source=root/'experiments/spectral_reachability'/(m.BRIDGE+'.py'))
    def put(k,value):
        refs[k].write_bytes(m.raw(value));a[k]=dict(path=str(refs[k]),sha256=m.sha(refs[k]))
    put('prepare',bp)
    br.update(prepared=a['prepare']['path'],prepare_sha256=a['prepare']['sha256'])
    taskfile=o/'cp10_tasks.json';taskfile.write_bytes(m.raw(tasks))
    h=m.sha(taskfile);br.update(tasks_sha256=h,artifact_sha256={'cp10_tasks.json':h})
    put('report',br)
    bc.update(prepared=a['prepare']['path'],prepare_sha256=a['prepare']['sha256'],report_sha256=a['report']['sha256'])
    put('completion',bc)
    bs.update(report_sha256=a['report']['sha256'],completion_sha256=a['completion']['sha256'],completion=bc,
              bindings={str(taskfile):h})
    put('receipt',bs)
    a['source']=dict(path=str(refs['source']),sha256=m.sha(refs['source']))
    monkeypatch.setattr(m,'ROOT',root)
    p=m.build(a)
    path=root/'results'/(m.NAME+'_prepare_RAM.json');m.write_new(path,p,32*m.MiB)
    assert m.verify(str(path),m.sha(path))==p
    return p,path,m.sha(path),root


def test_full_temp_build_store_publication_reader_consumer(tmp_path,monkeypatch):
    p,path,h,root=temp_build(tmp_path,monkeypatch)
    output=root/'results'/(m.NAME+'_RAM')
    store=m.Store(output,free=lambda:10*1024**3)
    store.put('binding.json',dict(prepared=str(path),prepare_sha256=h))
    r=m.execute(p,lambda t,s:NumpyNative(t,s),store,permit_ram=True)
    before=dict(original_state_dict=p['expected_model_hash'],all_buffers=m.digest('RAM buffers'))
    classes=[dict(name='fixture',class_name='RAM_NOT_MODEL',path=m.__file__,sha256=m.sha(m.__file__))]
    store.put('classes.json',dict(before=classes,after=classes),final=True)
    store.event(dict(kind='closure',model_before=before,model_after=before,sources_verified=True))
    report=dict(schema=m.NAME+'.report',status=r['status'],prepared=str(path),prepare_sha256=h,
                result=r,errors=[],model_before=before,model_after=before,host=dict(device='RAM_FIXTURE_NOT_CUDA'),
                precision=[m.native.precision_contract()]*2,sources_verified=True,bindings_sha256=m.digest(p['bindings']),
                CUDA_peak_bytes=[0,0],elapsed_seconds=store.clock()-store.started,contract=m.contract(),**m.FLAGS)
    publication=m.publish(store,report)
    receipt=m._consume(str(output),publication['report_sha256'],publication['completion_sha256'],str(path),h,permit_fixture=True)
    assert receipt['status']=='RAM_fixture_only_NOT_evidence' and len(receipt['probes'])==5
    assert receipt['verification']['saved_SVD_calls']==7
    assert len(receipt['plans'])==5 and [(x['root_slot'],x['role']) for x in receipt['plans']]==m.ORDER
    for i,plan in enumerate(receipt['plans']):
        copy=deepcopy(plan);hplan=copy.pop('witness_sha256');assert m.digest(copy)==hplan
        assert plan['native_call_ids']==dict(signal=i+2,whole=7+2*i,segmented=8+2*i)
        assert plan['raw_float32_bits']==np.asarray(plan['raw5'],'<f4').tobytes().hex()
        assert plan['causal_raw10_bits']==np.asarray(plan['causal_raw10'],'<f4').tobytes().hex()
    receipt_path=root/'results'/(m.NAME+'_consume_RAM.json')
    m.write_new(receipt_path,receipt)
    with pytest.raises(ValueError,match='actual complete own receipt'):
        m.witness(str(receipt_path),m.sha(receipt_path),0)
    # Internal explicit RAM path exercises the full exporter without forging a
    # CUDA status or making a fixture eligible for the public CLI.
    w=m._witness(str(receipt_path),m.sha(receipt_path),0,permit_fixture=True)
    assert w['plan']==receipt['plans'][0] and w['evidence_kind']=='RAM_fixture_NOT_evidence'
    bad=deepcopy(receipt);bad['plans'][0]['raw5'][0][0]+=.25
    bad_path=root/'results'/(m.NAME+'_bad_RAM.json');m.write_new(bad_path,bad)
    with pytest.raises(ValueError,match='full witness content hash'):
        m._witness(str(bad_path),m.sha(bad_path),0,permit_fixture=True)
    with pytest.raises(ValueError,match='actual CUDA preflight'):
        m.consume(str(output),publication['report_sha256'],publication['completion_sha256'],str(path),h)
    events,tail=m.journal(output)
    assert events[0]['kind']=='denominator' and tail==m.read(output/'report.json')['journal_tail']
    j=output/'events.jsonl';j.write_bytes(j.read_bytes().replace(b'"kind":"denominator"',b'"kind":"denominatoX"',1))
    with pytest.raises(ValueError):m.journal(output)


def test_real_store_refusal_no_callback_or_fake_full_transport(tmp_path):
    clock=[0.];store=m.Store(tmp_path/'store',clock=lambda:clock[0],free=lambda:10*1024**3)
    # Force exactly the ordinary allowance exhausted; reserve remains available.
    store.bytes=m.LIMITS['output_bytes']-m.LIMITS['final_reserve_bytes']
    called=[];ledger=m.core.Ledger(m.caps(),sink=store.event);ledger.scope=dict(test=True)
    with pytest.raises(ValueError,match='output cap'):
        ledger.call('signal_compile',lambda:called.append(True))
    assert called==[] and store.broken and store.recovery
    assert ledger.attempted['signal_compile_calls']==1 and ledger.completed['signal_compile_calls']==0
    store.event(dict(kind='execution_failure',error={'type':'ValueError','message':'ordinary quota exhausted'}))
    store.close()
    events,_=m.journal(store.output)
    assert [e['kind'] for e in events]==['resource_failure','execution_failure']
    # Intent was not durable: saved strict must not fabricate it or certify this.
    with pytest.raises(ValueError):m.resource_census(ledger.receipt(),events)


def test_final_fsync_wall_failure_is_retained(tmp_path):
    clock=[0.];store=m.Store(tmp_path/'late',clock=lambda:clock[0],free=lambda:10*1024**3)
    original=store.put
    def put(name,value,final=False):
        result=original(name,value,final)
        if name=='completion.json':clock[0]=1801.
        return result
    store.put=put
    with pytest.raises(ValueError,match='publication wall exceeded'):
        m.publish(store,dict(prepared='/fixture/prepare',prepare_sha256='f'*64))
    assert (store.output/'report.json').exists() and (store.output/'publication_failure.json').exists()


def test_cold_import_cli_and_auth_before_all_IO():
    code=f'import sys; from experiments.spectral_reachability import {m.NAME} as m; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules; m.main(["--help"])'
    x=subprocess.run([sys.executable,'-c',code],cwd=REAL_ROOT,capture_output=True,text=True)
    assert x.returncode==0 and all(s in x.stdout for s in ('prepare','run','consume','verify'))
    with pytest.raises(ValueError,match='MAIN explicit'):
        m.run('absent','f'*64,'absent','not authorized')


def test_digest_domain_and_absolute_path():
    assert m.digest([1,2])==hashlib.sha256(b'[1,2]').hexdigest()
    assert m.digest([1,2])!=hashlib.sha256(b'[1,2]\n').hexdigest()
    with pytest.raises(ValueError,match='canonical absolute'):
        m.absolute('relative.json')
