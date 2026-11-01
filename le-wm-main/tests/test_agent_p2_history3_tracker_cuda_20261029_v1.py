"""RAM arithmetic, metadata and saved-array ABI only; never load a model."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_history3_tracker_cuda_20261029_v1 as m


@pytest.fixture
def store(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path)
    (tmp_path/'results').mkdir()
    return m.Store(tmp_path/'results'/ (m.TAG+'_fixture'))


def kernels(fail=False):
    def check(x):
        m.need(type(x) is np.ndarray and x.dtype==np.float32 and np.isfinite(x).all(),'RAM f32')
    def ae(a):
        if fail: raise RuntimeError('fixture failed AE')
        return np.repeat(a.sum(-1,keepdims=True,dtype=np.float32),192,axis=-1)
    return m.core.low.Kernels(lambda xs,dim:np.concatenate(xs,axis=dim),
        lambda xs,dim:np.stack(xs,axis=dim),check,ae,lambda z,a:z+a/np.float32(16),lambda x:x.copy(),'ram_fixture')


def values():
    return dict(z=np.arange(576,dtype=np.float32).reshape(1,3,192)/1024,
                a=np.arange(20,dtype=np.float32).reshape(1,2,10)/1024,
                target=np.full((1,192),.25,np.float32))


def test_fixed_protocol():
    p=m.protocol(); s=m.spec()
    assert (s.horizon,s.samples,s.iterations,s.topk,s.seed,s.segments)==(3,8,2,2,0,(1,2))
    assert p['per_root']['predict_API']==12 and p['per_root']['predictor_samples']==54
    assert p['total_predict_API']==24 and p['total_predictor_samples']==108
    assert not p['scientific_admission'] and not p['raw_inverse'] and p['FLOPs'] is None
    assert p['encoder']==p['high']==p['AD']==p['physics']==0


def test_real_saved_abi_metadata_and_inputs_only():
    # Narrow fixed paths: small saved arrays and JSON, never weights/full consume.
    base=m.ROOT/'results'
    pm=m.read(base/'agent_p2_hilewm_computation_parity_20261029_v1_prepare_MAIN01.json',m.ACTUAL['manifest'])
    folder=base/'agent_p2_hilewm_computation_parity_20261029_v1_cuda_MAIN01'
    pr=m.read(folder/'report.json',m.ACTUAL['report'])
    receipt=m.read(base/'agent_p2_hilewm_computation_parity_20261029_v1_consume_MAIN01.json',m.ACTUAL['receipt'])
    slots,arr=m.select_inputs(pm,pr,folder)
    assert [s['root'] for s in slots]==[0,1] and receipt['all_values_exact'] is True
    assert all(x['z'].shape==(1,3,192) and x['a'].shape==(1,2,10) and x['target'].shape==(1,192) for x in arr)
    er=m.read(pm['export']['report'],pm['export']['report_sha256'])
    state=m.expected_state(er,pm['expected_hi_hashes']); assert len(state)==425
    bad=deepcopy(er); bad['rows'][0]['array']['value_sha256']='0'*64
    with pytest.raises(ValueError):m.expected_state(bad,pm['expected_hi_hashes'])


@pytest.mark.parametrize('fault',['order','root','history','actions','rgb','goal'])
def test_real_saved_join_mutations_without_fullstrict(fault):
    b=m.ROOT/'results'; pm=m.read(b/'agent_p2_hilewm_computation_parity_20261029_v1_prepare_MAIN01.json',m.ACTUAL['manifest'])
    folder=b/'agent_p2_hilewm_computation_parity_20261029_v1_cuda_MAIN01'
    pr=m.read(folder/'report.json',m.ACTUAL['report'])
    if fault=='order':pr['rows'][0]['pixels'][0]['image']=1
    elif fault=='root':pm['slots'][0]['root']=1
    elif fault=='history':pm['slots'][0]['z'][0][0]+=1
    elif fault=='actions':pm['slots'][0]['a'][0][0]+=1
    elif fault=='rgb':pm['slots'][0]['pixel_sha256'][0]='0'*64
    else:pr['rows'][0]['pixels'][3]['paths']['hi']['output']=pr['rows'][0]['pixels'][0]['paths']['hi']['output']
    with pytest.raises(ValueError):m.select_inputs(pm,pr,folder)


def test_prepare_verify_fixture_and_exclusive_cli(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(m,'ROOT',tmp_path); (tmp_path/'results').mkdir()
    auth={k:dict(path='/fixture/'+k,sha256=v) for k,v in m.ACTUAL.items()}
    manifest=dict(schema='explicit_RAM_fixture_not_HOST',authorities=auth,payload=[0,1])
    monkeypatch.setattr(m,'build',lambda a:deepcopy(manifest))
    p=tmp_path/'results'/(m.TAG+'_prepare_fixture.json')
    out=m.prepare(p,auth); assert m.verify(p,out['sha256'])==manifest
    m.main(['verify','--manifest',str(p),'--manifest-sha256',out['sha256']])
    assert json.loads(capsys.readouterr().out)['verified'] is True
    with pytest.raises(FileExistsError):m.prepare(p,auth)
    manifest['payload']=[1,0]
    with pytest.raises(ValueError):m.verify(p,out['sha256'])


def test_full_actual_receipt_comparison_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path); (tmp_path/'results').mkdir()
    def write(n,v):
        p=tmp_path/n; m.durable(p,m.encode(v)); return dict(path=str(p),sha256=m.sha(p))
    p=write('prepare.json',{'fixture':True})
    r=write('report.json',dict(manifest=p['path'],manifest_sha256=p['sha256']))
    original=dict(all_values_exact=True,raw_deployment=False,scientific_admission=False,detail=[1,2])
    c=write('receipt.json',original); auth=dict(manifest=p,report=r,receipt=c)
    def consume(report,rh,out):m.durable(out,m.encode(original))
    monkeypatch.setattr(m.parity,'consume',consume) # RAM owner callback, no production invocation.
    assert m.certify_parity(auth)[2]==original
    original['detail']=[2,1]
    with pytest.raises(ValueError,match='complete actual parity receipt'):m.certify_parity(auth)


@pytest.mark.parametrize('key',['manifest','report','receipt'])
def test_no_future_authority(key):
    hashes=dict(m.ACTUAL); hashes[key]='0'*64
    with pytest.raises(ValueError,match='fixed actual'):
        m.authorities('/no',hashes['manifest'],'/no',hashes['report'],'/no',hashes['receipt'])


@pytest.mark.parametrize('shape',[(),(1,),(1,3,192),(1,2,10)])
@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.int64])
def test_durable_array_exact_roundtrip(store,shape,dtype):
    a=np.zeros(shape,dtype=dtype); d=store.array('root0_array_0.npy',a,0)
    actual=m.load_saved_array(store.path,d)
    assert actual.shape==a.shape and actual.dtype==a.dtype and actual.tobytes()==a.tobytes()
    with pytest.raises(ValueError):store.array('root0_array_0.npy',a,0)


@pytest.mark.parametrize('fault',['shape','dtype','data_sha','file_sha','size','path','nan','trailer','symlink'])
def test_saved_array_mutations(store,fault):
    d=store.array('root0_array_0.npy',np.zeros((1,3),np.float32),0)
    if fault=='shape':d['shape']=[3]
    elif fault=='dtype':d['dtype']='<f8'
    elif fault=='data_sha':d['data_sha256']='0'*64
    elif fault=='file_sha':d['file_sha256']='0'*64
    elif fault=='size':d['file_bytes']+=1
    elif fault=='path':d['file']='../root0_array_0.npy'
    elif fault=='symlink':
        p=store.path/d['file']; saved=store.path/'original'; p.rename(saved); p.symlink_to(saved)
    else:
        p=store.path/d['file']; b=p.read_bytes()
        if fault=='trailer':b+=b'x'
        else:b=b[:-4]+np.array(np.nan,np.float32).tobytes()
        p.write_bytes(b); d.update(file_bytes=len(b),file_sha256=hashlib.sha256(b).hexdigest())
    with pytest.raises((ValueError,OSError)):m.load_saved_array(store.path,d)


def test_core_ram_full_paid_trace_disk_roundtrip_not_host(store):
    v=values(); r,a=m.core.track_fixture(v['z'],v['a'],v['target'],spec=m.spec(),kernels=kernels())
    assert r['completed'],r['errors']
    row=dict(root=0); m.persist_core(store,row,r,a)
    saved=m.read(store.path/row['trace']['file'],row['trace']['file_sha256'])
    arr={k:m.load_saved_array(store.path,d) for k,d in row['arrays'].items()}
    assert m.core.validate_saved(saved,arr)==m.core.validate_saved(r,a)
    assert saved['backend']=='ram_fixture_not_production'
    assert saved['predict_API_debited']==12 and saved['predictor_samples_debited']==54
    assert len(arr)==len(a) and not saved['real_meter_authenticated']


def test_partial_disk_preservation_and_no_promotion(store):
    v=values(); r,a=m.core.track_fixture(v['z'],v['a'],v['target'],spec=m.spec(),kernels=kernels(True))
    assert not r['completed'] and r['errors']
    row=dict(root=1); m.persist_core(store,row,r,a)
    assert not row['trace_completed'] and (store.path/row['trace']['file']).exists()
    with pytest.raises(ValueError):m.analyze_saved({},dict(schema=m.TAG+'.report',completed=False,errors=['partial']),store.path)


def test_file_census_and_chronological_journal(store):
    store.put('z.json',b'{}');store.put('a.json',b'{}')
    r=dict(files=store.files,write_journal=store.journal,file_bytes_before_report=store.used)
    store.put('report.json',m.encode(r),final=True)
    r=json.loads(m.encode(r)) # dict order sorted, ledger chronology not sorted.
    assert m.validate_files(r,store.path)['files']==2
    r['write_journal'][0]['returned']=False
    with pytest.raises(ValueError):m.validate_files(r,store.path)


@pytest.mark.parametrize('fault',['extra','missing','bytes','size','total','journal'])
def test_file_census_mutations(store,fault):
    store.put('a.json',b'{}'); r=dict(files=deepcopy(store.files),write_journal=deepcopy(store.journal),file_bytes_before_report=store.used)
    store.put('report.json',m.encode(r),final=True)
    if fault=='extra':(store.path/'extra').write_bytes(b'x')
    elif fault=='missing':(store.path/'a.json').unlink()
    elif fault=='bytes':(store.path/'a.json').write_bytes(b'[]')
    elif fault=='size':r['files']['a.json']['file_bytes']+=1
    elif fault=='total':r['file_bytes_before_report']+=1
    else:r['write_journal'].append(deepcopy(r['write_journal'][0]))
    with pytest.raises(ValueError):m.validate_files(r,store.path)


def test_hard_caps_no_write_no_refund_footer(store,monkeypatch):
    store.used=m.CAP-m.RESERVE
    with pytest.raises(ValueError):store.put('blocked',b'x')
    assert store.journal[-1]['returned'] is False and not (store.path/'blocked').exists()
    store.put('report.json',b'{}',final=True)
    store.used=0;store.root_used[0]=m.ROOT_CAP-m.ROOT_RESERVE
    with pytest.raises(ValueError):store.put('root_array',b'x',0)
    store.put('root_footer',b'{}',0,footer=True)
    monkeypatch.setattr(m.time,'monotonic',lambda:store.start+m.WALL+1)
    with pytest.raises(ValueError):store.guard()


def test_resource_integer_policy(store):
    g=m.GPUResourceGuard(store);g.observe(1,2);g.observe(3,4)
    assert g.snapshot()['max_allocated']==3 and g.snapshot()['dispatch_guard_calls']==2
    with pytest.raises(ValueError):g.observe(m.ALLOCATED+1,4)
    with pytest.raises(ValueError):g.observe(1,m.RESERVED+1)


def test_constructor_full_metadata_not_status_only():
    c=dict(own_sha256={'code':'x','test':'y'},tensor_copy_attempts=425,tensor_copy_returns=425,state_key_count=425,forward_calls=0,physics=0)
    manifest=dict(expected_construction=deepcopy(c),sources={'code':'x','test':'y'})
    m.validate_construction(c,manifest)
    c['state_key_count']=424
    with pytest.raises(ValueError):m.validate_construction(c,manifest)


def test_unauthorized_run_before_io_or_torch(tmp_path):
    with pytest.raises(ValueError,match='MAIN authorization'):
        m.run_cuda('/missing','0'*64,tmp_path/'no',main_authorized=False)
    assert not (tmp_path/'no').exists()


def test_cli_protocol_safe_subprocess():
    script='import sys; from experiments.spectral_reachability import '+m.TAG+' as m; m.main(["protocol"]); assert "torch" not in sys.modules'
    r=subprocess.run([sys.executable,'-B','-c',script],cwd=m.ROOT,text=True,capture_output=True,check=True)
    assert json.loads(r.stdout)==m.protocol()


def test_source_closure_existing_files_only():
    closure=m.closure()
    assert all(closure[str(p)]==h for p,h in m.PINS.items())
    assert str(m.ROOT/'tests'/('test_'+m.TAG+'.py')) in closure
    assert str(m.ROOT/'innerdocs'/(m.TAG+'.md')) in closure
