"""RAM arithmetic/orchestration/saved ABI only; no model/full strict execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_target_controller_cuda_20261029_v1 as m


def fixture():
    rows=[]
    for root in (0,1):
        for arm in m.ARMS:
            p=dict(z=np.full((1,3,192),root,np.float32),a=np.zeros((1,2,10),np.float32),target=np.full((1,192),.25,np.float32))
            rows.append(dict(root=root,arm=arm,condition='off',status='missing' if arm=='arbitrary' else 'assembled_not_dispatched',
                controller_payload=None if arm=='arbitrary' else {k:a.tolist() for k,a in p.items()},
                reason='original missing' if arm=='arbitrary' else None,target_sha256=m.core.digest(p['target'].tobytes())))
    return dict(schema=m.delivery.TAG+'.prepare',rows=rows,denominator=dict(roots=2,arms=3,total=6,assembled=4,missing=2,dispatched=0))


def kernels():
    def check(x):m.need(type(x) is np.ndarray and x.dtype==np.float32 and np.isfinite(x).all(),'RAM finite f32')
    return m.core.low.Kernels(lambda xs,dim:np.concatenate(xs,axis=dim),lambda xs,dim:np.stack(xs,axis=dim),check,
        lambda a:np.repeat(a.sum(-1,keepdims=True,dtype=np.float32),192,axis=-1),
        lambda z,a:z+a/np.float32(16),lambda x:x.copy(),'ram_fixture')


@pytest.fixture
def store(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    return m.Store(tmp_path/'results'/(m.TAG+'_fixture'))


def test_fixed_protocol_and_independent_caps():
    p=m.protocol();s=p['spec']
    assert [s[k] for k in ('horizon','samples','iterations','topk','var_scale','seed','segments')]==[3,8,2,2,1.,0,[1,2]]
    assert p['total_predict_API']==48 and p['total_predictor_samples']==216 and p['denominator']==6
    assert p['output_cap']==1024*1024**2 and p['footer_reserve']==128*1024**2 and p['per_root_file_cap']==256*1024**2
    assert p['wall_seconds']==1800 and p['FLOPs'] is None
    assert not any(p[k] for k in ('scientific_admission','raw_inverse','fair_baseline','time_matched','equal_generation_budget'))
    assert p['encoder']==p['high']==p['AD']==p['physics']==0


def test_real_delivery_metadata_only_and_source_pins():
    d=m.read(m.DELIVERY_PATH,m.DELIVERY_SHA);values=m.select_payloads(d)
    assert tuple(values)==m.ACTIVE and len(d['rows'])==6
    assert all(m.sha(p)==h for p,h in m.PINS.items())
    assert d['limitations']['prior_physical_time'] is None


@pytest.mark.parametrize('fault',['order','extra_payload','missing_payload','arbitrary_filled','arbitrary_started','denominator','target_sha','shape'])
def test_payload_selection_mutations(fault):
    d=fixture()
    if fault=='order':d['rows'][0]['root']=1
    elif fault=='extra_payload':d['rows'][0]['controller_payload']['witness']=[]
    elif fault=='missing_payload':d['rows'][0]['controller_payload'].pop('a')
    elif fault=='arbitrary_filled':d['rows'][2]['controller_payload']=d['rows'][0]['controller_payload']
    elif fault=='arbitrary_started':d['rows'][2]['status']='assembled_not_dispatched'
    elif fault=='denominator':d['denominator']['total']=4
    elif fault=='target_sha':d['rows'][0]['target_sha256']='0'*64
    else:d['rows'][0]['controller_payload']['target']=[[0.]]
    with pytest.raises(ValueError):m.select_payloads(d)


def test_all_four_inputs_sealed_before_invoke_and_unknown_failed_attempt(store):
    d=fixture();rows=m.input_rows(d);values=m.select_payloads(d)
    with pytest.raises(ValueError,match='all four input files'):m.execute_slots(store,rows,values,None,None,{})
    m.persist_inputs(store,rows,values)
    assert len(store.files)==12 and all(len(rows[i]['input_files'])==3 for i in m.ACTIVE)
    seen=[]
    def invoke(p):
        assert set(p)=={'z','a','target'} and len(store.files)==12
        seen.append(p);p['a'][:]=12
        raise RuntimeError('fixture interrupted call')
    with pytest.raises(RuntimeError):m.execute_slots(store,rows,values,invoke,lambda:{'frozen':'value'},{'frozen':'value'})
    assert len(seen)==1 and np.all(values[0]['a']==0)
    assert rows[0]['status']=='attempted' and rows[0]['debits']['known'] is False
    assert rows[1]['status']=='not_started' and rows[2]['status']=='missing'
    assert m.totals(rows)['actual_predict_API_debited'] is None
    assert m.totals(rows)['known_predict_API_debited']==0


def test_ram_return_persisted_but_never_promoted_to_host(store):
    d=fixture();rows=m.input_rows(d);values=m.select_payloads(d);m.persist_inputs(store,rows,values)
    def invoke(p):return m.core.track_fixture(p['z'],p['a'],p['target'],spec=m.tracker.spec(),kernels=kernels())
    with pytest.raises(ValueError,match='HOST only'):m.execute_slots(store,rows,values,invoke,lambda:{},{})
    row=rows[0];assert row['status']=='partial' and row['debits']==dict(known=True,predict_API=12,predictor_samples=54)
    cr=m.read(store.path/row['trace']['file'],row['trace']['file_sha256'])
    arr={k:m.load_array(store.path,d) for k,d in row['arrays'].items()}
    assert cr['backend']=='ram_fixture_not_production' and m.core.validate_saved(cr,arr)['all_mean_replays_verified']
    assert m.totals(rows)['actual_predict_API_debited']==12
    assert rows[1]['status']=='not_started' and rows[2]['status']=='missing'


@pytest.mark.parametrize('shape',[(),(1,),(1,3,192),(1,2,10)])
@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.int64])
def test_saved_arrays_exact_roundtrip(store,shape,dtype):
    x=np.zeros(shape,dtype);desc=store.array('slot0_array_0.npy',x,0)
    assert m.delivery.same(m.load_array(store.path,desc),x)


@pytest.mark.parametrize('fault',['dtype','shape','data','file','size','path','missing_slot','trailer','symlink','nan'])
def test_saved_array_mutations(store,fault):
    desc=store.array('slot0_array_0.npy',np.zeros(3,np.float32),0);p=store.path/desc['file']
    if fault=='dtype':desc['dtype']='<f8'
    elif fault=='shape':desc['shape']=[1,3]
    elif fault=='data':desc['data_sha256']='0'*64
    elif fault=='file':desc['file_sha256']='0'*64
    elif fault=='size':desc['file_bytes']+=1
    elif fault=='path':desc['file']='../slot0_array_0.npy'
    elif fault=='missing_slot':desc['file']='slot2_array_0.npy'
    elif fault=='symlink':q=store.path/'original';p.rename(q);p.symlink_to(q)
    else:
        b=p.read_bytes();b=b+b'x' if fault=='trailer' else b[:-4]+np.array(np.nan,np.float32).tobytes()
        p.write_bytes(b);desc.update(file_bytes=len(b),file_sha256=hashlib.sha256(b).hexdigest())
    with pytest.raises((ValueError,OSError)):m.load_array(store.path,desc)


def test_job_root_caps_and_footer_retention(store,monkeypatch):
    store.root_used[0]=m.ROOT_CAP-m.ROOT_RESERVE
    with pytest.raises(ValueError):store.put('slot0_array',b'x',0)
    assert store.journal[-1]['returned'] is False
    store.put('slot0_footer',b'{}',0,footer=True)
    store.put('slot3_data',b'x',1)
    store.used=m.CAP-m.RESERVE
    with pytest.raises(ValueError):store.put('blocked',b'x')
    store.put('report.json',b'{}',final=True)
    monkeypatch.setattr(m.time,'monotonic',lambda:store.start+m.WALL+1)
    with pytest.raises(ValueError):store.guard()


def file_report(store):
    store.put('slot0.json',b'{}',0);store.put('slot3.json',b'{}',1)
    r=dict(files=store.files,write_journal=store.journal,file_bytes_before_report=store.used,root_file_bytes=store.root_used)
    store.put('report.json',m.encode(r),final=True)
    return json.loads(m.encode(r))


def test_file_census_chronology_and_root_join(store):
    r=file_report(store);m.validate_files(r,store.path)
    r['root_file_bytes']['0']+=1
    with pytest.raises(ValueError):m.validate_files(r,store.path)


@pytest.mark.parametrize('fault',['extra','missing','tamper','journal','total'])
def test_file_census_mutations(store,fault):
    r=file_report(store)
    if fault=='extra':(store.path/'extra').write_bytes(b'x')
    elif fault=='missing':(store.path/'slot0.json').unlink()
    elif fault=='tamper':(store.path/'slot0.json').write_bytes(b'[]')
    elif fault=='journal':r['write_journal'].append(deepcopy(r['write_journal'][0]))
    else:r['file_bytes_before_report']+=1
    with pytest.raises(ValueError):m.validate_files(r,store.path)


def test_partial_rejected_before_cuda_claim():
    with pytest.raises(ValueError,match='complete four HOST'):
        m.analyze_saved({},dict(schema=m.TAG+'.report',completed=False,errors=['partial']),'/missing')


def test_prepare_verify_fixture_cli_and_exclusive(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    artifact=dict(schema=m.TAG+'.explicit_RAM_fixture',delivery=dict(path='/fixture',sha256='a'*64),rows=fixture()['rows'])
    monkeypatch.setattr(m,'build',lambda path,h:deepcopy(artifact))
    p=tmp_path/'results'/(m.TAG+'_prepare_fixture.json');o=m.prepare(p)
    assert m.verify(p,o['sha256'])==artifact
    m.main(['verify','--manifest',str(p),'--manifest-sha256',o['sha256']]);assert json.loads(capsys.readouterr().out)['verified']
    with pytest.raises(FileExistsError):m.prepare(p)
    artifact['rows'][0]['arm']='other'
    with pytest.raises(ValueError):m.verify(p,o['sha256'])


def test_build_fixed_authority_no_future_or_fake_path():
    with pytest.raises(ValueError,match='fixed actual delivery'):m.build(m.DELIVERY_PATH,'0'*64)
    with pytest.raises(ValueError,match='fixed actual delivery'):m.build('/fixture',m.DELIVERY_SHA)


def test_build_fixture_joins_old425_without_old_verify_or_consume(monkeypatch):
    d=fixture();d.update(authorities=dict(tracker_prepare=dict(path='/fixture/old.json',sha256='b'*64)),sources={'delivery_source':'a'*64})
    old=dict(sources={'original_model_npy':'a'*64},export={'fixture':True},expected_state={'original425':'literal'},
             expected_construction={'fixture_constructor':True})
    def read(path,h):return deepcopy(d if str(path)==str(m.DELIVERY_PATH) else old)
    monkeypatch.setattr(m,'read',read);monkeypatch.setattr(m.delivery,'build',lambda:deepcopy(d))
    monkeypatch.setattr(m,'closure',lambda:{'owned_source':'a'*64});monkeypatch.setattr(m,'sha',lambda p:'a'*64)
    # Artifact itself uses fixed actual SHA; keep its byte digest distinguishable.
    monkeypatch.setattr(m,'sha',lambda p:m.DELIVERY_SHA if str(p)==str(m.DELIVERY_PATH) else 'a'*64)
    def forbidden(*args,**kwargs):raise AssertionError('old full strict must not be called')
    monkeypatch.setattr(m.tracker,'verify',forbidden);monkeypatch.setattr(m.tracker,'consume',forbidden)
    out=m.build(m.DELIVERY_PATH,m.DELIVERY_SHA)
    assert out['expected_state']==old['expected_state'] and out['sources']['original_model_npy']=='a'*64
    assert out['delivery_artifact']==d and out['protocol']['total_predict_API']==48
    d['sources']['original_model_npy']='b'*64
    with pytest.raises(ValueError,match='consistent source closures'):m.build(m.DELIVERY_PATH,m.DELIVERY_SHA)


def test_modelguard_before_first_invocation(store):
    d=fixture();r=m.input_rows(d);v=m.select_payloads(d);m.persist_inputs(store,r,v)
    with pytest.raises(ValueError,match='original425 before'):m.execute_slots(store,r,v,None,lambda:{'changed':True},{})
    assert all(row['debits']['predict_API']==0 for row in r)


def test_unauthorized_run_no_io(tmp_path):
    with pytest.raises(ValueError,match='MAIN authorization'):m.run_cuda('/missing','0'*64,tmp_path/'no')
    assert not (tmp_path/'no').exists()


def test_cold_cli_no_torch_and_no_cpu_fallback():
    script='import sys; from experiments.spectral_reachability import '+m.TAG+' as m; m.main(["protocol"]); assert "torch" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-c',script],cwd=m.ROOT,text=True,capture_output=True,check=True)
    assert json.loads(p.stdout)==m.protocol()


def test_new_closure_and_no_old_mutation():
    closure=m.closure()
    assert all(closure[str(p)]==h for p,h in m.PINS.items())
    assert str(m.ROOT/'innerdocs'/(m.TAG+'.md')) in closure
