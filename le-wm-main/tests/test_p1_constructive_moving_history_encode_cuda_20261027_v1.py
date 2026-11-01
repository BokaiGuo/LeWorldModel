"""All fixtures live in RAM. No model, simulator, tmpdir or actual inputs."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace as NS
import gzip
import json
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as m


def packet():
    return dict(rgb_history=np.stack([np.full((224,224,3),i,np.uint8) for i in range(3)]),
        raw_action_history=m.fixed_raw(),goal_rgb=np.full((224,224,3),3,np.uint8))


def archive(p):
    b=BytesIO();np.savez_compressed(b,**p);return b.getvalue()


def manifest():
    return dict(schema=m.NAME+'.approved_blind_manifest',approved_by='MAIN',purpose='calibration',with_target=False,
        history_steps=[0,5,10],raw_action_steps=10,slots=[dict(order=s,source_slot=s,
        blind_npz=f'/RAM/s{s}_blind_inputs.npz',blind_sha256=chr(97+s)*64) for s in range(2)])


def protocol():
    p=dict(slots=deepcopy(manifest()['slots']),normalization=m.normalization(),model_bindings=m.MODELS,
        arguments=dict(approved_manifest='/RAM/manifest.json',manifest_sha='c'*64),contract=m.contract(),
        expected_model_hash=m.low.MODEL_TENSOR_SHA,policy=str(m.POLICY))
    a=packet()
    for s in p['slots']:
        s.update(pixel_sha256=[m.digest(x.tobytes()) for x in [*a['rgb_history'],a['goal_rgb']]],
            raw_action_sha256=m.digest(a['raw_action_history'].tobytes()))
    return p


def exported():
    p=protocol();return m.task_export(p,'d'*64,p['slots'][0],packet(),[np.full(192,i,np.float32) for i in range(4)])


def reseal(e):
    e['task']['task_contract_sha256']=m.core.digest(e['task_contract'])
    # Deliberately do not use task_id() validation to hide structural attacks.
    e['task_id']=m.core.digest(e['task'])


def test_blind_decode_and_true_time_major_packets():
    a=m.decode_blind(archive(packet()));assert np.array_equal(a['raw_action_history'],m.fixed_raw())
    e=exported();i=m.task_interfaces(e);assert set(i)=={'gn_task','d2p_packet','q','task_id'}
    t=i['gn_task'];raw=np.asarray(t['root_raw_action_history'],np.float32)
    expected=(raw.reshape(2,10)-np.tile(np.asarray(m.normalization()['mean'],np.float32),5))/np.tile(np.asarray(m.normalization()['std'],np.float32),5)
    assert np.array_equal(t['root_context']['action_history'],expected)
    assert np.array_equal(raw[:5],np.tile(np.asarray([.1,0],np.float32),(5,1)))
    assert np.array_equal(raw[5:],np.tile(np.asarray([0,.1],np.float32),(5,1)))
    assert not np.array_equal(expected[0],expected[1]) and t['horizon']==2
    i['gn_task']['goal'][0]=99;assert e['task']['goal'][0]==3


@pytest.mark.parametrize('bad',['state','certificate','sourcepack','witness','candidate_actions'])
def test_privileged_npz_members_rejected(bad):
    p=packet();p[bad]=np.zeros(2)
    with pytest.raises(ValueError,match='blind-only'):m.decode_blind(archive(p))


@pytest.mark.parametrize('bad',['zero','reversed','float64','packetshape','NaN','rawround','rgbfloat','rgbcount','goalshape'])
def test_packet_must_be_exact_actual_contract(bad):
    p=packet()
    if bad=='zero':p['raw_action_history'][:]=0
    elif bad=='reversed':p['raw_action_history']=p['raw_action_history'][::-1].copy()
    elif bad=='float64':p['raw_action_history']=p['raw_action_history'].astype(float)
    elif bad=='packetshape':p['raw_action_history']=p['raw_action_history'].reshape(2,10)
    elif bad=='NaN':p['raw_action_history'][0,0]=np.nan
    elif bad=='rawround':p['raw_action_history'][0,0]=np.nextafter(np.float32(.1),np.float32(1))
    elif bad=='rgbfloat':p['rgb_history']=p['rgb_history'].astype(float)
    elif bad=='rgbcount':p['rgb_history']=p['rgb_history'][:2]
    elif bad=='goalshape':p['goal_rgb']=p['goal_rgb'][:1]
    with pytest.raises(ValueError):m.decode_blind(archive(p))


def test_manifest_exact_two_approved_absolute_slots():
    assert m.manifest_slots(manifest(),Path('/RAM'))==manifest()['slots']


@pytest.mark.parametrize('bad',['extra','oldschema','target','boolslot','boolorder','reorder','one','duplicatepath','duplicatehash','relative','statepath','rawcount'])
def test_manifest_rejects_changed_contract(bad):
    p=manifest()
    if bad=='extra':p['certificate']='forbidden'
    elif bad=='oldschema':p['schema']=m.low.NAME+'.approved_blind_whitelist'
    elif bad=='target':p['with_target']=True
    elif bad=='boolslot':p['slots'][0]['source_slot']=False
    elif bad=='boolorder':p['slots'][0]['order']=False
    elif bad=='reorder':p['slots'].reverse()
    elif bad=='one':p['slots'].pop()
    elif bad=='duplicatepath':p['slots'][1]['blind_npz']=p['slots'][0]['blind_npz']
    elif bad=='duplicatehash':p['slots'][1]['blind_sha256']=p['slots'][0]['blind_sha256']
    elif bad=='relative':p['slots'][0]['blind_npz']='s0_blind_inputs.npz'
    elif bad=='statepath':p['slots'][0]['state_path']='forbidden'
    elif bad=='rawcount':p['raw_action_steps']=True
    with pytest.raises(ValueError):m.manifest_slots(p,Path('/RAM'))


@pytest.mark.parametrize('bad',['extra','oldschema','slot','blind','target','root','outcome','norm','model','goalsha','Q','cap','rawzero','rawrounded','tokenszero','tokensswap','latent64','tc_extra','tc_slot','tc_rawhash'])
def test_export_rejects_tamper_even_with_recomputed_hashes(bad):
    e=exported();t=e['task'];c=e['task_contract']
    if bad=='extra':e['witness']={}
    elif bad=='oldschema':e['schema']=m.low.NAME+'.task_export'
    elif bad=='slot':e['source_slot']=False
    elif bad=='blind':e['blind_sha256']='e'*64
    elif bad=='target':e['with_target']=True
    elif bad=='root':e['native_root_sha256']='e'*64
    elif bad=='outcome':e['P1_admission']=True
    elif bad=='norm':c['normalization']['mean'][0]=0;t['action_mean'][0]=0;t['root_context']['action_history']=m.core.normalized_tokens(t,m.fixed_raw()).tolist()
    elif bad=='model':c['model_bindings']=dict(c['model_bindings'],extra='e'*64)
    elif bad=='goalsha':t['goal_source_sha256']='e'*64
    elif bad=='Q':t['projection'][0][0]=0
    elif bad=='cap':t['packet_logit_cap']=1
    elif bad=='rawzero':t['root_raw_action_history']=np.zeros((10,2)).tolist();t['root_context']['action_history']=m.core.normalized_tokens(t,np.zeros((10,2))).tolist()
    elif bad=='rawrounded':t['root_raw_action_history'][0][0]=.1
    elif bad=='tokenszero':t['root_context']['action_history']=np.zeros((2,10)).tolist()
    elif bad=='tokensswap':t['root_context']['action_history'].reverse()
    elif bad=='latent64':t['goal'][0]=1.0000000000000002
    elif bad=='tc_extra':c['certificate']='bad'
    elif bad=='tc_slot':c['source_slot']=False
    elif bad=='tc_rawhash':c['raw_action_sha256']='e'*64
    reseal(e)
    with pytest.raises(ValueError):m.task_interfaces(e)


def test_load_task_external_hash_and_full_envelope(monkeypatch):
    e=exported();monkeypatch.setattr(m,'sha',lambda p:'a'*64);monkeypatch.setattr(m,'read',lambda p:deepcopy(e))
    assert m.load_task('/RAM/task.gz','a'*64)['task_id']==e['task_id']
    with pytest.raises(ValueError):m.load_task('/RAM/task.gz','b'*64)
    e['extra']='forbidden'
    with pytest.raises(ValueError):m.load_task('/RAM/task.gz','a'*64)


class RAMStore:
    def __init__(self):self.output=Path('/RAM/out');self.files={};self.events=[]
    def event(self,e):self.events.append(deepcopy(e))
    def latent(self,index,z):
        s=BytesIO();np.save(s,z,allow_pickle=False);name=f'image_{index:02d}.native192.npy';self.files[name]=s.getvalue()
        assert np.array_equal(np.load(BytesIO(self.files[name]),allow_pickle=False),z)
        return dict(path=str(self.output/name),sha256=m.digest(self.files[name]),dtype='float32',shape=[192])
    def json(self,name,e):self.files[name]=gzip.compress(m.canonical(e),mtime=0)


def mock_encoding(monkeypatch,fail=None):
    p=protocol();s=RAMStore();l=m.Ledger(2,s.event);calls=[];rows=[dict(status='blocked_not_attempted',images=[]) for _ in p['slots']];exports=[]
    monkeypatch.setattr(m,'load_blind',lambda *a:packet())
    monkeypatch.setattr(m,'sha',lambda path:m.digest(s.files[Path(path).name]))
    monkeypatch.setattr(m,'read',lambda path:json.loads(gzip.decompress(s.files[Path(path).name])))
    def encode(rgb):
        assert rgb.dtype==np.uint8 and rgb.shape==(224,224,3);calls.append(rgb.copy())
        for k,shape in [('encoder',(1,3,224,224)),('projector',(1,192))]:
            def returned():
                if fail==(len(calls),k):raise RuntimeError('mock module failure')
            l.dispatch(k,NS(device=NS(type='cuda'),shape=shape),returned)
        return np.random.default_rng(len(calls)).normal(size=192).astype(np.float32)
    return p,s,l,calls,rows,exports,encode


def test_all_eight_no_dedup_two_usable_tasks_and_small_receipt(monkeypatch):
    p,s,l,calls,rows,exports,encode=mock_encoding(monkeypatch)
    m.encode_all(p,'d'*64,encode,l,s,rows,exports)
    assert len(calls)==8 and np.array_equal(calls[0],calls[4]) and len(exports)==2
    assert l.counts()=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')}
    assert all(r['status']=='task_sealed' for r in rows) and len(s.files)==10
    for row in rows:assert set(m.load_task(row['task_path'],row['task_file_sha256']))=={'gn_task','d2p_packet','q','task_id'}
    assert sum(map(len,s.files.values()))+len(m.canonical(s.events))<128*1024
    for name,raw in s.files.items():
        if name.endswith('.npy'):
            z=np.load(BytesIO(raw),allow_pickle=False);assert z.shape==(192,) and z.dtype==np.float32


@pytest.mark.parametrize('fail',[(1,'encoder'),(1,'projector'),(5,'encoder'),(8,'projector')])
def test_failed_module_retains_census_and_prior_task_without_retry(monkeypatch,fail):
    p,s,l,calls,rows,exports,encode=mock_encoding(monkeypatch,fail)
    with pytest.raises(RuntimeError):m.encode_all(p,'d'*64,encode,l,s,rows,exports)
    assert len(calls)==fail[0] and len(exports)==(1 if fail[0]>4 else 0)
    assert l.entries[-1]['completed'] is False and any(e['kind']=='module_error' for e in s.events)
    if fail[0]<=4:assert rows[1]['status']=='blocked_not_attempted'
    else:assert rows[0]['status']=='task_sealed'
    assert len([n for n in s.files if n.endswith('.npy')])==fail[0]-1


def test_build_exact_manifest_models_and_input_bindings_without_privileged_read(monkeypatch):
    opened=[];manifest_sha='c'*64
    monkeypatch.setattr(m,'sha',lambda p:m.MODELS.get(str(p),manifest_sha))
    monkeypatch.setattr(m,'read',lambda path:(opened.append(str(path)) or manifest()))
    monkeypatch.setattr(m,'load_blind',lambda *a:packet())
    monkeypatch.setattr(m.low,'native_metadata',m.normalization)
    monkeypatch.setattr(m,'source_files',lambda:[Path('/RAM/own.py')])
    # Explicit blind hashes are distinct; hash lookup mirrors those two inputs.
    orig=m.sha;monkeypatch.setattr(m,'sha',lambda p:next((s['blind_sha256'] for s in manifest()['slots'] if s['blind_npz']==str(p)),orig(p)))
    p=m.build('/RAM/manifest.json',manifest_sha)
    assert opened==['/RAM/manifest.json'] and len(p['slots'])==2 and p['contract']['encoder_cap']==8
    assert all(s['blind_npz'] in p['bindings_sha256'] for s in p['slots'])
    assert p['normalization']==m.normalization() and p['arguments']['manifest_sha']==manifest_sha


def test_verify_requires_full_rebuild(monkeypatch):
    p=dict(arguments=dict(approved_manifest='/RAM/m',manifest_sha='c'*64))
    monkeypatch.setattr(m,'sha',lambda *a:'a'*64);monkeypatch.setattr(m,'read',lambda *a:deepcopy(p));monkeypatch.setattr(m,'build',lambda **a:deepcopy(p))
    assert m.verify('/RAM/p','a'*64)==p
    with pytest.raises(ValueError):m.verify('/RAM/p','b'*64)
    monkeypatch.setattr(m,'build',lambda **a:dict(p,changed=True))
    with pytest.raises(ValueError):m.verify('/RAM/p','a'*64)


def test_no_torch_load_failure_is_durable_partial_all_two_slots(monkeypatch):
    p=protocol();events=[];saved={}
    class EmptyOutput:
        def iterdir(self):return iter(())
    class S:
        output=EmptyOutput()
        def __init__(self,*a):pass
        def event(self,e):events.append(e)
        def json(self,n,v,**kw):saved[n]=deepcopy(v)
    monkeypatch.setattr(m,'verify',lambda *a:deepcopy(p));monkeypatch.setattr(m,'Store',S)
    monkeypatch.setitem(sys.modules,'torch',None)
    with pytest.raises(ValueError,match='partial moving'):m.run('/RAM/p','a'*64,'/RAM/out')
    r=saved['report.json'];assert r['model_unchanged'] is None and len(r['rows'])==2 and r['errors']
    assert all(row['status']=='blocked_not_attempted' for row in r['rows'])
    assert r['resources']=={k:dict(attempted=0,completed=0) for k in ('encoder','projector')}
