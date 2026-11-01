"""Prepare, caps and failure-path mocks. No neural/physics imports or calls."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import p4_terminal20_cuda_encoding_20261027_v1 as m


def tensor(kind='encoder',device='cuda'):
    return SimpleNamespace(device=SimpleNamespace(type=device),shape=(1,3,224,224) if kind=='encoder' else (1,192))


def test_contract():
    c=m.contract()
    assert c['historical_runs']==1 and c['encoder_cap']==c['existing_projector_cap']==2
    assert c['predict_cap']==c['AD_cap']==c['physics_cap']==0
    assert not c['P1_admission'] and not c['P4_admission'] and not c['cpu_neural_fallback']


def test_complete_dispatch_order():
    journal=[]; ledger=m.Ledger(journal.append)
    for arm in m.ARMS:
        ledger.arm=arm
        for kind in ('encoder','existing_projector'):
            assert ledger.dispatch(kind,tensor(kind),lambda:17,(),{})==17
    assert len(ledger.entries)==4 and all(e['completed'] for e in ledger.entries)
    assert len(journal)==8
    with pytest.raises(ValueError): ledger.dispatch('encoder',tensor(),lambda:0,(),{})
    assert len(ledger.entries)==4 and len(ledger.blocked)==1


@pytest.mark.parametrize('device',['cpu','meta',None])
def test_cpu_rejected_before_callback(device):
    ledger=m.Ledger(lambda e:None); ledger.arm='imagined'; called=[]
    with pytest.raises(ValueError): ledger.dispatch('encoder',tensor(device=device),lambda:called.append(1),(),{})
    assert not called and not ledger.entries and len(ledger.blocked)==1


@pytest.mark.parametrize('shape',[(2,3,224,224),(1,3,256,256),(1,1,3,224,224),()])
def test_shape_cap(shape):
    ledger=m.Ledger(lambda e:None); ledger.arm='imagined'; x=tensor(); x.shape=shape
    with pytest.raises(ValueError): ledger.dispatch('encoder',x,lambda:None,(),{})
    assert not ledger.entries


def test_projector_cannot_skip_encoder():
    ledger=m.Ledger(lambda e:None); ledger.arm='imagined'
    with pytest.raises(ValueError,match='sequence'): ledger.dispatch('existing_projector',tensor('existing_projector'),lambda:0,(),{})


def test_failed_entry_nonrefundable():
    ledger=m.Ledger(lambda e:None); ledger.arm='imagined'
    def fail(): raise RuntimeError('first error')
    with pytest.raises(RuntimeError): ledger.dispatch('encoder',tensor(),fail,(),{})
    assert len(ledger.entries)==1 and not ledger.entries[0]['completed']
    assert ledger.entries[0]['error']['message']=='first error'
    with pytest.raises(ValueError): ledger.dispatch('encoder',tensor(),lambda:0,(),{})


def test_async_error_not_completed():
    def synchronize(): raise RuntimeError('CUDA asynchronous error mock')
    ledger=m.Ledger(lambda e:None,synchronize); ledger.arm='imagined'
    with pytest.raises(RuntimeError): ledger.dispatch('encoder',tensor(),lambda:1,(),{})
    assert not ledger.entries[0]['completed']


@pytest.mark.parametrize('kind',['predict','predictor','action_encoder','pred_proj','AD.grad','AD.jacobian'])
def test_forbidden_accounting(kind):
    journal=[]; ledger=m.Ledger(journal.append)
    with pytest.raises(RuntimeError): ledger.forbid(kind)()
    assert ledger.blocked[0]['kind']==kind and journal[0]['event']=='blocked'


def protocol():
    return dict(endpoints=[dict(arm=arm,root_sha256='mock-root',primitive_step=20) for arm in m.ARMS],
                goal=np.zeros(192).tolist(),q=np.eye(192).tolist())


def test_two_latents_saved(tmp_path):
    p=protocol(); ledger=m.Ledger(lambda e:None)
    def save(arm,z):
        path=tmp_path/(arm+'.npy'); np.save(path,z,allow_pickle=False); return path,m.sha(path)
    rows,errors=m.encode_endpoints(p,lambda e:np.ones(192,np.float32),save,ledger,lambda e:None)
    assert not errors and len(rows)==2 and all(r['status']=='encoded_and_saved' for r in rows)
    assert all(r['original_Q_cost']==1. for r in rows)
    assert np.load(rows[0]['latent_path'],allow_pickle=False).dtype==np.float32


@pytest.mark.parametrize('output',[np.zeros(191,np.float32),np.zeros(192,np.float64),np.full(192,np.nan,np.float32)])
def test_invalid_native_stops_next_arm(output):
    called=[]; ledger=m.Ledger(lambda e:None)
    def encode(e): called.append(e['arm']); return output
    rows,errors=m.encode_endpoints(protocol(),encode,lambda *a:None,ledger,lambda e:None)
    assert called==['imagined'] and len(errors)==1 and rows[1]['status']=='blocked_not_attempted'


def test_save_failure_preserves_first_error():
    ledger=m.Ledger(lambda e:None); calls=[]
    def save(*args): raise OSError('mock ENOSPC')
    rows,errors=m.encode_endpoints(protocol(),lambda e:np.zeros(192,np.float32),save,ledger,lambda e:calls.append(e))
    assert errors[0]['message']=='mock ENOSPC' and rows[0]['status']=='failed'
    assert rows[1]['status']=='blocked_not_attempted' and ledger.arm is None


def test_journal_error_keeps_partial_rows():
    rows=[]; errors=[]; ledger=m.Ledger(lambda e:None)
    def journal(e): raise OSError('journal unavailable')
    with pytest.raises(OSError): m.encode_endpoints(protocol(),lambda e:np.zeros(192,np.float32),
        lambda *a:('saved.npy','saved-sha'),ledger,journal,rows,errors)
    assert len(rows)==1 and rows[0]['latent_path']=='saved.npy'


def test_png_sha(tmp_path):
    path=tmp_path/'rgb.png'; Image.fromarray(np.zeros((224,224,3),np.uint8)).save(path)
    rgb,pixel=m.pixels(path)
    assert m.pixels(path,m.sha(path),pixel)[1]==pixel and rgb.shape==(224,224,3)
    with pytest.raises(ValueError): m.pixels(path,'bad',pixel)
    with pytest.raises(ValueError): m.pixels(path,m.sha(path),'bad')


def test_wrong_png_dimensions(tmp_path):
    path=tmp_path/'rgb.png'; Image.fromarray(np.zeros((10,10,3),np.uint8)).save(path)
    with pytest.raises(ValueError): m.pixels(path)


def test_patch_restores_on_failure():
    obj=SimpleNamespace(x=3)
    with pytest.raises(RuntimeError):
        with m.ExitStack() as stack:
            m.patch(stack,obj,'x',5); assert obj.x==5; raise RuntimeError()
    assert obj.x==3


def test_prepare_seal_and_verify(tmp_path,monkeypatch):
    source=tmp_path/'source.py'; source.write_text('mock_source = True\n')
    p=dict(schema=m.TAG+'.protocol',contract=m.contract(),bindings={},model_bindings={},source_run='mock')
    monkeypatch.setattr(m,'collect',lambda *args:deepcopy(p))
    monkeypatch.setattr(m,'source_files',lambda:[source])
    output=tmp_path/'prepared'
    receipt=m.prepare(output)
    assert receipt['status']=='prepared_not_executed' and receipt['encoder_calls']==0
    assert m.verify(output,m.sha(output/'prepare.json'))['contract']==m.contract()
    with pytest.raises(ValueError): m.prepare(output)
    with pytest.raises(ValueError): m.verify(output,'bad-sha')
    source.write_text('changed = True\n')
    with pytest.raises(ValueError): m.verify(output,m.sha(output/'prepare.json'))


def test_no_heavy_prepare_imports():
    tree=ast.parse(Path(m.__file__).read_text())
    names=[v.name for n in tree.body if isinstance(n,ast.Import) for v in n.names]
    names += [n.module for n in tree.body if isinstance(n,ast.ImportFrom)]
    assert not any(x and any(bad in x for bad in ('torch','gym','pymunk','runtime','checkpoint_compat')) for x in names)


def test_real_saved_collect_only():
    p=m.collect()
    assert p['contract']['historical_runs']==1 and len(p['endpoints'])==2
    assert {e['arm'] for e in p['endpoints']}==set(m.ARMS)
    assert all(e['primitive_step']==20 and len(e['pixel_sha256'])==64 for e in p['endpoints'])
    assert 'q' not in p and 'goal' not in p  # Bind original arrays; no copied big protocol.


@pytest.mark.parametrize('file,key,value',[
    ('report.json','status','partial_or_integrity_fail'),
    ('report.json','model_unchanged',False),
    ('report.json','protocol_sha256','wrong'),
    ('report.json','journal_sha256','wrong'),
    ('prepare_receipt.json','protocol_sha256','wrong'),
])
def test_original_metadata_tamper_rejected(monkeypatch,file,key,value):
    original=m.read
    def read(path):
        data=original(path)
        if Path(path)==m.ORIGINAL/file: data[key]=value
        return data
    monkeypatch.setattr(m,'read',read)
    with pytest.raises(ValueError): m.collect()


def test_wrong_branch_root_rejected(monkeypatch):
    original=m.read
    def read(path):
        data=original(path)
        if Path(path)==m.ORIGINAL/'report.json': data['rows'][1]['root_sha256']='wrong-root'
        return data
    monkeypatch.setattr(m,'read',read)
    with pytest.raises(ValueError,match='one-root'): m.collect()


def test_prepare_low_disk_before_collect(tmp_path,monkeypatch):
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:SimpleNamespace(free=0))
    def collect(): raise AssertionError('must not collect')
    monkeypatch.setattr(m,'collect',collect)
    with pytest.raises(ValueError,match='headroom'): m.prepare(tmp_path/'prepared')


def test_journal_attempt_failure_never_dispatches():
    def journal(e): raise OSError('attempt journal failed')
    called=[]; ledger=m.Ledger(journal); ledger.arm='imagined'
    with pytest.raises(OSError): ledger.dispatch('encoder',tensor(),lambda:called.append(1),(),{})
    assert not called and len(ledger.entries)==1 and not ledger.entries[0]['completed']
