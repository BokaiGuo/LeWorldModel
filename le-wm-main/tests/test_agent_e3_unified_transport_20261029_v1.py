import hashlib
import io
import numpy as np
import pytest
from experiments.spectral_reachability import agent_e3_unified_transport_20261029_v1 as m


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def fixture(tmp_path):
    path=tmp_path/'job';s=m.Store(path)
    e=s.event('physical','intent','public_step',0,{})
    small=s.array('numerical','latent',np.zeros((1,3,192),np.float32))
    e2=s.event('numerical','intent','predict',1,{'input':small})
    hidden=s.array('encoding','encoder_hidden',np.zeros((1,257,192),np.float32))
    image=s.array('encoding','encoder_input',np.zeros((1,3,224,224),np.float32))
    s.event('encoding','intent','encoder',2,{'input':image})
    s.event('encoding','return','encoder',2,{'output':hidden})
    s.event('numerical','return','predict',1,{'output':small})
    s.event('physical','return','public_step',0,{})
    return path,s


def test_real_shape_disk_interleaving(tmp_path):
    path,s=fixture(tmp_path);r=m.consume(path,sha(path/'events.jsonl'))
    assert r['transport_complete'] and not r['full_native_strict']
    assert r['bytes']['encoding_arrays']>4*(257*192+3*224*224)
    assert [e['origin'] for e in r['events']]==['physical','numerical','encoding','encoding','numerical','physical']


def test_partial_failure_not_complete(tmp_path):
    path=tmp_path/'job';s=m.Store(path);s.event('physical','intent','public_step',0,{})
    s.event('physical','failure','public_step',0,{'exception':'failed'})
    r=m.consume(path,sha(path/'events.jsonl'));assert not r['transport_complete']
    assert r['counts']['physical:public_step']==dict(attempted=1,returned=0,failed=1)


def test_reject_shapes_and_unknown_inventory(tmp_path):
    path,s=fixture(tmp_path)
    with pytest.raises(ValueError):s.array('encoding','encoder_hidden',np.zeros((1,197,192),np.float32))
    (path/'unexpected').write_bytes(b'x')
    with pytest.raises(ValueError):m.consume(path,sha(path/'events.jsonl'))


def test_reject_tampered_array_and_journal(tmp_path):
    path,s=fixture(tmp_path);p=path/'a000001.npy';p.write_bytes(p.read_bytes()+b'x')
    with pytest.raises(ValueError):m.consume(path,sha(path/'events.jsonl'))
    with pytest.raises(ValueError):m.consume(path,'0'*64)


def test_tamper_event_order_even_with_new_journal_sha(tmp_path):
    path,s=fixture(tmp_path);p=path/'events.jsonl';lines=p.read_bytes().splitlines(keepends=True)
    lines[0],lines[1]=lines[1],lines[0];p.write_bytes(b''.join(lines))
    with pytest.raises(ValueError):m.consume(path,sha(p))
