"""Non-neural numeric/header/provenance audit contracts."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from experiments.spectral_reachability import p5_jepa_fresh_fidelity_audit_20261027_v1 as a


def stats():return dict(ACTION_MEAN=np.array([-.0087,.0068],np.float32),ACTION_STD=np.array([.2019,.2002],np.float32),PROPRIO_MEAN=np.zeros(4,np.float32),PROPRIO_STD=np.ones(4,np.float32))


def test_normalize_exact_native_dataset_projection():
    u=np.arange(30,dtype=np.float64).reshape(15,2)/1000
    prop=np.ones((4,4),np.float32);s=stats()
    pixels,packets,norm,controls=a.normalize(u,prop,s)
    np.testing.assert_array_equal(pixels,(100*u).astype(np.float32))
    np.testing.assert_array_equal(packets,((pixels/np.float32(100)-s['ACTION_MEAN'])/s['ACTION_STD']).reshape(3,10))
    assert packets.dtype==np.float32
    np.testing.assert_array_equal(norm,prop)
    for bank in controls:np.testing.assert_array_equal(bank[:2],packets[:2])
    np.testing.assert_array_equal(controls[3,-1].reshape(5,2),packets[-1].reshape(5,2)[::-1])


def test_mean_control_not_raw_zero():
    s=stats();_,_,_,c=a.normalize(np.ones((15,2),np.float64),np.zeros((4,4),np.float32),s)
    np.testing.assert_array_equal(c[1,-1].reshape(5,2)*s['ACTION_STD']+s['ACTION_MEAN'],np.tile(s['ACTION_MEAN'],(5,1)))


def test_target_is_fourth_frame_and_all_arms_retained():
    feat=np.zeros((4,256,384),np.float32);feat[3]=2
    successors={name:np.full((1,256,384),i,np.float32) for i,name in enumerate(a.ARMS)}
    costs={name:float((i-2)**2) for i,name in enumerate(a.ARMS)}
    result=a.recompute(feat,successors,costs)
    assert result['all_receipts_consistent']
    assert result['arms']['negative_normalized']['mse']==0
    assert set(result['arms'])==set(a.ARMS)
    assert result['arms']['reverse_steps']['ordering']=='actual_higher'


def test_receipt_tolerance_does_not_erase_exact_ordering():
    feat=np.zeros((4,256,384),np.float32)
    successors={k:np.ones((1,256,384),np.float32) for k in a.ARMS}
    successors['actual'][:]=np.nextafter(np.float32(1),np.float32(2))
    costs={k:float(np.mean(x.astype(np.float64)**2)) for k,x in successors.items()}
    result=a.recompute(feat,successors,costs)
    assert result['arms']['reverse_steps']['ordering']=='actual_higher'
    costs['actual']+=.001
    assert not a.recompute(feat,successors,costs)['all_receipts_consistent']


def test_safe_npy_header_shape_dtype_and_sha(tmp_path):
    p=tmp_path/'a.npy';np.save(p,np.arange(6,dtype=np.float32))
    result=a.load_array(p,a.sha(p),(6,),np.float32,tmp_path)
    assert isinstance(result,np.memmap) and not result.flags.writeable
    for shape,dtype,digest in [((3,),np.float32,a.sha(p)),((6,),np.float64,a.sha(p)),((6,),np.float32,'0'*64)]:
        with pytest.raises(ValueError):a.load_array(p,digest,shape,dtype,tmp_path)


def test_trailing_payload_and_symlink_rejected(tmp_path):
    p=tmp_path/'a.npy';np.save(p,np.ones(2,np.float32))
    with p.open('ab') as f:f.write(b'extra')
    with pytest.raises(ValueError):a.load_array(p,a.sha(p),(2,),np.float32,tmp_path)
    link=tmp_path/'link.npy';link.symlink_to(p)
    with pytest.raises(ValueError):a.sha(link)


@pytest.mark.parametrize('array',[np.ones(2,dtype=object),np.ones(2,np.float64),np.array([np.nan,0],np.float32)])
def test_object_wrong_dtype_nonfinite_rejected(tmp_path,array):
    p=tmp_path/'x.npy';np.save(p,array)
    with pytest.raises(ValueError):a.load_array(p,a.sha(p),(2,),np.float32,tmp_path)


def test_canonical_manifest_digest_is_not_file_digest(tmp_path):
    x={'b':2,'a':1};p=tmp_path/'m.json';p.write_text(json.dumps(x,indent=2))
    assert a.sha(p)!=a.digest(x)
    assert a.digest(x)==hashlib.sha256(b'{"a":1,"b":2}').hexdigest()


def test_json_nonfinite_rejected(tmp_path):
    p=tmp_path/'m.json';p.write_text('{"x":NaN}')
    with pytest.raises(ValueError):a.read_json(p)


def test_no_neural_physics_pickle_or_consumer_imports():
    tree=ast.parse(Path(a.__file__).read_text())
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            module=n.module if isinstance(n,ast.ImportFrom) else ' '.join(x.name for x in n.names)
            assert not any(x in module for x in ['torch','pymunk','gym','pygame','pickle','experiments','dinov2','transformers'])
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
            assert n.func.attr not in ['step','reset','forward','exec_module','add_safe_globals','safe_globals']
            if ast.unparse(n.func)=='np.load':
                args={k.arg:ast.literal_eval(k.value) for k in n.keywords}
                assert args['allow_pickle'] is False and args['mmap_mode']=='r'


def test_primary_v11_v10_contrast_noncausal_scope():
    p=a.protocol()
    assert p['primary'].endswith('v11') and p['contrast'].endswith('v10')
    assert 'NOT asserted to cause ordering' in p['version_boundary']
    assert not p['prospective_control'] and not p['control_benefit_claim'] and not p['P5_admission']
    assert p['neural_calls']==p['physics_calls']==0
