"""Saved actual arrays and tampering counterexamples; never executes a model."""
from copy import deepcopy
from io import BytesIO
import gzip

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import p1_line_search_gn_rgb_saved6_audit_20261027_v1 as a


@pytest.fixture(scope='module')
def bundle(): return a.load_bundle()


def npy(z):
    f=BytesIO(); np.save(f,z,allow_pickle=False); return f.getvalue()


def test_actual_raw_six_complete_recompute(bundle):
    r=a.audit(bundle)
    assert r['all_checks'] and r['raw_NPY']==r['raw_PNG']==6 and r['raw_image_events']==36
    assert r['resources']=={k:dict(attempted=6,completed=6) for k in ('encoder','projector')}
    assert [x['actual_Q_costs'] for x in r['roots']]==[
        [1.3611217371806512,1.3106902300273984,1.2682560823041582],
        [2.0772934423283536,2.040062158820375,2.0283601639713478]]
    assert r['rows'][0]['native_prediction_mse']==r['rows'][3]['native_prediction_mse']==0.
    assert all(x['object_position_delta']==x['object_angle_delta']==0. and not x['recorded_official_success'] for x in r['roots'])
    assert r['new_neural_calls']==r['new_physics']==0 and not r['P1_admission']


@pytest.mark.parametrize('mutant',[
    'NPYbit','NPYswap','PNGbit','PNGswap','Q','goal','prediction_frame','prediction_context','task_goal',
    'step_order','root','native_history','budget','return_missing','attempt_duplicate','forbidden_event',
    'analysis_actual','analysis_prediction','analysis_native','latent_path','latent_dtype','partial_row',
    'physical_state','physical_chain','selected_command','model_binding',
])
def test_saved_mutants_rejected(bundle,mutant):
    b=deepcopy(bundle)
    if mutant=='NPYbit':
        z=a.native(b['npydata'][1]).copy(); z[0]+=np.float32(.01); b['npydata'][1]=npy(z)
    elif mutant=='NPYswap': b['npydata'][1],b['npydata'][2]=b['npydata'][2],b['npydata'][1]
    elif mutant=='PNGbit':
        rgb=a.pixels(b['pngdata'][1]); rgb[0,0,0]^=1; f=BytesIO(); Image.fromarray(rgb).save(f,format='PNG'); b['pngdata'][1]=f.getvalue()
    elif mutant=='PNGswap': b['pngdata'][1],b['pngdata'][2]=b['pngdata'][2],b['pngdata'][1]
    elif mutant=='Q': b['prepare']['root_metrics'][0]['q'][0][0]+=.1
    elif mutant=='goal': b['prepare']['root_metrics'][0]['goal'][0]+=.1
    elif mutant=='prediction_frame': b['prepare']['frames'][1]['prediction_native192'][0]+=.1
    elif mutant=='prediction_context': b['predictions'][0]['contexts'][1]['latent_history'][-1][0]+=.1
    elif mutant=='task_goal': b['tasks'][0]['goal'][0]+=.1
    elif mutant=='step_order': b['prepare']['frames'][1]['step']=10
    elif mutant=='root': b['prepare']['frames'][1]['root_sha256']='0'*64
    elif mutant=='native_history': b['predictions'][0]['contexts'][0]['action_history'][0][0]+=.1
    elif mutant=='budget': b['report']['resources']['encoder']['completed']=5
    elif mutant=='return_missing': b['events'].pop(2)
    elif mutant=='attempt_duplicate': b['events'].insert(2,deepcopy(b['events'][1]))
    elif mutant=='forbidden_event': b['events'][1]['entry']['resource']='predict'
    elif mutant=='analysis_actual': b['analysis']['rows'][1]['actual_Q_mse']+=.1
    elif mutant=='analysis_prediction': b['analysis']['rows'][1]['predicted_Q_mse']+=.1
    elif mutant=='analysis_native': b['analysis']['rows'][1]['native_prediction_mse']+=.1
    elif mutant=='latent_path': b['report']['rows'][0]['latent']['path']='/old/frame.npy'
    elif mutant=='latent_dtype': b['report']['rows'][0]['latent']['dtype']='float64'
    elif mutant=='partial_row': b['report']['rows'][0]['status']='failed'
    elif mutant=='physical_state': b['physical']['rows'][0]['checkpoints'][1]['state'][0]+=.1
    elif mutant=='physical_chain': b['physical_events'][0]['event_sha256']='0'*64
    elif mutant=='selected_command': b['physical_prepared']['plans'][0]['commands'][0][0]+=.1
    elif mutant=='model_binding':
        path=next(p for p in b['prepare']['model_bindings'] if p.endswith('weights.pt'))
        b['prepare']['model_bindings'][path]='0'*64
    with pytest.raises(ValueError): a.audit(b)


@pytest.mark.parametrize('z',[np.zeros(192,np.float64),np.zeros((1,192),np.float32),np.full(192,np.nan,np.float32)])
def test_invalid_NPY_no_cast(z):
    with pytest.raises(ValueError): a.native(npy(z))


def test_NPY_trailing_bytes_and_raw_roundtrip():
    z=np.random.default_rng(20261027).normal(size=192).astype(np.float32); encoded=npy(z)
    assert np.array_equal(z,a.native(encoded))
    with pytest.raises(ValueError): a.native(encoded+b'extra')


@pytest.mark.parametrize('mode,size',[('L',(224,224)),('RGB',(112,112))])
def test_invalid_PNG(mode,size):
    f=BytesIO(); Image.new(mode,size).save(f,format='PNG')
    with pytest.raises(ValueError): a.pixels(f.getvalue())


@pytest.mark.parametrize('fault',['truncate','CRC','trailing'])
def test_gzip_complete_CRC_required(tmp_path,fault):
    data=gzip.compress(b'{"actual":true}')
    if fault=='truncate': data=data[:-5]
    elif fault=='trailing': data+=b'bad'
    else:
        data=bytearray(data); data[-8]^=1; data=bytes(data)
    path=tmp_path/'tiny.json.gz'; path.write_bytes(data)
    with pytest.raises((ValueError,a.zlib.error)): a.read(path)


def test_duplicate_JSON_keys_rejected(tmp_path):
    path=tmp_path/'tiny.json'; path.write_bytes(b'{"Q":1,"Q":2}')
    with pytest.raises(ValueError,match='duplicate'): a.read(path)


def test_source_hash_drift_rejected(bundle,monkeypatch):
    path=next(p for p in bundle['bindings'] if p.endswith('p1_line_search_gn_rgb_cuda_20261027_v1.py')); old=a.sha
    monkeypatch.setattr(a,'sha',lambda p:'0'*64 if str(p)==path else old(p))
    with pytest.raises(ValueError,match='SHA mismatch'): a.load_bundle()


def test_postread_TOCTOU(bundle,monkeypatch):
    path=next(iter(bundle['bindings'])); old=a.sha
    monkeypatch.setattr(a,'sha',lambda p:'0'*64 if str(p)==path else old(p))
    with pytest.raises(ValueError,match='TOCTOU'): a.audit(bundle)


def test_small_saved_output_exclusive(tmp_path):
    out=tmp_path/'audit.json'; result=a.execute(out)
    assert result['bytes']<256*1024 and result['sha256']==a.sha(out) and a.read(out)['all_checks']
    with pytest.raises(ValueError,match='exclusive'): a.execute(out)


def test_independent_source_imports_no_producer_or_model():
    import ast
    tree=ast.parse(a.Path(a.__file__).read_text())
    names=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert all(not str(n).startswith(('experiments','torch','stable_worldmodel')) for n in names)
