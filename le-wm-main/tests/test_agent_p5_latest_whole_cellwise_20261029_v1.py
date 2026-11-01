"""Only synthetic saved fixtures; never models/physics/production consumption."""
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_latest_whole_cellwise_20261029_v1 as m


def ref(tmp_path,name,a):
    p=tmp_path/(name+'.npy');buf=io.BytesIO();np.save(buf,a,allow_pickle=False);p.write_bytes(buf.getvalue())
    return dict(directory=str(tmp_path),array=dict(file=p.name,sha256=m.sha(p),file_bytes=p.stat().st_size,
        nbytes=a.nbytes,dtype=str(a.dtype),shape=list(a.shape)))


def test_signed_zero_is_not_bitwise_equal_and_ULP_is_numeric():
    a=np.array([0.,-0.],np.float32);b=np.zeros(2,np.float32)
    d=m.diff(a,b)
    assert not d['bitwise_equal'] and d['numeric_equal'] and d['differing_float32_words']==1
    assert d['max_ULP']==d['numeric_nonzero']==d['maxabs']==0


@pytest.mark.parametrize('x',(-1.,0.,1.))
def test_adjacent_float32_ULP(x):
    a=np.array([x],np.float32);b=np.nextafter(a,np.float32(np.inf))
    assert m.diff(a,b)['max_ULP']==1
    assert m.diff(a,b)['maxabs']==abs(float(a[0])-float(b[0]))


@pytest.mark.parametrize('bad',('nan','inf','shape','dtype','empty'))
def test_compare_domain_refusal(bad):
    a=np.zeros(2,np.float32);b=a.copy()
    if bad=='nan':a[0]=np.nan
    elif bad=='inf':a[0]=np.inf
    elif bad=='shape':b=np.zeros(3,np.float32)
    elif bad=='dtype':b=b.astype(np.float64)
    else:a=b=np.zeros(0,np.float32)
    with pytest.raises(ValueError):m.diff(a,b)


def test_full3_different_with_identical_final20_not_hidden():
    a=np.zeros((1,3,2),np.float32);b=a.copy();b[:,0]=3
    d=m.slots(a,b,output=True)
    assert not d['full3']['bitwise_equal'] and d['final20']['bitwise_equal']
    assert d['per_slot'][0]['maxabs']==3
    assert 'latest15' not in d
    assert 'final20' not in m.slots(a,b)


def cell_arrays():
    return ((np.zeros((1,3,2),np.float32),np.zeros((1,3,1),np.float32),np.zeros((1,3,1),np.float32)),
        (np.zeros((1,3,2),np.float32),np.zeros((1,3,1),np.float32)))


@pytest.mark.parametrize('inputchange,outputchange,status',[
    (False,False,'identical_inputs_and_outputs_no_distinct_contrast'),
    (False,True,'identical_inputs_different_outputs_execution_numeric_seam'),
    (True,False,'different_inputs_identical_outputs_mechanism_unresolved'),
    (True,True,'different_inputs_different_outputs')])
def test_no_unjustified_architecture_conclusions(inputchange,outputchange,status):
    a=cell_arrays();b=deepcopy(a)
    if inputchange:b[0][0][:,0]=1
    if outputchange:b[1][0][:,0]=1
    v=m.compare_cell(a,b)
    assert v['classification']==status and v['architectural_shielding_certified'] is False


def request(i=0):
    return dict(index=i,root_slot=0,condition='off',task_id='fixture',selection_digest='fixed',
        aligned_history_global_frames=[5,10,15],raw_outgoing16_to20=[[0.,0.]]*5,
        actual_inputs=['a'],rolled_inputs=['p'],AA={'a':1},common_label={'l':1},H2_label={'l':2},label_png={'rgb':'fixed'})


@pytest.mark.parametrize('bad',('index','boolindex','condition','task_id','selection_digest','raw_outgoing16_to20','common_label','actual_inputs','aligned_history_global_frames'))
def test_identity_rejects_time_root_action_label_metadata_mismatch(bad):
    a=request();b=deepcopy(a)
    if bad=='boolindex':b['index']=False
    elif bad=='index':b['index']=1
    else:b[bad]='wrong'
    with pytest.raises(ValueError):m.identity(a,b,0)


def test_load_ref_real_npy_and_trailing_or_sha_failure(tmp_path):
    a=np.zeros((1,3,2),np.float32);r=ref(tmp_path,'array',a);used={}
    np.testing.assert_array_equal(m.load_ref(r,used),a)
    p=tmp_path/r['array']['file'];p.write_bytes(p.read_bytes()+b'\n')
    with pytest.raises(ValueError):m.load_ref(r,{})
    r['array'].update(file_bytes=p.stat().st_size,sha256=m.sha(p))
    with pytest.raises(ValueError):m.load_ref(r,{})


def test_load_ref_symlink_rejected(tmp_path):
    r=ref(tmp_path,'array',np.zeros(2,np.float32));p=tmp_path/'link.npy';p.symlink_to(tmp_path/'array.npy')
    r['array']['file']='link.npy'
    with pytest.raises(ValueError):m.load_ref(r,{})


def native_fixture(tmp_path):
    a=tuple(np.zeros(s,np.float32) for s in m.latest.SHAPES);p=tuple(x.copy() for x in a)
    # Rolled first2 legitimately different in the fixture; latest MUST NOT inherit whole PP.
    p[0][:,:2]=3;p[2][:,:2]=4;p[0][:,2]=5;p[2][:,2]=6
    req=request();req['actual_inputs']=[ref(tmp_path,'a'+str(i),x) for i,x in enumerate(a)]
    req['rolled_inputs']=[ref(tmp_path,'p'+str(i),x) for i,x in enumerate(p)]
    req['AA']=ref(tmp_path,'aa',np.zeros((1,3,256,384),np.float32));req['PP']=req['AA']
    output=(np.zeros((1,3,256,384),np.float32),None,np.zeros((1,3,256,16),np.float32))
    def tree(x):
        if isinstance(x,np.ndarray):
            n=len(list(tmp_path.glob('io*.npy')));return {'$array':ref(tmp_path,'io'+str(n),x)['array']}
        if isinstance(x,tuple):return {'$tuple':[tree(y) for y in x]}
        if isinstance(x,dict):return {'$dict':{k:tree(v) for k,v in x.items()}}
        return x
    calls=[]
    for arm in m.latest.ARMS:
        args=m.latest.splice_inputs(a,p,arm)
        calls.append(dict(branch=0,module='predictor',status='returned',arm=arm,inputs=tree((args,{})),outputs=tree(output)))
    row=dict(predictions={arm:c['outputs']['$tuple'][0]['$array'] for arm,c in zip(m.latest.ARMS,calls,strict=True)})
    return req,dict(calls=calls,rows=[row])


def test_actual_latest_PP_native_io_not_whole_PP(tmp_path):
    q,r=native_fixture(tmp_path)
    args,out=m.native_cell('latest',0,'PP',tmp_path,r,q,{},{});
    assert np.all(args[0][:,:2]==0) and np.all(args[0][:,2]==5) and out[0].shape==(1,3,256,384)


def test_reject_whole_PP_input_substitution_in_latest(tmp_path):
    q,r=native_fixture(tmp_path)
    # Actual recorded latest PP input poisoned with rolled WHOLE first slots.
    r['calls'][2]['inputs']['$tuple'][0]['$tuple'][0]={'$array':q['rolled_inputs'][0]['array']}
    with pytest.raises(ValueError):m.native_cell('latest',0,'PP',tmp_path,r,q,{},{})


def test_source_evidence_no_import_and_prior_slots_allowed():
    bindings={str(m.ROOT/p):h for p,h in m.SOURCE_PINS.items()}
    imported={k:str(m.ROOT/p) for k,p in (
        ('app.plan_common.models.AdaLN_vit','external/jepa-wms/app/plan_common/models/AdaLN_vit.py'),
        ('src.models.utils.modules','external/jepa-wms/src/models/utils/modules.py'))}
    v=m.evidence([{'bindings':bindings}]*2,[{'imported_sources':imported}]*2,{})
    assert v['final20_can_attend_input_frames']==[5,10,15]
    assert v['block_time_allow_matrix'][2]==[True,True,True]
    assert v['earlier_history_architecturally_masked'] is False
    assert any(s['symbol']=='build_action_block_causal_attention_mask' for r in v['sources'] for s in r['snippets'])


def test_wrong_executed_source_rejected():
    with pytest.raises(ValueError):m.evidence([{'bindings':{}}]*2,[],{})


def test_both_stricts_must_rebuild_exactly(tmp_path,monkeypatch):
    p=tmp_path/'strict.json';v={'status':'fake'};p.write_bytes(m.raw(v))
    monkeypatch.setattr(m.latest,'consume',lambda *a:{'status':'different'})
    with pytest.raises(ValueError):m.bundle('latest',tmp_path,m.REPORT_PINS['latest'],p,m.sha(p),{})


def test_cli_help_imports_no_torch_or_physics():
    code='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,check=True,capture_output=True)
    r=subprocess.run([sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,'analyze','--help'],cwd=m.ROOT,capture_output=True,check=True,text=True)
    assert '--latest-report-sha' in r.stdout and '--whole-strict-sha' in r.stdout


def test_all164_fourcells_all80_analyze_fixture_and_reverify(tmp_path,monkeypatch):
    # Tiny synthetic values, not substituted scientific results. Original strict
    # consumer is separately exercised/rejection tested; this tests join wiring.
    a=[ref(tmp_path,'base'+str(i),np.zeros((1,3,2),np.float32)) for i in range(3)]
    requests=[]
    for i in range(164):
        q=request(i);q['root_slot'],q['condition']=m.latest.bridge.factor.ORDER[i//41]
        q['actual_inputs']=a;q['rolled_inputs']=a;requests.append(q)
    manifest=dict(requests=requests,expected_model={'fixture':'not_a_model'},expected_strict_loads={'fixture':True})
    report=dict(model_before=manifest['expected_model'],protocol=dict(pairs=m.protocol()['pairs']))
    monkeypatch.setattr(m,'bundle',lambda *args:(tmp_path,report,manifest))
    monkeypatch.setattr(m,'evidence',lambda *args:{'fixture_only':True})
    monkeypatch.setattr(m,'read',lambda *args:{})
    def native(kind,i,cell,*args):
        arrays=cell_arrays()
        if i==163 and cell=='PP' and kind=='latest':arrays[0][0][:,0]=1
        return arrays
    monkeypatch.setattr(m,'native_cell',native)
    args=dict(latest_output=str(tmp_path),latest_report_sha=m.REPORT_PINS['latest'],latest_strict_receipt=str(tmp_path/'ls'),
        latest_strict_sha='a'*64,whole_output=str(tmp_path),whole_report_sha=m.REPORT_PINS['whole'],whole_strict_receipt=str(tmp_path/'ws'),whole_strict_sha='b'*64)
    result=m.analyze(**args)
    assert len(result['branches'])==164 and len(result['pairs'])==80
    assert sum(sum(c.values()) for c in result['classification_counts'].values())==656
    assert result['classification_counts']['PP']['different_inputs_identical_outputs_mechanism_unresolved']==1
    assert result['branches'][163]['cells']['PP']['architectural_shielding_certified'] is False
    assert result['new_neural_calls']==result['new_physics_calls']==0
    # Real exclusive output writer/fsync/JSON codec and deterministic verify.
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    out=tmp_path/'results'/(m.NAME+'_fixture.json');receipt=m.write_new(result,out)
    with pytest.raises(ValueError):m.write_new(result,out)
    monkeypatch.setattr(m,'analyze',lambda **kwargs:result)
    monkeypatch.setattr(m,'read',lambda path,expected,used:json.loads(Path(path).read_bytes()))
    assert m.verify(out,receipt['sha256'])['status']=='saved_cellwise_comparison_recomputed'


def test_full_fixture_array_results_not_just_equal_mse():
    a=cell_arrays();b=deepcopy(a)
    a[1][0][:,2]=[1.,-1.];b[1][0][:,2]=[-1.,1.]
    assert np.mean(a[1][0]**2)==np.mean(b[1][0]**2)
    assert not m.compare_cell(a,b)['outputs']['visual']['final20']['bitwise_equal']
