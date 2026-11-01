"""Numeric/JSON fixtures, no physics or neural imports/execution."""
import ast
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from experiments.spectral_reachability import p5_jepa_panel_audit_20261027_v1 as a


def fixture_trajectory(move_object=False):
    full=[];trans=[]
    for i in range(16):
        full.append(dict(bodies=dict(agent=dict(position=[float(i),0.]),block=dict(position=[float(i) if move_object else 10.,20.],angle=.3)),env=dict(n_contact_points=0)))
        trans.append(dict(kind='step',info=dict(final_coverage=.2),reward=.2/.95) if i else dict(kind='reset'))
    return full,trans


def test_agent_motion_is_not_object_motion_or_C2():
    full,trans=fixture_trajectory()
    result=a.trajectory_summary(full,trans)
    assert result['agent']['path_length_px']==15
    assert result['object']['path_length_px']==0
    assert result['agent_only_motion'] and result['object_pose_exactly_stationary']
    assert not result['C2_admission']
    assert result['coverage_reset'] is None and len(result['coverage_frames1_to15'])==15


def test_object_motion_still_not_C2_admission():
    full,trans=fixture_trajectory(True)
    result=a.trajectory_summary(full,trans)
    assert result['object']['path_length_px']==15
    assert not result['agent_only_motion'] and not result['object_pose_exactly_stationary']
    assert not result['C2_admission']


def test_native_unwrapped_angles_and_no_false_zero_after_rotation():
    full,trans=fixture_trajectory()
    for i,frame in enumerate(full):
        frame['bodies']['block']['angle']=float(i)*2*np.pi/15
    result=a.trajectory_summary(full,trans)
    assert result['object_angle_path_abs_rad']==pytest.approx(2*np.pi)
    assert not result['object_pose_exactly_stationary'] and not result['agent_only_motion']


@pytest.mark.parametrize('change',[lambda rows:rows.pop(),lambda rows:rows.reverse(),lambda rows:rows[0].update(seed=2026102701),lambda rows:rows[1].update(seed=rows[0]['seed'])])
def test_fixed_cohort_no_subset_reorder_replacement(change):
    rows=[dict(seed=s) for s in a.SEEDS]
    change(rows)
    with pytest.raises(ValueError):
        a.fixed_rows(rows)


@pytest.mark.parametrize('payload',['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'])
def test_strict_json(payload,tmp_path):
    path=tmp_path/'a.json';path.write_text(payload)
    with pytest.raises(ValueError):
        a.read_json(path)


def test_safe_typed_numeric_json():
    value=dict(dtype='<f4',shape=[2],values=[1.,2.])
    np.testing.assert_array_equal(a.vector(value,(2,),np.float32),[1,2])
    for mutation in [dict(dtype='|O'),dict(shape=[3]),dict(values=[float('nan'),0])]:
        bad={**value,**mutation}
        with pytest.raises(ValueError):
            a.vector(bad,(2,),np.float32)


def test_helper_import_no_neural_physics_and_protocol_scope():
    before=set(sys.modules)
    module=a.helpers(a.ROOT)
    assert not {n.split('.')[0] for n in set(sys.modules)-before}&{'torch','pymunk','gym','pygame','transformers'}
    assert module.NAME==a.HELPER
    p=a.protocol()
    assert not any(p[k] for k in ('neural_calls','physics_calls','checkpoint_decoding','C2_admission','P5_admission'))


def test_import_call_ast_has_no_env_neural_or_pickle_decode():
    tree=ast.parse(Path(a.__file__).read_text())
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            name=n.module if isinstance(n,ast.ImportFrom) else ' '.join(x.name for x in n.names)
            assert not any(x in name for x in ('torch','pymunk','gym','pygame','pickle','transformers'))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
            assert n.func.attr not in ('step','reset','forward','backward','load','set_state','add_safe_globals')


def test_initial_root_identity_depends_on_seed_and_input():
    rgb=np.zeros((4,224,224,3),np.uint8);prop=np.zeros((4,4),np.float32)
    first=a.root_identity(a.SEEDS[0],rgb,prop)
    assert first!=a.root_identity(a.SEEDS[1],rgb,prop)
    prop[0,0]=1
    assert first!=a.root_identity(a.SEEDS[0],rgb,prop)


def fidelity_fixture(tmp_path):
    op=a.helpers(a.ROOT)
    controls=np.zeros((4,3,10),np.float32)
    physical=[];inputs={};prepared=[];rows=[]
    model={name:dict(all_cuda_finite=True,eval=True,frozen=True) for name in ('official_encoder','proprio_encoder','predictor')}
    for seed in a.SEEDS:
        physical.append(dict(seed=seed,identity=str(seed),input_sha256=dict(rgb='r',normalized_proprio='p',packets='a'),roles=dict(a=dict(object_pose_exactly_stationary=True))))
        inputs[seed]=dict(controls=controls)
        prepared.append(dict(seed=seed,root_sha256=str(seed),factual=str(tmp_path/f'seed_{seed}'),input_sha256=physical[-1]['input_sha256'],controls={arm:controls[i].tolist() for i,arm in enumerate(a.ARMS)}))
        feat=np.zeros((4,256,384),np.float32);feat[3]=3
        arrays=dict(official_patch_features=feat,**{arm+'_successor':np.full((1,256,384),i+1,np.float32) for i,arm in enumerate(a.ARMS)})
        artifacts={}
        for key,value in arrays.items():
            path=tmp_path/f'seed_{seed}_{key}.npy';np.save(path,value,allow_pickle=False)
            artifacts[key]=dict(path=str(path),sha256=a.sha(path),shape=list(value.shape),dtype='float32',bytes=path.stat().st_size)
        costs={arm:float((i+1-3)**2) for i,arm in enumerate(a.ARMS)}
        rows.append(dict(seed=seed,status='completed',error=None,finalization_errors=[],root_sha256=str(seed),counts=dict(official_encoder=1,proprio_encoder=1,predictor=4),attempted_counts=dict(official_encoder=1,proprio_encoder=1,predictor=4),
                         frame_counts=dict(official_encoder=4,proprio_encoder=3,predictor=12),attempted_frame_counts=dict(official_encoder=4,proprio_encoder=3,predictor=12),arm_status={arm:'completed' for arm in a.ARMS},model_unchanged=True,model_after=model,
                         artifacts=artifacts,costs=costs,actual_minus_control={arm:costs['actual']-costs[arm] for arm in a.ARMS[1:]}))
    manifest=dict(schema=a.CUDA,roots=prepared,protocol=dict(fixture=True))
    report=dict(schema=a.CUDA,status='fresh_native_panel_fidelity_recorded',protocol=manifest['protocol'],manifest_sha256=a.digest(manifest),error=None,finalization_errors=[],source_check_errors=[],sources_unchanged=True,new_physical_steps=0,P5_admission=False,
                model_unchanged=True,model_before=model,model_after=model,strict_loads=dict(predictor=dict(exact_match=True,missing_keys=[],unexpected_keys=[],shape_mismatches=[],dtype_mismatches=[])),counts=dict(official_encoder=3,proprio_encoder=3,predictor=12),attempted_counts=dict(official_encoder=3,proprio_encoder=3,predictor=12),
                frame_counts=dict(official_encoder=12,proprio_encoder=9,predictor=36),attempted_frame_counts=dict(official_encoder=12,proprio_encoder=9,predictor=36),rows=rows)
    mp=tmp_path/'manifest.json';rp=tmp_path/'report.json'
    mp.write_text(json.dumps(manifest));rp.write_text(json.dumps(report))
    return dict(cuda_manifest=str(mp),cuda_report=str(rp),panel=str(tmp_path)),physical,inputs,op,report


def test_all3_saved_array_fidelity_arithmetic_fixture(tmp_path):
    frozen,physical,inputs,op,_=fidelity_fixture(tmp_path)
    checks={}
    def check(name,condition):
        checks[name]=bool(condition)
        assert condition,name
    result=a.audit_fidelity(frozen,physical,inputs,op,check)
    assert len(result['rows'])==3 and all(checks.values())
    assert result['actual_higher_counts']=={arm:3 for arm in a.ARMS[1:]}
    assert all(x['arithmetic']['arms']['actual']['mse']==4 for x in result['rows'])


@pytest.mark.parametrize('mutation',['cost','attempted','identity','missing_seed'])
def test_fidelity_mismatch_not_silently_accepted(tmp_path,mutation):
    frozen,physical,inputs,op,report=fidelity_fixture(tmp_path)
    if mutation=='cost':
        report['rows'][0]['costs']['actual']+=.01
    elif mutation=='attempted':
        report['rows'][0]['attempted_counts']['predictor']=5
    elif mutation=='identity':
        report['rows'][0]['root_sha256']='wrong'
    else:
        report['rows'].pop()
    Path(frozen['cuda_report']).write_text(json.dumps(report))
    def check(name,condition):
        a.require(condition,name)
    with pytest.raises(ValueError):
        a.audit_fidelity(frozen,physical,inputs,op,check)
