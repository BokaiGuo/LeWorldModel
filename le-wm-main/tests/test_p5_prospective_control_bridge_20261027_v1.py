"""Nonneural fixtures plus actual saved preflight; no native runtime imports."""
from copy import deepcopy
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import p5_prospective_control_bridge_20261027_v1 as m


def ready():
    return dict(schema=m.TAG,status='ready_for_main_CUDA_compile',blockers=[],contract=m.contract(),
        compiler_inputs=dict(allowed_data_files={'context':'abc','independent_goal':'def'}),compiler_inputs_sha256='inputs',
        runner_sources=dict(compiler=dict(path='new_compiler.py',sha256='code')),
        model_identity={'native_frozen':'weights'},prefix_env_commands=np.zeros((10,2)).tolist(),
        factual_reference_packet=np.full((5,2),.1).tolist(),root_identity_sha256='root',goal=dict(rgb_sha256='def'))


def receipt(p):
    roles=('response_columns','goal_features','nominal_successor','ridge_successor','selected_replay','gram','rhs','ridge_delta')
    return dict(schema=m.TAG+'.compiler',status='selected_packet_replayed',prepared_sha256=m.digest(p),
        compiler_inputs_sha256=p['compiler_inputs_sha256'],compiler_data_bindings=p['compiler_inputs']['allowed_data_files'],
        goal_rgb_sha256=p['goal']['rgb_sha256'],compiler_source=p['runner_sources']['compiler'],backend='host_cuda_only',
        model_before=p['model_identity'],model_after=p['model_identity'],used_observed_frame15=False,
        physical_calls=0,completed=m.contract()['compiler']['expected_completed'],
        attempted=m.contract()['compiler']['expected_completed'],selected_replay_exact=True,
        artifact_roles={k:k+'.npy' for k in roles},artifact_bindings={k+'.npy':'hash' for k in roles},
        partial_work_unknown=False,candidates=[dict(id='nominal',env_packet=np.zeros((5,2)).tolist(),independent_goal_mse=.5),
            dict(id='ridge',env_packet=np.full((5,2),.2).tolist(),independent_goal_mse=.4)],
        selected_id='ridge',selected_env_packet=np.full((5,2),.2).tolist())


def test_minimal_three_arm_freeze_and_exact_packet_lengths():
    p=ready();r=receipt(p);rows=m.frozen_branches(p,r)
    assert [v['arm'] for v in rows]==list(m.ARMS)
    assert all(v['seed']==m.SEED and v['status']=='planned_not_constructed' for v in rows)
    assert all(np.shape(v['suffix_env_packet'])==(5,2) and np.shape(v['prefix_env_commands'])==(10,2) for v in rows)
    assert rows[0]['suffix_env_packet']==r['selected_env_packet']
    assert rows[1]['suffix_env_packet']==np.zeros((5,2)).tolist()
    assert rows[2]['suffix_env_packet']==p['factual_reference_packet']


def test_nominal_first_tie_not_outcome_reselection():
    p=ready();r=receipt(p);r['candidates'][1]['independent_goal_mse']=.5
    r['selected_id']='nominal';r['selected_env_packet']=np.zeros((5,2)).tolist()
    rows=m.frozen_branches(p,r)
    assert rows[0]['suffix_env_packet']==rows[1]['suffix_env_packet']
    assert len(rows)==3  # Keep duplicate-action roles; never replace them.


@pytest.mark.parametrize('fault',['backend','frame15','model','counts','attempt','physical','source','inputs','data_read',
    'goal','argmin','selected_packet','support','short_packet','nan','unknown_work','replay','roles','candidate_order'])
def test_compile_receipt_fail_closed(fault):
    p=ready();r=deepcopy(receipt(p))
    if fault=='backend':r['backend']='cpu'
    elif fault=='frame15':r['used_observed_frame15']=True
    elif fault=='model':r['model_after']={'changed':'weights'}
    elif fault=='counts':r['completed']['predictor']=12
    elif fault=='attempt':r['attempted']['predictor']=14
    elif fault=='physical':r['physical_calls']=1
    elif fault=='source':r['compiler_source']['sha256']='changed'
    elif fault=='inputs':r['compiler_inputs_sha256']='changed'
    elif fault=='data_read':r['compiler_data_bindings']['frame15']='future'
    elif fault=='goal':r['goal_rgb_sha256']='factual15'
    elif fault=='argmin':r['selected_id']='nominal'
    elif fault=='selected_packet':r['selected_env_packet'][0][0]+=.1
    elif fault=='support':r['candidates'][1]['env_packet'][0][0]=2
    elif fault=='short_packet':r['candidates'][1]['env_packet'].pop()
    elif fault=='nan':r['candidates'][0]['independent_goal_mse']=float('nan')
    elif fault=='unknown_work':r['partial_work_unknown']=True
    elif fault=='replay':r['selected_replay_exact']=False
    elif fault=='roles':r['artifact_roles'].pop('goal_features')
    else:r['candidates'].reverse()
    with pytest.raises(ValueError):m.frozen_branches(p,r)


def test_blocked_prepare_cannot_compile_or_freeze():
    p=ready();p['status']='blocked_pre_execution_inputs';p['blockers']=['independent goal missing']
    with pytest.raises(ValueError,match='blocked'):m.compiler_bundle(p,m.digest(p))
    with pytest.raises(ValueError,match='blocked'):m.frozen_branches(p,{})


def goal_fixture(tmp_path):
    rgb=tmp_path/'separate_goal.npy';np.save(rgb,np.full((224,224,3),77,np.uint8),allow_pickle=False)
    source=tmp_path/'reviewed_goal_source.py';source.write_text('"""Fixture provenance, not actual generation."""\n')
    receipt=tmp_path/'generation.json'
    receipt.write_text(json.dumps(dict(schema=m.TAG+'.goal_generation',goal_rgb_sha256=m.sha(rgb.read_bytes()),
        object_goal_pose=[256.,256.,float(np.pi/4)],used_factual_future=False,
        method='separate native goal scene; no branch execution',source_bindings={str(source):m.sha(source.read_bytes())})))
    spec=dict(schema=m.TAG+'.goal',rgb_path=str(rgb),rgb_sha256=m.sha(rgb.read_bytes()),
        object_goal_pose=[256.,256.,float(np.pi/4)],agent_goal_role='scene nuisance; no agent success claim',
        origin='independently_declared_native_goal_scene',generation_receipt=str(receipt),
        generation_receipt_sha256=m.sha(receipt.read_bytes()))
    path=tmp_path/'goal_spec.json';path.write_text(json.dumps(spec))
    return path,spec


def test_independent_goal_schema_fixture_only(tmp_path):
    path,spec=goal_fixture(tmp_path)
    assert m.goal_asset(m.saved.Inputs(),path,[])==spec


@pytest.mark.parametrize('fault',['renamed_factual','future_origin','wrong_pose','fake_agent_success','unknown_field','receipt_future'])
def test_goal_leakage_or_provenance_rejected(tmp_path,fault):
    path,spec=goal_fixture(tmp_path);forbidden=[]
    if fault=='renamed_factual':forbidden=[m.sha(np.full((224,224,3),77,np.uint8).tobytes())]
    elif fault=='future_origin':spec['origin']='factual_frame15'
    elif fault=='wrong_pose':spec['object_goal_pose'][0]+=1
    elif fault=='fake_agent_success':spec['agent_goal_role']='agent reaches object goal equals success'
    elif fault=='unknown_field':spec['factual_target_path']='frame15.npy'
    else:
        receipt=Path(spec['generation_receipt']);r=json.loads(receipt.read_text());r['used_factual_future']=True
        receipt.write_text(json.dumps(r));spec['generation_receipt_sha256']=m.sha(receipt.read_bytes())
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError):m.goal_asset(m.saved.Inputs(),path,forbidden)


def test_real_saved_factual_agent_object_distinction_and_explicit_blockers():
    result=m.prepare()
    assert result['status']=='blocked_pre_execution_inputs' and len(result['blockers'])==3
    segments=result['factual_diagnostic']['segments']
    assert not segments[0]['object_pose_exactly_unchanged_at_all_saved_steps']
    for row in segments[1:]:
        assert row['object_pose_exactly_unchanged_at_all_saved_steps']
        assert row['object_origin_displacement']==[0.,0.]
        assert np.linalg.norm(row['agent_displacement'])>0
        assert row['final_native_coverage']==row['max_native_coverage']==0
    assert result['new_neural_calls']==result['new_physics_steps']==0
    assert not result['P5_admission'] and not result['C2_established']
    assert len(result['prefix_identity'])==11
    assert [f['frame'] for f in result['compiler_inputs']['frames']]==[0,5,10]
    assert all('frame_15' not in p for p in result['compiler_inputs']['allowed_data_files'])


def test_ready_compiler_bundle_never_loads_factual_target(tmp_path):
    path,_=goal_fixture(tmp_path)
    result=m.prepare(path,Path(m.__file__),Path(m.__file__))  # Metadata fixture, not production runners.
    assert result['status']=='ready_for_main_CUDA_compile'
    bundle=m.compiler_bundle(result,m.digest(result))
    assert set(bundle)=={'rgb_context','raw_proprio_context','history_env_packets','goal_rgb'}
    assert bundle['rgb_context'].shape==(3,224,224,3)
    assert bundle['raw_proprio_context'].shape==(3,4)
    assert bundle['history_env_packets'].shape==(2,5,2)
    assert len(result['compiler_inputs']['allowed_data_files'])==7
    with pytest.raises(ValueError,match='retained protocol SHA'):m.compiler_bundle(result,'bad')


def test_prepare_freeze_durability_end_to_end_fixture(tmp_path,monkeypatch):
    # Redirect ONLY the new consumer output root; real old factual inputs read-only.
    goal,_=goal_fixture(tmp_path)
    p=m.prepare(goal,Path(m.__file__),Path(m.__file__))
    seal=m.create_seal(files=[Path(m.__file__)],args_json=json.dumps(dict(goal_spec=str(goal),
        compiler_source=m.__file__,executor_source=m.__file__)),output_dir=tmp_path/'source_seal')
    p['audit_source_seal']=seal
    prepared=tmp_path/'prepared.json';m.write(prepared,p)
    r=receipt(p)
    artifacts={}
    for role,relative in r['artifact_roles'].items():
        file=tmp_path/relative;file.write_bytes(b'fixture-not-neural-array')
        artifacts[str(file)]=m.sha(file.read_bytes());r['artifact_roles'][role]=str(file)
    r['artifact_bindings']=artifacts
    receipt_path=tmp_path/'compiler.json';m.write(receipt_path,r)
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'REPO',tmp_path)
    # Rebuild uses original REPO; retain real preparation while testing storage transaction.
    monkeypatch.setattr(m,'prepare',lambda *args: {k:v for k,v in p.items() if k!='audit_source_seal'})
    output=tmp_path/'results'/(m.TAG+'_freeze_fixture')
    rows=m.freeze_output(prepared,receipt_path,output)
    assert len(rows)==3
    final=json.loads((output/'freeze_receipt.json').read_text())
    assert set(final['census'])==set(m.ARMS)
    assert final['status']=='all_three_branches_durable_not_executed'
    assert final['selected_packet_sha256']==m.sha((output/'selected_packet.json').read_bytes())
    with pytest.raises(ValueError,match='exclusive'):m.freeze_output(prepared,receipt_path,output)


def test_no_neural_or_physics_import_or_execution_commands():
    tree=ast.parse(Path(m.__file__).read_text())
    names=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    names += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
    assert not any(n and n.split('.')[0] in ('torch','gym','gymnasium','pymunk','app') for n in names)
    assert not any(isinstance(n,ast.FunctionDef) and n.name in ('run_cuda','run_physics','step','reset') for n in ast.walk(tree))


def test_c1_not_c2_contract_and_unmodified_source_task_predicate():
    c=m.contract()
    assert not c['P5_admission'] and not c['P1_admission'] and not c['generalization']
    assert c['C2_status'].startswith('not_established')
    assert c['execution']['expected']['space_step']==3*(1+15*10)
    assert sum(c['compiler']['predictor_phases'].values())==13
    assert 'mixed-scene proxy' in c['evaluation']['image_metric']
