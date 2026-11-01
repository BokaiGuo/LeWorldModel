"""Pure metadata/static goal tests; no actual environment is constructed."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast
import subprocess

import numpy as np
import pytest

from experiments.spectral_reachability import p5_native_goal_scene_20261027_v1 as m


class Body:
    def __init__(self,position,angle=0.,cog=(0.,0.)):
        self.position=position;self.angle=angle;self.center_of_gravity=cog
        self.velocity=(0.,0.);self.angular_velocity=0.

    def local_to_world(self,v):
        c,s=np.cos(self.angle),np.sin(self.angle)
        return np.asarray(self.position)+np.array([[c,-s],[s,c]])@np.asarray(v)


class Poly:
    # Test polygon only, not a claim to reproduce native T physics/geometry.
    radius=0.
    def get_vertices(self):return [[0.,0.],[40.,0.],[40.,40.],[0.,40.]]


def scene():
    block=Body(m.GOAL[:2],m.GOAL[2],cog=(0.,20.));block.shapes=[Poly()]
    return NS(goal_pose=m.GOAL.copy(),agent=Body(m.AGENT.copy()),block=block,
        _get_goal_pose_body=lambda pose:Body(pose[:2],pose[2]))


def test_native_transform_metadata_not_center_distance():
    witness=m.geometry_witness(scene())
    assert witness['polygons'][0]['max_abs_vertex_error']==0
    assert witness['actual_block_cog']==[0.,20.]
    assert witness['official_goal_cog']==[0.,0.]
    assert not witness['coverage_measured'] and not witness['force_contact_claim']


@pytest.mark.parametrize('fault',['agent_shift','object_shift','rotation','goal','velocity','angular_velocity',
    'empty_shapes','unknown_shape','wrong_world_transform','nonfinite'])
def test_geometric_scene_fail_closed(fault):
    env=scene()
    if fault=='agent_shift':env.agent.position=[65.,448.]
    elif fault=='object_shift':env.block.position=[257.,256.]
    elif fault=='rotation':env.block.angle+=.01
    elif fault=='goal':env.goal_pose=[256.,256.,0.]
    elif fault=='velocity':env.block.velocity=[.0001,0.]
    elif fault=='angular_velocity':env.agent.angular_velocity=.1
    elif fault=='empty_shapes':env.block.shapes=[]
    elif fault=='unknown_shape':env.block.shapes=[object()]
    elif fault=='wrong_world_transform':env.block.local_to_world=lambda v:np.asarray(v)+[256,256]
    else:env.block.shapes[0].get_vertices=lambda:[[float('nan'),0.],[0.,0.],[1.,1.]]
    with pytest.raises(ValueError):m.geometry_witness(env)


def test_prior_constructor_no_future_or_runtime_parameters():
    c=m.constructor();p=m.protocol()
    assert c['reset_to_state']==[64.,448.,256.,256.,float(np.pi/4),0.,0.]
    assert c['shape']=='T' and c['block_cog'] is None and c['legacy'] is False
    assert p['expected']['control_step']==0 and p['expected']['space_step']==1
    assert p['expected']['native_render']==p['expected']['native_get_obs']==1
    assert p['seed']==2026102799
    assert not p['P5_admission'] and not p['C2_established']


def test_actual_prepare_only_reads_source_metadata_not_trajectories():
    result=m.prepare()
    assert result['status']=='prepared_goal_not_generated'
    assert not result['physics_executed'] and not result['neural_executed'] and not result['goal_rgb_available']
    assert not any('/factual_main_v1/' in p or 'frame_15' in p or 'cuda_main_v1' in p for p in result['bindings'])


def test_authorization_and_overlay_guards_before_any_runtime_read(monkeypatch):
    with pytest.raises(ValueError,match='authorization'):m.run_main('absent','none','absent')
    monkeypatch.delenv('PYTHONPATH',raising=False)
    with pytest.raises(ValueError,match='overlay'):m.run_main('absent','none','absent',True)


def test_no_neural_process_for_scene_generation(monkeypatch):
    monkeypatch.setenv('PYTHONPATH',str(m.REPO/'.jepa-physics-site'))
    monkeypatch.setenv('PYTHONNOUSERSITE','1')
    monkeypatch.setitem(m.sys.modules,'torch',NS())
    with pytest.raises(ValueError,match='neural process'):m.run_main('absent','none','absent',True)


def test_capture_failure_retains_original_and_close_failure(tmp_path,monkeypatch):
    operations=[]
    class Fake:
        def __init__(self,**kwargs):operations.append('construct')
        def seed(self,seed):operations.append('seed')
        def reset(self):raise ValueError('injected reset failure, no physics')
        def close(self):raise RuntimeError('injected close failure')
    calls=NS(context=None,receipt=lambda:{})
    journal=NS(append=lambda event:operations.append(event['phase']))
    monkeypatch.setattr(m.base,'rng_state',lambda env:{'fixture':True})
    assets,errors=m.capture_scene(Fake,tmp_path,journal,calls)
    assert assets is None
    assert [e['phase'] for e in errors]==['capture_scene','close']
    assert 'reset failure' in errors[0]['message'] and 'close failure' in errors[1]['message']
    assert (tmp_path/'constructor_rng.json').exists() and (tmp_path/'pre_reset_rng.json').exists()


def test_capture_saves_return_before_geometry_guard(tmp_path,monkeypatch):
    order=[]
    class Fake:
        def __init__(self,**kwargs):self.kwargs=kwargs
        def seed(self,seed):order.append('seed')
        def reset(self):order.append('reset');return {},np.zeros(7)
        def close(self):order.append('close')
    calls=NS(context=None,receipt=lambda:{})
    journal=NS(append=lambda event:None)
    monkeypatch.setattr(m.base,'rng_state',lambda env:{})
    monkeypatch.setattr(m.base,'capture',lambda *args:(order.append('raw_assets') or {'frame':0,'files':{}}))
    def bad_geometry(env):
        order.append('geometry');raise ValueError('fixture mismatch')
    monkeypatch.setattr(m,'geometry_witness',bad_geometry)
    assets,errors=m.capture_scene(Fake,tmp_path,journal,calls)
    assert order==['seed','reset','raw_assets','geometry','close']
    assert assets is None and errors[0]['message']=='fixture mismatch'


def test_protocol_native_source_binding_matches_official_reset():
    source=m.ENV.read_text();tree=ast.parse(source)
    env=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PushTEnv')
    fns={n.name:n for n in env.body if isinstance(n,ast.FunctionDef)}
    reset=ast.get_source_segment(source,fns['reset'])
    setter=ast.get_source_segment(source,fns['_set_state'])
    render=ast.get_source_segment(source,fns['_render_frame'])
    assert 'self._set_state(state)' in reset
    assert setter.count('self.space.step(')==1
    assert 'space.step(' not in render
    assert 'goal_body.local_to_world(v)' in render


def test_no_top_level_physics_and_no_factual_or_cuda_imports():
    tree=ast.parse(Path(m.__file__).read_text())
    top_imports=[a.name for n in tree.body if isinstance(n,ast.Import) for a in n.names]
    assert not any(n in ('gym','pymunk','torch') for n in top_imports)
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(n and ('factual_nonzero' in n or 'fresh_fidelity' in n or 'jepa_native_action_response' in n) for n in imports)
    assert not any(isinstance(n,ast.FunctionDef) and n.name in ('run_cuda','train') for n in ast.walk(tree))


def test_no_generator_execution_in_prepare(monkeypatch):
    monkeypatch.setattr(m,'capture_scene',lambda *unused:pytest.fail('prepare executed native collector'))
    monkeypatch.setattr(m.base,'load_runtime',lambda *unused:pytest.fail('prepare imported native runtime'))
    assert m.prepare()['protocol']['agent_nuisance']==[64.,448.]


def test_output_reuse_forbidden(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'REPO',tmp_path);(tmp_path/'results').mkdir()
    path=tmp_path/'results'/(m.TAG+'_fixture')
    m.directory(path)
    with pytest.raises(ValueError,match='exclusive'):m.directory(path)


def test_prepare_output_seal_and_no_fake_goal(tmp_path,monkeypatch):
    out=tmp_path/'goal_prepare';out.mkdir()
    monkeypatch.setattr(m,'directory',lambda path:out)
    result=m.prepare_output('unused')
    m.verify_seal(output_dir=result['source_seal']['output_dir'],expected_manifest_sha256=result['source_seal']['manifest_sha256'])
    assert (out/'protocol.json').exists()
    assert not (out/'goal_spec.json').exists() and not (out/'generation.json').exists()
    assert not any(p.suffix=='.npy' for p in out.iterdir())


def test_goal_spec_is_publication_commit_after_terminal_report():
    source=Path(m.__file__).read_text();tree=ast.parse(source)
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_main')
    text=ast.get_source_segment(source,run)
    assert text.index("base.save_json(out/'report.json'") < text.index("base.save_json(out/'generation.json'")
    assert text.index("base.save_json(out/'generation.json'") < text.index("base.save_json(out/'goal_spec.json'")


def test_independent_prepare_process_has_no_torch_modules():
    code=('import sys; from experiments.spectral_reachability import p5_native_goal_scene_20261027_v1 as m; '
        'r=m.prepare(); assert not r["physics_executed"] and not r["neural_executed"]; '
        'assert not any(k=="torch" or k.startswith("torch.") for k in sys.modules); print("no_torch_prepare_verified")')
    completed=subprocess.run([m.sys.executable,'-B','-c',code],cwd=m.REPO,capture_output=True,text=True,check=True)
    assert completed.stdout.strip()=='no_torch_prepare_verified'
