"""Non-neural fake-env contracts; never imports official env or executes physics."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.spectral_reachability import p5_native_c1_executor_20261027_v1 as e


def test_source_prepare_nonphysical_exact_contract():
    before=set(sys.modules)
    m=e.source_prepare(e.ROOT)
    assert m['status']=='prepared_executor_source_no_branch_authority'
    assert m['arms']==['compiled','nominal','factual_reference']
    assert m['seed']==2026102701
    assert m['contract']['execution']['expected']['space_step']==453
    assert not m['physics_executed'] and not m['neural_executed']
    assert not {n.split('.')[0] for n in set(sys.modules)-before}&{'pymunk','gym','pygame','torch','transformers'}


def test_prefix_gate_exact_files_not_just_numeric_state(tmp_path):
    ref=tmp_path/'reference';new=tmp_path/'native';ref.mkdir();new.mkdir()
    for name in e.NAMES:
        (ref/name).write_bytes(b'fixed bytes')
        (new/name).write_bytes(b'fixed bytes')
    expected=dict(frame=10,files={n:dict(path=str(ref/n),sha256=e.sha(ref/n)) for n in e.NAMES})
    record=dict(frame=10,files={n:dict(sha256=e.sha(new/n)) for n in e.NAMES})
    e.prefix_gate(new,expected,record)
    (new/'transition.json').write_bytes(b'fixed bytes\n')
    with pytest.raises(e.IntegrityFailure,match='byte mismatch'):
        e.prefix_gate(new,expected,record)


def test_wrapped_object_angle_and_agent_not_goal():
    public=dict(bodies=dict(agent=dict(position=[1,2],velocity=[3,4]),block=dict(position=[256.,256.],angle=float(np.pi/4+2*np.pi))),
                env=dict(goal_pose=dict(values=[256.,256.,float(np.pi/4)]),coverage_arr=[.4,.5]))
    result=e.object_metrics(public)
    assert result['object_xy_goal_error']==0 and result['object_angle_goal_error']<1e-14
    assert result['agent_xy']==[1,2] and 'agent_goal_error' not in result
    assert not result['C2_established'] and not result['P5_admission']


@pytest.mark.parametrize('arm',e.ARMS)
@pytest.mark.parametrize('mismatch_frame',[None,0,10])
@pytest.mark.parametrize('close_failure',[False,True])
def test_fake_arm_independent_prefix_gate_suffix_binding(tmp_path,monkeypatch,arm,mismatch_frame,close_failure):
    panel,v2,base,_,_=e.dependencies(e.ROOT)
    monkeypatch.setattr(base,'public_state',lambda env:dict(fixture_seed=env._seed,frame=env.i))
    writer=panel.BudgetIO(base)
    out=tmp_path/'out';writer.mkdir(out);writer.mkdir(out/arm)
    factual=tmp_path/'reference';writer.mkdir(factual)
    inputs=v2.action_inputs(v2.native_stats(e.ROOT))
    prefix=inputs['raw_env_commands'][:10]
    suffix=np.full((5,2),.01,np.float64) if arm=='compiled' else np.zeros((5,2),np.float64) if arm=='nominal' else inputs['raw_env_commands'][10:]
    journal=writer.journal(out/arm/'journal.jsonl')
    records=[];gates=[];instances=[]
    class Calls:
        context=None
        def receipt(self):
            return dict(fixture_only=True)
    class Env:
        def __init__(self,**kwargs):
            self._seed=123;self.i=0;self.closed=False;self.commands=[]
            self.agent=SimpleNamespace(position=[10.,20.])
            self.np_random=np.random.default_rng(123);self.random_state=np.random.RandomState(123)
            instances.append(self)
        def seed(self,value):
            self._seed=value;self.np_random=np.random.default_rng(value);self.random_state=np.random.RandomState(value)
        def arrays(self):
            state=np.array([10,20,30,40,0,0,0],np.float32)
            return dict(visual=np.full((224,224,3),self.i,np.uint8),proprio=state[[0,1,5,6]]),state
        def reset(self):
            assert self._seed==e.SEED
            return self.arrays()
        def step(self,u):
            if self.i>=10:
                assert gates==list(range(11))
            last=json.loads((out/arm/'journal.jsonl').read_text().splitlines()[-1])['event']
            assert last['phase']=='intent' and last['operation']=='step'
            command=json.loads((out/arm/f'command_{self.i+1:02d}.json').read_text())
            np.testing.assert_array_equal(command['raw_env_argument'],u)
            np.testing.assert_array_equal(command['absolute_target'],np.array([10,20])+100*u)
            self.commands.append(u.copy());self.i+=1
            obs,state=self.arrays();return obs,0.,False,dict(state=state)
        def close(self):
            self.closed=True
            if close_failure:
                raise RuntimeError('fixture close failure')
    refenv=Env();refenv.seed(e.SEED)
    identities=[]
    for i in range(11):
        refenv.i=i;obs,state=refenv.arrays()
        transition=dict(kind='reset') if i==0 else dict(kind='step',reward=0.,done=False,info=dict(state=state))
        rec=panel.capture(base,writer,refenv,obs,state,transition,factual/f'frame_{i:02d}')
        identities.append(dict(frame=i,files={name:dict(path=str(factual/f'frame_{i:02d}'/name),sha256=rec['files'][name]['sha256']) for name in e.NAMES}))
    if mismatch_frame is not None:
        path=factual/f'frame_{mismatch_frame:02d}/rgb.npy'
        path.write_bytes(path.read_bytes()+b'corrupt')
    manifest=dict(native_constructor=base.CTOR,prefix_identity=identities)
    branch=dict(arm=arm,seed=e.SEED,prefix_env_commands=prefix.tolist(),suffix_env_packet=suffix.tolist())
    if mismatch_frame is None:
        if close_failure:
            with pytest.raises(RuntimeError,match='fixture close failure'):
                e.collect_arm(panel,base,writer,Env,branch,manifest,out,journal,Calls(),records,gates)
        else:
            e.collect_arm(panel,base,writer,Env,branch,manifest,out,journal,Calls(),records,gates)
        assert len(records)==16 and gates==list(range(11))
        np.testing.assert_array_equal(np.array(instances[-1].commands),np.concatenate((prefix,suffix)))
    else:
        with pytest.raises(e.IntegrityFailure):
            e.collect_arm(panel,base,writer,Env,branch,manifest,out,journal,Calls(),records,gates)
        assert len(instances[-1].commands)==mismatch_frame
        assert len(records)==mismatch_frame+1
        assert not (out/arm/'command_11.json').exists()
    assert instances[-1].closed and instances[-1] is not refenv
    if close_failure:
        last=json.loads((out/arm/'journal.jsonl').read_text().splitlines()[-1])['event']
        assert last['phase']=='close_failed' and 'fixture close failure' in last['error']
        assert (last['original_error'] is None)==(mismatch_frame is None)
        if mismatch_frame is not None:
            assert last['original_error'].startswith('IntegrityFailure:')
    journal.close()


def test_no_neural_pickle_or_restore_api():
    tree=ast.parse(Path(e.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            name=node.module if isinstance(node,ast.ImportFrom) else ' '.join(n.name for n in node.names)
            assert not any(x in name for x in ('torch','pickle','transformers'))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in ('_set_state','set_state','forward','backward','train','add_safe_globals')


def test_source_manifest_never_dispatch_authority(tmp_path):
    m=e.source_prepare(e.ROOT)
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        e.run(e.ROOT,m,path,'0'*64,tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()


def test_real_current_bridge_prepare_schema_no_invented_fields():
    _,_,base,bridge,_=e.dependencies(e.ROOT)
    # Actual bound bridge prepare, actual goal spec. Test module is only a
    # source-path schema fixture, NOT a claim of a real compiled CUDA receipt.
    goal=e.ROOT/'results/p5_native_goal_scene_20261027_v1_scene_main_v1/goal_spec.json'
    if not goal.exists():
        pytest.skip('main-generated independent goal asset unavailable')
    before=set(sys.modules)
    prepared=bridge.prepare(goal_spec=goal,compiler_source=Path(__file__),executor_source=Path(e.__file__))
    assert prepared['schema']==e.BRIDGE and prepared['status']=='ready_for_main_CUDA_compile'
    assert prepared['native_constructor']==base.CTOR
    source=prepared['native_environment_source']
    assert source==dict(path=str(bridge.ENV),sha256=e.sha(bridge.ENV))
    assert prepared['contract']==bridge.contract()
    assert prepared['runner_sources']['executor']==dict(path=str(Path(e.__file__).resolve()),sha256=e.sha(e.__file__))
    assert not {n.split('.')[0] for n in set(sys.modules)-before}&{'pymunk','gym','pygame','torch'}


def test_endpoint_failure_retains_all_terminal_reports_without_native_import(tmp_path,monkeypatch):
    panel,v2,base,bridge,seals=e.dependencies(e.ROOT)
    monkeypatch.setattr(e,'dependencies',lambda root:(panel,v2,base,bridge,seals))
    monkeypatch.setattr(e,'verify_ready',lambda root,m:None)
    class FixtureClass:
        def __init__(self):
            raise AssertionError('no fixture/native env may be constructed in orchestration test')
    source=Path(FixtureClass.__init__.__code__.co_filename).resolve()
    runtime=SimpleNamespace(ENV_REL=str(source),probe=lambda *args:dict(status='native_import_ready_physics_not_executed'))
    monkeypatch.setattr(base,'load_runtime',lambda root:(runtime,{}))
    monkeypatch.setattr(e,'import_bound',lambda *args:SimpleNamespace(PushTEnv=FixtureClass))
    monkeypatch.setitem(sys.modules,'pymunk',SimpleNamespace(fixture_only=True))
    expected={k:v//3 for k,v in bridge.contract()['execution']['expected'].items()}
    class FixtureCalls:
        def __init__(self,*args):
            self.counts=expected.copy();self.space_steps=[dict(dt=.01)]
        def start(self):
            return None
        def stop(self):
            return None
        def receipt(self):
            return dict(python_call_entries=self.counts,fixture_only=True)
    monkeypatch.setattr(base,'NativeCalls',FixtureCalls)
    attempts=[]
    def fake_capture(panel,base,writer,cls,branch,m,out,journal,calls,records,gates):
        attempts.append(branch['arm'])
        records.extend(dict(frame=i) for i in range(16));gates.extend(range(11))
        raw=out/branch['arm']/'frame_15';writer.mkdir(raw)
        writer.save_json(raw/'full_public_state.json',dict(fixture_only=True))
    monkeypatch.setattr(e,'collect_arm',fake_capture)
    def broken_metrics(public):
        raise ValueError('fixture endpoint diagnostic failed')
    monkeypatch.setattr(e,'object_metrics',broken_metrics)
    parent=tmp_path/'results'/e.NAME;parent.mkdir(parents=True)
    path=tmp_path/'manifest.json';path.write_text('{}')
    branches=[dict(arm=arm,prefix_env_commands=np.zeros((10,2)).tolist(),suffix_env_packet=np.zeros((5,2)).tolist()) for arm in e.ARMS]
    manifest=dict(contract=bridge.contract(),branches=branches,selected_packet=dict(packet=np.zeros((5,2)).tolist()),native_environment_source=dict(path=str(source),sha256=e.sha(source)))
    result=e.run(tmp_path,manifest,path,e.sha(path),parent/'fixture_run')
    assert attempts==list(e.ARMS)
    assert result['status']=='native_c1_three_branch_execution_incomplete'
    for row in result['rows']:
        arm=row['arm'];out=parent/'fixture_run'/arm
        saved=json.loads((out/'report.json').read_text())
        assert saved['status']=='guard_abort' and 'fixture endpoint' in saved['endpoint_error']
        assert json.loads((out/'journal.jsonl').read_text().splitlines()[-1])['event']['phase']=='terminal'
    assert (parent/'fixture_run/report.json').exists()
