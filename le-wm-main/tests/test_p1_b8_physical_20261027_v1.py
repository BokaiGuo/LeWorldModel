"""Pure memory/saved arrays. No tmp_path, new results, model or simulator."""
from copy import deepcopy
import gzip
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p1_b8_physical_20261027_v1 as m


def arguments():
    return dict(direct_prepared=str(m.ROOT/'results/p1_observed_direct_cuda_20261027_v2_prepare01.json'),
        direct_prepare_sha='5217f0e2ccab0da8d270383885aec86b61a85788bf5b88f863ffe79ef8cb9da9',
        direct_report=str(m.ROOT/'results/p1_observed_direct_cuda_20261027_v2_cuda_main01/report.json'),
        direct_report_sha='954af696d88eecebdcdc7345de702809517491967263f395a8f2819753d889d5')


@pytest.fixture(scope='module')
def actual():
    if not Path(arguments()['direct_report']).is_file():pytest.skip('saved B8 evidence absent')
    return m.collect(**arguments())


def test_actual_four_full_J_trajectories_same_source(actual):
    assert len(actual['plans'])==4 and len({p['root_sha256'] for p in actual['plans']})==2
    assert [p['arm'] for p in actual['plans']]==list(m.ARMS)*2
    assert actual['inherited_direct']['jacobian']==dict(attempted=32,completed=32)
    assert actual['inherited_direct']['predict']==dict(attempted=280,completed=280)
    for p in actual['plans']:
        report=m.gz(p['origin']);winner=report['direct_runs'][0][-1]['incumbent']
        assert p['commands']==winner['commands'] and p['prediction_sha256']==m.source.digest(winner)
        assert p['observed_equals_parent'] and p['history_semantics']=='observed_not_padding_even_when_numerically_identical'
        assert p['search_cost']['jacobian']==dict(attempted=8,completed=8)


@pytest.mark.parametrize('defect',['rawJ','J','trajectory','commands','resources','goal','Q'])
def test_actual_complete_evidence_tamper_rejected(actual,defect):
    plan=actual['plans'][0];report=m.gz(plan['origin']);task=m.gz(plan['task_file'])['task']
    events=m.gz(Path(plan['origin']).parent/'direct_journal.jsonl.gz',lines=True)
    if defect=='rawJ':next(e for e in events if e['kind']=='jacobian_row')['raw_control_derivative'][0]+=.1
    elif defect=='J':report['direct_runs'][0][1]['derivative']['raw_terminal_jacobian'][0][0]+=.1
    elif defect=='trajectory':report['direct_runs'][0].pop(4)
    elif defect=='commands':report['direct_runs'][0][-1]['incumbent']['commands'][0][0]+=.01
    elif defect=='resources':report['extended_journal'].pop()
    elif defect=='goal':task['goal'][0]+=.01
    else:task['projection'][0][0]=2
    with pytest.raises(ValueError):m.component(task,plan['arm'],report,events)


@pytest.mark.parametrize('defect',['rgb','goal','state','raw'])
def test_nonstationary_observed_bridge_refused(actual,defect):
    plan=actual['plans'][0];raw=m.source.load_raw_source(plan['raw_source_file'],actual['source_plan'])
    packet=dict(state=raw['state'].copy(),goal=raw['goal'].copy(),goal_rgb=raw['goal_rgb'].copy(),
        rgb_history=raw['rgb_history'].copy(),raw_action_history=raw['raw_action_history'].copy(),
        observed_states=np.stack([raw['state']]*3),metadata=dict(root_sha256=plan['root_sha256'],parent_root_sha256=plan['root_sha256']))
    m.stationary_parent(packet,plan['raw_source_file'])
    if defect=='rgb':packet['rgb_history'][1,0,0,0]^=1
    elif defect=='goal':packet['goal'][0]+=.1
    elif defect=='state':packet['observed_states'][1,0]+=.1
    else:packet['raw_action_history'][0,0]+=.1
    with pytest.raises(ValueError,match='stationary'):m.stationary_parent(packet,plan['raw_source_file'])


@pytest.mark.parametrize('defect',['partial','extra','oversize'])
def test_gzip_strict_memory_only(defect):
    raw=gzip.compress(b'{}\n')
    if defect=='partial':raw=raw[:-4]
    elif defect=='extra':raw+=raw
    with pytest.raises(ValueError):m.decode_gzip(raw,limit=1 if defect=='oversize' else 1024)


def test_unauthorized_never_verifies_or_constructs(monkeypatch):
    def forbidden(*a):raise AssertionError('must not be called')
    monkeypatch.setattr(m,'verify',forbidden)
    with pytest.raises(ValueError,match='authorization'):m.run('none','a'*64,'none',forbidden)


def test_four_mib_and_reserve_memory_only(monkeypatch):
    store=object.__new__(m.Store);store.used=m.CAP-1;store.path=Path('/nonexistent_memory_only')
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:SimpleNamespace(free=m.FLOOR+m.CAP))
    with pytest.raises(ValueError,match='4MiB'):store.check(2)
    store.used=0
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:SimpleNamespace(free=m.FLOOR))
    with pytest.raises(ValueError,match='floor'):store.check(1)


class MemoryStore:
    """No actual directory, PNG encoding or byte writes."""
    def __init__(self,path):self.path=Path('/b8_memory_only');self.events=[];self.pngs=[];self.report=None
    def check(self,*a):return None
    def emit(self,kind,**v):self.events.append(dict(kind=kind,**deepcopy(v)))
    def png(self,name,image):
        self.pngs.append((name,image.copy()))
        return dict(path=str(self.path/name),file_sha256='f'*64,rgb_sha256=m.core.sha(image.tobytes()))
    def json(self,name,value):self.report=deepcopy(value)


class ArrayEnv:
    """No gym/Pymunk: returns saved state arrays, records raw dispatch only."""
    relative=True;action_scale=100;render_mode='rgb_array';render_size=224
    def __init__(self,packet,terminal=None,fail_step=None):
        self.unwrapped=self;self.packet=packet;self.state=packet['state'].copy();self.goal=packet['goal'].copy()
        self.block=SimpleNamespace(shapes=[]);self.seeds=[];self.commands=[];self.closed=False
        self.terminal=terminal;self.fail_step=fail_step
    def reset(self,seed,options=None):
        self.seeds.append((seed,deepcopy(options)))
        if options is not None:self.state=options['state'].copy();self.goal=options['goal_state'].copy()
        return dict(state=self.state.copy()),dict(goal_state=self.goal.copy(),goal=self.packet['goal_rgb'].copy())
    def render(self):return self.packet['rgb_history'][0].copy()
    def _get_obs(self):return self.state.copy()
    def eval_state(self,goal,state):return False,float(np.linalg.norm(state-goal))
    def step(self,command):
        self.commands.append(command.copy());i=len(self.commands)
        if i==self.fail_step:raise RuntimeError('mock_step_first')
        return dict(state=self.state.copy()),-1.,i==self.terminal,False,{}
    def close(self):self.closed=True


@pytest.mark.parametrize('mode',['complete','terminal','failure_close'])
def test_real_branch_recipe_guard_and_four_arm_census_in_memory(actual,monkeypatch,mode):
    data=deepcopy(actual);sourceplan=data['source_plan'];envs=[];stores=[]
    sourceplan['source_closure'][str(Path(inspect.getsourcefile(ArrayEnv)).resolve())]='e'*64
    sourceplan['environment_sha256']='e'*64
    prepared=dict(inputs=data)
    monkeypatch.setattr(m,'verify',lambda *a:prepared)
    def store_factory(path):
        store=MemoryStore(path);stores.append(store);return store
    monkeypatch.setattr(m,'Store',store_factory)
    # Only final virtual journal hash is mocked. Raw source hashes still use
    # sealed physical.sha directly and are checked against actual parent files.
    monkeypatch.setattr(m,'sha',lambda *a:'f'*64)
    # The authenticated constructor is replaced by an array fake. Its local
    # closure is necessarily not the real source plan; read real parent arrays
    # against the ORIGINAL plan, not that deliberately synthetic test closure.
    original_load=m.source.load_raw_source
    monkeypatch.setattr(m.source,'load_raw_source',lambda path,plan:original_load(path,actual['source_plan']))
    def sample(goal,shapes,slot,proposal,attempt,plan):
        p=next(p for p in data['plans'] if p['candidate_id']==slot['candidate_id'])
        return m.source.load_raw_source(p['raw_source_file'],actual['source_plan'])['state'].copy()
    monkeypatch.setattr(m.source,'sample_state',sample)
    def factory():
        plan=data['plans'][len(envs)];packet=m.source.load_raw_source(plan['raw_source_file'],actual['source_plan'])
        env=ArrayEnv(packet,terminal=3 if mode=='terminal' else None,fail_step=3 if mode=='failure_close' else None)
        if mode=='failure_close':
            def close():raise RuntimeError('mock_close_second')
            env.close=close
        envs.append(env);return env
    r=m.run('memory','a'*64,'memory',factory,authorize_physics=True)
    assert len(envs)==4
    for plan,env in zip(data['plans'],envs):
        assert env.seeds[0][0]==plan['source_metadata']['selected_attempt']['random_reset_seed']
        assert env.seeds[1][0]==sourceplan['placement_reset_seed']
        np.testing.assert_array_equal(env.commands,np.asarray(plan['commands'],np.float32)[:len(env.commands)])
    if mode=='complete':
        assert r['status']=='physical_calibration_recorded',r['errors']
        assert r['counts']=={k:dict(attempted=n,completed=n) for k,n in dict(reset=8,step=40,render=44).items()}
        assert len(stores[0].pngs)==12 and all(env.closed for env in envs)
        assert all(len(row['guard_receipt']['observations'])==11 for row in r['rows'])
    elif mode=='terminal':
        assert r['status']=='partial_or_failed' and all(len(env.commands)==3 for env in envs)
        assert all(row['status']=='environment_terminated' and not row['full_horizon'] for row in r['rows'])
    else:
        assert r['status']=='partial_or_failed' and r['counts']['step']==dict(attempted=12,completed=8)
        assert all(row['first_error']['message']=='mock_step_first' for row in r['rows'])
        assert all(row['secondary_errors'][-1]['message']=='mock_close_second' for row in r['rows'])
    assert not r['P1_admission'] and not r['stall_certified'] and r['new_independent_samples']==0


def test_small_build_not_written(actual,monkeypatch):
    monkeypatch.setattr(m,'collect',lambda **kw:actual)
    p=m.build(**arguments());assert len(m.source.canonical(p).encode())<=m.SMALL
    assert p['contract']['output_cap_bytes']==4*1024**2 and not p['contract']['stall_certified']
