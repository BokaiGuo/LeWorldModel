"""Pure memory/real saved arrays; no tmp, result, simulator or neural work."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p1_gn_fixedsegment_physical_20261027_v1 as m


def arguments():
    return dict(diagnostic_prepared=str(m.ROOT/'results/p1_gn_step_diagnostic_cuda_20261027_v1_prepare01.json'),
        diagnostic_prepare_sha='76a4248cd61d1d6e2a0056e6eb6cb9c41ef1b2b7fd5bd6ba07770a9ecbfa01ec',
        diagnostic_report=str(m.ROOT/'results/p1_gn_step_diagnostic_cuda_20261027_v1_cuda_main01/report.json'),
        diagnostic_report_sha='2636b3a89cbe96b97d64dce853f506a837e4191e05b90dd5b754146fde658848')


@pytest.fixture(scope='module')
def actual():return m.collect(**arguments())


@pytest.fixture(scope='module')
def bundle(actual):
    a=arguments();folder=Path(a['diagnostic_report']).parent;p=m.read(a['diagnostic_prepared'])
    return (p,m.read(a['diagnostic_report']),[m.read(folder/f'root_{i}.selection.json') for i in range(2)],
        [m.read(folder/f'root_{i}.result.json') for i in range(2)],
        [m.read(folder/f'eval_{i:02d}.json') for i in range(16)],
        [json.loads(line) for line in (folder/'events.jsonl').read_bytes().splitlines()],
        [m.diagnostic.read(root['task_path'])['task'] for root in p['roots']])


def test_actual_two_original_selected_plans(actual):
    assert len(actual['plans'])==2 and len({p['root_sha256'] for p in actual['plans']})==2
    assert [p['candidate_id'] for p in actual['plans']]==[0,1]
    for p in actual['plans']:
        assert p['arm']==m.ARM and p['selected_scale']==.25 and p['observed_equals_parent']
        assert len(p['commands'])==10
        assert p['selection_sha256']==m.source.digest({k:v for k,v in p.items() if k!='selection_sha256'})
        assert p['raw_source_sha256']==m.sha(p['raw_source_file'])
    assert actual['inherited']['diagnostic']=={k:dict(attempted=32,completed=32) for k in ('predict','action_encoder')}


@pytest.mark.parametrize('defect',['argmin','scale','commands','history','Q','entry','event','missing_trial','replay'])
def test_actual_frozen_selection_receipts_reject_tamper(bundle,defect):
    p,r,s,v,e,events,t=deepcopy(bundle)
    if defect=='argmin':s[0]['winner_index']=0
    elif defect=='scale':p['roots'][0]['candidates'][0]['scale']=.75
    elif defect=='commands':v[0]['verified_prediction']['commands'][0][0]+=.01
    elif defect=='history':e[0]['contexts'][1]['action_history'][0][0]+=.1
    elif defect=='Q':t[0]['projection'][0][0]=2
    elif defect=='entry':r['module_entries'][0]['completed']=False
    elif defect=='event':events.pop()
    elif defect=='missing_trial':e.pop(1)
    else:e[7]['contexts'][-1]['latent_history'][-1][0]+=.1
    with pytest.raises(ValueError):m.check_diagnostic(p,r,s,v,e,events,t)


def test_stable_zero_incumbent_tie_not_forced_new_scale():
    values=[dict(label=label,cost=1.) for label in ['incumbent','scale_0','scale_1','scale_2','scale_3']]
    assert m.diagnostic.select(values)==0


class MemoryStore:
    """No directory or PNG writes. Charge full raw RGB bytes conservatively."""
    def __init__(self,path):self.path=Path('/gn_memory_only');self.used=0;self.events=[];self.pngs=[];self.report=None
    def check(self,size=65536):assert self.used+size<=m.CAP
    def emit(self,kind,**values):
        event=dict(kind=kind,**deepcopy(values));self.check(len(m.source.canonical(event).encode()))
        self.used+=len(m.source.canonical(event).encode());self.events.append(event)
    def png(self,name,image):
        self.check(image.nbytes);self.used+=image.nbytes;self.pngs.append((name,image.copy()))
        return dict(path=str(self.path/name),file_sha256='f'*64,rgb_sha256=m.core.sha(image.tobytes()))
    def json(self,name,value):
        self.check(len(m.source.canonical(value).encode()));self.used+=len(m.source.canonical(value).encode());self.report=deepcopy(value)


class ArrayEnv:
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
    def eval_state(self,goal,state):
        d=state-goal;angle=min(abs(d[4]),2*np.pi-abs(d[4]))
        return bool(np.linalg.norm(d[:4])<20 and angle<np.pi/9),float(np.linalg.norm(d))
    def step(self,command):
        self.commands.append(command.copy());i=len(self.commands)
        if i==self.fail_step:raise RuntimeError('first_step_error')
        return dict(state=self.state.copy()),-1.,i==self.terminal,False,{}
    def close(self):self.closed=True


@pytest.mark.parametrize('mode',['complete','terminal','step_close_failure','bad_initial'])
def test_exact_original_branch_memory_only(actual,monkeypatch,mode):
    data=deepcopy(actual);sp=data['source_plan'];stores=[];envs=[]
    sp['source_closure'][str(Path(inspect.getsourcefile(ArrayEnv)).resolve())]='e'*64;sp['environment_sha256']='e'*64
    p=dict(arguments={},material_sha256=m.source.digest(data),inherited=data['inherited'])
    monkeypatch.setattr(m,'verify',lambda *a:p);monkeypatch.setattr(m,'collect',lambda **kw:data)
    monkeypatch.setattr(m,'sha',lambda *a:'f'*64) # virtual final journal only
    original_load=m.source.load_raw_source
    monkeypatch.setattr(m.source,'load_raw_source',lambda path,plan:original_load(path,actual['source_plan']))
    def sample(goal,shapes,slot,proposal,attempt,plan):
        raw=next(p['raw_source_file'] for p in data['plans'] if p['candidate_id']==slot['candidate_id'])
        state=original_load(raw,actual['source_plan'])['state'].copy()
        if mode=='bad_initial':state[0]+=1
        return state
    monkeypatch.setattr(m.source,'sample_state',sample)
    def store_factory(path):
        store=MemoryStore(path);stores.append(store);return store
    monkeypatch.setattr(m,'Store',store_factory)
    def factory():
        plan=data['plans'][len(envs)];packet=original_load(plan['raw_source_file'],actual['source_plan'])
        env=ArrayEnv(packet,terminal=3 if mode=='terminal' else None,fail_step=3 if mode=='step_close_failure' else None)
        if mode=='step_close_failure':
            def close():raise RuntimeError('second_close_error')
            env.close=close
        envs.append(env);return env
    r=m.run('memory','a'*64,'memory',factory,authorize_physics=True)
    assert len(envs)==2
    for plan,env in zip(data['plans'],envs):
        assert [s for s,_ in env.seeds]==[plan['source_metadata']['selected_attempt']['random_reset_seed'],sp['placement_reset_seed']]
        if env.commands:np.testing.assert_array_equal(env.commands,np.asarray(plan['commands'],np.float32)[:len(env.commands)])
    assert stores[0].used<m.CAP and stores[0].report is not None
    if mode=='complete':
        assert r['status']=='physical_calibration_recorded',r['errors']
        assert r['counts']=={k:dict(attempted=n,completed=n) for k,n in dict(reset=4,step=20,render=22).items()}
        assert len(stores[0].pngs)==6 and all(env.closed for env in envs)
        assert all(len(row['guard_receipt']['observations'])==11 for row in r['rows'])
        for row in r['rows']:
            assert [c['step'] for c in row['object_only']['checkpoints']]==[0,5,10]
            assert all(c['success'] is None and c['file_sha256'] for c in row['object_only']['checkpoints'])
            assert row['object_only']['final']['success'] is None
        assert r['internal_counts']==dict(reset_physics_completed_implied=8,reset_physics_attempted_upper_bound=8,
            goal_renders_completed_implied=4,actual_substeps=None)
    elif mode=='terminal':
        assert r['status']=='partial_or_failed' and all(len(env.commands)==3 for env in envs)
        assert all(row['status']=='environment_terminated' and not row['full_horizon'] for row in r['rows'])
    elif mode=='bad_initial':
        assert r['status']=='partial_or_failed' and r['counts']['step']['attempted']==0
        assert all(not row['initial_exact'] for row in r['rows'])
    else:
        assert r['status']=='partial_or_failed' and r['counts']['step']==dict(attempted=6,completed=4)
        assert all(row['first_error']['message']=='first_step_error' for row in r['rows'])
        assert all(row['secondary_errors'][-1]['message']=='second_close_error' for row in r['rows'])
    assert not r['P1_admission'] and not r['stall_certified'] and r['new_independent_samples']==0


def test_authorization_before_any_input_or_factory(monkeypatch):
    def forbidden(*a):raise AssertionError('must not run')
    monkeypatch.setattr(m,'verify',forbidden)
    with pytest.raises(ValueError,match='authorization'):m.run('none','a'*64,'none',forbidden)


def test_two_mib_total_and_floor_without_writes(monkeypatch):
    store=object.__new__(m.Store);store.path=Path('/memory');store.used=m.CAP-1
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:SimpleNamespace(free=m.FLOOR+m.CAP))
    with pytest.raises(ValueError,match='2MiB'):store.check(2)
    store.used=0;monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:SimpleNamespace(free=m.FLOOR))
    with pytest.raises(ValueError,match='floor'):store.check(1)


def test_failed_write_bytes_never_refunded(monkeypatch):
    store=object.__new__(m.Store);store.used=0
    file=SimpleNamespace(is_file=lambda:True,stat=lambda:SimpleNamespace(st_size=m.CAP-1))
    store.path=SimpleNamespace(exists=lambda:True,iterdir=lambda:[file])
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:SimpleNamespace(free=m.FLOOR+m.CAP))
    with pytest.raises(ValueError,match='2MiB'):store.check(2)


def test_low_start_disk_never_constructs_factory(actual,monkeypatch):
    p=dict(arguments={},material_sha256=m.source.digest(actual),inherited=actual['inherited'])
    monkeypatch.setattr(m,'verify',lambda *a:p);monkeypatch.setattr(m,'collect',lambda **kw:actual)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:SimpleNamespace(free=m.FLOOR+m.CAP-1))
    def forbidden():raise AssertionError('factory forbidden before disk gate')
    # No mkdir/open occurs: Store checks free space before creating anything.
    with pytest.raises(ValueError,match='reserve'):m.run('memory','a'*64,'/tmp/GN_FIXEDSEGMENT_GATE_MUST_NOT_CREATE',forbidden,authorize_physics=True)


def test_build_small_complete_new_seal_no_write(actual,monkeypatch):
    monkeypatch.setattr(m,'collect',lambda **kw:actual)
    p=m.build(**arguments());raw=m.source.canonical(p).encode()
    files=[Path(m.__file__),m.ROOT/'tests'/('test_'+m.NAME+'.py'),m.ROOT/'innerdocs'/(m.NAME+'.md')]
    assert len(raw)+sum(f.stat().st_size for f in files)<=100*1024
    assert p['material_sha256']==m.source.digest(actual) and p['schema']==m.NAME+'.prepare'
    assert p['plans']==actual['plans'] and p['contract']['resets']==4
