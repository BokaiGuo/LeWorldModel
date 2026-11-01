"""MAIN-run saved arrays and memory mocks only; no neural/physics/tmp writes."""
from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p1_line_search_gn_physical_20261027_v1 as m


def arguments():
    return dict(search_prepared=str(m.ROOT/'results/p1_line_search_gn_cuda_20261027_v1_prepare01.json'),
        search_prepare_sha='e1b6d9783da6a4e32d360326869fe1b64fef8c3f4f168095be4b7f8c215b26d9',
        search_report=str(m.ROOT/'results/p1_line_search_gn_cuda_20261027_v1_cuda_main01/report.json'),
        search_report_sha='5f9825f98ed40f637c9d48e49c7c53428d1380e4234e3a1e5503a76f74c4bf1e')


@pytest.fixture(scope='module')
def actual():return m.collect(**arguments())


@pytest.fixture(scope='module')
def component():
    a=arguments();p=m.read(a['search_prepared']);folder=Path(a['search_report']).parent
    return (m.search.read(p['roots'][0]['task_path'])['task'],m.b8.gz(folder/'root_0.json.gz'),
        [e for e in m.b8.gz(folder/'events.jsonl.gz',lines=True) if e['root_index']==0])


def test_actual_fixed_two_branches(actual):
    assert len(actual['plans'])==2 and len({p['root_sha256'] for p in actual['plans']})==2
    for plan in actual['plans']:
        assert plan['arm']==m.ARM and plan['observed_equals_parent']
        assert plan['selection_sha256']==m.source.digest({k:v for k,v in plan.items() if k!='selection_sha256'})
        winner=m.b8.gz(plan['origin'])['final_prediction']
        assert plan['commands']==winner['commands'] and plan['prediction_sha256']==m.source.digest(winner)
    assert actual['inherited']['line_search']=={k:dict(attempted=2*v,completed=2*v) for k,v in m.search.budget().items()}


@pytest.mark.parametrize('defect',['selection','final_raw','missing_trial','delta','endpoint','raw_J','gradient','history','Q','completion','replay'])
def test_component_rejects_tampering(component,defect):
    task,row,events=deepcopy(component)
    def event(kind):return next(e for e in events if e['kind']==kind)
    if defect=='selection':row['steps'][0]['selected_index']=99
    elif defect=='final_raw':row['final_prediction']['commands'][0][0]+=.01
    elif defect=='missing_trial':events.remove(event('evaluation_complete'))
    elif defect=='delta':event('line_proposals_frozen')['delta'][0]+=.1
    elif defect=='endpoint':event('line_proposals_frozen')['endpoint'][0][0]+=.1
    elif defect=='raw_J':event('jacobian_row')['raw_control_derivative'][0]+=.1
    elif defect=='gradient':event('derivative_complete')['derivative']['objective_gradient'][0]+=.1
    elif defect=='history':event('native_packet')['context']['action_history'][0][0]+=.1
    elif defect=='Q':task['projection'][0][0]=2
    elif defect=='completion':events.remove(event('operation_completed'))
    else:event('update_verified')['prediction']['cost']+=.1
    with pytest.raises(ValueError):m.check_component(task,row,events)


def test_stable_zero_candidate_retained():
    assert m.search.select(dict(cost=1.),[dict(cost=1.) for _ in range(5)])==0
    with pytest.raises(ValueError):m.search.select(dict(cost=1.),[dict(cost=0.) for _ in range(4)])


class MemoryStore:
    def __init__(self,path):self.path=Path('/line_search_memory_only');self.used=0;self.events=[];self.pngs=[];self.report=None
    def check(self,size=65536):assert self.used+size<=m.CAP
    def emit(self,kind,**values):
        event=dict(kind=kind,**deepcopy(values));size=len(m.source.canonical(event).encode())
        self.check(size);self.used+=size;self.events.append(event)
    def png(self,name,image):
        self.check(image.nbytes);self.used+=image.nbytes;self.pngs.append((name,image.copy()))
        return dict(path=str(self.path/name),file_sha256='f'*64,rgb_sha256=m.core.sha(image.tobytes()))
    def json(self,name,value):
        size=len(m.source.canonical(value).encode());self.check(size);self.used+=size;self.report=deepcopy(value)


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
def test_two_branch_memory_execution(actual,monkeypatch,mode):
    data=deepcopy(actual);sp=data['source_plan'];stores=[];envs=[]
    sp['source_closure'][str(Path(inspect.getsourcefile(ArrayEnv)).resolve())]='e'*64;sp['environment_sha256']='e'*64
    p=dict(arguments={},material_sha256=m.source.digest(data),inherited=data['inherited'])
    monkeypatch.setattr(m,'verify',lambda *a:p);monkeypatch.setattr(m,'collect',lambda **kw:data)
    monkeypatch.setattr(m,'sha',lambda *a:'f'*64)
    load=m.source.load_raw_source
    monkeypatch.setattr(m.source,'load_raw_source',lambda path,plan:load(path,actual['source_plan']))
    def sample(goal,shapes,slot,proposal,attempt,plan):
        raw=next(p['raw_source_file'] for p in data['plans'] if p['candidate_id']==slot['candidate_id'])
        state=load(raw,actual['source_plan'])['state'].copy()
        if mode=='bad_initial':state[0]+=1
        return state
    monkeypatch.setattr(m.source,'sample_state',sample)
    def store_factory(path):
        store=MemoryStore(path);stores.append(store);return store
    monkeypatch.setattr(m,'Store',store_factory)
    def factory():
        plan=data['plans'][len(envs)];packet=load(plan['raw_source_file'],actual['source_plan'])
        env=ArrayEnv(packet,terminal=3 if mode=='terminal' else None,fail_step=3 if mode=='step_close_failure' else None)
        if mode=='step_close_failure':
            def close():raise RuntimeError('second_close_error')
            env.close=close
        envs.append(env);return env
    r=m.run('memory','a'*64,'memory',factory,authorize_physics=True)
    assert len(envs)==2 and stores[0].report is not None and stores[0].used<m.CAP
    for plan,env in zip(data['plans'],envs):
        assert [s for s,_ in env.seeds]==[plan['source_metadata']['selected_attempt']['random_reset_seed'],sp['placement_reset_seed']]
        if env.commands:np.testing.assert_array_equal(env.commands,np.asarray(plan['commands'],np.float32)[:len(env.commands)])
    if mode=='complete':
        assert r['status']=='physical_calibration_recorded',r['errors']
        assert r['counts']=={k:dict(attempted=n,completed=n) for k,n in dict(reset=4,step=20,render=22).items()}
        assert len(stores[0].pngs)==6 and all(e.closed for e in envs)
        for row in r['rows']:
            assert len(row['guard_receipt']['observations'])==11
            assert [x['step'] for x in row['object_only']['checkpoints']]==[0,5,10]
            assert row['object_only']['final']['success'] is None
    elif mode=='terminal':
        assert r['status']=='partial_or_failed' and all(len(e.commands)==3 for e in envs)
        assert all(row['status']=='environment_terminated' and not row['full_horizon'] for row in r['rows'])
    elif mode=='bad_initial':
        assert r['status']=='partial_or_failed' and r['counts']['step']['attempted']==0
    else:
        assert r['status']=='partial_or_failed' and r['counts']['step']==dict(attempted=6,completed=4)
        assert all(row['first_error']['message']=='first_step_error' for row in r['rows'])
        assert all(row['secondary_errors'][-1]['message']=='second_close_error' for row in r['rows'])
    assert not r['P1_admission'] and not r['stall_certified'] and r['new_independent_samples']==0


def test_authorization_first(monkeypatch):
    def forbidden(*a):raise AssertionError('must not execute')
    monkeypatch.setattr(m,'verify',forbidden)
    with pytest.raises(ValueError,match='authorization'):m.run('none','a'*64,'none',forbidden)


def test_quota_floor_no_writes(monkeypatch):
    store=object.__new__(m.Store);store.path=Path('/memory');store.used=m.CAP-1
    monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:SimpleNamespace(free=m.FLOOR+m.CAP))
    with pytest.raises(ValueError,match='2MiB'):store.check(2)
    store.used=0;monkeypatch.setattr(m.shutil,'disk_usage',lambda *a:SimpleNamespace(free=m.FLOOR))
    with pytest.raises(ValueError,match='floor'):store.check(1)


def test_build_small_new_seal_no_write(actual,monkeypatch):
    monkeypatch.setattr(m,'collect',lambda **kw:actual)
    p=m.build(**arguments())
    files=[Path(m.__file__),m.ROOT/'tests'/('test_'+m.NAME+'.py'),m.ROOT/'innerdocs'/(m.NAME+'.md')]
    assert len(m.source.canonical(p).encode())+1+sum(f.stat().st_size for f in files)<=m.SMALL
    assert p['plans']==actual['plans'] and p['schema']==m.NAME+'.prepare'
    assert p['material_sha256']==m.source.digest(actual) and p['contract']['public_steps']==20
