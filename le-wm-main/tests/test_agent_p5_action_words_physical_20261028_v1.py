"""RAM routing/receipts only: no native physics, neural imports, or science evidence."""
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_action_words_physical_20261028_v1 as m


def module(name):
    path = m.ROOT/'experiments/spectral_reachability'/f'{name}.py'
    spec = importlib.util.spec_from_file_location('ram_'+name,path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


PILOT = module('p5_jepa_factual_pilot_20261027_v1')
PANEL = module(m.PANEL)
NONZERO = module('p5_jepa_factual_nonzero_20261027_v2')
PRODUCER = module(m.PRODUCER)


class Body:
    def __init__(self):
        self.position = np.zeros(2,np.float64)
        self.velocity = np.zeros(2,np.float64)


class Space:
    def __init__(self):
        self.ticks = []

    def step(self, dt):
        # Record-only RAM token; NOT a physical integrator.
        self.ticks.append(dt)


class Env:
    """Analytic token producer, deliberately not the installed environment."""
    instances = []
    fail_frame = None
    early_frame = None
    close_fails = False
    reset_mutation = False
    calls_started = 0

    def __init__(self, **ctor):
        assert ctor == PILOT.CTOR and ctor['reset_to_state'] is None
        self.seed()
        self.frame = 0
        self.closed = False
        self.instances.append(self)

    def seed(self, value=None):
        self.seed_value = value

    def _setup(self):
        self.space = Space()
        self.agent, self.block, self.static = Body(), Body(), Body()
        self.agent.position[:] = [11.,13.]
        self.coverage = []
        self.latest = None

    def _set_state(self):
        self.space.step(.01)

    def _get_goal_pose_body(self):
        return Body()

    def _render_frame(self):
        self._get_goal_pose_body()
        rgb = np.zeros((224,224,3),np.uint8)
        rgb[0,0,0] = self.frame
        return rgb

    def _get_obs(self):
        return np.asarray([*self.agent.position,0.,0.,0.,*self.agent.velocity],np.float32)

    def obs(self):
        state = self._get_obs()
        return dict(visual=self._render_frame(),proprio=state[[0,1,5,6]]),state

    def reset(self):
        self._setup()
        self._set_state()
        if self.reset_mutation:
            self.agent.position[0] += 1
        return self.obs()

    def step(self, u):
        self.frame += 1
        if self.frame == self.fail_frame:
            raise RuntimeError('RAM injected native attempt failure')
        self.latest = (self.agent.position+100*u).tolist()
        # Token update, no dynamics or native calls; checks action routing only.
        self.agent.position = self.agent.position+100*u
        for _ in range(10):
            self.space.step(.01)
        self._get_goal_pose_body()
        self.coverage.append(0.)
        obs,state = self.obs()
        info = dict(state=state, pos_agent=self.agent.position.copy(),
                    vel_agent=self.agent.velocity.copy(), block_pose=np.zeros(3,np.float64),
                    goal_pose=np.array([256.,256.,np.pi/4]), n_contacts=0,
                    max_coverage=0.,final_coverage=0.)
        return obs,0.,self.frame == self.early_frame,info

    def close(self):
        if self.close_fails:
            raise RuntimeError('RAM close failure')
        self.closed = True


def public(env):
    return dict(bodies=dict(agent=dict(position=env.agent.position.tolist(),velocity=env.agent.velocity.tolist()),
                block=dict(position=[0.,0.],angle=0.)),
                env=dict(seed=env.seed_value,goal_pose=dict(dtype='<f8',shape=[3],values=[256.,256.,float(np.pi/4)]),
                         coverage_arr=env.coverage.copy(),latest_action=env.latest,
                         n_contact_points=0,success_threshold=.95))


def base():
    # Facade, never mutate any frozen module globals or native implementation.
    return SimpleNamespace(**{k:getattr(PILOT,k) for k in
        ('CTOR','canonical','mkdir','save_json','save_array','Journal','operation','NativeCalls','numeric_json')},
        public_state=public,rng_state=lambda env:dict(seed=env.seed_value))


@pytest.fixture(autouse=True)
def reset_fixture():
    Env.instances = []
    Env.fail_frame = Env.early_frame = None
    Env.reset_mutation = Env.close_fails = False
    yield
    assert sys.getprofile() is None


def fixture_manifest(tmp_path):
    b = base()
    seals,roots = {},[]
    prefix = np.asarray(NONZERO.SCHEDULE,np.float64).reshape(15,2)[:10]
    original = tmp_path/'original'
    original.mkdir()
    for seed in m.SEEDS:
        factual = original/f'seed_{seed}'
        factual.mkdir()
        dest = factual/'a'
        dest.mkdir()
        writer = m.Writer(b,tmp_path)
        env = Env(**b.CTOR)
        env.seed(seed)
        b.save_json(dest/'pre_reset_rng.json',b.rng_state(env))
        obs,state = env.reset()
        record = PANEL.capture(b,writer,env,obs,state,dict(kind='reset'),dest/'frame_00')
        for frame,u in enumerate(prefix,1):
            cmd = m.command_for(frame,u,np.asarray(env.agent.position,np.float64),record['files']['full_public_state.json']['sha256'])
            b.save_json(dest/f'command_{frame:02d}.json',cmd)
            obs,reward,done,info = env.step(u.copy())
            record = PANEL.capture(b,writer,env,obs,info['state'],dict(kind='step',reward=reward,done=done,info=info),dest/f'frame_{frame:02d}')
        env.close()
        for p in dest.rglob('*'):
            if p.is_file():
                seals[str(p)] = m.sha(p)
        roots.append(dict(seed=seed,factual=str(factual),root_sha256=m.digest(dict(seed=seed)),
            goal_source=dict(path=str(dest/'frame_10/full_public_state.json'),sha256=m.sha(dest/'frame_10/full_public_state.json')),
            plans={k:dict(raw=v.tolist()) for k,v in PRODUCER.programs().items()}))
    pm = dict(roots=roots)
    manifest = dict(schema=m.TAG,policy=m.policy(),branches=m.branches_from(pm,b,NONZERO.SCHEDULE,seals))
    Env.instances = []
    return manifest,b,pm,seals


def run_fixture(tmp_path, mutation=None):
    manifest,b,pm,seals = fixture_manifest(tmp_path)
    if mutation:
        mutation(manifest)
    out = tmp_path/'run'
    def load():
        return Env,lambda:b.NativeCalls(Env,SimpleNamespace(Space=Space,Body=Body)),dict(status='RAM_only_not_physics')
    report = m.execute(manifest,'a'*64,out,PANEL,b,load,lambda:None)
    return manifest,report,out


@pytest.fixture(scope='module')
def complete(tmp_path_factory):
    # One complete 12-branch RAM layout for repeated saved mutation cases.
    return run_fixture(tmp_path_factory.mktemp('all12_words'))


def test_full12_order_and_meter(complete):
    manifest,r,out = complete
    assert r['status']=='all12_native_replays_recorded'
    assert [(x['seed'],x['word']) for x in r['rows']]==[(s,w) for s in m.SEEDS for w in m.WORDS]
    assert len(r['rows'])==12
    for row in r['rows']:
        assert row['prefix_gates']==list(range(11))
        assert row['actual_suffix_frames']==10
        assert row['meter']['python_call_entries']==m.EXPECTED
        assert row['operation_meter']==dict(durable_intents=dict(construct=1,seed=1,reset=1,step=20,close=1),
                                           durable_returns=dict(construct=1,seed=1,reset=1,step=20,close=1))
        assert row['last_native_metrics']['task_success'] is None
    consumed = m.validate_saved(manifest,'a'*64,out/'report.json',m.sha(out/'report.json'))
    assert len(consumed['verified_cases'])==12 and not consumed['scientific_admission']


def test_full_size_regression(complete):
    _,r,out = complete
    records = m.inventory(out)
    size = sum(x['bytes'] for x in records.values())+(out/'report.json').stat().st_size
    nodes = 1+len(list(out.rglob('*')))
    assert size+8192*nodes < m.CAP
    assert nodes < m.NODES-64
    assert size < m.PAYLOAD-m.RESERVE
    assert len([p for p in records if p.endswith('rgb.npy')])==252


@pytest.mark.parametrize('field',['rgb.npy','state.npy','proprio.npy','full_public_state.json','transition.json'])
def test_prefix_byte_gate_each_field(tmp_path,field):
    d = tmp_path/'actual'
    d.mkdir()
    reference = tmp_path/'reference'
    reference.mkdir()
    expected = dict(frame=10,files={})
    for name in m.NAMES:
        (d/name).write_bytes(b'original')
        (reference/name).write_bytes(b'original')
        expected['files'][name]=dict(path=str(reference/name),sha256=m.sha(reference/name),bytes=8)
    (d/field).write_bytes(b'mutation')
    with pytest.raises(m.IntegrityFailure,match='prefix byte mismatch'):
        m.prefix_gate(d,expected)


def test_prefix_mutation_stops_before_suffix(tmp_path):
    def mutation(manifest):
        Env.reset_mutation = True
    _,r,_ = run_fixture(tmp_path,mutation)
    assert r['rows'][0]['status']=='integrity_failure'
    assert len(Env.instances)==1 and Env.instances[0].closed
    assert r['rows'][0]['meter']['python_call_entries'].get('control_step',0)==0
    assert all(x['status']=='not_run' for x in r['rows'][1:])
    assert len(r['rows'])==12


def test_failed_attempt_no_refund_or_padding(tmp_path):
    _,r,_ = run_fixture(tmp_path,lambda manifest:setattr(Env,'fail_frame',12))
    row = r['rows'][0]
    assert row['status']=='engineering_failure'
    assert row['operation_meter']['durable_intents']['step']==12
    assert row['operation_meter']['durable_returns']['step']==11
    assert row['actual_suffix_frames']==1
    assert row['meter']['python_call_entries']['control_step']==12
    assert row['close']['returned'] and row['close']['original_error'] is not None
    assert len(r['rows'])==12 and all(x['status']=='not_run' for x in r['rows'][1:])


def test_close_failure_blocks_later_and_stops_profiler(tmp_path):
    _,r,_ = run_fixture(tmp_path,lambda manifest:setattr(Env,'close_fails',True))
    assert r['rows'][0]['close']['status']=='failed'
    assert r['rows'][0]['close']['attempted'] and not r['rows'][0]['close']['returned']
    assert len(Env.instances)==1
    assert all(x['status']=='not_run' for x in r['rows'][1:])


def test_early_stop_retained_no_extra_steps(tmp_path):
    _,r,_ = run_fixture(tmp_path,lambda manifest:setattr(Env,'early_frame',12))
    assert len(r['rows'])==12
    assert all(x['status']=='native_early_stop' and x['actual_suffix_frames']==2 for x in r['rows'])
    assert all(x.frame==12 and x.closed for x in Env.instances)


@pytest.mark.parametrize('change', ['missing_word','seed','raw_shape','goal','command','closure'])
def test_assembler_fails_closed(tmp_path,change):
    _,b,pm,seals = fixture_manifest(tmp_path)
    if change=='missing_word':
        del pm['roots'][0]['plans']['yx']
    elif change=='seed':
        pm['roots'][0]['seed'] += 1
    elif change=='raw_shape':
        pm['roots'][0]['plans']['xy']['raw'] = [[0.,0.]]
    elif change=='goal':
        pm['roots'][0]['goal_source']['sha256']='0'*64
    elif change=='command':
        p = Path(pm['roots'][0]['factual'])/'a/command_01.json'
        c = m.read_json(p)
        c['raw_env_argument']=[0.,0.]
        p.write_bytes(m.canonical(c)+b'\n')
        seals[str(p)]=m.sha(p)
    else:
        seals.clear()
    with pytest.raises(ValueError):
        m.branches_from(pm,b,NONZERO.SCHEDULE,seals)


def test_floor_wall_reserve_and_failed_write_no_refund(tmp_path):
    b = base()
    writer = m.Writer(b,tmp_path,clock=lambda:0.,free=lambda:0)
    with pytest.raises(ValueError,match='floor'):
        writer.charge(1)
    writer = m.Writer(b,tmp_path,clock=lambda:0.,free=lambda:m.FLOOR)
    writer.clock = lambda:m.WALL
    with pytest.raises(ValueError,match='wall'):
        writer.charge(1)
    writer.charge(1,final=True)
    assert writer.payload==1
    writer = m.Writer(b,tmp_path)
    writer.payload = m.PAYLOAD-m.RESERVE
    with pytest.raises(ValueError,match='payload'):
        writer.charge(1)
    writer.charge(1,final=True)
    p = tmp_path/'exists.json'
    b.save_json(p,{})
    writer = m.Writer(b,tmp_path)
    with pytest.raises(FileExistsError):
        writer.save_json(p,dict(charged=True))
    assert writer.payload>0 and writer.nodes==1


def test_json_strict_and_no_symlink(tmp_path):
    p = tmp_path/'d.json'
    p.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError,match='duplicate'):
        m.read_json(p)
    p.write_text('{"x":NaN}')
    with pytest.raises(ValueError,match='nonfinite'):
        m.read_json(p)
    link = tmp_path/'link'
    link.symlink_to(p)
    with pytest.raises(ValueError,match='nonsymlink'):
        m.sha(link)


def copy_frame(tmp_path,complete,frame=11):
    _,r,out = complete
    src = out/r['rows'][0]['case']/f'frame_{frame:02d}'
    dest = tmp_path/'frame'
    dest.mkdir()
    for p in src.iterdir():
        (dest/p.name).write_bytes(p.read_bytes())
    return dest,copy.deepcopy(r['rows'][0]),m.read_json(src/'full_public_state.json')['env']['goal_pose']


@pytest.mark.parametrize('change',['done_true','done_int','goal','state7','reward','coverage','proprio','rgb'])
def test_native_frame_mutations(tmp_path,complete,change):
    dest,_,goal = copy_frame(tmp_path,complete)
    t = m.read_json(dest/'transition.json')
    p = m.read_json(dest/'full_public_state.json')
    if change=='done_true':
        t['done']=True
    elif change=='done_int':
        t['done']=0
    elif change=='goal':
        p['env']['goal_pose']['values'][0]+=1
    elif change=='state7':
        p['bodies']['agent']['velocity'][0]+=1
    elif change=='reward':
        t['reward']=1
    elif change=='coverage':
        p['env']['coverage_arr'].pop()
    else:
        name = 'proprio' if change=='proprio' else 'rgb'
        a = np.load(dest/(name+'.npy'),allow_pickle=False)
        a = a+1 if name=='proprio' else a.astype(np.float32)
        with (dest/(name+'.npy')).open('wb') as f:
            np.save(f,a,allow_pickle=False)
    (dest/'transition.json').write_bytes(m.canonical(t)+b'\n')
    (dest/'full_public_state.json').write_bytes(m.canonical(p)+b'\n')
    with pytest.raises(ValueError):
        m.validate_frame(dest,11,goal)


def test_native_done_false_even_above_threshold(tmp_path,complete):
    dest,_,goal = copy_frame(tmp_path,complete)
    p = m.read_json(dest/'full_public_state.json')
    p['env']['coverage_arr'][-1] = 1.
    t = m.read_json(dest/'transition.json')
    t['info']['final_coverage']=t['info']['max_coverage']=t['reward']=1.
    t['done']=False
    (dest/'transition.json').write_bytes(m.canonical(t)+b'\n')
    (dest/'full_public_state.json').write_bytes(m.canonical(p)+b'\n')
    m.validate_frame(dest,11,goal)
    assert m.native_metrics(p)['task_success'] is None
    t['done']=True
    (dest/'transition.json').write_bytes(m.canonical(t)+b'\n')
    with pytest.raises(ValueError,match='done=False'):
        m.validate_frame(dest,11,goal)


@pytest.mark.parametrize('field',['report_sha','manifest_sha','word','denominator','promotion','raw'])
def test_saved_contract_mutations(tmp_path,complete,field):
    manifest,r,out = complete
    changed = copy.deepcopy(r)
    man = copy.deepcopy(manifest)
    if field=='report_sha':
        with pytest.raises(ValueError,match='SHA'):
            m.validate_saved(man,'a'*64,out/'report.json','0'*64)
        return
    if field=='manifest_sha':
        changed['manifest_sha256']='0'*64
    elif field=='word':
        changed['rows'][0]['word']='winner'
    elif field=='denominator':
        changed['rows'].pop()
    elif field=='promotion':
        changed['scientific_admission']=True
    else:
        man['branches'][0]['suffix'][0][0]+=.1
    # Separate report outside original immutable RAM layout; validation must fail.
    p = tmp_path/'report.json'
    p.write_bytes(m.canonical(changed)+b'\n')
    with pytest.raises(ValueError):
        m.validate_saved(man,'a'*64,p,m.sha(p))


def test_actual_native_source_contract_read_only():
    p = m.ROOT/'external/jepa-wms/evals/simu_env_planning/envs/pusht_env/pusht_env.py'
    text = p.read_text()
    assert 'done = False  # coverage > self.success_threshold' in text
    assert 'tuple(self.agent.velocity)' in text
    assert PILOT.CTOR['with_target'] is True
    assert PILOT.CTOR['reset_to_state'] is None
    assert m.policy()['expected_total']['space_step']==2412
    assert m.policy()['expected_total']['native_render']==252


def test_pinned_actual_authorities_read_only():
    for rel,expected in m.PINNED.items():
        assert m.sha(m.ROOT/rel)==expected
    c = m.read_json(m.ROOT/'results/consume_p5_words_MAIN_20261028_v1_MAIN01.json')
    assert c['status']=='all12_saved_program_predictions_verified'
    assert c['physical_release'] is False
    assert c['manifest_sha256'] == PRODUCER.digest(m.read_json(m.ROOT/f'results/{m.PRODUCER}_prepare_MAIN01/manifest.json'))


def test_cli_run_requires_main_authority(tmp_path):
    p = tmp_path/'manifest.json'
    p.write_text('{}')
    with pytest.raises(ValueError,match='confirm-physics'):
        m.main(['run','--manifest',str(p),'--manifest-sha256',m.sha(p)])


def test_no_existing_output_or_profiler_replacement(tmp_path):
    manifest,b,_,_ = fixture_manifest(tmp_path)
    with pytest.raises(ValueError,match='new output'):
        m.execute(manifest,'a'*64,tmp_path,PANEL,b,lambda:None,lambda:None)
    calls = b.NativeCalls(Env,SimpleNamespace(Space=Space,Body=Body))
    def existing(frame,event,arg):
        return None
    sys.setprofile(existing)
    try:
        with pytest.raises(ValueError,match='existing profiler'):
            calls.start()
        assert sys.getprofile() is existing
    finally:
        sys.setprofile(None)
