"""Record-only RAM fakes: no native simulator/model, no physical science claim."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_contact_source_native_20261028_v1 as m


C,B,P,H = m.helpers()
G = C.load(C.GEOMETRY)
SAMPLE = C.read_json(C.ROOT/'results/p5_jepa_factual_panel_20261027_v3/factual_main_v1/seed_2026102702/a/frame_00/full_public_state.json')


class Body:
    def __init__(self):
        self.position = np.zeros(2,np.float64)
        self.velocity = np.zeros(2,np.float64)
        self.angle = self.angular_velocity = 0.


class Space:
    def __init__(self):
        self.trigger = None

    def step(self,dt):
        # No integrator: executes a token callback for observer routing tests.
        if self.trigger:
            self.trigger()


class Env:
    instances,trace = [],[]
    fail_at = None
    close_failure = False
    corrupt_prefix = False
    wrong_increment = False
    static_contact = False
    no_contact = False

    def __init__(self,**ctor):
        assert ctor==C.constructor(B.CTOR)
        self.ctor = ctor
        self.index = len(self.instances)
        self.instances.append(self)
        self.seed()
        self.frame = 0
        self.closed = False

    def seed(self,value=None):
        self.seed_value = value

    def _setup(self):
        self.space = Space()
        self.agent,self.block,self.space.static_body = Body(),Body(),Body()
        self.agent.position = np.array(C.STATE[:2])
        self.block.position = np.array(C.STATE[2:4])
        self.n_contact_points = 0
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
        if self.corrupt_prefix and self.index==1 and self.frame==5:
            rgb[0,0,0]+=1
        return rgb

    def _get_obs(self):
        return np.array([*self.agent.position,*self.block.position,self.block.angle%(2*np.pi),*self.agent.velocity],np.float32)

    def observation(self):
        state = self._get_obs()
        return dict(visual=self._render_frame(),proprio=state[[0,1,5,6]]),state

    def reset(self):
        self._setup()
        self._set_state()
        return self.observation()

    def _handle_collision(self,arbiter,space,data):
        if not self.wrong_increment:
            self.n_contact_points += len(arbiter.contact_point_set.points)

    def step(self,u):
        self.frame += 1
        self.trace.append((self.index,self.frame,u.copy()))
        if self.fail_at==self.frame:
            raise RuntimeError('RAM original step failed')
        self.n_contact_points = 0
        self.latest = (self.agent.position+100*u).tolist()
        self.agent.position += 100*u
        self.agent.velocity = u.copy()
        other = self.space.static_body if self.static_contact else self.block
        arbiter = SimpleNamespace(shapes=[SimpleNamespace(body=self.agent),SimpleNamespace(body=other)],
            contact_point_set=SimpleNamespace(points=[SimpleNamespace(point_a=(1.,2.),point_b=(1.,2.),distance=-.1)],normal=(1.,0.)),total_impulse=(1.,0.))
        self.space.trigger = (lambda:self._handle_collision(arbiter,self.space,None)) if u[0]>0 and not self.no_contact else None
        for _ in range(10):
            self.space.step(.01)
        if u[0]>0:
            self.block.position[0] += 1.
            self.block.angle += .01
        self._get_goal_pose_body()
        self.coverage.append(.2)
        obs,state = self.observation()
        info = dict(state=state,pos_agent=self.agent.position.copy(),vel_agent=self.agent.velocity.copy(),
            block_pose=np.array([*self.block.position,self.block.angle]),goal_pose=np.asarray(m.GOAL['values']),
            n_contacts=int(np.ceil(self.n_contact_points/10)),final_coverage=.2,max_coverage=.2)
        return obs,float(np.clip(.2/.95,0,1)),False,info

    def close(self):
        if self.close_failure:
            raise RuntimeError('RAM close failed')
        self.closed = True


def public(env):
    result = copy.deepcopy(SAMPLE)
    for role in ('agent','block'):
        body = getattr(env,role)
        result['bodies'][role].update(position=body.position.tolist(),velocity=body.velocity.tolist(),
                                     angle=body.angle,angular_velocity=body.angular_velocity)
    result['env'].update(seed=env.seed_value,goal_pose=copy.deepcopy(m.GOAL),
        n_contact_points=env.n_contact_points,coverage_arr=env.coverage.copy(),latest_action=env.latest,success_threshold=.95)
    return result


def base():
    return SimpleNamespace(**{k:getattr(B,k) for k in ('CTOR','canonical','mkdir','save_json','save_array','Journal','operation','NativeCalls','numeric_json')},
        public_state=public,rng_state=lambda env:dict(seed=env.seed_value))


def manifest():
    candidate = C.geometry_candidate(G,SAMPLE)
    goal = C.read_json(C.ROOT/f'results/{C.GOAL}_scene_main_v1/goal_spec.json')
    contract = C.contract(B.CTOR,candidate,dict(ACTION_MEAN=[-.0087,.0068],ACTION_STD=[.2019,.2002]),goal['rgb_sha256'])
    return dict(schema=m.TAG,source_contract=contract,policy=m.policy(),bindings={})


@pytest.fixture(autouse=True)
def clean():
    Env.instances,Env.trace = [],[]
    Env.fail_at = None
    Env.close_failure = Env.corrupt_prefix = Env.wrong_increment = Env.static_contact = Env.no_contact = False
    yield
    assert sys.getprofile() is None


def execute(tmp_path,mutate=None):
    b = base()
    man = manifest()
    if mutate:
        mutate()
    def loader():
        return Env,m.make_meter(b,Env,SimpleNamespace(Space=Space,Body=Body)),dict(status='native_import_ready_physics_not_executed',RAM_token_only=True),G
    out = tmp_path/'run'
    r = m.execute(man,'a'*64,out,C,b,P,H,loader,lambda:None)
    return man,r,out


@pytest.fixture(scope='module')
def complete(tmp_path_factory):
    return execute(tmp_path_factory.mktemp('contact_native_complete'))


def test_both_prefixes_gate_before_any_suffix_and_full_meter(complete):
    man,r,out = complete
    assert r['status']=='all2_native_source_traces_recorded'
    assert r['meter']['observer_errors']==[]
    assert len(r['meter']['callbacks'])==100
    assert r['meter']['python_call_entries']['native_collision_callback']==100
    expected = dict(man['source_contract']['generator_private']['expected_native_call_entries'])
    actual = dict(r['meter']['python_call_entries'])
    actual.pop('native_collision_callback')
    assert actual==expected
    events = H.read_journal(out/'journal.jsonl')
    gate = next(i for i,e in enumerate(events) if e['phase']=='all_prefixes_byte_gated')
    before = [e for e in events[:gate] if e['phase']=='intent' and e['operation']=='step']
    assert len(before)==20 and all(e['details']['raw_env_argument']==[0.,0.] for e in before)
    after = [e for e in events[gate+1:] if e['phase']=='intent' and e['operation']=='step']
    assert len(after)==20 and all(e['details']['frame']>10 for e in after)
    consumed = m.validate_saved(man,'a'*64,out/'report.json',m.sha(out/'report.json'),H,G)
    assert consumed['status']=='all2_saved_source_traces_verified'
    assert consumed['source_witness_observed'] and consumed['blind_publication_allowed']
    assert len(consumed['private_suffix_frame_rows'])==20
    assert not consumed['scientific_admission'] and consumed['task_success'] is None


def test_complete_size_and_all_actual_frames(complete):
    _,r,out = complete
    assert sum(1 for p in out.rglob('rgb.npy'))==42
    assert len(r['rows'])==2
    assert len(r['meter']['space_step_events'])==402
    assert sum(x['bytes'] for x in H.inventory(out).values())<H.PAYLOAD-H.RESERVE
    assert all([x['frame'] for x in row['records']]==list(range(21)) for row in r['rows'])


def test_prefix_mutation_zero_suffix_dispatch_and_both_closed(tmp_path):
    _,r,out = execute(tmp_path,lambda:setattr(Env,'corrupt_prefix',True))
    assert r['status']=='native_source_incomplete' and 'prefix byte gate' in r['global_error']
    assert len(Env.instances)==2 and all(e.closed for e in Env.instances)
    assert len(Env.trace)==20 and all(f<=10 and np.array_equal(u,[0.,0.]) for _,f,u in Env.trace)
    assert all(len(row['records'])==11 for row in r['rows'])


def test_failed_native_attempt_not_refunded_and_cleanup(tmp_path):
    man,r,out = execute(tmp_path,lambda:setattr(Env,'fail_at',12))
    assert r['status']=='native_source_incomplete'
    assert r['operation_meter']['durable_intents']['step']==22
    assert r['operation_meter']['durable_returns']['step']==21
    assert all(e.closed for e in Env.instances)
    assert [len(row['records']) for row in r['rows']]==[12,11]
    result = m.validate_saved(man,'a'*64,out/'report.json',m.sha(out/'report.json'),H,G)
    assert result['status']=='partial_source_retained_not_verified'
    assert result['source_witness_observed'] is None and not result['blind_publication_allowed']


def test_close_failure_preserves_original_and_releases_profile(tmp_path):
    def mutation():
        Env.fail_at=12
        Env.close_failure=True
    _,r,_ = execute(tmp_path,mutation)
    assert 'original step failed' in r['global_error']
    assert all(row['close']['status']=='failed' for row in r['rows'])
    assert r['profiler_released']


def test_original_callback_count_mismatch_fails_next_capture(tmp_path):
    _,r,_ = execute(tmp_path,lambda:setattr(Env,'wrong_increment',True))
    assert r['status']=='native_source_incomplete'
    assert 'callback/public counter' in r['global_error']
    assert all(e.closed for e in Env.instances)


@pytest.mark.parametrize('case',['static_contact','no_contact'])
def test_non_agent_block_or_absent_callback_not_witness(tmp_path,case):
    man,r,out = execute(tmp_path,lambda:setattr(Env,case,True))
    assert r['status']=='all2_native_source_traces_recorded'
    c = m.validate_saved(man,'a'*64,out/'report.json',m.sha(out/'report.json'),H,G)
    assert not c['source_contact_observed'] and not c['source_witness_observed']
    assert not c['blind_publication_allowed']


@pytest.mark.parametrize('key',['counter_after','returned','space_step_index','substep','body_roles','observer_errors','point'])
def test_callback_receipt_mutations(complete,key):
    _,r,_ = complete
    meter = copy.deepcopy(r['meter'])
    e = meter['callbacks'][0]
    if key=='counter_after':
        e['counter_after']+=1
    elif key=='returned':
        e['returned']=False
    elif key=='space_step_index':
        e['space_step_index']=0
    elif key=='substep':
        e['substep_in_operation']=0
    elif key=='body_roles':
        e['body_roles']=['agent','invented']
    elif key=='observer_errors':
        meter['observer_errors']=['failed observer']
    else:
        e['points'][0]['distance']=float('nan')
    with pytest.raises(ValueError):
        m.validate_callbacks(meter)


def test_reset_contract_and_static_gap_not_actual_contact():
    b = base()
    env = Env(**C.constructor(B.CTOR))
    env.seed(C.SEED)
    env.reset()
    p = public(env)
    m.reset_gate(p,manifest()['source_contract'],G)
    p['bodies']['agent']['position'][0]+=1
    with pytest.raises(ValueError,match='post-reset'):
        m.reset_gate(p,manifest()['source_contract'],G)
    assert manifest()['source_contract']['generator_private']['geometry']['actual_contacts'] is None


def test_existing_profiler_not_replaced():
    calls = m.make_meter(base(),Env,SimpleNamespace(Space=Space,Body=Body))
    def prior(frame,event,arg):
        return None
    sys.setprofile(prior)
    try:
        with pytest.raises(ValueError,match='existing profiler'):
            calls.start()
        assert sys.getprofile() is prior
    finally:
        sys.setprofile(None)


def test_rejected_runtime_retains_probe_and_two_not_run(tmp_path):
    b = base()
    man = manifest()
    probe = dict(status='runtime_probe_failed',problems=['RAM wrong dependency'])
    out = tmp_path/'rejected_runtime'
    r = m.execute(man,'a'*64,out,C,b,P,H,lambda:(None,None,probe,None),lambda:None)
    assert r['status']=='native_source_incomplete'
    assert len(r['rows'])==2 and all(x['status']=='not_run' for x in r['rows'])
    assert m.read_json(out/'runtime_probe.json')==probe
    assert Env.instances==[] and r['meter'] is None
    c = m.validate_saved(man,'a'*64,out/'report.json',m.sha(out/'report.json'),H,G)
    assert c['status']=='partial_source_retained_not_verified'


def test_cli_run_requires_main_permission(tmp_path):
    with pytest.raises(ValueError,match='confirm-physics'):
        m.main(['run','--manifest',str(tmp_path/'manifest.json'),'--manifest-sha256','a'*64])


def test_prepare_missing_contract_fails_closed(tmp_path):
    with pytest.raises(ValueError,match='nonsymlink'):
        m.prepare(tmp_path/'missing.json','0'*64)


def test_pinned_frozen_owners_and_no_neural_native_imports():
    for name,h in m.PINS.items():
        assert m.sha(m.ROOT/'experiments/spectral_reachability'/f'{name}.py')==h
    before = set(sys.modules)
    m.helpers()
    assert not {x for x in set(sys.modules)-before if x.split('.')[0] in ('torch','pymunk','pygame','gym')}


def test_publish_requires_exact_actual_positive_receipt(monkeypatch,tmp_path):
    r = tmp_path/'receipt.json'
    B.save_json(r,dict(status='RAM_not_authority'))
    monkeypatch.setattr(m,'consume',lambda *a:dict(status='all2_saved_source_traces_verified',blind_publication_allowed=False))
    with pytest.raises(ValueError,match='positive source witness'):
        m.publish(tmp_path/'m.json','a'*64,tmp_path/'private/report.json','b'*64,r,m.sha(r),tmp_path/'public')
    assert not (tmp_path/'public').exists()


def test_positive_ram_publication_excludes_private_witness(monkeypatch,tmp_path,complete):
    man,r,out = complete
    receipt = m.validate_saved(man,'a'*64,out/'report.json',m.sha(out/'report.json'),H,G)
    receipt_path = tmp_path/'private_consume.json'
    B.save_json(receipt_path,receipt)
    manifest_path = tmp_path/'native_manifest.json'
    B.save_json(manifest_path,man)
    # Explicit test seam only: production consume still requires full actual
    # prepare/source/contract authority, not these synthetic token captures.
    monkeypatch.setattr(m,'consume',lambda *args:receipt)
    public_dir = tmp_path/'public_context'
    result = m.publish(manifest_path,m.sha(manifest_path),out/'report.json',m.sha(out/'report.json'),
                       receipt_path,m.sha(receipt_path),public_dir)
    payload = m.read_json(Path(result['public_payload']))
    assert set(payload['inputs'])==set(C.blind_schema()['files'])
    assert payload['source_witness_included'] is False
    assert payload['scientific_admission'] is False
    assert all(Path(x['path']).parent==public_dir for x in payload['inputs'].values())
    assert {p.name for p in public_dir.iterdir()}=={k+'.npy' for k in C.blind_schema()['files']}
    assert not any(key in payload for key in ('seed','source_contract','contact','reference_suffix','receipt','block','geometry'))
    assert np.load(public_dir/'history_normalized.npy',allow_pickle=False).any()
    assert not np.load(public_dir/'history_raw_u.npy',allow_pickle=False).any()
    lineage = m.read_json(Path(result['private_lineage']))
    assert lineage['source_receipt_sha256']==m.sha(receipt_path)
    assert Path(result['private_lineage']).parent!=public_dir
    with pytest.raises(ValueError,match='exclusive'):
        m.publish(manifest_path,m.sha(manifest_path),out/'report.json',m.sha(out/'report.json'),
                  receipt_path,m.sha(receipt_path),public_dir)


def test_json_hash_duplicate_and_symlink(tmp_path):
    p = tmp_path/'x.json'
    p.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError,match='duplicate'):
        m.read_json(p)
    with pytest.raises(ValueError,match='FILE SHA'):
        m.read_json(p,'f'*64)
    link = tmp_path/'link'
    link.symlink_to(p)
    with pytest.raises(ValueError,match='nonsymlink'):
        m.sha(link)
