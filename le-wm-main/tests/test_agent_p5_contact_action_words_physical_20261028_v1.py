"""Record-only RAM fakes: no native simulator/model, no physical science claim."""
import copy
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_contact_action_words_physical_20261028_v1 as m


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
    
    original=contract['generator_private']['plans'][0]
    from experiments.spectral_reachability import agent_p5_native_action_words_20261028_v2 as words
    plans=[dict(arm=word,seed=original['seed'],constructor=original['constructor'],prefix_raw_u=original['prefix_raw_u'],
                reference_suffix_raw_u=words.programs()[word].reshape(10,2).tolist()) for word in m.ARMS]
    contract['generator_private'].update(plans=plans,expected_native_call_entries=dict(m.EXPECTED))
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
    def loader():
        return Env,m.make_meter(b,Env,SimpleNamespace(Space=Space,Body=Body)),dict(status='native_import_ready_physics_not_executed',RAM_token_only=True),G
    out = tmp_path/'run'

    # Generate inert saved source PREFIX with recorder only, no Space integrator.
    source=tmp_path/'original_prefix'
    source.mkdir()
    env=Env(**C.constructor(B.CTOR))
    env.seed(C.SEED)
    obs,state=env.reset()
    writer=H.Writer(b,tmp_path)
    records=[]
    for frame in range(11):
        if frame:
            raw=np.zeros(2,np.float64)
            command=H.command_for(frame,raw,np.array(env.agent.position,np.float64),records[-1]['files']['full_public_state.json']['sha256'])
            writer.save_json(source/f'command_{frame:02d}.json',command)
            obs,reward,done,info=env.step(raw)
            transition=dict(kind='step',reward=reward,done=done,info=info)
            state=info['state']
        else:
            transition=dict(kind='reset')
        record=P.capture(b,writer,env,obs,state,transition,source/f'frame_{frame:02d}')
        records.append(record)
    writer.save_json(source/'pre_reset_rng.json',b.rng_state(env))
    man['prefix_identity']=[dict(frame=i,files={name:dict(path=str(source/f'frame_{i:02d}'/name),
        sha256=m.sha(source/f'frame_{i:02d}'/name),bytes=(source/f'frame_{i:02d}'/name).stat().st_size) for name in m.NAMES}) for i in range(11)]
    man['prefix_commands']=[dict(path=str(source/f'command_{i:02d}.json'),sha256=m.sha(source/f'command_{i:02d}.json')) for i in range(1,11)]
    man['seeded_rng']=dict(path=str(source/'pre_reset_rng.json'),sha256=m.sha(source/'pre_reset_rng.json'))
    man['public_payload_sha256']='f'*64
    pub=tmp_path/'public'
    pub.mkdir()
    inputs={}
    for i in (0,5,10):
        p=pub/f'context_rgb_{i:02d}.npy'
        B.save_array(p,np.load(source/f'frame_{i:02d}/rgb.npy',allow_pickle=False))
        inputs[f'context_rgb_{i:02d}']=dict(path=str(p),sha256=m.sha(p))
    for key,a in [('context_proprio',np.stack([np.load(source/f'frame_{i:02d}/proprio.npy',allow_pickle=False) for i in (0,5,10)])),
                  ('history_raw_u',np.zeros((10,2),np.float64))]:
        p=pub/(key+'.npy'); B.save_array(p,a); inputs[key]=dict(path=str(p),sha256=m.sha(p))
    man['public_inputs']=inputs
    Env.instances,Env.trace=[],[]
    if mutate:
        mutate()
    r = m.execute(man,'a'*64,out,C,b,P,H,loader,lambda:None)


    return man,r,out


@pytest.fixture(scope='module')
def complete(tmp_path_factory):
    return execute(tmp_path_factory.mktemp('contact_native_complete'))


def test_both_prefixes_gate_before_any_suffix_and_full_meter(complete):
    man,r,out = complete
    assert r['status']=='all4_native_word_traces_recorded'
    assert r['meter']['observer_errors']==[]
    assert len(r['meter']['callbacks'])==150
    assert r['meter']['python_call_entries']['native_collision_callback']==150
    expected = dict(man['source_contract']['generator_private']['expected_native_call_entries'])
    actual = dict(r['meter']['python_call_entries'])
    actual.pop('native_collision_callback')
    assert actual==expected
    events = H.read_journal(out/'journal.jsonl')
    gate = next(i for i,e in enumerate(events) if e['phase']=='all_prefixes_byte_gated')
    before = [e for e in events[:gate] if e['phase']=='intent' and e['operation']=='step']
    assert len(before)==40 and all(e['details']['raw_env_argument']==[0.,0.] for e in before)
    after = [e for e in events[gate+1:] if e['phase']=='intent' and e['operation']=='step']
    assert len(after)==40 and all(e['details']['frame']>10 for e in after)
    consumed = m.validate_saved(man,'a'*64,out/'report.json',m.sha(out/'report.json'),H,G)
    assert consumed['status']=='all4_saved_native_words_verified'
    assert consumed['physical_release'] is False
    assert len(consumed['private_suffix_frame_rows'])==40
    assert not consumed['scientific_admission'] and consumed['task_success'] is None


def test_complete_size_and_all_actual_frames(complete):
    _,r,out = complete
    assert sum(1 for p in out.rglob('rgb.npy'))==84
    assert len(r['rows'])==4
    assert len(r['meter']['space_step_events'])==804
    assert sum(x['bytes'] for x in H.inventory(out).values())<H.PAYLOAD-H.RESERVE
    assert all([x['frame'] for x in row['records']]==list(range(21)) for row in r['rows'])


def test_prefix_mutation_zero_suffix_dispatch_and_both_closed(tmp_path):
    _,r,out = execute(tmp_path,lambda:setattr(Env,'corrupt_prefix',True))
    assert r['status']=='native_source_incomplete' and 'prefix byte gate' in r['global_error']
    assert len(Env.instances)==4 and all(e.closed for e in Env.instances)
    assert len(Env.trace)==40 and all(f<=10 and np.array_equal(u,[0.,0.]) for _,f,u in Env.trace)
    assert all(len(row['records'])==11 for row in r['rows'])


def test_failed_native_attempt_not_refunded_and_cleanup(tmp_path):
    man,r,out = execute(tmp_path,lambda:setattr(Env,'fail_at',12))
    assert r['status']=='native_source_incomplete'
    assert r['operation_meter']['durable_intents']['step']==42
    assert r['operation_meter']['durable_returns']['step']==41
    assert all(e.closed for e in Env.instances)
    assert [len(row['records']) for row in r['rows']]==[12,11,11,11]
    result = m.validate_saved(man,'a'*64,out/'report.json',m.sha(out/'report.json'),H,G)
    assert result['status']=='partial_source_retained_not_verified'
    assert result['private_suffix_frame_rows'] is None and not result['physical_release']


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
    assert r['status']=='all4_native_word_traces_recorded'
    c = m.validate_saved(man,'a'*64,out/'report.json',m.sha(out/'report.json'),H,G)
    assert all(x['agent_block_callback_points']==0 for x in c['private_suffix_frame_rows'])
    assert c['scientific_admission'] is False


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
    assert len(r['rows'])==4 and all(x['status']=='not_run' for x in r['rows'])
    assert m.read_json(out/'runtime_probe.json')==probe
    assert Env.instances==[] and r['meter'] is None
    c = m.validate_saved(man,'a'*64,out/'report.json',m.sha(out/'report.json'),H,G)
    assert c['status']=='partial_source_retained_not_verified'


def test_cli_run_requires_main_permission(tmp_path):
    with pytest.raises(ValueError,match='confirm-physics'):
        m.main(['run','--manifest',str(tmp_path/'manifest.json'),'--manifest-sha256','a'*64])


def test_prepare_missing_contract_fails_closed(tmp_path):
    with pytest.raises(ValueError,match='four-word'):
        m.prepare(tmp_path/'missing.json','0'*64,tmp_path/'m.json','0'*64,tmp_path/'r.json','0'*64,tmp_path/'c.json','0'*64)


def test_pinned_frozen_owners_and_no_neural_native_imports():
    for name,h in m.PINS.items():
        assert m.sha(m.ROOT/'experiments/spectral_reachability'/f'{name}.py')==h
    before = set(sys.modules)
    m.helpers()
    assert not {x for x in set(sys.modules)-before if x.split('.')[0] in ('torch','pymunk','pygame','gym')}


def test_policy_separate_physical_axes():
    p=m.policy()
    assert p['arms']==['xy','yx','x_minus_x','zero_zero'] and p['denominator']==4
    assert p['expected_native_call_entries']==m.EXPECTED
    assert m.EXPECTED==dict(env_construct=4,seed=8,reset=4,control_step=80,space_step=804,
        native_render=84,native_get_obs=84,close=4,setup=4,set_state=4,space_construct=4,
        native_goal_body=164,body_construct=176)
    assert p['wall_seconds']==900 and p['free_floor_bytes']==200*1024**2


def rebind_output(out,r):
    # Test-only tamper of temporary recorder output, never MAIN results.
    journal=out/'journal.jsonl'
    import json
    entries=[json.loads(x) for x in journal.read_text().splitlines()]
    entries[-1]['event']['summary_sha256']=m.digest({k:v for k,v in r.items() if k!='artifacts'})
    previous=None
    for e in entries:
        e['previous_sha256']=previous
        e['entry_sha256']=m.digest({k:v for k,v in e.items() if k!='entry_sha256'})
        previous=e['entry_sha256']
    journal.write_text(''.join(m.canonical(e).decode()+'\n' for e in entries))
    r['artifacts']=H.inventory(out)
    (out/'report.json').write_text(m.canonical(r).decode()+'\n')


@pytest.mark.parametrize('change',['denominator','done','callback','pose','command','close','space','goal','wall'])
def test_saved_mutations_fail_closed(tmp_path,change):
    man,r,out=execute(tmp_path)
    if change=='denominator': r['rows']=r['rows'][:3]
    elif change=='callback': r['meter']['callbacks'][0]['returned']=False
    elif change=='close': r['rows'][0]['close']['returned']=False
    elif change=='space': r['meter']['space_step_events'][0]['dt']=.02
    elif change=='wall': r['wall_seconds']=901
    else:
        arm='xy'
        if change=='done':
            path=out/arm/'frame_11/transition.json'; v=m.read_json(path); v['done']=True
        elif change=='pose':
            path=out/arm/'frame_11/full_public_state.json'; v=m.read_json(path); v['bodies']['block']['position'][0]+=1
        elif change=='goal':
            path=out/arm/'frame_11/full_public_state.json'; v=m.read_json(path); v['env']['goal_pose']['values'][0]+=1
        else:
            path=out/arm/'command_11.json'; v=m.read_json(path); v['raw_env_argument'][0]=0
        path.write_text(m.canonical(v).decode()+'\n')
        if change in ('done','pose','goal'):
            name=path.name
            r['rows'][0]['records'][11]['files'][name]=dict(sha256=m.sha(path),bytes=path.stat().st_size)
    rebind_output(out,r)
    with pytest.raises(ValueError): m.validate_saved(man,'a'*64,out/'report.json',m.sha(out/'report.json'),H,G)


def test_live_prefix_public_identity_rejects_numeric_drift(tmp_path):
    man,r,out=execute(tmp_path)
    p=Path(man['public_inputs']['context_proprio']['path'])
    a=np.load(p,allow_pickle=False); a[1,0]+=1
    with p.open('wb') as f: np.save(f,a,allow_pickle=False)
    man['public_inputs']['context_proprio']['sha256']=m.sha(p)
    with pytest.raises(ValueError,match='published raw proprio'): m.bind_public_prefix(man)


def test_saved_actual_numeric_authority_without_running_prepare():
    """Only frozen-source hashes and actual saved JSON; no model/physics load."""
    assert m.sha(m.ROOT/'experiments/spectral_reachability'/(m.NUMERIC+'.py'))==m.NUMERIC_SHA
    assert m.sha(m.ROOT/'experiments/spectral_reachability'/(m.NATIVE+'.py'))==m.NATIVE_SHA
    mp=m.ROOT/f'results/{m.NUMERIC}_prepare_MAIN01.json'
    rp=m.ROOT/f'results/{m.NUMERIC}_cuda_MAIN01/report.json'
    cp=m.ROOT/f'results/{m.NUMERIC}_consume_MAIN01.json'
    model=m.read_json(mp,m.MODEL_MANIFEST_SHA)
    r=m.read_json(rp,m.MODEL_REPORT_SHA)
    receipt=m.read_json(cp,m.MODEL_RECEIPT_SHA)
    assert model['protocol']['words']==list(m.ARMS)
    assert receipt['status']=='all4_saved_public_action_words_verified'
    assert r['meter']['completed']['predictor']==8 and len(r['artifacts'])==43
    assert receipt['report_sha256']==m.MODEL_REPORT_SHA


@pytest.fixture
def prepare_inputs(tmp_path,monkeypatch,complete):
    """Synthetic saved authority layout, not permission to run a simulator."""
    man,r,out=complete
    original=tmp_path/'source'/'original_prefix'
    shutil.copytree(out.parent/'original_prefix',original)
    monkeypatch.setattr(m,'SOURCE_PREFIX',original)
    # source inventory authenticated by injected fixture SHA, never old suffix.
    source_report=original.parent/'report.json'
    B.save_json(source_report,dict(artifacts={str(p.relative_to(original.parent)):
        dict(sha256=m.sha(p),bytes=p.stat().st_size) for p in original.rglob('*') if p.is_file()}))
    monkeypatch.setattr(m,'SOURCE_REPORT_SHA',m.sha(source_report))
    goal=C.read_json(C.ROOT/f'results/{C.GOAL}_scene_main_v1/goal_spec.json')
    contract=C.contract(B.CTOR,C.geometry_candidate(G,SAMPLE),
        dict(ACTION_MEAN=[-.0087,.0068],ACTION_STD=[.2019,.2002]),goal['rgb_sha256'])
    old=dict(source_contract=contract,bindings={},runtime_manifest='fixture_runtime')
    native_path=tmp_path/'native.json'; B.save_json(native_path,old)
    from experiments.spectral_reachability import agent_p5_native_action_words_20261028_v2 as words
    model=dict(public_payload_sha256='f'*64,protocol=dict(words=list(m.ARMS)),
        plans={w:dict(raw=words.programs()[w].tolist()) for w in m.ARMS},inputs=man['public_inputs'],
        bindings={},model_spec=dict(loader_manifest=dict(seals={})))
    model_path=tmp_path/'model.json'; B.save_json(model_path,model)
    report_path=tmp_path/'model_report.json'; B.save_json(report_path,dict(artifacts={}))
    actual=dict(status='all4_saved_public_action_words_verified',scientific_admission=False)
    receipt_path=tmp_path/'consume.json'; B.save_json(receipt_path,actual)
    numeric=SimpleNamespace(consume=lambda *args:actual,PUBLIC_SHA='f'*64,owner=words)
    native=SimpleNamespace(verify=lambda v:m.require(v==old,'fixture original source contract'))
    monkeypatch.setattr(m,'upstreams',lambda:(native,numeric))
    for name,path in (('NATIVE_MANIFEST_SHA',native_path),('MODEL_MANIFEST_SHA',model_path),
                      ('MODEL_REPORT_SHA',report_path),('MODEL_RECEIPT_SHA',receipt_path)):
        monkeypatch.setattr(m,name,m.sha(path))
    return dict(native_path=native_path,native_sha=m.sha(native_path),model_path=model_path,model_sha=m.sha(model_path),
                report_path=report_path,report_sha=m.sha(report_path),receipt_path=receipt_path,receipt_sha=m.sha(receipt_path))


def test_prepare_true_layout_all_four_actual_words(prepare_inputs):
    prepared=m.prepare(**prepare_inputs)
    assert prepared['policy']['denominator']==4
    plans=prepared['source_contract']['generator_private']['plans']
    assert [p['arm'] for p in plans]==list(m.ARMS)
    assert [p['reference_suffix_raw_u'] for p in plans]!=C.reference_queues()['push_x'].tolist()
    m.verify(prepared)
    assert len(prepared['prefix_identity'])==11
    assert prepared['source_contract']['generator_private']['expected_native_call_entries']==m.EXPECTED
    assert all(len(p['prefix_raw_u'])==len(p['reference_suffix_raw_u'])==10 for p in plans)


@pytest.mark.parametrize('key',['native_sha','model_sha','report_sha','receipt_sha'])
def test_prepare_external_actualhash_mandatory(prepare_inputs,key):
    inputs=dict(prepare_inputs); inputs[key]='0'*64
    with pytest.raises(ValueError,match='four-word'): m.prepare(**inputs)


def test_prepare_rejects_resealed_wrong_consume(prepare_inputs,monkeypatch):
    path=prepare_inputs['receipt_path']
    path.write_text(m.canonical(dict(status='all4_saved_public_action_words_verified',scientific_admission=True)).decode())
    prepare_inputs['receipt_sha']=m.sha(path)
    monkeypatch.setattr(m,'MODEL_RECEIPT_SHA',m.sha(path))
    with pytest.raises(ValueError,match='exact strict'): m.prepare(**prepare_inputs)


def test_prepare_rejects_resealed_changed_word(prepare_inputs,monkeypatch):
    path=prepare_inputs['model_path']; v=m.read_json(path)
    v['plans']['xy']['raw'][0][0][0]=0
    path.write_text(m.canonical(v).decode())
    prepare_inputs['model_sha']=m.sha(path); monkeypatch.setattr(m,'MODEL_MANIFEST_SHA',m.sha(path))
    with pytest.raises(ValueError,match='raw word'): m.prepare(**prepare_inputs)


def test_prefix_gate_checks_every_source_file_before_suffix(tmp_path):
    def change(): Env.corrupt_prefix=True
    _,r,_=execute(tmp_path,change)
    assert r['status']=='native_source_incomplete'
    assert len(Env.trace)==40 and all(f<=10 for _,f,u in Env.trace)
    assert len(Env.instances)==4 and all(e.closed for e in Env.instances)


