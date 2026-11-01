"""RAM fake native API only: no real environment, neural model or physics."""
from copy import deepcopy
import io
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p4_fresh_source_20261028_v1 as m
import test_p1_constructive_moving_history_source_20261027_v1 as h


class RAM(h.MemoryStore):
    def check(self):return None
    def file_sha(self,name):
        import hashlib
        return hashlib.sha256(b''.join(m.raw(e) for e in self.events)).hexdigest()


class Env(h.FakeEnv):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.goal_pose=np.array([256.,256.,np.pi/4]);self.render_action=False;self.render_size=224


def fake_state(goal,shapes,slot,*unused):
    shift=30.*(slot['root_seed']-203027101)
    return np.array([100.+shift,100.,150.+shift,150.,0.,0.,0.])


def run(monkeypatch,*,base=Env,duplicate=False,p=None,store=None):
    monkeypatch.setattr(m.moving.source,'sample_state',
                        (lambda *a:fake_state(None,None,dict(root_seed=203027101))) if duplicate else fake_state)
    p=m.build('calibration_only') if p is None else p;store=store or RAM();envs=[]
    def factory(meter):
        env=m.live.metered_class(base,meter)();envs.append(env);return env
    result=m.execute(p,store,factory,authorization=m.AUTH)
    return result,store,envs


def test_prepare_exact_new_four_slots_without_old_global_mutation():
    old=m.moving.build(60)
    p=m.build('calibration_only')
    assert len(old['slots']) == 2 and p['slots'] == m.slots() and len(p['slots']) == 4
    assert p['slots'][0]['root_seed'] == 203027101 and p['slots'][3]['reset_seed'] == 203027204
    assert p['freshness'] is p['confirmed'] is False
    assert p['recipe'] == old['recipe'] and p['witness'] == old['witness']
    assert m.verify(p) == p and m.moving.build(60) == old


@pytest.mark.parametrize('kwargs',[dict(mode='fresh_candidate'),dict(mode=''),dict(mode='calibration_only',wall_seconds=0),
                                  dict(mode='calibration_only',wall_seconds=1201),dict(mode='calibration_only',wall_seconds=True),
                                  dict(mode='calibration_only',history_sha='0'*64)])
def test_missing_certificate_or_invalid_prepare_refused(kwargs):
    with pytest.raises((ValueError,TypeError)):m.build(**kwargs)


@pytest.mark.parametrize('key,value',[('slots',[]),('mode','fresh_candidate'),('caps',{}),('freshness',True),
                                    ('raw_warmup',[[0.,0.]]*10),('wall_seconds',1201)])
def test_reconstruction_tampering_refused(key,value):
    p=m.build('calibration_only');p[key]=value
    with pytest.raises((ValueError,KeyError)):m.verify(p)


def test_full_native_episode_reuse_four_slot_blind_witness_goal_closure(monkeypatch):
    r,s,envs=run(monkeypatch)
    assert r['status']=='fixed_slots_processed',r['errors']
    assert r['sourcepack_complete'] and r['reachable_count']==4 and len(r['rows'])==4 and len(envs)==1
    assert r['freshness'] is r['confirmed'] is r['P1_admission'] is r['P4_admission'] is False
    assert all(x['freshness'] is x['confirmed'] is False and x['replay_native_success']==[True,True] for x in r['rows'])
    assert len(envs[0].resets)==24
    assert r['counts']['reset']==dict(attempted=24,completed=24)
    assert r['counts']['goal_bind']==dict(attempted=8,completed=8)
    assert r['counts']['goal_setter']==dict(attempted=32,completed=32)
    assert r['counts']['physics_step']['completed']==48+10*r['counts']['public_step']['completed']
    for row in r['rows']:
        assert [p['phase'] for p in row['phases']]==['construction','replay0','replay1']
        assert all(p['status']=='returned' for p in row['phases'])
        assert row['actual_root_identity'] is not None
        with np.load(io.BytesIO(s.files[row['blind_payload']]),allow_pickle=False) as z:
            assert set(z.files)=={'rgb_history','raw_action_history','goal_rgb'}
            assert m.moving.exact(z['raw_action_history'],m.moving.warmup_commands())
            assert z['rgb_history'].shape==(3,224,224,3)
        with np.load(io.BytesIO(s.files[row['certificate']]),allow_pickle=False) as z:
            assert set(z.files)==set('goal goal_rgb witness_commands witness_logits future_states observed_states root_history raw_action_history'.split())
    events=s.events
    assert sum(e['kind']=='actual_root_pre_witness_check' for e in events)==4
    assert sum(e['kind']=='replay_root_pre_witness_check' for e in events)==8
    assert np.array_equal(envs[0].goal_pose,[256.,256.,np.pi/4])
    assert not envs[0]._live_spaces  # frozen subclass restores every fake space hook.


def test_duplicate_rejected_before_witness_without_replacement(monkeypatch):
    r,s,_=run(monkeypatch,duplicate=True)
    assert r['reachable_count']==1 and len(r['rows'])==4
    assert [x['status'] for x in r['rows']]==['reachable_native_replayed']+['rejected_fixed_candidate']*3
    assert not any(n.startswith('s1_construction_commit') for n in s.files)
    assert not any(n.startswith('s1_blind') for n in s.files)
    assert r['counts']['reset']['completed']==12


def test_historical_pose_exclusion_applies_before_witness(monkeypatch):
    p=m.build('calibration_only')
    # RAM verified plan fixture: changing the real prepare would be rejected.
    _,store,_=run(monkeypatch)
    e=next(e for e in store.events if e['kind']=='actual_root_pre_witness_check')
    p['historical_roots']=[e['identity']]
    monkeypatch.setattr(m,'verify',lambda x:x)
    r,s,_=run(monkeypatch,p=p)
    assert r['rows'][0]['status']=='rejected_fixed_candidate'
    assert 's0_construction_commit.json' not in s.files


def test_engineering_failure_retains_pending_and_all_four_rows(monkeypatch):
    class Bad(Env):
        def __init__(self):super().__init__(fail_step=True,close_error=True)
    r,s,_=run(monkeypatch,base=Bad)
    assert r['status']=='engineering_failed' and not r['sourcepack_complete']
    assert len(r['rows'])==4 and [x['status'] for x in r['rows'][1:]]==['not_started']*3
    assert r['counts']['public_step']==dict(attempted=1,completed=0) and r['pending']
    assert 'first fake physics failure' in r['first_error']['message']
    assert 'secondary fake close failure' in str(r['errors'])
    assert [p['status'] for p in r['rows'][0]['phases']]==['failed_pending','not_started','not_started']
    assert all(len(x['phases'])==3 for x in r['rows'])


def test_native_guard_failure_keeps_four_and_real_failed_observation(monkeypatch):
    class Invisible(Env):
        def render(self):return np.zeros((224,224,3),np.uint8)
    r,s,_=run(monkeypatch,base=Invisible)
    assert len(r['rows'])==4 and r['reachable_count']==0
    assert all(x['status']=='rejected_fixed_candidate' for x in r['rows'])
    assert any(e['kind']=='fresh_observation' for e in s.events)
    assert not any('blind_inputs' in f for f in s.files)


def test_no_authority_never_constructs_environment():
    with pytest.raises(ValueError):m.execute(m.build('calibration_only'),RAM(),lambda meter:pytest.fail('factory'),authorization='no')


@pytest.mark.parametrize('seeds,spans',[([203027201],[]),([203027101],[]),([] ,[[203027101,203027105]])])
def test_cross_namespace_reserved_conflicts_never_replace(seeds,spans):
    with pytest.raises(ValueError):m.seed_check(m.slots(),seeds,spans)


def test_half_open_span_and_pose_alias_rules():
    m.seed_check(m.slots(),[],[[1,203027101]])
    a=dict(root_state=[100.,100.,200.,200.,0.,0.,0.],physical_pose=[100.,100.,200.,200.,0.],
           root_state_sha256='a',root_snapshot_sha256='b',history_pixels_sha256=['c'])
    b=dict(a,root_state=[100.,100.,200.,200.,2*np.pi,9.,10.],root_state_sha256='other')
    assert m.duplicate(a,b)
    b['root_state'][0]+=1.;b.update(root_snapshot_sha256='d',history_pixels_sha256=['e'])
    assert not m.duplicate(a,b)


def test_durable_operation_intent_before_actual_entry_and_return(monkeypatch):
    r,s,_=run(monkeypatch)
    attempts={};completed={}
    for e in s.events:
        if e['kind']=='operation_attempted':attempts[e['operation_id']]=e
        if e['kind']=='operation_completed':completed[e['operation_id']]=e
    assert set(attempts)==set(completed)
    assert all(attempts[i]['index']<completed[i]['index'] and
               attempts[i]['resource']==completed[i]['resource'] for i in attempts)
    for k,c in r['counts'].items():
        assert sum(e['resource']==k for e in attempts.values())==c['attempted']


def test_final_report_write_failure_not_complete(monkeypatch):
    class Broken(RAM):
        def put(self,name,value,final=False):
            if final:raise OSError('RAM final fail')
            return super().put(name,value,final)
    r,_,_=run(monkeypatch,store=Broken())
    assert not r['sourcepack_complete'] and r['status']=='engineering_failed' and not r['freshness']


def certificate_fixture():
    state=[100.,100.,200.,200.,0.,0.,0.]
    identity=m.exclusion.state_identity(state)
    root=dict(root_state=state,**identity,root_snapshot_sha256='1'*64,
              history_pixels_sha256=['2'*64]*3,raw_history_sha256='3'*64)
    content=dict(schema='moving_fresh_source_exclusions.v1',protocol=dict(path=str(m.PROTOCOL),sha256=m.PROTOCOL_SHA),
                 project_root=str(m.ROOT),cutoff_utc='2026-10-28T00:00:00Z',observed_roots=[root],
                 used_or_reserved_root_seeds=[1],used_or_reserved_reset_seeds=[2],reserved_reset_spans=[[3,4]],
                 root_goal_sha256s=[],unresolved_sources=[],source_sha256={str(m.exclusion.__file__):m.sha(m.exclusion.__file__)},
                 evidence_files=[dict(family=f,path='/RAM/'+f,sha256='4'*64) for f in m.exclusion.REQUIRED_FAMILIES])
    approval=dict(schema=m.exclusion.NAME+'.MAIN_coverage_approval',authority='MAIN',approved=True,
                  manifest_content_sha256=m.exclusion.digest(content),cutoff_utc=content['cutoff_utc'],
                  coverage_scope=list(m.exclusion.REQUIRED_FAMILIES))
    cert=dict(content,coverage_content_sha256=m.exclusion.digest(content),approval=dict(authority='MAIN',approved=True,
              coverage_scope=approval['coverage_scope'],external_file=dict(path='/RAM/approval',sha256='5'*64)),
              approval_errors=[],source_run_preconditions_met=True,status='coverage_approved_pending_new_root_identity_checks',
              request_sha256='6'*64,projected_file_bytes=0,expanded_metadata_bytes=0)
    return cert,approval


def test_external_history_certificate_contract_in_RAM_only(monkeypatch):
    cert,approval=certificate_fixture()
    monkeypatch.setattr(m,'read',lambda path,expected:deepcopy(cert if path=='/RAM/cert' else approval))
    assert m.history_certificate('/RAM/cert','7'*64)==cert
    # A RAM certificate demonstrates syntax only, not real historical evidence.


@pytest.mark.parametrize('mutation',['unresolved','approval','content','scope','cutoff','state','snapshot','families','original_source'])
def test_history_certificate_mutations_fail_closed(monkeypatch,mutation):
    cert,approval=certificate_fixture()
    if mutation=='unresolved':cert['unresolved_sources']=['unknown reset']
    elif mutation=='approval':approval['approved']=False
    elif mutation=='content':cert['coverage_content_sha256']='0'*64
    elif mutation=='scope':approval['coverage_scope']=[]
    elif mutation=='cutoff':approval['cutoff_utc']='2026-10-27T00:00:00Z'
    elif mutation=='state':cert['observed_roots'][0]['root_state'][0]+=1.
    elif mutation=='snapshot':cert['observed_roots'][0]['root_snapshot_sha256']=None
    elif mutation=='families':cert['evidence_files']=[]
    elif mutation=='original_source':cert['source_sha256']={}
    # For downstream semantic mutations, bind the modified content too. This
    # proves rejection is not merely an invalidated checksum.
    if mutation in ('state','snapshot','families','original_source'):
        extra={'coverage_content_sha256','approval','approval_errors','source_run_preconditions_met','status',
               'request_sha256','projected_file_bytes','expanded_metadata_bytes'}
        cert['coverage_content_sha256']=m.exclusion.digest({k:v for k,v in cert.items() if k not in extra})
        approval['manifest_content_sha256']=cert['coverage_content_sha256']
    monkeypatch.setattr(m,'read',lambda path,expected:deepcopy(cert if path=='/RAM/cert' else approval))
    with pytest.raises((ValueError,KeyError)):m.history_certificate('/RAM/cert','7'*64)


def test_budget_and_wall_refuse_before_actual_entry():
    p=m.build('calibration_only');meter=m.Meter(p,RAM(),m.time.monotonic())
    meter.caps['constructor']=0;entered=[]
    with pytest.raises(ValueError):meter.call('constructor',lambda:entered.append(True))
    assert not entered and meter.counts['constructor']['attempted']==0
    meter=m.Meter(p,RAM(),m.time.monotonic()-1201)
    with pytest.raises(ValueError):meter.call('constructor',lambda:entered.append(True))
    assert not entered


def test_witness_commit_precedes_any_witness_public_step(monkeypatch):
    class Order(RAM):
        def emit(self,kind,**fields):
            if kind=='operation_attempted' and fields['resource']=='public_step':
                namespace=fields['episode_id'];slot=int(namespace[1]);phase=namespace[3:]
                warms=sum(e['kind']=='warmup_return' and e['slot']==slot and e['phase']==phase for e in self.events)
                if warms==10:assert f's{slot}_{phase}_commit.json' in self.files
            super().emit(kind,**fields)
    r,_,_=run(monkeypatch,store=Order())
    assert r['reachable_count']==4


def test_wrong_actual_latest_action_is_not_published(monkeypatch):
    class Wrong(Env):
        def step(self,command):
            v=super().step(command);self.latest_action=np.zeros(2,np.float32);return v
    r,s,_=run(monkeypatch,base=Wrong)
    assert r['status']=='engineering_failed' and r['reachable_count']==0
    assert 'latest_action' in r['first_error']['message']
    assert not any('blind_inputs' in f for f in s.files)


def test_goal_pose_drift_never_yields_blind_or_fresh_qualification(monkeypatch):
    class Changed(Env):
        def _set_goal_state(self,goal):
            super()._set_goal_state(goal)
            if self._live_meter.region=='goal_bind':self.goal_pose[0]+=1.
    r,s,_=run(monkeypatch,base=Changed)
    assert r['status']=='engineering_failed' and not r['freshness']
    assert 'goal_pose' in r['first_error']['message'] and not any('blind_inputs' in f for f in s.files)


def test_prepare_verify_serialization_and_external_sha_in_RAM(monkeypatch):
    import hashlib
    output=m.ROOT/'results'/(m.NAME+'_RAM_prepare.json');memory={};original_open=m.Path.open
    class Buffer(io.BytesIO):
        def fileno(self):return 999
        def __exit__(self,*unused):memory['prepared']=self.getvalue();self.close()
    def opener(path,*a,**kw):
        if path==output:return Buffer()
        return original_open(path,*a,**kw)
    original_sha=m.sha;original_read=m.read
    monkeypatch.setattr(m.Path,'open',opener)
    monkeypatch.setattr(m.os,'fsync',lambda fd:None)
    monkeypatch.setattr(m.os,'open',lambda *a,**k:999)
    monkeypatch.setattr(m.os,'close',lambda fd:None)
    monkeypatch.setattr(m,'sha',lambda path:hashlib.sha256(memory['prepared']).hexdigest() if m.Path(path)==output else original_sha(path))
    result=m.prepare(output,mode='calibration_only')
    def read(path,expected=None,cap=4*1024**2):
        if m.Path(path)!=output:return original_read(path,expected,cap)
        m.require(expected==m.sha(path),'external RAM prepare SHA')
        return m.json.loads(memory['prepared'])
    monkeypatch.setattr(m,'read',read)
    assert m.load_prepared(output,result['prepare_sha256'])['slots']==m.slots()
    with pytest.raises(ValueError):m.load_prepared(output,'0'*64)


def test_no_run_authorization_no_inputs_or_native_factory():
    with pytest.raises(ValueError):m.run('absent','bad','absent',authorization='no')
