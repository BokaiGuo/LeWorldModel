"""Pure metadata/array fake-environment contracts; no real generator execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import p1_conditional_procedural_source_20261027_v1 as m


def exclusions():
    return dict(schema=m.NAME+'.exclusions',environment='swm/PushT-v1',
        files={'fixture_metadata':'a'*64,'fixture_coverage_review':'b'*64},root_hashes=[],seed_families=[],reset_spans=[],
        unknown_sources=[],coverage=dict(status='externally_reviewed_complete',authority_sha256='b'*64,
            scope='array_fixture_not_real_history'),physical_future_labels_read=False)


def plan():
    path=Path(m.__file__).resolve();backend=Path(__file__).resolve()
    closure={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (path,backend)}
    return dict(schema=m.NAME+'.plan',environment='swm/PushT-v1',purpose='fresh_confirmation',
        generator_sha256=closure[str(path)],environment_sha256=closure[str(backend)],source_closure=closure,
        reset_semantics=dict(internal_physics_steps_per_reset=0,internal_renders_per_reset=0,authority_source_sha256=closure[str(backend)]),
        exclusion_id=m.digest(exclusions()),proposal_limit_per_seed=2,reset_attempts_per_proposal=2,
        placement_reset_seed=0,slots=[dict(candidate_id=i,root_seed=800+i,reset_seed_span=[900+4*i,904+4*i],
            split=s) for i,s in enumerate(('calibration','confirmation'))],
        recipe=dict(position_jitter=16.,angle_offset_range=[-.8,.8],edge_fraction_range=[.2,.8],
            pusher_surface_offset=15.),
        domain=dict(canvas_xy=[0.,512.],contact_gap_range=[-1.,1.],
            nonterminal_position_tolerance=1.,nonterminal_angle_tolerance=.01),
        history=dict(mode='repeat_current_rgb_causal_padding_v1',frames=3,packet_steps=5,
            zero_raw_actions_are_padding=True,public_warmup_steps=0),
        rgb_shape=[4,4,3],artifact_byte_limit=2_000_000,
        frozen_before_generation=True,physical_future_labels_read=False)


class Vertex:
    def __init__(self,x,y):self.x=x;self.y=y


class Shape:
    def get_vertices(self):return [Vertex(-10,-10),Vertex(10,-10),Vertex(10,10),Vertex(-10,10)]
    def point_query(self,point):
        class Query:distance=15.
        return Query()


class Body:
    shapes=[Shape()]


class AgentShape:radius=15.


class Agent:shapes=[AgentShape()]


class ArrayEnvironment:
    """Fake only: it has no dynamics and step is deliberately fatal."""
    def __init__(self,fail_reset=None):
        self.unwrapped=self;self.block=Body();self.agent=Agent();self.calls=[];self.fail_reset=fail_reset
    def reset(self,*,seed,options=None):
        self.calls.append(('reset',seed,options is not None))
        if len(self.calls)==self.fail_reset:raise RuntimeError('first reset failure')
        goal=np.array([200,200,250,250,0,0,0.],float)
        state=goal.copy()
        if options is not None:state=np.array(options['state']);goal=np.array(options['goal_state'])
        return {'state':state},{'goal_state':goal,'goal':np.full((4,4,3),80,np.uint8)}
    def render(self):self.calls.append(('render',));return np.full((4,4,3),100,np.uint8)
    def step(self,*args):raise AssertionError('public physics steps forbidden in this version')


def execute(tmp_path,p=None,e=None,env=None):
    p=plan() if p is None else p;e=exclusions() if e is None else e
    env=ArrayEnvironment() if env is None else env
    out=tmp_path/'source'
    r=m.run_main_only(p,e,out,env,authorization=m.MAIN_AUTHORIZATION,
        expected_plan_sha256=m.digest(p),expected_exclusion_sha256=m.digest(e))
    return r,out,env


def test_metadata_plan_has_finite_exact_reset_schedule():
    r=m.validate_plan(plan(),exclusions())
    assert r['ready'] and r['maximum_random_resets']==8 and r['maximum_total_resets']==16
    assert r['maximum_public_steps']==0


@pytest.mark.parametrize('defect',['span','seed','overlap','warmup','future','bound','source','exclusion'])
def test_plan_corruption_or_undeclared_physics_rejected(defect):
    p=plan()
    if defect=='span':p['slots'][0]['reset_seed_span'][1]+=1
    elif defect=='seed':p['slots'][1]['root_seed']=800
    elif defect=='overlap':p['slots'][1]['reset_seed_span']=[901,905]
    elif defect=='warmup':p['history']['public_warmup_steps']=10
    elif defect=='future':p['future_success']=object()
    elif defect=='bound':p['proposal_limit_per_seed']=True
    elif defect=='source':p['generator_sha256']='0'*64
    else:p['exclusion_id']='0'*64
    with pytest.raises(ValueError):m.validate_plan(p,exclusions())


def test_unknown_historical_sources_failclosed_before_any_reset(tmp_path):
    e=exclusions();e['unknown_sources']=[dict(source='missing_legacy_seed',reason='unknown')]
    p=plan();p['exclusion_id']=m.digest(e);env=ArrayEnvironment()
    assert not m.validate_plan(p,e)['ready']
    with pytest.raises(ValueError):execute(tmp_path,p,e,env)
    assert env.calls==[] and not (tmp_path/'source').exists()


@pytest.mark.parametrize('kind',['seed','span','reserved'])
def test_used_and_reserved_source_exclusions_apply_before_generation(kind):
    e=exclusions()
    if kind=='seed':e['seed_families']=[dict(seed=800,usage='used',evidence='a'*64)]
    else:e['reset_spans']=[dict(span=[899,902],usage='reserved_not_opened' if kind=='reserved' else 'used',evidence='a'*64)]
    p=plan();p['exclusion_id']=m.digest(e)
    r=m.validate_plan(p,e)
    assert not r['ready'] and r['blocked_slots'][0]['candidate_id']==0


def test_fake_entry_archives_every_proposal_attempt_and_causal_raw_source(tmp_path):
    r,out,env=execute(tmp_path)
    assert r['status']=='completed_source_only' and r['new_public_steps']==0
    assert r['resources']['reset']==dict(attempted=16,completed=16)
    assert len(r['slots'])==2 and len(r['slots'][0]['proposals'])==2
    assert all(len(x['attempts'])==2 for x in r['slots'][0]['proposals'])
    events=m.read_journal(out/'journal.jsonl')
    assert events[0]['kind']=='run_started' and events[-1]['kind']=='run_finished'
    assert len([e for e in events if e['kind']=='operation_started' and e['operation']=='random_reset'])==8
    assert len([e for e in events if e['kind']=='proposal_started'])==4
    packet=m.load_raw_source(out/r['slots'][0]['raw_source_file'],r['plan'])
    assert packet['rgb_history'].shape==(3,4,4,3)
    assert np.array_equal(packet['rgb_history'][0],packet['rgb_history'][2])
    assert np.count_nonzero(packet['raw_action_history'])==0
    assert r['history_semantics']=='causal_padding_not_observed_transitions'
    assert not r['freshness_certified_globally'] and not r['P1_admission']
    assert packet['metadata']['selected_attempt']['random_reset_seed']==900


def test_failure_first_error_kept_and_no_hidden_retry_or_replacement(tmp_path):
    r,out,env=execute(tmp_path,env=ArrayEnvironment(fail_reset=1))
    assert r['status']=='failed' and r['first_error']['message']=='first reset failure'
    assert len(env.calls)==1
    assert r['resources']['reset']==dict(attempted=1,completed=0)
    assert r['slots'][0]['status']=='failed' and r['slots'][1]['status']=='not_started'
    assert m.read_journal(out/'journal.jsonl')[-1]['kind']=='run_finished'
    assert json.loads((out/'sourcepack.json').read_text())['first_error']==r['first_error']


def test_durable_prefix_exists_before_reset_and_after_each_attempt(tmp_path):
    class CheckingEnv(ArrayEnvironment):
        def reset(self,**kw):
            events=m.read_journal(tmp_path/'source'/'journal.jsonl')
            assert events[-1]['kind']=='operation_started'
            return super().reset(**kw)
    execute(tmp_path,env=CheckingEnv())


def test_refuses_output_overwrite_and_missing_execution_authorization(tmp_path):
    p=plan();env=ArrayEnvironment()
    with pytest.raises(ValueError):m.run_main_only(p,exclusions(),tmp_path/'source',env,authorization='',
        expected_plan_sha256=m.digest(p),expected_exclusion_sha256=m.digest(exclusions()))
    r,out,env=execute(tmp_path)
    saved=(out/'sourcepack.json').read_bytes()
    with pytest.raises(FileExistsError):m.run_main_only(p,exclusions(),out,env,authorization=m.MAIN_AUTHORIZATION,
        expected_plan_sha256=m.digest(p),expected_exclusion_sha256=m.digest(exclusions()))
    assert (out/'sourcepack.json').read_bytes()==saved


def test_current_metadata_audit_excludes_more_than_latest_four_without_reading_outcomes():
    e=m.collect_current_exclusions()
    assert {202610241,202710241,202753000,202753010}<={r['seed'] for r in e['seed_families']}
    assert {202610281,202710281,202753100}<={r['seed'] for r in e['seed_families'] if r['usage']=='reserved_not_opened'}
    assert len(e['root_hashes'])>=20 and e['unknown_sources']
    assert e['coverage']['status']=='unknown' and not e['physical_future_labels_read']
    assert 'metadata_authority_content_id' in e['files']


def test_import_has_no_model_environment_or_hidden_factory_dependency():
    code='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert not any(x in sys.modules for x in ("torch","gymnasium","pymunk","stable_worldmodel"))'
    subprocess.run([sys.executable,'-c',code],check=True)


def test_sourcepack_census_and_array_bridge_to_existing_admission(tmp_path):
    from experiments.spectral_reachability import p1_conditional_task_admission_20261027_v1 as a
    r,out,env=execute(tmp_path)
    ids={row['metadata']['root_sha256']:str(i+1)*64 for i,row in enumerate(r['slots'])}
    bundle=m.to_admission_scan(r['plan'],exclusions(),r,out,ids)
    census=a.procedural_scan(**bundle)
    assert len(census['identities'])==2
    assert census['identities'][0]['source']['reset_seed_span']==[900,904]
    assert census['identities'][0]['task_id']=='1'*64


@pytest.mark.parametrize('defect',['missing_attempt','wrong_seed','raw_hash','journal','extra_task','failed_run'])
def test_bridge_refuses_partial_or_tampered_sourcepack(defect,tmp_path):
    r,out,env=execute(tmp_path);ids={row['metadata']['root_sha256']:str(i+1)*64 for i,row in enumerate(r['slots'])}
    if defect=='missing_attempt':r['slots'][0]['proposals'][0]['attempts'].pop()
    elif defect=='wrong_seed':r['slots'][0]['proposals'][0]['attempts'][0]['random_reset_seed']+=1
    elif defect=='raw_hash':r['slots'][0]['raw_source_sha256']='0'*64
    elif defect=='journal':r['journal_sha256']='0'*64
    elif defect=='extra_task':ids['f'*64]='a'*64
    else:r['status']='failed'
    with pytest.raises(ValueError):m.to_admission_scan(r['plan'],exclusions(),r,out,ids)


def test_journal_torn_or_changed_prefix_never_becomes_completed(tmp_path):
    r,out,env=execute(tmp_path);path=out/'journal.jsonl'
    raw=path.read_bytes();path.write_bytes(raw[:-1])
    with pytest.raises(ValueError):m.read_journal(path)
    path.write_bytes(raw.replace(b'run_started',b'RUN_started',1))
    with pytest.raises(ValueError):m.read_journal(path)


def test_secondary_journal_cleanup_errors_do_not_replace_first_reset_failure(tmp_path,monkeypatch):
    original=m.DurableJournal.append
    def faulty(self,kind,**kwargs):
        if kind in ('run_failed','run_finished'):raise OSError('secondary journal failure')
        return original(self,kind,**kwargs)
    monkeypatch.setattr(m.DurableJournal,'append',faulty)
    r,out,env=execute(tmp_path,env=ArrayEnvironment(fail_reset=1))
    assert r['first_error']['message']=='first reset failure'
    assert r['secondary_errors'] and not r['durable_journal_complete']


def test_no_real_environment_cli_entry(tmp_path):
    result=subprocess.run([sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,'--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--audit-exclusions' in result.stdout
    assert '--run-main' in result.stdout and '--authorize-reset-render' in result.stdout


def test_cli_unknown_source_gate_precedes_environment_import_or_creation(tmp_path):
    e=exclusions();e['unknown_sources']=[dict(source='untraced_history',reason='unknown')]
    p=plan();p['exclusion_id']=m.digest(e)
    request=tmp_path/'request.json';request.write_text(json.dumps(dict(plan=p,exclusions=e,
        plan_sha256=m.digest(p),exclusion_sha256=m.digest(e))))
    result=subprocess.run([sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,
        '--run-main',str(request),'--authorize-reset-render','--output',str(tmp_path/'never_created')],capture_output=True,text=True)
    assert result.returncode==2,result.stderr
    assert json.loads(result.stdout)['status']=='unknown'
    assert not (tmp_path/'never_created').exists()


def test_storage_floor_checked_before_any_reset(tmp_path,monkeypatch):
    from collections import namedtuple
    Usage=namedtuple('Usage','total used free')
    monkeypatch.setattr(m.shutil,'disk_usage',lambda path:Usage(1000,900,100))
    env=ArrayEnvironment()
    with pytest.raises(ValueError):execute(tmp_path,env=env)
    assert env.calls==[] and not (tmp_path/'source').exists()


def test_initial_domain_rejections_keep_full_scan_and_bridge_empty_pool(tmp_path):
    p=plan();p['domain']['contact_gap_range']=[1.,2.]
    r,out,env=execute(tmp_path,p=p)
    assert r['status']=='completed_source_only'
    assert all(row['status']=='metadata_rejected' for row in r['slots'])
    assert r['resources']['reset']['completed']==16 and r['resources']['render']['completed']==8
    assert all(not attempt['domain']['checks']['contact'] for row in r['slots'] for prop in row['proposals'] for attempt in prop['attempts'])
    bundle=m.to_admission_scan(p,exclusions(),r,out,{})
    assert all(x['status']=='metadata_rejected' and x['identity']['root_sha256'] is None for x in bundle['receipts'])


def test_raw_source_records_exact_selected_attempt_after_root_exclusion(tmp_path):
    base,out,env=execute(tmp_path)
    excluded=base['slots'][0]['metadata']['root_sha256']
    e=exclusions();e['root_hashes']=[excluded];p=plan();p['exclusion_id']=m.digest(e)
    second=tmp_path/'second';second.mkdir()
    r,out,env=execute(second,p=p,e=e)
    assert r['slots'][0]['proposals'][0]['attempts'][0]['root_excluded_or_duplicate']
    packet=m.load_raw_source(out/r['slots'][0]['raw_source_file'],p)
    assert packet['metadata']['selected_attempt']['attempt_index']==1
    assert packet['metadata']['selected_attempt']['random_reset_seed']==901


def test_render_failure_cannot_relabel_completed_resets_as_unattempted(tmp_path):
    class BrokenRender(ArrayEnvironment):
        def render(self):raise RuntimeError('render failed after placement')
    r,out,env=execute(tmp_path,env=BrokenRender())
    assert r['first_error']['message']=='render failed after placement'
    assert r['resources']==dict(reset=dict(attempted=2,completed=2),render=dict(attempted=1,completed=0))
    assert r['slots'][0]['proposals'][0]['attempts'][0]['status']=='started'
    assert r['status']=='failed'


def test_source_drift_refused_before_reset(tmp_path):
    p=plan();test_path=str(Path(__file__).resolve());p['source_closure'][test_path]='0'*64
    env=ArrayEnvironment()
    with pytest.raises(ValueError):execute(tmp_path,p=p,env=env)
    assert env.calls==[]


def test_journal_failure_before_reset_never_performs_unlogged_work(tmp_path,monkeypatch):
    original=m.DurableJournal.append
    def fail_before_call(self,kind,**data):
        if kind=='operation_started':raise OSError('durable start failed')
        return original(self,kind,**data)
    monkeypatch.setattr(m.DurableJournal,'append',fail_before_call)
    r,out,env=execute(tmp_path)
    assert not env.calls and r['resources']['reset']['attempted']==0
    assert r['first_error']['message']=='durable start failed'


def test_calibration_exploration_can_generate_without_claiming_global_confirmation(tmp_path):
    e=exclusions();e['coverage']['status']='unknown';e['coverage']['authority_sha256']=None
    e['unknown_sources']=[dict(source='legacy_root',reason='reset_attempt_receipt_missing')]
    p=plan();p['purpose']='calibration_exploration';p['exclusion_id']=m.digest(e)
    for slot in p['slots']:slot['split']='calibration'
    checked=m.validate_plan(p,e)
    assert checked['ready'] and checked['status']=='calibration_exploration_only'
    assert not checked['fresh_confirmation_ready']
    r,out,env=execute(tmp_path,p=p,e=e)
    assert r['status']=='completed_calibration_source_only' and not r['freshness_certified_globally']
    assert all(row['slot']['split']=='calibration' for row in r['slots'])
    ids={row['metadata']['root_sha256']:str(i+1)*64 for i,row in enumerate(r['slots'])}
    bundle=m.to_admission_scan(p,e,r,out,ids)
    assert all(x['identity']['split']=='calibration' for x in bundle['receipts'])
    p['purpose']='fresh_confirmation'
    assert not m.validate_plan(p,e)['ready']


def test_calibration_mode_cannot_smuggle_confirmation_slots_or_known_reuse(tmp_path):
    p=plan();p['purpose']='calibration_exploration'
    with pytest.raises(ValueError):m.validate_plan(p,exclusions())
    for slot in p['slots']:slot['split']='calibration'
    e=exclusions();e['seed_families']=[dict(seed=800,usage='used',evidence='a'*64)]
    p['exclusion_id']=m.digest(e)
    assert not m.validate_plan(p,e)['ready']


def test_reset_internal_physics_is_not_conflated_with_zero_public_steps(tmp_path):
    p=plan();p['reset_semantics']['internal_physics_steps_per_reset']=2
    p['reset_semantics']['internal_renders_per_reset']=1
    checked=m.validate_plan(p,exclusions())
    assert checked['maximum_reset_internal_physics_steps']==32
    r,out,env=execute(tmp_path,p=p)
    assert r['new_public_steps']==0
    assert r['reset_internal_physics_steps']['completed_reset_implied']==32
    assert r['reset_internal_physics_steps']['actual_observed'] is None


def test_prepare_calibration_is_metadata_only_and_binds_real_environment_reset_code(tmp_path):
    (tmp_path/'results').mkdir()
    environment=m.PROJECT/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py'
    base=plan()
    config=dict(root_seeds=[17001,17002],reset_seed_start=18000,proposal_limit_per_seed=2,
        reset_attempts_per_proposal=2,placement_reset_seed=0,recipe=base['recipe'],domain=base['domain'],
        rgb_shape=[224,224,3],artifact_byte_limit=8*1024**2,
        environment_source_path=str(environment),source_closure_paths=[])
    request=m.prepare_calibration_request(config,tmp_path)
    assert request['plan']['purpose']=='calibration_exploration'
    assert request['plan']['reset_semantics']['internal_physics_steps_per_reset']==2
    assert request['plan']['reset_semantics']['internal_renders_per_reset']==1
    assert m.validate_plan(request['plan'],request['exclusions'])['ready']
    assert not m.validate_plan(request['plan'],request['exclusions'])['fresh_confirmation_ready']
