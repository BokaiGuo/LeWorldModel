"""RAM and bounded saved-metadata ABI only; no upstream consume/model/native run."""
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p2_subgoal_gap_protocol_20261029_v1 as m


def budget(mode=m.MODES[0],cap=4,batch=3):
    return dict(mode=mode,cap=cap,matched_model_sha256='1'*64,matched_callable_sha256='2'*64,
        batch_limit=batch,scope=list(m.PHASES),failed_attempts_refunded=False)


def put(path,obj):
    path.write_bytes(m.raw(obj));return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def minimal_plan(root,arm):
    return dict(root_slot=root,arm=arm,raw_steps=15,program_count=3,compiled_nodes=[0,5,10],condition='off',
        edges=[{},{},{}],task_id=str(root)*64,task_file=f'/RAM/task{root}.json',task_sha256='3'*64,
        model_hash='1'*64,root_context=dict(fixture_history=root),goal=[root],normalization=dict(fixture=True),selection_digest='4'*64)


@pytest.fixture
def catalog(tmp_path):
    paths={k:tmp_path/(k+'.json') for k in m.CATALOG}
    cr=put(paths['compiled_report'],dict(schema='fixture.report'))
    hr=put(paths['prior_report'],dict(schema='fixture.Hi.report'))
    plans=[minimal_plan(r,a) for r in (0,1) for a in ('joint_H3_no_composition','three_program_imagined_recompile')]
    put(paths['compiled_receipt'],dict(schema='agent_p2_three_program_20261028_v1.consume',plans=plans,planner_report_sha256=cr['sha256']))
    put(paths['prior_receipt'],dict(schema='agent_hilewm_low_cem_20261029_v1.consume',physical_release=False,report_sha256=hr['sha256']))
    put(paths['prior_rgb_receipt'],dict(schema='RAM_high_FIRST'))
    put(paths['safe_weights_export'],dict(schema='RAM_export_NO_WEIGHTS'))
    return paths


def test_current_real_named_catalog_ABI_only_no_scientific_or_strict():
    # Explicitly allowed saved metadata check; no scores/outcomes used.
    inv=m.inventory()
    assert len(inv['compiled'])==2 and [p['root_slot'] for p in inv['compiled']]==[0,1]
    assert inv['full_strict_rebuilt'] is inv['checkpoint_loaded'] is False
    assert inv['new_neural_calls']==inv['new_physics_calls']==0
    assert inv['files']['prior_receipt']['status']=='metadata_read_NOT_full_strict'


def test_catalog_fixed_pointers_not_score_or_winner_selection(catalog):
    inv=m.inventory(catalog)
    assert [r['receipt_pointer'] for r in inv['compiled']]==['/plans/1','/plans/3']
    assert 'predicted_Q' not in str(inv) and 'native_success' not in str(inv)
    assert inv['files']['compiled_receipt']['sha256']==m.ref(catalog['compiled_receipt'])['sha256']


@pytest.mark.parametrize('change',['order','reportsha','program','nodes','root','model'])
def test_catalog_mutations_failclosed(catalog,change):
    p=catalog['compiled_receipt'];u=m.scoring.parse(p.read_bytes())
    if change=='order':u['plans'].reverse()
    elif change=='reportsha':u['planner_report_sha256']='0'*64
    elif change=='program':u['plans'][1]['program_count']=1
    elif change=='nodes':u['plans'][1]['compiled_nodes']=[0,10]
    elif change=='root':u['plans'][1]['root_slot']=1
    else:u['plans'][3]['model_hash']='0'*64
    p.write_bytes(m.raw(u))
    with pytest.raises(ValueError):m.inventory(catalog)


def test_missing_prior_is_missing_not_zero_or_random(catalog):
    paths=dict(catalog,prior_receipt=catalog['prior_receipt'].parent/'absent.json')
    inv=m.inventory(paths)
    assert inv['files']['prior_receipt']['sha256'] is inv['files']['prior_receipt']['bytes'] is None
    assert inv['files']['prior_receipt']['status']=='missing'


def test_default_prepare_actual_ABI_is_blocked_and_full6(tmp_path):
    p=m.prepare()
    assert p['status']=='blocked_missing_assets' and len(p['slots'])==6
    assert set(p['blockers'])=={'MAIN_predeclared_predict_mode_cap_scope_missing',*['missing_'+a for a in m.ASSETS]}
    assert p['contract']['budget'] is None and not p['contract']['independent_confirmation']
    assert not p['production_ready'] and not p['dispatch_allowed'] and not p['full_strict_rebuilt']
    assert all(r['proposal'] is r['observed'] is r['native_success'] is r['resources'] is None for r in p['slots'])
    dest=tmp_path/(m.NAME+'_RAM.json');info=m.write_new(dest,p)
    assert m.verify(dest,info['sha256'])['verified']
    with pytest.raises(ValueError):m.write_new(dest,p)


@pytest.mark.parametrize('change',['mode','cap','batch','scope','refund','sha'])
def test_budget_declaration_mutations(change):
    b=budget()
    if change=='mode':b['mode']='equal_FLOPs'
    elif change=='cap':b['cap']=True
    elif change=='batch':b['batch_limit']=0
    elif change=='scope':b['scope'].remove('AD_primal')
    elif change=='refund':b['failed_attempts_refunded']=True
    else:b['matched_model_sha256']='future SHA'
    with pytest.raises(ValueError):m.budget_check(b)


def test_meter_api_and_samples_distinct_failure_no_refund():
    b=budget(cap=2);meter=m.FixtureMeter(b)
    for n,failed in ((3,False),(1,True)):
        i=meter.before_dispatch('proposal',n,'1'*64,'2'*64);meter.after_return(i,failed=failed)
    assert meter.report()['attempted']==dict(predict_API=2,predictor_forward_samples=4)
    assert meter.report()['completed']==dict(predict_API=1,predictor_forward_samples=3)
    assert meter.report()['fixture_only'] and not meter.report()['real_meter_authenticated']
    with pytest.raises(ValueError,match='predispatch'):meter.before_dispatch('controller',1,'1'*64,'2'*64)
    assert meter.attempted['predict_API']==2


def test_meter_sample_cap_batch_pending_and_boundary():
    mtr=m.FixtureMeter(budget(m.MODES[1],cap=4))
    mtr.before_dispatch('AD_primal',3,'1'*64,'2'*64)
    with pytest.raises(ValueError):mtr.before_dispatch('whole_replay',2,'1'*64,'2'*64)
    assert mtr.report()['events'][0]['status']=='pending' and mtr.completed['predict_API']==0
    with pytest.raises(ValueError):mtr.before_dispatch('controller',4,'1'*64,'2'*64)
    with pytest.raises(ValueError):mtr.before_dispatch('controller',1,'9'*64,'2'*64)
    with pytest.raises(ValueError):mtr.after_return(-1)
    mtr.after_return(0)
    with pytest.raises(ValueError):mtr.after_return(0)


def rows():
    out=[]
    for root in (0,1):
        for arm in m.ARMS:
            out.append(dict(root_slot=root,arm=arm,fixture_only=True,target=[[0.]*192 for _ in m.CP],
                observed=[[.01]*192,None,[.03]*192],predicted=[[0.]*192 for _ in m.CP],native_success=False))
    return out


def test_fixture_tracking_full_denominator_native_and_tau_separate():
    r=rows();r[1].update(target=None,observed=None,predicted=None,native_success=None)
    p=m.fixture_measure(r)
    assert p['realization_ratio'] is None and len(p['rows'])==6
    assert p['rows'][1]['actual_tracking'] is None and p['rows'][1]['native_success'] is None
    assert p['rows'][0]['predicted_tracking'][0]['mse']==0 and p['rows'][0]['actual_tracking'][0]['l2']>0
    p=m.fixture_measure(r,tolerance=.2)
    assert p['realization_ratio'][0]==dict(step=5,denominator=6,known=5,realized=5)
    assert p['realization_ratio'][1]==dict(step=10,denominator=6,known=0,realized=0)
    assert p['realization_ratio'][2]['realized']==0
    assert p['rows'][0]['native_success'] is False


@pytest.mark.parametrize('change',['order','real','missing','observed','short','tau'])
def test_fixture_cannot_fake_prior_or_delete_missing(change):
    r=rows();t=.2
    if change=='order':r.reverse()
    elif change=='real':r[0]['fixture_only']=False
    elif change=='missing':r.pop()
    elif change=='observed':r[0]['observed'][0][0]=True
    elif change=='short':r[0]['predicted'].pop()
    else:t=float('nan')
    with pytest.raises(ValueError):m.fixture_measure(r,t)


def test_dependency_ref_source_bytes_not_pickle_or_weights(tmp_path):
    p=tmp_path/'native.py';p.write_text('# fixture source; never execute\n')
    r={k:v for k,v in m.ref(p).items() if k!='bytes'}
    assert m.check_ref(r)==p.read_bytes()
    with pytest.raises(ValueError):m.check_ref(dict(r,sha256='0'*64))
    weight=tmp_path/'no_load.pt';weight.write_bytes(b'not loaded')
    with pytest.raises(ValueError):m.ref(weight)


def test_actual_declaration_bound_but_cannot_self_release(tmp_path):
    d=dict(schema=m.NAME+'.declaration',route='current_P2_frozen192',budget=None,assets=dict.fromkeys(m.ASSETS))
    r=put(tmp_path/'declaration.json',d)
    p=m.prepare(r['path'],r['sha256'])
    assert p['arguments']['declaration']==r['path'] and not p['production_ready']
    d['route']='Hi_latent_cast_to_P2';r=put(tmp_path/'wrong.json',d)
    with pytest.raises(ValueError):m.prepare(r['path'],r['sha256'])
    with pytest.raises(ValueError):m.prepare(r['path'],None)


def test_asset_claims_do_not_bypass_actual_pairing(tmp_path):
    dep=put(tmp_path/'metadata.json',dict(fixture_only=True))
    obj=dict(schema=m.NAME+'.prior_pairing',model_hash='1'*64,bindings={k:dep for k in m.COMMON},
        method='HiLeWM_hierarchical_adaptation',fallback=False,raw_inverse_available=True)
    with pytest.raises(ValueError,match='pairing field'):m.asset_check('prior_pairing',obj,'1'*64)
    obj['method']='random Gaussian prior'
    with pytest.raises(ValueError):m.asset_check('prior_pairing',obj,'1'*64)


@pytest.fixture
def complete_declaration(tmp_path):
    # ALL refs are synthetic local metadata, never a real pairing certificate.
    dep=put(tmp_path/'coordinate.json',dict(fixture_only=True))
    code=tmp_path/'predictor.py';code.write_text('# RAM source, never executed\n')
    predictor={k:v for k,v in m.ref(code).items() if k!='bytes'}
    objects={role:dict(schema=m.NAME+'.'+role,model_hash='1'*64,bindings={k:dep for k in m.COMMON}) for role in m.ASSETS}
    objects['arbitrary_optimizer'].update(method='direct_latent_optimization',parameter_shape=[3,192],uses_action_rollout_as_target=False)
    objects['prior_pairing'].update(method='HiLeWM_hierarchical_adaptation',fallback=False,raw_inverse_available=True)
    objects['prior_pairing']['bindings'].update(target_delivery=dep,prior_checkpoint_inventory=dep)
    objects['common_controller'].update(checkpoint_steps=list(m.CP),same_callable_all_arms=True,privileged_compiled_warmstart=False)
    objects['common_controller']['bindings']['predictor_callable']=predictor
    objects['native_rebuild'].update(roots=[0,1],warmup=10,reset_saved_poststate=False)
    objects['tracking_criterion'].update(latent_l2_tolerance=.1,chosen_before_outcomes=True)
    b=budget();b['matched_callable_sha256']=predictor['sha256']
    return dict(schema=m.NAME+'.declaration',route='current_P2_frozen192',budget=b,
        assets={role:put(tmp_path/(role+'.json'),obj) for role,obj in objects.items()})


def test_complete_fixture_assets_still_never_self_authorize(complete_declaration,catalog,tmp_path,monkeypatch):
    inventory=m.inventory(catalog);monkeypatch.setattr(m,'inventory',lambda:deepcopy(inventory))
    r=put(tmp_path/'complete_declaration.json',complete_declaration)
    p=m.prepare(r['path'],r['sha256'])
    assert not p['blockers'] and p['status']=='ABI_prepared_MAIN_full_strict_and_runtime_review_pending'
    assert p['production_ready'] is p['dispatch_allowed'] is p['full_strict_rebuilt'] is False
    assert p['contract']['P1_admission'] is p['contract']['equal_FLOPs'] is False


@pytest.mark.parametrize('mutation',['coordinate','callable','model','raw_inverse','arbitrary','privilege','poststate','tolerance'])
def test_complete_asset_mutations_failclosed(complete_declaration,catalog,tmp_path,monkeypatch,mutation):
    inv=m.inventory(catalog);monkeypatch.setattr(m,'inventory',lambda:deepcopy(inv))
    d=deepcopy(complete_declaration)
    if mutation=='callable':d['budget']['matched_callable_sha256']='0'*64
    else:
        role={'coordinate':'prior_pairing','model':'prior_pairing','raw_inverse':'prior_pairing',
            'arbitrary':'arbitrary_optimizer','privilege':'common_controller',
            'poststate':'native_rebuild','tolerance':'tracking_criterion'}[mutation]
        obj=m.bound(d['assets'][role])
        if mutation=='coordinate':obj['bindings']['encoder']=put(tmp_path/'different_encoder.json',dict(other_coordinate=True))
        elif mutation=='model':obj['model_hash']='9'*64
        elif mutation=='raw_inverse':obj['raw_inverse_available']=False
        elif mutation=='arbitrary':obj['uses_action_rollout_as_target']=True
        elif mutation=='privilege':obj['privileged_compiled_warmstart']=True
        elif mutation=='poststate':obj['reset_saved_poststate']=True
        else:obj['latent_l2_tolerance']=0
        d['assets'][role]=put(tmp_path/'changed_asset.json',obj)
    r=put(tmp_path/'changed_declaration.json',d)
    with pytest.raises(ValueError):m.prepare(r['path'],r['sha256'])


def test_cli_help_prepare_stdout_verify_and_tamper(tmp_path,capsys):
    dest=tmp_path/(m.NAME+'_CLI.json')
    m.main(['prepare','--output',str(dest)])
    m.main(['verify','--prepared',str(dest),'--prepare-sha',m.ref(dest)['sha256']])
    assert '"verified":true' in capsys.readouterr().out
    data=m.scoring.parse(dest.read_bytes());data['dispatch_allowed']=True
    changed=tmp_path/'changed.json';changed.write_bytes(m.raw(data))
    with pytest.raises(ValueError):m.verify(changed,m.ref(changed)['sha256'])


def test_cold_subprocess_forbids_production_dependencies_and_fullstrict():
    code='''
import builtins
original=builtins.__import__
def guarded(name,*a,**kw):
    assert name.split('.')[0] not in ('torch','numpy','gymnasium','pymunk','pygame','requests'),name
    assert 'agent_p2_three_program_20261028_v1' not in name,name
    assert 'agent_hilewm_low_cem_20261029_v1' not in name,name
    return original(name,*a,**kw)
builtins.__import__=guarded
from experiments.spectral_reachability import agent_p2_subgoal_gap_protocol_20261029_v1 as m
p=m.prepare()
assert p['full_strict_rebuilt'] is False and p['dispatch_allowed'] is False
'''
    run=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
