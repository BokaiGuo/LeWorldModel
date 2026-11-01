"""Array fixtures establish a contract, not a model task or physical result."""
from copy import deepcopy
import json
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p1_conditional_task_admission_20261027_v1 as m


def context():
    return dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,
        action_dim=10,latent_history=np.zeros((3,192)).tolist(),action_history=np.zeros((2,10)).tolist())


def task():
    c=context();raw=np.zeros((10,2))
    c['action_history']=((raw.astype(np.float32).reshape(2,10)-np.tile(np.array([-.0087,.0068],np.float32),5))/
        np.tile(np.array([.2019,.2002],np.float32),5)).tolist()
    return dict(environment='array_fixture_not_PushT',task_contract_sha256='f'*64,root_context=c,
        root_raw_action_history=raw.tolist(),goal=[0.]*192,
        projection=np.eye(192).tolist(),model_sha256='a'*64,model_config_sha256='b'*64,
        metric_source_sha256='c'*64,goal_source_sha256='d'*64,
        representation='native_projected_cls',latent_dim=192,horizon=2,packet_steps=5,control_dim=2,
        action_mean=[-.0087,.0068],action_std=[.2019,.2002],packet_logit_cap=float(.75*np.sqrt(5)),
        feedback='open_loop_model_only')


def prediction(t,value=.5):
    logits=np.zeros((2,10));raw=np.tanh(logits.astype(np.float32)).reshape(10,2)
    tokens=m.normalized_tokens(t,raw)
    initial=deepcopy(t['root_context']);contexts=[]
    for k in range(2):
        z=np.zeros(192);z[0]=value
        initial=dict(initial,latent_history=initial['latent_history'][1:]+[z.tolist()],
            action_history=initial['action_history'][1:]+[tokens[k].tolist()])
        contexts.append(deepcopy(initial))
    return dict(task_id=m.task_id(t),logits=logits.tolist(),commands=raw.tolist(),
        contexts=contexts,whole_replay=deepcopy(contexts[-1]),segmented_replay=deepcopy(contexts),
        cost=value*value/192)


def policy():
    entries=[]
    for i,split in enumerate(('calibration','confirmation','confirmation')):
        entries.append(dict(candidate_id=i,root_sha256=str(i)*64,task_id=m.task_id(task()),
            source=dict(kind='dataset_episode',dataset_sha256='e'*64,episode=i,start_row=0),split=split))
    return dict(schema=m.NAME+'.policy',environment='array_fixture_not_PushT',
        task_contract_sha256='f'*64,direct=dict(method='full_H2_GN',implementation_sha256='1'*64,configuration_sha256='2'*64,
            B1=1,B2=2,initial_logits=[np.zeros((2,10)).tolist()],incumbent_rule='strict_improvement'),
        scan=entries,metadata_scan=None,pool_counts=dict(calibration=1,confirmation=2),required_confirmation=2,
        excluded_roots=[],excluded_sources=[],exclusion_sources={'history_manifest':'3'*64},
        budgets=dict(direct=dict(predict=22,action_encoder=22,jacobian=2,svd=2),
            producer=dict(predict=2,action_encoder=2,jacobian=0,svd=0),
            witness=dict(predict=6,action_encoder=6,jacobian=0,svd=0)),
        source_bindings={'runner':'4'*64,'witness_producer':'6'*64},physical_future_labels_read=False)


def contract(p=None):
    p=policy() if p is None else p
    calibration=[e for e in m.source_pool(p)['accepted'] if e['split']=='calibration']
    authority=dict(schema=m.NAME+'.threshold_authority',environment=p['environment'],
        policy_id=m.digest(p),task_contract_sha256=p['task_contract_sha256'],
        values=dict(epsilon_plateau=.00001,epsilon_escape=.0001,tau=.0001),
        calibration_roots=[e['root_sha256'] for e in calibration],calibration_sources=[deepcopy(e['source']) for e in calibration],derivation=None,
        source_bindings={'external_reviewed_fixture_protocol':'5'*64},
        frozen_before_confirmation=True)
    return m.freeze_contract(p,authority,m.digest(authority))


def row(entry=None):
    entry=policy()['scan'][1] if entry is None else entry
    t=task();run=[]
    # Same finite incumbent continuously survives B1 -> B2, never restarted.
    for i in range(3):
        derivative=None if i==0 else dict(task_id=m.task_id(t),at_logits=np.zeros((2,10)).tolist(),
            at_terminal=prediction(t,.5)['contexts'][-1]['latent_history'][-1],
            raw_terminal_jacobian=np.zeros((192,20)).tolist(),objective_gradient=[0.]*20)
        run.append(dict(iteration=i,incumbent=prediction(t,.5),proposal=None if i==0 else prediction(t,.6),accepted=False,derivative=derivative))
    events=[]
    def charge(phase,evaluation_id,counts):
        for resource,n in counts.items():
            for _ in range(n):events.append(dict(index=len(events),phase=phase,evaluation_id=evaluation_id,
                resource=resource,samples=1,completed=True))
    for i in range(3):charge('direct',f'run_0_iter_{i}',dict(predict=6,action_encoder=6))
    for i in (1,2):charge('direct',f'opt_0_iter_{i}',dict(predict=2,action_encoder=2,jacobian=1,svd=1))
    charge('producer','producer',dict(predict=2,action_encoder=2))
    charge('witness','witness',dict(predict=6,action_encoder=6))
    return dict(identity=deepcopy(entry),status='completed',task=t,direct_runs=[run],
        witness=prediction(t,.1),journal=events,source_bindings=policy()['source_bindings'],physical_future_labels_read=False)


def test_threshold_authority_absence_is_unknown_no_reacher_defaults():
    c=m.freeze_contract(policy())
    assert c['status']=='unknown' and not c['ready']
    r=m.check_task(c,row())
    assert r['status']=='unknown' and not r['model_task_eligible'] and not r['P1_admission']
    assert 'missing_threshold_authority' in r['unknown_reasons']


def test_continuous_plateau_and_complete_h2_target_witness_are_only_model_screen():
    r=m.check_task(contract(),row())
    assert r['finite_budget_plateau'] and r['budget_escape'] and r['target_reached']
    assert r['model_task_eligible'] and not r['P1_admission']
    assert not r['physical_mechanism_established'] and not r['physical_efficacy_measured']


@pytest.mark.parametrize('defect',['q','goal','actions','partial_horizon','history','layout','replay','task_id','cap','score'])
def test_complete_task_witness_integrity(defect):
    r=row();w=r['witness']
    if defect=='q':r['task']['projection'][0][0]=0
    elif defect=='goal':r['task']['goal'][0]=1
    elif defect=='actions':w['commands'][0][0]=.2
    elif defect=='partial_horizon':w['contexts'].pop()
    elif defect=='history':w['contexts'][1]['action_history'][0][0]+=1e-12
    elif defect=='layout':w['contexts'][1]['layout']='wrong'
    elif defect=='replay':w['whole_replay']['latent_history'][-1][0]+=1e-12
    elif defect=='task_id':w['task_id']='9'*64
    elif defect=='cap':w['logits'][0]=[100.]*10
    else:w['cost']+=.1
    with pytest.raises(ValueError):m.check_task(contract(),r)


@pytest.mark.parametrize('defect',['budget_jump','restart_at_B1','lost_incumbent','wrong_initialization','missing_restart','incomplete','budget','source_binding'])
def test_direct_continuity_and_budget_receipts_fail_closed(defect):
    r=row()
    if defect=='budget_jump':r['direct_runs'][0][2]['iteration']=3
    elif defect=='restart_at_B1':r['direct_runs'][0][2]['incumbent']=prediction(r['task'],.7)
    elif defect=='lost_incumbent':r['direct_runs'][0][1]['accepted']=True
    elif defect=='wrong_initialization':r['direct_runs'][0][0]['incumbent']['logits'][0][0]=.1
    elif defect=='missing_restart':r['direct_runs']=[]
    elif defect=='incomplete':r['journal'][0]['completed']=False
    elif defect=='budget':r['journal'].pop()
    else:r['source_bindings']['runner']='0'*64
    with pytest.raises(ValueError):m.check_task(contract(),r)


def test_improvement_escape_alone_does_not_mean_target_reached():
    p=policy();c=contract(p);r=row();r['witness']=prediction(r['task'],.4)
    result=m.check_task(c,r)
    assert result['budget_escape'] and not result['target_reached']
    assert not result['model_task_eligible']


def test_source_episode_exclusion_and_fixed_scan_no_failure_replacement():
    p=policy();p['scan'].insert(0,dict(p['scan'][0],candidate_id=50,root_sha256='6'*64,
        source=dict(kind='dataset_episode',dataset_sha256='e'*64,episode=99,start_row=0)))
    p['excluded_sources']=[deepcopy(p['scan'][0]['source'])]
    pool=m.source_pool(p)
    assert pool['rejections'][0]['reason']=='historical_source_family'
    assert len(pool['accepted'])==3
    rows=[row(e) for e in pool['accepted']]
    rows[1]=dict(identity=rows[1]['identity'],status='failed',journal=[],error='model failed',physical_future_labels_read=False)
    result=m.select_cohort(contract(p),rows)
    assert result['selected_roots']==[] and result['status']=='incomplete_model_pool'
    assert len(result['rows'])==3


def test_calibration_confirmation_separation_and_insufficient_stop():
    c=contract();rows=[row(e) for e in policy()['scan']]
    result=m.select_cohort(c,rows)
    assert result['selected_roots']==['1'*64,'2'*64]
    assert '0'*64 not in result['selected_roots']
    rows[-1]['witness']=prediction(rows[-1]['task'],.5)
    result=m.select_cohort(c,rows)
    assert result['selected_roots']==[] and result['status']=='insufficient_model_eligible_roots'


def test_authority_cannot_reuse_confirmation_sources_or_reacher_environment():
    p=policy();c=contract();a=deepcopy(c['authority'])
    for key,value in [('environment','Reacher'),('calibration_roots',['1'*64]),('policy_id','0'*64)]:
        bad=deepcopy(a);bad[key]=value
        with pytest.raises(ValueError):m.freeze_contract(p,bad,m.digest(bad))
    with pytest.raises(ValueError):m.freeze_contract(p,a,'0'*64)


def test_forbidden_future_field_is_rejected_without_reading_its_value():
    r=row();r['physical_future']=object()
    with pytest.raises(ValueError):m.check_task(contract(),r)


def test_failed_row_work_is_counted_without_neural_retry():
    c=contract();rows=[row(e) for e in policy()['scan']]
    rows[1]=dict(identity=rows[1]['identity'],status='failed',journal=rows[1]['journal'][:3],
        error='fixture failure',physical_future_labels_read=False)
    r=m.select_cohort(c,rows)
    assert r['selected_roots']==[] and r['resources']['attempted']['predict']==63
    assert r['resources']['completed']['predict']==63


def test_current_four_roots_have_unknown_preconditions_not_negative_admission():
    r=m.audit_current(m.PROJECT)
    assert len(r['roots'])==4 and len(r['arms'])==8
    assert r['candidate_count']==80 and r['gradient_plateau']=='unverified'
    assert r['two_stage_target_reachability']=='unverified' and not r['fresh_conditional_cohort']
    assert not r['P1_admission'] and not r['model_task_eligible']
    assert sum(len(a['greedy_tradeoff_indices']) for a in r['arms'])==3


def test_import_is_array_only():
    script='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert not any(x in sys.modules for x in ("torch","gymnasium","pymunk","h5py"))'
    subprocess.run([sys.executable,'-c',script],check=True)


@pytest.mark.parametrize('bad',['H1_only','wrong_gradient','wrong_condition'])
def test_direct_fullH2_derivative_binding(bad):
    r=row();d=r['direct_runs'][0][1]['derivative']
    if bad=='H1_only':d['raw_terminal_jacobian']=np.zeros((192,10)).tolist()
    elif bad=='wrong_gradient':d['objective_gradient'][0]=1
    else:d['at_terminal'][0]+=.1
    with pytest.raises(ValueError):m.check_task(contract(),r)


def procedural_policy():
    p=policy()
    p['scan']=[dict(candidate_id=i,root_sha256=e['root_sha256'],task_id=e['task_id'],split=e['split'],
        source=dict(kind='procedural_seed',generator_sha256='a'*64,environment_sha256='b'*64,
            reset_recipe_sha256='c'*64,root_seed=800+i,reset_seed_span=[800+i,801+i],proposal_index=0)) for i,e in enumerate(p['scan'])]
    p['excluded_sources']=[]
    p['metadata_scan']=scan_bundle(p)
    p['source_bindings']['metadata_scan_plan']=p['metadata_scan']['plan_sha256']
    return p


def test_procedural_roots_do_not_require_dataset_episode():
    p=procedural_policy();pool=m.source_pool(p)
    assert pool['complete'] and len(pool['accepted'])==3
    assert 'source_episode' not in pool['accepted'][0]


def test_procedural_seed_family_excluded_and_never_shared_between_splits():
    p=procedural_policy()
    p['scan'][1]['source']['root_seed']=800
    p['scan'][1]['source']['proposal_index']=7
    pool=m.source_pool(p)
    assert not pool['complete'] and pool['rejections'][0]['reason']=='pool_source_family_duplicate'
    p=procedural_policy();p['excluded_sources']=[deepcopy(p['scan'][0]['source'])]
    p['excluded_sources'][0]['generator_sha256']='9'*64
    assert m.source_pool(p)['rejections'][0]['reason']=='historical_source_family'


def recipe():
    return dict(schema=m.NAME+'.calibration_recipe',environment=policy()['environment'],
        task_contract_sha256='f'*64,method='empirical_quantiles_v1',minimum_rows=1,
        quantiles=dict(epsilon_plateau=.5,epsilon_escape=.5,tau=.5),
        offsets=dict(epsilon_plateau=0.,epsilon_escape=0.,tau=0.),
        source_bindings={'derivation_code':'7'*64},frozen_before_calibration=True)


def test_independent_arrays_can_derive_freezable_threshold_authority():
    p=procedural_policy();r=recipe();p['source_bindings']['calibration_recipe']=m.digest(r)
    rows=[row(p['scan'][0])];rows[0]['source_bindings']=p['source_bindings']
    a=m.calibrate_authority(p,r,rows,m.digest(r))
    c=m.freeze_contract(p,a,m.digest(a))
    assert c['ready'] and a['values']['tau']==rows[0]['witness']['cost']
    assert a['values']['epsilon_plateau']==0
    bad=deepcopy(a);bad['values']['tau']+=1
    with pytest.raises(ValueError):m.freeze_contract(p,bad,m.digest(bad))


def test_calibration_rejects_confirmation_and_undeclared_recipe_or_future():
    p=procedural_policy();r=recipe();p['source_bindings']['calibration_recipe']=m.digest(r)
    confirmation=row(p['scan'][1]);confirmation['source_bindings']=p['source_bindings']
    with pytest.raises(ValueError):m.calibrate_authority(p,r,[confirmation],m.digest(r))
    with pytest.raises(ValueError):m.calibrate_authority(p,r,[], '0'*64)
    r['physical_future']=object()
    with pytest.raises(ValueError):m.calibrate_authority(p,r,[],None)


def test_current_audit_distinguishes_procedural_provenance_and_real_missing_fields():
    a=m.audit_current()
    assert a['provenance']['source_episode']=='not_applicable_procedural_roots'
    assert a['provenance']['root_seed']==202753010
    assert a['provenance']['per_root_proposal_index']=='not_recorded'
    for r in a['roots']:
        assert r['source_episode']=='not_applicable_procedural_roots'
        assert 'direct_runs' in r['missing_screening_fields']
        assert 'witness' in r['missing_screening_fields']


def scan_bundle(p=None):
    p=procedural_policy() if p is None else p
    plan=dict(schema=m.NAME+'.procedural_scan',generator_sha256='a'*64,environment_sha256='b'*64,
        reset_recipe_sha256='c'*64,proposal_limit_per_seed=16,physical_future_labels_read=False,
        frozen_before_generation=True,slots=[dict(candidate_id=e['candidate_id'],root_seed=e['source']['root_seed'],
            reset_seed_span=e['source']['reset_seed_span'],split=e['split']) for e in p['scan']])
    receipts=[dict(identity=deepcopy(e),status='completed',error=None) for e in p['scan']]
    return dict(plan=plan,plan_sha256=m.digest(plan),receipts=receipts)


def test_finite_procedural_scan_checks_all_slots_and_failures_without_replacement():
    b=scan_bundle()
    assert m.procedural_scan(**b)['identities']==procedural_policy()['scan']
    b['receipts'][1]['status']='failed';b['receipts'][1]['error']='initial render failed'
    b['receipts'][1]['identity']['root_sha256']=None;b['receipts'][1]['identity']['task_id']=None
    result=m.procedural_scan(**b)
    assert len(result['identities'])==2 and result['failed_slots']==1
    b['receipts'].pop()
    with pytest.raises(ValueError):m.procedural_scan(**b)


@pytest.mark.parametrize('defect',['seed','order','proposal','hash','extra_future'])
def test_procedural_scan_cannot_expand_or_change_declared_seed_reset_pool(defect):
    b=scan_bundle()
    if defect=='seed':b['receipts'][0]['identity']['source']['root_seed']+=99
    elif defect=='order':b['receipts'].reverse()
    elif defect=='proposal':b['receipts'][0]['identity']['source']['proposal_index']=16
    elif defect=='hash':b['plan_sha256']='0'*64
    else:b['receipts'][0]['future']=object()
    with pytest.raises(ValueError):m.procedural_scan(**b)


def test_distinct_parent_seeds_cannot_reuse_underlying_random_reset_span():
    p=procedural_policy();p['scan'][1]['source']['reset_seed_span']=[800,802]
    assert not m.source_pool(p)['complete']
    b=scan_bundle();b['plan']['slots'][1]['reset_seed_span']=[800,802]
    b['receipts'][1]['identity']['source']['reset_seed_span']=[800,802]
    b['plan_sha256']=m.digest(b['plan'])
    with pytest.raises(ValueError):m.procedural_scan(**b)


def test_root_history_also_requires_strict_normalized_raw_action_binding():
    t=task();t['root_context']['action_history'][0][0]+=1e-12
    with pytest.raises(ValueError):m.validate_task(t)


def test_cli_calibrate_freeze_screen_and_exclusive_output(tmp_path):
    p=procedural_policy();r=recipe();p['source_bindings']['calibration_recipe']=m.digest(r)
    rows=[row(e) for e in p['scan']]
    for item in rows:item['source_bindings']=deepcopy(p['source_bindings'])
    def invoke(flag,payload,out):
        source=tmp_path/(flag+'.input.json');source.write_text(json.dumps(payload))
        return subprocess.run([sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,
            '--'+flag,str(source),'--output',str(out)],capture_output=True,text=True)
    out=tmp_path/'authority.json'
    result=invoke('calibrate',dict(policy=p,recipe=r,rows=rows[:1],recipe_sha256=m.digest(r)),out)
    assert result.returncode==0,result.stderr
    a=json.loads(out.read_text());frozen=tmp_path/'contract.json'
    assert invoke('freeze',dict(policy=p,authority=a,authority_sha256=m.digest(a)),frozen).returncode==0
    c=json.loads(frozen.read_text());screened=tmp_path/'screen.json'
    # Calibrated strict escape equals this fixture's improvement, so the
    # fixture MUST NOT pass merely because calibration produced thresholds.
    result=invoke('screen',dict(contract=c,rows=rows),screened)
    assert result.returncode==2
    assert json.loads(screened.read_text())['selected_roots']==[]
    saved=screened.read_bytes()
    assert invoke('screen',dict(contract=c,rows=rows),screened).returncode!=0
    assert screened.read_bytes()==saved
