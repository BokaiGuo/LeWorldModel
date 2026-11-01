"""RAM protocol/identity/native arithmetic, no inference/physics or tmp files."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_baseline_matched_rgb_comparison_20261028_v1 as m


def plan(arm='line_search_GN',slot=0):
    return dict(source_slot=slot,arm=arm,task_export=f'/task{slot}',task_sha256=m.gn.EXPORT_SHAS[slot],
        task_id=f'task{slot}',blind_sha256='b'*64,commands=np.zeros((10,2),np.float32).tolist(),
        logits=np.zeros((2,10),np.float32).tolist(),selection_sha256='c'*64,model_terminal_cost=.5)


def record_fixture():
    p=plan();root=[100.,100.,200.,200.,0.,3.,4.];goal=[0.]*7
    checkpoints=[dict(step=k,state=deepcopy(root),observation_state=deepcopy(root),snapshot={'state':deepcopy(root)},
        pixel_sha256='d'*64,file_sha256='e'*64) for k in (0,5,10)]
    metric=m.native(goal,root)
    row=dict(root_slot=0,arm=p['arm'],initial_exact=True,full_horizon=True,own_returned=10,first_error=None,secondary_errors=[],
        checkpoints=checkpoints,root_state=root,goal=goal,initial_metrics=metric,final_metrics=metric,native_success=False,
        selection_digest='a'*64,status='horizon_completed')
    task=dict(root_context={'layout':'states_with_between_state_actions_v1','latent_history':np.zeros((3,192)).tolist(),
        'action_history':np.zeros((2,10)).tolist()},goal=np.ones(192).tolist(),projection=np.eye(192).tolist(),
        root_raw_action_history=m.gn.moving.fixed_raw().astype(float).tolist(),horizon=2,packet_steps=5)
    export=dict(source_slot=0,task_id=p['task_id'],blind_sha256=p['blind_sha256'],task_contract=dict(
        pixel_sha256=['d'*64]*4,normalization=m.gn.moving.normalization(),model_bindings=m.arrays.MODELS,
        raw_action_sha256=m.digest(m.gn.moving.fixed_raw().tobytes()),with_target=False))
    return p,row,export,task


def all_records():
    records=[]
    base=m.record(*record_fixture(),{5:np.ones(192,np.float32),10:np.full(192,2,np.float32)},
        {'cost_axis':'fixture'},'f'*64,{'source':'fixture'})
    for s,a in m.EXPECTED:
        r=deepcopy(base);r.update(root_slot=s,arm=a)
        r['identity']['source_slot']=s;r['identity']['root_sha256']=str(s)*64
        r['identity_sha256']=m.fingerprint(r['identity'])
        r['domain']['search_support']=a  # intentionally different mechanisms
        records.append(r)
    return records


def test_all10_failures_retained_fixed_predeclared_order():
    r=m.assemble(list(reversed(all_records())),[])
    assert [(x['root_slot'],x['arm']) for x in r['rows']]==m.EXPECTED
    assert r['fixed_denominator']==r['native_failures']==10 and r['native_successes']==0
    assert len(r['identity_partitions'])==2 and r['all_expected_same_identity'] is True
    assert all(x['all_five_present'] and x['same_admissible_packet_domain'] and not x['same_search_support'] for x in r['identity_partitions'])
    assert r['rankings'] is None and r['P1_admission'] is r['cross_paper_ranking'] is False


@pytest.mark.parametrize('change',['goal_latent','normalization','root_snapshot','task_sha256','model_bindings','raw_history','with_target'])
def test_identity_difference_never_pooled(change):
    rs=all_records();rs[0]['identity'][change]={'fixture':'different_valid_saved_identity'}
    rs[0]['identity_sha256']=m.fingerprint(rs[0]['identity'])
    r=m.assemble(rs,[])
    assert len(r['rows'])==10 and len(r['identity_partitions'])==3 and r['all_expected_same_identity'] is False
    own=next(x for x in r['identity_partitions'] if x['identity_sha256']==rs[0]['identity_sha256'])
    assert own['arms']==[m.beam.ARM] and own['descriptive_joint_allowed'] is False


@pytest.mark.parametrize('fault',['missing','duplicate','wrongarm','boolslot','fingerprint','missingcheckpoint','steps','success'])
def test_invalid_census_failclosed(fault):
    rs=all_records()
    if fault=='missing':rs.pop()
    elif fault=='duplicate':rs[0]=deepcopy(rs[1])
    elif fault=='wrongarm':rs[0]['arm']='old_beam16x4'
    elif fault=='boolslot':rs[0]['root_slot']=False
    elif fault=='fingerprint':rs[0]['identity_sha256']='0'*64
    elif fault=='missingcheckpoint':rs[0]['checkpoints'].pop()
    elif fault=='steps':rs[0]['own_returned']=9
    else:rs[0]['native_success']=True
    with pytest.raises(ValueError):m.assemble(rs,[])


def test_native_full7_velocity_changes_distance_not_success_rule():
    g=[0.]*7;s=[0.,0.,0.,0.,0.,3.,4.]
    r=m.native(g,s);assert r['native_distance']==5. and r['native_reward']==-5. and r['native_success'] is True
    s[0]=30.;r=m.native(g,s);assert r['native_success'] is False and r['object_xy_error']==0.
    assert r['success_includes_agent_and_object'] is True


def test_original_angle_no_silent_wrapping():
    r=m.native([0.]*7,[0.,0.,0.,0.,7.,0.,0.])
    assert r['native_angle_term']<0 and r['native_success'] is True and r['wrapped_allxy_success'] is False


@pytest.mark.parametrize('n',[5,6,8])
def test_native_wrong_dimension_rejected(n):
    with pytest.raises(ValueError):m.native([0.]*n,[0.]*n)


def test_record_original_root_not_fabricated_encode():
    args=record_fixture();r=m.record(*args,{5:np.ones(192,np.float32),10:np.full(192,2,np.float32)}, {},'f'*64,{})
    assert r['root_Q_original_task']==1. and [x['Q'] for x in r['checkpoints']]==[1.,0.,1.]
    assert r['checkpoints'][0]['latent_source']=='original_task_root_not_new_RGB_encode'
    assert r['checkpoints'][1]['latent_source']=='actual_saved_RGB_encoding'
    assert r['native_success'] is False


@pytest.mark.parametrize('fault',['raw','ball','normalizeraw','nan'])
def test_selected_action_witness_failclosed(fault):
    p=plan()
    if fault=='raw':p['commands'][0][0]=.1
    elif fault=='ball':p['logits'][0][0]=2.
    elif fault=='normalizeraw':p['commands']=(np.asarray(p['commands'])+.01).tolist()
    else:p['logits'][0][0]=float('nan')
    with pytest.raises(ValueError):m.domain(p)


def test_beam_real_fixed_bank_is_not_old_constant_packet_support():
    p=plan(m.beam.ARM);b=m.beam.producer.bank(0);p.update(selected_path=[0,1],bank_sha256=m.beam.producer.core.digest(b),
        commands=b['commands'][0]+b['commands'][1])
    d=m.domain(p);assert d['equal_search_coverage'] is False and 'finite subset' in d['search_support']
    assert d['admissible']['packets']==2 and d['selected_logit_norms'][0]==0.


def test_different_admissible_support_never_claims_same():
    rs=all_records();rs[0]['domain']['admissible']['logit_cap']=.5
    r=m.assemble(rs,[]);assert r['identity_partitions'][0]['same_admissible_packet_domain'] is False
    assert r['equal_total_compute'] is r['equal_FLOPs'] is False


def test_missing_real_checkpoint_cannot_be_padded():
    p,r,e,t=record_fixture();r['checkpoints'].pop(1)
    with pytest.raises(ValueError):m.record(p,r,e,t,{5:np.ones(192),10:np.ones(192)}, {},'f'*64,{})


def test_bad_saved_native_metric_rejected():
    p,r,e,t=record_fixture();r['final_metrics']['native_distance']+=1
    with pytest.raises(ValueError):m.record(p,r,e,t,{5:np.ones(192),10:np.ones(192)}, {},'f'*64,{})


def test_preserve_original_legacy_epsilon_not_silent_strict_ball_claim():
    p=plan('full10_conditional_D2P');p['logits'][0][0]=float(.75*np.sqrt(5))+1e-13
    p['commands']=np.tanh(np.asarray(p['logits'],np.float32)).reshape(10,2).astype(float).tolist()
    d=m.domain(p);assert d['validation_epsilon']==1e-12 and d['strictly_inside_closed_balls'] is False
    assert d['logit_norm_excess'][0]>0
    p['arm']='CEM'
    with pytest.raises(ValueError):m.domain(p)  # no borrowing GN/D2P receipt epsilon
