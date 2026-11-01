"""Pure resource/mock tests plus saved receipt audit. No neural/physics."""
from copy import deepcopy
import json
import pytest
from experiments.spectral_reachability import p2_full_resource_matching_20261027_v1 as m


def bindings():
    return {k:'a'*64 for k in ('root_sha256','task_sha256','model_sha256','protocol_sha256')}


def test_forward_equality_never_upgrades_unknown_or_AD_mismatch():
    joint=m.forward(168); cond=m.add(m.forward(168),m.vector(jacobian_192x10_calls=40,math_vjp_192x10_directions=7680))
    assert m.validate_claim(joint,cond,'predict_frontier_equal')['predict_frontier_equal']
    with pytest.raises(ValueError,match='168/168'):m.validate_claim(joint,cond,'full_resource_equal')
    assert not m.compare(joint,joint)['full_resource_equal']
    assert m.compare(joint,joint)['status']=='unknown_full_cost'


def test_derivative_shapes_cannot_be_collapsed_into_Jacobian_call_count():
    a=m.vector(jacobian_192x20_calls=1,math_vjp_192x20_directions=192,jacobian_scalar_entries=3840)
    b=m.vector(jacobian_192x10_calls=1,math_vjp_192x10_directions=192,jacobian_scalar_entries=1920)
    r=m.compare(a,b)
    assert not r['measured_vector_equal'] and 'jacobian_scalar_entries' in r['differences']
    assert a['math_vjp_192x20_directions']==b['math_vjp_192x10_directions']


def test_fully_known_equal_declared_vector_is_not_hardware_attestation():
    a=m.vector(**{k:0 for k in m.UNKNOWN})
    r=m.validate_claim(a,deepcopy(a),'full_resource_equal')
    assert r['full_resource_equal'] and not r['hardware_cost_equivalence_certified']
    b=deepcopy(a);b['flops']=None
    assert not m.compare(a,b)['full_resource_equal']


@pytest.mark.parametrize('axis',['svd_192x10_calls','bounded_solve_10_calls','visual_encoder_calls',
    'projector_calls','selected_steps','warmup_steps','root_collection_resets','action_encoder_context_tokens'])
def test_each_real_resource_axis_can_break_full_equality(axis):
    a=m.vector();b=m.vector(**{axis:1})
    assert axis in m.compare(a,b)['differences']
    with pytest.raises(ValueError):m.validate_claim(a,b,'full_resource_equal')


@pytest.mark.parametrize('value',[-1,True,1.5,'192'])
def test_invalid_counts_and_unknown_axes_rejected(value):
    with pytest.raises(ValueError):m.vector(predict_calls=value)
    with pytest.raises(ValueError):m.vector(official_success=1)


def test_shared_once_and_unknown_propagation_peak_max():
    a=m.vector(predict_calls=4,peak_gpu_bytes=30);b=m.vector(predict_calls=164,peak_gpu_bytes=20)
    out=m.sum_disjoint([dict(work_id='shared',vector=a),dict(work_id='exclusive',vector=b)])
    assert out['predict_calls']==168 and out['peak_gpu_bytes']==30 and out['flops'] is None
    with pytest.raises(ValueError):m.sum_disjoint([dict(work_id='shared',vector=a)]*2)
    assert m.add(a,m.vector())['peak_gpu_bytes'] is None


def test_feasible_common_cap_allocations_are_not_equal_consumption():
    p=m.next_protocol();a=p['allocations']
    assert [a[x]['candidates'] for x in m.ARMS]==[80,40]
    assert not p['strict_equal_vector_feasible_for_positive_candidates']
    assert a[m.ARMS[0]]['unspent']['jacobian_192x10_calls']==40
    assert a[m.ARMS[1]]['unspent']['jacobian_192x10_calls']==0
    assert all(x['no_padding'] and not x['strict_full_resource_ready'] for x in a.values())
    r=m.candidate_recipe(m.ARMS[1]);cap=deepcopy(p['common_per_root_cap'])
    cap['jacobian_192x10_calls']=7
    allocation=m.allocate(cap,r['fixed'],r['per_useful_candidate_cap'],40)
    assert allocation['candidates']==7  # derivative axis, not forward axis, binds
    cap['selected_steps']=9
    with pytest.raises(ValueError,match='mandatory'):m.allocate(cap,r['fixed'],r['per_useful_candidate_cap'],40)


def test_zero_capacity_no_fake_padding_or_free_unit():
    r=m.candidate_recipe(m.ARMS[0]); cap=deepcopy(r['fixed'])
    a=m.allocate(cap,r['fixed'],r['per_useful_candidate_cap'],80)
    assert a['candidates']==0 and not a['feasible']
    with pytest.raises(ValueError):m.allocate(cap,r['fixed'],m.vector(),80)
    cap['predict_calls']=None
    with pytest.raises(ValueError):m.allocate(cap,r['fixed'],r['per_useful_candidate_cap'],80)


def test_reservation_predispatch_partial_failure_nonrefund_and_bindings():
    ledger=m.BudgetLedger(m.forward(2),bindings()); unit=m.forward(1)
    ledger.reserve('one',unit,'candidate_evaluation')
    ledger.finish('one',m.vector(action_encoder_calls=1,action_encoder_samples=1,action_encoder_context_tokens=3))
    assert ledger.reserved['predict_calls']==1 and ledger.completed['predict_calls']==0
    ledger.reserve('two',unit,'candidate_evaluation')
    with pytest.raises(ValueError,match='predispatch'):ledger.reserve('three',unit,'candidate_evaluation')
    assert ledger.reserved['predict_calls']==2 and len(ledger.events)==2
    with pytest.raises(ValueError):ledger.finish('one',unit)
    assert ledger.events[0]['bindings']==bindings()
    receipt=ledger.receipt()
    assert receipt['pending']==['two'] and not receipt['full_resource_equal']
    receipt['reserved']['predict_calls']=0
    assert ledger.reserved['predict_calls']==2


def test_padding_duplicate_ids_unknown_bindings_and_overcompletion_rejected():
    ledger=m.BudgetLedger(m.forward(2),bindings());unit=m.forward(1)
    with pytest.raises(ValueError):ledger.reserve('padding',unit,'padding')
    ledger.reserve('real',unit,'candidate_generation')
    with pytest.raises(ValueError):ledger.reserve('real',unit,'candidate_generation')
    with pytest.raises(ValueError):ledger.finish('real',m.forward(2))
    assert ledger.events[0]['completed'] is None and ledger.completed['predict_calls']==0
    with pytest.raises(ValueError):m.BudgetLedger(m.forward(2),dict(bindings(),root_sha256='missing'))


def operation_fixture(zero_radius=False):
    ops=[]
    def pair(e,terminal=None):
        ops.extend([dict(e,completed=False),dict(e,completed=True,**(terminal or {}))])
    pair(dict(kind='jacobian',shape=[192,20],vectorize=True))
    pair(dict(kind='bounded_ridge_svd',svd_calls_requested=1))
    pair(dict(kind='mode_svd',svd_calls_requested=1))
    pair(dict(kind='suffix_jacobian',index=0,shape=[192,10],vectorize=True))
    n=int(not zero_radius)
    pair(dict(kind='suffix_solve',index=0,svd_calls_requested=n),dict(svd_calls=n,bounded_solve_calls=n))
    local=dict(jacobian=[[0.]*10 for _ in range(192)],svd_calls=n,bounded_solve_calls=n)
    results=[dict(arm=m.ARMS[1],attempts=[dict(index=0,suffix_compilation=local)])]
    return ops,results


@pytest.mark.parametrize('zero',[False,True])
def test_actual_operation_shapes_and_zero_radius_solve_not_assumed(zero):
    shared,cond=m.count_operations(*operation_fixture(zero))
    assert shared['jacobian_192x20_calls']==1 and shared['svd_192x20_calls']==2
    assert cond['jacobian_192x10_calls']==1 and cond['math_vjp_192x10_directions']==192
    assert cond['svd_192x10_calls']==cond['bounded_solve_10_calls']==int(not zero)


@pytest.mark.parametrize('damage',['shape','completion','pair','kind','missing','solve'])
def test_operation_tamper_cannot_disappear(damage):
    ops,results=operation_fixture()
    if damage=='shape':ops[6]['shape']=[192,20]
    elif damage=='completion':ops[7]['completed']=False
    elif damage=='pair':ops[7]['index']=1
    elif damage=='kind':ops[-2]['kind']=ops[-1]['kind']='other_work'
    elif damage=='missing':ops.pop()
    else:ops[-1]['svd_calls']=0
    with pytest.raises(ValueError):m.count_operations(ops,results)


def test_actual_source_bound_resource_audit_and_no_total_compute_claim():
    r=m.audit_current();u=r['paired_unique'];a=r['standalone_attributed']
    assert u['predict_calls']==u['action_encoder_calls']==1328
    assert u['jacobian_192x20_calls']==4 and u['jacobian_192x10_calls']==160
    assert u['math_vjp_192x20_directions']==768 and u['math_vjp_192x10_directions']==30720
    assert u['svd_192x20_calls']==8 and u['svd_192x10_calls']==160
    assert u['bounded_solve_20_calls']==4 and u['bounded_solve_10_calls']==160
    assert u['visual_encoder_calls']==u['projector_calls']==8
    assert u['warmup_steps']==120 and u['selected_steps']==80
    assert a[m.ARMS[0]]['predict_calls']==a[m.ARMS[1]]['predict_calls']==672
    assert all(row['standalone_planner'][arm]['predict_calls']==168 for row in r['roots'] for arm in m.ARMS)
    assert r['comparison']['predict_frontier_equal'] and not r['comparison']['full_resource_equal']
    assert 'flops' in r['comparison']['unknown_axes'] and not r['P2_admission']
    print('\nACTUAL_RESOURCES='+json.dumps(dict(comparison=r['comparison'],paired_unique=u),sort_keys=True))
