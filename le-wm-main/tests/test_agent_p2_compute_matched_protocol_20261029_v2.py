"""RAM-only wall-cap protocol/accounting, no torch/physics/model execution."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_compute_matched_protocol_20261029_v2 as m


class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now


def ledger(arm='joint',clock=None):return m.Ledger(0,arm,clock or Clock())


def predict(l):
    p=l.intent('predict')
    for kind in ('action_encoder','predictor','pred_proj'):
        k=l.intent(kind);l.returned(k)
    l.returned(p)


def unit(l,name):
    assert l.start_unit(name)
    costs=m.unit_cost(name)
    ad=None
    if costs['jacobian']:
        h,d=m.AD_SPEC[name];ad=l.intent('jacobian',horizon=h,dimension=d)
    h=m.AD_SPEC[name][0] if ad is not None else 0
    for _ in range(h):predict(l)
    if ad is not None:l.returned(ad)
    for _ in range(costs['predict']-h):predict(l)
    for _ in range(costs['root_SVD']):i=l.intent('root_SVD');l.returned(i)
    for _ in range(costs['root_ridge']):i=l.intent('root_ridge');l.returned(i)
    l.finish_unit()


def test_common_caps_not_equal_elapsed_or_equal_flops():
    p=m.protocol({})
    assert p['comparison']=='wall_capped_not_equal_FLOPs'
    assert p['wall_cap_seconds_per_root_arm']==450
    assert p['equal_FLOPs'] is p['equal_total_compute'] is False
    a=ledger();b=ledger('composition');a.clock.now=1;b.clock.now=2
    assert m.compare_reports([a.report(),b.report()])['elapsed_equality_required'] is False


def test_all_fixed_units_and_roots_no_outcome_schedule():
    joint=m.schedule('joint');comp=m.schedule('composition')
    assert joint[:2]==comp[:2]==['nominal','bank']
    assert joint[2:44]==comp[2:44]==['seed:'+str(i) for i in range(42)]
    assert len(joint)==len(comp)==129  # 2 setup +42 seed+84 optimize+replay
    assert joint[44]=='joint:0:0' and joint[86]=='joint:1:0'
    assert comp[44:48]==['H2:0','H1:0','H2:1','H1:1']


def test_natural_completion_no_padding_and_different_actual_costs():
    reports=[]
    for arm in ('joint','composition'):
        l=ledger(arm)
        for label in m.schedule(arm):unit(l,label)
        assert not l.start_unit('anything')
        reports.append(l.report())
    assert reports[0]['attempted']['predict']==1902
    assert reports[1]['attempted']['predict']==1062  # own common full42 scan, not old936
    assert all(r['attempted']['jacobian']==85 for r in reports)
    assert all(r['stop']=='natural_completion' for r in reports)


def test_wall_deadline_does_not_dispatch_or_pad():
    c=Clock();l=ledger(clock=c);c.now=450
    assert not l.start_unit('nominal')
    assert l.report()['attempted']['predict']==0
    assert l.report()['stop']=='wall_cap'


def test_inflight_overrun_is_reported_not_hidden_or_equalized():
    c=Clock();l=ledger(clock=c);assert l.start_unit('nominal')
    i=l.intent('predict');c.now=451;l.returned(i,success=False)
    assert l.report()['wall_overrun'] is True
    assert l.report()['attempted']['predict']==1 and l.report()['completed']['predict']==0
    with pytest.raises(ValueError):l.intent('predict')


def test_failure_no_refund_preserves_all_intents_and_stops_arm():
    l=ledger();assert l.start_unit('nominal');i=l.intent('predict');l.returned(i,success=False)
    r=l.report();assert r['attempted']['predict']==1 and r['completed']['predict']==0
    assert r['stop']=='operation_failure' and r['errors']
    assert not l.start_unit('nominal')


def test_jacobian_primal_counts_and_no_incomplete_AD_commit():
    l=ledger();unit(l,'nominal');assert l.start_unit('bank')
    i=l.intent('jacobian',horizon=3,dimension=30)
    predict(l)
    with pytest.raises(ValueError):l.returned(i)
    assert l.report()['attempted']['mathematical_VJP']==192
    assert l.report()['completed']['jacobian']==0


def test_extra_or_wrong_order_unit_and_shape_fail_closed():
    l=ledger()
    with pytest.raises(ValueError):l.start_unit('bank')
    unit(l,'nominal');assert l.start_unit('bank')
    with pytest.raises(ValueError):l.intent('jacobian',horizon=2,dimension=20)


def test_counter_cap_checked_before_callback_and_replay_reserve_kept():
    l=ledger();l.attempted['predict']=m.caps()['predict']-6
    assert not l.start_unit('nominal')
    assert l.report()['stop']=='resource_cap'


def witness(cost=1.):
    c=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=[[0.]*192]*2+[[cost**.5]*192],action_history=m.old.kernel.normalized_history(np.zeros((10,2),np.float32)).tolist())
    return dict(logits=[0.]*30,contexts=[deepcopy(c) for _ in range(3)])


def test_anytime_incumbent_only_complete_model_witness_stable_ties():
    l=ledger();unit(l,'nominal')
    z=[0.]*192
    assert l.commit(witness(),z,'baseline')
    assert not l.commit(witness(),z,'tie')
    assert l.commit(witness(.25),z,'improved')
    assert l.report()['incumbent']['tag']=='improved'
    bad=witness();bad['contexts']=bad['contexts'][:2]
    with pytest.raises(ValueError):l.commit(bad,z,'partial')


def test_clock_must_be_finite_monotone():
    c=Clock();l=ledger(clock=c);c.now=-1
    with pytest.raises(ValueError):l.report()


def test_failed_child_cannot_be_reported_as_successful_predict():
    l=ledger();assert l.start_unit('nominal');p=l.intent('predict')
    i=l.intent('action_encoder');l.returned(i,success=False)
    with pytest.raises(ValueError):l.returned(p)


def test_after_deadline_improvement_not_used_for_capped_incumbent():
    c=Clock();l=ledger(clock=c);unit(l,'nominal');l.commit(witness(),[0.]*192,'before')
    c.now=451
    assert not l.commit(witness(.25),[0.]*192,'late')
    assert l.report()['incumbent']['tag']=='before'
    assert l.report()['commits'][-1]['late'] is True


def test_resource_stop_can_spend_replay_reserve_without_fake_unit_completion():
    l=ledger();unit(l,'nominal')
    l.attempted['predict']=m.caps()['predict']-6
    assert not l.start_unit('bank')
    unit(l,'replay')
    assert l.report()['stop']=='resource_cap'
    assert l.report()['export_complete'] is True
    assert l.report()['units_completed']==2


@pytest.mark.parametrize('key',['action_history','latent_history'])
def test_partial_context_or_wrong_action_history_rejected(key):
    l=ledger();unit(l,'nominal');w=witness();w['contexts'][1][key]=[]
    with pytest.raises(ValueError):l.commit(w,[0.]*192,'bad')


def test_prepare_reuses_real_saved_binding_not_v1_equality_status(tmp_path,monkeypatch):
    import json
    (tmp_path/'report.json').write_text(json.dumps(dict(prepared='/saved/p',prepare_sha256='psha')))
    monkeypatch.setattr(m.prior,'build',lambda *a:dict(authority={'receipt_sha256':m.prior.RECEIPT_SHA},bindings_sha256={'/saved/model':'modelSHA'}))
    monkeypatch.setattr(m.old,'verify',lambda *a:dict(slots=[dict(root_slot=i,condition='off') for i in (0,1)],expected_model_hash='modelSHA'))
    monkeypatch.setattr(m.old,'sha',lambda *a:'ownSHA')
    p=m.build(tmp_path,'/saved/receipt')
    assert p['comparison']=='wall_capped_not_equal_FLOPs'
    assert p['expected_model_hash']=='modelSHA' and len(p['slots'])==2
    assert p['bindings_sha256']['/saved/model']=='modelSHA'


def test_pair_report_rejects_nonfinite_elapsed_not_requires_equal():
    a=ledger().report();b=ledger('composition').report();b['elapsed_seconds']=float('nan')
    with pytest.raises(ValueError):m.compare_reports([a,b])


def test_sealed_elapsed_not_inflated_by_later_other_arm_or_consume():
    c=Clock();l=ledger(clock=c);c.now=450;l.start_unit('nominal');l.seal()
    c.now=9999
    assert l.report()['elapsed_seconds']==450
    assert not l.start_unit('nominal')
    with pytest.raises(ValueError):l.intent('predict')


def test_actual_memory_caps_reported_not_used_as_equality_gate():
    l=ledger();assert l.report()['peak_allocated_bytes'] is None
    l.sample_memory(1024,2048)
    assert l.report()['peak_allocated_bytes']==1024
    l.sample_memory(4*1024**3+1,6*1024**3)
    assert l.report()['stop']=='memory_cap'
    assert not l.start_unit('nominal')


def test_natural_finish_requires_explicit_seal_and_cannot_arbitrary_stop():
    l=ledger()
    with pytest.raises(ValueError):l.seal()


def test_no_outcome_access_in_prepare_and_zero_resource_caps_forbidden_ops():
    p=m.protocol({})
    assert p['physical_outcome_access'] is False
    assert all(p['resource_caps_per_root_arm'][k]==0 for k in ('encoder','projector','physics','training'))
    l=ledger();assert l.start_unit('nominal')
    with pytest.raises(ValueError):l.intent('encoder')
