"""Pure saved-state/latent arithmetic fixtures; no simulator or model execution."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import p4_multiround_mechanism_20261027_v1 as m


@pytest.fixture(scope='module')
def g():
    return m.geometry.contract(m.saved.Inputs())


def states(end=20):
    return [dict(primitive_step=i,state=[300.,300.,100.+i,150.,.01*i,2.*i,-float(i)]) for i in range(end+1)]


@pytest.mark.parametrize('seed',range(5))
def test_feedback_residual_identity_and_direction(seed):
    rng=np.random.default_rng(seed)
    observed,predicted,goal=rng.normal(size=(3,192));q=rng.normal(size=(192,192))*.1
    value=m.drift(observed,predicted,goal,q)
    assert value['projected_mse']>=0
    assert value['observed_minus_predicted_Q_cost']==pytest.approx(value['projected_mse']+value['goal_cross_term'])
    assert abs(value['identity_error'])<1e-10


def test_feedback_error_magnitude_does_not_determine_goal_direction():
    z=np.zeros(192);goal=np.ones(192);q=np.eye(192)
    toward=m.drift(np.full(192,.2),z,goal,q)
    away=m.drift(np.full(192,-.2),z,goal,q)
    assert toward['projected_mse']==away['projected_mse']
    assert toward['observed_minus_predicted_Q_cost']<0<away['observed_minus_predicted_Q_cost']
    assert toward['goal_cross_term']<0<away['goal_cross_term']


def test_zero_goal_direction_cosine_is_not_fabricated():
    z=np.zeros(192)
    value=m.drift(z,z,z,np.eye(192))
    assert value['raw_l2']==0 and value['drift_vs_predicted_goal_direction']['cosine'] is None


def test_official_squared_distance_decomposition_keeps_velocity():
    goal=np.zeros(7);state=np.array([3,4,6,8,2,5,12])
    p=m.official_components(state,goal)
    assert p==dict(agent_xy_squared=25.,object_xy_squared=100.,unwrapped_angle_squared=4.,pusher_velocity_squared=169.)
    assert sum(p.values())==np.linalg.norm(state)**2


def test_full_segment_pose_progress_telescopes(g):
    value=m.physical_segment(states(),5,10,np.zeros(7),g)
    assert value['status']=='complete_segment' and len(value['intervals'])==5
    assert value['net']['original_pose']['cost_change']==pytest.approx(sum(v['pose_change'] for v in value['intervals']))
    assert value['net']['origin_vs_fixed_goal']['exact_goal_distance_reduction']==pytest.approx(
        sum(v['origin_goal_distance_reduction'] for v in value['intervals']))
    assert value['net']['origin_vs_fixed_goal']['exact_goal_distance_reduction']<0
    assert value['force_contact'] is None and value['causal_attribution'] is None


@pytest.mark.parametrize('last', [5,7,9])
def test_partial_segment_never_padded(g,last):
    value=m.physical_segment(states(last),5,10,np.zeros(7),g)
    assert value['status']=='partial_observed_segment' and value['last_observed']==last
    assert len(value['intervals'])==last-5


def test_unobserved_segment_and_duplicate_fail_closed(g):
    value=m.physical_segment(states(7),10,15,np.zeros(7),g)
    assert value['status']=='not_observed' and value['net'] is None
    s=states();s.append(deepcopy(s[0]))
    with pytest.raises(ValueError,match='duplicate'):m.physical_segment(s,5,10,np.zeros(7),g)


def fixture_arm(arm):
    p=dict(physical_goal=np.zeros(7).tolist(),goal=np.zeros(192).tolist(),q=np.eye(192).tolist())
    row=dict(arm=arm,classification='complete20',steps=states())
    events=[]
    def context(value):
        return dict(latent_history=np.full((3,192),value).tolist())
    for start in (5,10,15):
        observed=context(.2);predicted=context(.3);chosen=observed if arm=='actual' else predicted
        events.append(dict(kind='planning_context',arm=arm,boundary=start,observed_context=observed,
            predicted_context=predicted,chosen_context=chosen))
        winner=dict(candidate_id='suffix_0',predicted_terminal=np.full(192,.1).tolist(),nonlinear_terminal_cost=.01)
        events.append(dict(kind='selection',arm=arm,boundary=start,selection_sha256='fixture_only',
            compiler_plan=dict(context=chosen,nominal_terminal=np.full(192,.3).tolist(),
                candidates=[winner],selected_candidate_id='suffix_0')))
    return p,row,events


@pytest.mark.parametrize('arm',['actual','imagined'])
def test_three_rounds_terminal_latent_not_imputed(g,arm):
    p,row,events=fixture_arm(arm)
    value=m.analyze_arm(p,row,events,g)
    assert len(value['rounds'])==3 and value['complete20']
    assert all(r['physical']['status']=='complete_segment' for r in value['rounds'])
    assert [r['next_latent_status'] for r in value['rounds']]==['saved_next_boundary','saved_next_boundary','not_saved_at20']
    assert value['rounds'][-1]['next_boundary_prediction_discrepancy'] is None
    assert 'observed_latent_Q_cost_change' not in value['rounds'][-1]
    assert all(r['model']['predicted_reduction_from_chosen']>0 for r in value['rounds'])
    assert value['observed_pose_change_sum']>0
    assert value['rounds'][0]['boundary_feedback']['selected_context']==('observed' if arm=='actual' else 'predicted')


def test_model_score_tampering_rejected(g):
    p,row,events=fixture_arm('actual')
    events[1]['compiler_plan']['candidates'][0]['nonlinear_terminal_cost']=10.
    with pytest.raises(ValueError,match='model score'):m.analyze_arm(p,row,events,g)


def test_guard_partial_round_slots_preserved(g):
    p,row,events=fixture_arm('actual')
    row.update(classification='guard_aborted',steps=states(12))
    events=[e for e in events if e['boundary']<=10]
    value=m.analyze_arm(p,row,events,g)
    assert not value['complete20'] and len(value['rounds'])==3
    assert [r['physical']['status'] for r in value['rounds']]==[
        'complete_segment','partial_observed_segment','not_observed']
    assert value['rounds'][1]['next_boundary_prediction_discrepancy'] is None


def test_blocked_has_no_fabricated_zero_progress(g):
    p,row,events=fixture_arm('actual')
    row.update(classification='blocked',steps=[])
    value=m.analyze_arm(p,row,[],g)
    assert value['observed_pose_change_sum'] is None
    assert all(r['physical']['status']=='not_observed' for r in value['rounds'])


def test_actual_saved_audit_no_run_prepare(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('saved arithmetic must not invoke runtime/physics/neural entry point')
    monkeypatch.setattr(m.outcomes.runtime,'run',forbidden)
    monkeypatch.setattr(m.outcomes.runtime,'prepare',forbidden)
    result=m.audit(m.DEFAULT_OUTCOMES)
    assert result['status']=='saved_round_decomposition_recorded'
    assert result['inputs_unchanged'] and result['new_neural_calls']==result['new_physics_steps']==0
    assert not result['P4_admission'] and not result['causal_attribution']
    assert len(result['rows'])==2 and sum(len(r['rounds']) for r in result['rows'])==6


def test_new_exclusive_output_required(tmp_path):
    with pytest.raises(ValueError):m.run(tmp_path)
