"""Saved arrays and counterexamples only; no model/env imports or execution."""
from copy import deepcopy
import math
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p1_goal_consistent_conditional_outcome_audit_20261027_v1 as m


def test_pose_and_official_do_not_conflate_agent_or_velocity():
    goal=[0.]*7;state=[100.,0.,3.,4.,0.,10.,0.]
    r=m.labels(state,goal)
    assert r['pose']['pose_cost']==5
    assert r['official']['official_state_distance']==pytest.approx(math.sqrt(10125))
    assert not r['official']['official_success'] and r['official']['object_only_success']


def test_official_angle_uses_original_not_improved_wrap():
    r=m.labels([0.,0.,0.,0.,2*math.pi+.1,0.,0.],[0.]*7)
    assert r['official']['angle_error']==pytest.approx(-.1)
    assert r['pose']['angle_distance']==pytest.approx(.1)


@pytest.fixture(scope='module')
def bundle():return m.load_bundle(m.PROJECT)[0]


def test_actual_saved_artifacts_all_pass(bundle):
    r=m.audit(bundle)
    assert r['pair_counts']=={'full_H2_pair':4}
    assert r['budget']['selected']==dict(attempts=80,returns=80)
    assert r['budget']['warmup']==dict(attempts=120,returns=120)
    assert r['upstream_budget']['predict']==528 and r['upstream_budget']['jacobians']==88
    assert r['encoding_budget']['encoder_attempts']==r['encoding_budget']['projector_attempts']==8
    assert r['new_neural_calls']==r['new_physical_steps']==0
    assert r['comparison']['pose_improved_roots']==4
    assert r['comparison']['official_distance_improved_roots']==1


@pytest.mark.parametrize('damage',['state','pose','official','raw_action','history','guard','fresh','count','horizon'])
def test_branch_tampering_rejected(bundle,damage):
    key=next(iter(bundle['rows']));r=deepcopy(bundle['rows'][key]);p=bundle['plans'][key]
    if damage=='state':r['final_state'][2]+=1
    elif damage=='pose':r['pose']['pose_cost']+=1
    elif damage=='official':r['official']['official_state_distance']+=1
    elif damage=='raw_action':r['commands'][0][0]+=.01
    elif damage=='history':r['context_identity']['normalized_between_history'][0][0]+=.01
    elif damage=='guard':r['guard']['observations'][0]['canonical_agent_pixels']=0
    elif damage=='fresh':r['endpoint_fresh']=False
    elif damage=='count':r['physical_env_step_returns']=9
    else:r['planned_horizon_tokens']=3
    with pytest.raises(ValueError):m.inspect_branch(p,r,bundle['events'][key],bundle['pixels'][key],bundle['packets'][key[0]],bundle['protocol']['references'][key[0]])


@pytest.mark.parametrize('damage',['pose','latent','Qcost','rgb','metric'])
def test_endpoint_encoding_tampering_rejected(bundle,damage):
    key=next(iter(bundle['rows']));e=deepcopy(bundle['encodings'][key]);p=bundle['plans'][key]
    if damage=='latent':e['terminal_vector'][0]+=1
    elif damage=='Qcost':e['actual_Q_costs']['old_Q']+=1
    elif damage=='rgb':e['terminal_rgb_sha256']='bad'
    elif damage=='metric':e['active_metric_id']='bad'
    else:e['q_sha256']['goal_patch_Q']='bad'
    with pytest.raises(ValueError):m.inspect_encoding(p,bundle['rows'][key],e)


def test_q_scores_same_vector_and_different_projectors():
    z=np.array([2.,3.]+[0.]*190);q=np.diag([1.,0.]+[0.]*190)
    assert m.qcost(z,[0.]*192,q)==4/192


@pytest.mark.parametrize('damage',['command','phase','index','return','foreign'])
def test_budget_events_not_trusted_as_summary_counts(bundle,damage):
    events=deepcopy(bundle['budget_events'])
    k=next(i for i,e in enumerate(events) if e['kind']=='env_step' and e['phase']=='selected' and not e['completed'])
    if damage=='command':events[k]['command'][0]+=.1
    elif damage=='phase':events[k]['phase']='warmup'
    elif damage=='index':events[k]['index']+=1
    elif damage=='return':events.pop(k+1)
    else:events[k]['root_sha256']='foreign'
    with pytest.raises(ValueError):m.physical_budget(events,bundle['plans'])


def test_partial_pair_not_aggregated_as_complete(bundle):
    rows=deepcopy(bundle['rows']);key=next(iter(rows));rows[key]['full_horizon_completed']=False
    pairs=m.pair_rows(bundle['protocol']['roots'],rows,bundle['encodings'])
    assert pairs[0]['classification']!='full_H2_pair'
    assert pairs[0]['new_minus_old_pose_cost'] is None


@pytest.mark.parametrize('damage',['missing','index','foreign','extra'])
def test_encoding_journal_counterexamples(bundle,damage):
    events=deepcopy(bundle['encoding_events'])
    if damage=='missing':events.pop(4)
    elif damage=='index':events[3]['index']+=1
    elif damage=='foreign':events[2]['root_sha256']='foreign'
    else:events.append(dict(kind='predict',completed=True))
    with pytest.raises(ValueError):m.encoding_budget(events,bundle['plans'],bundle['encodings'])


@pytest.mark.parametrize('damage',['missing_module','missing_jacobian','replay','summary'])
def test_upstream_budget_not_double_counted_or_summary_only(bundle,damage):
    b=dict(bundle,compile_events=deepcopy(bundle['compile_events']),compilation=deepcopy(bundle['compilation']))
    events=b['compile_events'][bundle['protocol']['roots'][0]]
    if damage=='missing_module':events.pop(next(i for i,e in enumerate(events) if e['kind']=='predict'))
    elif damage=='missing_jacobian':events.pop(next(i for i,e in enumerate(events) if e['kind']=='jacobian'))
    elif damage=='replay':
        event=next(e for e in events if e['kind']=='replay' and e['completed'])
        event['commands']=event['commands'][:5]
    else:b['compilation']['resources']['completed_predict_samples']+=1
    with pytest.raises(ValueError):m.compiler_budget(b)


def test_import_never_loads_consumer_model_or_simulator():
    subprocess.run([sys.executable,'-c','import sys; from experiments.spectral_reachability import '+m.NAME+
        '; assert "torch" not in sys.modules; assert "gymnasium" not in sys.modules; assert "pymunk" not in sys.modules; '
        'assert "experiments.spectral_reachability.p1_goal_consistent_conditional_physical_20261027_v1" not in sys.modules'],check=True)
