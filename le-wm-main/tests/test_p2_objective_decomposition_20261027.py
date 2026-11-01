from copy import deepcopy
import numpy as np
import pytest

from experiments.spectral_reachability.p2_objective_decomposition_20261027 import decompose, metric


def data():
    c, p, o = [], [], []
    for root in range(4):
        compiled = dict(goal_vectors=[np.zeros(192).tolist()],
                        scoring=dict(metric=dict(projection=np.eye(192).tolist())))
        c.append(dict(root_sha256=str(root), compiled=compiled))
        plans, branches = [], []
        for arm, value in [('composition', 1.), ('flat_search', 2.)]:
            vector = [value]*192
            actions = [[0., 0.]]*10
            edge = dict(actions=actions, output_context=dict(latent_history=[vector]))
            plans.append(dict(arm=arm, witness=dict(independent_full_replay_exact=True, edges=[edge]),
                              selected_model_cost=value**2))
            branches.append(dict(arm=arm, planned_commands=10, completed_commands=10,
                remaining_commands_not_executed=0, status='completed', final_vector=vector,
                trace=[dict(primitive_step=i+1, command=a, state=[0.]*7) for i,a in enumerate(actions)],
                final_state=[0.]*7, pose=dict(pose_cost=value),
                official=dict(official_state_distance=value, official_success=False)))
        p.append(dict(root_sha256=str(root), rows=plans))
        o.append(dict(root_sha256=str(root), branches=branches))
    return [dict(rows=rows) for rows in (c,p,o)]


def test_identity_join_not_array_order():
    c,p,o = data()
    o['rows'].reverse()
    result = decompose(c,p,o)
    assert all(r['real_q_vs_object_pose_order_same'] for r in result['rows'])
    assert result['P2_admission'] is False


@pytest.mark.parametrize('fault', ['action', 'duplicate_root', 'score', 'nan', 'missing_arm'])
def test_reject_corruption(fault):
    c,p,o = data()
    if fault == 'action':
        o['rows'][0]['branches'][0]['trace'][0]['command'] = [.1,.2]
    elif fault == 'duplicate_root':
        o['rows'][1]['root_sha256'] = '0'
    elif fault == 'score':
        p['rows'][0]['rows'][0]['selected_model_cost'] += .1
    elif fault == 'nan':
        o['rows'][0]['branches'][0]['final_vector'][0] = float('nan')
    else:
        p['rows'][0]['rows'].pop()
    with pytest.raises(ValueError):
        decompose(c,p,o)


def test_abort_kept_not_equal_horizon():
    c,p,o = data()
    b = o['rows'][0]['branches'][0]
    b.update(completed_commands=5, remaining_commands_not_executed=5,
             status='observation_unavailable', trace=b['trace'][:5])
    result = decompose(c,p,o)
    assert len(result['rows']) == 4
    assert result['rows'][0]['fixed_horizon_pair'] is False


def test_real_representation_can_disagree_without_prediction_error():
    c,p,o = data()
    o['rows'][0]['branches'][0]['pose']['pose_cost'] = 9.
    result = decompose(c,p,o)['rows'][0]
    assert result['predicted_vs_real_q_order_same'] is True
    assert result['real_q_vs_object_pose_order_same'] is False
