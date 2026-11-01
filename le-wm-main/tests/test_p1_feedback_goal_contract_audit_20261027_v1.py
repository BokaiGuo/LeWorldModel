"""Non-neural contract tests; fixtures are not physical efficacy evidence."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.spectral_reachability import p1_feedback_goal_contract_audit_20261027_v1 as a


def fixture():
    q = np.eye(192); goal = np.ones(192)
    case = dict(root_sha256=a.ROOT, root_index=1, role='selected', arm='actual',
                prefix=np.zeros((5,2)).tolist(), context={'example':'fixture only'})
    plan = dict(**deepcopy(case), q=q.tolist(), goal=goal.tolist(), independent_replay_exact=True,
                candidates=[], selected_candidate_id='suffix_9')
    for i in range(10):
        pred = np.ones(192)*i/10
        plan['candidates'].append(dict(candidate_id=f'suffix_{i}', suffix_params=[0.]*10,
            commands=np.zeros((10,2)).tolist(), predicted_terminal=pred.tolist(),
            nonlinear_terminal_cost=a.projected_cost(q,goal,pred)))
    return plan, case, q, goal


def test_all_candidates_recomputed():
    report = a.audit_plan(*fixture())
    assert len(report['candidate_costs']) == 10
    assert report['selected_candidate_id'] == 'suffix_9'


@pytest.mark.parametrize('damage', ['root','context','goal','q','cost','duplicate','prefix','raw',
                                  'nan','winner','missing','replay','wrong_endpoint'])
def test_reject_plan_damage(damage):
    p,c,q,g = fixture()
    if damage == 'root': p['root_index'] = 2
    if damage == 'context': p['context'] = {'wrong':'history'}
    if damage == 'goal': p['goal'][0] = 0
    if damage == 'q': p['q'][0][0] = 0
    if damage == 'cost': p['candidates'][0]['nonlinear_terminal_cost'] += .1
    if damage == 'duplicate': p['candidates'][1]['candidate_id'] = 'suffix_0'
    if damage == 'prefix': p['candidates'][0]['commands'][0][0] = .1
    if damage == 'raw': p['candidates'][0]['commands'][5][0] = .1
    if damage == 'nan': p['candidates'][0]['predicted_terminal'][0] = float('nan')
    if damage == 'winner': p['selected_candidate_id'] = 'suffix_0'
    if damage == 'missing': p['candidates'].pop()
    if damage == 'replay': p['independent_replay_exact'] = False
    if damage == 'wrong_endpoint': p['candidates'][0]['predicted_terminal'] = p['candidates'][1]['predicted_terminal']
    with pytest.raises(ValueError): a.audit_plan(p,c,q,g)


def goal_fixture():
    goals = np.zeros((5,192)); goals[1:, :4] = np.eye(4)
    q, rank = a.projection_from_goals(goals)
    assert rank == 4
    target = np.array([170.,252.,256.,256.,.7,0.,0.])
    geometry = []
    for off in [[0.,0.],[-32.,0.],[32.,0.],[0.,-32.],[0.,32.]]:
        state = target.copy(); state[:2] += off
        geometry.append(dict(offset=off, requested_goal=state.tolist(), actual_goal_render_state=state.tolist()))
    return goals,geometry,q


def test_finite_goal_official_counterexample():
    rows = a.finite_witnesses(*goal_fixture())
    assert rows[0]['official_success_vs_original']
    assert all(not r['official_success_vs_original'] and r['official_state_distance_vs_original']==32
               and r['Q_goal_MSE']==0 and r['object_xy_error']==0 for r in rows[1:])


@pytest.mark.parametrize('damage',['actual_render','object_target','order','bad_q','nonfinite'])
def test_reject_witness_damage(damage):
    g,geo,q = goal_fixture()
    if damage == 'actual_render': geo[0]['actual_goal_render_state'][2] += 1
    if damage == 'object_target':
        geo[1]['requested_goal'][2] += 1; geo[1]['actual_goal_render_state'][2] += 1
    if damage == 'order': geo[1],geo[2] = geo[2],geo[1]
    if damage == 'bad_q': q = np.eye(192)
    if damage == 'nonfinite': geo[0]['actual_goal_render_state'][0] = float('nan')
    with pytest.raises(ValueError): a.finite_witnesses(g,geo,q)


def test_official_strict_boundaries():
    g = np.zeros(7); s = g.copy(); s[0] = 20
    assert not a.official(g,s)[0]
    s[0] = 0; s[4] = np.pi/9
    assert not a.official(g,s)[0]
    s[4] = 2*np.pi-.01
    assert a.official(g,s)[0]


def test_ast_evaluator_rejects_threshold_change(tmp_path):
    env = a.REPO/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py'
    a.check_evaluator(env)
    text = env.read_text().replace('pos_diff < 20', 'pos_diff < 21')
    changed = tmp_path/'changed.py'; changed.write_text(text)
    with pytest.raises(ValueError): a.check_evaluator(changed)


def test_input_hash_and_end_check(tmp_path):
    p = tmp_path/'data.json'; p.write_text(json.dumps({'fixed':1}))
    inputs = a.Inputs(); inputs.load(p); inputs.unchanged()
    with pytest.raises(ValueError): inputs.bind(p,'0'*64)
    p.write_text('{}')
    with pytest.raises(ValueError): inputs.unchanged()


def test_source_stats_are_packet_major(tmp_path):
    p = tmp_path/'stats.py'
    p.write_text('PUBLIC_ACTION_MEAN=np.asarray([1.,2.],dtype=np.float32)\n'
                 'PUBLIC_ACTION_STD=np.asarray([3.,4.],dtype=np.float32)\n')
    mean,std = a.public_action_stats(p)
    np.testing.assert_array_equal(mean,[1,2]*5)
    np.testing.assert_array_equal(std,[3,4]*5)
    p.write_text('PUBLIC_ACTION_MEAN=np.asarray([1.,2.])')
    with pytest.raises(ValueError): a.public_action_stats(p)


def test_no_neural_or_simulator_imports():
    import ast
    tree = ast.parse(a.Path(a.__file__).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [v.name for n in ast.walk(tree) if isinstance(n,ast.Import) for v in n.names]
    assert not any(n and n.split('.')[0] in {'torch','gym','gymnasium','stable_worldmodel'} for n in imports)
