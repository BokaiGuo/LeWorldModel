"""Read-only saved-array attribution and tiny mocks; no model or simulator imports."""
from pathlib import Path
import hashlib
import json
import math
from collections import Counter

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'results/p2_composition_predict_frontier_physical_20261027_v1_main01'
REPORT_SHA = '9144c3f25f2880af62d94dd88257918fa508b5ddbea51d98e9ccbc3d3b7532e2'
PREPARE_SHA = 'ea9e30764c6f5a12e84e109987eedb3d425ed413d8af5cc66b884157035f37a2'
PLANNER_SHA = '4023a0e3896a90407188d260a45dd334e0802ba17ecd59ef62ea931603eba173'
ARMS = ('joint_h2_no_composition', 'conditional_suffix_composition')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def indexed(rows, fields=('root_sha256','method')):
    out = {tuple(r[k] for k in fields):r for r in rows}
    if len(out) != len(rows):
        raise ValueError('duplicate stable identity')
    return out


def sign(x):
    if not math.isfinite(x):
        raise ValueError('nonfinite delta')
    return (x > 0) - (x < 0)


def attribution(predicted_delta, actual_q_delta, pose_delta):
    """Exact paired ordering only, not significance thresholds or causal proof."""
    p,q,o = map(sign, (predicted_delta,actual_q_delta,pose_delta))
    return dict(prediction_reversal=p*q == -1, proxy_mismatch=q*o == -1,
        pose_harm=o > 0, tie_in_chain=0 in (p,q,o))


@pytest.mark.parametrize('p,q,o,rev,proxy,harm', [
    (-1,1,-1,True,True,False), (1,1,1,False,False,True),
    (-1,1,1,True,False,True), (1,-1,1,True,True,True),
    (-1,-1,1,False,True,True), (0,1,1,False,False,True)])
def test_mock_attribution_distinguishes_two_layers_and_expected_loss(p,q,o,rev,proxy,harm):
    r = attribution(p,q,o)
    assert (r['prediction_reversal'],r['proxy_mismatch'],r['pose_harm']) == (rev,proxy,harm)
    assert r['tie_in_chain'] == (p == 0)


def test_mock_identity_join_ignores_order_but_rejects_duplicates():
    rows = [dict(root_sha256=str(i),method=a,value=i) for i in range(4) for a in ARMS]
    assert indexed(rows) == indexed(rows[::-1])
    with pytest.raises(ValueError): indexed(rows+[rows[0]])
    with pytest.raises(ValueError): attribution(float('nan'),1,1)


def same_number(a,b):
    assert np.isclose(a,b,atol=1e-10,rtol=1e-12), (a,b)


def score(z, goal, q):
    z,goal,q = np.asarray(z,float),np.asarray(goal,float),np.asarray(q,float)
    assert z.shape == goal.shape == (192,) and q.shape == (192,192)
    assert np.isfinite(z).all() and np.isfinite(goal).all() and np.isfinite(q).all()
    r = q@(z-goal)
    return float(np.mean(r*r))


def saved_audit():
    assert sha(RUN/'report.json') == REPORT_SHA
    report = load(RUN/'report.json'); execution = load(RUN/'execution_protocol.json')
    prepared = Path(execution['prepared'])
    assert sha(prepared/'prepare.json') == report['consumer_prepare_sha256'] == execution['prepare_sha256'] == PREPARE_SHA
    receipt = load(prepared/'prepare.json'); protocol = load(prepared/'protocol.json')
    assert sha(prepared/'protocol.json') == receipt['protocol_sha256']
    assert sha(Path(protocol['planner_output'])/'report.json') == report['planner_report_sha256'] == PLANNER_SHA
    for path,h in {**protocol['sources_sha256'],**protocol['input_bindings']}.items():
        assert sha(path) == h
    assert report['status'] == 'guarded_calibration_recorded' and report['failures'] == []
    assert report['source_verified'] is report['model_unchanged'] is report['full_horizon_all_branches'] is True
    assert all(report[f'P{i}_admission'] is False for i in range(1,6))
    assert sha(RUN/'physical_report.json') == report['physical_report_sha256']
    physical = load(RUN/'physical_report.json')
    assert physical['rows'] == report['rows'] and physical['failures'] == []
    assert physical['physical_call_receipt'] == report['physical_call_receipt']
    plans, rows, enc = map(indexed, (protocol['plans'],report['rows'],report['encodings']))
    assert len(plans) == len(rows) == len(enc) == 8 and plans.keys() == rows.keys() == enc.keys()
    assert sum(r['official']['official_success'] for r in rows.values()) == 0
    assert sum(r['official']['object_only_success'] for r in rows.values()) == 0
    budget_path = RUN/'physical_budget.events.jsonl'
    assert sha(budget_path) == report['physical_budget_journal_sha256'] == physical['physical_budget_journal_sha256']
    budget = [json.loads(line) for line in budget_path.read_text().splitlines()]
    census = Counter((e['kind'],e['phase'],e['completed']) for e in budget)
    expected_steps = dict(root_collection=0,warmup=120,selected=80)
    expected_resets = dict(root_collection=32,warmup=8,selected=0)
    for kind,table,name in (('env_step',expected_steps,'steps'),('env_reset',expected_resets,'resets')):
        for phase,n in table.items():
            assert census[kind,phase,False] == census[kind,phase,True] == n
            assert report['physical_call_receipt'][name][phase] == dict(attempts=n,returns=n)
    assert len(budget) == 480 and not any('error' in e for e in budget)
    stats = report['encoding_stats']
    assert all(stats[k+'_attempts'] == stats[k+'_completed'] == 8 for k in ('encoder','projector'))
    assert stats['forbidden_predict_attempts'] == 0
    encoding_events = [json.loads(line) for line in (RUN/'encoding.events.jsonl').read_text().splitlines()]
    for kind in ('encoder','projector'):
        assert [e['index'] for e in encoding_events if e['kind'] == kind and not e['completed']] == list(range(8))
        assert [e['index'] for e in encoding_events if e['kind'] == kind and e['completed']] == list(range(8))
    result = []
    for key,p in plans.items():
        r,e = rows[key],enc[key]
        assert r['selection_sha256'] == e['selection_sha256'] == p['selection_sha256']
        assert r['active_metric_id'] == e['active_metric_id'] == p['active_metric_id']
        assert r['status'] == 'completed' and not r['failures'] and r['failure'] is None
        assert r['full_horizon_completed'] and r['endpoint_fresh'] and r['endpoint_scope'] == 'full_H2'
        assert r['physical_env_step_attempts'] == r['physical_env_step_returns'] == r['planned_commands'] == 10
        assert e['status'] == 'completed' and e['failure'] is None and e['attempted_encode_samples'] == 1
        assert sha(r['terminal_rgb_path']) == e['terminal_rgb_sha256'] == r['terminal_rgb_sha256']
        stem = f"root_{p['root_index']}_{p['method']}"
        assert load(RUN/(stem+'.physical.json')) == r
        assert load(RUN/(stem+'.encoding.json')) == e
        assert sha(r['journal_path']) == r['journal_sha256']
        journal = [json.loads(line) for line in Path(r['journal_path']).read_text().splitlines()]
        attempts = [v for v in journal if v['kind'] == 'attempt']
        returns = [v for v in journal if v['kind'] == 'step_returned']
        assert [v['primitive_step'] for v in attempts] == [v['primitive_step'] for v in returns] == list(range(1,11))
        assert [v['command'] for v in attempts] == p['commands']
        images = [v for v in journal if 'rgb_path' in v]
        assert len(images) == len(r['guard']['observations']) == 11 and images[-1]['primitive_step'] == 10
        for v,guard in zip(images,r['guard']['observations']):
            assert sha(v['rgb_path']) == v['rgb_file_sha256']
            with Image.open(v['rgb_path']) as image: rgb = np.asarray(image)
            assert rgb.shape == (224,224,3) and rgb.dtype == np.uint8
            assert int(np.all(rgb == [65,105,225],axis=-1).sum()) == guard['canonical_agent_pixels'] > 0
        qs = protocol['root_metrics'][p['root_sha256']]
        assert p['goal'] == qs['goal']
        actual = score(e['terminal_vector'],qs['goal'],qs['metrics']['source_Q']['projection'])
        predicted = score(p['selected_predicted_context']['latent_history'][-1],qs['goal'],qs['metrics']['source_Q']['projection'])
        same_number(actual,e['actual_costs']['source_Q'])
        same_number(predicted,e['predicted_costs']['source_Q']);same_number(predicted,p['predicted_costs']['source_Q'])
        state,goal = np.asarray(r['final_state']),np.asarray(r['context_identity']['goal'])
        xy = float(np.linalg.norm(state[2:4]-goal[2:4]))
        angle = float(abs(np.arctan2(np.sin(state[4]-goal[4]),np.cos(state[4]-goal[4]))))
        same_number(xy,r['pose']['xy_distance']);same_number(angle,r['pose']['angle_distance'])
        same_number(xy+20*angle,r['pose']['pose_cost'])
        same_number(float(np.linalg.norm(state-goal)),r['official']['official_state_distance'])
        official_angle = min(abs(goal[4]-state[4]),2*np.pi-abs(goal[4]-state[4]))
        assert r['official']['official_success'] == bool(np.linalg.norm(goal[:4]-state[:4]) < 20 and official_angle < np.pi/9)
        assert r['official']['object_only_success'] == bool(xy < 20 and official_angle < np.pi/9)
    for root in sorted({key[0] for key in plans},key=lambda root:rows[root,ARMS[0]]['root_index']):
        a,b = (rows[root,arm] for arm in ARMS); x,y = (enc[root,arm] for arm in ARMS)
        assert a['context_identity'] == b['context_identity']
        assert a['active_metric_id'] == b['active_metric_id']
        dp = y['predicted_costs']['source_Q']-x['predicted_costs']['source_Q']
        dq = y['actual_costs']['source_Q']-x['actual_costs']['source_Q']
        do = b['pose']['pose_cost']-a['pose']['pose_cost']
        official = b['official']['official_state_distance']-a['official']['official_state_distance']
        pair = next(c for c in report['comparisons'] if c['root_sha256'] == root)
        same_number(do,pair['right_minus_left_pose']);same_number(official,pair['right_minus_left_official'])
        result.append(dict(root_index=a['root_index'],root_sha256=root,
            predicted_Q=[x['predicted_costs']['source_Q'],y['predicted_costs']['source_Q']],
            actual_Q=[x['actual_costs']['source_Q'],y['actual_costs']['source_Q']],
            delta_predicted_Q=dp,delta_actual_Q=dq,delta_pose=do,delta_official=official,
            **attribution(dp,dq,do)))
    assert sha(RUN/'report.json') == REPORT_SHA
    return result


def test_actual_saved_receipts_and_all_eight_endpoint_arrays():
    rows = saved_audit()
    assert [(r['prediction_reversal'],r['proxy_mismatch'],r['pose_harm']) for r in rows] == [
        (True,True,False),(False,False,True),(True,False,True),(True,True,True)]
    print('\nACTUAL_SAVED_ATTRIBUTION='+json.dumps(rows,sort_keys=True,allow_nan=False))
