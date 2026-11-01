"""Non-neural consumer contracts and durable physical fixtures, not efficacy."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.spectral_reachability import pusht_full_packet_physical_20261027 as p


@pytest.fixture
def fixture():
    from experiments.spectral_reachability.pusht_full_packet_compiler_20261027 import (
        spectral_bank,bounded_ridge,CAP,DAMPING,RADII,
    )
    from experiments.spectral_reachability.pusht_full_packet_chart_20261027 import chart_matrix
    goal=np.ones(192);q=np.eye(192);jac=np.eye(192,10);origin=np.zeros(192)
    compiled=dict(goal_vectors=[goal.tolist()],scoring=dict(metric=dict(projection=q.tolist())))
    source=dict(protocol=dict(roots=[f'root{i}' for i in range(4)]),
        rows=[dict(root_sha256=f'root{i}',compiled=deepcopy(compiled)) for i in range(4)])
    rows=[]
    for i in range(4):
        arms=[]
        for arm in p.ARMS:
            basis=chart_matrix(arm);d=basis.shape[1]
            bank=spectral_bank(jac@basis,origin,goal,q)
            candidates=[]
            for j in range(10):
                prefix_params=np.asarray(bank['params'][j],dtype=np.float32)
                nominal=np.zeros(192);response=np.eye(192,d)
                own=bounded_ridge(q@response,q@(goal-nominal))
                suffix_params=np.asarray(own['params'],dtype=np.float32)
                prefix=np.tanh(basis.astype(np.float32)@prefix_params).reshape(5,2)
                suffix=np.tanh(basis.astype(np.float32)@suffix_params).reshape(5,2)
                predicted_prefix=np.full(192,j/20);predicted_terminal=np.full(192,.1)
                stage=float(np.mean((goal-predicted_prefix)**2))
                candidates.append(dict(candidate_id=j,kind=bank['kinds'][j],commands=np.concatenate((prefix,suffix)).tolist(),
                    prefix_commands=prefix.tolist(),suffix_commands=suffix.tolist(),
                    prefix_params=prefix_params.tolist(),suffix_params=suffix_params.tolist(),
                    nominal_terminal=nominal.tolist(),jacobian=response.tolist(),
                    predicted_prefix=predicted_prefix.tolist(),predicted_terminal=predicted_terminal.tolist(),
                    stage_cost=stage,own_reach_value=own['own_reach_value'],objective=stage+own['own_reach_value'],
                    nonlinear_terminal_cost=float(np.mean((goal-predicted_terminal)**2)),independent_model_replay_exact=True))
            arms.append(dict(arm=arm,chart=basis.tolist(),bank=bank,selected_candidate_id=9,greedy_candidate_id=9,candidates=candidates))
        rows.append(dict(root_index=i,root_sha256=f'root{i}',arms=arms,
            root_jacobian_full=jac.tolist(),root_nominal_terminal=origin.tolist()))
    report=dict(status='plans_sealed',protocol=dict(roots=source['protocol']['roots'],
        damping=DAMPING,norm_cap_atanh_per_packet=CAP,spectral_radii=RADII),plans=rows,
        gates=dict(all_eighty_candidates=True),all_plans_before_physical_execution=True)
    return report,source


def test_all_eighty_and_full_actions_not_collapsed(fixture):
    snapshot=deepcopy(fixture)
    plans=p.plans_from_saved(*fixture)
    assert len(plans)==80 and len({(r['root_sha256'],r['method']) for r in plans})==80
    assert all(x['planned_commands']==10 and x['planned_horizon_tokens']==2 for x in plans)
    full=next(x for x in plans if x['arm']=='full' and x['candidate_id']=='2')
    assert not np.array_equal(full['commands'][:5],np.repeat([full['commands'][0]],5,axis=0))
    assert {x['selected_candidate_id'] for x in plans}=={'9'}
    assert {x['zero_candidate_id'] for x in plans}=={'0'}
    assert fixture==snapshot


@pytest.mark.parametrize('mutation',['status','root','root_index','arm','count','duplicate','selected','greedy',
    'commands','prefix','suffix_constant','vector','cost','zero','unsafe_id','bool_command'])
def test_invalid_plans_fail_before_environment(fixture,mutation):
    report,source=deepcopy(fixture)
    arm=report['plans'][0]['arms'][0]; candidate=arm['candidates'][0]
    if mutation=='status':report['status']='pending'
    elif mutation=='root':report['plans'].pop()
    elif mutation=='root_index':report['plans'][0]['root_index']=True
    elif mutation=='arm':arm['arm']='full'
    elif mutation=='count':arm['candidates'].pop()
    elif mutation=='duplicate':arm['candidates'][1]['candidate_id']=0
    elif mutation=='selected':arm['selected_candidate_id']=20
    elif mutation=='greedy':arm['greedy_candidate_id']=20
    elif mutation=='commands':candidate['commands'][0][0]=1.1
    elif mutation=='prefix':candidate['prefix_commands'][0][0]=.1
    elif mutation=='suffix_constant':
        candidate['suffix_commands'][0][0]=.2; candidate['commands'][5][0]=.2
    elif mutation=='vector':candidate['predicted_prefix']=[0.]
    elif mutation=='cost':candidate['objective']=float('nan')
    elif mutation=='zero':candidate['prefix_commands']=[[.1,.1]]*5; candidate['commands'][:5]=[[.1,.1]]*5
    elif mutation=='unsafe_id':candidate['candidate_id']='../../bad'
    else:candidate['commands']=[[True,False]]*10
    with pytest.raises(ValueError):p.plans_from_saved(report,source)


class Env:
    def __init__(self,missing_at=None):self.n=0;self.missing_at=missing_at
    def render(self):
        image=np.zeros((224,224,3),np.uint8)
        if self.n!=self.missing_at:image[10:12,10:12]=[65,105,225]
        return image
    def step(self,commands):self.n+=1;return None,0.,False,False,{}


def execute(tmp_path,fixture,missing_at=None):
    plan=p.plans_from_saved(*fixture)[0]
    row=p.durable.physical_branch(plan=plan,repetition=0,output=tmp_path,reset=lambda:{},
        env=Env(missing_at),state_of=lambda env:np.array([env.n,0,0,0,0.]),evaluate=lambda state:dict(official={},pose={}))
    p.write(tmp_path/'physical_report.json',dict(rows=[row]))
    return row,plan


def test_prefix_and_terminal_read_exact_saved_steps(tmp_path,fixture):
    row,plan=execute(tmp_path,fixture)
    paths=[]
    def setup():
        assert (tmp_path/'physical_report.json').is_file()
        def encode(path):paths.append(path.name);return [0.]*192
        return encode
    receipts=p.encode_observations([row],[plan],tmp_path,setup)
    assert [r['kind'] for r in receipts]==['prefix','terminal']
    assert all(r['comparable'] and r['attempted_encode_samples']==1 for r in receipts)
    assert paths[0].endswith('_rgb_5.png') and paths[1].endswith('_rgb_10.png')


@pytest.mark.parametrize('stop,prefix,terminal',[(0,False,False),(3,False,False),(5,True,False),(7,True,False),(10,True,True)])
def test_guard_aborts_no_padding_and_comparison_census(tmp_path,fixture,stop,prefix,terminal):
    row,plan=execute(tmp_path,fixture,missing_at=stop)
    results=p.encode_observations([row],[plan],tmp_path,lambda:lambda path:[0.]*192)
    assert [r['comparable'] for r in results]==[prefix,terminal]
    assert row['completed_commands']==stop
    assert row['status']=='observation_unavailable'


def test_encoding_setup_failure_preserves_physics(tmp_path,fixture):
    row,plan=execute(tmp_path,fixture)
    physics=next(tmp_path.glob('*.physical.json')); raw=physics.read_bytes()
    def fail():raise RuntimeError('CUDA unavailable')
    receipts=p.encode_observations([row],[plan],tmp_path,fail)
    assert all(r['status']=='encoding_setup_failed' and r['attempted_encode_samples']==0 for r in receipts)
    assert physics.read_bytes()==raw


def test_prefix_encoder_failure_still_attempts_terminal(tmp_path,fixture):
    row,plan=execute(tmp_path,fixture)
    def setup():
        def encode(path):
            if path.name.endswith('_rgb_5.png'):raise RuntimeError('encoder failed')
            return [0.]*192
        return encode
    receipts=p.encode_observations([row],[plan],tmp_path,setup)
    assert [r['status'] for r in receipts]==['encoding_failed','completed']
    assert sum(r['attempted_encode_samples'] for r in receipts)==2


def test_changed_journal_fail_closed(tmp_path,fixture):
    row,plan=execute(tmp_path,fixture)
    row['journal_sha256']='changed'
    receipts=p.encode_observations([row],[plan],tmp_path,lambda:lambda path:pytest.fail('no encode allowed'))
    assert all(not r['comparable'] and r['attempted_encode_samples']==0 for r in receipts)


def test_missing_census_materialized_all_eighty(tmp_path,fixture):
    plans=p.plans_from_saved(*fixture); rows=[]
    p.materialize_missing(plans,rows,tmp_path)
    assert len(rows)==len(list(tmp_path.glob('*.physical.json')))==80
    assert all(r['status']=='blocked' and r['attempted_commands']==0 for r in rows)
    comparisons=p.comparisons(plans,rows)
    assert len(comparisons)==8
    assert all(not role['full_horizon_comparable'] for c in comparisons for role in c['roles'].values())
    assert comparisons[0]['roles']['selected']['candidate_id']=='9'
    assert comparisons[0]['roles']['greedy']['candidate_id']=='9'
    assert comparisons[0]['roles']['zero']['candidate_id']=='0'


def test_no_encoding_before_physics(tmp_path):
    with pytest.raises(ValueError):p.encode_observations([],[],tmp_path,lambda:pytest.fail('setup forbidden'))


@pytest.mark.parametrize('mutation',['false_gate','unsealed_physical','chart','cap','params','objective',
    'selected_reordered','greedy_reordered','replay','kind'])
def test_compiler_metric_basis_and_selection_rechecked(fixture,mutation):
    report,source=deepcopy(fixture)
    arm=report['plans'][0]['arms'][1];c=arm['candidates'][1]
    if mutation=='false_gate':report['gates']['all_eighty_candidates']=False
    elif mutation=='unsealed_physical':report['all_plans_before_physical_execution']=False
    elif mutation=='chart':arm['chart'][0][0]=2
    elif mutation=='cap':report['protocol']['norm_cap_atanh_per_packet']=99
    elif mutation=='params':c['suffix_params'][0]+=1e-3
    elif mutation=='objective':c['objective']+=1e-3
    elif mutation=='selected_reordered':arm['selected_candidate_id']=8
    elif mutation=='greedy_reordered':arm['greedy_candidate_id']=8
    elif mutation=='kind':c['kind']='wrong'
    else:c['independent_model_replay_exact']=False
    with pytest.raises(ValueError):p.plans_from_saved(report,source)


def test_q_goal_metrics_from_saved_physical_images(tmp_path,fixture):
    row,plan=execute(tmp_path,fixture)
    results=p.encode_observations([row],[plan],tmp_path,lambda:lambda path:[0.]*192,
        metrics={row['root_sha256']:(np.ones(192),2*np.eye(192))})
    assert all(r['observed_Q_goal_cost']==4 for r in results)
    assert all('projected_prediction_mse' in r for r in results)
