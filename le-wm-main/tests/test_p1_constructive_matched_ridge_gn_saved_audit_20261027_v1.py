"""Pure-array and actual saved regression; no tmp fixtures or model execution."""
import ast
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_matched_ridge_gn_saved_audit_20261027_v1 as a

REPORT=a.io.ROOT/'results/p1_constructive_matched_ridge_gn_cuda_20261027_v1_cuda_main01/report.json'
# Locally observed immutable snapshot; MAIN independently confirms before formal CLI receipt.
REPORT_SHA='7eed1f6ab7543e6dc4be367c5141ab23cb44dc014fae770fc58fe813422137d5'
PREPARE_SHA='cec47cb477b06cbbecc9a402f6f17415d3a67cec576679aab906dc68a733bf05'


def test_normal_equations_match_mean_lambda_not_old_lambda():
    rng=np.random.default_rng(20261027);j=rng.normal(size=(192,20));r=rng.normal(size=192)
    d=a.direction(j,r);left,s,right=np.linalg.svd(j/np.sqrt(192),full_matrices=False)
    matched=-right.T@(s*(left.T@(r/np.sqrt(192)))/(s*s+.002/192))
    old=-right.T@(s*(left.T@(r/np.sqrt(192)))/(s*s+.002))
    np.testing.assert_allclose(d,matched,rtol=1e-12,atol=1e-13)
    assert not np.allclose(d,old,rtol=1e-6,atol=1e-8)
    assert 192*a.LAMBDA_MEAN==a.LAMBDA_RAW==.002


def test_import_boundary():
    tree=ast.parse(Path(a.__file__).read_text())
    imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    imports += [v.name for n in ast.walk(tree) if isinstance(n,ast.Import) for v in n.names]
    assert not any(any(k in name for k in ('torch','matched_ridge_gn_cuda','planners_cuda','encode_cuda','task_admission')) for name in imports)
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('optimize','consume','selected','task_export','gn_audit','audit') for n in ast.walk(tree))


@pytest.mark.parametrize('bad',['7eed1f6','x'*64,'A'*64,None])
def test_external_sha_must_be_full(bad):
    with pytest.raises(ValueError):a.external_sha(bad)


@pytest.fixture(scope='module')
def actual():
    if not REPORT.exists():pytest.skip('MAIN saved variant receipt not installed')
    r=a.util.read(REPORT,REPORT_SHA);p=a.util.read(r['prepared'],PREPARE_SHA)
    events=[a.io.parse(line) for line in a.util.load(REPORT.parent/'events.jsonl.gz').splitlines()]
    material=[]
    for i in range(2):
        t=a.util.export_task(p['slots'][i]['task_export'],p['slots'][i]['task_sha256'])
        result=a.util.read(REPORT.parent/r['rows'][i]['result_file'])
        group=[e['event'] for e in events if e['source_slot']==i]
        material.append((t,result,group))
    return r,p,material


def test_actual_complete_saved_arithmetic(actual):
    result=a.audit(REPORT,REPORT_SHA,PREPARE_SHA)
    assert result['status']=='matched_ridge_saved_arithmetic_consistent_not_admitted'
    assert result['jacobians_recomputed']==8 and result['raw_VJP_rows_checked']==1536
    assert result['resources']['predict']==dict(attempted=156,completed=156)
    assert result['resources']['action_encoder']==dict(attempted=156,completed=156)
    assert result['event_count']==8734
    assert [r['model_terminal_cost'] for r in result['rows']]==[1.0146352777046366,1.4551953375149658]
    assert result['new_neural_calls']==result['new_physics']==0
    assert result['P1_admission'] is result['equal_budget_claim'] is False


@pytest.mark.parametrize('bad',['meanlambda','rawlambda','old_direction','delta','endpoint','rawrow','J','gradient','ADpoint','history','action','cost','scale','selection','missingtrial','reorder','missingreturn','duplicateAPI','batchedAPI','sample_units'])
def test_root_mutations_rejected(actual,bad):
    t,r,events=deepcopy(actual[2][0])
    def event(kind):return next(e for e in events if e['kind']==kind)
    if bad=='meanlambda':r['lambda_mean']=.002
    if bad=='rawlambda':event('line_proposals_frozen')['lambda_unnormalized']=.384
    if bad=='old_direction':
        d=event('derivative_complete')['derivative'];j=np.asarray(d['raw_terminal_jacobian']);residual=np.asarray(d['at_terminal'])-t['goal']
        event('line_proposals_frozen')['delta']=np.linalg.solve(j.T@j+.384*np.eye(20),-j.T@residual).tolist()
    if bad=='delta':event('line_proposals_frozen')['delta'][0]+=1
    if bad=='endpoint':event('line_proposals_frozen')['endpoint'][0][0]+=.1
    if bad=='rawrow':event('jacobian_row')['raw_control_derivative'][0]+=1
    if bad=='J':event('derivative_complete')['derivative']['raw_terminal_jacobian'][0][0]+=1
    if bad=='gradient':event('derivative_complete')['derivative']['objective_gradient'][0]+=1
    if bad=='ADpoint':event('derivative_complete')['derivative']['at_logits'][0][0]+=.1
    if bad=='history':event('evaluation_complete')['prediction']['contexts'][0]['action_history'][0][0]+=.1
    if bad=='action':event('evaluation_request')['commands'][0][0]+=.1
    if bad=='cost':event('evaluation_complete')['prediction']['cost']+=1
    if bad=='scale':event('line_proposals_frozen')['scales'][0]=2.
    if bad=='selection':event('selection_complete')['selected_index']=99
    if bad=='missingtrial':events.remove(event('evaluation_complete'))
    if bad=='reorder':
        i=events.index(event('line_proposals_frozen'));j=events.index(event('selection_complete'));events[i],events[j]=events[j],events[i]
    if bad=='missingreturn':events.remove(event('operation_completed'))
    if bad=='duplicateAPI':events.append(event('AD_API_completed'))
    if bad=='batchedAPI':event('AD_API_attempt')['entry']['is_grads_batched']=True
    if bad=='sample_units':
        event('operation_attempted')['event']['samples']=2
        event('operation_completed')['event']['samples']=2
    with pytest.raises(ValueError):a.audit_root(t,r,events)


@pytest.mark.parametrize('bad',['partial','commands','blind','totals','model'])
def test_report_mutations_rejected(actual,monkeypatch,bad):
    r=deepcopy(actual[0]);original=a.util.read
    if bad=='partial':r['status']='partial_or_failed'
    if bad=='commands':r['rows'][0]['commands'][0][0]+=.1
    if bad=='blind':r['rows'][0]['blind_sha256']='0'*64
    if bad=='totals':r['resources']['predict']['completed']-=1
    if bad=='model':r['model_hash_after']='0'*64
    monkeypatch.setattr(a.util,'read',lambda path,expected=None:r if Path(path)==REPORT else original(path,expected))
    with pytest.raises(ValueError):a.audit(REPORT,REPORT_SHA,PREPARE_SHA)


def test_actual_lambda_diagnostics_all_eight_present(actual):
    all_iterations=[]
    for t,r,events in actual[2]:
        _,diagnostics=a.audit_root(t,r,events);all_iterations.extend(diagnostics)
    assert len(all_iterations)==8
    assert all(d['raw_VJP_rows']==192 and d['lambda_mean']==.002/192 and d['lambda_unnormalized']==.002 for d in all_iterations)
    # Error values are reported without an extra scientific passing threshold.
    assert all(np.isfinite(d['normal_equation_residual_norm']) and d['max_delta_difference']>=0 for d in all_iterations)
