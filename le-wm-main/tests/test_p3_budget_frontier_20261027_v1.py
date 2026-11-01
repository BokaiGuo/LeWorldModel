"""Saved real receipts + deliberate counterexamples; no neural/physics execution."""
import ast
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.spectral_reachability import p3_budget_frontier_20261027_v1 as audit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence():
    paths = [audit.saved.MAIN,audit.saved.BINDING,audit.saved.PHYSICAL,
             audit.ORIGINAL,audit.RUNTIME,audit.PHYSICAL_PROTOCOL,audit.OLD]
    return [audit.saved.decode((ROOT/p).read_bytes()) for p in paths]


def reseal(plan):
    plan['selection_sha256'] = audit.sha(audit.canonical(
        {k:v for k,v in plan.items() if k!='selection_sha256'}).encode())


def test_real_receipt_census(evidence):
    r = audit.audit(*evidence)
    assert r['classifications'] == {'full-path':2,'guard-partial':2,'unmatched':4}
    assert r['totals'] == dict(planning_predict_samples=1152,verification_predict_samples=19,
        predict_samples=1171,action_encoder_samples=1171,processed_action_context_tokens=3513,
        physical_step_attempts=154,physical_step_returns=154,terminal_encode_samples=16)
    assert len(r['arm_costs']) == 8 and len(r['rows']) == 16
    assert r['guard_aborted_executions'] == 4
    assert all(v==0 for v in r['derivative_totals'].values())
    assert all(v is None for v in r['unknown_or_excluded_costs'].values())
    assert r['repeats_are_dependent'] and r['shared_planning_charged_once']
    assert r['previous_audit_arithmetic_reproduced'] and not r['P3_admission']
    assert not r['gates']['P1_passed'] and len(r['remaining_gaps']) == 7
    full = [c for c in r['comparisons'] if c['classification']=='full-path']
    assert all(c['saved_counter_frontier_relation']=='tie' and c['identical_selected_path'] for c in full)
    assert all(c['total_pipeline_frontier_relation']=='unknown_cost_not_ordered' for c in r['comparisons'])
    assert all(c['saved_counter_frontier_relation']=='ineligible' for c in r['comparisons'] if c not in full)


def test_independent_of_old_audit_function(evidence,monkeypatch):
    def forbidden(*args):
        raise AssertionError('must reconstruct independently')
    monkeypatch.setattr(audit.saved,'audit',forbidden)
    assert audit.audit(*evidence)['totals']['predict_samples']==1171


def test_row_order_independent(evidence):
    expected = audit.audit(*evidence)
    for i,key in ((0,'corrections'),(1,'rows'),(2,'rows'),(2,'encodings'),(3,'rows'),
                  (4,'paired_plan'),(5,'plans'),(6,'rows'),(6,'comparisons')):
        if key=='paired_plan':
            evidence[i][key]['roots'].reverse()
        else:
            evidence[i][key].reverse()
    assert audit.audit(*evidence)==expected


@pytest.mark.parametrize('i,key',[(0,'corrections'),(1,'rows'),(2,'rows'),(2,'encodings'),
                                 (3,'rows'),(5,'plans'),(6,'rows'),(6,'comparisons')])
@pytest.mark.parametrize('mutation',['missing','duplicate','foreign'])
def test_no_filter_or_ambiguous_join(evidence,i,key,mutation):
    rows = evidence[i][key]
    if mutation=='missing':
        rows.pop()
    elif mutation=='duplicate':
        rows.append(deepcopy(rows[0]))
    else:
        rows[0]['root_sha256']='0'*64
    with pytest.raises(ValueError):
        audit.audit(*evidence)


@pytest.mark.parametrize('field',['task_id','frozen_model_id','root_id','protocol_id'])
def test_wrong_predict_binding_rejected(evidence,field):
    evidence[3]['rows'][0]['prediction_receipt']['binding'][field]='foreign'
    with pytest.raises(ValueError,match='binding'):
        audit.audit(*evidence)


@pytest.mark.parametrize('phase',[0,1])
@pytest.mark.parametrize('field',['predict_calls','predict_samples','failed_predict_calls'])
def test_prediction_omission_and_double_charge(evidence,phase,field):
    evidence[3]['rows'][0]['prediction_receipt']['phases'][phase][field]+=1
    with pytest.raises(ValueError):
        audit.audit(*evidence)


@pytest.mark.parametrize('field',list(audit.BACKWARD)+['predict_samples','rollout_samples','encoder_samples'])
def test_no_counter_silent_loss(evidence,field):
    evidence[3]['rows'][0]['resources'][field]+=1
    with pytest.raises(ValueError):
        audit.audit(*evidence)


def test_empty_layer_not_counted_as_search(evidence):
    r=audit.audit(*evidence)
    adaptive=[a for a in r['arm_costs'] if a['method']==audit.METHODS[1]]
    assert [a['actual_search_horizon'] for a in adaptive]==[2,3,2,3]
    assert all(a['planning_predict_samples']==144 for a in adaptive)


@pytest.mark.parametrize('field',['goal_vectors_sha256','projection_sha256','bank','predictor_context'])
def test_task_or_action_contract_drift_downgrades(evidence,field):
    # Resealed counterexample: not a hash-corruption test. Same counts do not
    # make a changed task/action contract comparable.
    root=evidence[4]['protocol']['roots'][1]
    p=next(p for p in evidence[5]['plans'] if p['root_sha256']==root and p['method']==audit.METHODS[1])
    p[field]={'foreign':True} if field in ('bank','predictor_context') else '0'*64
    reseal(p)
    for b in evidence[2]['rows']:
        if b['root_sha256']==root and b['method']==audit.METHODS[1]:
            b['selection_sha256']=p['selection_sha256']
    for b in evidence[6]['rows']:
        if b['root_sha256']==root and b['method']==audit.METHODS[1]:
            b['selection_sha256']=p['selection_sha256']
    r=audit.audit(*evidence)
    assert r['classifications']=={'guard-partial':2,'unmatched':6}
    assert not r['gates']['all_pairs_contract_matched']


@pytest.mark.parametrize('target',['context_identity','guard'])
def test_same_counts_not_same_context_or_feedback(evidence,target):
    root=evidence[4]['protocol']['roots'][1]
    b=next(b for b in evidence[2]['rows'] if b['root_sha256']==root and b['method']==audit.METHODS[1])
    if target=='context_identity':
        b[target]['goal_rgb_sha256']='0'*64
    else:
        b[target]['scope']='changed feedback contract'
        old=next(x for x in evidence[6]['rows'] if x['root_sha256']==root and
                 x['method']==b['method'] and x['repetition']==b['repetition'])
        old['guard']=deepcopy(b['guard'])
    r=audit.audit(*evidence)
    c=next(c for c in r['comparisons'] if c['root_sha256']==root and c['repetition']==b['repetition'])
    assert c['classification']=='unmatched' and c['saved_counter_frontier_relation']=='ineligible'


def test_abort_not_promoted_to_full(evidence):
    b=next(b for b in evidence[2]['rows'] if b['guard']['abstained'])
    b['full_horizon_completed']=True
    with pytest.raises(ValueError,match='completion'):
        audit.audit(*evidence)


@pytest.mark.parametrize('field',['predict_samples','action_backward_calls'])
def test_bool_not_counter(evidence,field):
    evidence[3]['rows'][0]['resources'][field]=True
    with pytest.raises(ValueError,match='integer'):
        audit.audit(*evidence)


@pytest.mark.parametrize('mutation',['duplicate','missing'])
def test_kernel_receipts_not_filtered(evidence,mutation):
    kernels=evidence[3]['rows'][0]['search']['layers'][0]['kernels']
    if mutation=='duplicate':
        kernels.append(deepcopy(kernels[0]))
    else:
        kernels.pop()
    with pytest.raises(ValueError):
        audit.audit(*evidence)


def test_pareto_is_vector_not_scalar():
    assert audit.pareto({'p':1,'vjp':0},{'p':1,'vjp':0},[2,3],[2,3])=='tie'
    assert audit.pareto({'p':1,'vjp':0},{'p':2,'vjp':0},[2,3],[2,3])=='fixed_dominates'
    assert audit.pareto({'p':2,'vjp':0},{'p':1,'vjp':0},[2,3],[2,3])=='adaptive_dominates'
    assert audit.pareto({'p':1,'vjp':2},{'p':2,'vjp':0},[2,3],[2,3])=='tradeoff'
    assert audit.pareto({'jacobian':None},{'jacobian':None},[2,3],[2,3])=='unknown_cost_not_ordered'


def test_new_exclusive_output(tmp_path):
    (tmp_path/'results'/audit.NAME).mkdir(parents=True)
    with pytest.raises(ValueError,match='exclusive NEW'):
        audit.run(tmp_path)


def test_stdlib_only_dependency_chain():
    for module in (audit,audit.saved):
        tree=ast.parse(Path(module.__file__).read_text())
        names={n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
        names|={a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
        assert names <= {'__future__','argparse','collections','copy','json','pathlib','hashlib','math','experiments'}
