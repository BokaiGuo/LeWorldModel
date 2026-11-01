"""Pure saved-array checks and tamper counterexamples; never CUDA or physics."""
from copy import deepcopy
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import p3_joint_scheduler_saved_outcome_audit_20261027_v1 as m


@pytest.fixture(scope='module')
def bundle():return m.load_bundle(m.PROJECT)[0]


def test_actual_saved_audit(bundle):
    r=m.audit(bundle)
    assert r['status']=='saved_arithmetic_verified'
    assert r['denominator']==dict(roots=4,arms=16,scheduler_pairs=8,new_independent_roots=0)
    assert r['budget']['predict']==r['budget']['action_encoder']==1066
    assert r['budget']['search_predict_included']==960
    assert r['budget']['shared_predict_included']==40
    assert r['budget']['verification_predict_included']==66
    assert r['budget']['jacobians']==8 and r['budget']['execution_svds']==32
    assert sum(x['stratum']=='mixed_H' for x in r['pairs'])==1
    assert sum(x['stratum']=='same_H' for x in r['pairs'])==7
    assert r['new_neural_calls']==r['new_physical_steps']==0


def test_cost_is_own_Q_not_cross_metric():
    z=[2.,3.]+[0.]*190
    assert m.own_cost(z,[0.]*192,np.diag([1.,0.]+[0.]*190))==4/192


def test_uniform_formula_ignores_chi_but_dynamic_uses_fixed_rule():
    assert m.allocation('uniform',.8,.1)['blocks']=={'2':5,'3':5}
    assert m.allocation('dynamic',.8,.1)['blocks']=={'2':3,'3':7}
    assert m.allocation('dynamic',0.,0.)['blocks']=={'2':7,'3':3}


@pytest.mark.parametrize('damage',['cost','index','missing','actions','history','winner','metric','allocation'])
def test_arm_tampering_rejected(bundle,damage):
    record=bundle['protocol']['records'][0];root=bundle['report']['roots'][0]
    arm='old_Q__uniform';source=deepcopy(root['arms'][arm])
    if damage=='cost':source['candidates'][0]['cost']+=1
    elif damage=='index':source['candidates'][0]['index']=1
    elif damage=='missing':source['candidates'].pop()
    elif damage=='actions':source['candidates'][2]['actions'][0][0]+=.01
    elif damage=='history':source['candidates'][0]['context']['action_history'][0][0]+=.01
    elif damage=='winner':source['winner']=deepcopy(source['candidates'][-1])
    elif damage=='metric':source['active_metric_id']='bad'
    else:source['allocation']['blocks']['2']+=1
    with pytest.raises(ValueError):m.check_arm(record,arm,source,root['compilation'],bundle['protocol']['normalization'])


@pytest.mark.parametrize('damage',['response','residual','chi','params','radius','nominal'])
def test_response_tampering_rejected(bundle,damage):
    record=bundle['protocol']['records'][0];comp=deepcopy(bundle['report']['roots'][0]['compilation'])
    c=comp['metrics']['old_Q']['2']
    if damage=='response':c['response'][0][0]+=.01
    elif damage=='residual':c['residual'][0]+=.01
    elif damage=='chi':c['signal']['chi']+=.01
    elif damage=='params':c['solve']['params'][0]+=.01;c['proposal_logits']=c['solve']['params']
    elif damage=='radius':c['solve']['radius']+=.1
    else:comp['raw']['2']['nominal'][0]+=.1
    with pytest.raises(ValueError):m.check_compilation(record,comp,bundle['protocol']['normalization'])


@pytest.mark.parametrize('damage',['module_missing','module_role','jacobian','svd','candidate','extra'])
def test_journal_not_summary_only(bundle,damage):
    root=bundle['report']['roots'][0];record=bundle['protocol']['records'][0]
    events=deepcopy(bundle['events'][root['root_sha256']])
    if damage=='module_missing':events.pop(next(i for i,e in enumerate(events) if e['kind']=='predict'))
    elif damage=='module_role':next(e for e in events if e['kind']=='predict')['role']='old_Q__dynamic'
    elif damage=='jacobian':events.pop(next(i for i,e in enumerate(events) if e['kind']=='jacobian'))
    elif damage=='svd':events.pop(next(i for i,e in enumerate(events) if e['kind']=='svd'))
    elif damage=='candidate':next(e for e in events if e['kind']=='candidate')['index']+=1
    else:events.append(dict(kind='predict',completed=True))
    with pytest.raises(ValueError):m.check_journal(record,root,events,bundle['protocol']['arm_order'][0])


def test_duplicate_candidate_records_retained():
    bank=m.bank(0,2,3,[0.]*20)
    assert len(bank)==3 and np.array_equal(bank[0],bank[1])
    assert not np.array_equal(bank[1],bank[2])


@pytest.mark.parametrize('damage',['root_missing','root_duplicate','arm_missing','summary','inherited','validation'])
def test_census_and_resource_tampering_rejected(bundle,damage):
    b=dict(bundle,report=dict(bundle['report']))
    if damage=='root_missing':b['report']['roots']=bundle['report']['roots'][:-1]
    elif damage=='root_duplicate':b['report']['roots']=[bundle['report']['roots'][0]]*4
    elif damage=='arm_missing':
        roots=list(bundle['report']['roots']);roots[0]=dict(roots[0],arms=dict(roots[0]['arms']))
        roots[0]['arms'].pop('old_Q__dynamic');b['report']['roots']=roots
    elif damage=='summary':
        b['report']['resources']=deepcopy(bundle['report']['resources']);b['report']['resources']['attempted']['predict']+=1
    elif damage=='inherited':
        b['report']['resources']=deepcopy(bundle['report']['resources']);b['report']['resources']['inherited']['cached_input_history_cost']='free'
    else:b['report']['source_validation_calls']=dict(attempted=2,completed=1)
    with pytest.raises(ValueError):m.audit(b)


def test_pairs_never_compare_different_active_Q(bundle):
    root=bundle['report']['roots'][0];arms=dict(root['arms'])
    arms['old_Q__dynamic']=dict(arms['old_Q__dynamic'],active_metric_id='another Q')
    with pytest.raises(ValueError):m.compare(root['root_sha256'],'old_Q',arms)


def test_root_order_not_an_identity_join(bundle):
    b=dict(bundle,report=dict(bundle['report'],roots=list(reversed(bundle['report']['roots']))))
    assert m.audit(b)['budget']['predict']==1066


def test_import_is_non_neural():
    subprocess.run([sys.executable,'-c','import sys; from experiments.spectral_reachability import '+m.NAME+
        '; assert not {"torch","gymnasium","pymunk"}.intersection(sys.modules); '
        'assert "experiments.spectral_reachability.p3_joint_horizon_goal_metric_scheduler_20261027_v1" not in sys.modules'],check=True)
