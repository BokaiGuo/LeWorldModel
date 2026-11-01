"""RAM-only saved arithmetic tests; no model, CUDA, environment, or physics."""
from copy import deepcopy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_teacher_library40_rank_error_20261028_v1 as m


def row(i,actual=True):
    task,rem=divmod(i,10);ri,li=divmod(rem,5);slot,cond=((0,'off'),(0,'on'),(1,'off'),(1,'on'))[task]
    goal=[0.0]*192;pred=[float(i+1)]*192
    r=dict(index=i,root_slot=slot,condition=cond,role=('Z','S')[ri],library_index=li,library_label=m.rgb.teacher.LIBRARY[li],
        job_digest=f'j{i}',anchor_id=f'a{task}{ri}',task_id=f't{task}',blind_sha256=f'b{task}',plan_digest=f'p{i}',goal192=goal,goal_binding=dict(x=1),
        suffix_raw_bits=(np.zeros((5,2),'<f4')+li).tobytes().hex(),predicted_Q=float((i+1)**2),actual_Q=float(i*i) if actual else None,
        predicted192=pred,native_success=False,missing_reason=None if actual else 'missing',actual_model_error=None,cross_term=0.,quadratic_term=0.,identity_residual=0.)
    return r


def test_fixed40_order_and_missing12():
    rows=[row(i,i!=12) for i in range(40)]
    a=m.assemble(rows)
    # The five-word rank vector is indivisible: one missing member makes all
    # five rank errors in that fixed group unknown; the slot denominator stays40.
    assert len(a['rows'])==40 and a['rank_rows']==dict(fixed=40,known=35,unknown=5)
    assert all(a['rows'][i]['rank_status']=='unknown_incomplete_fixed_library' for i in range(10,15))
    assert a['within_role_summary']['fixed_pair_denominator']==80
    assert a['within_role_summary']['available_pairs']==76
    assert a['within_role_summary']['unknown_pairs']==4
    assert a['same_word_ZS_summary']['fixed_pair_denominator']==20
    assert a['same_word_ZS_summary']['available_pairs']==19
    assert a['same_word_ZS_summary']['unknown_pairs']==1


def test_exact_zero_tie_is_not_reversal():
    a=row(0);b=row(1);a['predicted_Q']=b['predicted_Q']=2.;a['actual_Q']=b['actual_Q']=3.
    p=m.pair(a,b,'within_role_library')
    assert p['classification']=='both_tie' and p['reversal'] is False and p['sign_agreement'] is True
    assert m.pair_summary([p])['observed_reversals']==0


def test_discordant_is_attribution_only_not_selection():
    a=row(0);b=row(1);a['predicted_Q']=1.;b['predicted_Q']=4.;a['actual_Q']=9.;b['actual_Q']=2.
    p=m.pair(a,b,'within_role_library')
    assert p['classification']=='discordant' and p['reversal'] is True
    assert 'winner' not in p and 'winner' not in m.pair_summary([p])


@pytest.mark.parametrize('kind', ['cross_task','cross_condition','cross_goal','cross_role_word','cross_root','bad_suffix'])
def test_illegal_matching_rejected(kind):
    a=row(0);b=row(1)
    if kind=='cross_task':b['task_id']='other';b['root_slot']=1
    elif kind=='cross_condition':b['condition']='on'
    elif kind=='cross_goal':b['goal192'][0]=1
    elif kind=='cross_role_word':b['role']='S';b['library_index']=1
    elif kind=='cross_root':b['root_slot']=1
    elif kind=='bad_suffix':b['suffix_raw_bits']='different'
    with pytest.raises(ValueError):m.matching(a,b,'same_word_ZS' if kind in ('cross_role_word','bad_suffix') else 'within_role_library')


def test_unknown_pair_does_not_become_failure_or_zero():
    a=row(12,False);b=row(13,True)
    p=m.pair(a,b,'within_role_library')
    assert p['available'] is False and p['actual_delta'] is None and p['error_delta'] is None
    assert p['classification']=='unknown_missing_terminal' and p['missing_indices']==[12]


def test_exact_q_decomposition_and_residual():
    g=np.linspace(-1,1,192);a=g+.25;p=a+.5
    terms=m.scalar_terms(p,a,g)
    assert terms['actual_Q']==pytest.approx(.0625)
    assert terms['latent_MSE']==pytest.approx(.25)
    assert terms['cross_term']==pytest.approx(.25)
    assert terms['Q_error']==pytest.approx(.5)
    assert terms['identity_residual']==pytest.approx(0.)
    missing=m.scalar_terms(p,None,g)
    assert missing['actual_Q'] is None and missing['Q_error'] is None
    with pytest.raises(ValueError):m.scalar_terms(p[:191],a,g)


def test_ranks_are_midranked_and_not_reordered():
    assert m.midranks([3.,1.,1.,2.])==[4,1.5,1.5,3]
    assert m.midranks([1.,None,3.])==[None,None,None]


def test_fixed_summary_unknown_full_rate_not_claimed():
    p=m.pair(row(0),row(1),'within_role_library')
    s=m.pair_summary([p])
    assert s['observed_rate_denominator']==1 and s['observed_rate_is_full_fixed_library'] is True
    rows=[row(i,i!=12) for i in range(40)];a=m.assemble(rows)
    s=a['within_role_summary']
    assert s['observed_rate_is_full_fixed_library'] is False and s['full_fixed_library_reversal_rate'] is None
    assert s['full_rate_unknown_reason']=='missing_terminal'


def test_no_new_execution_symbols_in_contract():
    rows=[row(i,i!=12) for i in range(40)];a=m.assemble(rows)
    assert a['rank_rows']['fixed']==40
    assert all(x is None or isinstance(x,(int,float)) for x in [a['within_role_summary']['observed_reversal_rate']])
    assert m.NAME.endswith('_v1')
