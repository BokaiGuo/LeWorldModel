"""Synthetic algebra only; no CUDA, physical replay or production results."""
from copy import deepcopy
from fractions import Fraction
import math

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_live_raster_analysis_20261028_v1 as m


def scores():
    # Deliberately crossing fixture rankings, not measured research values.
    vals=((1.,5.,2.,7.),(4.,2.,6.,1.),(3.,3.,3.,3.),(8.,6.,4.,2.))
    return [dict(word=w,terminal_raster=t,goal_raster=g,status='computed',full_patch_goal_mse=vals[i][j])
        for i,w in enumerate(m.WORDS) for j,(t,g) in enumerate(m.CONDITIONS)]


def coverage():
    return [dict(word=w,coverage=.1*(i+1),reward=.1,done=False,task_success=None,
        source_transition_sha256=str(i)*64,source='fixture original frame20, not frame15') for i,w in enumerate(m.WORDS)]


def exact(value):return Fraction(int(value['numerator']),int(value['denominator']))


def test_full_denominators_and_directed_symmetry():
    r=m.analyze_scores(scores(),coverage())
    assert len(r['cross_scores'])==16
    assert len(r['directed_rankings'])==48
    assert len(r['directed_reversals'])==72
    table={(x['a'],x['b'],x['terminal'],x['goal']):x for x in r['directed_rankings']}
    for (a,b,t,g),x in table.items():
        other=table[b,a,t,g]
        assert exact(x['cost_a_minus_b'])==-exact(other['cost_a_minus_b'])
        assert x['sign']==-other['sign']
    assert r['reversal_census']['comparisons']==36
    assert r['reversal_census']['independent_sample_count'] is None


def test_exact_factorial_and_simple_effects():
    r=m.factorial([Fraction(x) for x in (1,5,2,7)])
    assert exact(r['mean'])==Fraction(15,4)
    assert exact(r['terminal_main_effect'])==Fraction(3,2)
    assert exact(r['goal_main_effect'])==Fraction(9,2)
    assert exact(r['interaction'])==1
    assert exact(r['terminal_effect_goal_visible'])==1
    assert exact(r['terminal_effect_goal_white'])==2
    assert exact(r['goal_effect_terminal_visible'])==4
    assert exact(r['goal_effect_terminal_white'])==5
    assert [exact(x) for x in r['reconstructed']]==[1,5,2,7]


def test_binary64_atoms_not_rounded_effect_subtractions():
    vals=[m.fraction(x) for x in (1e16,1e16+2,1e16+4,1e16+8)]
    r=m.factorial(vals)
    assert exact(r['interaction'])==2
    assert [exact(x) for x in r['reconstructed']]==vals


def test_both_reversal_directions_preserved():
    r=m.analyze_scores(scores(),coverage())
    directed={(x['a'],x['b'],str(x['from_condition']),str(x['to_condition'])):x for x in r['directed_reversals']}
    assert any(x['direction']=='a_to_b' for x in directed.values())
    assert any(x['direction']=='b_to_a' for x in directed.values())
    for (a,b,old,new),x in directed.items():
        y=directed[b,a,old,new]
        assert x['strict_reversal']==y['strict_reversal']
        if x['direction']=='a_to_b':assert y['direction']=='b_to_a'
    assert {x['intervention_axis'] for x in directed.values()}=={'goal_only','terminal_only','both'}


def test_ties_not_strict_reversal():
    rows=scores()
    for x in rows:x['full_patch_goal_mse']=1.
    r=m.analyze_scores(rows,coverage())
    assert all(x['sign']==0 for x in r['directed_rankings'])
    assert not any(x['strict_reversal'] for x in r['directed_reversals'])
    assert r['reversal_census']['tie_involved']==36


def test_one_ulp_is_not_tie():
    rows=scores()
    for x in rows:x['full_patch_goal_mse']=1.
    rows[4]['full_patch_goal_mse']=float(np.nextafter(1.,math.inf))
    r=m.analyze_scores(rows,coverage())
    x=next(x for x in r['directed_rankings'] if (x['a'],x['b'],x['terminal'],x['goal'])==('xy','yx','visible','visible'))
    assert x['sign']==-1 and exact(x['cost_a_minus_b'])<0


@pytest.mark.parametrize('mutation',['missing','duplicate','reorder','unknown_word','unknown_color','null','nan','inf','negative','bool','status'])
def test_score_mutations_failclosed(mutation):
    rows=scores()
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows[1]=deepcopy(rows[0])
    elif mutation=='reorder':rows.reverse()
    elif mutation=='unknown_word':rows[0]['word']='selected'
    elif mutation=='unknown_color':rows[0]['terminal_raster']='new'
    elif mutation=='status':rows[0]['status']='not_computed'
    else:rows[0]['full_patch_goal_mse']={'null':None,'nan':float('nan'),'inf':math.inf,'negative':-1.,'bool':True}[mutation]
    with pytest.raises(ValueError):m.analyze_scores(rows,coverage())


@pytest.mark.parametrize('mutation',['missing','order','null','success','sha'])
def test_legacy_no_missing_to_zero(mutation):
    rows=coverage()
    if mutation=='missing':rows.pop()
    elif mutation=='order':rows.reverse()
    elif mutation=='null':rows[0]['coverage']=None
    elif mutation=='success':rows[0]['task_success']=True
    else:rows[0]['source_transition_sha256']=''
    with pytest.raises(ValueError):m.analyze_scores(scores(),rows)


def test_legacy_separate_no_same_time_metric():
    r=m.analyze_scores(scores(),coverage())
    assert r['legacy_coverage_frame']==20 and r['raster_frame']==15
    assert r['native_vs_raster_rank_agreement'] is None
    assert not r['C2_established'] and not r['scientific_admission']
    assert r['legacy_native_coverage_separate']==coverage()


def test_full_word_average_is_fixed_not_result_selected():
    r=m.analyze_scores(scores(),coverage())
    vals=m.validate_scores(scores())
    means=[sum(vals[w,t,g] for w in m.WORDS)/4 for t,g in m.CONDITIONS]
    assert r['fixed_four_word_average_factorial']==m.factorial(means)


def test_prepare_only_does_not_read_production_result(monkeypatch):
    from types import SimpleNamespace as NS
    monkeypatch.setattr(m,'encoder',lambda:NS(WORDS=m.WORDS,RASTERS=m.COLORS,SHAPE=m.SHAPE,__file__='/fixture/source'))
    monkeypatch.setattr(m,'sources',lambda:{'/fixture/own':'f'*64})
    p=m.prepare()
    assert p['actual_report'] is None and p['status']=='analysis_protocol_only_no_actual_results'
    assert p['protocol']['score_denominator']==16


def test_rational_sign_and_display_overflow_not_false_zero():
    x=Fraction(10)**500
    a=m.atom(x)
    assert a['binary64_display'] is None and int(a['numerator'])>0
    assert m.sign(x)==1 and m.sign(-x)==-1 and m.sign(Fraction())==0
