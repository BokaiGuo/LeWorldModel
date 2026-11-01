"""Synthetic algebra only; no CUDA, physical replay or production results."""
from copy import deepcopy
from fractions import Fraction
import math
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_live_raster_analysis_20261028_v3 as m


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


@pytest.fixture
def saved_ram(monkeypatch):
    """Synthetic arrays / mocked authority only; no production output or model."""
    memory={};shape=m.SHAPE;keys=tuple(s+'__'+c for s in (*m.WORDS,'goal') for c in m.COLORS)
    arrays={k:np.full(shape,(i+1)/32,np.float32) for i,k in enumerate(keys)}
    records={}
    for k,a in arrays.items():
        buf=io.BytesIO();np.save(buf,a,allow_pickle=False);b=buf.getvalue();path='/ram-encoder/'+k+'_patch.npy'
        memory[path]=b;records[k]=dict(path=path,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),shape=list(shape),dtype='float32')
    computed=[]
    for w in m.WORDS:
        for t,g in m.CONDITIONS:
            d=arrays[w+'__'+t].astype(np.float64)-arrays['goal__'+g].astype(np.float64)
            computed.append(dict(word=w,terminal_raster=t,goal_raster=g,status='computed',full_patch_goal_mse=float(np.mean(d**2))))
    receipt=dict(cross_condition=computed,legacy_native_coverage=coverage(),fixed_within_state_raster_difference=[])
    report=dict(token_artifacts=records,cross_condition=computed)
    native=dict(targets=[dict(state_id=w,frame=15) for w in m.WORDS]+[dict(state_id='goal',frame=0)],old_failure=dict(sha256='f'*64))
    def add(path,x):memory[path]=m.canonical(x)+b'\n';return hashlib.sha256(memory[path]).hexdigest()
    native_sha=add('/ram-raster/prepare.json',native)
    ep=dict(render_authority=dict(render_manifest='/ram-raster/prepare.json',render_manifest_sha256=native_sha,
        render_report='/ram-raster/report.json'),render_receipt=dict(status='strict_live_raster_complete_v2',
        baseline_to_current_anchor=[dict(state_id=s,fixture=True) for s in (*m.WORDS,'goal')],
        source_comparisons=[dict(fixture_index=i) for i in range(65)]))
    add('/ram-encoder/prepare.json',ep);add('/ram-encoder/report.json',report);add('/ram-encoder/receipt.json',receipt)
    pp=dict(bindings={'fixture':'not_research'});add('/ram-analysis/prepare.json',pp)
    oldread,oldstat,oldload=Path.read_bytes,Path.stat,np.load
    monkeypatch.setattr(Path,'read_bytes',lambda p:memory[str(p)] if str(p) in memory else oldread(p))
    monkeypatch.setattr(Path,'stat',lambda p,*a,**k:NS(st_size=len(memory[str(p)]),st_mode=0o100644) if str(p) in memory else oldstat(p,*a,**k))
    monkeypatch.setattr(m,'sha',lambda p:hashlib.sha256(memory[str(p)]).hexdigest())
    monkeypatch.setattr(np,'load',lambda p,*a,**k:oldload(io.BytesIO(memory[str(p)]),*a,**k))
    monkeypatch.setattr(m,'verify',lambda *args:pp)
    monkeypatch.setattr(m,'encoder',lambda:NS(KEYS=keys,consume=lambda *args:receipt))
    return memory,report,receipt,ep,pp,add


def call_saved():
    return m.analyze('/ram-analysis/prepare.json',m.sha('/ram-analysis/prepare.json'),
        '/ram-encoder/prepare.json',m.sha('/ram-encoder/prepare.json'),
        '/ram-encoder/report.json',m.sha('/ram-encoder/report.json'),
        '/ram-encoder/receipt.json',m.sha('/ram-encoder/receipt.json'))


def test_complete_saved_algebra_pipeline_no_encoder(saved_ram):
    memory,report,receipt,ep,pp,add=saved_ram
    result=call_saved()
    assert result['status']=='all16_saved_raster_algebra_recorded'
    assert len(result['analysis']['directed_reversals'])==72
    assert len(result['raster_time_diagnostics']['source_comparisons'])==65
    assert result['raster_time_diagnostics']['baseline_to_current_anchor']==ep['render_receipt']['baseline_to_current_anchor']
    add('/ram-analysis/report.json',result)
    assert m.consume('/ram-analysis/report.json',m.sha('/ram-analysis/report.json'))['status']=='saved_algebra_verified'


@pytest.mark.parametrize('mutation',['score','receipt','array','frame','v1_receipt','time_missing','diff_missing'])
def test_saved_binding_mutations_rejected(saved_ram,mutation):
    memory,report,receipt,ep,pp,add=saved_ram
    if mutation=='score':
        report['cross_condition'][0]['full_patch_goal_mse']+=.01;add('/ram-encoder/report.json',report)
        # Mock authority shares scores: independent array arithmetic must reject.
    elif mutation=='receipt':
        wrong=deepcopy(receipt);wrong['cross_condition'].pop();add('/ram-encoder/receipt.json',wrong)
    elif mutation=='array':memory['/ram-encoder/xy__visible_patch.npy']+=b'wrong'
    elif mutation=='frame':
        native=json.loads(memory['/ram-raster/prepare.json']);native['targets'][0]['frame']=20
        ep['render_authority']['render_manifest_sha256']=add('/ram-raster/prepare.json',native);add('/ram-encoder/prepare.json',ep)
    else:
        if mutation=='v1_receipt':ep['render_receipt']['status']='strict_live_raster_complete'
        elif mutation=='time_missing':ep['render_receipt']['baseline_to_current_anchor'].pop()
        else:ep['render_receipt']['source_comparisons'].pop()
        add('/ram-encoder/prepare.json',ep)
    with pytest.raises(ValueError):call_saved()


def test_v3_preserves_all_predeclared_function_asts():
    import ast
    root=Path(m.__file__).parent
    old=ast.parse((root/'agent_p5_live_raster_analysis_20261028_v2.py').read_text())
    new=ast.parse(Path(m.__file__).read_text())
    old_functions={n.name:ast.dump(n,include_attributes=False) for n in old.body if isinstance(n,ast.FunctionDef)}
    new_functions={n.name:ast.dump(n,include_attributes=False) for n in new.body if isinstance(n,ast.FunctionDef)}
    assert old_functions==new_functions
    changes=[]
    assert len(old.body)==len(new.body)
    for a,b in zip(old.body,new.body):
        if ast.dump(a)!=ast.dump(b):
            assert isinstance(a,ast.Assign) and isinstance(b,ast.Assign)
            assert len(a.targets)==len(b.targets)==1
            assert isinstance(a.targets[0],ast.Name) and a.targets[0].id==b.targets[0].id
            changes.append(a.targets[0].id)
    assert changes==['NAME','ENCODER_SHA']


def test_final_encoder_authority_not_stale_source():
    assert m.ENCODER=='main_p5_live_raster_encoder_20261028_v2'
    assert m.ENCODER_SHA=='4f3b1d42a56097e56863c85bc4fe47a0c22e7d8b242b9bc252b4967a70a57a7e'
    assert m.protocol()['score_denominator']==16
