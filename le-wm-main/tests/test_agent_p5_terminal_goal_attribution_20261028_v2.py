"""RAM-only v2 contract tests; no model, CUDA, physics, or network."""
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p5_terminal_goal_attribution_20261028_v2 as m

def toks(v): return {w:np.full((3,*m.SHAPE),v+i,np.float32) for i,w in enumerate(m.WORDS)}
def native(): return {w:dict(state=[1.,2.,3.,4.,0.,5.,6.],proprio=[1.,2.,5.,6.]) for w in m.WORDS}

def test_protocol_v2_closes_four_gaps():
    p=m.protocol(); assert p['schema']==m.TAG and p['denominator']==4
    assert p['new_api']=={'official_encoder':4,'proprio_encoder':0,'predictor':0}
    assert p['goal_reencoded'] is False and p['predictor_rerun'] is False
    assert p['FLOPs'] is None and p['task_success'] is None

def test_metrics_full_tokens_and_native_separate():
    goal=np.zeros(m.SHAPE,np.float32); r=m.metric(goal,toks(1),toks(2),native())
    assert [x['word'] for x in r]==list(m.WORDS) and all('native' in x for x in r)
    assert all(x['task_success'] is None for x in r)

@pytest.mark.parametrize('bad',['goal','missing','shape','dtype','nan','realnan'])
def test_metric_fail_closed(bad):
    goal=np.zeros(m.SHAPE,np.float32); p=toks(1); r=toks(2)
    if bad=='goal': goal=np.zeros((2,3),np.float32)
    elif bad=='missing': p.pop('xy')
    elif bad=='shape': p['xy']=p['xy'][:2]
    elif bad=='dtype': p['xy']=p['xy'].astype(np.float64)
    elif bad=='nan': goal[0,0]=np.nan
    else: r['xy'][0,0,0]=np.nan
    with pytest.raises(ValueError): m.metric(goal,p,r,native())

def test_meter_caps_and_failed_attempt_no_refund():
    events=[]; x=m.Meter(lambda k,v:events.append((k,v)))
    for _ in range(4): x.before(); x.after()
    with pytest.raises(ValueError): x.before()
    assert x.a==x.c==4 and len(events)==8

def test_strict_preflight_rejects_missing_authorities(tmp_path):
    with pytest.raises((ValueError,FileNotFoundError)):
        m.prepare(*([tmp_path/'missing']*9))

def test_verify_rejects_tampered_manifest():
    with pytest.raises((ValueError,KeyError)):
        m.verify({'schema':m.TAG,'status':'prepared_strict_saved_join','protocol':m.protocol(),'bindings':{},'authority':{}})

def test_cli_requires_nine_paths(tmp_path):
    with pytest.raises(ValueError,match='nine authority'):
        m.main(['prepare','--manifest',str(tmp_path/'m')])

def test_v1_untouched_and_new_only():
    assert m.sha(m.ROOT/'experiments/spectral_reachability/agent_p5_terminal_goal_attribution_20261028_v1.py')==m.V1_SHA
    assert (m.ROOT/'innerdocs'/(m.TAG+'.md')).is_file()

def test_output_is_not_a_science_gate():
    assert m.protocol()['scientific_admission'] is False
    assert m.protocol()['C2_established'] is False
    assert m.protocol()['physics']==0

def test_consume_contract_mentions_model_and_import_closure():
    text=(m.ROOT/'experiments/spectral_reachability'/(m.TAG+'.py')).read_text()
    assert 'model_before' in text and 'model_after' in text and 'imported source closure' in text
