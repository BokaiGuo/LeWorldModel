"""RAM-only diagnostics; no CUDA, model, physics, or private outcome execution."""
import json
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p5_terminal_goal_attribution_20261028_v1 as m

def toks(v): return {w:np.full((3,*m.SHAPE),v+i,np.float32) for i,w in enumerate(m.WORDS)}
def native(): return {w:dict(state=[1.,2.,3.,4.,0.,5.,6.],proprio=[1.,2.,5.,6.],coverage=.1,native_reward=.1,native_done=False) for w in m.WORDS}

def test_protocol_fixed_four_and_separate_axes():
    p=m.protocol(); assert p['words']==list(m.WORDS) and p['denominator']==4
    assert p['new_api']=={'official_encoder':4,'proprio_encoder':0,'predictor':0}
    assert p['reused_goal_encoder']==1 and p['reused_predictor_forward']==0
    assert p['FLOPs'] is None and p['task_success'] is None and not p['tie_rule'].startswith('minimum')

def test_metrics_preserve_all_words_and_units():
    goal=np.zeros(m.SHAPE,np.float32); out=m.metrics(goal,toks(1),toks(2),native())
    assert [r['word'] for r in out]==list(m.WORDS) and len(out)==4
    assert all(r['predicted_goal_mse']<r['real_terminal_goal_mse'] for r in out)
    assert all(r['task_success'] is None and 'coverage' in r['native'] for r in out)
    assert len({r['model_prediction_minus_real'] for r in out})==4

@pytest.mark.parametrize('bad', ['goal_shape','goal_dtype','pred_shape','pred_dtype','nan','missing'])
def test_metrics_fail_closed(bad):
    goal=np.zeros(m.SHAPE,np.float32); pred=toks(1); real=toks(2)
    if bad=='goal_shape': goal=np.zeros((2,3),np.float32)
    elif bad=='goal_dtype': goal=goal.astype(np.float64)
    elif bad=='pred_shape': pred['xy']=pred['xy'][:2]
    elif bad=='pred_dtype': pred['xy']=pred['xy'].astype(np.float64)
    elif bad=='nan': real['xy'][0,0,0]=np.nan
    else: pred.pop('xy')
    with pytest.raises(ValueError): m.metrics(goal,pred,real,native())

def test_no_selection_or_coverage_relabel():
    rows=m.metrics(np.zeros(m.SHAPE,np.float32),toks(1),toks(1),native())
    assert len(rows)==4 and all(r['predicted_goal_mse']==r['real_terminal_goal_mse'] for r in rows)
    assert m.protocol()['native_metric'].startswith('native terminal coverage')

def test_meter_exact_four_no_refund():
    ev=[]; x=m.Meter(lambda k,v:ev.append((k,v)))
    for _ in range(4): x.before(1); x.after(1)
    with pytest.raises(ValueError): x.before(1)
    assert x.attempted==x.completed=={'official_encoder':4} and len(ev)==8

def test_output_store_rejects_existing_and_nonexclusive(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path); (tmp_path/'results').mkdir()
    p=tmp_path/'results'/f'{m.TAG}_x'; x=m.Store(p); assert p.is_dir()
    with pytest.raises(ValueError): m.Store(p)

def test_authority_constants_and_no_private_scoring():
    assert m.GOAL_RECEIPT and m.PHYS_RECEIPT_SHA and m.NUM_RECEIPT_SHA
    assert not m.protocol()['private_physics_not_planner'] is False
    assert (m.ROOT/'innerdocs'/(m.TAG+'.md')).is_file()

def test_cli_requires_all_authority_inputs(tmp_path):
    with pytest.raises(ValueError,match='all external authorities'):
        m.main(['prepare','--manifest',str(tmp_path/'x')])
    with pytest.raises(ValueError,match='actual manifest SHA'):
        m.main(['verify','--manifest',str(tmp_path/'x')])

