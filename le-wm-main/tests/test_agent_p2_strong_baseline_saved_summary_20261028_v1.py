"""Saved/RAM-only baseline summary tests."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_p2_strong_baseline_saved_summary_20261028_v1 as m

def test_real_saved_inputs_are_sealed_and_protocols_are_separate():
    x=m.build(); assert x['status'].startswith('saved_'); assert set(x['local'])=={'kinodynamic','no_composition','sampling'}
    assert all(v['status']=='protocol_mismatch_no_direct_comparison' for v in x['comparisons'].values())
    assert x['author']['status']=='unavailable'; assert x['P1_admission'] is False

def test_real_denominators_and_local_values_are_not_paper_values():
    x=m.build(); assert x['denominators']=={'kinodynamic':2,'no_composition':2,'sampling':4}
    assert all('local_values' in x['local'][k] for k in x['local'])
    assert x['author']['values'] is None

@pytest.mark.parametrize('damage',['status','schema','protocol','errors','admission','row'])
def test_saved_summary_mutation_rejected(tmp_path,damage):
    x=m.build();
    if damage=='status': x['status']='bad'
    if damage=='schema': x['schema']='other'
    if damage=='protocol': x['local']['sampling']['protocol']['history']='fake'
    if damage=='errors': x['local']['kinodynamic']['raw_keys'].append('errors')
    if damage=='admission': x['P1_admission']=True
    if damage=='row': x['local']['no_composition']['local_values'].pop()
    path=tmp_path/'summary.json'; m.write(path,x)
    with pytest.raises(ValueError): m.verify(path,m.sha(path))

def test_author_values_always_separate_and_optional(tmp_path):
    author=tmp_path/'author.json'; author.write_text(json.dumps({'kinodynamic':{'reported':0.1}}))
    x=m.build(author); assert x['author']['values']['kinodynamic']['reported']==.1
    assert all('author' not in x['local'][k] for k in x['local'])

def test_protocol_mismatch_never_returns_delta():
    a={'protocol':{'horizon':2}}; b={'protocol':{'horizon':10}}
    x=m.compare(a,b); assert x['status'].endswith('no_direct_comparison') and x['delta'] is None

def test_report_errors_and_unknown_values_fail_closed(tmp_path,monkeypatch):
    original=m.LOCAL['sampling']; report=deepcopy(m.load_local()['sampling']['report']); report['errors']=['unknown']
    p=tmp_path/'report.json'; p.write_text(json.dumps(report)); monkeypatch.setitem(m.LOCAL,'sampling',(p,m.sha(p)))
    with pytest.raises(ValueError): m.load_local()

def test_cold_import_does_not_import_neural_runtime():
    import os,subprocess,sys
    code="""import sys
class G:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','gym','gymnasium','pymunk','stable_worldmodel'}: raise AssertionError(fullname)
sys.meta_path.insert(0,G())
from experiments.spectral_reachability import agent_p2_strong_baseline_saved_summary_20261028_v1 as m
assert m.compare({'protocol':{}},{'protocol':{}})['status'].endswith('allowed')
"""
    p=subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert p.returncode==0,p.stdout+p.stderr
