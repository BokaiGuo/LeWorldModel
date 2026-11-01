import copy
import hashlib
import pytest
from experiments.spectral_reachability import agent_p5_fixed80_error_decomposition_20261029_v1 as m

@pytest.fixture
def fixture():
 rows=[]; pairs=[]
 for i in range(164):
  panel=i//41
  rows.append(dict(index=i,root_slot=panel//2,condition=('off','on')[panel%2],task_id='task'+str(panel),lewm_actual_goal_Q=float(i),native_final_metrics={k:float(-i) for k in m.METRICS},jepa_common_label_MSE={k:float(i+j) for j,k in enumerate(m.JEPA)}))
 for i in range(80):
  slot,j=divmod(i,20); minus=41*slot+1+2*j
  pairs.append(dict(slot=slot,coord=j%10,h=(.01,.001)[j//10],minus_index=minus,plus_index=minus+1))
 a=dict(schema='agent_p5_fixed164_physical_model_join_20261029_v1.analysis',status='fixed164_saved_identity_join_only',known_roots=2,fresh_roots=0,baseline_indices=[0,41,82,123],P1_admission=False,P5_admission=False,rows=rows,pairs=pairs)
 return dict(schema='agent_p5_fixed164_physical_model_persist_20261029_v1.result',denominator=dict(m.EXPECTED),cross_model_spaces_comparable=False,P1_admission=False,P5_admission=False,full_strict_rerun=False,source_closure={'fixture':'sha'},analysis=a)

def test_fixed_denominator_directions_and_same_space(fixture):
 a=m.analyze(fixture); assert len(a['pairs'])==80 and a['known_roots']==2
 p=a['pairs'][0]; assert p['lewm_actual_Q_plus_minus']==1 and p['native_plus_minus']['native_distance']==-1
 assert p['direction_agreement']['native_distance']=='opposite'
 assert p['jepa_within_branch_error_contrasts']['short_20_minus_H1_20']==dict(minus=2,plus=2)
 assert a['direction_summary']['native_distance']['all']['counts']['opposite']==80

def test_zero_and_missing_never_imputed(fixture):
 a=fixture['analysis']['rows']; a[1]['native_final_metrics']['native_distance']=None; a[2]['jepa_common_label_MSE']['H1_20']=None
 a[2]['lewm_actual_goal_Q']=a[1]['lewm_actual_goal_Q']
 p=m.analyze(fixture)['pairs'][0]
 assert p['native_plus_minus']['native_distance'] is None and p['direction_agreement']['native_distance']=='missing'
 assert p['jepa_MSE_plus_minus']['H1_20'] is None and p['jepa_within_branch_error_contrasts']['short_20_minus_H1_20']['plus'] is None
 assert p['lewm_actual_Q_plus_minus']==0 and p['direction_agreement']['object_position_error']=='one_zero'
 assert m.agreement('zero','zero')=='both_zero'

@pytest.mark.parametrize('mutation',[
 lambda x:x['analysis']['pairs'].pop(), lambda x:x['analysis']['rows'].pop(),
 lambda x:x['analysis']['pairs'][0].update(plus_index=4), lambda x:x['analysis']['pairs'][0].update(h=.02),
 lambda x:x['analysis']['rows'][2].update(task_id='wrong'), lambda x:x['analysis']['rows'][1].update(lewm_actual_goal_Q=float('nan')),
 lambda x:x['analysis']['rows'][1]['native_final_metrics'].pop('native_distance'),
 lambda x:x['analysis']['rows'][1]['jepa_common_label_MSE'].pop('H1_20'), lambda x:x.update(cross_model_spaces_comparable=True),
])
def test_fail_closed(fixture,mutation):
 mutation(fixture)
 with pytest.raises((ValueError,KeyError)): m.analyze(fixture)

def test_exclusive_and_exact_verify(tmp_path,fixture,monkeypatch):
 import sys
 monkeypatch.setattr(m,'build',lambda:m.analyze(fixture)); monkeypatch.setattr(m,'target',lambda p: __import__('pathlib').Path(p))
 p=tmp_path/'a.json'; data=m.canonical(m.analyze(fixture)); m.write(p,data)
 with pytest.raises(FileExistsError): m.write(p,data)
 sha=hashlib.sha256(data).hexdigest(); monkeypatch.setattr(sys,'argv',['x','verify','--analysis',str(p),'--sha256',sha]); m.main()
 p.write_bytes(data.replace(b'"known_roots":2',b'"known_roots":3')); sha=hashlib.sha256(p.read_bytes()).hexdigest()
 monkeypatch.setattr(sys,'argv',['x','verify','--analysis',str(p),'--sha256',sha])
 with pytest.raises(ValueError): m.main()
