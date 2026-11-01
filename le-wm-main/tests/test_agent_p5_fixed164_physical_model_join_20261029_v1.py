import copy
import hashlib
import pytest
from experiments.spectral_reachability import agent_p5_fixed164_physical_model_join_20261029_v1 as m


@pytest.fixture(scope='module')
def fixture():
 c=dict(schema='agent_p5_short_start_comparison_20261029_v1.analysis',status='all164_and80_common_H2_label_saved_rescored',evidence=dict(new_complete_strict_sha256=m.FILES['short_strict'][1],new_report_sha256='report'),bindings={str(m.ROOT/m.FILES['physical'][0]):m.FILES['physical'][1]},rows=[],pairs=[])
 s=dict(schema='agent_p5_short_start_full_roll_20261029_v1.consume',status='all164_short_start_causal_roll_saved_verified',branches=164,pairs=80,report_sha256='report')
 p=dict(schema='agent_observed_raw_coordinate_physical_20261028_v1.consume',status='observed_raw164_physical_recorded_not_admitted',fixed_denominator=164,known_roots=2,fresh_roots=0,physical_report_sha256='physical_report',rows=[])
 r=dict(schema='agent_observed_raw_coordinate_rgb_20261028_v1.consume',status='raw164_RGB_verified_not_admitted',fixed_denominator=164,known_roots=2,fresh_roots=0,physical_input_sha256=dict(physical_receipt_sha=m.FILES['physical'][1],physical_report_sha='physical_report'),rows=[])
 for i in range(164):
  panel=i//41; ident=dict(index=i,root_slot=panel//2,condition=('off','on')[panel%2],task_id='task'+str(panel),selection_digest='selection'+str(i))
  commands=[[0.1,0.0]]*10; h=m.rawhash(commands); cp=dict(step=10,file_sha256='png'+str(i),pixel_sha256='pixels'+str(i),state=[i]*7); metrics=dict(native_success=False)
  c['rows'].append({**ident,'common_MSE':{'short_20':0.5}})
  p['rows'].append({**ident,'plan':dict(index=i,commands=commands,raw_float32_sha256=h,producer_job_sha256='job'+str(i),model_task_binding={'task':panel}), 'initial_exact':True,'status':'horizon_completed','checkpoints':[cp],'final_metrics':metrics,'final_state':[i]*7,'native_success':False})
  r['rows'].append({**ident,'raw_float32_sha256':h,'producer_job_sha256':'job'+str(i),'model_task_binding':{'task':panel},'prefix_reference_exact':True,'physical_status':'horizon_completed','latent_missing':False,'source':copy.deepcopy(cp),'evaluation':dict(final_metrics=copy.deepcopy(metrics),final_state=[i]*7,native_success=False),'actual_Q':0.1})
 for i in range(80):
  panel=i//20; j=i%20; minus=41*panel+1+2*j
  c['pairs'].append(dict(slot=panel,minus_index=minus,plus_index=minus+1,coord=j%10,h=(.01,.001)[j//10],root_slot=panel//2,condition=('off','on')[panel%2]))
 return dict(comparison=c,short_strict=s,physical=p,rgb=r)


def test_full_fixed_denominator_and_baselines(fixture):
 x=m.analyze(fixture)
 assert len(x['rows'])==164 and len(x['pairs'])==80 and x['baseline_indices']==[0,41,82,123]
 assert x['known_roots']==2 and x['fresh_roots']==0 and x['P5_admission'] is False


@pytest.mark.parametrize('mutation',[
 lambda x:x['rgb']['rows'][2].update(task_id='wrong'),
 lambda x:x['rgb']['rows'][2].update(selection_digest='wrong'),
 lambda x:x['rgb']['rows'][2].update(raw_float32_sha256='wrong'),
 lambda x:x['physical']['rows'][2]['plan']['commands'][0].__setitem__(0,0.2),
 lambda x:x['rgb']['rows'][2].update(producer_job_sha256='wrong'),
 lambda x:x['rgb']['rows'][2]['source'].update(file_sha256='wrong'),
 lambda x:x['rgb']['rows'][2]['evaluation'].update(native_success=True),
 lambda x:x['rgb']['rows'][2].update(latent_missing=True),
 lambda x:x['comparison']['pairs'][0].update(coord=2),
 lambda x:x['comparison']['pairs'][0].update(h=.02),
 lambda x:x['comparison']['rows'].pop(),
 lambda x:x['physical'].update(fresh_roots=1),
 lambda x:x['short_strict'].update(report_sha256='wrong'),
 lambda x:x['comparison']['rows'][1]['common_MSE'].update(short_20=float('nan')),
])
def test_mismatch_is_fail_closed(fixture,mutation):
 x=copy.deepcopy(fixture); mutation(x)
 with pytest.raises((ValueError,KeyError)): m.analyze(x)


def test_duplicate_and_nonfinite_refused():
 with pytest.raises(ValueError): m.unique([('x',1),('x',2)])
 with pytest.raises(ValueError): m.rawhash([[float('inf'),0]]*10)


def test_cli_help(capsys,monkeypatch):
 import sys
 monkeypatch.setattr(sys,'argv',['p5-join','--help'])
 with pytest.raises(SystemExit) as e: m.main()
 assert e.value.code==0 and 'analyze' in capsys.readouterr().out


def test_verify_rebuild_and_mutation(fixture,tmp_path,monkeypatch,capsys):
 import sys
 data=m.canonical(m.analyze(fixture)); path=tmp_path/'analysis.json'; path.write_bytes(data)
 digest=hashlib.sha256(data).hexdigest(); monkeypatch.setattr(m,'load',lambda:fixture)
 monkeypatch.setattr(sys,'argv',['join','verify','--analysis',str(path),'--sha256',digest]); m.main()
 assert 'exact_saved_join_verified' in capsys.readouterr().out
 path.write_bytes(data.replace(b'"fresh_roots":0',b'"fresh_roots":1'))
 monkeypatch.setattr(sys,'argv',['join','verify','--analysis',str(path),'--sha256',hashlib.sha256(path.read_bytes()).hexdigest()])
 with pytest.raises(ValueError): m.main()
