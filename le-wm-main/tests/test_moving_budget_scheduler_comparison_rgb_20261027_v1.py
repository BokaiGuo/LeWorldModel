"""Only RAM arrays/JSON mocks; no actual build, prepare, neural or simulator."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import moving_budget_scheduler_comparison_rgb_20261027_v1 as m
from test_moving_budget_scheduler_rgb_cuda_20261027_v1 import fixture as physical_fixture


def fixture():
    physical,events,p,manifest,tasks,selected=physical_fixture()
    images=m.rgb.assemble(physical,events,p,manifest,tasks,selected)
    report=dict(schema=m.rgb.NAME+'.report',status='moving_budget_scheduler_rgb_encoded_not_admitted',errors=[],contract=m.rgb.contract(),
        fixed_branch_denominator=8,fixed_image_denominator=16,P1_admission=False,P2_admission=False,P3_admission=False,training=False,
        resources={k:dict(attempted=16,completed=16) for k in ('encoder','projector')},rows=[],prepared='/RAM/rgb_prepare.json',prepare_sha256='b'*64,
        inputs=dict(physical_report='/RAM/physical/report.json',physical_sha='c'*64,physical_prepare_sha='d'*64,physical_source_sha=m.rgb.PHYSICAL_SOURCE_SHA),
        artifact_sha256={})
    receipt=dict(schema=m.rgb.NAME+'.consume',status='scheduler_rgb_verified_not_admitted',errors=[],rows=[],report_sha256='e'*64,
        fixed_branch_denominator=8,fixed_image_denominator=16,prediction_status='unknown_not_recomputed',P1_admission=False,P2_admission=False,P3_admission=False)
    for item in images:
        met=m.rgb.compare(item,np.full(192,item['index']/10,np.float32));met.update(rawpixel_vs_prefix5_mse=0.,actual_Q_change_from_prefix5=0.,rawpixel_mean=0.)
        if item['role']=='terminal10':met['actual_Q_change_from_prefix5']=met['actual_Q']-report['rows'][-1]['metrics']['actual_Q']
        row={k:item[k] for k in ('index','branch','root_slot','source_slot','arm','role','task_id','task_export','task_sha256','blind_sha256')}
        row.update(status='encoded',provenance=deepcopy(item),metrics=met,rawpixel=dict(file='mock.png'),latent=dict(file='mock.npy'))
        report['rows'].append(row)
        own={k:item[k] for k in ('root_slot','arm','role','selection_digest','task_id','root_sha256','compiler_report_sha256')}
        own.update({k:deepcopy(row[k]) for k in ('metrics','rawpixel','latent')});receipt['rows'].append(own)
    base=dict(schema=m.comparison.NAME+'.report',status='saved_comparison_assembled_not_admitted',errors=[],fixed_branch_denominator=8,
        resources=dict(scheduler_actual=dict(predict=520)),RGB_status=deepcopy(m.comparison.MISSING_RGB),rows=[],**m.comparison.FLAGS)
    base['inputs']=dict(physical_output='/RAM/physical',physical_report_sha256='c'*64,physical_prepare_sha256='d'*64,
        physical_source_sha256=m.rgb.PHYSICAL_SOURCE_SHA,scheduler_output='/RAM/scheduler',scheduler_report_sha256='c'*64,bindings_sha256={})
    for b in physical['rows']:
        plan=b['selected_plan']
        base['rows'].append(dict(source_slot=b['root_slot'],arm=b['arm'],root_sha256=plan['root_sha256'],task_id=plan['task_id'],blind_sha256=plan['blind_sha256'],
            full_horizon=True,native_steps=10,truncated=False,physical_selection_digest=b['selection_digest'],model_selection_sha256=plan['compiler_selection_sha256'],
            checkpoint_references=deepcopy(b['checkpoints']),raw_initial_state=b['root_state'],raw_goal=b['goal'],model_Q=.1+b['branch_index']/100,
            actual_RGB_sameQ=deepcopy(m.comparison.MISSING_RGB)))
    return base,report,receipt


def test_complete8_16_join_preserves_old_report_and_predictions_unknown():
    args=fixture();before=deepcopy(args);result=m.assemble(*args)
    assert args==before and result['original_comparison']==before[0]
    assert len(result['rows'])==result['fixed_branch_denominator']==8 and result['fixed_image_denominator']==16
    for row in result['rows']:
        assert row['predicted_native192'] is row['predicted_prefix_Q'] is None
        assert row['prefix5']['metrics']['predicted_Q'] is row['terminal10']['metrics']['predicted_Q'] is None
        assert row['comparison_row']['actual_RGB_sameQ']['status']=='missing_not_encoded'
        assert row['actual_terminal_minus_original_model_Q']==row['terminal10']['metrics']['actual_Q']-row['original_planner_terminal_Q']
    assert all(result[k] is False for k in m.comparison.FLAGS)
    assert result['resources']['new_neural_calls']==result['resources']['new_physics_calls']==0


def test_identity_pairing_not_input_order():
    a=fixture();expected=m.assemble(*a)
    for rows in (a[0]['rows'],a[1]['rows'],a[2]['rows']):rows.reverse()
    actual=m.assemble(*a)
    assert actual['rows']==expected['rows'] and actual['D_minus_reference']==expected['D_minus_reference']


@pytest.mark.parametrize('bad',['base_schema','rgb_schema','receipt_schema','partial','missing','duplicate','arm','bool','root','task','blind',
    'selection','modelselection','compiler','physical','prepare','source','path','ownmetric','checkpoint','rootstate','rootQ','Qchange',
    'Qname','prediction','negative','nan','claim','receiptclaim','modelSHA','ownroot'])
def test_strict_join_fail_closed(bad):
    b,r,o=fixture();p=r['rows'][0]['provenance']
    if bad=='base_schema':b['schema']='old'
    elif bad=='rgb_schema':r['schema']='old'
    elif bad=='receipt_schema':o['schema']='old'
    elif bad=='partial':b['rows'][0]['full_horizon']=False
    elif bad=='missing':r['rows'].pop()
    elif bad=='duplicate':r['rows'][1]=deepcopy(r['rows'][0])
    elif bad=='arm':r['rows'][0]['arm']='oldGN'
    elif bad=='bool':b['rows'][0]['source_slot']=False
    elif bad in ('root','task','blind'):p[dict(root='root_sha256',task='task_id',blind='blind_sha256')[bad]]='ab'*32
    elif bad=='selection':p['selection_digest']='0'*64
    elif bad=='modelselection':p['compiler_selection_sha256']='0'*64
    elif bad=='compiler':p['compiler_authority']='other'
    elif bad in ('physical','prepare','source','path'):r['inputs'][dict(physical='physical_sha',prepare='physical_prepare_sha',source='physical_source_sha',path='physical_report')[bad]]='0'*64
    elif bad=='ownmetric':o['rows'][0]['metrics']['actual_Q']+=1
    elif bad=='checkpoint':p['checkpoint']['pixel_sha256']='0'*64
    elif bad=='rootstate':b['rows'][0]['raw_initial_state'][0]+=1
    elif bad=='rootQ':r['rows'][0]['metrics']['root_Q']+=1;o['rows'][0]['metrics']=deepcopy(r['rows'][0]['metrics'])
    elif bad=='Qchange':r['rows'][0]['metrics']['actual_Q_change']+=1;o['rows'][0]['metrics']=deepcopy(r['rows'][0]['metrics'])
    elif bad=='Qname':p['Q']='fitted'
    elif bad=='prediction':p['predicted']=[0.]*192
    elif bad in ('negative','nan'):
        r['rows'][0]['metrics']['actual_Q']=-1 if bad=='negative' else float('nan');o['rows'][0]['metrics']=deepcopy(r['rows'][0]['metrics'])
    elif bad=='claim':b['P3_admission']=True
    elif bad=='receiptclaim':o['P1_admission']=True
    elif bad=='modelSHA':p['compiler_report_sha256']='0'*64
    else:o['rows'][0]['root_sha256']='0'*64
    with pytest.raises(ValueError):m.assemble(b,r,o)


def test_D_minus_reference_arithmetic_and_ties_not_fake_reversal():
    b,r,o=fixture()
    for row in b['rows']:row['model_Q']=.5
    result=m.assemble(b,r,o)
    for ref,data in result['D_minus_reference'].items():
        vals=[]
        for row in data['per_root']:
            s=row['source_slot'];d=next(x for x in result['rows'] if (x['source_slot'],x['arm'])==(s,m.rgb.ARMS[1]));a=next(x for x in result['rows'] if (x['source_slot'],x['arm'])==(s,ref))
            delta=d['terminal10']['metrics']['actual_Q']-a['terminal10']['metrics']['actual_Q'];vals.append(delta)
            assert row['actual_terminal_Q_D_minus_reference']==delta and row['terminal_model_actual_order_reversed'] is None
        assert data['mean_actual_terminal_Q_D_minus_reference']==float(np.mean(vals))


def test_build_only_frozen_saved_providers_and_full_closure(monkeypatch):
    b,r,o=fixture();comparison_path='/RAM/comparison/report.json';rgb_output='/RAM/rgb';rh='e'*64;h='f'*64
    pins={**m.PINS,comparison_path:m.COMPARISON_SHA,rgb_output+'/report.json':rh,r['prepared']:r['prepare_sha256']}
    monkeypatch.setattr(m,'sha',lambda f:pins.get(str(f),h));monkeypatch.setattr(m.comparison,'read',lambda *a:deepcopy(b))
    calls=[]
    monkeypatch.setattr(m.comparison,'inputs',lambda **a:calls.append(('old',a)) or deepcopy(b))
    monkeypatch.setattr(m.rgb,'consume',lambda *a:calls.append(('rgb',a)) or deepcopy(o))
    monkeypatch.setattr(m.rgb,'read',lambda path,*a:deepcopy(r) if Path(path).name=='report.json' else dict(bindings={}))
    result=m.build(comparison_path,m.COMPARISON_SHA,rgb_output,rh)
    assert len(result['rows'])==8 and [x[0] for x in calls]==['old','rgb']
    assert result['inputs']['rgb_report_sha256']==rh and len(result['inputs']['bindings_sha256'])>=11
    monkeypatch.setattr(m.comparison,'inputs',lambda **a:dict(b,status='altered'))
    with pytest.raises(ValueError,match='fully reproduced'):m.build(comparison_path,m.COMPARISON_SHA,rgb_output,rh)


def test_wrong_approved_comparison_SHA_refuses_before_reads(monkeypatch):
    monkeypatch.setattr(m.comparison,'read',lambda *a:pytest.fail('no reads'))
    with pytest.raises(ValueError,match='MAIN-approved'):m.build('/RAM/old','a'*64,'/RAM/rgb','e'*64)


def test_writer_floor_no_file_writes_or_native_imports(monkeypatch):
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:NS(free=m.FLOOR+m.CAP-1))
    monkeypatch.setattr(Path,'exists',lambda path:False)
    monkeypatch.setattr(Path,'mkdir',lambda *a:pytest.fail('no writes'))
    with pytest.raises(ValueError):m.write_new('/RAM/new',dict(mock=True))
    assert not any(x in sys.modules for x in ('torch','gymnasium','pymunk','stable_worldmodel'))


def test_CLI_explicit_external_SHA_and_new_output(monkeypatch,capsys):
    calls=[]
    monkeypatch.setattr(m,'build',lambda **a:calls.append(a) or dict(mock=True))
    monkeypatch.setattr(m,'write_new',lambda output,result:dict(output=output))
    monkeypatch.setattr(sys,'argv',[m.NAME,'--comparison-report','/RAM/old/report.json','--comparison-sha',m.COMPARISON_SHA,
        '--rgb-output','/RAM/rgb','--rgb-report-sha','e'*64,'--output','/RAM/new'])
    m.main();assert calls[0]['rgb_report_sha']=='e'*64 and 'output' not in calls[0]
