"""RAM-only saved bridge fixtures; injected receipts are not CUDA evidence."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from io import BytesIO
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.spectral_reachability import moving_overlay_task_metric_bridge_20261027_v1 as b


def fixture():
    base=dict(schema=b.COMPARISON+'.report',status='saved_physical_comparison_not_admitted',
        root_denominator=2,branch_denominator=8,P1_admission=False,model_repaired=False,
        independent_confirmation=False,equal_total_FLOPs=False,actual_neural_calls=0,actual_physics_calls=0,
        rows=[],binding=dict(source_sha256='a'*64,files=dict(physical_report=dict(file='/physical/report.json',sha256='b'*64))),
        physical_counts={'public_step':{'attempted':160,'completed':160}},physical_errors=[],
        inputs_final_verified=True,descriptive={'original':'unchanged'},planner_cost_receipt={'nested':'not summed'})
    report=dict(schema=b.RGB+'.report',status='encoded_sealed',P1_admission=False,training=False,
        cuda_verified=True,source_verified=True,model_unchanged=True,errors=[],
        input_binding=dict(physical_report='/physical/report.json',physical_sha='b'*64),
        rows=[],images=[],prediction_status='not_recomputed_encoding_only',
        shapes=dict(pixels=[1,3,224,224],hidden=[1,257,192],cls=[1,192],projected=[1,192]))
    for s in (0,1):
        for c in ('off','on'):
            report['images'].append(dict(index=len(report['images']),kind='goal',root_slot=s,condition=c,
                path=f'/paired/s{s}_{c}.npz',file_sha256='c'*64,pixel_sha256=f'{s*2+(c=="on")+20:064x}',latent={'file':f'g{s}_{c}.npy.gz'}))
    for row_index,(s,c,arm) in enumerate(b.CELLS):
        goal=np.asarray([1,2,3,4,.2,.3,.4],np.float64)+s
        initial=goal+np.asarray([5,0,8,0,.1,.2,.1])
        root=b.digest(np.concatenate([initial,goal]).tobytes())
        row=dict(root_slot=s,condition=c,arm=arm,selection_digest=f'{row_index+1:064x}',
            root_sha256=root,goal=goal.tolist(),source_status='horizon_completed',
            checkpoint_refs=[],missing_checkpoints=[],first_error=None,secondary_errors=[],
            reported_native_success=False,actual_RGB=dict(actual_latent_Q=None,status='missing_not_encoded'))
        rgbrow=dict(root_slot=s,condition=c,arm=arm,selection_digest=row['selection_digest'],goal_image=s*2+(c=='on'),checkpoints=[])
        for t,name,factor in ((0,'initial',1.),(5,'prefix5',.8),(10,'terminal10',.6)):
            state=goal+(initial-goal)*factor;metrics=b.physical.metrics(goal,state);row[name]=metrics
            cp=dict(step=t,png=f'b{row_index}_{t}.png',file_sha256=f'{row_index*3+t+60:064x}',pixel_sha256=f'{row_index*3+t+160:064x}')
            row['checkpoint_refs'].append(cp)
            idx=len(report['images']);latent=dict(file=f'i{idx}.npy.gz',sha256='d'*64,array_sha256='e'*64,shape=[192],dtype='float32')
            report['images'].append(dict(index=idx,kind='checkpoint',row=row_index,root_slot=s,condition=c,step=t,
                path='/physical/'+cp['png'],file_sha256=cp['file_sha256'],pixel_sha256=cp['pixel_sha256'],
                physical=dict(cp,state=state.tolist(),snapshot=dict(state=state.tolist())),latent=latent,status='encoded'))
            rgbrow['checkpoints'].append(dict(step=t,image=idx,latent=latent,Q=float(10-t*.5),status='encoded'))
        base['rows'].append(row);report['rows'].append(rgbrow)
    return base,report,deepcopy(report['rows'])


def test_complete8_saved_join_and_original_preserved():
    base,report,receipt=fixture();original=deepcopy(base)
    result=b.assemble(base,report,receipt)
    assert len(result['rows'])==8 and result['branch_denominator']==8
    assert base==original and result['physical_comparison']==base
    first=result['rows'][0]['checkpoints'][0]
    assert first['actual_latent_Q']==10 and first['errors']['object_position']['squared_l2']==64
    assert first['errors']['agent_position']['squared_l2']==25
    assert result['P1_admission'] is False and result['latent_improvement_implies_task_improvement'] is False
    assert result['physical_counts']==base['physical_counts']


def test_explicit_identity_join_not_array_position():
    base,report,receipt=fixture();expected=b.assemble(base,report,receipt)
    base['rows'].reverse();receipt.reverse()
    assert b.assemble(base,report,receipt)['rows']==expected['rows']


@pytest.mark.parametrize('mutation',['selection','condition','duplicate','missing_row','cp_bool','q_nan','q_negative',
    'latent','png','file_sha','pixel_sha','cp_state','root','goal','metrics','physical_sha','physical_path','p1','hidden197'])
def test_conflicts_failclosed(mutation):
    base,report,receipt=fixture();row=receipt[0];im=report['images'][4]
    if mutation=='selection':row['selection_digest']='f'*64
    elif mutation=='condition':row['condition']='on'
    elif mutation=='duplicate':receipt[1]=deepcopy(receipt[0])
    elif mutation=='missing_row':receipt.pop()
    elif mutation=='cp_bool':row['checkpoints'][0]['step']=False
    elif mutation=='q_nan':row['checkpoints'][0]['Q']=float('nan')
    elif mutation=='q_negative':row['checkpoints'][0]['Q']=-1.
    elif mutation=='latent':row['checkpoints'][0]['latent']['array_sha256']='f'*64
    elif mutation=='png':im['path']='/another/'+Path(im['path']).name
    elif mutation=='file_sha':im['file_sha256']='f'*64
    elif mutation=='pixel_sha':im['pixel_sha256']='f'*64
    elif mutation=='cp_state':im['physical']['snapshot']['state'][0]=99
    elif mutation=='root':base['rows'][0]['root_sha256']='f'*64
    elif mutation=='goal':base['rows'][0]['goal'][0]=999
    elif mutation=='metrics':base['rows'][0]['terminal10']['velocity_error_l2']=999
    elif mutation=='physical_sha':report['input_binding']['physical_sha']='f'*64
    elif mutation=='physical_path':report['input_binding']['physical_report']='/other/report.json'
    elif mutation=='p1':base['P1_admission']=True
    else:report['shapes']['hidden']=[1,197,192]
    with pytest.raises(ValueError):b.assemble(base,report,receipt)


def test_missing_real_checkpoint_stays_null_and_no_terminal_substitution():
    base,report,receipt=fixture();row=base['rows'][0];missing=report['rows'][0]['checkpoints'][2]
    # No image exists for this terminal: remove it, reindex later references.
    index=missing['image'];report['images'].pop(index)
    for i,image in enumerate(report['images']):image['index']=i
    for r in report['rows']:
        for cp in r['checkpoints']:
            if cp['image']==index:cp.update(image=None,latent=None,Q=None,status='missing_physical_checkpoint')
            elif cp['image'] is not None and cp['image']>index:cp['image']-=1
    row['checkpoint_refs'].pop();row['missing_checkpoints']=[10];row['terminal10']=None
    receipt=deepcopy(report['rows'])
    result=b.assemble(base,report,receipt);cp=result['rows'][0]['checkpoints'][2]
    assert cp['actual_latent_Q'] is None and cp['errors'] is None
    assert cp['status']=='missing_physical_checkpoint'
    assert result['paired_on_minus_off'][0]['checkpoints'][2]['delta']['actual_latent_Q'] is None
    assert result['paired_on_minus_off'][0]['checkpoints'][2]['delta']['object_position_l2'] is None


def test_wrapped_angle_and_native_negative_angle_preserved():
    goal=[0.]*7;state=[0,0,0,0,4*math.pi,3,4]
    errors=b.decompose(goal,state)
    assert errors['velocity']['l2']==5.
    assert errors['angle']['native_term']<0 and errors['angle']['absolute_wrapped']<1e-12
    assert errors['native']['negative_native_angle'] is True
    assert errors['raw_squared_total']==pytest.approx(sum(errors[k]['squared_l2'] for k in ('agent_position','object_position','velocity'))+errors['angle']['squared_raw'])


def test_goal_velocity_not_assumed_zero_and_no_units_sum_with_latent():
    goal=[0,0,0,0,0,2,3];state=[3,4,0,0,0,5,7]
    e=b.decompose(goal,state)
    assert e['velocity']['delta']==[3.,4.] and e['velocity']['l2']==5.
    assert 'latent' not in e


def test_latent_lower_task_worse_is_descriptive_not_pass():
    base,report,receipt=fixture()
    raw=report['images'][6]['physical'];raw['state'][2]+=50
    raw['snapshot']['state']=raw['state'].copy()
    base['rows'][0]['terminal10']=b.physical.metrics(base['rows'][0]['goal'],raw['state'])
    result=b.assemble(base,report,receipt)
    temporal=result['rows'][0]['changes']['terminal10_minus_initial0']
    assert temporal['actual_latent_Q']==-5. and temporal['object_position_l2']>0
    assert 'task_pass' not in result and result['P1_admission'] is False


def test_cli_help_imports_no_torch(capsys):
    assert 'torch' not in sys.modules
    with pytest.raises(SystemExit):b.main(['--help'])
    assert '--rgb-report-sha' in capsys.readouterr().out


def test_material_missing_real_hash_rejected_before_consumer():
    with pytest.raises(ValueError):b.material('/base',None,'/rgb',None,read=lambda _:pytest.fail('invalid SHA before read'),consumer=lambda *a:pytest.fail('no consume'))


def test_external_material_joins_ownconsume_only_after_sha(monkeypatch):
    base,report,receipt=fixture();payloads={}
    basepath='/base.json';rgbpath='/rgb/report.json'
    source=b'RAM comparison source';rgbsource=b'RAM RGB source'
    base['binding']['source_sha256']=b.digest(source)
    payloads.update({basepath:b.canonical(base),rgbpath:b.canonical(report),str(b.COMPARISON_FILE):source,str(b.RGB_FILE):rgbsource})
    monkeypatch.setattr(b,'RGB_SOURCE_SHA',b.digest(rgbsource))
    seen=[]
    def consume(output,h):seen.append((output,h));return receipt
    out=b.material(basepath,b.digest(payloads[basepath]),'/rgb',b.digest(payloads[rgbpath]),read=payloads.__getitem__,consumer=consume)
    assert seen==[('/rgb',b.digest(payloads[rgbpath]))] and out['inputs']['rgb_report']['sha256']==b.digest(payloads[rgbpath])
    payloads[rgbpath]+=b' '
    with pytest.raises(ValueError):b.material(basepath,b.digest(payloads[basepath]),'/rgb',out['inputs']['rgb_report']['sha256'],read=payloads.__getitem__,consumer=lambda *a:pytest.fail('hash first'))


@pytest.mark.parametrize('failure',['new','exists','floor','oversize'])
def test_write_new_ram_exclusive_floor_cap_and_fsync(monkeypatch,failure):
    payloads=[];modes=[];synced=[]
    class RAMFile(BytesIO):
        def fileno(self):return 8
        def close(self):payloads.append(self.getvalue());super().close()
    def opened(self,mode):modes.append(mode);return RAMFile()
    monkeypatch.setattr(Path,'exists',lambda _:failure=='exists')
    monkeypatch.setattr(Path,'is_symlink',lambda _:False)
    monkeypatch.setattr(Path,'open',opened)
    monkeypatch.setattr(b.shutil,'disk_usage',lambda _:SimpleNamespace(free=0 if failure=='floor' else b.FLOOR+b.CAP))
    monkeypatch.setattr(b.os,'fsync',lambda fd:synced.append(fd))
    monkeypatch.setattr(b.os,'open',lambda *a:9)
    monkeypatch.setattr(b.os,'close',lambda *a:None)
    value={'data':'x'*b.CAP} if failure=='oversize' else {'P1_admission':False}
    if failure=='new':
        h=b.write_new('/RAM_NEW_metric_bridge.json',value)
        assert modes==['xb'] and h==b.digest(payloads[0]) and synced==[8,9]
    else:
        with pytest.raises(ValueError):b.write_new('/RAM_NEW_metric_bridge.json',value)
        assert modes==[]


def test_cli_ram_dispatch_no_actual_outputs(monkeypatch,capsys):
    calls=[];args=['--comparison','/comp','--comparison-sha','a'*64,'--rgb-output','/rgb',
        '--rgb-report-sha','b'*64,'--output','/new.json']
    monkeypatch.setattr(b,'material',lambda *a:calls.append(a) or {'status':'saved_task_metric_bridge_not_admitted'})
    monkeypatch.setattr(b,'write_new',lambda *a:'c'*64)
    assert b.main(args)==0
    assert calls==[('/comp','a'*64,'/rgb','b'*64)]
    assert json.loads(capsys.readouterr().out)['P1_admission'] is False


def test_condition_goal_and_declared_images_not_dropped():
    base,report,receipt=fixture();report['rows'][0]['goal_image']=1
    with pytest.raises(ValueError):b.assemble(base,report,receipt)
    base,report,receipt=fixture();report['images'].append(dict(report['images'][-1],index=28))
    with pytest.raises(ValueError):b.assemble(base,report,receipt)


def test_no_real_future_sha_no_saved_run(monkeypatch):
    for sha in ('future', '0', None):
        with pytest.raises(ValueError):b.material('/base',sha,'/rgb','a'*64,
            read=lambda _:pytest.fail('invalid external SHA before reads'),consumer=lambda *a:pytest.fail('never consume'))
