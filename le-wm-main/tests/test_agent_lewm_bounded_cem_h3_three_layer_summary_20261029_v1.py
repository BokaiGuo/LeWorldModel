"""RAM and temporary disk only; no imported encoder/physical/producer modules."""
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import math
import numpy as np
import pytest
from experiments.spectral_reachability import agent_lewm_bounded_cem_h3_three_layer_summary_20261029_v1 as m


def put(p,x):p.write_bytes(m.encode(x));return m.sha(p)


def fixture():
    rows=[];specs=[];comparisons=[];plans=[];native_rows=[]
    for root in (0,1):
        goal=[0.]*7
        plan=dict(task_id='task'+str(root),task_sha256=str(root)*64,selection_digest='a'*64,native_goal=goal)
        plans.append(plan);actual=dict(plan=deepcopy(plan),selection_digest=plan['selection_digest'],
            own_returned=10 if root==0 else 15,status='native_stopped' if root==0 else 'horizon_completed',checkpoints=[],responses=[])
        for step in m.CP:
            i=len(rows);present=not(root==0 and step==15)
            z=None if not present else [float(np.float32((root+step+1)/16))]*192
            pred=None if step==0 else [float(np.float32((root+step+2)/16))]*192
            state=[0.,0.,0.,0.,0.,256. if root==0 else 32.,0.]
            cp=dict(step=step,png='r%d.png'%i,file_sha256='b'*64,pixel_sha256='c'*64,state=state,snapshot=dict(state=state))
            source=dict(cp,path='/RAM_ONLY/r%d.png'%i) if present else None
            # Deliberately native-success true despite large velocity distance.
            native_metrics=dict(native_success=True,native=dict(native_success=True)) if present else None
            response=dict(step=step,state=state,native_success=True,reward=-state[5]) if present and step else None
            if present:actual['checkpoints'].append(deepcopy(cp))
            if response is not None:actual['responses'].append(deepcopy(response))
            spec=dict(index=i,plan_index=root,root_slot=root,checkpoint=step,absolute_step=10+step,condition='off',
                arm='bounded_native_CEM_H3_actualmean',**plan,physical_status=actual['status'],actual_returned=actual['own_returned'],
                source=source,native_metrics=native_metrics,response=response,goal192=[0.]*192,predicted192=pred,
                model_Q=None if pred is None else pred[0]**2,causal_raw10_bits=None if not present else '00'*80,
                planned_raw10_bits='00'*80,Q_sha256='d'*64,model_hash='e'*64,
                missing_reason=None if present else 'native_stop_before_checkpoint',
                status='not_started' if present else 'missing_physical_checkpoint',latent=None)
            row=deepcopy(spec);row['actual192']=z
            if present:row.update(status='encoded',latent=dict(fixture=True),pixels={},preprocessed={},modules=[])
            aq=None if z is None else z[0]**2;pq=spec['model_Q'];err=None if aq is None or pq is None else pq-aq
            comparisons.append(dict(index=i,root_slot=root,checkpoint=step,absolute_step=10+step,task_id=plan['task_id'],
                model_Q=pq,actual_encoded_Q=aq,prediction_vs_encoded_MSE=None if z is None or pred is None else (pred[0]-z[0])**2,
                model_minus_actual_Q=err,native_metrics=deepcopy(native_metrics),native_success=True if present else None,
                latent_missing=not present,anchor_reencode=None))
            specs.append(spec);rows.append(row)
        native_rows.append(actual)
    counts={k:dict(attempted=7 if k in ('encoder','projector') else 0,completed=7 if k in ('encoder','projector') else 0) for k in m.KINDS}
    phy=dict(schema=m.PHY+'.consume',status='bounded_CEM_H3_known2_native_recorded_not_admitted',errors=[],
        fixed_denominator=2,known_roots=2,fresh_roots=0,plans=plans,rows=native_rows,counts=dict(public=45,physics=458,PNG=147))
    resources=dict(predict=192,AE=68)
    prep=dict(schema=m.RGB+'.prepare',rows=specs,bindings={},arguments={},inherited_model_resources=resources,inherited_model_wall_seconds=12.)
    pp=dict(schema=m.PHY+'.prepare',plans=deepcopy(plans),planner_authority=dict(resources=resources))
    rgb=dict(schema=m.RGB+'.consume',status='all_available8_CEM_H3_RGB_verified_NOT_admitted',fixed_denominator=8,
        trajectory_denominator=2,known_roots=2,fresh_roots=0,encoded=7,missing=1,rows=rows,comparisons=comparisons,
        full_upstream_rebuilt_here=False,own_native_encoding_saved_reconstructed_here=True,
        consume_new_neural_calls=0,consume_new_physics=0,inherited_physical_counts=phy['counts'],counts=counts,
        inherited_model_resources=dict(predict=192,AE=68),inherited_model_wall_seconds=12.,bindings={},**{k:False for k in m.FLAGS})
    return rgb,prep,phy,pp


def disk_fixture(tmp,mp):
    rgb,prep,phy,pp=fixture()
    for row in rgb['rows']:
        if row['actual192'] is None:continue
        a=np.asarray(row['actual192'],np.float32);stream=io.BytesIO();np.save(stream,a,allow_pickle=False)
        p=tmp/('r%d_latent.npy.gz'%row['index']);p.write_bytes(gzip.compress(stream.getvalue(),mtime=0))
        row['latent']=dict(file=p.name,sha256=m.sha(p),array_sha256=hashlib.sha256(a.tobytes()).hexdigest(),shape=[192],dtype='float32')
        rgb['bindings'][str(p)]=m.sha(p)
    paths={k:tmp/(k+'.json') for k in ('rgb_receipt','rgb_prepare','physical_receipt','physical_prepare')}
    specs={}
    for k,x in (('physical_receipt',phy),('physical_prepare',pp)):
        specs[k]=dict(path=str(paths[k]),sha256=put(paths[k],x))
    mp.setattr(m,'PHYSICAL_SHA',specs['physical_receipt']['sha256']);mp.setattr(m,'PHYSICAL_PREP_SHA',specs['physical_prepare']['sha256'])
    prep['bindings'][str(m.ROOT/'experiments/spectral_reachability'/(m.RGB+'.py'))]=m.RGB_SOURCE_SHA
    prep['arguments']=dict(physical_receipt=specs['physical_receipt']['path'],physical_receipt_sha=m.PHYSICAL_SHA,
        physical_prepared=specs['physical_prepare']['path'],physical_prepare_sha=m.PHYSICAL_PREP_SHA)
    specs['rgb_prepare']=dict(path=str(paths['rgb_prepare']),sha256=put(paths['rgb_prepare'],prep))
    rgb['source_authorities']=prep['arguments'];rgb['prepare_sha256']=specs['rgb_prepare']['sha256']
    rgb['report_sha256']='7'*64;rgb['completion_sha256']='8'*64
    rgb['completion']=dict(report_sha256=rgb['report_sha256'],prepare_sha256=rgb['prepare_sha256'],prepared=str(paths['rgb_prepare']))
    specs['rgb_receipt']=dict(path=str(paths['rgb_receipt']),sha256=put(paths['rgb_receipt'],rgb))
    mp.setattr(m,'RGB_PREPARE_SHA',specs['rgb_prepare']['sha256']);mp.setattr(m,'RGB_RECEIPT_SHA',specs['rgb_receipt']['sha256'])
    mp.setattr(m,'RGB_REPORT_SHA',rgb['report_sha256']);mp.setattr(m,'RGB_COMPLETION_SHA',rgb['completion_sha256'])
    manifest=dict(schema=m.NAME+'.inputs',trust='MAIN_complete_external_receipt_SHAs_not_rebuilt',inputs=specs)
    p=tmp/'inputs.json';h=put(p,manifest)
    return p,h,manifest


def test_all8_all12_arithmetic_null_native_success():
    r=m.summarize(*fixture())
    assert len(r['rows'])==8 and len(r['pairs'])==12 and r['available']['actual_positions']==7
    assert r['rows'][3]['actual_Q'] is r['rows'][3]['native'] is None
    assert r['rows'][3]['model_Q'] is not None
    assert r['rows'][0]['model_Q'] is None and r['rows'][0]['native']['native_success'] is True
    assert r['rows'][0]['native']['velocity_squared']==65536 and r['rows'][0]['native']['native7_distance']==256
    assert sum(p['native_delta'] is None for p in r['pairs'])==3
    for c in r['comparison_census'].values():assert sum(c.values())==12


@pytest.mark.parametrize('change',['missing','order','cp','task','goal','success','reward','state','mean','counts','claim','modelhash'])
def test_bad_identity_or_arithmetic_rejected(change):
    rgb,prep,phy,pp=fixture()
    if change=='missing':rgb['rows'][3]['actual192']=[0.]*192
    elif change=='order':rgb['rows'].reverse()
    elif change=='cp':rgb['rows'][0]['absolute_step']=0
    elif change=='task':rgb['rows'][0]['task_id']='other'
    elif change=='goal':rgb['rows'][0]['native_goal'][5]=100.
    elif change=='success':rgb['comparisons'][0]['native_success']=False
    elif change=='reward':rgb['rows'][1]['response']['reward']=0.
    elif change=='state':rgb['rows'][1]['source']['state'][0]=1.
    elif change=='mean':rgb['comparisons'][1]['actual_encoded_Q']+=1e-9
    elif change=='counts':rgb['counts']['predict_API']['completed']=1
    elif change=='claim':rgb['fair_ranking']=True
    else:rgb['rows'][1]['model_hash']='f'*64
    with pytest.raises(ValueError):m.summarize(rgb,prep,phy,pp)


def test_full_tempdisk_cli_build_and_wholeverify(tmp_path,monkeypatch,capsys):
    p,h,_=disk_fixture(tmp_path,monkeypatch);output=tmp_path/(m.NAME+'_RAM.json')
    m.main(['analyze','--inputs',str(p),'--inputs-sha',h,'--output',str(output)])
    assert json.loads(capsys.readouterr().out)['sha256']==m.sha(output)
    x=m.read(output,m.sha(output));assert len(x['bindings'])==15
    assert x['new_neural_calls']==x['new_physics_calls']==x['new_fullstrict_calls']==0
    m.main(['verify','--summary',str(output),'--summary-sha',m.sha(output)])
    assert json.loads(capsys.readouterr().out)['verified'] is True
    with pytest.raises(ValueError):m.main(['analyze','--inputs',str(p),'--inputs-sha',h,'--output',str(output)])


@pytest.mark.parametrize('change',['receipt','latent','orphanref','source','wrongsha','manifestkey','duplicate','nonfinite'])
def test_disk_authority_and_array_tamper(tmp_path,monkeypatch,change):
    p,h,spec=disk_fixture(tmp_path,monkeypatch)
    if change=='receipt':Path(spec['inputs']['rgb_receipt']['path']).write_bytes(b'{}')
    elif change=='latent':next(tmp_path.glob('*.npy.gz')).write_bytes(b'changed')
    elif change in ('source','orphanref'):
        k='rgb_prepare' if change=='source' else 'rgb_receipt';path=Path(spec['inputs'][k]['path']);x=m.read(path,spec['inputs'][k]['sha256'])
        if change=='source':x['bindings'][str(m.ROOT/'experiments/spectral_reachability'/(m.RGB+'.py'))]='f'*64
        else:x['bindings'][str(tmp_path/'extra'/'r0_latent.npy.gz')]='f'*64
        spec['inputs'][k]['sha256']=put(path,x);h=put(p,spec)
        if change=='source':
            monkeypatch.setattr(m,'RGB_PREPARE_SHA',spec['inputs'][k]['sha256'])
            rs=spec['inputs']['rgb_receipt'];rp=Path(rs['path']);rx=m.read(rp,rs['sha256'])
            rx['prepare_sha256']=m.RGB_PREPARE_SHA;rx['completion']['prepare_sha256']=m.RGB_PREPARE_SHA
            rs['sha256']=put(rp,rx);monkeypatch.setattr(m,'RGB_RECEIPT_SHA',rs['sha256']);h=put(p,spec)
        else:monkeypatch.setattr(m,'RGB_RECEIPT_SHA',spec['inputs'][k]['sha256'])
    elif change=='wrongsha':h='f'*64
    elif change=='manifestkey':spec['inputs']['future']={};h=put(p,spec)
    elif change=='duplicate':p.write_bytes(b'{"schema":1,"schema":2}');h=m.sha(p)
    else:p.write_bytes(b'{"x":NaN}');h=m.sha(p)
    with pytest.raises(ValueError):m.analyze(str(p),h)


@pytest.mark.parametrize('key',['report_sha256','completion_sha256','prepare_sha256'])
def test_resealed_receipt_still_rejects_wrong_publication_binding(tmp_path,monkeypatch,key):
    p,h,spec=disk_fixture(tmp_path,monkeypatch);s=spec['inputs']['rgb_receipt'];path=Path(s['path'])
    x=m.read(path,s['sha256']);x[key]='f'*64;s['sha256']=put(path,x)
    monkeypatch.setattr(m,'RGB_RECEIPT_SHA',s['sha256']);h=put(p,spec)
    with pytest.raises(ValueError):m.analyze(str(p),h)


def test_exact_sign_no_product_underflow_or_epsilon():
    assert m.compare(1e-300,-1e-300)=='reversal'
    assert m.compare(0.,-0.)=='both_tie' and m.compare(None,1.)=='null'
    assert m.native([0,0,0,0,4*math.pi,0,0],[0]*7)['negative_native_angle'] is True
    with pytest.raises(ValueError):m.vector([True]*192,192,True)


def test_all_native_coordinate_contributions_independent_known_arithmetic():
    n=m.native([3.,4.,6.,8.,.5,12.,5.],[0.]*7)
    assert n['axis_squared']==[9.,16.,36.,64.,.25,144.,25.]
    assert (n['agent_xy_squared'],n['object_xy_squared'],n['velocity_squared'])==(25.,100.,169.)
    assert n['native7_squared']==294.25 and n['joint_xy_squared']==125.
    assert n['native7_distance']==math.sqrt(294.25) and n['native_success'] is False


def test_wholeverify_rejects_resealed_local_summary_edit(tmp_path,monkeypatch):
    p,h,_=disk_fixture(tmp_path,monkeypatch);x=m.analyze(str(p),h)
    x['summary']['rows'][3]['actual_Q']=0.
    out=tmp_path/(m.NAME+'_edited.json');sh=put(out,x)
    with pytest.raises(ValueError):m.main(['verify','--summary',str(out),'--summary-sha',sh])
