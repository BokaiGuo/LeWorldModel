"""Independent six-frame saved receipt arithmetic; NumPy/PIL/stdlib only."""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import shutil
import zlib

import numpy as np
from PIL import Image

NAME = 'p1_line_search_gn_rgb_saved6_audit_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'results/p1_line_search_gn_rgb_cuda_20261027_v1_cuda_main01/report.json'
REPORT_SHA = 'e1ec726ec0d54ef7347030abdaa10a2bdf4f5d4767e327970a2a29c819723dcd'
PREPARE_SHA = 'cf384f59734fd7ae247dd3d0bf82414d26b5d000fb5f7851eb36cca07c625e0f'
PHYSICAL_SHA = '3a41fe83f375e5bb3f390e9f3a0e8e17dd70c2d49a7ebe3a99079f5774de85c1'
PHYSICAL_PREPARE_SHA = 'f36f915ccfc2687da056810d1241da433190eff430bb54fa7b7b35a6af3818d5'
MAX_BYTES = 32*1024**2


def require(ok, message):
    if not ok: raise ValueError(message)


def canonical(v): return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(raw): return hashlib.sha256(raw).hexdigest()
def source_digest(v): return digest(json.dumps(v,sort_keys=True,allow_nan=False).encode())


def sha(path):
    path=Path(path); require(path.is_file() and not path.is_symlink(),'regular nonsymlink input: '+str(path))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def bytes_at(path):
    require(Path(path).stat().st_size<=MAX_BYTES,'bounded raw input')
    return Path(path).read_bytes()


def read(path):
    data=bytes_at(path)
    if str(path).endswith('.gz'):
        d=zlib.decompressobj(31); data=d.decompress(data,MAX_BYTES+1)
        require(d.eof and not d.unused_data and not d.unconsumed_tail,'complete bounded single gzip')
    require(len(data)<=MAX_BYTES,'bounded expanded input')
    def unique(items):
        result={}
        for k,v in items:
            require(k not in result,'duplicate JSON key'); result[k]=v
        return result
    value=json.loads(data,object_pairs_hook=unique); canonical(value); return value


def array(v, shape):
    a=np.asarray(v)
    require(a.dtype.kind in 'fi' and a.shape==shape and np.isfinite(a).all(),'finite aligned '+str(shape))
    return a.astype(float)


def native(data):
    stream=BytesIO(data); z=np.load(stream,allow_pickle=False)
    require(isinstance(z,np.ndarray) and z.dtype==np.float32 and z.shape==(192,) and np.isfinite(z).all(),
            'raw native float32 NPY192, no cast')
    require(stream.tell()==len(data),'no trailing NPY bytes')
    return z


def pixels(data):
    with Image.open(BytesIO(data)) as image:
        require(image.format=='PNG' and image.mode=='RGB' and image.size==(224,224),'original224 RGB PNG')
        rgb=np.asarray(image).copy()
    require(rgb.dtype==np.uint8 and rgb.shape==(224,224,3),'exact uint8 RGB')
    return rgb


def prediction(task,pred,taskid):
    require(source_digest(task)==taskid==pred['task_id'],'task/goal/Q/context exact source identity')
    require(task['horizon']==2 and task['packet_steps']==5 and task['control_dim']==2 and
            task['representation']=='native_projected_cls' and task['feedback']=='open_loop_model_only','original whole H2 contract')
    logits=array(pred['logits'],(2,10)); rawcommands=np.tanh(logits.astype(np.float32)).reshape(10,2)
    require(np.array_equal(rawcommands,pred['commands']) and (np.linalg.norm(logits,axis=1)<=task['packet_logit_cap']).all(),'original feasible float32 tanh action plan')
    std=array(task['action_std'],(2,)); require((std>0).all(),'normalizer positive')
    initial=task['root_context']
    require((initial['layout'],initial['context_length'],initial['latent_dim'],initial['action_dim'])==
            ('states_with_between_state_actions_v1',3,192,10),'root context native layout')
    array(initial['latent_history'],(3,192)); array(initial['action_history'],(2,10))
    rootraw=array(task['root_raw_action_history'],(10,2)).astype(np.float32).reshape(2,10)
    root_tokens=(rootraw-np.tile(np.asarray(task['action_mean'],np.float32),5))/np.tile(std.astype(np.float32),5)
    require(np.array_equal(root_tokens,initial['action_history']),'source root raw normalized history')
    tokens=(rawcommands.reshape(2,10)-np.tile(np.asarray(task['action_mean'],np.float32),5))/np.tile(std.astype(np.float32),5)
    require(len(pred['contexts'])==2,'wholeH2 two predictions')
    previous=task['root_context']
    for k,c in enumerate(pred['contexts']):
        require((c['layout'],c['context_length'],c['latent_dim'],c['action_dim'])==('states_with_between_state_actions_v1',3,192,10),'prediction context layout')
        array(c['latent_history'],(3,192)); array(c['action_history'],(2,10))
        require(c['latent_history'][:2]==previous['latent_history'][1:] and
                np.array_equal(c['action_history'],np.vstack((previous['action_history'],tokens[k]))[-2:]),'literal prediction history chronology')
        previous=c
    require(pred['segmented_replay']==pred['contexts'] and pred['whole_replay']==pred['contexts'][-1],'saved independent H2 replay identity')
    q=array(task['projection'],(192,192)); goal=array(task['goal'],(192,))
    require(np.allclose(q,q.T,atol=1e-8,rtol=0) and np.allclose(q@q,q,atol=1e-8,rtol=0),'original orthogonal Q')
    require(float(np.mean((q@(np.asarray(previous['latent_history'][-1])-goal))**2))==pred['cost'],'original final predicted Q cost')
    return [task['root_context']['latent_history'][-1]]+[c['latent_history'][-1] for c in pred['contexts']]


def load_bundle(report=REPORT, report_sha=REPORT_SHA):
    report=Path(report).resolve(); folder=report.parent; bindings={}
    def bound(path,expected):
        path=Path(path).absolute(); require(sha(path)==expected,'SHA mismatch: '+str(path))
        bindings[str(path)]=expected; return path
    r=read(bound(report,report_sha))
    require(report_sha==REPORT_SHA,'fixed actual encoder receipt')
    names={'binding.json','image.events.jsonl','analysis.json'}|{f'frame_{i:02d}.native192.npy' for i in range(6)}
    require(set(r['artifact_sha256'])==names and {p.name for p in folder.iterdir()}==names|{'report.json'},'exact raw six NPY output census')
    for name,h in r['artifact_sha256'].items(): bound(folder/name,h)
    binding=read(folder/'binding.json')
    require(binding['prepare_sha256']==r['prepare_sha256']==PREPARE_SHA,'actual original encode preparation')
    p=read(bound(binding['prepared'],PREPARE_SHA))
    for path,h in p['bindings_sha256'].items(): bound(path,h)
    arguments=p['arguments']
    require(arguments['physical_report_sha']==PHYSICAL_SHA and arguments['physical_prepare_sha']==PHYSICAL_PREPARE_SHA,'fixed actual physical pair')
    physical=read(bound(arguments['physical_report'],PHYSICAL_SHA)); physical_folder=Path(arguments['physical_report']).parent
    prepared=read(bound(arguments['physical_prepared'],PHYSICAL_PREPARE_SHA))
    require(physical['prepare_sha256']==PHYSICAL_PREPARE_SHA,'actual physical execution preparation')
    journal=bound(physical_folder/'journal.jsonl',physical['journal_sha256'])
    events=[json.loads(line) for line in bytes_at(journal).splitlines()]
    tasks=[]; predictions=[]; pngdata=[]
    for plan in prepared['plans']:
        require(str(Path(plan['task_file']).absolute()) in bindings and str(Path(plan['origin']).absolute()) in bindings,'raw task and actual prediction sealed')
        task=read(bound(plan['task_file'],plan['task_file_sha256']))['task']
        component=read(bound(plan['origin'],plan['origin_sha256']))
        tasks.append(task); predictions.append(component['final_prediction'])
    for f in p['frames']:
        require(Path(f['path']).parent==physical_folder,'actual physical PNG, not old endpoint')
        pngdata.append(bytes_at(bound(f['path'],f['file_sha256'])))
    npydata=[bytes_at(folder/f'frame_{i:02d}.native192.npy') for i in range(6)]
    image_events=[json.loads(line) for line in bytes_at(folder/'image.events.jsonl').splitlines()]
    return dict(report=r,prepare=p,physical=physical,physical_prepared=prepared,physical_events=events,tasks=tasks,
        predictions=predictions,pngdata=pngdata,npydata=npydata,events=image_events,analysis=read(folder/'analysis.json'),
        folder=str(folder),bindings=bindings)


def audit(b):
    r,p=b['report'],b['prepare']; physical,prepared=b['physical'],b['physical_prepared']
    require(r['schema']=='p1_line_search_gn_rgb_cuda_20261027_v1.report' and r['status']=='six_actual_rgb_latents_saved' and
            r['errors']==[] and r['first_error'] is None,'actual successful six frame receipt')
    require(p['schema']=='p1_line_search_gn_rgb_cuda_20261027_v1.prepare' and p['contract']['frames']==6 and
            p['contract']['encoder_cap']==p['contract']['projector_cap']==6 and p['contract']['checkpoints']==[0,5,10], 'six frame frozen contract')
    require(r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and r['model_unchanged'] is True and r['source_verified'] is True,'recorded frozen model state')
    for key in ('new_predict','new_actionencoder','new_AD','new_physics'): require(r[key]==0,'no new '+key)
    require(r['training'] is False and r['new_Q'] is False and r['P1_admission'] is False,'scope')
    require(physical['status']=='physical_calibration_recorded' and physical['errors']==[] and
            physical['counts']=={k:dict(attempted=v,completed=v) for k,v in dict(reset=4,render=22,step=20).items()},'actual physical full counts')
    previous=None
    for index,e in enumerate(b['physical_events']):
        copy=dict(e); h=copy.pop('event_sha256')
        require(e['index']==index and e['previous_sha256']==previous and source_digest(copy)==h,'physical hash chain')
        previous=h
    require([e['receipt'] for e in b['physical_events'] if e['kind']=='branch_closed']==physical['rows'],'all actual physical row receipts')
    inherited_physical={k:dict(attempted=sum(e['kind']==k+'_attempted' for e in b['physical_events']),
                               completed=sum(e['kind']==k+'_returned' for e in b['physical_events'])) for k in ('reset','render','step')}
    require(inherited_physical==physical['counts'],'inherited physical raw operation census')
    require(len(prepared['plans'])==len(physical['rows'])==len(b['tasks'])==len(b['predictions'])==len(p['root_metrics'])==2,'two roots complete')
    require(len({plan['root_sha256'] for plan in prepared['plans']})==2,'distinct original roots')
    require(len(p['frames'])==len(r['rows'])==len(b['pngdata'])==len(b['npydata'])==6 and len(b['events'])==36,'all6 frames/no retries/duplicates retained')
    rows=[]; counts={k:dict(attempted=0,completed=0) for k in ('encoder','projector')}; expected_events=[]; summaries=[]
    for rootindex,(plan,row,task,pred,metric) in enumerate(zip(prepared['plans'],physical['rows'],b['tasks'],b['predictions'],p['root_metrics'])):
        root=plan['root_sha256']; taskid=plan['task_id']; predicted=prediction(task,pred,taskid)
        for filename,key in (('weights.pt','model_sha256'),('config.json','model_config_sha256')):
            require(task[key]==next(h for path,h in p['model_bindings'].items() if Path(path).name==filename),'task original encoder checkpoint identity')
        require(plan['arm']==row['arm']=='line_search_GN' and row['root_sha256']==root and row['selection_sha256']==plan['selection_sha256'],'physical plan/root pairing')
        require(row['status']=='completed' and row['full_horizon'] is True and row['initial_exact'] is True and
                row['first_error'] is None and row['secondary_errors']==[] and row['attempted_steps']==row['returned_steps']==10,'all own actual steps')
        require(source_digest(pred)==plan['prediction_sha256'] and pred['commands']==plan['commands'],'original selected prediction/commands seal')
        require(metric==dict(root_index=rootindex,root_sha256=root,task_id=taskid,q=task['projection'],goal=task['goal']),'original task goal/Q exact, no cross-root projection')
        require([v['step'] for v in row['checkpoints']]==[0,5,10] and len(row['trace'])==10,'all actual checkpoints/steps')
        for step,trace in enumerate(row['trace'],1):
            require(trace['step']==step and trace['attempted'] is True and trace['returned'] is True and trace['command']==pred['commands'][step-1],'all exact dispatched commands')
        cps=[e['receipt'] for e in b['physical_events'] if e['kind']=='checkpoint_saved' and e['selection_sha256']==plan['selection_sha256']]
        require(cps==row['checkpoints'],'physical journal PNG export chronology')
        for ci,step in enumerate((0,5,10)):
            i=3*rootindex+ci; f=p['frames'][i]; cp=row['checkpoints'][ci]
            cppath=Path(cp['path']); cppath=cppath if cppath.is_absolute() else ROOT/cppath
            require(f==dict(order=i,root_index=rootindex,root_sha256=root,arm='line_search_GN',step=step,
                path=str(cppath.absolute()),file_sha256=cp['file_sha256'],pixel_sha256=cp['rgb_sha256'],
                prediction_native192=predicted[ci],task_id=taskid),'frame original pixel/prediction/goal task binding')
            if step: require(cp['state']==row['trace'][step-1]['actual_state'],'saved checkpoint state chronology')
            require(digest(b['pngdata'][i])==cp['file_sha256'],'raw PNG file SHA')
            rgb=pixels(b['pngdata'][i]); require(digest(rgb.tobytes())==cp['rgb_sha256'],'raw PNG pixel SHA')
            record=r['rows'][i]; latent=record['latent']; name=f'frame_{i:02d}.native192.npy'
            require(record==dict(order=i,root_sha256=root,step=step,status='saved',latent=latent) and
                    latent==dict(path=str(Path(b['folder'])/name),sha256=r['artifact_sha256'][name],dtype='float32',shape=[192]),'independent raw NPY export identity')
            require(digest(b['npydata'][i])==latent['sha256'],'raw NPY file SHA')
            z=native(b['npydata'][i]).astype(float); goal=array(task['goal'],(192,)); q=array(task['projection'],(192,192)); model=array(predicted[ci],(192,))
            actual=float(np.mean((q@(z-goal))**2)); imagined=float(np.mean((q@(model-goal))**2)); mse=float(np.mean((z-model)**2))
            rows.append(dict(order=i,root_sha256=root,step=step,task_id=taskid,actual_Q_mse=actual,predicted_Q_mse=imagined,native_prediction_mse=mse))
            expected_events.append(dict(kind='frame_attempt',row=dict(order=i,root_sha256=root,step=step,status='attempted')))
            for ki,resource in enumerate(('encoder','projector')):
                entry=dict(index=2*i+ki,frame=i,resource=resource,completed=False)
                expected_events.append(dict(kind='module_attempt',entry=entry)); counts[resource]['attempted']+=1
                expected_events.append(dict(kind='module_return',entry=dict(entry,completed=True))); counts[resource]['completed']+=1
            expected_events.append(dict(kind='frame_saved',row=record))
        start=np.asarray(row['checkpoints'][0]['state']); end=np.asarray(row['checkpoints'][-1]['state'])
        summaries.append(dict(root_index=rootindex,root_sha256=root,
            actual_Q_costs=[v['actual_Q_mse'] for v in rows[-3:]],predicted_Q_costs=[v['predicted_Q_mse'] for v in rows[-3:]],
            native_prediction_mse=[v['native_prediction_mse'] for v in rows[-3:]],
            object_position_delta=float(np.linalg.norm(end[2:4]-start[2:4])),object_angle_delta=float(end[4]-start[4]),
            recorded_official_success=row['trace'][-1]['native_official_evaluation']['success']))
    require(expected_events==b['events'],'ALL raw encoder/projector attempts/returns and frame order exact')
    require(r['resources']==counts=={k:dict(attempted=6,completed=6) for k in counts},'raw encoder6/projector6 accounting')
    expected_analysis=dict(rows=rows,new_Q=False,training=False,P1_admission=False,scope='original-Q descriptive calibration, not causal efficacy')
    require(b['analysis']==expected_analysis,'independent six rows originalQ/native MSE exact')
    for path,h in b['bindings'].items(): require(sha(path)==h,'postread TOCTOU: '+path)
    return dict(schema=NAME+'.report',status='saved6_raw_arithmetic_verified_not_admitted',all_checks=True,
        rows=rows,roots=summaries,resources=counts,inherited_physical_resources=inherited_physical,
        raw_NPY=6,raw_PNG=6,raw_image_events=36,input_sha256=b['bindings'],
        new_neural_calls=0,new_physics=0,new_Q=False,training=False,P1_admission=False,
        limits=['saved NPY/PNG/receipt arithmetic, not independent encoder reexecution',
                'recorded frozen tensor hashes and bound model files, no model reloaded by auditor',
                'original-Q calibration cost changes are not physical success; no cross-Q subtraction',
                'source acquisition/planner/physical costs inherited separately, not full pipeline compute equivalence'])


def execute(output):
    r=audit(load_bundle()); paths=[Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    r['audit_source_sha256']={str(p):sha(p) for p in paths}; data=canonical(r); output=Path(output).resolve()
    require(len(data)<256*1024 and not output.exists(),'exclusive small audit output')
    require(shutil.disk_usage(output.parent).free>=200*1024**2+len(data),'200MiB floor')
    with output.open('xb') as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
    return dict(output=str(output),sha256=sha(output),bytes=len(data),status=r['status'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); print(canonical(execute(args.output)).decode().strip())
