"""Encode all 24 actual physical checkpoints; diagnose, never choose actions."""
import argparse
import json
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_witness_physical_20261027_v1 as physical
from experiments.spectral_reachability import p1_orientation_cuda_main_20261027_v1 as storage

NAME='p1_witness_rgb_cuda_main_20261027_v1'
ROOT=storage.ROOT
RUN=ROOT/'results/p1_witness_physical_20261027_v1_main01'
PREPARED=ROOT/'results/p1_witness_physical_20261027_v1_prepared01'
RUN_SHA='48df27e54eca33159c44a834ce0be1a13723553dc9d64dde996b7b4ca9eaf747'
PREPARE_SHA='e4fc7b7adeac3a23036196adaadfbbe69d8873f1e8649ebbf5fe9e8fb1eca0ff'
io=storage.native.io
require=storage.require


def collect():
    require(io.sha(RUN/'report.json')==RUN_SHA,'fixed physical report')
    p=physical.verify(PREPARED,PREPARE_SHA); r=physical.read(RUN/'report.json')
    require(r['status']=='physical_calibration_recorded' and r['errors']==[] and
        r['counts']=={k:dict(attempted=n,completed=n) for k,n in dict(reset=16,render=88,step=80).items()},'all-eight actual acquisition')
    require(io.sha(RUN/'journal.jsonl')==r['journal_sha256'],'physical journal SHA')
    require(len(r['rows'])==len(p['inputs']['plans'])==8,'all physical branches')
    wp=physical.witness.verify(p['inputs']['arguments']['witness_prepared'],p['inputs']['arguments']['witness_prepare_sha'])
    tasks={t['root_sha256']:t['task'] for t in wp['inputs']['tasks']}
    frames=[];bindings={str(RUN/'report.json'):RUN_SHA,str(RUN/'journal.jsonl'):r['journal_sha256'],
        str(PREPARED/'prepare.json'):PREPARE_SHA,**p['sources'],**p['inputs']['bindings']}
    direct=physical.read(p['inputs']['arguments']['direct_report'])
    for row,plan in zip(r['rows'],p['inputs']['plans']):
        require(all(row[k]==plan[k] for k in ('root_sha256','arm','selection_sha256')) and
            row['status']=='completed' and row['full_horizon'] is True and row['initial_exact'] is True and
            row['returned_steps']==row['attempted_steps']==10 and len(row['checkpoints'])==3,'same full original physical plan order')
        task=tasks[row['root_sha256']]
        if plan['arm']=='zero':
            ref=next(x for x in direct['direct'] if x['root_sha256']==row['root_sha256'] and x['method']=='full_H2_GN')
            path=Path(ref['report_path']);require(io.sha(path)==ref['report_sha256'],'zero prediction original source')
            pred=physical.read(path)['direct_runs'][0][0]['incumbent']
        else:
            path=Path(plan['origin']); report=physical.read(path)
            pred=report['witness'] if plan['arm']=='two_stage' else report['direct_runs'][0][-1]['incumbent']
            require(physical.source.digest(pred)==plan['prediction_sha256'],'original chosen prediction')
        physical.core.check_prediction(task,pred)
        require(pred['commands']==plan['commands'],'predicted and physical exact commands')
        bindings[str(path)]=io.sha(path)
        predictions=[task['root_context']['latent_history'][-1],*[c['latent_history'][-1] for c in pred['contexts']]]
        for index,check in enumerate(row['checkpoints']):
            step=(0,5,10)[index];require(check['step']==step,'all initial/mid/terminal frames')
            path=Path(check['path']);path=path if path.is_absolute() else ROOT/path
            require(path.parent==RUN,'own actual RGB path')
            io.pixels(path,check['file_sha256'],check['rgb_sha256']);bindings[str(path)]=check['file_sha256']
            state=row['trace'][step-1]['actual_state'] if step else row['checkpoints'][0]['state']
            require(check['state']==state,'actual checkpoint chronology')
            frames.append(dict(order=len(frames),root_sha256=row['root_sha256'],arm=row['arm'],step=step,
                path=str(path),file_sha256=check['file_sha256'],pixel_sha256=check['rgb_sha256'],
                predicted_native192=predictions[index],goal=task['goal'],q=task['projection'],
                physical_diagnostic=(row['initial_pose_diagnostics'] if step==0 else row['trace'][step-1]['pose_diagnostics'])))
    original,_=storage.native.native_metadata()
    files=set(io.source_files()+[Path(__file__),Path(physical.__file__),Path(storage.__file__),
        Path(storage.screen.__file__),Path(storage.native.__file__),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')])
    return dict(schema=NAME+'.protocol',frames=frames,bindings=bindings,
        sources_sha256={str(f.resolve()):io.sha(f) for f in files},**original,
        encoder_cap=24,projector_cap=24,predict_cap=0,AD_cap=0,physical_steps_cap=0,
        output_cap_bytes=storage.CAP,disk_floor_bytes=storage.FLOOR,training=False,new_Q=False,P1_admission=False,
        scope='all actual calibration checkpoints; diagnostic comparison, not action selection or independent outcome')


def analyze(frames,values):
    z=np.asarray(values,dtype=float)
    require(len(frames)==24 and z.shape==(24,192) and np.isfinite(z).all(),'complete finite all24')
    rows=[]
    for i,(f,v) in enumerate(zip(frames,z)):
        require(f['order']==i and f['step']==(0,5,10)[i%3],'fixed complete chronological census')
        q=np.asarray(f['q']);g=np.asarray(f['goal']);pred=np.asarray(f['predicted_native192'])
        require(q.shape==(192,192) and pred.shape==g.shape==(192,) and
            all(np.isfinite(a).all() for a in (q,g,pred)),'native fixed goal/Q')
        rows.append(dict(root_sha256=f['root_sha256'],arm=f['arm'],step=f['step'],
            native_prediction_mse=float(np.mean((pred-v)**2)),
            predicted_Q_mse=float(np.mean((q@(pred-g))**2)),
            actual_Q_mse=float(np.mean((q@(v-g))**2)),physical_diagnostic=f['physical_diagnostic']))
    return dict(rows=rows,frames=24,training=False,new_Q=False,P1_admission=False,
        scope='descriptive calibration model/metric/physics attribution only; no paired independent efficacy claim')


def prepare(output):
    output=Path(output).absolute();require(not output.exists(),'new prepare')
    require(shutil.disk_usage(output.parent).free>storage.FLOOR+storage.CAP,'full reserve')
    p=collect();output.mkdir();storage.write(output/'protocol.json',p)
    storage.write(output/'prepare.json',dict(protocol_sha256=io.sha(output/'protocol.json')))
    return io.sha(output/'prepare.json')


def verify(folder,expected):
    folder=Path(folder);require(io.sha(folder/'prepare.json')==expected,'external prepare SHA')
    r=physical.read(folder/'prepare.json');require(io.sha(folder/'protocol.json')==r['protocol_sha256'],'protocol SHA')
    p=physical.read(folder/'protocol.json');require(p==collect(),'all input/code/model reconstruction');return p


def run(folder,expected,output):
    p=verify(folder,expected);output=Path(output).absolute()
    require(not output.exists() and shutil.disk_usage(output.parent).free>storage.FLOOR+storage.CAP,'new output/full reserve')
    output.mkdir();values=[];events=[];errors=[];model=None;before=after=None;host=None
    with (output/'image.events.jsonl').open('x') as journal:
        def emit(e):
            raw=storage.catalog.canonical(e)
            require(sum(f.stat().st_size for f in output.iterdir())+len(raw)<storage.CAP-250000,'output cap')
            require(shutil.disk_usage(output).free>storage.FLOOR+len(raw),'disk floor')
            events.append(e);journal.write(raw.decode());journal.flush();os.fsync(journal.fileno())
        try:
            import torch
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
            host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
            torch.backends.cuda.matmul.allow_tf32=False
            model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
            io.frozen(model);before=io.tensor_hash(model)
            encode=storage.screen.native_image_encoder(model,device,p0.image_transform(),emit)
            for f in p['frames']:
                image,_=io.pixels(f['path'],f['file_sha256'],f['pixel_sha256'])
                emit(dict(kind='frame_started',order=f['order'],root_sha256=f['root_sha256'],arm=f['arm'],step=f['step']))
                values.append(encode(image).tolist());torch.cuda.synchronize(device)
                emit(dict(kind='frame_completed',order=f['order']))
            storage.write(output/'analysis.json',analyze(p['frames'],values))
        except BaseException as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)))
        finally:
            if model is not None:
                try:io.frozen(model);after=io.tensor_hash(model)
                except BaseException as exc:errors.append(dict(stage='final_model',message=str(exc)))
            try:verify(folder,expected)
            except BaseException as exc:errors.append(dict(stage='final_source',message=str(exc)))
            counts={k:dict(attempted=sum(e.get('samples',0) for e in events if e['kind']=='native_image_attempted' and e['resource']==k),
                completed=sum(e['counts'][k] for e in events if e['kind']=='native_image_completed')) for k in ('encoder','projector')}
            complete=(not errors and len(values)==24 and before is not None and before==after and
                counts=={k:dict(attempted=24,completed=24) for k in counts})
            result=dict(status='all24_physical_checkpoints_encoded' if complete else 'partial_or_failed',errors=errors,
                resources=counts,host=host,model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
                prepare_sha256=expected,new_predict_calls=0,new_physical_steps=0,training=False,new_Q=False,P1_admission=False)
            storage.write(output/'report.json',result)
    require(complete,'partial checkpoint encoding; preserve artifacts, no retry');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha');parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    result=prepare(a.output) if a.command=='prepare' else (verify(a.prepared,a.prepare_sha)['schema'] if a.command=='verify' else run(a.prepared,a.prepare_sha,a.output))
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
