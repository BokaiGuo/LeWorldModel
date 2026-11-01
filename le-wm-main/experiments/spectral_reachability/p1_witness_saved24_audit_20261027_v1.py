"""Independently recompute original-Q metrics from saved CUDA latent arrays only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import p1_witness_rgb_cuda_main_20261027_v1 as acquisition

TAG='p1_witness_saved24_audit_20261027_v1'
PROJECT=Path(__file__).resolve().parents[2]
RUN=PROJECT/'results/p1_witness_rgb_cuda_main_20261027_v1_cuda_main01'
PREPARED=PROJECT/'results/p1_witness_rgb_cuda_main_20261027_v1_prepared01'
REPORT_SHA='255acbd9e7bdc66df8fced64120a13787520a821832a28a9c2fe6b5e4ed60da6'
ANALYSIS_SHA='4f37ac7744d552e570b799ab4256c3a63ab0bea8ebe368a5a9d83d487a84bab0'
EVENTS_SHA='d5350184c19207ef4c3cbdd30af02c3303a335436b6bddffc3fd74ec4c505c9b'
PREPARE_SHA='ce6b4971fe748718b4e685478c065d5b3ac2ca4a6f08a3d1d9503f555ed54d3f'
require=acquisition.require
CAP=512*1024
FLOOR=200*1024**2


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def keyed(rows):
    out={}
    for row in rows:
        key=(row['root_sha256'],row['arm'],row['step'])
        require(key not in out,'duplicate root/arm/checkpoint')
        out[key]=row
    return out


def recompute(frames,events):
    require(len(frames)==24 and len(keyed(frames))==24 and len(events)==120,'all24 census')
    roots={f['root_sha256'] for f in frames}
    require(len(roots)==2 and set(keyed(frames))=={(r,a,s) for r in roots
        for a in ('zero','full_H2_GN','full_H2_gradient','two_stage') for s in (0,5,10)},'fixed two-root four-arm three-checkpoint panel')
    rows=[];latents={}
    for order,f in enumerate(frames):
        start,encoder,projector,completed,end=events[5*order:5*order+5]
        require(start==dict(kind='frame_started',order=order,root_sha256=f['root_sha256'],arm=f['arm'],step=f['step'])
            and f['order']==order and end==dict(kind='frame_completed',order=order),'frame-scoped chronological mapping')
        for e,name in ((encoder,'encoder'),(projector,'projector')):
            require(e==dict(kind='native_image_attempted',resource=name,samples=1),'actual module attempt')
        require(completed['kind']=='native_image_completed' and completed['counts']==dict(encoder=1,projector=1)
            and completed['rgb_sha256']==f['pixel_sha256'],'native raw-array pixel binding')
        z,p,g,q=(np.asarray(x,dtype=np.float64) for x in
            (completed['latent'],f['predicted_native192'],f['goal'],f['q']))
        require(z.shape==p.shape==g.shape==(192,) and q.shape==(192,192)
            and all(np.isfinite(x).all() for x in (z,p,g,q)),'finite native shape')
        # Explicit quadratic forms, independent of producer.analyze's mean-square implementation.
        ep=q@(p-g);ea=q@(z-g);error=p-z
        row=dict(root_sha256=f['root_sha256'],arm=f['arm'],step=f['step'],
            predicted_Q_mse=float(ep@ep/192),actual_Q_mse=float(ea@ea/192),
            native_prediction_mse=float(error@error/192))
        rows.append(row);latents[(f['root_sha256'],f['arm'],f['step'])]=z
    initials=[]
    bykey=keyed(frames)
    for root in sorted(roots):
        zero=latents[root,'zero',0];pixel=bykey[root,'zero',0]['pixel_sha256']
        group=[f for f in frames if f['root_sha256']==root and f['step']==0]
        exact=all(np.array_equal(latents[root,f['arm'],0],zero) for f in group)
        samepixels=all(f['pixel_sha256']==pixel for f in group)
        require(samepixels,'initial RGB differs between arms')
        initials.append(dict(root_sha256=root,initial_pixel_identical=samepixels,initial_latents_bitwise_identical=exact,
            max_abs_initial_difference=max(float(np.max(np.abs(latents[root,f['arm'],0]-zero))) for f in group)))
    return rows,initials


def audit():
    bindings={RUN/'report.json':REPORT_SHA,RUN/'analysis.json':ANALYSIS_SHA,
              RUN/'image.events.jsonl':EVENTS_SHA,PREPARED/'prepare.json':PREPARE_SHA}
    require(all(sha(p)==h for p,h in bindings.items()),'pinned actual input bytes')
    p=acquisition.verify(PREPARED,PREPARE_SHA)  # pure source/asset reconstruction, no loader/run
    report=read(RUN/'report.json');analysis=read(RUN/'analysis.json')
    require(report['status']=='all24_physical_checkpoints_encoded' and report['errors']==[] and
        report['model_unchanged'] and report['model_hash_before']==report['model_hash_after'] and
        report['prepare_sha256']==PREPARE_SHA and report['resources']=={k:dict(attempted=24,completed=24) for k in ('encoder','projector')}
        and report['new_predict_calls']==report['new_physical_steps']==0,'actual resources/frozen model')
    require(p['encoder_cap']==p['projector_cap']==24 and p['AD_cap']==p['predict_cap']==p['physical_steps_cap']==0,'declared caps')
    rows,initials=recompute(p['frames'],[json.loads(line) for line in (RUN/'image.events.jsonl').read_text().splitlines()])
    actual=keyed(rows);saved=keyed(analysis['rows']);require(actual.keys()==saved.keys(),'saved analysis census')
    for key,row in actual.items():
        for field in ('predicted_Q_mse','actual_Q_mse','native_prediction_mse'):
            require(np.isclose(row[field],saved[key][field],rtol=0,atol=2e-12),'independent metric '+field)
    # Independently join the physical checkpoints by root/arm/step, never zip branch arrays.
    physical=read(acquisition.RUN/'report.json');source_rows=keyed([
        dict(root_sha256=r['root_sha256'],arm=r['arm'],**c) for r in physical['rows'] for c in r['checkpoints']])
    require(source_rows.keys()==actual.keys(),'all24 physical mapping')
    for f in p['frames']:
        c=source_rows[f['root_sha256'],f['arm'],f['step']]
        require(c['file_sha256']==f['file_sha256'] and c['rgb_sha256']==f['pixel_sha256']
                and Path(c['path']).resolve()==Path(f['path']).resolve(),'physical PNG acquisition mapping')
    gains=[]
    for root in sorted({r['root_sha256'] for r in rows}):
        two=actual[root,'two_stage',10]
        for arm in ('zero','full_H2_GN','full_H2_gradient'):
            peer=actual[root,arm,10]
            gains.append(dict(root_sha256=root,comparator=arm,
                predicted_Q_improvement=peer['predicted_Q_mse']-two['predicted_Q_mse'],
                actual_Q_improvement=peer['actual_Q_mse']-two['actual_Q_mse']))
    require(all(sha(path)==h for path,h in bindings.items()),'inputs changed during audit')
    return dict(schema=TAG+'.report',status='saved24_arithmetic_checks_completed',rows=rows,
        initial_duplicates=initials,terminal_two_stage_improvement=gains,
        resources=report['resources'],physical_mapping_checks=24,
        input_bindings={str(k):v for k,v in bindings.items()},audit_source_sha256=sha(__file__),
        P1_admission=False,new_neural_calls=0,new_physical_steps=0,
        scope='Original inherited Q/goal only. Same roots and outcomes, not independent efficacy confirmation; no action selection or tuning.',
        limitation='Saved latent arrays and invocation receipts are checked; no independent model re-encoding or weight execution occurred.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    path=parser.parse_args().output.resolve()
    require(TAG in path.name and not path.exists(),'exclusive new output')
    result=audit();raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    require(len(raw)<=CAP and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'small output / 200MiB floor')
    with path.open('xb') as stream: stream.write(raw);stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(path),sha256=sha(path),status=result['status'])))


if __name__=='__main__':main()
