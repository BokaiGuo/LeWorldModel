"""Pure RAM six-slot RGB roster join. No I/O, upstream authentication or HOST run.

Caller must later authenticate full native and paid receipts before production use.
This helper cannot issue a ready flag or substitute for either strict consumer.
"""
from copy import deepcopy
import hashlib

import numpy as np

NAME='agent_p2_six_target_rgb_roster_20261029_v1'
NATIVE='agent_p2_six_target_physical_20261029_v1'
PAID='agent_p2_six_target_paid_replay_20261029_v1'
ARMS=('prior_FIRST','program_first_postcondition','arbitrary')
CP=(0,5,10,15)
FRAME_KEYS=('step','png','file_sha256','pixel_sha256','state','snapshot')


def need(value,message):
    if not value:raise ValueError(message)


def f32(x,shape):
    need(type(x) is np.ndarray and x.dtype==np.float32 and x.shape==shape and np.isfinite(x).all(),'original finite f32 shape')
    return x


def same(a,b,message):
    need(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),message)


def digest(x):return hashlib.sha256(x.tobytes()).hexdigest()


def vectors(plan,paid_row,tr,arrays):
    """Select only actual-token predictions, with complete replay-history joins."""
    need(paid_row['status']=='returned' and tr['completed'] is True and tr['errors']==[] and
         tr['backend']=='host_cuda_actual_tokens' and tr['blocked'] is None,'complete actual-token replay, never normalized fixture relabel')
    z=f32(arrays[tr['inputs']['z']],(1,3,192));a=f32(arrays[tr['inputs']['a']],(1,2,10))
    target=f32(arrays[tr['inputs']['target']],(1,192))
    for k,v in (('z',z),('a',a),('target',target)):
        same(np.asarray(plan['controller_payload'][k],np.float32),v,'original controller input '+k)
        need(plan['payload_sha256'][k]==digest(v),'complete payload bytes SHA')
    raw=np.asarray(plan['commands'],np.float32);f32(raw,(15,2))
    need((np.abs(raw)<=1).all() and raw.tobytes().hex()==plan['raw_float32_bits'] and digest(raw)==plan['raw_float32_sha256'],'raw15 exact ordered bytes/domain')
    ar=tr['arithmetic'];need(len(ar)==4 and [x['kind'] for x in ar]==['mul','add','sub','div'] and all(x['returned'] is True for x in ar),'actual arithmetic sequence')
    same(f32(arrays[ar[1]['output']],(15,2)),raw,'actual CUDA add output, not CPU inverse or mean')
    same(f32(arrays[tr['inputs']['sealed_raw']],(15,2)),raw,'sealed raw matches actual output')
    tokens=f32(arrays[tr['actual_tokens']],(1,1,3,10))
    need(len(tr['replays'])==2,'both paid full replays')
    w,s=tr['replays']
    need([x['kind'] for x in (w,s)]==['whole','segmented'] and all(x['returned'] is True for x in (w,s)),'whole and segmented returned')
    pred=f32(arrays[w['rollout']['prediction']],(1,1,3,192))
    same(pred,f32(arrays[s['rollout']['prediction']],(1,1,3,192)),'all whole/seg prediction bytes')
    for replay in (w,s):
        roll=replay['rollout'];need(len(roll['checkpoints'])==3,'all checkpoint histories')
        for j,c in enumerate(roll['checkpoints']):
            zz=f32(arrays[c['z']],(1,4+j,192));aa=f32(arrays[c['a']],(1,3+j,10))
            same(zz,np.concatenate((z,pred[0,:,:j+1,:]),axis=1),'full causal z history')
            same(aa,np.concatenate((a,tokens[0,:,:j+1,:]),axis=1),'full actual-token history')
        for k in ('z','a'):same(arrays[roll['final'][k]],arrays[roll['checkpoints'][-1][k]],'entire final context')
    return [z[0,-1],*(pred[0,0,j] for j in range(3))]


def assemble(prepared,native_receipt,paid_report,get_paid_trace):
    """Join already loaded dicts/callback arrays, NOT certify upstream authority.

    get_paid_trace(i) -> (actual trace dict, array-key -> original f32 ndarray).
    No scores, image loading, full-strict calls, model imports or durable writes.
    """
    need(prepared['schema']==NATIVE+'.prepare' and native_receipt['schema']==NATIVE+'.consume' and
         paid_report['schema']==PAID+'.report','real six schemas')
    need(native_receipt['fixed_denominator']==6 and native_receipt['errors']==[] and
         native_receipt['planner_authority']==prepared['planner_authority'],'whole native/prepare authority equality')
    args=prepared['arguments']
    need(paid_report['manifest_sha256']==args['replay_manifest_sha'] and paid_report['completed'] is True and
         paid_report['errors']==[],'same actual paid manifest, not old normalized report')
    need(len(prepared['plans'])==len(native_receipt['rows'])==len(paid_report['rows'])==6,'all six fixed denominator')
    out=[]
    for i,(p,n,pr) in enumerate(zip(prepared['plans'],native_receipt['rows'],paid_report['rows'],strict=True)):
        key=(i,i//3,'off',ARMS[i%3])
        need((p['index'],p['root_slot'],p['condition'],p['arm'])==key and
             (n['index'],n['root_slot'],n['condition'],n['arm'])==key and
             (pr['slot'],pr['root'],pr['condition'],pr['arm'])==key,'slot/root/time fixed join')
        need(n['plan']==p and n['task_id']==p['task_id'] and n['selection_digest']==p['selection_digest'],'complete plan/task seal join')
        need(n['status'] in ('horizon_completed','native_stopped') and n['initial_exact'] is True and
             n['first_error'] is None and n['secondary_errors']==[],'clean native source, not engineering failure as missing')
        need(type(n['own_returned']) is int and 0<=n['own_returned']<=15 and
             (n['status']!='horizon_completed' or n['own_returned']==15),'actual physical time')
        need(n['goal']==p['native_goal'],'task goal, never tracking target')
        responses=n['responses'];need([x['step'] for x in responses]==list(range(1,n['own_returned']+1)),'every actual response')
        for j,r in enumerate(responses):
            need(type(r['terminated']) is bool and type(r['truncated']) is bool and
                 (j==len(responses)-1 or not(r['terminated'] or r['truncated'])),'native early stop preserved')
        if responses:
            need((n['status']=='native_stopped')==(responses[-1]['terminated'] or responses[-1]['truncated']),'no arbitrary short horizon')
        trajectory=n['trajectory'];frames=[f for f in trajectory if f['phase']=='plan']
        need([f['step'] for f in frames]==list(range(n['own_returned']+1)) and
             all(f['overlay_index']==0 and f['condition']=='off' for f in frames),'one actual canonical off frame per step')
        cps=[{k:f[k] for k in FRAME_KEYS} for f in frames if f['step'] in CP]
        need(cps==n['checkpoints'] and n['missing_checkpoints']==[cp for cp in CP if cp>n['own_returned']],'no substituted or padded cp')
        tr,arrays=get_paid_trace(i);vs=vectors(p,pr,tr,arrays)
        for j,cp in enumerate(CP):
            frame=None if cp>n['own_returned'] else frames[cp]
            response=None if cp==0 or frame is None else responses[cp-1]
            if frame is not None:
                need(frame['snapshot']['state']==frame['state'] and (response is None or response['state']==frame['state']),'same-time RGB/fullstate/return')
                need(len(frame['state'])==7 and np.isfinite(np.asarray(frame['state'],np.float64)).all(),'native7 finite')
                need(frame['png'] not in ('','.','..') and '/' not in frame['png'] and '\\' not in frame['png'],'flat source PNG name')
            out.append(dict(index=4*i+j,slot=i,root=i//3,condition='off',arm=ARMS[i%3],cp_public_step=cp,
                status='available_unencoded' if frame is not None else 'missing_physical_checkpoint',
                reason=None if frame is not None else 'native_checkpoint_not_reached',
                frame=deepcopy(frame),response=deepcopy(response),native_goal=deepcopy(n['goal']),
                task_id=p['task_id'],selection_digest=p['selection_digest'],
                raw15_sha256=p['raw_float32_sha256'],raw_prefix_bits=np.asarray(p['commands'][:cp],np.float32).tobytes().hex(),
                predicted192=vs[j].tolist(),predicted_sha256=digest(vs[j]),
                prediction_origin='observed_history' if cp==0 else 'paid_actual_deployment_tokens',
                prediction_pointer=dict(trace=deepcopy(pr['trace']),array=tr['inputs']['z'] if cp==0 else tr['replays'][0]['rollout']['prediction'],
                    index=[0,2] if cp==0 else [0,0,j-1]),
                latent=None,metrics=None,scientific_admission=False))
    return dict(schema=NAME+'.roster',denominator=dict(slots=6,checkpoints=24),rows=out,
        encoder_samples_available=sum(x['frame'] is not None for x in out),
        upstream_full_authentication_performed=False,host_ready=False,new_neural_calls=0,new_physics_calls=0)
