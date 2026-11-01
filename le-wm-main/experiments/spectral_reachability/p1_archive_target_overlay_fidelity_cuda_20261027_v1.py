"""MAIN-only paired rendering observation effect; no goal, planner or admission.

Explicit externally pinned overlay producer report/prepare/source; 30 independent
visual encodings, 24 predictions, zero AD. No environment or neural CPU fallback.
"""
from __future__ import annotations
import argparse
from collections.abc import Mapping
from copy import deepcopy
import gzip
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_archive_target_overlay_probe_20261027_v1 as producer
from experiments.spectral_reachability import p1_constructive_offline_fidelity_cuda_20261027_v2 as fidelity
from experiments.spectral_reachability import p1_constructive_blind_planners_cuda_20261027_v1 as sealed

NAME='p1_archive_target_overlay_fidelity_cuda_20261027_v1'
ROOT,enc=sealed.ROOT,fidelity.enc
require,sha,canonical=sealed.require,sealed.sha,sealed.canonical
EPISODES=(1,6229)
STEPS=(0,5,10,15,20)
CONDITIONS=('archive','nativeoff','nativeon')
KINDS=('H1_10_to_15','fresh_H2_10_to_15','fresh_H2_15_to_20','teacher_forced_15_to_20')
CAP,FLOOR,FINAL=8*1024**2,200*1024**2,512*1024
PREPARE_CAP=256*1024
REQUEST_KEYS={'episode_ordinal','condition','kind','latent_history','raw_packets','tokens'}


def contract():
    return dict(episodes=list(EPISODES),conditions=list(CONDITIONS),checkpoints=list(STEPS),
        order='episode then condition then checkpoint / prediction kind',prediction_kinds=list(KINDS),
        counts=dict(encoder=30,projector=30,predict=24,predictor=24,action_encoder=24,pred_proj=24),
        raw='original archive float32 raw[0:20]; no clip/tanh/rescale or /100',normalizer='original public frozen mean/std',
        context='states0/5/10, raw0:5/5:10/10:15; H2 rolls own predicted15; teacher forced uses own actual15',
        scoring='same-condition native192 MSE/persistence only; no goals/Q/selection/fit',
        no_dedup=True,offreplay='validation-only duplicate, NOT one of30 encoding samples; exact file/pixel/fullsnapshot checks',
        no_retry=True,AD=0,physics=0,training=False,Q_fit=False,CPU_neural=False,
        SDP='MATH',matmul_TF32=False,cudnn_TF32=False,independent_confirmation=False,P1_admission=False,
        training_lineage_verified=False,ranking_repaired_claim=False,scope='paired rendering observation effect, fixed factual calibration',
        wall_seconds=600,allocated_bytes=3*1024**3,reserved_bytes=4*1024**3,
        output_cap_bytes=CAP,disk_floor_bytes=FLOOR,prepare_cap_bytes=PREPARE_CAP,final_reserve_bytes=FINAL,
        array_storage='original NPY bytes losslessly gzip; all full native module input/output tensor leaves')


def read(path,expected=None,cap=2*1024**2):
    path=Path(path);require(path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'bounded regular JSON')
    before=sha(path);require(expected is None or before==expected,'external JSON SHA')
    value=json.loads(path.read_bytes());require(sha(path)==before,'JSON TOCTOU');return value


def array_id(a):return producer.prior.array_id(a)


def bits(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype.str==b.dtype.str and a.shape==b.shape and a.tobytes()==b.tobytes()


def image(a):
    require(isinstance(a,np.ndarray) and a.dtype==np.uint8 and a.shape==(224,224,3),'unaltered RGB224 uint8');return a


def journal(raw,tail):
    require(len(raw)<=16*1024**2 and raw.endswith(b'\n'),'bounded complete producer journal')
    events=[json.loads(line) for line in raw.splitlines()];previous=None
    for k,e in enumerate(events):
        require(e['index']==k and e['previous_sha256']==previous and
            e['event_sha256']==producer.digest({a:v for a,v in e.items() if a!='event_sha256'}),'producer event hash chain')
        previous=e['event_sha256']
    require(events and previous==tail,'producer report journal tail');return events


def assemble(p,r,events,payload,decode_frame):
    """Pure explicit producer schema adapter; labels/geometry never model inputs."""
    require(p['schema']==producer.NAME+'.prepare' and p['contract']==producer.contract() and
        p['selection_fixed_before_execution'] is True and p['archive_future_state_for_evaluation_only'] is True,
        'fixed producer prepare, not old alignment schema')
    require(r['schema']==producer.NAME+'.report' and r['status']=='overlay_probe_processed' and r['prepare_id']==producer.digest(p) and
        r['contract']==p['contract'] and r['errors']==[] and r['first_error'] is None and r['inputs_final_verified'] is True and
        r['fixed_denominator']==2 and r['expected_frames']==30 and r['new_neural_calls']==0 and r['P1_admission'] is False and
        r['scientific_winner'] is None and r['approved_probe_report_sha256']==producer.REPORT_SHA and
        r['approved_probe_prepare_sha256']==producer.PREPARE_SHA,'complete actual full20 overlay report')
    require(r['counts']=={k:dict(attempted=v,completed=v) for k,v in p['contract']['caps'].items()},'complete actual physical census')
    require(events[0]['kind']=='run_started' and events[0]['prepare_id']==r['prepare_id'] and events[0]['contract']==p['contract'] and
        events[-1]['kind']=='run_finished' and events[-1]['status']==r['status'] and events[-1]['counts']==r['counts'] and
        events[-1]['errors']==[],'producer durable begin/end')
    require(len(payload)==len(r['rows'])==2 and [a['episode_ordinal'] for a in r['rows']]==list(EPISODES),'both original episodes in order')
    counts={k:dict(attempted=0,completed=0) for k in r['counts']};stack=[]
    for e in events:
        if e['kind'] in ('operation_attempted','operation_completed'):
            k=e['resource'];counts[k]['attempted' if e['kind']=='operation_attempted' else 'completed']+=1
            if e['kind']=='operation_attempted':stack.append(k)
            else:require(stack and stack.pop()==k,'producer nested operation balance')
    require(not stack and counts==r['counts'],'actual physical attempts/returns, including reset internals')
    result=[]
    for episode,row,data in zip(EPISODES,r['rows'],payload,strict=True):
        require(row['status']=='overlay_full20_completed' and row['steps_completed']==20 and row['missing_checkpoints']==[] and
            len(row['frames'])==15 and len(row['off_replay_checks'])==5,'full fixed denominator; no partial padding')
        raw=data['actions'];archive=data['comparison_pixels']
        require(raw.dtype==np.float32 and raw.shape==(20,2) and np.isfinite(raw).all() and
            archive.dtype==np.uint8 and archive.shape==(5,224,224,3),'literal fixed archive slices')
        finished=[e['receipt'] for e in events if e['kind']=='branch_finished' and e['receipt']['episode_ordinal']==episode]
        frames=[e['receipt'] for e in events if e['kind']=='fresh_overlay_frame' and e['episode_ordinal']==episode]
        transitions=[e for e in events if e['kind']=='native_return' and e['episode_ordinal']==episode]
        require(finished==[row] and frames==row['frames'] and [e['step'] for e in transitions]==list(range(1,21)), 'durable branch/frames/full20 commands')
        for j,e in enumerate(transitions):
            command=raw[j];require(e['requested_raw']==e['actual_raw']==array_id(command) and
                e['requested_float32_bits']==e['actual_raw_bits']==command.tobytes().hex() and
                (j==19 or (e['terminated'] is False and e['truncated'] is False)),'literal source raw commands; no premature continuation')
        require(transitions[-1]['terminated']==row['terminated'] and transitions[-1]['truncated']==row['truncated'],'terminal native flags retained')
        off=[];on=[];goal_config=None
        for j,step in enumerate(STEPS):
            triplet=row['frames'][3*j:3*j+3];arrays=[];first=triplet[0]['pre']
            require(set(first)=={'snapshot','goal_state','goal_pose','goal_rgb','renderer'},'exact invariant fields')
            stable={k:v for k,v in first.items() if k!='snapshot'}
            if goal_config is None:goal_config=stable
            require(stable==goal_config and first['goal_state']==row['native_goal'] and first['goal_pose']==row['goal_pose'] and
                first['renderer']==row['renderer'],'same native goal and renderer across checkpoints')
            require(first['snapshot']==(row['placed_snapshot'] if step==0 else transitions[step-1]['snapshot']), 'checkpoint actual state/snapshot at original time')
            for k,(f,flag) in enumerate(zip(triplet,(False,True,False),strict=True)):
                name=f'e{episode}_s{step:02}_{k}_'+('on' if flag else 'off')+'.png'
                require(f['step']==step and f['overlay_index']==k and type(f['overlay_index']) is int and f['with_target'] is flag and
                    f['png']==name and f['pre']==f['post']==first,'explicit frame identity and invariant full snapshot')
                a=image(decode_frame(f));require(array_id(a)==f['pixels'],'decoded file pixels receipt')
                require(f['comparison']==producer.compare(a,archive[j]),'unaltered archive comparison arithmetic')
                if k==1:require(f['old_off_match'] is None,'on frame is not old off')
                else:require(f['old_off_match']==dict(file=True,pixels=True,state=True,snapshot=True),'original lag0 off full binding')
                require(f['off_replay_match'] is (True if k==2 else None),'offreplay receipt role')
                arrays.append(a)
            require(bits(arrays[0],arrays[2]) and triplet[0]['png_sha256']==triplet[2]['png_sha256'] and
                triplet[0]['pixels']==triplet[2]['pixels'] and row['off_replay_checks'][j]==
                dict(step=step,first_frame_index=3*j,last_frame_index=3*j+2,bit_exact=True),'off replay exact bytes/pixels and frame indices')
            off.append(arrays[0]);on.append(arrays[1])
        result.append(dict(episode_ordinal=episode,raw_actions=raw.copy(),
            frames=dict(archive=archive.copy(),nativeoff=np.stack(off),nativeon=np.stack(on))))
    return result


def material(source_report,source_sha,producer_source_sha,producer_prepare_sha):
    args=dict(source_report=str(Path(source_report).absolute()),source_sha=source_sha,
        producer_source_sha=producer_source_sha,producer_prepare_sha=producer_prepare_sha)
    for h in (source_sha,producer_source_sha,producer_prepare_sha):enc.core.hash_value(h)
    require(sha(producer.__file__)==producer_source_sha,'externally pinned final producer source')
    rp=Path(args['source_report']);r=read(rp,source_sha)
    bp=rp.parent/'execution_binding.json';binding=read(bp)
    require(set(binding)=={'prepared','prepare_file_sha256'} and binding['prepare_file_sha256']==producer_prepare_sha,'explicit producer execution binding')
    pp=Path(binding['prepared']);require(pp.is_absolute(),'absolute producer prepare path');p=read(pp,producer_prepare_sha)
    payload=producer.verify(p)  # Saved-only source/slice/old-PNG validation; NEVER producer.execute.
    bindings=dict(p['bindings']);bindings.update({str(rp):source_sha,str(pp):producer_prepare_sha,str(bp):sha(bp),str(Path(producer.__file__).resolve()):producer_source_sha})
    jp=rp.parent/'journal.jsonl';require(jp.is_file() and not jp.is_symlink() and jp.stat().st_size<=16*1024**2,'bounded producer journal')
    bindings[str(jp)]=sha(jp);events=journal(jp.read_bytes(),r['journal_chain_tail'])
    def decode(f):
        require(Path(f['png']).name==f['png'],'flat producer PNG');path=rp.parent/f['png']
        require(path.is_file() and not path.is_symlink(),'regular nonsymlink producer PNG')
        a=producer.old_png(path,f['png_sha256']);bindings[str(path)]=f['png_sha256'];return a
    data=assemble(p,r,events,payload,decode)
    require({f.name for f in rp.parent.iterdir()}=={'report.json','execution_binding.json','journal.jsonl'}|
        {f['png'] for row in r['rows'] for f in row['frames']},'exact complete producer output census')
    files=sealed.source_files()+[Path(fidelity.__file__),Path(fidelity.old.__file__),Path(__file__),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    bindings.update({str(Path(f).resolve()):sha(f) for f in files});bindings.update(enc.MODELS)
    normalization=enc.native_metadata()
    for path,h in bindings.items():require(sha(path)==h,'final source/slice/PNG/model binding '+path)
    inputs=[dict(episode_ordinal=d['episode_ordinal'],raw_actions=array_id(d['raw_actions']),
        frames={c:[array_id(a) for a in d['frames'][c]] for c in CONDITIONS}) for d in data]
    own=dict(schema=NAME+'.prepare',arguments=args,contract=contract(),bindings=bindings,input_receipts=inputs,
        producer_prepare_id=r['prepare_id'],normalization=normalization,policy=str(enc.POLICY),expected_model_hash=enc.MODEL_TENSOR_SHA)
    require(len(canonical(own))<PREPARE_CAP,'prepare exceeds256KiB: report design problem, never raise limit')
    return own,data


def prepare(output,**arguments):
    p,_=material(**arguments);b=canonical(p);path=Path(output).absolute()
    require(not path.exists() and shutil.disk_usage(path.parent).free>=FLOOR+len(b),'exclusive prepare +200MiB floor')
    with path.open('xb') as f:require(f.write(b)==len(b),'short prepare');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(b))


def verify(prepared,expected):
    p=read(prepared,expected,PREPARE_CAP);current,data=material(**p['arguments'])
    require(p==current,'entire own prepare reconstruction');return p,data


class Store(sealed.Store):
    def _write(self,name,raw,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..') and not (self.output/name).is_symlink(),'flat nonsymlink output')
        used=sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(raw),'8MiB incl partial/final +200MiB floor')
        with (self.output/name).open('ab' if append else 'xb') as f:require(f.write(raw)==len(raw),'short durable write');f.flush();os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)


def save_array(store,name,a):
    a=np.asarray(a);require(a.dtype in (np.dtype('uint8'),np.dtype('float32')) and np.isfinite(a).all(),'original RGB or native float32 arrays')
    b=BytesIO();np.save(b,a,allow_pickle=False);raw=b.getvalue();compressed=gzip.compress(raw,compresslevel=6,mtime=0)
    name=name+'.npy.gz';store._write(name,compressed)
    return dict(file=name,sha256=enc.digest(compressed),npy_sha256=enc.digest(raw),raw_sha256=array_id(a)['sha256'],shape=list(a.shape),dtype=a.dtype.str)


def prediction_request(episode,condition,kind,zs,raw,prefix_prediction=None):
    require(episode in EPISODES and condition in CONDITIONS and kind in KINDS and len(zs)==5,'fixed request identity')
    for z in zs:enc.native(z)
    require(raw.dtype==np.float32 and raw.shape==(20,2) and np.isfinite(raw).all(),'original20 raw actions')
    if kind in KINDS[:2]:history=np.stack(zs[:3]);packets=raw[:15]
    else:
        last=zs[3] if kind==KINDS[3] else prefix_prediction
        enc.native(last);history=np.stack((zs[1],zs[2],last));packets=raw[5:20]
    packets=packets.reshape(3,5,2).copy()
    return dict(episode_ordinal=episode,condition=condition,kind=kind,latent_history=history,
        raw_packets=packets,tokens=fidelity.normalized(packets))


def mse(a,b):return float(np.mean((np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))**2))


def execute(data,encode,predict,store,images,rows):
    """Array-only scheduler; backend requests never carry a target or another condition."""
    require(len(data)==2 and len(images)==30 and len(rows)==24,'all fixed denominators')
    latent={};outputs=[]
    for slot,d in enumerate(data):
        require(d['episode_ordinal']==EPISODES[slot],'fixed source order')
        save_array(store,f'episode_{d["episode_ordinal"]}.raw20',d['raw_actions'])
        for ci,c in enumerate(CONDITIONS):
            zs=[]
            for j,step in enumerate(STEPS):
                index=15*slot+5*ci+j;row=images[index];row['status']='attempted'
                row['pixels']=save_array(store,f'image_{index:02d}.pixels',image(d['frames'][c][j]))
                store.event(dict(kind='image_attempt',receipt=deepcopy(row)))
                z=encode(d['frames'][c][j].copy(),dict(index=index,episode_ordinal=d['episode_ordinal'],condition=c,step=step))
                enc.native(z);row.update(status='encoded',latent=save_array(store,f'image_{index:02d}.native192',z))
                store.event(dict(kind='image_saved',receipt=deepcopy(row)));zs.append(z.copy())
            latent[d['episode_ordinal'],c]=zs
    for slot,d in enumerate(data):
        ep=d['episode_ordinal']
        for ci,c in enumerate(CONDITIONS):
            zs=latent[ep,c];prefix=None;independent=None
            for k,kind in enumerate(KINDS):
                index=12*slot+4*ci+k;row=rows[index];row['status']='attempted'
                request=prediction_request(ep,c,kind,zs,d['raw_actions'],prefix)
                row['identity']={a:request[a] for a in ('episode_ordinal','condition','kind')}
                row['inputs']={a:save_array(store,f'request_{index:02d}.{a}',request[a]) for a in ('latent_history','raw_packets','tokens')}
                store.event(dict(kind='prediction_attempt',receipt=deepcopy(row)))
                value=predict(deepcopy(request),index)
                require(isinstance(value,np.ndarray) and value.dtype==np.float32 and value.shape==(3,192) and np.isfinite(value).all(),'full native3x192 prediction')
                row.update(status='saved',output=save_array(store,f'request_{index:02d}.prediction',value))
                store.event(dict(kind='prediction_saved',receipt=deepcopy(row)))
                if k==0:independent=value.copy()
                if k==1:
                    require(bits(value,independent),'fresh H2 first prediction must exactly reproduce independent H1, no tolerance/retry')
                    prefix=value[-1].copy()
                outputs.append(dict(**row['identity'],last=value[-1].copy(),last_input=request['latent_history'][-1].copy()))
    store.event(dict(kind='all24_predictions_durable_before_scoring'))
    scores=[]
    for out in outputs:
        zs=latent[out['episode_ordinal'],out['condition']];target=zs[3 if out['kind'] in KINDS[:2] else 4]
        scores.append(dict(**{k:out[k] for k in ('episode_ordinal','condition','kind')},
            actual_successor_step=15 if out['kind'] in KINDS[:2] else 20,
            MSE=mse(out['last'],target),persistence_last_input_MSE=mse(out['last_input'],target),
            persistence_origin10_MSE=mse(zs[2],target)))
    distances=[dict(episode_ordinal=ep,step=s,nativeoff_nativeon_MSE=mse(latent[ep,'nativeoff'][j],latent[ep,'nativeon'][j]))
        for ep in EPISODES for j,s in enumerate(STEPS)]
    return dict(scope='paired rendering observation effect',scores=scores,paired_latent_distances=distances,
        training_lineage_verified=False,ranking_repaired_claim=False,P1_admission=False)


def census():
    images=[dict(index=15*s+5*ci+j,episode_ordinal=ep,condition=c,step=step,status='not_started')
        for s,ep in enumerate(EPISODES) for ci,c in enumerate(CONDITIONS) for j,step in enumerate(STEPS)]
    rows=[dict(index=12*s+4*ci+k,episode_ordinal=ep,condition=c,kind=kind,status='not_started')
        for s,ep in enumerate(EPISODES) for ci,c in enumerate(CONDITIONS) for k,kind in enumerate(KINDS)]
    return images,rows


class NativeTrace:
    """Read-only native tensor capture; actual attempts precede dispatch and disk saves."""
    def __init__(self,model,store,sync,inference):
        self.model,self.store,self.sync,self.inference=model,store,sync,inference
        self.entries=[];self.hooks=[];self.pending={};self.active=None;self.failed=False;self.predict_original=None
    def tree(self,value,stem):
        if hasattr(value,'detach'):
            require(getattr(getattr(value,'device',None),'type',None)=='cuda','CUDA-only native module tensor')
            a=value.detach().cpu().numpy();require(a.dtype==np.float32 and np.isfinite(a).all(),'full finite native float32 tensor')
            return dict(tensor=save_array(self.store,stem,a))
        if isinstance(value,Mapping):return {str(k):self.tree(v,stem+'.'+str(k)) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [self.tree(v,stem+f'.{k}') for k,v in enumerate(value)]
        require(value is None or type(value) in (bool,int,float,str),'explicit native structured output');return value
    def begin(self,kind,args,kwargs):
        require(self.active is not None and not self.failed and self.inference(),'scoped frozen inference; no retry')
        phase,index,expected=self.active
        order=('encoder','projector') if phase=='image' else ('action_encoder','predict','predictor','pred_proj')
        own=[e for e in self.entries if e['phase']==phase and e['sample']==index]
        require(len(own)<len(order) and kind==order[len(own)],'exact native call order/cap')
        e=dict(index=len(self.entries),phase=phase,sample=index,resource=kind,completed=False)
        self.entries.append(e);self.store.event(dict(kind='native_module_attempt',entry=deepcopy(e)))
        e['inputs']=self.tree(args,f'module_{e["index"]:03d}.inputs');e['kwargs']=self.tree(kwargs,f'module_{e["index"]:03d}.kwargs')
        if phase=='prediction' and kind in ('action_encoder','predict'):
            want=expected['tokens'] if kind=='action_encoder' else expected['latent_history']
            require(bits(args[0].detach().cpu().numpy(),want[None]),'actual full3 context/tokens exactly requested')
        self.store.event(dict(kind='native_inputs_saved',entry=deepcopy(e)));return e
    def end(self,e,out):
        if out is None:self.failed=True;return
        self.sync();e['output']=self.tree(out,f'module_{e["index"]:03d}.output');e['completed']=True
        self.store.event(dict(kind='native_module_return',entry=deepcopy(e)))
    def pre(self,kind):
        def before(module,args,kwargs):
            e=self.begin(kind,args,kwargs);self.pending[module]=e
        return before
    def post(self,module,args,kwargs,out):
        e=self.pending.pop(module,None)
        if e is not None:self.end(e,out)
    def __enter__(self):
        try:
            for kind in ('encoder','projector','action_encoder','predictor','pred_proj'):
                module=getattr(self.model,kind)
                self.hooks.append(module.register_forward_pre_hook(self.pre(kind),with_kwargs=True))
                self.hooks.append(module.register_forward_hook(self.post,with_kwargs=True,always_call=True))
            self.predict_owned='predict' in self.model.__dict__;self.predict_original=self.model.predict
            def predict(*args,**kwargs):
                e=self.begin('predict',args,kwargs)
                try:
                    value=self.predict_original(*args,**kwargs);self.end(e,value);return value
                except BaseException:self.failed=True;raise
            self.model.predict=predict;return self
        except BaseException as e:self.__exit__(type(e),e,e.__traceback__);raise
    def __exit__(self,typ,exc,tb):
        failures=[]
        if self.predict_original is not None:
            try:
                if self.predict_owned:self.model.predict=self.predict_original
                else:delattr(self.model,'predict')
            except BaseException as e:failures.append(e)
        for hook in reversed(self.hooks):
            try:hook.remove()
            except BaseException as e:failures.append(e)
        if failures:
            if exc is not None:exc.add_note('native hook cleanup: '+repr(failures))
            else:raise failures[0]
    def counts(self):
        return {k:dict(attempted=sum(e['resource']==k for e in self.entries),completed=sum(e['resource']==k and e['completed'] for e in self.entries)) for k in contract()['counts']}
    def finish(self,phase,index):
        own=[e for e in self.entries if e['phase']==phase and e['sample']==index]
        require(len(own)==(2 if phase=='image' else 4) and all(e['completed'] for e in own),'all actual native module returns')


def run(prepared,expected,output):
    p,data=verify(prepared,expected);store=Store(output);images,rows=census()
    errors=[];model=trace=None;before=after=host=None;source_ok=False;forbidden=[];limits=fidelity.Limits()
    def error(phase,e):errors.append(enc.error(phase,e))
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        store.put('input_receipts.json',p)
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        limits.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
        limits.free=lambda:shutil.disk_usage(store.output).free
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'same frozen checkpoint')
        transform=p0.image_transform();trace=NativeTrace(model,store,lambda:torch.cuda.synchronize(device),lambda:not torch.is_grad_enabled())
        def encode(rgb,meta):
            limits.check();trace.active=('image',meta['index'],None)
            with torch.inference_mode():
                x=p0._image_tensor(rgb.copy(),transform=transform,device=device)
                require(x.device.type=='cuda' and tuple(x.shape)==(1,1,3,224,224),'native independent CUDA RGB sample')
                z=model.encode({'pixels':x})['emb']
                require(z.device.type=='cuda' and tuple(z.shape)==(1,1,192) and z.dtype==torch.float32,'native projected CLS192')
                a=z[0,0].detach().cpu().numpy().copy()
            trace.finish('image',meta['index']);limits.check();return a
        def predict(r,index):
            require(set(r)==REQUEST_KEYS,'no targets, future labels, states or Q in predictor request')
            limits.check();trace.active=('prediction',index,r)
            with torch.inference_mode():
                z=torch.from_numpy(r['latent_history'].copy()).to(device)[None]
                tokens=torch.from_numpy(r['tokens'].copy()).to(device)[None]
                value=model.predict(z,model.action_encoder(tokens))
                require(value.device.type=='cuda' and tuple(value.shape)==(1,3,192) and value.dtype==torch.float32,'native full predicted3x192')
                a=value[0].detach().cpu().numpy().copy()
            trace.finish('prediction',index);limits.check();return a
        def ad_event(e):forbidden.append(deepcopy(e));store.event(e)
        with fidelity.old.no_ad(torch,ad_event,errors),sdpa_kernel(SDPBackend.MATH),trace:
            analysis=execute(data,encode,predict,store,images,rows)
            store.put('analysis.json',analysis)
        sealed.exact_counts(trace.counts(),contract()['counts']);require(not forbidden,'zero actual AD attempts')
    except BaseException as e:
        errors.insert(0,enc.error('execution',e))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e:error('error_journal',e)
    finally:
        for row in images+rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model)
            except BaseException as e:error('final_model',e)
        try:verify(prepared,expected);source_ok=True
        except BaseException as e:error('final_source',e)
        try:store.close()
        except BaseException as e:error('journal_finalize',e)
        hashes={}
        try:files=list(store.output.iterdir())
        except BaseException as e:error('artifact_listing',e);files=[]
        for f in files:
            try:require(f.is_file(),'flat actual artifact');hashes[f.name]=sha(f)
            except BaseException as e:error('artifact_hash',e)
        complete=not errors and source_ok and before is not None and before==after and all(x['status']=='encoded' for x in images) and all(x['status']=='saved' for x in rows)
        r=dict(schema=NAME+'.report',status='overlay_factual_fidelity_saved_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,contract=p['contract'],images=images,requests=rows,
            resources=trace.counts() if trace is not None else {k:dict(attempted=0,completed=0) for k in contract()['counts']},
            module_ledger=deepcopy(trace.entries) if trace is not None else [],forbidden_AD_attempts=forbidden,
            errors=errors,first_error=errors[0] if errors else None,artifact_sha256=hashes,source_verified=source_ok,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,
            host=host,limits=limits.receipt(),AD=0,physics=0,training=False,P1_admission=False,scientific_winner=None)
        try:store.put('report.json',r,final=True)
        except BaseException as e:error('final_report',e);complete=False;print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'failure preserved; no retry: '+str(errors));return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run'))
    for arg in ('source-report','source-sha','producer-source-sha','producer-prepare-sha','prepared','prepare-sha','output'):p.add_argument('--'+arg)
    a=p.parse_args()
    if a.command=='prepare':r=prepare(a.output,source_report=a.source_report,source_sha=a.source_sha,producer_source_sha=a.producer_source_sha,producer_prepare_sha=a.producer_prepare_sha)
    elif a.command=='verify':own,data=verify(a.prepared,a.prepare_sha);r=dict(status='verified_no_execution',episodes=len(data),contract=own['contract'])
    else:r=run(a.prepared,a.prepare_sha,a.output)
    print(canonical({k:v for k,v in r.items() if k not in ('images','requests','module_ledger')}).decode(),end='')


if __name__=='__main__':main()
