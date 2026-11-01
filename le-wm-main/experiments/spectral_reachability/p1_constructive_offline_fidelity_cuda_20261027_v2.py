"""Stage A only: fixed dispersed recorded episodes, teacher-forced CUDA fidelity.

Prepare/verify are selected saved-data reads. Only MAIN may invoke run.
No environment, training, action optimization, target-conditioned selection or P1 admission.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import gzip
import importlib.metadata
from io import BytesIO
import os
from pathlib import Path
import shutil
import time
import zlib
import numpy as np
from experiments.spectral_reachability import p1_constructive_offline_fidelity_cuda_20261027_v1 as old

base, enc = old.base, old.enc
ROOT, DATA = old.ROOT, old.DATA
NAME = 'p1_constructive_offline_fidelity_cuda_20261027_v2'
STEPS, WINDOWS, CONDITIONS = (0, 5, 10, 15, 20), (10, 15), ('F', 'Z', 'M')
CAP, FLOOR, FINAL = 32*1024**2, 200*1024**2, 1024**2
WALL, ALLOCATED, RESERVED = 600, 3221225472, 4294967296
V1_SHA = 'fb73c3cb05f4dc2bcf47a6214fd3b67be8e875b5e666c01a15f11748d4b74b04'
require, sha, read, canonical = old.require, old.sha, old.read, old.canonical
version, array_receipt, png_bytes = old.version, old.array_receipt, old.png_bytes


def episode_indices(n):
    require(type(n) is int and n >= 4, 'at least four episode ordinals, no replacement')
    return (0, n//3, 2*n//3, n-1)


def slice_episode(f, index):
    n = len(f['ep_len']); rows = f['pixels'].shape[0]
    require(type(index) is int and index in episode_indices(n), 'fixed dispersed ordinal')
    require(f['ep_offset'].shape == f['ep_len'].shape == (n,), 'episode metadata shape')
    for key in ('ep_offset', 'ep_len', 'episode_idx', 'step_idx'):
        require(f[key].dtype.kind in 'iu', 'integer episode/time metadata')
    offset, length = int(f['ep_offset'][index]), int(f['ep_len'][index])
    following = int(f['ep_offset'][index+1]) if index+1 < n else rows
    require(length >= 21 and offset >= 0 and offset+length == following <= rows, 'full21 and exact next/end boundary')
    require((index == 0 and offset == 0) or (index > 0 and int(f['ep_offset'][index-1])+int(f['ep_len'][index-1]) == offset), 'previous boundary')
    for key in ('action', 'episode_idx', 'step_idx'):
        require(f[key].shape[0] == rows, 'aligned column rows')
    require(f['pixels'].shape[1:] == (224,224,3) and f['pixels'].dtype == np.uint8 and f['action'].shape[1:] == (2,) and f['action'].dtype == np.float32, 'native RGB/action layout')
    ids, steps = (np.asarray(f[k][offset:offset+21]) for k in ('episode_idx','step_idx'))
    require(ids.shape == steps.shape == (21,) and np.array_equal(ids, np.full(21,index)) and np.array_equal(steps,np.arange(21)), 'no crossing/time gaps')
    actions = np.asarray(f['action'][offset:offset+20])
    require(actions.dtype == np.float32 and actions.shape == (20,2) and np.isfinite(actions).all(), 'literal finite expert raw actions; NO action-ball restriction')
    frames = np.asarray(f['pixels'][[offset+s for s in STEPS]])
    require(frames.dtype == np.uint8 and frames.shape == (5,224,224,3), 'five unaltered RGB')
    record = dict(episode_index=index,offset=offset,length=length,next_or_end_offset=following,
        frame_local_indices=list(STEPS),frame_global_indices=[offset+s for s in STEPS],
        action_local_slice=[0,20],action_global_slice=[offset,offset+20],
        packets=[dict(local=[s,s+5],global_slice=[offset+s,offset+s+5]) for s in (0,5,10,15)],
        episode_idx=array_receipt(ids),step_idx=array_receipt(steps),actions=array_receipt(actions),
        raw_actions=actions.astype(float).tolist(),frames=[dict(step=s,pixels=array_receipt(a),png_sha256=enc.digest(png_bytes(a))) for s,a in zip(STEPS,frames,strict=True)])
    return record, dict(actions=actions,frames=frames)


def correlations(payload):
    """Descriptive only; never replace, deduplicate, or infer independent lineage."""
    return [dict(slots=[i,j],action_max_abs_difference=float(np.max(np.abs(a['actions'].astype(float)-b['actions']))),
        equal_pixel_steps=[s for k,s in enumerate(STEPS) if np.array_equal(a['frames'][k],b['frames'][k])])
        for i,a in enumerate(payload) for j,b in enumerate(payload) if i < j]


def contract():
    return dict(episode_rule='[0,floor(E/3),floor(2E/3),E-1]; no outcome substitutions',frames=list(STEPS),
        windows=[dict(window=10,history_steps=[0,5,10],packet_starts=[0,5,10],label=15),
                 dict(window=15,history_steps=[5,10,15],packet_starts=[5,10,15],label=20)],
        conditions=list(CONDITIONS),teacher_forced=True,mask='zero only final action embedding AFTER factual action_encoder; history unchanged',
        zero='raw zero outgoing packet then public normalization; not normalized-zero',
        raw_actions='unaltered expert float32; no clip/tanh/domain-ball rejection',
        counts=dict(encoder=20,projector=20,predict=24,predictor=24,action_encoder=24,pred_proj=24),
        tokens=dict(per_call=3,latent_input=72,action_packet_input=72,pred_proj_samples=72,raw_coordinates_per_packet=10),
        AD=0,grad=0,engine=0,physics=0,reset=0,render=0,training=False,optimization=False,
        label_policy='backend request whitelist has no target; all24 outputs durable before any score',
        normalizer='unchanged public constants; training provenance unknown; no refit, no extra /100',
        time_alignment='local loader action[t:t+5] with pixels[t] -> pixels[t+5]; collector causal provenance unknown',
        dataset_identity='stat version + selected slices, NOT whole-file SHA or certified dataset revision',
        independence='four fixed ordinals, NOT four certified independent demonstrations; lineage unknown',
        counterfactual_truth='F only factual label; Z/M have no physical counterfactual target; M is embedding intervention',
        wall_seconds=WALL,allocated_bytes=ALLOCATED,reserved_bytes=RESERVED,output_cap_bytes=CAP,
        final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,prepare_cap_bytes=256*1024,png_cap_bytes=1024**2,
        SDP='MATH',TF32=False,no_retries=True,no_dedup=True,CPU_neural=False,paired_render_B=False,
        scientific_threshold=None,selection=False,P1_admission=False,independent_confirmation=False)


def material():
    import h5py
    import hdf5plugin  # local codec registration, no resolver/download
    require(sha(old.__file__) == V1_SHA, 'frozen v1 utility source unchanged')
    before = version(DATA); records=[]; payload=[]
    with h5py.File(DATA,'r',rdcc_nbytes=4*1024**2) as f:
        require(len(f['ep_len'])==18685,'known local dataset E=18685; no silent different source')
        indices=episode_indices(len(f['ep_len']))
        for i in indices:
            r,d=slice_episode(f,i); records.append(r); payload.append(d)
        schema={k:dict(shape=list(f[k].shape),dtype=f[k].dtype.str) for k in ('pixels','action','ep_len','ep_offset','episode_idx','step_idx')}
        def attrs(obj):
            result={}
            for k,v in obj.attrs.items():
                if isinstance(v,bytes): result[k]=dict(bytes_hex=v.hex())
                elif isinstance(v,(np.ndarray,np.generic)): result[k]=v.tolist()
                else: result[k]=v
            canonical(result)  # fail closed on unsupported non-JSON metadata
            return result
        attributes=dict(file=attrs(f),columns={k:attrs(f[k]) for k in schema})
        keys=sorted(f.keys())
    require(version(DATA) == before, 'stable HDF5 during selected reads')
    normalizer=enc.native_metadata()
    require(normalizer['training_provenance_verified'] is False,'unknown checkpoint training stats lineage')
    require(normalizer['mean']==base.d2p.ACTION_MEAN.astype(float).tolist() and normalizer['std']==base.d2p.ACTION_STD.astype(float).tolist(),'same frozen public constants')
    files=base.source_files()+[Path(__file__),Path(old.__file__),Path(old.rgb.__file__),Path(h5py.__file__),Path(hdf5plugin.__file__),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),
        ROOT/'innerdocs/p1_constructive_offline_live_fidelity_design_20261027_v1.md']
    site=Path(h5py.__file__).parents[1]/'stable_worldmodel/data'
    files += [site/'dataset.py',site/'formats/hdf5.py']
    bindings={str(Path(p).resolve()):sha(p) for p in files}; bindings.update(enc.MODELS)
    for path,h in bindings.items(): require(sha(path) == h,'source/model seal')
    require(version(DATA) == before,'final HDF5 stat unchanged')
    return dict(schema=NAME+'.prepare',dataset=str(DATA),file_version=before,dataset_schema=schema,
        episode_indices=list(indices),episodes=records,correlation=correlations(payload),contract=contract(),
        dataset_keys=keys,attributes=attributes,versions={k:importlib.metadata.version(k) for k in ('numpy','pillow','torch','h5py','hdf5plugin','stable-worldmodel')},
        normalization=normalizer,policy=str(enc.POLICY),expected_model_hash=enc.MODEL_TENSOR_SHA,
        bindings=bindings,hdf5_version=h5py.version.hdf5_version),payload


def prepare(output):
    p,_=material(); raw=canonical(p); path=Path(output).absolute()
    require(not path.exists() and len(raw)<256*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive small prepare +200MiB floor')
    with path.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw))


def verify(path,expected):
    require(sha(path)==expected,'external prepare SHA'); p=read(path); current,payload=material()
    require(p==current,'full selected byte/version/source reconstruction'); return p,payload


class Store:
    def __init__(self,output):
        self.output=Path(output).absolute()
        require(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'exclusive32MiB +200MiB reserve')
        self.output.mkdir(); self.comp=zlib.compressobj(6,zlib.DEFLATED,31); self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..') and not (self.output/name).is_symlink(),'flat nonsymlink output')
        used=sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(raw),'cap incl partial/final and disk floor')
        with (self.output/name).open('ab' if append else 'xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    def event(self,e):
        require(not self.closed and not self.broken,'no journal retry')
        try: self._write('events.jsonl.gz',self.comp.compress(canonical(e))+self.comp.flush(zlib.Z_SYNC_FLUSH),append=True)
        except BaseException: self.broken=True; raise
    def put(self,name,value,final=False):
        raw=canonical(value); self._write(name,gzip.compress(raw,mtime=0) if name.endswith('.gz') else raw,final=final)
    def close(self):
        require(not self.closed and not self.broken,'cannot finalize broken journal'); self.closed=True
        self._write('events.jsonl.gz',self.comp.flush(zlib.Z_FINISH),append=True,final=True)


def save_array(store,name,a):
    b=BytesIO(); np.save(b,a,allow_pickle=False); store._write(name,b.getvalue())
    r=array_receipt(a)
    return dict(file=name,sha256=sha(store.output/name),raw_sha256=r['sha256'],shape=r['shape'],dtype=r['dtype'])


def normalized(raw):
    a=np.asarray(raw)
    require(a.dtype==np.float32 and a.shape==(3,5,2) and np.isfinite(a).all(),'three literal raw packets')
    return ((a-base.d2p.ACTION_MEAN)/base.d2p.ACTION_STD).reshape(3,10).astype(np.float32)


REQUEST_KEYS={'slot','episode','window','condition','latent_history','raw_packets','tokens'}


def requests(latents,payload,indices):
    require(len(latents)==len(payload)==len(indices)==4,'fixed four episode slots')
    result=[]; labels=[]
    for slot,(zs,d,ep) in enumerate(zip(latents,payload,indices,strict=True)):
        require(len(zs)==5,'all five actual latents')
        for z in zs: enc.native(z)
        a=d['actions']; require(a.dtype==np.float32 and a.shape==(20,2) and np.isfinite(a).all(),'raw20x2')
        for window in WINDOWS:
            start=window//5-2; raw=a[start*5:(start+3)*5].reshape(3,5,2).copy()
            labels.append(dict(slot=slot,window=window,current=zs[start+2].copy(),target=zs[start+3].copy()))
            for kind in CONDITIONS:
                packets=raw.copy()
                if kind=='Z': packets[-1]=0
                result.append(dict(slot=slot,episode=ep,window=window,condition=kind,
                    latent_history=np.stack(zs[start:start+3]),raw_packets=packets,tokens=normalized(packets)))
    return result,labels


def mask_outgoing(embedding,kind):
    require(kind in CONDITIONS and tuple(embedding.shape)==(1,3,192),'full action embedding')
    out=embedding.copy() if isinstance(embedding,np.ndarray) else embedding.clone()
    if kind=='M': out[:,-1,:]=0
    return out


def score(outputs,labels):
    require(len(outputs)==24 and len(labels)==8,'all fixed denominators before scoring')
    rows=[]
    for label in labels:
        group=[r for r in outputs if (r['slot'],r['window'])==(label['slot'],label['window'])]
        require([r['condition'] for r in group]==list(CONDITIONS),'complete ordered F/Z/M')
        target=label['target'].astype(float); values={r['condition']:r['prediction'][-1].astype(float) for r in group}
        errors={k:float(np.mean((v-target)**2)) for k,v in values.items()}
        rows.append(dict(slot=label['slot'],window=label['window'],fidelity_mse=errors,
            label_step=label['window']+5,label_array=array_receipt(label['target']),current_array=array_receipt(label['current']),
            zero_minus_factual=errors['Z']-errors['F'],mask_minus_factual=errors['M']-errors['F'],
            persistence_mse=float(np.mean((label['current'].astype(float)-target)**2)),
            actual_latent_change_l2=float(np.linalg.norm(label['current'].astype(float)-target)),
            factual_vs_zero_mse=float(np.mean((values['F']-values['Z'])**2)),
            factual_vs_mask_mse=float(np.mean((values['F']-values['M'])**2))))
    return dict(windows=rows,window_denominator=8,episode_denominator=4,
        mean_fidelity_mse={k:float(np.mean([r['fidelity_mse'][k] for r in rows])) for k in CONDITIONS},
        counterfactual_truth=False,independent_demonstrations='unknown',P1_admission=False,threshold=None)


class Limits:
    def __init__(self,clock=time.monotonic,memory=lambda:(0,0),free=lambda:FLOOR+CAP):
        self.clock,self.memory,self.free=clock,memory,free; self.start=clock()
        self.peak_allocated=self.peak_reserved=0; self.minimum_free=None; self.checks=0
    def check(self):
        self.checks+=1
        require(self.clock()-self.start<WALL,'wall600s cap')
        a,r=self.memory(); self.peak_allocated=max(self.peak_allocated,a); self.peak_reserved=max(self.peak_reserved,r)
        require(a<=ALLOCATED and r<=RESERVED,'allocated3GiB/reserved4GiB cap')
        free=self.free(); self.minimum_free=free if self.minimum_free is None else min(free,self.minimum_free)
        require(free>=FLOOR,'200MiB disk floor')
    def receipt(self):
        return dict(checks=self.checks,peak_allocated_bytes=self.peak_allocated,peak_reserved_bytes=self.peak_reserved,
            minimum_observed_free_bytes=self.minimum_free,elapsed_seconds=self.clock()-self.start)


class PredictLedger:
    """predict API and predictor.forward are separate counters, not 48 predictions."""
    ORDER=('action_encoder','predict','predictor','pred_proj')
    def __init__(self,sink,limits,sync=lambda:None):
        self.sink,self.limits,self.sync=sink,limits,sync; self.entries=[]; self.failed=False
    def dispatch(self,kind,fn,*args,**kwargs):
        self.limits.check()
        require(not self.failed and len(self.entries)<96 and kind==self.ORDER[len(self.entries)%4],'fixed24 requests/call order, no retry')
        # Only the predict API may be pending around its native child calls.
        require(all(e['completed'] or e['resource']=='predict' for e in self.entries),'no retry after incomplete child')
        e=dict(index=len(self.entries),request=len(self.entries)//4,resource=kind,tokens=3,completed=False)
        self.entries.append(e); self.sink(dict(kind='prediction_module_attempt',entry=deepcopy(e)))
        try:
            value=fn(*args,**kwargs); self.sync(); self.limits.check(); e['completed']=True
            self.sink(dict(kind='prediction_module_return',entry=deepcopy(e))); return value
        except BaseException as exc:
            self.failed=True; e['error']=enc.error(kind,exc); raise
    def counts(self):
        return {k:dict(attempted=sum(e['resource']==k for e in self.entries),completed=sum(e['resource']==k and e['completed'] for e in self.entries),
            attempted_tokens=3*sum(e['resource']==k for e in self.entries),completed_tokens=3*sum(e['resource']==k and e['completed'] for e in self.entries)) for k in self.ORDER}


@contextmanager
def prediction_guard(model,torch,ledger,errors):
    undo=[]
    def patch(obj,key,value): undo.append((obj,key,getattr(obj,key))); setattr(obj,key,value)
    def validate(x,shape):
        require(x.device.type=='cuda' and x.dtype==torch.float32 and tuple(x.shape)==shape and not torch.is_grad_enabled(),'native CUDA float32 inference shape')
    def wrap(kind,fn):
        def call(*args,**kwargs):
            validate(args[0],(3,192) if kind=='pred_proj' else (1,3,10) if kind=='action_encoder' else (1,3,192))
            if kind in ('predict','predictor'): validate(args[1],(1,3,192))
            def execute():
                out=fn(*args,**kwargs)
                validate(out,(3,192) if kind=='pred_proj' else (1,3,192))
                require(bool(torch.isfinite(out).all()),'finite module output'); return out
            return ledger.dispatch(kind,execute)
        return call
    def denied(*args,**kwargs):
        ledger.failed=True; ledger.sink(dict(kind='forbidden_visual_entry')); raise ValueError('no fresh visual work during prediction')
    try:
        for obj,key,kind in ((model,'predict','predict'),(model.predictor,'forward','predictor'),(model.action_encoder,'forward','action_encoder'),(model.pred_proj,'forward','pred_proj')):
            patch(obj,key,wrap(kind,getattr(obj,key)))
        for m in (model.encoder,model.projector): patch(m,'forward',denied)
        yield
    finally:
        for obj,key,value in reversed(undo):
            try: setattr(obj,key,value)
            except BaseException as exc: errors.append(enc.error('restore_'+key,exc))


def native_predict(model,torch,device,request):
    require(set(request)==REQUEST_KEYS,'prediction whitelist, no target/label')
    with torch.inference_mode():
        z=torch.from_numpy(request['latent_history'].copy()).to(device)[None]
        tokens=torch.from_numpy(request['tokens'].copy()).to(device)[None]
        before=model.action_encoder(tokens); after=mask_outgoing(before,request['condition'])
        require(torch.equal(before[:,:2],after[:,:2]),'historical embeddings bitwise unchanged')
        pred=model.predict(z,after)
        return dict(prediction=pred[0].detach().cpu().numpy().copy(),
            embedding_before=before[0].detach().cpu().numpy().copy(),embedding_after=after[0].detach().cpu().numpy().copy())


def execute_requests(reqs,backend,store,rows,limits):
    require(len(reqs)==len(rows)==24,'fixed24 requests')
    outputs=[]
    for i,r in enumerate(reqs):
        require(set(r)==REQUEST_KEYS,'no labels in backend request')
        limits.check(); rows[i]['status']='attempted'
        meta={k:r[k] for k in ('slot','episode','window','condition')}
        inputs={k:save_array(store,f'request_{i:02d}.{k}.npy',r[k]) for k in ('latent_history','raw_packets','tokens')}
        store.event(dict(kind='prediction_request',index=i,identity=meta,inputs=inputs))
        result=backend(deepcopy(r))
        for k in ('prediction','embedding_before','embedding_after'):
            require(result[k].dtype==np.float32 and result[k].shape==(3,192) and np.isfinite(result[k]).all(),'native complete finite outputs')
        require(np.array_equal(result['embedding_before'][:2],result['embedding_after'][:2]),'unchanged history embeddings')
        require(np.array_equal(result['embedding_after'],mask_outgoing(result['embedding_before'][None],r['condition'])[0]),'exact outgoing-only mask contract')
        saved={k:save_array(store,f'request_{i:02d}.{k}.npy',result[k]) for k in result}
        saved['last_native192']=save_array(store,f'request_{i:02d}.last_native192.npy',result['prediction'][-1])
        rows[i].update(status='saved',identity=meta,inputs=inputs,outputs=saved)
        store.event(dict(kind='prediction_saved',row=deepcopy(rows[i])))
        outputs.append(dict(**meta,prediction=result['prediction'].copy()))
    return outputs


def run(prepared,expected,output):
    limits=Limits(); p,payload=verify(prepared,expected); limits.check(); store=Store(output)
    errors=[]; model=None; before=after=host=None; source_ok=False; forbidden_ad=[]
    images=[dict(index=i,slot=i//5,episode=p['episode_indices'][i//5],step=STEPS[i%5],status='not_started') for i in range(20)]
    rows=[dict(index=i,status='not_started') for i in range(24)]
    visual=enc.Ledger(5,store.event); pred=PredictLedger(store.event,limits)
    def event(e): limits.check(); store.event(e)
    def ad_event(e): forbidden_ad.append(deepcopy(e)); store.event(e)
    visual.sink=event
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        store.put('selected_source_receipts.json',p)
        for i,d in enumerate(payload): save_array(store,f'slot_{i}_raw20x2.npy',d['actions'])
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
        limits.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
        limits.free=lambda:shutil.disk_usage(store.output).free
        limits.check(); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        limits.check(); enc.frozen(model); before=enc.tensor_hash(model)
        require(before==p['expected_model_hash'],'original frozen tensor SHA')
        visual.sync=pred.sync=lambda:torch.cuda.synchronize(device); visual.no_grad=lambda:not torch.is_grad_enabled()
        transform=p0.image_transform(); latents=[]
        with old.no_ad(torch,ad_event,errors),sdpa_kernel(SDPBackend.MATH):
            with enc.guard(model,torch,visual,errors):
                for slot,d in enumerate(payload):
                    zs=[]
                    for j,a in enumerate(d['frames']):
                        limits.check(); i=slot*5+j; row=images[i]; row['status']='attempted'; visual.image=i
                        raw=png_bytes(a); require(len(raw)<=1024**2,'PNG1MiB cap')
                        name=f'image_{i:02d}.png'; store._write(name,raw)
                        row['png']=dict(file=name,sha256=sha(store.output/name),pixels=array_receipt(a))
                        store.event(dict(kind='image_attempt',row=deepcopy(row)))
                        with torch.inference_mode():
                            x=p0._image_tensor(a.copy(),transform=transform,device=device)
                            require(x.device.type=='cuda' and tuple(x.shape)==(1,1,3,224,224),'native CUDA pixels-only input')
                            z=model.encode({'pixels':x})['emb']
                            require(z.device.type=='cuda' and z.dtype==torch.float32 and tuple(z.shape)==(1,1,192),'native CLS192')
                            value=z[0,0].detach().cpu().numpy().copy(); enc.native(value)
                        require([(e['resource'],e['completed']) for e in visual.entries if e['image']==i]==[('encoder',True),('projector',True)],'both visual completions')
                        row.update(status='encoded',latent=save_array(store,f'image_{i:02d}.native192.npy',value))
                        store.event(dict(kind='image_saved',row=deepcopy(row))); zs.append(value); visual.image=None
                    latents.append(zs)
            reqs,labels=requests(latents,payload,p['episode_indices'])
            store.put('prediction_order.json',[{k:r[k] for k in ('slot','episode','window','condition')} for r in reqs])
            with prediction_guard(model,torch,pred,errors):
                outputs=execute_requests(reqs,lambda r:native_predict(model,torch,device,r),store,rows,limits)
            store.event(dict(kind='all24_predictions_durable_before_scoring'))
            store.put('diagnostics.json',score(outputs,labels))
        limits.check()
    except BaseException as exc:
        errors.insert(0,enc.error('execution',exc))
        try: store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary: errors.append(enc.error('error_journal',secondary))
    finally:
        for row in images+rows:
            if row['status']=='attempted': row['status']='failed_partial'
        if model is not None:
            try: enc.frozen(model); after=enc.tensor_hash(model)
            except BaseException as exc: errors.append(enc.error('final_model',exc))
        try: verify(prepared,expected); source_ok=True
        except BaseException as exc: errors.append(enc.error('final_source',exc))
        try: limits.check()
        except BaseException as exc: errors.append(enc.error('final_limits',exc))
        try: store.close()
        except BaseException as exc: errors.append(enc.error('journal_finalize',exc))
        counts=dict(**visual.counts(),**pred.counts())
        complete=(not errors and not visual.blocked and not forbidden_ad and not pred.failed and source_ok and before is not None and before==after
            and all(r['status']=='encoded' for r in images) and all(r['status']=='saved' for r in rows)
            and all(counts[k]['attempted']==counts[k]['completed']==v for k,v in contract()['counts'].items()))
        hashes={}
        for f in store.output.iterdir():
            if f.is_file():
                try: hashes[f.name]=sha(f)
                except BaseException as exc: errors.append(enc.error('artifact_hash',exc)); complete=False
        report=dict(schema=NAME+'.report',status='stage_A_fidelity_saved_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,episode_indices=p['episode_indices'],
            contract=p['contract'],images=images,requests=rows,resources=counts,errors=errors,first_error=errors[0] if errors else None,
            module_ledger=dict(visual=visual.entries,prediction=pred.entries),blocked_visual=visual.blocked,
            forbidden_AD_attempts=forbidden_ad,prediction_failed=pred.failed,
            source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before else None,
            host=host,limits=limits.receipt(),artifact_sha256=hashes,
            AD=0 if complete else None,physics=0,selection=False,P1_admission=False,independent_confirmation=False)
        try: store.put('report.json',report,final=True)
        except BaseException as exc:
            complete=False; errors.append(enc.error('final_report',exc)); print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial retained; no retry: '+str(errors)); return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=('prepare','verify','run'))
    for key in ('prepared','prepare-sha','output'): parser.add_argument('--'+key)
    args=parser.parse_args()
    if args.command=='prepare': result=prepare(args.output)
    elif args.command=='verify': result=dict(status='saved_only_verified',episode_indices=verify(args.prepared,args.prepare_sha)[0]['episode_indices'])
    else: result=run(args.prepared,args.prepare_sha,args.output)
    print(canonical(result).decode(),end='')


if __name__=='__main__': main()
