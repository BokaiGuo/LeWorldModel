"""Fixed local HDF5 factual fidelity diagnostic; MAIN-only frozen CUDA.

No simulator, training, normalizer fit, action selection, or P1 admission.
Prepare/verify read ONLY fixed slices, never hash the entire HDF5 asset.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
from io import BytesIO
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_selected_rgb_cuda_20261027_v1 as rgb

base,enc=rgb.planner,rgb.enc
ROOT=base.ROOT
NAME='p1_constructive_offline_fidelity_cuda_20261027_v1'
DATA=Path('/home/gbk/.stable_worldmodel/datasets/pusht_expert_train.h5')
EPISODES=(0,1,2,3)
STEPS=(0,5,10,15,20)
KINDS=('factual','zero','sign_flipped')
CAP,FLOOR,FINAL=8*1024**2,200*1024**2,256*1024
require,sha,read,canonical=base.require,base.sha,base.read,base.canonical
Store=base.Store


def version(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'local regular HDF5 only')
    s=path.stat();return {k:int(getattr(s,k)) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')}


def array_receipt(value):
    a=np.ascontiguousarray(value)
    require(a.dtype.kind in 'uif' and np.isfinite(a).all(),'finite numeric selected bytes')
    return dict(dtype=a.dtype.str,shape=list(a.shape),sha256=enc.digest(a.tobytes()))


def png_bytes(a):
    from PIL import Image
    require(a.dtype==np.uint8 and a.shape==(224,224,3),'unaltered RGB224')
    b=BytesIO();Image.fromarray(a).save(b,format='PNG');return b.getvalue()


def slice_episode(f,index):
    """Explicit offsets/lengths and 21 local indices; no resolver/cache/whole columns."""
    require(type(index) is int and index in EPISODES,'fixed episode only')
    nrows=f['pixels'].shape[0];n=len(f['ep_len'])
    require(index+1<n and f['ep_offset'].shape==f['ep_len'].shape,'fixed episode plus next-offset boundary')
    offset=int(f['ep_offset'][index]);length=int(f['ep_len'][index]);next_offset=int(f['ep_offset'][index+1])
    require(length>=21 and offset>=0 and offset+length==next_offset<=nrows,'episode full21 + exact next offset; no substitution')
    require(index==0 and offset==0 or index>0 and int(f['ep_offset'][index-1])+int(f['ep_len'][index-1])==offset,'previous episode boundary')
    for key in ('action','episode_idx','step_idx'):
        require(f[key].shape[0]==nrows,'column row alignment')
    require(f['pixels'].shape[1:]==(224,224,3) and f['pixels'].dtype==np.uint8 and
        f['action'].shape[1:]==(2,) and f['action'].dtype==np.float32,'native RGB/action storage dtype/shape')
    ids=np.asarray(f['episode_idx'][offset:offset+21]);steps=np.asarray(f['step_idx'][offset:offset+21])
    require(ids.dtype.kind in 'iu' and steps.dtype.kind in 'iu' and ids.shape==steps.shape==(21,) and
        np.array_equal(ids,np.full(21,index)) and np.array_equal(steps,np.arange(21)),'no episode crossing or time gaps')
    actions=np.asarray(f['action'][offset:offset+20])
    require(actions.shape==(20,2) and np.isfinite(actions).all() and (abs(actions)<=1).all(),'literal factual actions in frozen raw domain; never clip')
    frames=np.asarray(f['pixels'][[offset+s for s in STEPS]])
    require(frames.shape==(5,224,224,3),'all five fixed frames')
    audit=dict(episode_index=index,offset=offset,length=length,next_offset=next_offset,
        frame_local_indices=list(STEPS),frame_global_indices=[offset+s for s in STEPS],
        action_local_slice=[0,20],action_global_slice=[offset,offset+20],
        packet_slices=[dict(local=[s,s+5],global_slice=[offset+s,offset+s+5],from_frame=s,to_frame=s+5) for s in (0,5,10,15)],
        episode_idx=array_receipt(ids),step_idx=array_receipt(steps),actions=array_receipt(actions),raw_actions=actions.astype(float).tolist(),
        frames=[dict(step=s,pixels=array_receipt(a),png_sha256=enc.digest(png_bytes(a))) for s,a in zip(STEPS,frames,strict=True)],
        action_time_semantics='storage action[t:t+5] paired with pixels[t] -> pixels[t+5], matching local loader frameskip/reshape; collector causal lineage unknown')
    return audit,dict(frames=frames,actions=actions)


def contract():
    return dict(episodes=list(EPISODES),frames=list(STEPS),root_step=10,H1_target_step=15,H2_target_step=20,
        H1=list(KINDS),H2='fresh factual10:20, recompute both packets; no factual H1 reuse',
        encoder=20,projector=20,predict=20,action_encoder=20,
        H1_evaluations=12,H2_evaluations=4,predict_including_H2_prefix=20,AD=0,physics=0,training=False,
        prediction_tokens=dict(predict_calls=20,latent_tokens_per_call=3,total_latent_input_tokens=60,
            action_encoder_calls=20,action_packet_tokens_per_call=3,total_action_packet_tokens=60,raw_coordinates_per_token=10),
        normalizer=dict(mean=base.d2p.ACTION_MEAN.astype(float).tolist(),std=base.d2p.ACTION_STD.astype(float).tolist(),
            raw_divisor=1,training_provenance_verified=False,fit=False),
        goal='actual z15 for fixed H1 fidelity scores only; actual z20 for H2 only; never model inputs or selection',
        no_early_exit=True,no_dedup=True,no_retries=True,new_Q=False,Q='identity_native192_mean_squared_error',
        file_binding='stat version + explicit selected array bytes; NOT whole-HDF5 hash or official dataset revision',
        numerical=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),selection=False,
        scientific_pass_threshold=None,P1_admission=False,P2_admission=False,independent_confirmation=False,
        scope='four fixed local recorded episodes; factual fidelity and noncausal counterfactual controls, not success/theory',
        output_cap_bytes=CAP,reserve_bytes=FLOOR)


def material():
    import h5py
    import hdf5plugin  # installed codec registration only, no installation/resolver
    before=version(DATA);records=[];payload=[]
    with h5py.File(DATA,'r',rdcc_nbytes=4*1024**2) as f:
        for index in EPISODES:
            a,d=slice_episode(f,index);records.append(a);payload.append(d)
        schema={k:dict(shape=list(f[k].shape),dtype=f[k].dtype.str) for k in ('pixels','action','ep_len','ep_offset','episode_idx','step_idx')}
    require(version(DATA)==before,'HDF5 stat changed during selected reads')
    normalizer=enc.native_metadata()
    require(normalizer['training_provenance_verified'] is False,'public normalizer is not certified training lineage')
    files=base.source_files()+[Path(__file__),Path(rgb.__file__),Path(h5py.__file__),Path(hdf5plugin.__file__),
        ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    site=Path(h5py.__file__).parents[1]/'stable_worldmodel/data'
    files += [site/'dataset.py',site/'formats/hdf5.py']
    bindings={str(Path(p).resolve()):sha(p) for p in files}
    bindings.update(enc.MODELS)
    for path,h in bindings.items():require(sha(path)==h,'source/model file seal')
    require(version(DATA)==before,'final HDF5 stat unchanged')
    return dict(schema=NAME+'.prepare',dataset=str(DATA),file_version=before,dataset_schema=schema,episodes=records,
        normalization=normalizer,contract=contract(),policy=str(enc.POLICY),expected_model_hash=enc.MODEL_TENSOR_SHA,
        bindings=bindings,hdf5_version=h5py.version.hdf5_version),payload


def prepare(output):
    p,_=material();raw=canonical(p);path=Path(output).absolute()
    require(not path.exists() and len(raw)<256*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive small prepare +200MiB')
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw))


def verify(path,expected):
    require(sha(path)==expected,'external prepare SHA');p=read(path);current,payload=material()
    require(p==current,'full selected bytes/stat/source/model reconstruction');return p,payload


def save_array(store,name,a):
    b=BytesIO();np.save(b,a,allow_pickle=False);store._write(name,b.getvalue())
    receipt=array_receipt(a)
    return dict(file=name,sha256=sha(store.output/name),dtype=receipt['dtype'],shape=receipt['shape'],raw_sha256=receipt['sha256'])


def context(latents,actions):
    require(len(latents)==5,'all five actual latents retained')
    for z in latents:enc.native(z)
    c=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=np.asarray(latents[:3]).astype(float).tolist(),
        action_history=base.d2p.normalized_history(actions[:10]).astype(float).tolist())
    base.d2p.validate_context(c);return c


def diagnose(latents,actions,backend,sink):
    """Array callback kernel. Targets never reach backend or action proposals."""
    c=context(latents,actions);a=np.asarray(actions)
    require(a.dtype==np.float32 and a.shape==(20,2) and np.isfinite(a).all() and (abs(a)<=1).all(),'exact saved raw20x2')
    outputs=[]
    for kind,commands in zip(KINDS,(a[10:15].copy(),np.zeros((5,2),np.float32),-a[10:15].copy()),strict=True):
        sink(dict(kind='H1_request',control=kind,context=c,commands=commands.astype(float).tolist()))
        after=backend.step(deepcopy(c),commands.copy());base.d2p.validate_advance(c,commands,after)
        require(after['latent_history'][:2]==c['latent_history'][1:],'H1 complete chronology')
        z=np.asarray(after['latent_history'][-1]);score=float(np.mean((z-np.asarray(latents[3],float))**2))
        row=dict(kind=kind,horizon=1,commands=commands.astype(float).tolist(),context=after,target_step=15,fidelity_mse=score)
        outputs.append(row);sink(dict(kind='H1_result',result=row))
    sink(dict(kind='H2_request',context=c,commands=a[10:20].astype(float).tolist()))
    after=backend.replay(deepcopy(c),a[10:20].copy());base.d2p.validate_advance(c,a[10:20],after)
    require(after['latent_history'][0]==c['latent_history'][-1],'H2 full original root retained')
    row=dict(kind='factual',horizon=2,commands=a[10:20].astype(float).tolist(),context=after,target_step=20,
        fidelity_mse=float(np.mean((np.asarray(after['latent_history'][-1])-np.asarray(latents[4],float))**2)))
    outputs.append(row);sink(dict(kind='H2_result',result=row))
    return dict(outputs=outputs,root_context=c,persistence_H1_mse=float(np.mean((np.asarray(latents[2],float)-latents[3])**2)),
        persistence_H2_mse=float(np.mean((np.asarray(latents[2],float)-latents[4])**2)),
        selection=False,P1_admission=False,counterfactual_truth='unknown; recorded target belongs ONLY to factual actions')


class PredictLedger(base.D2PLedger):
    def begin(self,r):
        require(sum(e['resource']==r for e in self.neural)<20,'predispatch20 cap')
        return super().begin(r)
    def counts(self):return {k:v for k,v in super().counts().items() if k in ('predict','action_encoder')}


@contextmanager
def no_ad(torch,sink,problems):
    undo=[]
    def deny(*a,**kw):sink(dict(kind='forbidden_AD'));raise ValueError('offline fidelity prohibits AD')
    try:
        groups=[(torch.autograd,('grad','backward','_engine_run_backward')),(torch.Tensor,('backward',)),
            (torch.autograd.functional,('jacobian','hessian','jvp','vjp','hvp','vhp')),
            (torch.func,('jacrev','jacfwd','hessian','jvp','vjp','grad','grad_and_value'))]
        for obj,keys in groups:
            for key in keys:
                if hasattr(obj,key):undo.append((obj,key,getattr(obj,key)));setattr(obj,key,deny)
        yield
    finally:
        for obj,key,value in reversed(undo):
            try:setattr(obj,key,value)
            except BaseException as e:problems.append(enc.error('restore_AD',e))


def encode_all(payload,callback,ledger,store,rows):
    latents=[]
    for ep,d in enumerate(payload):
        zs=[]
        for j,a in enumerate(d['frames']):
            i=ep*5+j;row=rows[i];row['status']='attempted';ledger.image=i
            name=f'episode_{ep}_step_{STEPS[j]:02d}.png';store._write(name,png_bytes(a))
            row['png']=dict(file=name,sha256=sha(store.output/name),pixels=array_receipt(a))
            store.event(dict(kind='image_attempt',episode=ep,step=STEPS[j],receipt=deepcopy(row)))
            z=callback(a.copy());enc.native(z)
            require([(e['resource'],e['completed']) for e in ledger.entries if e['image']==i]==[('encoder',True),('projector',True)],'actual encoder/projector completions')
            row.update(status='encoded',latent=save_array(store,f'image_{i:02d}.native192.npy',z));zs.append(z.copy())
            store.event(dict(kind='image_saved',receipt=deepcopy(row)));ledger.image=None
        latents.append(zs)
    return latents


def run(prepared,expected,output):
    p,payload=verify(prepared,expected);store=Store(output);problems=[];model=None;before=after=host=None;source_ok=False
    images=[dict(index=i,episode=i//5,step=STEPS[i%5],status='not_started') for i in range(20)]
    rows=[dict(episode=i,status='not_started') for i in EPISODES]
    visual=enc.Ledger(5,store.event);pred=PredictLedger(store.event)
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        store.put('selected_payload_receipts.json',dict(dataset=p['dataset'],file_version=p['file_version'],episodes=p['episodes']))
        for i,d in enumerate(payload):
            rows[i]['actions']=save_array(store,f'episode_{i}_raw20x2.npy',d['actions'])
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen tensor SHA')
        visual.sync=pred.sync=lambda:torch.cuda.synchronize(device);visual.no_grad=lambda:not torch.is_grad_enabled()
        transform=p0.image_transform()
        with sdpa_kernel(SDPBackend.MATH):
            with enc.guard(model,torch,visual,problems):
                def callback(a):
                    with torch.inference_mode():
                        x=p0._image_tensor(a,transform=transform,device=device)
                        require(x.device.type=='cuda' and tuple(x.shape)==(1,1,3,224,224),'native CUDA batch1 pixels')
                        z=model.encode({'pixels':x})['emb']
                        require(z.device.type=='cuda' and z.dtype==torch.float32 and tuple(z.shape)==(1,1,192),'native projected CLS192')
                        return z[0,0].detach().cpu().numpy().copy()
                latents=encode_all(payload,callback,visual,store,images)
            backend=base.d2p.TorchBackend(model,device)
            with no_ad(torch,store.event,problems),base.module_guard(model,pred):
                for ep,d in enumerate(payload):
                    rows[ep]['status']='attempted'
                    def sink(e):store.event(dict(episode=ep,event=e))
                    result=diagnose(latents[ep],d['actions'],backend,sink)
                    name=f'episode_{ep}.diagnostic.json.gz';store.put(name,result)
                    rows[ep].update(status='diagnostic_recorded',result_file=name)
    except BaseException as e:
        problems.append(enc.error('execution',e))
        try:store.event(dict(kind='execution_error',error=problems[0]))
        except BaseException as e:problems.append(enc.error('error_journal',e))
    finally:
        for row in images+rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model)
            except BaseException as e:problems.append(enc.error('final_model',e))
        try:verify(prepared,expected);source_ok=True
        except BaseException as e:problems.append(enc.error('final_source',e))
        try:store.close()
        except BaseException as e:problems.append(enc.error('journal_finalize',e))
        complete=not problems and not visual.blocked and source_ok and before is not None and before==after and all(r['status']=='encoded' for r in images) and all(r['status']=='diagnostic_recorded' for r in rows) and visual.counts()=={k:dict(attempted=20,completed=20) for k in ('encoder','projector')} and pred.counts()=={k:dict(attempted=20,completed=20) for k in ('predict','action_encoder')}
        hashes={}
        for f in store.output.iterdir():
            if f.is_file():
                try:hashes[f.name]=sha(f)
                except BaseException as e:problems.append(enc.error('artifact_hash',e));complete=False
        report=dict(schema=NAME+'.report',status='offline_fidelity_recorded_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,images=images,rows=rows,resources=dict(**visual.counts(),**pred.counts()),
            contract=p['contract'],errors=problems,first_error=problems[0] if problems else None,artifact_sha256=hashes,
            source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before else None,
            host=host,new_AD=0 if complete else None,new_physics=0,training=False,selection=False,P1_admission=False,P2_admission=False)
        try:store.put('report.json',report,final=True)
        except BaseException as e:complete=False;problems.append(enc.error('final_report',e));print(canonical(dict(errors=problems)).decode(),flush=True)
    require(complete,'partial retained; no retry: '+str(problems));return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run'))
    for key in ('prepared','prepare-sha','output'):p.add_argument('--'+key)
    a=p.parse_args()
    if a.command=='prepare':r=prepare(a.output)
    elif a.command=='verify':r=dict(status='verified_saved_bytes_only',episodes=len(verify(a.prepared,a.prepare_sha)[0]['episodes']))
    else:r=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(r).decode(),end='')


if __name__=='__main__':main()
