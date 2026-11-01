"""MAIN-only CUDA F/Z/M, fixed paired live contexts and factual own-view labels.

New32 encodings +24 H1; no simulator, optimizer, AD, outcome selection or training.
Z/M are representation/action-conditioning controls, NOT new physical labels.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_constructive_live_overlay_fidelity_20261027_v1_cuda as previous

render=previous.render
base,enc,rgb,saved=render.base,render.enc,render.rgb,render.saved
ROOT=base.ROOT
NAME='p1_constructive_live_overlay_fidelity_20261027_v2_cuda'
VIEWS,ROLES=previous.VIEWS,previous.ROLES
CONTROLS=('F','Z','M')
OLD_SHA='67b028da9d5204f86a9be51596269a4ece2e66d13c3ca8f9a255fc13dfe1b2dd'
RENDER_SHA='27becaabeb6a54cc34810cb53655957fded9b0d7145470d95c9b599c7ce7534f'
FLOOR,CAP=render.FLOOR,render.CAP
require,sha,read,canonical=base.require,base.sha,base.read,base.canonical


def contract():
    return dict(branches=4,views=list(VIEWS),controls=list(CONTROLS),images=32,encoder=32,projector=32,
        predict=24,action_encoder=24,model_predict_entries=24,pred_proj_calls=24,
        predictor_input_tokens=72,action_encoder_packet_tokens=72,pred_proj_token_samples=72,
        context='own view warmup5,warmup10,prefix5; actual zero packet then original prefix5',
        F='original suffix5 raw',Z='only outgoing raw5x2 zeros, then frozen public normalization; history untouched',
        M='F input tokens; actual AE call; clone output and zero ONLY embedding[:, -1, :]',
        labels='own view actual terminal10 ONLY; Z/M have no counterfactual physical truth',
        order='root-major branch, off/on, F/Z/M; all requests sealed before predictions; all predictions before evaluation',
        no_dedup=True,no_cache=True,no_retry=True,selection=False,AD=0,optimization=False,physics=0,training=False,
        normalization='frozen public float32; no extra /100 or fit; training lineage unknown',
        numerical=dict(SDP='MATH',TF32=False),wall_seconds=600,gpu_allocated_cap=3*1024**3,gpu_reserved_cap=4*1024**3,
        output_cap_bytes=CAP,reserve_bytes=FLOOR,scientific_pass_threshold=None,P1_admission=False,P2_admission=False,
        training_overlay_provenance='unknown',interpretation='known-task own-view conditioning ablation, not training-overlay causal attribution')


def build(paired_report,paired_sha):
    require(sha(previous.__file__)==OLD_SHA and sha(render.__file__)==RENDER_SHA,'frozen v1 renderer/input infrastructure')
    old=previous.build(paired_report,paired_sha)  # Saved inputs only, never v1 run or its CUDA result.
    p=deepcopy(old);p.update(schema=NAME+'.prepare',contract=contract())
    files=[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    p['bindings'].update({str(f.resolve()):sha(f) for f in files})
    return p


def prepare(output,**args):
    p=build(**args);value=canonical(p);path=Path(output).absolute()
    require(not path.exists() and len(value)<256*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(value),'exclusive small prepare/200MiB')
    with path.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(value),contract=p['contract'])


def verify(path,expected):
    require(sha(path)==expected,'external v2 prepare SHA');p=read(path)
    require(p==build(**p['arguments']),'complete paired inputs and new source reconstruction');return p


def make_requests(branches,observed):
    """Whitelist observed3 only: no terminal key or label parameter."""
    require(len(branches)==4 and [(b['branch'],b['source_slot'],b['arm']) for b in branches]==
        [(i*2+j,i,a) for i in range(2) for j,a in enumerate(base.ARMS)],'original fixed4 branches')
    require(set(observed)=={(b,v,r) for b in range(4) for v in VIEWS for r in ROLES[:3]},'observed-only whitelist; no terminals')
    requests=[]
    for b in branches:
        actions=previous.physical_commands(b['commands'])
        between=np.concatenate((np.zeros((5,2),np.float32),actions[:5]))
        for view in VIEWS:
            zs=[observed[(b['branch'],view,r)] for r in ROLES[:3]]
            for z in zs:enc.native(z)
            c=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
                latent_history=np.asarray(zs).astype(float).tolist(),action_history=base.d2p.normalized_history(between).astype(float).tolist())
            base.d2p.validate_context(c)
            for control in CONTROLS:
                raw=np.zeros((5,2),np.float32) if control=='Z' else actions[5:].copy()
                requests.append(dict(index=len(requests),branch=b['branch'],source_slot=b['source_slot'],arm=b['arm'],view=view,control=control,
                    context=deepcopy(c),commands=raw.astype(float).tolist(),raw_between=between.astype(float).tolist()))
    return requests


def action_tokens(context,commands):
    base.d2p.validate_context(context);raw=np.asarray(commands,dtype=np.float32)
    require(raw.shape==(5,2) and np.isfinite(raw).all() and (abs(raw)<=1).all() and np.array_equal(raw,commands),'literal float32 outgoing5x2')
    return np.concatenate((np.asarray(context['action_history'],np.float32),base.d2p.normalized_history(raw)),axis=0)


def mask_embedding(embedding,control):
    require(control in CONTROLS and tuple(embedding.shape)==(1,3,192),'F/Z/M full AE embedding')
    out=embedding.clone() if hasattr(embedding,'clone') else embedding.copy()
    if control=='M':out[:,-1,:]=0
    return out


class NativeBackend:
    def __init__(self,model,device):
        require(getattr(device,'type',None)=='cuda','HOST CUDA only, no CPU neural');self.model,self.device=model,device
    def predict(self,context,commands,control,sink):
        import torch
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        tokens=action_tokens(context,commands)
        raw=torch.tensor(commands,device=self.device,dtype=torch.float32)
        history=torch.tensor(context['action_history'],device=self.device,dtype=torch.float32)[None]
        states=torch.tensor(context['latent_history'],device=self.device,dtype=torch.float32)[None]
        arr=lambda t:t.detach().cpu().numpy().copy()
        with torch.no_grad():
            # Same GPU normalize_live_packet path as sealed make_transition, not a CPU neural path.
            outgoing=p0.packets.normalize_live_packet(raw,mean=p0.PUBLIC_ACTION_MEAN,std=p0.PUBLIC_ACTION_STD)
            actual=torch.cat((history,outgoing[None,None]),dim=1)
            require(tuple(actual.shape)==(1,3,10) and actual.device.type=='cuda' and torch.isfinite(actual).all(),'native AE input')
            require(np.array_equal(arr(actual)[0],tokens),'GPU/public float32 token conformance')
            sink(dict(kind='action_tokens',tokens=arr(actual).tolist()))
            embedding=self.model.action_encoder(actual)
            require(tuple(embedding.shape)==(1,3,192) and embedding.dtype==torch.float32 and torch.isfinite(embedding).all(),'actual AE output')
            conditioned=mask_embedding(embedding,control)
            before,after=arr(embedding),arr(conditioned)
            require(np.array_equal(before[:,:2],after[:,:2]) and
                (np.count_nonzero(after[:,-1])==0 if control=='M' else np.array_equal(before,after)),'only outgoing embedding mask')
            sink(dict(kind='embedding_condition',control=control,actual_before=before.tolist(),used_after=after.tolist()))
            prediction=self.model.predict(states,conditioned)
            require(prediction.device.type=='cuda' and prediction.dtype==torch.float32 and tuple(prediction.shape)==(1,3,192) and torch.isfinite(prediction).all(),'full native post-pred_proj prediction')
            result=arr(prediction)[0]
        return dict(predicted3=result.astype(float).tolist(),action_tokens=tokens.astype(float).tolist(),
            embedding_before=before[0].astype(float).tolist(),embedding_used=after[0].astype(float).tolist())


def predict_all(requests,backend,sink):
    expected=[(b,v,c) for b in range(4) for v in VIEWS for c in CONTROLS]
    require([(r['branch'],r['view'],r['control']) for r in requests]==expected and
        [r['index'] for r in requests]==list(range(24)),'all24 fixed requests, no reordering/dedup')
    outputs=[]
    for r in requests:
        sink(dict(kind='H1_attempt',request=deepcopy(r)))
        result=backend.predict(deepcopy(r['context']),np.asarray(r['commands'],np.float32),r['control'],
            lambda e:sink(dict(kind='condition_detail',request_index=r['index'],detail=e)))
        a=np.asarray(result['predicted3'])
        require(a.shape==(3,192) and np.isfinite(a).all() and np.array_equal(a,a.astype(np.float32)),'native prediction3 saved exactfloat32')
        outputs.append(dict(request_index=r['index'],**result));sink(dict(kind='H1_completed',output=deepcopy(outputs[-1])))
    return outputs


def evaluate(requests,outputs,labels):
    require(len(requests)==len(outputs)==24 and set(labels)=={(b,v) for b in range(4) for v in VIEWS},'all24 predictions and own8 labels')
    for z in labels.values():enc.native(z)
    groups=[];mse=lambda a,b:float(np.mean((np.asarray(a,float)-np.asarray(b,float))**2))
    for b in range(4):
        for view in VIEWS:
            indices=[i for i,r in enumerate(requests) if r['branch']==b and r['view']==view]
            require(len(indices)==3 and [requests[i]['control'] for i in indices]==list(CONTROLS),'F/Z/M group')
            predicted={};errors={};target=labels[(b,view)];current=requests[indices[0]]['context']['latent_history'][-1]
            for i in indices:
                require(outputs[i]['request_index']==requests[i]['index']==i,'prediction/request identity')
                predicted[requests[i]['control']]=outputs[i]['predicted3'][-1]
                errors[requests[i]['control']]=mse(predicted[requests[i]['control']],target)
            persistence=mse(current,target)
            groups.append(dict(branch=b,source_slot=requests[indices[0]]['source_slot'],arm=requests[indices[0]]['arm'],view=view,
                errors=errors,persistence_mse=persistence,F_minus_persistence=errors['F']-persistence,
                Z_minus_F=errors['Z']-errors['F'],M_minus_F=errors['M']-errors['F'],
                F_vs_Z_output_mse=mse(predicted['F'],predicted['Z']),F_vs_M_output_mse=mse(predicted['F'],predicted['M']),
                target_actual_change_mse=persistence,labels='factual own-view only; no Z/M physical truth',P1_admission=False))
    return groups


class PredictLedger(base.D2PLedger):
    def __init__(self,sink,check):super().__init__(sink);self.check=check
    def begin(self,r):
        self.check();require(sum(e['resource']==r for e in self.neural)<24,'fixed24 predispatch cap');return super().begin(r)
    def counts(self):return {k:v for k,v in super().counts().items() if k in ('predict','action_encoder')}


@contextmanager
def prediction_receipts(model,sink,check,sync=lambda:None):
    receipts=[];hooks=[];original=model.predict;owned='predict' in model.__dict__;pending=None
    def begin(resource):
        check();require(sum(e['resource']==resource for e in receipts)<24,'method/projection24 cap')
        e=dict(resource=resource,index=len(receipts),completed=False);receipts.append(e)
        sink(dict(kind='prediction_attempt',receipt=deepcopy(e)));return e
    def end(e):sync();e['completed']=True;sink(dict(kind='prediction_completed',receipt=deepcopy(e)))
    def call(*a,**kw):e=begin('model.predict');out=original(*a,**kw);end(e);return out
    def pre(module,args):
        nonlocal pending
        x=args[0];require(pending is None and tuple(x.shape)==(3,192) and x.device.type=='cuda','native flattened pred_proj3tokens')
        pending=begin('pred_proj');pending['tokens']=3
    def post(module,args,out):
        nonlocal pending
        require(pending is not None and out is not None,'pred_proj returned');end(pending);pending=None
    try:
        model.predict=call;hooks.append(model.pred_proj.register_forward_pre_hook(pre));hooks.append(model.pred_proj.register_forward_hook(post))
        yield receipts
        require(pending is None,'no unfinished projection')
    finally:
        for hook in reversed(hooks):hook.remove()
        if owned:model.predict=original
        else:delattr(model,'predict')


def run(prepared,expected,output):
    p=verify(prepared,expected);store=render.Store(output);errors=[];model=None;before=after=host=None;source_ok=False
    start=time.monotonic();method=[];requests=[];outputs=[];groups=[];pred=None;scope={}
    images=[dict(**item,status='not_started') for item in p['images']]
    rows=[dict(index=i,branch=b,view=v,control=c,status='not_started') for i,(b,v,c) in enumerate((b,v,c) for b in range(4) for v in VIEWS for c in CONTROLS)]
    def event(e):store.event(dict(scope=deepcopy(scope),event=e))
    visual=enc.Ledger(8,event)
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        def check():
            store.check();require(time.monotonic()-start<600,'wall600 before dispatch')
            require(torch.cuda.memory_allocated(device)<=contract()['gpu_allocated_cap'] and torch.cuda.memory_reserved(device)<=contract()['gpu_reserved_cap'],'GPU3/4GiB caps')
        def sync():torch.cuda.synchronize(device);check()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen tensors')
        visual.sync=sync;visual.no_grad=lambda:not torch.is_grad_enabled();pred=PredictLedger(event,check);pred.sync=sync
        transform=p0.image_transform();observed={};labels={}
        with sdpa_kernel(SDPBackend.MATH),saved.no_ad(torch,event,errors):
            with enc.guard(model,torch,visual,errors):
                for image in images:
                    scope=dict(image=image['index']);check();image['status']='attempted';visual.image=image['index'];event(dict(kind='image_attempt',image=deepcopy(image)))
                    a=rgb.png(image['png_path'],image['sha256'],image['pixel_sha256'])
                    with torch.inference_mode():
                        x=p0._image_tensor(a,transform=transform,device=device)
                        require(x.device.type=='cuda' and tuple(x.shape)==(1,1,3,224,224),'native batch1 pixels')
                        z=model.encode({'pixels':x})['emb'];require(z.device.type=='cuda' and z.dtype==torch.float32 and tuple(z.shape)==(1,1,192),'native projected192')
                        z=z[0,0].detach().cpu().numpy().copy()
                    enc.native(z);require([(e['resource'],e['completed']) for e in visual.entries if e['image']==image['index']]==[('encoder',True),('projector',True)],'actual all32 visual completions')
                    image.update(status='encoded',latent=saved.save_array(store,f'image_{image["index"]:02}.native192.npy',z))
                    if image['role']=='terminal10':labels[(image['branch'],image['view'])]=z.copy()
                    else:observed[(image['branch'],image['view'],image['role'])]=z.copy()
                    event(dict(kind='image_saved',image=deepcopy(image)));visual.image=None
            requests=make_requests(p['branches'],observed);store.put('requests.json.gz',requests)
            def sink(e):
                nonlocal scope
                if e['kind']=='H1_attempt':
                    r=e['request'];scope={k:r[k] for k in ('index','branch','view','control')};rows[r['index']]['status']='attempted'
                if e['kind']=='H1_completed':
                    o=e['output'];i=o['request_index'];name=f'prediction_{i:02}.json.gz';store.put(name,o)
                    rows[i].update(status='prediction_saved',result_file=name)
                event(e)
            with base.module_guard(model,pred),prediction_receipts(model,event,check,sync) as method:
                outputs=predict_all(requests,NativeBackend(model,device),sink)
            scope=dict(stage='evaluation_only');groups=evaluate(requests,outputs,labels);store.put('fzm_metrics.json.gz',groups)
    except BaseException as exc:
        errors.insert(0,enc.error('execution',exc))
        try:event(dict(kind='execution_error',error=errors[0]))
        except BaseException as exc:errors.append(enc.error('error_journal',exc))
    finally:
        for row in images+rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model)
            except BaseException as exc:errors.append(enc.error('final_model',exc))
        try:verify(prepared,expected);source_ok=True
        except BaseException as exc:errors.append(enc.error('final_source',exc))
        complete=not errors and not visual.blocked and source_ok and before is not None and before==after and len(groups)==8 and all(i['status']=='encoded' for i in images) and all(r['status']=='prediction_saved' for r in rows) and visual.counts()=={k:dict(attempted=32,completed=32) for k in ('encoder','projector')} and pred is not None and pred.counts()=={k:dict(attempted=24,completed=24) for k in ('predict','action_encoder')} and all(sum(e['resource']==r and e['completed'] for e in method)==24 for r in ('model.predict','pred_proj'))
        try:store.close()
        except BaseException as exc:errors.append(enc.error('journal_finalize',exc));complete=False
        report=dict(schema=NAME+'.report',status='paired_view_FZM_recorded_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,contract=p['contract'],images=images,rows=rows,groups=groups,errors=errors,
            resources=dict(**visual.counts(),**(pred.counts() if pred else {})),prediction_method_receipts=method,
            source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before else None,
            host=host,artifact_sha256={f.name:sha(f) for f in store.output.iterdir() if f.is_file()},
            new_physics=0,AD=0 if complete else None,training=False,selection=False,P1_admission=False,P2_admission=False,
            conditioning_ablation=True,counterfactual_physical_truth=False,training_overlay_provenance='unknown')
        store.put('report.json',report,final=True)
    require(complete,'partial retained; no retry: '+str(errors));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run'))
    for k in ('paired-report','paired-sha','prepared','prepare-sha','output'):parser.add_argument('--'+k)
    a=parser.parse_args()
    if a.command=='prepare':r=prepare(a.output,paired_report=a.paired_report,paired_sha=a.paired_sha)
    elif a.command=='verify':r=dict(status='saved_verified_only',images=len(verify(a.prepared,a.prepare_sha)['images']))
    else:r=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(r).decode(),end='')


if __name__=='__main__':main()
