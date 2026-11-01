"""MAIN CUDA-only factual H1 on 32 sealed paired-render images, no selection.

Eight H1 calls, own-view labels; counterfactual action dependence is NOT measured.
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
from experiments.spectral_reachability import p1_constructive_live_overlay_fidelity_20261027_v1 as render

base,enc,rgb,saved=render.base,render.enc,render.rgb,render.saved
ROOT=base.ROOT
NAME=render.NAME+'_cuda'
VIEWS=('off','on')
ROLES=('warmup5','warmup10','prefix5','terminal10')
require,sha,read,canonical=base.require,base.sha,base.read,base.canonical
FLOOR,CAP=render.FLOOR,render.CAP


def contract():
    return dict(branches=4,views=list(VIEWS),images=32,encoder=32,projector=32,predict=8,action_encoder=8,
        context='own view warmup5,warmup10,prefix5; between raw warmup5:10=zeros then original prefix5',
        outgoing='original sealed suffix5',label='own view actual terminal10, evaluator only',
        predictor_input_tokens=24,action_encoder_packet_tokens=24,pred_proj_token_samples=24,
        AD=0,training=False,optimization=False,physics=0,no_dedup=True,no_retry=True,selection=False,
        normalization='frozen public mean/std, no fit, training lineage unknown',
        action_dependence='counterfactual NOT measured: fixed8 factual calls cannot also execute zero/mask controls',
        numerical=dict(SDP='MATH',TF32=False),wall_seconds=600,gpu_allocated_cap=3*1024**3,gpu_reserved_cap=4*1024**3,
        output_cap_bytes=CAP,reserve_bytes=FLOOR,P1_admission=False,training_overlay_provenance='unknown',
        interpretation='known-task paired native-default marker pilot, own-view fidelity only; not training-contract causal attribution')


def assemble(paired,events,rp):
    require(paired['schema']==render.NAME+'.report' and paired['status']=='live_overlay_pairs_sealed_not_admitted' and
        paired['errors']==[] and paired['source_verified'] is True and paired['contract']==render.contract(),'complete paired render seal')
    require(events[0]['kind']=='run_started' and events[-1]['kind']=='run_finished' and events[-1]['complete'] is True and
        events[-1]['counts']==paired['counts']=={k:dict(attempted=v,completed=v) for k,v in render.contract()['caps'].items()},'renderer full actual counts')
    census={k:dict(attempted=0,completed=0) for k in paired['counts']};stack=[]
    for e in events:
        if e['kind'] in ('operation_attempted','operation_completed'):
            k=e['resource'];attempt=e['kind']=='operation_attempted';census[k]['attempted' if attempt else 'completed']+=1
            if attempt:stack.append(k)
            else:require(stack and stack.pop()==k,'renderer operation nesting')
    require(not stack and census==paired['counts'],'actual events resource reconstruction')
    require(len(paired['rows'])==len(rp['branches'])==4,'all4 branches')
    images=[];branches=[]
    for b,(row,item) in enumerate(zip(paired['rows'],rp['branches'],strict=True)):
        require(row['branch']==b and row['source_slot']==item['source_slot'] and row['arm']==item['arm'] and
            row['commands']==item['commands'] and row['status']=='replayed_and_paired' and row['first_error'] is None and
            row['secondary_errors']==[] and row['off_exact']==22 and row['public_returned']==20,'fixed complete original branch')
        off=[e['receipt'] for e in events if e['kind']=='off_receipt' and e['branch']==b]
        require(off==item['receipts'],'all88 off trajectory receipts, not just selected endpoints')
        pairs=[e['receipt'] for e in events if e['kind']=='pair_completed' and e['receipt']['branch']==b]
        require(row['pairs']==pairs and [(p['phase'],p['step']) for p in pairs]==list(render.SELECTED),'all4 fixed pairs/branch')
        for j,pair in enumerate(pairs):
            proof=pair['proof']
            require(pair['status']=='paired' and pair['source_slot']==item['source_slot'] and pair['arm']==item['arm'] and
                set(pair['views'])==set(VIEWS) and proof['before']==proof['after'] and proof['snapshot_equal'] is True and
                proof['goal_pose']==render.DEFAULT_POSE and proof['order']==['off','on'],'same snapshot/default marker pair')
            original=item['receipts'][render.ALL.index(render.SELECTED[j])]
            require(pair['views']['off']==dict(file=original['png'],sha256=original['file_sha256'],pixel_sha256=original['pixel_sha256']),'selected off original file identity')
        for view in VIEWS:
            for j,pair in enumerate(pairs):
                images.append(dict(index=len(images),branch=b,source_slot=item['source_slot'],arm=item['arm'],view=view,role=ROLES[j],**pair['views'][view]))
        branches.append(dict(branch=b,source_slot=item['source_slot'],arm=item['arm'],commands=item['commands'],selection_digest=item['selection_digest']))
    return images,branches


def build(paired_report,paired_sha):
    path=Path(paired_report).absolute();require(sha(path)==paired_sha,'external actual paired report SHA');r=read(path)
    rp=render.verify(r['prepared'],r['prepare_sha256'])
    bindings=dict(rp['bindings']);bindings.update({str(path):paired_sha,str(Path(r['prepared']).absolute()):r['prepare_sha256']})
    jp=path.parent/'events.jsonl.gz';require(sha(jp)==r['artifact_sha256'][jp.name],'paired events gzip SHA')
    require(jp.stat().st_size<=CAP,'small renderer events');plain,eof=base.inflate(jp.read_bytes(),32*1024**2)
    require(eof and plain.endswith(b'\n'),'complete single gzip journal');events=[base.decode(line) for line in plain.splitlines()]
    images,branches=assemble(r,events,rp);bindings[str(jp)]=sha(jp)
    # Authenticate EVERY saved off file, as well as all32 selected references.
    for item in rp['branches']:
        for c in item['receipts']:
            f=rgb.local(path.parent,c['png']);rgb.png(f,c['file_sha256'],c['pixel_sha256']);bindings[str(f)]=c['file_sha256']
    for image in images:
        f=rgb.local(path.parent,image['file']);rgb.png(f,image['sha256'],image['pixel_sha256'])
        image['png_path']=str(f);bindings[str(f)]=image['sha256']
    norm=enc.native_metadata();require(norm['training_provenance_verified'] is False,'unknown training normalizer lineage')
    bindings.update(enc.MODELS)
    files=render.source_files()+[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py')]
    bindings.update({str(Path(f).resolve()):sha(f) for f in files})
    for f,h in bindings.items():require(sha(f)==h,'source/model/PNG seal: '+f)
    return dict(schema=NAME+'.prepare',arguments=dict(paired_report=str(path),paired_sha=paired_sha),images=images,branches=branches,
        bindings=bindings,normalization=norm,policy=str(enc.POLICY),expected_model_hash=enc.MODEL_TENSOR_SHA,contract=contract())


def prepare(output,**args):
    p=build(**args);value=canonical(p);path=Path(output).absolute()
    require(not path.exists() and len(value)<256*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(value),'exclusive small CUDA prepare/200MiB')
    with path.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(value))


def verify(path,expected):
    require(sha(path)==expected,'external CUDA prepare SHA');p=read(path)
    require(p==build(**p['arguments']),'complete renderer/source/model/images reconstruction');return p


def context(zs,commands):
    require(len(zs)==4,'4 own-view observed frames')
    for z in zs:enc.native(z)
    a=physical_commands(commands)
    # terminal zs[3] is intentionally absent from every model input.
    between=np.concatenate((np.zeros((5,2),np.float32),a[:5]))
    c=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=np.asarray(zs[:3]).astype(float).tolist(),action_history=base.d2p.normalized_history(between).astype(float).tolist())
    base.d2p.validate_context(c);return c,a[5:].copy(),between


def physical_commands(commands):return render.physical.commands(commands)


def diagnose(zs,commands,backend,sink):
    c,outgoing,between=context(zs,commands)
    sink(dict(kind='H1_request',context=c,commands=outgoing.astype(float).tolist(),raw_between=between.astype(float).tolist()))
    after=backend.step(deepcopy(c),outgoing.copy());base.d2p.validate_advance(c,outgoing,after)
    z=np.asarray(after['latent_history'][-1],float);target=np.asarray(zs[3],float);current=np.asarray(zs[2],float)
    mse=lambda a,b:float(np.mean((a-b)**2))
    result=dict(context=c,after=after,commands=outgoing.astype(float).tolist(),raw_between=between.astype(float).tolist(),
        factual_mse=mse(z,target),persistence_mse=mse(current,target),predicted_change_mse=mse(z,current),
        factual_minus_persistence=mse(z,target)-mse(current,target),outgoing_raw_l2=float(np.linalg.norm(outgoing)),
        outgoing_token=base.d2p.normalized_history(outgoing).astype(float).tolist(),
        action_dependence_measured=False,counterfactual_action_control=None,selection=False,P1_admission=False)
    sink(dict(kind='H1_result',result=result));return result


class PredictLedger(saved.PredictLedger):
    def __init__(self,sink,check):super().__init__(sink);self.check=check
    def begin(self,r):
        self.check();require(sum(e['resource']==r for e in self.neural)<8,'fixed8 predispatch cap');return super().begin(r)


@contextmanager
def prediction_receipts(model,sink,check,sync=lambda:None):
    """Actual model.predict entry AND internal pred_proj receipts, not inferred calls."""
    receipts=[];hooks=[];original=model.predict;owned='predict' in model.__dict__;pending=None
    def call(*a,**kw):
        check();e=dict(resource='model.predict',index=len(receipts),completed=False)
        require(sum(x['resource']=='model.predict' for x in receipts)<8,'model.predict8 cap')
        receipts.append(e);sink(dict(kind='method_attempt',receipt=deepcopy(e)))
        out=original(*a,**kw);sync();e['completed']=True;sink(dict(kind='method_completed',receipt=deepcopy(e)));return out
    def pre(module,args):
        nonlocal pending
        # LeWM.predict flattens (B,T,D) -> (B*T,D) before pred_proj.
        check();x=args[0];require(pending is None and tuple(x.shape)==(3,192) and x.device.type=='cuda','pred_proj flattened CUDA3tokens')
        require(sum(x['resource']=='pred_proj' for x in receipts)<8,'pred_proj8 cap')
        pending=dict(resource='pred_proj',index=len(receipts),tokens=3,completed=False);receipts.append(pending)
        sink(dict(kind='module_attempt',receipt=deepcopy(pending)))
    def post(module,args,out):
        nonlocal pending
        require(pending is not None and out is not None,'pred_proj returned')
        sync();pending['completed']=True;sink(dict(kind='module_completed',receipt=deepcopy(pending)));pending=None
    try:
        model.predict=call;hooks.append(model.pred_proj.register_forward_pre_hook(pre));hooks.append(model.pred_proj.register_forward_hook(post))
        yield receipts
        require(pending is None,'no partial pred_proj return')
    finally:
        for hook in reversed(hooks):hook.remove()
        if owned:model.predict=original
        else:delattr(model,'predict')


def run(prepared,expected,output):
    p=verify(prepared,expected);store=render.Store(output);errors=[];model=None;before=after=host=None;source_ok=False
    start=time.monotonic();method=[];images=[dict(**item,status='not_started') for item in p['images']]
    rows=[dict(**b,view=v,status='not_started') for b in p['branches'] for v in VIEWS]
    visual=enc.Ledger(8,store.event);pred=None
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
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'frozen model tensors')
        def sync():torch.cuda.synchronize(device);check()
        visual.sync=sync;visual.no_grad=lambda:not torch.is_grad_enabled();pred=PredictLedger(store.event,check);pred.sync=sync
        transform=p0.image_transform();latents={}
        with sdpa_kernel(SDPBackend.MATH),saved.no_ad(torch,store.event,errors):
            with enc.guard(model,torch,visual,errors):
                for image in images:
                    check();image['status']='attempted';visual.image=image['index'];store.event(dict(kind='image_attempt',image=deepcopy(image)))
                    a=rgb.png(image['png_path'],image['sha256'],image['pixel_sha256'])
                    with torch.inference_mode():
                        x=p0._image_tensor(a,transform=transform,device=device)
                        require(x.device.type=='cuda' and tuple(x.shape)==(1,1,3,224,224),'original batch1 RGB preprocessing')
                        z=model.encode({'pixels':x})['emb'];require(z.device.type=='cuda' and z.dtype==torch.float32 and tuple(z.shape)==(1,1,192),'native192 existing projector')
                        z=z[0,0].detach().cpu().numpy().copy()
                    enc.native(z);require([(e['resource'],e['completed']) for e in visual.entries if e['image']==image['index']]==[('encoder',True),('projector',True)],'no inferred/dedup visual calls')
                    image.update(status='encoded',latent=saved.save_array(store,f'image_{image["index"]:02}.native192.npy',z))
                    latents[(image['branch'],image['view'],image['role'])]=z.copy();store.event(dict(kind='image_saved',image=deepcopy(image)));visual.image=None
            backend=base.d2p.TorchBackend(model,device)
            with base.module_guard(model,pred),prediction_receipts(model,store.event,check,sync) as method:
                for row in rows:
                    row['status']='attempted';b,v=row['branch'],row['view']
                    zs=[latents[(b,v,role)] for role in ROLES]
                    def sink(e):store.event(dict(branch=b,view=v,event=e))
                    result=diagnose(zs,row['commands'],backend,sink);name=f'b{b}_{v}.fidelity.json.gz';store.put(name,result)
                    row.update(status='factual_recorded',result_file=name,metrics={k:result[k] for k in ('factual_mse','persistence_mse','predicted_change_mse','factual_minus_persistence','action_dependence_measured')})
    except BaseException as exc:
        errors.insert(0,enc.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as exc:errors.append(enc.error('error_journal',exc))
    finally:
        for row in images+rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model)
            except BaseException as exc:errors.append(enc.error('final_model',exc))
        try:verify(prepared,expected);source_ok=True
        except BaseException as exc:errors.append(enc.error('final_source',exc))
        complete=not errors and not visual.blocked and source_ok and before is not None and before==after and all(i['status']=='encoded' for i in images) and all(r['status']=='factual_recorded' for r in rows) and visual.counts()=={k:dict(attempted=32,completed=32) for k in ('encoder','projector')} and pred is not None and pred.counts()=={k:dict(attempted=8,completed=8) for k in ('predict','action_encoder')} and all(sum(e['resource']==r and e['completed'] for e in method)==8 for r in ('model.predict','pred_proj'))
        try:store.close()
        except BaseException as exc:errors.append(enc.error('journal_finalize',exc));complete=False
        report=dict(schema=NAME+'.report',status='paired_view_factual_fidelity_recorded_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,contract=p['contract'],images=images,rows=rows,errors=errors,
            resources=dict(**visual.counts(),**(pred.counts() if pred else {})),prediction_method_receipts=method,
            source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before else None,
            host=host,artifact_sha256={f.name:sha(f) for f in store.output.iterdir() if f.is_file()},
            new_physics=0,AD=0 if complete else None,training=False,selection=False,P1_admission=False,
            action_dependence_measured=False,training_overlay_provenance='unknown')
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
