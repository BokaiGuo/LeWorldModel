"""Prepare two saved step20 RGBs for MAIN-only CUDA encoding. Never physics.

prepare/verify import no torch or old runtime. run is a separate explicit CLI.
Two arms of ONE historical run; derived encoder evidence, not P1/P4 admission.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sysconfig

import numpy as np
from PIL import Image

from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG = 'p4_terminal20_cuda_encoding_20261027_v1'
REPO = Path(__file__).resolve().parents[2]
ORIGINAL = REPO/'results/p4_multiround_feedback_runtime_20261027_v1_prepare_v1'
ARMS = ('imagined','actual')
MAX_SMALL = 16*1024*1024
RESERVE = 256*1024
MIN_FREE = 4*1024*1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def read(path):
    require(Path(path).stat().st_size <= MAX_SMALL, 'oversize small input')
    def unique(items):
        d={}
        for k,v in items:
            require(k not in d,'duplicate JSON key'); d[k]=v
        return d
    value=json.loads(Path(path).read_bytes(),object_pairs_hook=unique)
    canonical(value)
    return value


def write(path, value):
    raw=canonical(value); require(len(raw)<=RESERVE,'small output cap')
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def pixels(path, expected_file=None, expected_pixel=None):
    require(Path(path).stat().st_size<=1024*1024,'PNG cap')
    if expected_file is not None: require(sha(path)==expected_file,'PNG file SHA')
    with Image.open(path) as im:
        require(im.mode=='RGB' and im.size==(224,224),'exact original RGB PNG')
        rgb=np.asarray(im).copy()
    require(rgb.shape==(224,224,3) and rgb.dtype==np.uint8,'uint8 RGB')
    digest=hashlib.sha256(rgb.tobytes()).hexdigest()
    if expected_pixel is not None: require(digest==expected_pixel,'PNG pixel SHA')
    return rgb,digest


def contract():
    return dict(historical_runs=1,arms=list(ARMS),primitive_step=20,latent_dim=192,
        encoder_cap=2,existing_projector_cap=2,per_arm_each_cap=1,batch=1,time=1,
        device='cuda:0',cpu_neural_fallback=False,predict_cap=0,AD_cap=0,physics_cap=0,
        transform='original preflight_pusht_d2p_p0._image_tensor + run_pusht_counterfactual.image_transform',
        representation='original model.encode pixels-only -> existing encoder CLS -> existing model.projector -> native192',
        refit=False,retries=False,source_report_overwrite=False,P1_admission=False,P4_admission=False)


def source_files():
    site=Path(sysconfig.get_paths()['purelib'])
    relative=['stable_worldmodel/wm/lewm/lewm.py','stable_worldmodel/wm/lewm/module.py',
        'stable_pretraining/backbone/utils.py','stable_pretraining/data/dataset_stats.py',
        'transformers/models/vit/modeling_vit.py','transformers/models/vit/configuration_vit.py',
        'torchvision/transforms/v2/_container.py','torchvision/transforms/v2/_misc.py',
        'torchvision/transforms/v2/_geometry.py','torchvision/transforms/v2/functional/_misc.py',
        'torchvision/transforms/v2/functional/_geometry.py','torchvision/transforms/v2/functional/_type_conversion.py']
    local=[Path(__file__),REPO/f'tests/test_{TAG}.py',REPO/f'innerdocs/{TAG}.md',
        REPO/'experiments/checkpoint_compat.py',REPO/'experiments/gpu_preflight.py',
        REPO/'experiments/run_pusht_counterfactual.py',
        REPO/'experiments/spectral_reachability/preflight_pusht_d2p_p0.py',
        REPO/'experiments/spectral_reachability/d2p_prerun_source_seal.py']
    return sorted({p.resolve() for p in local+[site/r for r in relative]})


def collect(folder=ORIGINAL):
    """Only saved-data validation and streaming hashes; no large input copies."""
    folder=Path(folder).resolve()
    r,p,prepared=[read(folder/n) for n in ('report.json','protocol.json','prepare_receipt.json')]
    require(r['schema']=='p4_multiround_feedback_runtime_20261027_v1' and
        r['status']=='calibration_recorded' and not r['failures'] and r['model_unchanged'] is True,'complete real runtime required')
    require(r['protocol_sha256']==prepared['protocol_sha256']==sha(folder/'protocol.json'),'original protocol SHA')
    require(p['contract']['planned_steps_per_arm']==20 and p['contract']['boundaries']==[5,10,15], '20step3round only')
    require(sha(folder/'runtime.events.jsonl')==r['journal_sha256'],'original journal SHA')
    require((folder/'runtime.events.jsonl').stat().st_size<=MAX_SMALL,'bounded journal')
    events=[json.loads(line) for line in (folder/'runtime.events.jsonl').read_bytes().splitlines()]
    require([e['event_index'] for e in events]==list(range(len(events))),'journal order')
    root=p['root_sha256']
    require(len(r['rows'])==2 and {v['arm'] for v in r['rows']}==set(ARMS) and all(
        v['root_sha256']==root and v['status']=='completed' and v['completed_packets']==4 and
        v['selected_returns']==v['selected_attempts']==20 for v in r['rows']),'one-root two complete arms')
    goal=np.asarray(p['goal']); q=np.asarray(p['q'])
    require(goal.shape==(192,) and q.shape==(192,192) and np.isfinite(goal).all() and np.isfinite(q).all(),'original goal/Q')
    bindings={str(folder/n):sha(folder/n) for n in ('report.json','protocol.json','prepare_receipt.json','runtime.events.jsonl')}
    endpoints=[]
    for arm in ARMS:
        obs=[e for e in events if e.get('arm')==arm and e['kind']=='observation' and e.get('primitive_step')==20]
        require(len(obs)==1,'unique arm step20 observation')
        e=obs[0]; path=Path(e['rgb_path']).resolve()
        _,pixel=pixels(path,e['rgb_sha256'],e['pixel_sha256'])
        bindings[str(path)]=sha(path)
        endpoints.append(dict(root_sha256=root,arm=arm,primitive_step=20,event_index=e['event_index'],
            rgb_path=str(path),rgb_file_sha256=e['rgb_sha256'],pixel_sha256=pixel,state=e['state']))
    policy=Path(p['policy']).resolve()
    model_bindings={str(policy/n):p['bindings'][str(policy/n)] for n in ('weights.pt','config.json')}
    for path,h in model_bindings.items(): require(sha(path)==h,'original frozen checkpoint binding: '+path)
    config=read(policy/'config.json')
    require(config['_target_']=='stable_worldmodel.wm.lewm.LeWM' and
            config['encoder'].get('pretrained') is False and
            config['projector']['_target_']=='stable_worldmodel.wm.lewm.module.MLP',
            'original local-only model/projector configuration required')
    # Compare reused functions against HISTORICAL source seals, without copying
    # the large historical protocol or sealing a newly modified old runner.
    old_sources={}
    for seal in prepared['source_seals']:
        manifest=Path(seal['output_dir'])/'manifest.json'
        require(sha(manifest)==seal['manifest_sha256'],'historical manifest SHA')
        bindings[str(manifest)]=sha(manifest)
        for v in read(manifest)['files']: old_sources[v['source']]=v['sha256']
    historical={}
    for name in ('experiments/checkpoint_compat.py','experiments/run_pusht_counterfactual.py',
                 'experiments/spectral_reachability/preflight_pusht_d2p_p0.py'):
        path=str(REPO/name)
        require(path in old_sources and sha(path)==old_sources[path],'historical loader/transform drift: '+path)
        historical[path]=old_sources[path]
    return dict(schema=TAG+'.protocol',contract=contract(),source_report=str(folder/'report.json'),
        source_report_sha256=sha(folder/'report.json'),source_run=str(folder),root_sha256=root,
        endpoints=endpoints,policy=str(policy),model_bindings=model_bindings,bindings=bindings,
        historical_source_bindings=historical,goal_q_source=str(folder/'protocol.json'),
        goal_q_sha256=hashlib.sha256(canonical([p['goal'],p['q']])).hexdigest())


def prepare(output):
    output=Path(output).resolve(); require(not output.exists(),'new exclusive prepare directory')
    require(shutil.disk_usage(output.parent).free>=MIN_FREE,'insufficient preparation headroom')
    p=collect(); sources=source_files()
    p['sources_sha256']={str(f):sha(f) for f in sources}
    output.mkdir(exist_ok=False)
    write(output/'protocol.json',p)
    seal=create_seal(files=sources,args_json=json.dumps(dict(protocol_sha256=sha(output/'protocol.json'),contract=contract())),output_dir=output/'source_seal')
    receipt=dict(schema=TAG+'.prepare',status='prepared_not_executed',protocol_sha256=sha(output/'protocol.json'),
        source_seal=seal,encoder_calls=0,projector_calls=0,predict_calls=0,AD_calls=0,physics_steps=0,
        P1_admission=False,P4_admission=False)
    write(output/'prepare.json',receipt)
    return receipt


def verify(prepared,expected_sha):
    prepared=Path(prepared).resolve()
    require(sha(prepared/'prepare.json')==expected_sha,'prepare receipt SHA')
    r=read(prepared/'prepare.json'); p=read(prepared/'protocol.json')
    require(r['status']=='prepared_not_executed' and sha(prepared/'protocol.json')==r['protocol_sha256'] and
            p['contract']==contract() and p['schema']==TAG+'.protocol','prepared contract')
    seal=r['source_seal']; verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    for path,h in {**p['bindings'],**p['model_bindings'],**p['sources_sha256']}.items():
        require(sha(path)==h,'bound input/source changed: '+path)
    rebuilt=collect(p['source_run']); rebuilt['sources_sha256']=p['sources_sha256']
    require(rebuilt==p,'saved endpoint/model/goal/Q reconstruction')
    return p


class Ledger:
    """Nonrefundable pre-dispatch global2/per-arm1 caps with failed entries."""
    def __init__(self,journal,synchronize=lambda:None):
        self.journal,self.synchronize=journal,synchronize
        self.entries=[]; self.blocked=[]; self.arm=None

    def dispatch(self,kind,tensor,callback,args,kwargs):
        try:
            require(self.arm in ARMS and kind in ('encoder','existing_projector'),'explicit arm/entry kind')
            require(getattr(getattr(tensor,'device',None),'type',None)=='cuda','CPU neural forbidden')
            require(tuple(tensor.shape)==((1,3,224,224) if kind=='encoder' else (1,192)),'single-frame native input')
            same=[e for e in self.entries if e['kind']==kind]
            require(len(same)<2 and not any(e['arm']==self.arm for e in same),'entry cap before dispatch')
            require(all(e['completed'] for e in self.entries),'failed entry cannot be retried')
            expected=[(arm,k) for arm in ARMS for k in ('encoder','existing_projector')]
            require(len(self.entries)<4 and (self.arm,kind)==expected[len(self.entries)],'encoder/projector dispatch sequence')
        except BaseException as exc:
            self.blocked.append(dict(kind=kind,arm=self.arm,error=str(exc)))
            self.journal(dict(event='blocked',**self.blocked[-1])); raise
        e=dict(kind=kind,arm=self.arm,index=len(self.entries),completed=False)
        self.entries.append(e); self.journal(dict(event='attempt',**e))
        try:
            result=callback(*args,**kwargs); self.synchronize(); e['completed']=True
            return result
        except BaseException as exc:
            e['error']=dict(type=type(exc).__name__,message=str(exc)); raise
        finally:
            self.journal(dict(event='return_or_error',**deepcopy(e)))

    def forbid(self,kind):
        def denied(*args,**kwargs):
            self.blocked.append(dict(kind=kind,arm=self.arm,error='forbidden entry'))
            self.journal(dict(event='blocked',**self.blocked[-1])); raise RuntimeError('forbidden '+kind)
        return denied


def encode_endpoints(p,encode,save,ledger,journal,rows=None,errors=None):
    """Non-neural-testable sequencing; stop after first error, retain all slots."""
    rows=[] if rows is None else rows
    errors=[] if errors is None else errors
    for endpoint in p['endpoints']:
        row=dict(endpoint,status='blocked_not_attempted',latent_path=None)
        if not errors:
            ledger.arm=endpoint['arm']; row['status']='attempted'
            try:
                native=np.asarray(encode(endpoint))
                require(native.shape==(192,) and native.dtype==np.float32 and np.isfinite(native).all(),'finite native float32 latent192')
                # Persist native latent BEFORE metric arithmetic or next endpoint.
                path,digest=save(endpoint['arm'],native)
                row.update(latent_path=str(path),latent_sha256=digest,latent_dtype='float32',latent_shape=[192])
                residual=np.asarray(p['q'])@(np.asarray(p['goal'])-native.astype(float))
                row.update(status='encoded_and_saved',original_Q_cost=float(np.mean(residual**2)))
            except BaseException as exc:
                errors.append(dict(stage='endpoint',arm=endpoint['arm'],type=type(exc).__name__,message=str(exc)))
                row.update(status='failed',error=errors[-1])
            finally:
                ledger.arm=None
        rows.append(row); journal(dict(event='endpoint',row=deepcopy(row)))
    return rows,errors


def tensor_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        a=value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(tuple(a.shape)).encode())
        h.update(a.numpy().tobytes())
    return h.hexdigest()


def patch(stack,obj,name,value):
    old=getattr(obj,name); setattr(obj,name,value)
    stack.callback(setattr,obj,name,old)


def execute_cuda(prepared,expected_sha,output):
    p=verify(prepared,expected_sha)
    output=Path(output).resolve(); require(not output.exists(),'new exclusive derived output')
    require(shutil.disk_usage(output.parent).free>=MIN_FREE,'insufficient execution/final-receipt reserve')
    # One launch per preparation: no retry/refund after even partial dispatch.
    write(Path(prepared)/'launch.json',dict(output=str(output),prepare_sha256=expected_sha))
    output.mkdir(exist_ok=False)
    # Materially allocate final receipt bytes before any CUDA/model setup.
    with (output/'receipt.json').open('xb') as stream:
        os.posix_fallocate(stream.fileno(),0,RESERVE)
    write(output/'execution_binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected_sha))
    errors=[]; rows=[]; before=after=None; host=None; model=None
    with (output/'encoding.events.jsonl').open('x') as stream:
        def journal(value):
            raw=json.dumps(value,sort_keys=True,allow_nan=False)+'\n'
            require(stream.tell()+len(raw.encode())<=128*1024,'journal cap')
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        ledger=Ledger(journal)
        try:
            import torch
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
            host=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
            torch.backends.cuda.matmul.allow_tf32=False
            model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
            require(all(v.device.type=='cuda' and not v.requires_grad for v in model.parameters()) and
                    all(not m.training for m in model.modules()),'frozen CUDA eval model')
            before=tensor_hash(model); transform=p0.image_transform()
            ledger.synchronize=lambda:torch.cuda.synchronize(device)
            with ExitStack() as stack:
                for kind,module in (('encoder',model.encoder),('existing_projector',model.projector)):
                    original=module.forward
                    def counted(*args,_kind=kind,_original=original,**kwargs):
                        return ledger.dispatch(_kind,args[0],_original,args,kwargs)
                    patch(stack,module,'forward',counted)
                for obj,name,kind in ((model,'predict','predict'),(model.predictor,'forward','predictor'),
                    (model.action_encoder,'forward','action_encoder'),(model.pred_proj,'forward','pred_proj'),
                    (torch.autograd,'grad','AD.grad'),(torch.autograd,'backward','AD.backward'),
                    (torch.Tensor,'backward','AD.Tensor.backward')):
                    patch(stack,obj,name,ledger.forbid(kind))
                for name in ('jacobian','hessian','jvp','vjp','hvp','vhp'):
                    patch(stack,torch.autograd.functional,name,ledger.forbid('AD.'+name))
                for name in ('jvp','vjp','jacrev','jacfwd','hessian','grad','grad_and_value'):
                    patch(stack,torch.func,name,ledger.forbid('AD.func.'+name))
                def encode(e):
                    rgb,_=pixels(e['rgb_path'],e['rgb_file_sha256'],e['pixel_sha256'])
                    with torch.inference_mode():
                        x=p0._image_tensor(rgb,transform=transform,device=device)
                        require(x.device.type=='cuda' and tuple(x.shape)==(1,1,3,224,224),'exact CUDA image tensor')
                        z=model.encode({'pixels':x})['emb']
                        require(z.device.type=='cuda' and tuple(z.shape)==(1,1,192) and z.dtype==torch.float32 and
                                bool(torch.isfinite(z).all()),'native192 CUDA output')
                        require([(e['kind'],e['completed']) for e in ledger.entries if e['arm']==ledger.arm]==
                                [('encoder',True),('existing_projector',True)],'actual endpoint entry census')
                        return z[0,0].detach().cpu().numpy().copy()
                def save(arm,z):
                    path=output/f'{arm}.terminal20.native192.npy'
                    with path.open('xb') as target:
                        np.save(target,z,allow_pickle=False); target.flush(); os.fsync(target.fileno())
                    return path,sha(path)
                metric=read(p['goal_q_source'])
                work=dict(p,goal=metric['goal'],q=metric['q'])
                encode_endpoints(work,encode,save,ledger,journal,rows,errors)
            torch.cuda.synchronize(device)
        except BaseException as exc:
            errors.append(dict(stage='setup_or_execution',type=type(exc).__name__,message=str(exc)))
        finally:
            if model is not None:
                try:
                    after=tensor_hash(model)
                    require(all(not m.training for m in model.modules()) and all(not v.requires_grad and v.device.type=='cuda'
                        for v in model.parameters()),'final frozen CUDA eval model')
                except BaseException as exc: errors.append(dict(stage='final_model_hash',type=type(exc).__name__,message=str(exc)))
            try: verify(prepared,expected_sha)
            except BaseException as exc: errors.append(dict(stage='final_input_verify',type=type(exc).__name__,message=str(exc)))
            for endpoint in p['endpoints'][len(rows):]: rows.append(dict(endpoint,status='blocked_not_attempted',latent_path=None))
            if before is None or before!=after: errors.append(dict(stage='model_integrity',message='model hash unavailable or changed'))
            counts={kind:dict(attempted=sum(e['kind']==kind for e in ledger.entries),
                completed=sum(e['kind']==kind and e['completed'] for e in ledger.entries)) for kind in ('encoder','existing_projector')}
            complete=not errors and not ledger.blocked and all(c==dict(attempted=2,completed=2) for c in counts.values()) and all(r['status']=='encoded_and_saved' for r in rows)
            if not complete and not errors:
                errors.append(dict(stage='entry_integrity',message='incomplete entry/endpoint census or forbidden attempts'))
            receipt=dict(schema=TAG+'.derived_receipt',status='two_terminal20_latents_saved' if complete else 'partial_or_failed',
                source_report=p['source_report'],source_report_sha256=p['source_report_sha256'],
                root_sha256=p['root_sha256'],historical_runs=1,rows=rows,counts=counts,entries=ledger.entries,
                blocked=ledger.blocked,first_error=errors[0] if errors else None,errors=errors,
                forbidden_attempts={kind:sum(e['kind']==kind for e in ledger.blocked) for kind in sorted({e['kind'] for e in ledger.blocked})},
                journal_sha256=sha(output/'encoding.events.jsonl'),
                prepare_sha256=expected_sha,goal_q_sha256=p['goal_q_sha256'],model_bindings=p['model_bindings'],
                model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
                host=host,predict_calls=0,AD_calls=0,physics_steps=0,training=False,
                P1_admission=False,P4_admission=False,efficacy_verified=False,
                consumer_key=['source_report_sha256','root_sha256','arm','primitive_step'],
                note='new derived encoding of old real RGB; not a new rollout, batch, or independent root')
            raw=canonical(receipt); require(len(raw)<=RESERVE,'reserved receipt cap')
            with (output/'receipt.json').open('r+b') as target:
                target.write(raw); target.truncate(); target.flush(); os.fsync(target.fileno())
    if not complete: raise RuntimeError('terminal encoding incomplete; see NEW derived receipt, no retry')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify','run'))
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--prepare-sha')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.command=='prepare': result=prepare(args.output)
    elif args.command=='verify': result={'status':'verified_not_executed','root_sha256':verify(args.prepared,args.prepare_sha)['root_sha256']}
    else: result=execute_cuda(args.prepared,args.prepare_sha,args.output)
    print(json.dumps(result if args.command!='run' else {'status':result['status'],'counts':result['counts']},sort_keys=True))
