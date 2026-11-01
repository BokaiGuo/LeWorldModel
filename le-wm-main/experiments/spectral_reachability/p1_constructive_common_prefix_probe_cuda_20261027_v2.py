"""Common-prefix CUDA fidelity v2: default exact authentication, formal MATH.

4 default warm authentications + 92 MATH images + 84 MATH H1 predictions.
No tolerance gate, outcomes-based protocol change, new AD, training or physics.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_common_prefix_probe_cuda_20261027_v1 as prior

NAME='p1_constructive_common_prefix_probe_cuda_20261027_v2'
ROOT=prior.ROOT; base=prior.base; enc=prior.enc; physical=prior.physical
require,sha,read,canonical=prior.require,prior.sha,prior.read,prior.canonical
V1_SHA='a29fe7a195de3c9b4ee0b1b6f45a67dad0af5c6e6595337291fb0ac2f00e9713'
PHASES=('anchor_default','image_math','predict_math')


def contract():
    c=deepcopy(prior.contract()); c.update(images=96,formal_images=92,authentication_images=4,
        caps=dict(encoder=96,projector=96,predict=84,action_encoder=84),
        phase_caps=dict(anchor_default=dict(encoder=4,projector=4),image_math=dict(encoder=92,projector=92),
            predict_math=dict(action_encoder=84,predict=84)),
        authentication='4 original warm PNGs, exact float32 bits vs approved task under original default SDP; no tolerance',
        context='MATH actual [warm5,warm10,prefix5] only; normalized [raw zero5,own raw prefix5]; default authentication outputs NOT context inputs',
        image_order='4 extra default warm5/warm10 root order, then unchanged92 formal MATH slots',
        numerical=dict(anchor_default=dict(SDP='DEFAULT_ALL_ENABLED',matmul_TF32=False,cudnn_TF32=True),
            formal=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),batch=1,dtype='float32'),
        backend_difference='MATH warm minus original/default reported without acceptance tolerance; authentication remains strict',
        v1_preserved=True,scope='numerical protocol implementation repair; not a model negative result or P1 admission')
    return c


def material(**arguments):
    require(sha(prior.__file__)==V1_SHA,'frozen v1 primitive source')
    p,_=prior.material(**arguments)
    # Authenticate the old encoder's original pixel identities, without loading blind NPZ/certificates.
    pb=read(Path(p['arguments']['physical_report']).parent/'execution_binding.json'); pp=read(pb['prepared'])
    parent=read(pp['arguments']['planner_report']); slots=read(parent['prepared'])['slots']; proofs=[]
    for k,slot in enumerate(slots):
        export=read(slot['task_export']); tc=export['task_contract']
        bp=Path(slot['task_export']).parent/'binding.json'; binding=read(bp)
        require(binding['prepare_sha256']==tc['prepare_sha256'] and sha(binding['prepared'])==tc['prepare_sha256'],
            'original encoder prepare identity from approved task')
        original=read(binding['prepared']); s=original['slots'][k]
        require(original['schema']==enc.NAME+'.prepare' and s['order']==s['source_slot']==export['source_slot']==k and
            s['blind_sha256']==export['blind_sha256']==tc['blind_source_sha256'],'same original blind slot')
        for offset in (0,1):
            item=p['images'][2*k+offset]; require(item['available'],'fixed approved actual warm PNG required')
            require(item['role']==('warm5','warm10')[offset] and item['checkpoint']['pixel_sha256']==s['pixel_sha256'][offset+1],
                'actual warm PNG bytes must equal originally proved blind input pixels')
            proofs.append(dict(index=2*k+offset,root_slot=k,role=item['role'],pixel_sha256=s['pixel_sha256'][offset+1],
                original_encoder_prepare_sha256=tc['prepare_sha256'],blind_sha256=s['blind_sha256'],task_id=export['task_id']))
        p['bindings'][str(bp.resolve())]=sha(bp); p['bindings'][str(Path(binding['prepared']).resolve())]=tc['prepare_sha256']
    p.update(schema=NAME+'.prepare',contract=contract(),warm_pixel_proofs=proofs)
    for f in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        p['bindings'][str(f.resolve())]=sha(f)
    for f,h in p['bindings'].items(): require(sha(f)==h,'v2 final byte binding '+f)
    return p


def prepare(output,**arguments):
    p=material(**arguments); value=canonical(p); f=Path(output).absolute()
    require(not f.exists() and len(value)<=2*1024**2 and shutil.disk_usage(f.parent).free>=prior.FLOOR+len(value),'NEW small prepare/200MiB floor')
    with f.open('xb') as stream: require(stream.write(value)==len(value),'short prepare'); stream.flush(); os.fsync(stream.fileno())
    physical.syncdir(f.parent); return dict(prepared=str(f),prepare_sha256=sha(f),bytes=len(value))


def verify(path,expected):
    require(sha(path)==expected,'external v2 prepare SHA'); p=read(path,2*1024**2)
    require(p==material(**p['arguments']),'full v2 source/pixel/task/protocol reconstruction'); return p


class Ledger(prior.Ledger):
    def call(self,resource,fn):
        require(not self.failed and self.scope is not None and self.scope[0] in PHASES,'scoped phase/no retry')
        phase,index=self.scope; cap=contract()['phase_caps'][phase]
        require(resource in cap and type(index) is int and 0<=index<(4 if phase=='anchor_default' else 92 if phase=='image_math' else 84),
            'phase resource and fixed slot range')
        expected=('action_encoder','predict') if phase=='predict_math' else ('encoder','projector')
        current=[e['resource'] for e in self.entries if e['scope']==list(self.scope)]
        require(len(current)<2 and resource==expected[len(current)] and all(e['completed'] for e in self.entries),'exact nonretry per-slot order')
        require(sum(e['resource']==resource and e['phase']==phase for e in self.entries)<cap[resource] and
            sum(e['resource']==resource for e in self.entries)<contract()['caps'][resource],'predispatch phase/total caps')
        e=dict(index=len(self.entries),scope=list(self.scope),phase=phase,resource=resource,completed=False); self.entries.append(e)
        try:
            self.sink(dict(kind='module_attempt',entry=deepcopy(e))); result=fn(); self.sync(); e['completed']=True
            self.sink(dict(kind='module_completed',entry=deepcopy(e))); return result
        except BaseException as exc:
            self.failed=True
            try: self.sink(dict(kind='module_error',entry=deepcopy(e),error=str(exc)))
            except BaseException as secondary: exc.add_note('journal error '+str(secondary))
            raise
    def phase_counts(self):
        return {phase:{k:dict(attempted=sum(e['phase']==phase and e['resource']==k for e in self.entries),
            completed=sum(e['phase']==phase and e['resource']==k and e['completed'] for e in self.entries))
            for k in contract()['phase_caps'][phase]} for phase in PHASES}


@contextmanager
def numerical_phase(torch,attention,phase,sink):
    require(phase in PHASES,'fixed numerical phase'); old=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    default=phase=='anchor_default'; primary=None
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=default
    allowed=[attention.SDPBackend.MATH,attention.SDPBackend.FLASH_ATTENTION,attention.SDPBackend.EFFICIENT_ATTENTION,
        attention.SDPBackend.CUDNN_ATTENTION] if default else attention.SDPBackend.MATH
    try:
        with attention.sdpa_kernel(allowed):
            flags={name:bool(getattr(torch.backends.cuda,name+'_sdp_enabled')()) for name in ('math','flash','mem_efficient','cudnn')}
            require(flags==dict(math=True,flash=default,mem_efficient=default,cudnn=default),'actual SDP backend enable flags')
            sink(dict(kind='numerical_phase_enter',phase=phase,SDP_enabled=flags,matmul_TF32=False,cudnn_TF32=default))
            yield
    except BaseException as exc: primary=exc; raise
    finally:
        try: torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=old
        except BaseException as exc:
            if primary is not None: primary.add_note('restore TF32 '+str(exc))
            else: raise


def exact_bits(z,expected): return physical.bits(z,'<f4')==physical.bits(expected,'<f4')


def encode_slot(item,phase,encode,ledger,store,row):
    row['status']='attempted'; ledger.scope=(phase,item['index']); z=encode(item,phase); enc.native(z)
    # Persist output BEFORE any subsequent authentication gate.
    row.update(status='encoded',phase=phase,native192=z.astype(float).tolist(),float32_bits=physical.bits(z,'<f4'))
    store.event(dict(kind='image_saved',receipt=deepcopy(row)))
    require([(e['resource'],e['completed']) for e in ledger.entries if e['scope']==list(ledger.scope)]==
        [('encoder',True),('projector',True)],'actual two module completions per image')
    return z.copy()


def execute(p,encode,predict,ledger,store,anchors,image_rows,rows,differences):
    # Authentication stage: same four PNGs as formal warm inputs, original numerical backend.
    for item,row in zip(p['images'][:4],anchors,strict=True):
        require(item['available'],'original warm PNG missing; no substitution')
        z=encode_slot(item,'anchor_default',encode,ledger,store,row)
        task=p['tasks'][item['root_slot']]; target=np.asarray(task['root_context']['latent_history'][1+item['index']%2],np.float32)
        row['exact_original_task']=exact_bits(z,target)
        store.event(dict(kind='original_anchor_authenticated',receipt=deepcopy(row),expected_float32_bits=physical.bits(target,'<f4')))
        require(row['exact_original_task'],'DEFAULT original warm must match approved task EXACT bits; no tolerance')
    zs={}; predicted=[None]*84; labels=[None]*84
    for item,row in zip(p['images'],image_rows,strict=True):
        if not item['available']: row['status']='missing'; continue
        z=encode_slot(item,'image_math',encode,ledger,store,row); zs[item['index']]=z
        if item['index']<4:
            old=np.asarray(anchors[item['index']]['native192'],np.float32); delta=z.astype(float)-old.astype(float)
            diff=dict(index=item['index'],root_slot=item['root_slot'],role=item['role'],
                math_vs_original_exact_bits=exact_bits(z,old),max_abs=float(np.max(np.abs(delta))),
                mse=float(np.mean(delta*delta)),math_minus_original=delta.tolist(),acceptance_tolerance=None,
                default_anchor_authenticated=anchors[item['index']]['exact_original_task'])
            differences.append(diff); store.event(dict(kind='warm_backend_difference',receipt=deepcopy(diff)))
    for pos in p['positions']:
        pid=pos['position_id']; root=pos['root_slot']; task=p['tasks'][root]; ids=(2*root,2*root+1,4+pid)
        if not all(i in zs for i in ids):
            for row in rows[21*pid:21*pid+21]: row['status']='missing_observed_context'
            continue
        context=prior.observed_context(task,*(zs[i] for i in ids),pos['prefix_commands'])
        store.event(dict(kind='observed_MATH_context_frozen',position_id=pid,context=context,
            default_anchors_authenticated=[anchors[2*root+i]['exact_original_task'] for i in (0,1)]))
        for k in range(21):
            index=21*pid+k; row=rows[index]; ledger.scope=('predict_math',index); row['status']='attempted'
            commands=physical.raw5(p['contract']['probes'][k]['commands']); after=predict(deepcopy(context),commands.copy())
            base.d2p.validate_advance(context,commands,after)
            require([(e['resource'],e['completed']) for e in ledger.entries if e['scope']==list(ledger.scope)]==
                [('action_encoder',True),('predict',True)],'actual H1 module completions')
            predicted[index]=prior.vector(after['latent_history'][-1]).tolist()
            row.update(status='predicted',phase='predict_math',input_context=deepcopy(context),commands=commands.astype(float).tolist(),prediction_context=after)
            store.event(dict(kind='H1_saved',receipt=deepcopy(row)))
    for i,row in enumerate(p['rows']):
        if row['terminal_image'] in zs: labels[i]=zs[row['terminal_image']].astype(float).tolist()
    ledger.scope=None; return prior.arithmetic(p,predicted,labels)


def run(prepared,expected,output):
    p=verify(prepared,expected); store=base.Store(output); ledger=Ledger(store.event); errors=[]; model=None
    before=after=host=None; source_ok=False; analysis=None; differences=[]
    anchors=[dict(index=i,phase='anchor_default',status='not_started',provenance=deepcopy(p['images'][i])) for i in range(4)]
    images=[dict(index=x['index'],phase='image_math',status='not_started',provenance=x) for x in p['images']]
    rows=[dict(x,status='not_started') for x in p['rows']]
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn import attention
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model); before=enc.tensor_hash(model); require(before==p['expected_model_hash'],'original frozen model')
        transform=p0.image_transform(); backend=base.d2p.TorchBackend(model,device); ledger.sync=lambda:torch.cuda.synchronize(device)
        def encode(item,phase):
            c=item['checkpoint']; pixels=prior.rgb.png(item['png_path'],c['file_sha256'],c['pixel_sha256'])
            with numerical_phase(torch,attention,phase,store.event):
                x=p0._image_tensor(pixels,transform=transform,device=device); z=model.encode({'pixels':x})['emb']
                require(z.dtype==torch.float32 and tuple(z.shape)==(1,1,192),'original native CLS192')
                return z[0,0].detach().cpu().numpy().copy()
        def predict(context,commands):
            with numerical_phase(torch,attention,'predict_math',store.event): return backend.step(context,commands)
        with torch.inference_mode(),prior.guard(model,torch,ledger):
            analysis=execute(p,encode,predict,ledger,store,anchors,images,rows,differences)
    except BaseException as exc:
        errors.append(enc.error('execution',exc))
        try: store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary: errors.append(enc.error('error_journal',secondary))
    finally:
        ledger.scope=None
        for row in anchors+images+rows:
            if row['status']=='attempted': row['status']='failed_partial'
        if model is not None:
            try: enc.frozen(model); after=enc.tensor_hash(model)
            except BaseException as exc: errors.append(enc.error('final_model',exc))
        try: verify(prepared,expected); source_ok=True
        except BaseException as exc: errors.append(enc.error('final_source',exc))
        try: store.close()
        except BaseException as exc: errors.append(enc.error('final_journal',exc))
        full=all(x['status']=='encoded' and x.get('exact_original_task') is True for x in anchors) and all(x['status']=='encoded' for x in images) and all(x['status']=='predicted' for x in rows)
        if full:
            expected_phases={phase:{k:dict(attempted=v,completed=v) for k,v in caps.items()} for phase,caps in contract()['phase_caps'].items()}
            if ledger.phase_counts()!=expected_phases or ledger.counts()!={k:dict(attempted=v,completed=v) for k,v in contract()['caps'].items()}:
                errors.append(dict(stage='final_counts',message='actual phase/total counts differ from96/84'))
        closed=not errors and not ledger.failed and source_ok and before is not None and before==after and analysis is not None
        result=dict(schema=NAME+'.report',status=('common_prefix_native_fidelity_completed_not_admitted' if full else 'partial_saved_data_evaluated_not_admitted') if closed else 'partial_or_failed',
            inputs=p['arguments'],prepare_sha256=expected,anchors=anchors,images=images,rows=rows,analysis=analysis,
            warm_backend_differences=differences,resources=ledger.counts(),phase_resources=ledger.phase_counts(),errors=errors,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,
            source_verified=source_ok,host=host,contract=p['contract'],inherited_costs=p['inherited_costs'],new_AD=0,new_physics=0,
            P1_admission=False,P2_admission=False,training=False)
        hashes={}
        for f in store.output.iterdir():
            if f.is_file():
                try: hashes[f.name]=sha(f)
                except BaseException as exc: errors.append(enc.error('artifact_hash',exc)); closed=False
        result['artifact_sha256']=hashes
        if not closed: result['status']='partial_or_failed'
        try: store.put('report.json',result,final=True)
        except BaseException as exc: errors.append(enc.error('final_report',exc)); print(canonical(dict(errors=errors)).decode(),flush=True); closed=False
    require(closed,'failed partial retained; no retry '+str(errors)); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=('prepare','verify','run'))
    for key in ('physical-report','physical-sha','prepared','prepare-sha','output'): parser.add_argument('--'+key)
    a=parser.parse_args()
    if a.command=='prepare': result=prepare(a.output,physical_report=a.physical_report,physical_sha=a.physical_sha)
    elif a.command=='verify': result=dict(status='verified_no_execution',contract=verify(a.prepared,a.prepare_sha)['contract'])
    else:
        r=run(a.prepared,a.prepare_sha,a.output); result={k:v for k,v in r.items() if k not in ('rows','images','anchors','analysis','warm_backend_differences')}
    print(canonical(result).decode(),end='')


if __name__=='__main__': main()
