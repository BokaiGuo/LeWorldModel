"""Fixed model-only multiradius H1 capability responses; MAIN HOST CUDA only.

Consumes only source-bound model contexts/predictions and original approved tasks.
No physical endpoint labels enter this experiment; no encoding/AD/selection.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_common_prefix_probe_cuda_20261027_v2 as parent

NAME='p1_constructive_common_prefix_multiradius_cuda_20261027_v1'
ROOT=parent.ROOT; base=parent.base; enc=parent.enc; core=enc.core; physical=parent.physical
require,sha,read,canonical=parent.require,parent.sha,parent.read,parent.canonical
PARENT_SHA='73959e915161c3e756068bc30606c7c14a84c9806b52d47ca918b1dbcf598b46'
SOURCE_SHA='bbcc1d1c1dffab6f956fa9e7371217a31a642293ccbb43bb7328d23eff7a83f2'
RHOS=(.0625,.125,.25,.5)
CAP,FLOOR,FINAL=16*1024**2,200*1024**2,512*1024


def bank():
    out=[dict(index=0,rho=None,coordinate=None,sign=0,commands=np.zeros((5,2)).tolist(),logits=np.zeros(10).tolist(),reference_index=0)]
    for rho in RHOS:
        for j in range(10):
            for sign in (1,-1):
                x=np.zeros(10,np.float32); x[j]=sign*rho
                require(float(np.linalg.norm(x.astype(float)))<=physical.LOGIT_CAP,'original packet ball')
                out.append(dict(index=len(out),rho=rho,coordinate=j,sign=sign,logits=x.astype(float).tolist(),
                    commands=np.tanh(x).reshape(5,2).astype(float).tolist(),reference_index=1+2*j+(sign==-1) if rho==.25 else None))
    require(len(out)==81,'fixed81 actual H1 probes'); return out


def contract():
    return dict(roots=2,positions=4,radii=list(RHOS),nonzero_per_radius=20,shared_zero_per_position=1,
        per_position=81,planned=324,caps=dict(predict=324,action_encoder=324),encoder=0,projector=0,
        AD=0,J=0,VJP=0,SVD=0,solve=0,physics=0,training=False,selection=False,
        order='position0..3; zero; rho ascending; coordinate0..9 plus then minus',
        action='original raw time-major5x2 float32 tanh once; no scaling/projection/recentering/clip',
        direction='same approved blindtask goal-original_root latent per root, fixed across positions/radii',
        numerical=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False,batch=1,dtype='float32'),
        source_repeat='zero and rho .25 all21 per position must reproduce parent full H1 context exact float32 bits',
        reference_repeats_in_new_budget=84,nonreference_calls=240,no_padding=True,no_retries=True,
        partial='all324 rows retained; missing endpoints yield null columns; first neural or exact-reference failure stops, no retry',
        output_cap_bytes=CAP,disk_floor_bytes=FLOOR,P1_admission=False,P2_admission=False,
        interpretation='model internal finite logit-response stability only; no new physical radius labels or unique cause attribution')


def context_exact(a,b):
    core.context(a); core.context(b)
    return a==b and all(physical.bits(a[k],'<f4')==physical.bits(b[k],'<f4') for k in ('latent_history','action_history'))


def model_slots(r,p):
    """Whitelist projection: never accesses report.analysis or any terminal encoding/physical response."""
    require(r['schema']==parent.NAME+'.report' and r['status']=='common_prefix_native_fidelity_completed_not_admitted' and
        r['errors']==[] and r['model_unchanged'] is r['source_verified'] is True and
        r['model_hash_before']==r['model_hash_after']==p['expected_model_hash']==enc.MODEL_TENSOR_SHA,'complete frozen parent model receipt')
    require(r['contract']==p['contract']==parent.contract() and len(r['rows'])==len(p['rows'])==84 and len(p['positions'])==4,'fixed parent census')
    require(r['resources']=={k:dict(attempted=v,completed=v) for k,v in parent.contract()['caps'].items()},'parent actual96/84 counts')
    expected_phases={phase:{k:dict(attempted=v,completed=v) for k,v in c.items()} for phase,c in parent.contract()['phase_caps'].items()}
    require(r['phase_resources']==expected_phases and len(r['anchors'])==4 and
        all(x['exact_original_task'] is True for x in r['anchors']),'parent four exact default anchors')
    slots=[]
    for pid,pos in enumerate(p['positions']):
        require(pos['position_id']==pid,'position order'); root=pos['root_slot']; t=p['tasks'][root]; core.validate_task(t)
        require(np.array_equal(t['projection'],np.eye(192)),'original identityQ')
        # Only formal images0..7 are accessed, never formal terminal10 images8..91.
        inputs=[np.asarray(r['images'][i]['native192'],np.float32) for i in (2*root,2*root+1,4+pid)]
        c=parent.prior.observed_context(t,*inputs,pos['prefix_commands']); refs=[]
        for k in range(21):
            row=r['rows'][21*pid+k]; original=p['rows'][21*pid+k]
            require(row['status']=='predicted' and row['index']==original['index']==21*pid+k and
                row['position_id']==pid and row['root_slot']==root and row['probe_index']==k and
                row['selection_digest']==original['selection_digest'] and row['commands']==physical.probes()[k]['commands'] and
                context_exact(row['input_context'],c),'source-bound actual same context / original21 probe')
            base.d2p.validate_advance(c,physical.raw5(row['commands']),row['prediction_context']); refs.append(deepcopy(row['prediction_context']))
        slots.append(dict(position_id=pid,root_slot=root,root_sha256=pos['root_sha256'],position_sha256=pos['position_sha256'],
            task_id=core.task_id(t),context=c,context_sha256=core.digest(c),references=refs,
            original_root=deepcopy(t['root_context']['latent_history'][-1]),goal=deepcopy(t['goal']),
            role_aliases=[x['role'] for x in pos['role_aliases']]))
    return slots


def material(parent_report,parent_sha):
    require(parent_sha==PARENT_SHA and sha(parent.__file__)==SOURCE_SHA,'approved parent external/source SHA')
    path=Path(parent_report).resolve(); require(sha(path)==parent_sha,'actual parent report SHA'); r=read(path)
    bp=path.parent/'binding.json'; b=read(bp); require(b['prepare_sha256']==r['prepare_sha256'] and sha(b['prepared'])==r['prepare_sha256'],'parent prepare seal')
    p=read(b['prepared']); slots=model_slots(r,p)
    bindings={str(path):parent_sha,str(bp):sha(bp),str(Path(b['prepared']).resolve()):r['prepare_sha256']}
    # Hash inherited byte seals only: do not reopen producer sourcepack/certificates/physical outcomes as data.
    for f,h in p['bindings'].items(): require(sha(f)==h,'parent byte seal '+f); bindings[f]=h
    for f in (Path(parent.__file__),Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        bindings[str(f.resolve())]=sha(f)
    return dict(schema=NAME+'.prepare',arguments=dict(parent_report=str(path),parent_sha=parent_sha),slots=slots,
        bank=bank(),contract=contract(),bindings=bindings,policy=p['policy'],expected_model_hash=p['expected_model_hash'],
        inherited_costs=dict(parent_resources=r['resources'],parent_phases=r['phase_resources'],earlier=p['inherited_costs']))


def prepare(output,**arguments):
    p=material(**arguments); raw=canonical(p); path=Path(output).absolute()
    require(not path.exists() and len(raw)<=2*1024**2 and shutil.disk_usage(path.parent).free>=FLOOR+len(raw),'exclusive small prepare/200MiB floor')
    with path.open('xb') as f: require(f.write(raw)==len(raw),'short prepare'); f.flush(); os.fsync(f.fileno())
    physical.syncdir(path.parent); return dict(prepared=str(path),prepare_sha256=sha(path),bytes=len(raw))


def verify(path,expected):
    require(sha(path)==expected,'external prepare SHA'); p=read(path,2*1024**2)
    require(p==material(**p['arguments']),'source-bound model-only reconstruction'); return p


class Store(base.Store):
    def __init__(self,path):
        require(shutil.disk_usage(Path(path).absolute().parent).free>=FLOOR+CAP,'NEW16MiB +200MiB floor')
        super().__init__(path)
    def _write(self,name,raw,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..') and not (self.output/name).is_symlink(),'flat new output')
        used=sum(f.stat().st_size for f in self.output.iterdir() if f.is_file())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(raw),'whole16MiB cap/floor')
        with (self.output/name).open('ab' if append else 'xb') as f: require(f.write(raw)==len(raw),'short write'); f.flush(); os.fsync(f.fileno())
        physical.syncdir(self.output)


class Ledger:
    def __init__(self,sink,sync=lambda:None): self.sink,self.sync=sink,sync; self.scope=None; self.entries=[]; self.failed=False
    def call(self,resource,fn):
        require(not self.failed and resource in ('action_encoder','predict') and self.scope is not None,'forward-only/no encoding/no retry')
        pid,index=self.scope; require(type(pid) is int and 0<=pid<4 and type(index) is int and 0<=index<81,'fixed4x81 scope')
        current=[e['resource'] for e in self.entries if e['scope']==list(self.scope)]
        require(len(current)<2 and resource==('action_encoder','predict')[len(current)] and all(e['completed'] for e in self.entries),'one actual H1 per slot')
        require(sum(e['resource']==resource for e in self.entries)<324 and
            sum(e['resource']==resource and e['scope'][0]==pid for e in self.entries)<81,'predispatch global/root81 caps')
        e=dict(index=len(self.entries),scope=list(self.scope),resource=resource,completed=False); self.entries.append(e)
        try:
            self.sink(dict(kind='module_attempt',entry=deepcopy(e))); result=fn(); self.sync(); e['completed']=True
            self.sink(dict(kind='module_completed',entry=deepcopy(e))); return result
        except BaseException as exc:
            self.failed=True
            try: self.sink(dict(kind='module_error',entry=deepcopy(e),error=str(exc)))
            except BaseException as secondary: exc.add_note('journal '+str(secondary))
            raise
    def counts(self):
        return {k:dict(attempted=sum(e['resource']==k for e in self.entries),completed=sum(e['resource']==k and e['completed'] for e in self.entries)) for k in contract()['caps']}


def arithmetic(p,rows):
    require(len(rows)==324 and [r['index'] for r in rows]==list(range(324)),'fixed324 denominator')
    out=[]
    for slot in p['slots']:
        pid=slot['position_id']; group=rows[81*pid:81*pid+81]; d=np.asarray(slot['goal'])-slot['original_root']; norm=float(np.linalg.norm(d)); u=d/norm if norm else None
        z=[np.asarray(r['prediction_context']['latent_history'][-1],float) if r['status']=='predicted' else None for r in group]
        radii=[]; smallest=None
        for ri,rho in enumerate(RHOS):
            columns=[]
            for j in range(10):
                a,b=1+20*ri+2*j,2+20*ri+2*j; B=(z[a]-z[b])/(2*rho) if z[a] is not None and z[b] is not None else None
                E=z[a]+z[b]-2*z[0] if B is not None and z[0] is not None else None
                columns.append(dict(coordinate=j,B=None if B is None else B.tolist(),E=None if E is None else E.tolist(),
                    curvature_per_rho2=None if E is None else (E/rho**2).tolist(),
                    signed_unit=None if B is None or u is None else float(u@B),signed_raw=None if B is None or u is None else float(d@B)))
            matrix=None if any(c['B'] is None for c in columns) else np.asarray([c['B'] for c in columns]).T
            if ri==0: smallest=matrix
            cosine=None
            if matrix is not None and smallest is not None:
                n=float(np.linalg.norm(matrix)*np.linalg.norm(smallest)); cosine=float(np.sum(matrix*smallest)/n) if n else None
            signed=[c['signed_unit'] for c in columns]
            radii.append(dict(rho=rho,columns=columns,B_192x10=None if matrix is None else matrix.tolist(),
                B_frobenius=None if matrix is None else float(np.linalg.norm(matrix)),
                B_relative_change_vs_smallest=None if matrix is None or smallest is None or np.linalg.norm(smallest)==0 else float(np.linalg.norm(matrix-smallest)/np.linalg.norm(smallest)),
                B_cosine_vs_smallest=cosine,signed_l2=None if any(a is None for a in signed) else float(np.linalg.norm(signed))))
        out.append(dict(position_id=pid,root_slot=slot['root_slot'],planned=81,completed=sum(r['status']=='predicted' for r in group),
            direction=d.tolist(),direction_norm=norm,radii=radii))
    return dict(positions=out,planned=324,space='native192 model-only finite logit secants; not AD/physical u',P1_admission=False,P2_admission=False)


def execute(p,predict,ledger,store,rows):
    for slot in p['slots']:
        pid=slot['position_id']; c=slot['context']; store.event(dict(kind='context_frozen',position_id=pid,context=c,context_sha256=slot['context_sha256']))
        for probe in p['bank']:
            k=probe['index']; row=rows[81*pid+k]; row['status']='attempted'; ledger.scope=(pid,k)
            commands=physical.raw5(probe['commands']); after=predict(deepcopy(c),commands.copy()); base.d2p.validate_advance(c,commands,after)
            require([(e['resource'],e['completed']) for e in ledger.entries if e['scope']==list(ledger.scope)]==[('action_encoder',True),('predict',True)],'actual complete H1 modules')
            ref=probe['reference_index']; exact=None if ref is None else context_exact(after,slot['references'][ref])
            row.update(status='predicted',prediction_context=after,reference_exact=exact)
            store.event(dict(kind='H1_saved',receipt=deepcopy(row)))
            if exact is False: row['status']='reference_mismatch'
            require(exact is not False,'source zero/rho.25 exact H1 replay mismatch; no tolerance or retry')
    ledger.scope=None


def run(prepared,expected,output):
    p=verify(prepared,expected); store=Store(output); ledger=Ledger(store.event); model=None; errors=[]; before=after=host=None; source_ok=False; analysis=None
    rows=[dict(index=81*s['position_id']+b['index'],position_id=s['position_id'],root_slot=s['root_slot'],probe=deepcopy(b),
        context_sha256=s['context_sha256'],status='not_started') for s in p['slots'] for b in p['bank']]
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn import attention
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False); enc.frozen(model)
        before=enc.tensor_hash(model); require(before==p['expected_model_hash'],'same original frozen model')
        backend=base.d2p.TorchBackend(model,device); ledger.sync=lambda:torch.cuda.synchronize(device)
        with torch.inference_mode(),parent.prior.guard(model,torch,ledger),parent.numerical_phase(torch,attention,'predict_math',store.event):
            execute(p,backend.step,ledger,store,rows)
    except BaseException as exc:
        errors.append(enc.error('execution',exc))
        try: store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary: errors.append(enc.error('error_journal',secondary))
    finally:
        ledger.scope=None
        for row in rows:
            if row['status']=='attempted': row['status']='failed_partial'
        try: analysis=arithmetic(p,rows)
        except BaseException as exc: errors.append(enc.error('final_arithmetic',exc))
        if model is not None:
            try: enc.frozen(model); after=enc.tensor_hash(model)
            except BaseException as exc: errors.append(enc.error('final_model',exc))
        try: verify(prepared,expected); source_ok=True
        except BaseException as exc: errors.append(enc.error('final_source',exc))
        try: store.close()
        except BaseException as exc: errors.append(enc.error('final_journal',exc))
        full=all(r['status']=='predicted' and r['reference_exact'] is not False for r in rows)
        if full and ledger.counts()!={k:dict(attempted=v,completed=v) for k,v in contract()['caps'].items()}: errors.append(dict(stage='final_counts',message='not324actual'))
        closed=full and not errors and not ledger.failed and source_ok and before is not None and before==after
        result=dict(schema=NAME+'.report',status='model_multiradius_completed_not_admitted' if closed else 'partial_or_failed',
            prepare_sha256=expected,inputs=p['arguments'],rows=rows,analysis=analysis,resources=ledger.counts(),errors=errors,
            source_verified=source_ok,model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before is not None else None,
            host=host,contract=p['contract'],inherited_costs=p['inherited_costs'],new_encoder=0,new_projector=0,new_AD=0,new_physics=0,P1_admission=False,P2_admission=False)
        hashes={}
        for f in store.output.iterdir():
            if f.is_file():
                try: hashes[f.name]=sha(f)
                except BaseException as exc: errors.append(enc.error('artifact_hash',exc)); closed=False
        result['artifact_sha256']=hashes
        if not closed: result['status']='partial_or_failed'
        try: store.put('report.json',result,final=True)
        except BaseException as exc: errors.append(enc.error('final_report',exc)); print(canonical(dict(errors=errors)).decode(),flush=True); closed=False
    require(closed,'partial output retained; no retry '+str(errors)); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=('prepare','verify','run'))
    for key in ('parent-report','parent-sha','prepared','prepare-sha','output'): parser.add_argument('--'+key)
    a=parser.parse_args()
    if a.command=='prepare': result=prepare(a.output,parent_report=a.parent_report,parent_sha=a.parent_sha)
    elif a.command=='verify': result=dict(status='verified_no_execution',contract=verify(a.prepared,a.prepare_sha)['contract'])
    else:
        r=run(a.prepared,a.prepare_sha,a.output); result={k:v for k,v in r.items() if k not in ('rows','analysis')}
    print(canonical(result).decode(),end='')


if __name__=='__main__': main()
