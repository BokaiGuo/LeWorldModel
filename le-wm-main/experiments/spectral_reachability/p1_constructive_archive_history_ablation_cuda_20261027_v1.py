"""Fixed archive-history F/S/A/SA CUDA experiment; MAIN-only run.

Consume original v2 saved encodings/raw actions, never re-encode or read HDF5.
No learning, selection, simulator, privileged inputs or physical counterfactuals.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from io import BytesIO
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import zlib
import numpy as np
from experiments.spectral_reachability import p1_constructive_offline_fidelity_cuda_20261027_v2 as frozen

NAME='p1_constructive_archive_history_ablation_cuda_20261027_v1'
ROOT=frozen.ROOT
SOURCE_SHA='e2ef4ce025d52f62e1688b268d2dab03021c248c4c8c5a7e2bf54ae98352b5e9'
SOURCE_CODE_SHA='f11b7ed5ae9bb7fb64150c8a9e8285d206f0c57cd255e1f2f547071ce3d94e12'
SOURCE_PREPARE_SHA='580dd9e35431423885041da76bbdbca3a033808dae7db40aabbaff25e1c34adb'
EPISODES=(0,6228,12456,18684)
CONDITIONS=('F','S','A','SA')
CAP,FLOOR,FINAL=8*1024**2,200*1024**2,1024**2
require,sha,read,canonical=frozen.require,frozen.sha,frozen.read,frozen.canonical
enc=frozen.enc


def bits(a,b): return a.dtype.str==b.dtype.str and a.shape==b.shape and a.tobytes()==b.tobytes()


class Saved:
    def __init__(self,parent,manifest): self.parent=Path(parent).absolute(); self.manifest=manifest; self.bindings={}
    def raw(self,name):
        require(Path(name).name==name and name not in ('','.','..'),'flat saved child')
        path=self.parent/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size<=1024**2,'bounded regular saved child')
        require(name in self.manifest and sha(path)==self.manifest[name],'original report artifact binding')
        self.bindings[str(path)]=self.manifest[name]; return path.read_bytes()
    def array(self,name,shape,receipt=None):
        b=self.raw(name); stream=BytesIO(b); a=np.load(stream,allow_pickle=False)
        require(stream.tell()==len(b) and a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'exact finite float32 NPY entry')
        if receipt is not None:
            require(receipt==dict(file=name,sha256=self.manifest[name],raw_sha256=enc.digest(a.tobytes()),shape=list(shape),dtype=a.dtype.str),'complete saved file/array entry')
        return a


def make_requests(latents,raws):
    require(len(latents)==len(raws)==4,'fixed four archive slots')
    requests=[]; labels=[]
    for slot,(zs,raw) in enumerate(zip(latents,raws,strict=True)):
        require(len(zs)==5 and raw.dtype==np.float32 and raw.shape==(20,2) and np.isfinite(raw).all(),'all20 original raw actions, no domain repair')
        for z in zs: enc.native(z)
        for window in (10,15):
            start=window//5-2; history=np.stack(zs[start:start+3]); packets=raw[5*start:5*(start+3)].reshape(3,5,2).copy()
            labels.append(dict(slot=slot,episode=EPISODES[slot],window=window,current=zs[start+2].copy(),target=zs[start+3].copy()))
            for kind in CONDITIONS:
                h,a=history.copy(),packets.copy()
                if kind in ('S','SA'): h[:2]=h[-1]
                if kind in ('A','SA'): a[:2]=0  # literal raw zero THEN original public normalization
                requests.append(dict(slot=slot,episode=EPISODES[slot],window=window,condition=kind,
                    latent_history=h,raw_packets=a,tokens=frozen.normalized(a)))
    return requests,labels


def contract():
    return dict(episodes=list(EPISODES),windows=[10,15],conditions=list(CONDITIONS),H1_calls=32,
        interventions=dict(F='original factual states/history/actions',S='only two old states become current latent',
            A='only two historical raw packets become zero then public-normalized',SA='both S and A'),
        invariants=['same current native192','same outgoing factual raw packet/token','same actual successor label'],
        teacher_forced=True,label_policy='all32 outputs durable before scoring; no label/reference prediction reaches backend',
        exact_F='bitwise complete3x192 prediction and action embeddings against original v2; no allclose fallback',
        counts=dict(encoder=0,projector=0,predict=32,predictor=32,action_encoder=32,pred_proj=32),
        tokens=dict(per_call=3,latent_input=96,action_packet_input=96,pred_proj_samples=96),
        AD=0,grad=0,engine=0,physics=0,reset=0,render=0,optimization=False,training=False,selection=False,
        SDP='MATH',matmul_TF32=False,cudnn_TF32=False,raw_clip=False,CPU_neural=False,
        normalizer='original public mean/std; training provenance unknown; no refit or /100',
        counterfactual_physics=False,scientific_threshold=None,P1_admission=False,independent_confirmation=False,
        wall_seconds=600,allocated_bytes=3221225472,reserved_bytes=4294967296,
        output_cap_bytes=CAP,disk_floor_bytes=FLOOR,final_reserve_bytes=FINAL,prepare_cap_bytes=256*1024,
        no_retry=True,no_dedup=True,scope='conditional representation-history ablations; NOT live failure causality or training provenance')


def load_source(report_path,expected):
    require(expected==SOURCE_SHA and sha(report_path)==expected,'fixed actual v2 source report SHA')
    require(sha(frozen.__file__)==SOURCE_CODE_SHA,'original numerics source pinned')
    report=read(report_path)
    require(report['schema']==frozen.NAME+'.report' and report['status']=='stage_A_fidelity_saved_not_admitted' and
        report['errors']==[] and report['source_verified'] is True and report['model_unchanged'] is True and
        report['model_hash_before']==report['model_hash_after']==enc.MODEL_TENSOR_SHA,'complete original frozen-model saved source')
    require(report['episode_indices']==list(EPISODES) and report['contract']==frozen.contract(),'original fixed source protocol/numerics')
    require(report['AD']==report['physics']==0 and not report['blocked_visual'] and not report['forbidden_AD_attempts'],'original no forbidden work')
    require(all(report['resources'][k]['attempted']==report['resources'][k]['completed']==v for k,v in frozen.contract()['counts'].items()),'complete original neural census')
    path=Path(report['prepared']).absolute()
    require(report['prepare_sha256']==SOURCE_PREPARE_SHA and sha(path)==SOURCE_PREPARE_SHA,'original source prepare')
    original=read(path)
    require(original['schema']==frozen.NAME+'.prepare' and original['episode_indices']==list(EPISODES) and
        original['contract']==report['contract'] and original['expected_model_hash']==enc.MODEL_TENSOR_SHA,'original prepare source/model identity')
    require(original['normalization']==enc.native_metadata(),'unchanged public normalizer binding')
    for name,v in original['versions'].items(): require(importlib.metadata.version(name)==v,'source software version '+name)
    source=Saved(Path(report_path).absolute().parent,report['artifact_sha256'])
    require(json.loads(source.raw('binding.json'))==dict(prepared=str(path),prepare_sha256=SOURCE_PREPARE_SHA),'source execution binding')
    require(json.loads(source.raw('selected_source_receipts.json'))==original,'source saved metadata')
    require(len(report['images'])==20 and len(report['requests'])==24,'source fixed image/request denominators')
    latents=[]; raws=[]
    for slot,episode in enumerate(EPISODES):
        raw=source.array(f'slot_{slot}_raw20x2.npy',(20,2)); record=original['episodes'][slot]
        require(record['episode_index']==episode and frozen.array_receipt(raw)==record['actions'] and
            np.array_equal(raw,np.asarray(record['raw_actions'])),'original literal raw actions')
        raws.append(raw); zs=[]
        for j,step in enumerate((0,5,10,15,20)):
            i=5*slot+j; row=report['images'][i]
            require((row['index'],row['slot'],row['episode'],row['step'],row['status'])==(i,slot,episode,step,'encoded'),'original native image identity')
            zs.append(source.array(f'image_{i:02d}.native192.npy',(192,),row['latent']))
            png=row['png']; name=f'image_{i:02d}.png'; b=source.raw(name)
            require(png['file']==name and png['sha256']==enc.digest(b)==record['frames'][j]['png_sha256'] and
                png['pixels']==record['frames'][j]['pixels'],'source encoding PNG identity')
        latents.append(zs)
    requests,labels=make_requests(latents,raws); references=[]
    for slot in range(4):
        for w,window in enumerate((10,15)):
            old_index=6*slot+3*w; row=report['requests'][old_index]; r=requests[8*slot+4*w]
            identity={k:r[k] for k in ('slot','episode','window','condition')}
            require(row['index']==old_index and row['status']=='saved' and row['identity']==identity,'original F request identity')
            for key in ('latent_history','raw_packets','tokens'):
                a=source.array(f'request_{old_index:02d}.{key}.npy',r[key].shape,row['inputs'][key])
                require(bits(a,r[key]),'source original F '+key+' exact input')
            ref={k:source.array(f'request_{old_index:02d}.{k}.npy',(3,192),row['outputs'][k]) for k in ('prediction','embedding_before','embedding_after')}
            last=source.array(f'request_{old_index:02d}.last_native192.npy',(192,),row['outputs']['last_native192'])
            require(bits(last,ref['prediction'][-1]) and bits(ref['embedding_before'],ref['embedding_after']),'original factual output consistency')
            references.append(ref)
    bindings=dict(original['bindings']); bindings.update(source.bindings)
    bindings.update({str(Path(report_path).absolute()):expected,str(path):SOURCE_PREPARE_SHA})
    for path,h in bindings.items(): require(sha(path)==h,'source/model/array seal '+path)
    return dict(original=original,host=report['host'],bindings=bindings,requests=requests,labels=labels,references=references)


def material(source_report,source_sha):
    data=load_source(source_report,source_sha); original=data['original']
    files=[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    bindings=dict(data['bindings']); bindings.update({str(p.resolve()):sha(p) for p in files})
    p=dict(schema=NAME+'.prepare',source_report=str(Path(source_report).absolute()),source_sha256=source_sha,
        source_prepare_sha256=SOURCE_PREPARE_SHA,source_code_sha256=SOURCE_CODE_SHA,bindings=bindings,contract=contract(),
        policy=original['policy'],expected_model_hash=original['expected_model_hash'],normalization=original['normalization'],
        source_host=data['host'],source_v2_disk_telemetry='known mock-minimum bug; NOT adopted as actual disk telemetry')
    return p,data


def prepare(source_report,source_sha,output):
    p,_=material(source_report,source_sha); b=canonical(p); output=Path(output).absolute()
    require(not output.exists() and len(b)<256*1024 and shutil.disk_usage(output.parent).free>=FLOOR+len(b),'NEW small prepare+200MiB')
    with output.open('xb') as f: f.write(b); f.flush(); os.fsync(f.fileno())
    fd=os.open(output.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(prepared=str(output),prepare_sha256=sha(output),bytes=len(b))


def verify(path,expected):
    require(sha(path)==expected,'external new prepare SHA'); p=read(path)
    current,data=material(p['source_report'],p['source_sha256'])
    require(p==current,'complete saved source/version/model/array reconstruction'); return p,data


class Store(frozen.Store):
    def __init__(self,output):
        self.output=Path(output).absolute()
        require(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'exclusive8MiB+200MiB reserve')
        self.output.mkdir(); self.comp=zlib.compressobj(6,zlib.DEFLATED,31); self.closed=self.broken=False
        self._write('events.jsonl.gz',b'')
    def _write(self,name,raw,append=False,final=False):
        require(Path(name).name==name and name not in ('','.','..') and not (self.output/name).is_symlink(),'flat nonsymlink output')
        used=sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())
        require(used+len(raw)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(raw),'allbytes cap/final reserve/200MiB floor')
        with (self.output/name).open('ab' if append else 'xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)


class Ledger(frozen.PredictLedger):
    def dispatch(self,kind,fn,*args,**kwargs):
        self.limits.check()
        require(not self.failed and len(self.entries)<128 and kind==self.ORDER[len(self.entries)%4],'fixed32 requests, no retry')
        require(all(e['completed'] or e['resource']=='predict' for e in self.entries),'no retry after incomplete child')
        e=dict(index=len(self.entries),request=len(self.entries)//4,resource=kind,tokens=3,completed=False)
        self.entries.append(e)
        try:
            self.sink(dict(kind='prediction_module_attempt',entry=deepcopy(e)))
            value=fn(*args,**kwargs); self.sync(); self.limits.check(); e['completed']=True
            self.sink(dict(kind='prediction_module_return',entry=deepcopy(e))); return value
        except BaseException as exc:
            self.failed=True; e['error']=enc.error(kind,exc); raise


def native_predict(model,torch,device,r):
    require(set(r)==frozen.REQUEST_KEYS and r['condition'] in CONDITIONS,'no label/reference in neural callback')
    # Preserve the ORIGINAL native F forward, including its embedding clone and pred_proj.
    # S/A/SA act only on input arrays; none is the old outgoing-embedding mask condition.
    return frozen.native_predict(model,torch,device,dict(r,condition='F'))


def execute(requests,references,backend,store,rows,limits):
    require(len(requests)==len(rows)==32 and len(references)==8,'all fixed denominators')
    outputs=[]
    for i,r in enumerate(requests):
        require(set(r)==frozen.REQUEST_KEYS and r['condition']==CONDITIONS[i%4],'ordered input whitelist')
        limits.check(); rows[i]['status']='attempted'; meta={k:r[k] for k in ('slot','episode','window','condition')}
        inputs={k:frozen.save_array(store,f'request_{i:02d}.{k}.npy',r[k]) for k in ('latent_history','raw_packets','tokens')}
        store.event(dict(kind='request',index=i,identity=meta,inputs=inputs))
        out=backend(deepcopy(r)); require(set(out)=={'prediction','embedding_before','embedding_after'},'only native complete outputs')
        for a in out.values(): require(a.dtype==np.float32 and a.shape==(3,192) and np.isfinite(a).all(),'native finite3x192')
        require(bits(out['embedding_before'],out['embedding_after']),'no embedding mask in this experiment')
        saved={k:frozen.save_array(store,f'request_{i:02d}.{k}.npy',a) for k,a in out.items()}
        saved['last_native192']=frozen.save_array(store,f'request_{i:02d}.last_native192.npy',out['prediction'][-1])
        rows[i].update(status='saved',identity=meta,inputs=inputs,outputs=saved)
        if r['condition']=='F':
            ref=references[i//4]
            rows[i]['F_exact']={k:bits(out[k],ref[k]) for k in out}
            rows[i]['F_max_abs_delta']={k:float(np.max(np.abs(out[k].astype(float)-ref[k]))) for k in out}
        store.event(dict(kind='output_saved',row=deepcopy(rows[i])))
        if r['condition']=='F' and not all(rows[i]['F_exact'].values()):
            rows[i]['status']='saved_factual_reproduction_failed'
            store.event(dict(kind='factual_exact_reproduction_failed',row=deepcopy(rows[i])))
            raise ValueError('original v2 F bitwise reproduction failed; no tolerance/retry')
        outputs.append(dict(**meta,prediction=out['prediction'].copy()))
    return outputs


def score(outputs,labels):
    require(len(outputs)==32 and len(labels)==8,'score only after all32 saved')
    rows=[]
    for i,label in enumerate(labels):
        group=outputs[4*i:4*i+4]
        require([(r['slot'],r['window'],r['condition']) for r in group]==[(label['slot'],label['window'],k) for k in CONDITIONS],'fixed window/condition identities')
        target=label['target'].astype(float)
        errors={r['condition']:float(np.mean((r['prediction'][-1].astype(float)-target)**2)) for r in group}
        rows.append(dict(slot=label['slot'],episode=label['episode'],window=label['window'],label_step=label['window']+5,
            label_array=frozen.array_receipt(label['target']),fidelity_mse=errors,
            S_minus_F=errors['S']-errors['F'],A_minus_F=errors['A']-errors['F'],SA_minus_F=errors['SA']-errors['F'],
            interaction_SA_minus_S_minus_A_plus_F=errors['SA']-errors['S']-errors['A']+errors['F'],
            persistence_mse=float(np.mean((label['current'].astype(float)-target)**2))))
    return dict(windows=rows,window_denominator=8,episode_denominator=4,
        mean_fidelity_mse={k:float(np.mean([r['fidelity_mse'][k] for r in rows])) for k in CONDITIONS},
        physical_counterfactual_truth=False,training_provenance_verified=False,independent_demonstrations='unknown',
        scientific_threshold=None,P1_admission=False,interpretation='conditional input ablation on same factual successor; NOT physical successor prediction accuracy for intervened histories')


def run(prepared,expected,output):
    limits=frozen.Limits(free=lambda:shutil.disk_usage(Path(output).absolute().parent).free)
    p,data=verify(prepared,expected); limits.check(); store=Store(output)
    errors=[]; forbidden=[]; model=None; before=after=host=None; source_ok=False
    rows=[dict(index=i,status='not_started') for i in range(32)]; ledger=Ledger(store.event,limits)
    def ad_event(e): forbidden.append(deepcopy(e)); store.event(e)
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        store.put('source_binding.json',dict(source_report=p['source_report'],source_sha256=p['source_sha256']))
        store.put('request_order.json',[{k:r[k] for k in ('slot','episode','window','condition')} for r in data['requests']])
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0'); device=torch.device('cuda:0')
        for k in ('name','torch_cuda','torch_version'): require(host[k]==p['source_host'][k],'original CUDA/software numerics '+k)
        limits.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
        limits.free=lambda:shutil.disk_usage(store.output).free
        limits.check(); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        limits.check(); enc.frozen(model); before=enc.tensor_hash(model)
        require(before==p['expected_model_hash'],'original frozen tensor hash')
        ledger.sync=lambda:torch.cuda.synchronize(device)
        with frozen.old.no_ad(torch,ad_event,errors),sdpa_kernel(SDPBackend.MATH),frozen.prediction_guard(model,torch,ledger,errors):
            outputs=execute(data['requests'],data['references'],lambda r:native_predict(model,torch,device,r),store,rows,limits)
        store.event(dict(kind='all32_outputs_durable_before_label_join'))
        store.put('diagnostics.json',score(outputs,data['labels'])); limits.check()
    except BaseException as exc:
        errors.insert(0,enc.error('execution',exc))
        try: store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as secondary: errors.append(enc.error('error_journal',secondary))
    finally:
        for r in rows:
            if r['status']=='attempted': r['status']='failed_partial'
        if model is not None:
            try: enc.frozen(model); after=enc.tensor_hash(model)
            except BaseException as exc: errors.append(enc.error('final_model',exc))
        try: verify(prepared,expected); source_ok=True
        except BaseException as exc: errors.append(enc.error('final_source',exc))
        try: limits.check()
        except BaseException as exc: errors.append(enc.error('final_limits',exc))
        try: store.close()
        except BaseException as exc: errors.append(enc.error('journal_finalize',exc))
        counts=dict(encoder=dict(attempted=0,completed=0),projector=dict(attempted=0,completed=0),**ledger.counts())
        exact=sum(r['status']=='saved' and 'F_exact' in r and all(r['F_exact'].values()) for r in rows)
        complete=not errors and not forbidden and not ledger.failed and source_ok and before is not None and before==after and exact==8 and all(r['status']=='saved' for r in rows) and all(counts[k]['attempted']==counts[k]['completed']==v for k,v in contract()['counts'].items())
        hashes={}
        for path in store.output.iterdir():
            if path.is_file():
                try: hashes[path.name]=sha(path)
                except BaseException as exc: errors.append(enc.error('artifact_hash',exc)); complete=False
        report=dict(schema=NAME+'.report',status='history_ablation_saved_not_admitted' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,source_report=p['source_report'],source_sha256=p['source_sha256'],
            requests=rows,resources=counts,module_ledger=ledger.entries,forbidden_AD_attempts=forbidden,F_exact_windows=exact,F_expected_windows=8,
            errors=errors,first_error=errors[0] if errors else None,contract=p['contract'],source_verified=source_ok,
            model_hash_before=before,model_hash_after=after,model_unchanged=before==after if before else None,
            host=host,limits=limits.receipt(),artifact_sha256=hashes,AD=0 if complete else None,physics=0,P1_admission=False)
        try: store.put('report.json',report,final=True)
        except BaseException as exc:
            complete=False; errors.append(enc.error('final_report',exc)); print(canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial retained; no retry: '+str(errors)); return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=('prepare','verify','run'))
    for key in ('source-report','source-sha','prepared','prepare-sha','output'): parser.add_argument('--'+key)
    a=parser.parse_args()
    if a.command=='prepare': r=prepare(a.source_report,a.source_sha,a.output)
    elif a.command=='verify': r=dict(status='saved_only_verified',requests=len(verify(a.prepared,a.prepare_sha)[1]['requests']))
    else: r=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(r).decode(),end='')


if __name__=='__main__': main()
