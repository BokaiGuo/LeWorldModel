"""Intersection v2 fixed 24-slot native RGB encoding, MAIN CUDA only."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import agent_p2_intersection_rgb_roster_20261029_v1 as roster
from experiments.spectral_reachability import agent_p1_intersection_goal_adapter_20261029_v2 as authority
from experiments.spectral_reachability import agent_p2_contextual_physical_rgb_encode_20261028_v1 as bridge
from experiments.spectral_reachability import agent_p2_six_target_rgb_20261029_v1 as primitive

NAME = 'agent_p2_intersection_rgb_20261029_v1'
REPORT_SHA = 'aba81af75d6de452f517f8b28a8d60a4baa7599877e96376f25d04c884e56a88'
RECEIPT_SHA = 'e96462847db675dc0b0c5376067df3654edd40805d6e374ffa3f8f1d5224e6c4'
PREPARE_SHA = '619a3202a9e14c6d164375a46ee8e51e7e9fe217678e6fb6f3c2725e5219341c'
SOURCE_SHA = 'e2309903a600cc536d9d12ef746014ebfdaa0bea6a6097c326c43c46a9dbdc35'


def authenticated_roster(prepared, receipt, physical_output, report_sha, receipt_sha, prepare_sha, source_sha):
    """Read-only source-bound projection; caller must not call it a complete prepare."""
    authority.need((report_sha, receipt_sha, prepare_sha, source_sha) ==
                   (REPORT_SHA, RECEIPT_SHA, PREPARE_SHA, SOURCE_SHA), 'actual MAIN authority pins')
    output = Path(physical_output)
    authority.need(output.is_absolute() and output.is_dir() and not output.is_symlink(), 'absolute physical directory')
    p = json.loads(authority.read(Path(prepared), prepare_sha))
    r = json.loads(authority.read(Path(receipt), receipt_sha))
    report = json.loads(authority.read(output/'report.json', report_sha))
    authority.read(authority.SOURCE, source_sha)
    authority.read(authority.RECEIPT, authority.RECEIPT_SHA)
    authority.need(r['status'] == 'intersection6_native_recorded_not_admitted' and r['errors'] == []
                   and r['physical_report_sha256'] == report_sha, 'full MAIN strict receipt')
    authority.need(report['execution_binding']['prepared'] == str(prepared)
                   and report['execution_binding']['prepare_sha256'] == prepare_sha
                   and r['bindings'][str(prepared)] == prepare_sha
                   and r['bindings'][str(authority.SOURCE)] == source_sha
                   and r['bindings'][str(output/'report.json')] == report_sha, 'exact source and report binding')
    authority.project(p, Path(prepared), prepare_sha, authority.SOURCE, source_sha)
    planner=p['planner_authority']; planner_report=Path(planner['planner_output'])/'report.json'
    mr=json.loads(authority.read(planner_report,planner['planner_report_sha']))
    mp=json.loads(authority.read(Path(mr['prepared']),mr['prepare_sha256']))
    authority.need(mr['model_before']==mr['model_after']==mp['expected_model_hash']==bridge.native.MODEL_TENSOR_SHA
                   and Path(mp['policy'])==bridge.rgb.POLICY, 'exact original planner model and policy')
    rows = roster.assemble(p, r, output)
    for row in rows:
        if row['frame'] is not None:
            bridge.load_pixels(row['frame'])
    bridge.verify_bindings(r['bindings'])
    return dict(schema=NAME+'.authenticated_roster', rows=rows,
                fixed_denominator=24, known_roots=2, fresh_roots=0,
                inherited_strict_receipt_sha256=receipt_sha, physical_report_sha256=report_sha,
                physical_prepare_sha256=prepare_sha, physical_source_sha256=source_sha,
                upstream_full_strict_reexecuted=False, P1_admission=False, P2_admission=False,
                production_ready=False)


ROOT = authority.ROOT
CAP, FINAL, FLOOR, WALL, META = 256*1024**2, 32*1024**2, 200*1024**2, 900, 128*1024**2
ALLOC, RESERVED = 3*1024**3, 4*1024**3
BOUNDARY = 'budget start before verify through report fsync/hash; completion acknowledgement excluded'
need, sha, encode = primitive.need, primitive.sha, primitive.encode


def contract():
    return dict(slots=6, positions=24, checkpoints=list(roster.CP), known_roots=2, fresh_roots=0,
                max_encoder=24, max_projector=24, new_predict=0, new_AD=0, new_physics=0,
                training=False, cuda_only=True, batch=1, no_dedup=True, no_padding=True,
                hidden=[1,257,192], latent=192, wall_seconds=WALL, cap_bytes=CAP,
                final_reserve=FINAL, disk_floor=FLOOR, allocated_cap=ALLOC, reserved_cap=RESERVED,
                wall_boundary=BOUNDARY, FLOPs=None, P1_admission=False, P2_admission=False,
                preregistration_verified=False, physical_release=False)


def source_files():
    files = bridge.source_files() + [Path(__file__).resolve(), Path(roster.__file__).resolve(),
        Path(primitive.__file__).resolve(), Path(primitive.roster.__file__).resolve(), Path(authority.__file__).resolve(),
        ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md'),
        ROOT/'tests'/('test_'+roster.NAME+'.py'), ROOT/'innerdocs'/(roster.NAME+'.md'),
        ROOT/'experiments/checkpoint_compat.py', ROOT/'experiments/gpu_preflight.py',
        ROOT/'experiments/spectral_reachability/preflight_pusht_d2p_p0.py']
    return sorted(set(files))


def build(prepared, receipt, physical_output, report_sha, receipt_sha, prepare_sha, source_sha):
    a = authenticated_roster(prepared,receipt,physical_output,report_sha,receipt_sha,prepare_sha,source_sha)
    strict = json.loads(authority.read(Path(receipt),receipt_sha))
    bindings = dict(strict['bindings'])
    extra = {str(Path(receipt)):receipt_sha, str(Path(physical_output)/'report.json'):report_sha,
             str(Path(prepared)):prepare_sha, **bridge.rgb.MODELS,
             **{str(f):sha(f) for f in source_files()}}
    for path,h in extra.items():
        need(path not in bindings or bindings[path] == h, 'conflicting binding')
        bindings[path] = h
    bridge.verify_bindings(bindings)
    rows = a['rows']
    for row in rows:
        row.update(cp_public_step=row['checkpoint'], pixel_source=row['frame'],
                   status='not_started' if row['frame'] is not None else 'missing_physical_checkpoint',
                   pixels=None,preprocessed=None,latent=None,modules=[])
    n = sum(row['frame'] is not None for row in rows)
    need(len(rows)==24 and n<=24, 'fixed position denominator')
    return dict(schema=NAME+'.prepare', arguments=dict(prepared=str(prepared),receipt=str(receipt),
        physical_output=str(physical_output),report_sha=report_sha,receipt_sha=receipt_sha,
        prepare_sha=prepare_sha,source_sha=source_sha), contract=contract(), rows=rows,
        bindings=bindings, expected_calls={k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        expected_model_hash=bridge.native.MODEL_TENSOR_SHA, policy=str(bridge.rgb.POLICY),
        inherited_physical_counts=strict['counts'], inherited_strict_receipt_sha256=receipt_sha)


def read(path, expected=None):
    p=Path(path)
    need(p.is_absolute() and p.is_file() and not p.is_symlink() and p.stat().st_size<=META,'bounded absolute metadata')
    if expected is not None: need(sha(p)==expected,'metadata SHA mismatch')
    return json.loads(p.read_bytes())


def write(path, value):
    p=Path(path); data=encode(value)
    need(p.is_absolute() and p.name.startswith(NAME+'_') and not p.is_symlink() and
         len(data)<=META and shutil.disk_usage(p.parent).free>=FLOOR+len(data),'exclusive bounded output')
    with p.open('xb') as f: need(f.write(data)==len(data),'short write');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(p.parent)
    return sha(p)


def verify(path, expected):
    p=read(path,expected);need(p==build(**p['arguments']),'entire prepare reconstruction');return p


class Store(bridge.rgb.Store):
    def __init__(self, path, budget):
        need(Path(path).is_absolute() and Path(path).name.startswith(NAME+'_'),'own absolute directory')
        super().__init__(path,budget);self.index=0;self.order=[]
    def write(self,name,raw,final=False):
        h=super().write(name,raw,final=final);self.order.append(name);return h
    def event(self,e):
        self.put(f'event_{self.index:04d}.json',e);self.index+=1


def run(prepared, expected, output, authorized=False):
    need(authorized is True,'explicit MAIN CUDA authorization')
    budget=primitive.Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);errors=[];model=trace=None;before=after=None;precision=determinism=None;host=None;peaks={}
    store.put('binding.json',dict(prepared=str(prepared),prepare_sha256=expected))
    store.put('initial_roster.json',p['rows'])
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from torch.utils._python_dispatch import TorchDispatchMode
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0');torch.cuda.reset_peak_memory_stats()
        precision=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        determinism=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        def check():
            store.check();peaks.update(allocated=int(torch.cuda.max_memory_allocated()),reserved=int(torch.cuda.max_memory_reserved()))
            peaks['guard_checks']=peaks.get('guard_checks',0)+1
            need(peaks['allocated']<=ALLOC and peaks['reserved']<=RESERVED,'GPU caps')
        class Guard(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                check();v=func(*args,**(kwargs or {}));check();return v
        model=load_compatible_pretrained(p['policy'],device=torch.device('cuda:0')).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model);need(before==p['expected_model_hash'],'model identity')
        trace=primitive.Trace(model,torch,store,p['expected_calls'],check);transform=p0.image_transform()
        def encode_one(pix,row,scope):
            trace.scope=scope
            try:
                x=p0._image_tensor(pix,transform=transform,device=torch.device('cuda:0'))
                row['preprocessed']=trace.save(x,f'r{row["index"]}_preprocessed')
                bridge.same(x.detach().cpu().numpy(),bridge.pixels.preprocess_pixels(pix),'native preprocessing')
                z=model.encode({'pixels':x})['emb'];need(tuple(z.shape)==(1,1,192),'native output shape')
                return z[0,0].detach().cpu().numpy().copy()
            finally:
                row['modules']=deepcopy([e for e in trace.entries if e['scope']==scope]);trace.scope=None
        with trace,torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH),Guard():
            primitive.process(p,rows,store,encode_one)
        check()
    except BaseException as e: errors.append(bridge.native.error('encode',e))
    finally:
        if precision is not None: torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=precision
        if determinism is not None: torch.use_deterministic_algorithms(determinism[0],warn_only=determinism[1])
        try:
            if model is not None: bridge.native.frozen(model);after=bridge.native.tensor_hash(model)
            bridge.verify_bindings(p['bindings']);need(sha(prepared)==expected,'prepare changed');budget.check()
        except BaseException as e: errors.append(bridge.native.error('final_guards',e))
        entries=[] if trace is None else trace.entries
        complete=not errors and before==after==p['expected_model_hash'] and bridge.counts(entries)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()}
        seal=dict(rows=rows,modules=entries,counts=bridge.counts(entries),errors=errors)
        seal_sha=store.put('native_seal.json',seal,final=True)
        report=dict(schema=NAME+'.report',contract=contract(),completed=complete,prepared=str(prepared),prepare_sha256=expected,
            **seal,native_seal_sha256=seal_sha,host=host,model_before=before,model_after=after,gpu_peaks=peaks,
            resources=primitive.accounting(p,rows,entries),event_count=store.index,artifact_sha256=deepcopy(store.hashes),
            artifact_order=list(store.order),wall_before_report=budget.clock()-budget.started)
        store.put('report.json',report,final=True);elapsed=budget.clock()-budget.started
        done=dict(schema=NAME+'.completion',report_sha256=sha(store.path/'report.json'),complete=bool(complete and elapsed<=WALL),wall_inclusive=elapsed,boundary=BOUNDARY)
        store.put('completion.json',done,final=True)
    need(done['complete'],'partial artifacts retained, no retry: '+str(errors))
    return done


def mse(a,b):
    a=np.asarray(a,dtype=np.float32);b=np.asarray(b,dtype=np.float32)
    need(a.shape==b.shape==(192,) and np.isfinite(a).all() and np.isfinite(b).all(),'finite latent vectors')
    return math.fsum((float(x)-float(y))**2 for x,y in zip(a,b,strict=True))/192


def validate(p,r,events,load):
    need(r['schema']==NAME+'.report' and r['contract']==p['contract']==contract() and r['completed'] is True and r['errors']==[], 'complete report')
    need(r['model_before']==r['model_after']==p['expected_model_hash'],'model unchanged')
    need(r['host']['device']=='cuda:0' and 0<r['gpu_peaks']['allocated']<=ALLOC and 0<r['gpu_peaks']['reserved']<=RESERVED and r['gpu_peaks']['guard_checks']>0,'CUDA guards')
    need(len(p['rows'])==len(r['rows'])==24,'full denominator')
    modules=[];want=[];out=[]
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        mutable={'status','latent','pixels','preprocessed','modules'}
        need({k:v for k,v in spec.items() if k not in mutable}=={k:v for k,v in row.items() if k not in mutable},'immutable roster')
        z=None
        if spec['pixel_source'] is None:
            need(row==spec,'missing preserved')
        else:
            need(row['status']=='encoded','available frame encoded')
            pix=load(row['pixels']);bridge.same(pix,bridge.load_pixels(spec['pixel_source']),'exact PNG pixels')
            pre=load(row['preprocessed']);bridge.same(pre,bridge.pixels.preprocess_pixels(pix),'exact preprocessing')
            mods=row['modules'];scope=dict(row=row['index'],step=row['cp_public_step'],phase='encode')
            need([x['resource'] for x in mods]==['encoder','projector'] and all(x['scope']==scope and x['completed'] is True and len(x['inputs'])==1 for x in mods),'native module chain')
            en,proj=mods;ein,eout,pin,pout,z=(load(d) for d in (en['inputs'][0],en['output'],proj['inputs'][0],proj['output'],row['latent']))
            need(all(v.dtype==np.float32 and np.isfinite(v).all() for v in (pre,ein,eout,pin,pout,z)),'float32 native IO')
            bridge.physical.saved.overlay.visual_bridge(pre,ein,eout,pin,pout,z,[1,257,192])
            bridge.same(pre.reshape(1,3,224,224),ein,'preprocess encoder')
            bridge.same(eout[:,0],pin,'hidden CLS projector');bridge.same(pout[0],z,'projector output')
            want.append(dict(kind='image_attempt',scope=scope))
            for mod in mods:
                want.extend([dict(kind='module_attempt',entry={k:mod[k] for k in ('index','scope','resource')}|dict(completed=False)),dict(kind='module_return',entry=mod)])
            want.append(dict(kind='image_return',scope=scope,row=row));modules.extend(mods)
        pred=spec['predicted192'];goal=spec['goal192']
        out.append(dict(index=spec['index'],slot=spec['slot'],root=spec['root'],arm=spec['arm'],checkpoint=spec['checkpoint'],
            prediction_role=spec['prediction_role'],predicted192=pred,goal192=goal,native_goal=spec['native_goal'],
            model_Q=mse(pred,goal),actual_Q=None if z is None else mse(z,goal),
            model_actual_MSE=None if z is None else mse(pred,z),latent=row['latent'],frame=spec['frame'],
            native_response=spec['native_response'],missing_reason=spec['missing_reason'],raw15_bits=spec['raw15_bits'],
            prefix_bits=spec['prefix_bits'],selection_digest=spec['selection_digest']))
    need(modules==r['modules'] and [x['index'] for x in modules]==list(range(len(modules))) and events==want
         and r['counts']==bridge.counts(modules)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()},'full event/module ledger')
    need(r['resources']==primitive.accounting(p,r['rows'],modules),'resource accounting')
    return out


def consume(output, expected):
    folder=Path(output);need(folder.is_absolute() and folder.is_dir() and not folder.is_symlink(),'absolute output directory')
    need(sha(folder/'report.json')==expected,'report SHA')
    fs=list(folder.iterdir());need(all(f.is_file() and not f.is_symlink() for f in fs) and sum(f.stat().st_size for f in fs)<=CAP,'bounded regular census')
    files={f.name:f.read_bytes() for f in fs};r=read(folder/'report.json',expected);p=verify(r['prepared'],r['prepare_sha256']);done=read(folder/'completion.json')
    need(done['schema']==NAME+'.completion' and done['report_sha256']==expected and done['complete'] is True and done['boundary']==BOUNDARY
         and 0<=r['wall_before_report']<=done['wall_inclusive']<=WALL,'report fsync wall boundary')
    inventory=r['artifact_sha256'];need(set(files)==set(inventory)|{'report.json','completion.json'} and
        all(hashlib.sha256(files[k]).hexdigest()==h for k,h in inventory.items()),'full artifact hash census')
    order=r['artifact_order'];need(len(order)==len(set(order)) and set(order)==set(inventory) and
        order[:2]==['binding.json','initial_roster.json'] and order[-1]=='native_seal.json','durable full order')
    need(read(folder/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and
         read(folder/'initial_roster.json')==p['rows'],'original inputs')
    need(read(folder/'native_seal.json',r['native_seal_sha256'])=={k:r[k] for k in ('rows','modules','counts','errors')},'native seal')
    events=[read(folder/f'event_{i:04d}.json') for i in range(r['event_count'])]
    used={'binding.json','initial_roster.json','native_seal.json',*(f'event_{i:04d}.json' for i in range(len(events)))}
    need([order.index(f'event_{i:04d}.json') for i in range(len(events))]==sorted(order.index(f'event_{i:04d}.json') for i in range(len(events))),'event order')
    for i,e in enumerate(events):
        pos=order.index(f'event_{i:04d}.json')
        if e['kind']=='module_return':
            mod=e['entry'];attempt=next(j for j,x in enumerate(events[:i]) if x['kind']=='module_attempt' and x['entry']['index']==mod['index'])
            need(all(order.index(f'event_{attempt:04d}.json')<order.index(x['file'])<pos for x in [*mod['inputs'],mod['output']]),'module IO persistence order')
        if e['kind']=='image_return':
            row=e['row'];attempt=next(j for j,x in enumerate(events[:i]) if x['kind']=='image_attempt' and x['scope']==e['scope'])
            need(all(order.index(f'event_{attempt:04d}.json')<order.index(row[k]['file'])<pos for k in ('pixels','preprocessed','latent')),'image persistence order')
    def blob(name): used.add(name);return files[name]
    rows=validate(p,r,events,lambda d:bridge.rgb.load_array(d,blob))
    need(used==set(inventory),'no unreferenced files')
    bridge.verify_bindings(p['bindings']);need(all(sha(folder/k)==hashlib.sha256(v).hexdigest() for k,v in files.items()),'final file seals')
    return dict(schema=NAME+'.consume',status='fixed24_saved_rgb_verified_not_admitted',report_sha256=expected,
        prepare_sha256=r['prepare_sha256'],completion_sha256=sha(folder/'completion.json'),rows=rows,denominator=24,
        counts=r['counts'],resources=r['resources'],gpu_peaks=r['gpu_peaks'],bindings=p['bindings'],
        inherited_physical_counts=p['inherited_physical_counts'],known_roots=2,fresh_roots=0,
        preregistration_verified=False,P1_admission=False,P2_admission=False,FLOPs=None,new_physics=0,new_predict=0)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=('prepare','verify','run','consume','verify-consume'))
    for k in ('prepared','receipt','physical-output','report-sha','receipt-sha','prepare-sha','source-sha','output','rgb-output'):
        ap.add_argument('--'+k)
    ap.add_argument('--main-authorized',action='store_true');a=ap.parse_args()
    if a.command=='prepare': result=write(a.output,build(a.prepared,a.receipt,a.physical_output,a.report_sha,a.receipt_sha,a.prepare_sha,a.source_sha))
    elif a.command=='verify': verify(a.prepared,a.prepare_sha);result='verified'
    elif a.command=='run': result=run(a.prepared,a.prepare_sha,a.output,a.main_authorized)
    elif a.command=='consume': result=write(a.output,consume(a.rgb_output,a.report_sha))
    else:
        saved=read(a.output,a.receipt_sha);need(saved==consume(a.rgb_output,a.report_sha),'whole saved consume mismatch');result='verified'
    print(result)


if __name__=='__main__': main()
