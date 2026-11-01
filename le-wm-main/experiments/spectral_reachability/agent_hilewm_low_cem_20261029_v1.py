"""MAIN-only installed low CEM on paid RGB/high-FIRST; normalized integration only.

Cold import is NumPy-only. No encoder, foreign model class, pickle execution,
physical action release, training, or CPU neural fallback. All four known slots.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import io
import math
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import numpy as np
from experiments.spectral_reachability import agent_hilewm_rgb_high_low_20261028_v1 as base

TAG='agent_hilewm_low_cem_20261029_v1'
ROOT=base.ROOT
need,encode,digest=base.need,base.encode,base.digest
CAP,FLOOR,WALL,RESERVE=32*1024**2,200*1024**2,600,4*1024**2
PARENT_MANIFEST=ROOT/'results/agent_hilewm_rgb_high_low_20261028_v1_prepare_MAIN01/manifest.json'
PARENT_MH='a3255d38e9cf718f9560c1b9e381abe38750edd9e91d3f04e5ad332cf414d7e9'
PARENT_REPORT=ROOT/'results/agent_hilewm_rgb_high_low_20261028_v1_cuda_MAIN01/report.json'
PARENT_RH='083d0d9aa190b561e6ccf865b09d7224b97d404b16ba38c792bd8af386f5622a'
PARENT_RECEIPT=ROOT/'results/agent_hilewm_rgb_high_low_20261028_v1_consume_MAIN01.json'
PARENT_CH='40f28414682c9540e9e195b361a36dfc514f816358adbeacb379fedc460e4ef0'
PINS={str(Path(base.__file__).resolve()):'ad7e19e8c1024c51227307b219b32bdc448e85ca27b6a3a4cc8e5708258bda2d',
    str(ROOT/'tests/test_agent_hilewm_rgb_high_low_20261028_v1.py'):'aa5d476d324e2d35d40966015f38093c70da549f91fe610b9bc96342d57fd3eb',
    str(ROOT/'innerdocs/agent_hilewm_rgb_high_low_20261028_v1.md'):'86409c4cd6470a1030774c82d5fa8023576e7c7e456d6e0ad7caeed70c9ec540',
    str(base.seam.PACKAGE/'solver/cem.py'):'a6662e7f1f2c6ecaf0a8527255d531aa05ed583339aabdf2798d0206382ef51b'}
INFO=('z_hist','a_hist','z_subgoal')


def protocol():
    return dict(schema=TAG+'.protocol',order=[list(x) for x in base.ORDER],known_roots=2,fresh_roots=0,
        search=dict(samples=8,iterations=3,topk=3,horizon=2,action_block=5,grouped_width=10,
            seed_base=1729,var_scale=1.,initial='full zero (1,2,10)',receding_horizon=1,
            bounds=[-1.,1.],clamp=False,bounds_meaning='synthetic normalized dimensional Box only'),
        history='single paid encoded z_init; zero a_hist; append outgoing candidate before action embedding; ctx=min(3,z,embedded actions)',
        target='exact paid high FIRST, not terminal; immutable all4',
        resources=dict(low_cost=12,candidate_rows=96,action_encoder=24,low_predictor=24,low_pred_proj=24,
            predictor_sample_rows=192,predictor_context_tokens=288,action_context_tokens=480,projection_rows=288,
            encoder=0,projector=0,high_predictor=0,AD=0,physics=0,FLOPs=None),
        final_mean_evaluated=False,extra_winner_replay=0,raw_inverse='no registered frozen original mean/scale authority; hardfail',
        output_cap_bytes=CAP,floor_bytes=FLOOR,wall_seconds=WALL,final_reserve_bytes=RESERVE,
        physical_release=False,full_policy=False,official_baseline=False,common_backbone_equivalence=False,
        P1_admission=False,scientific_admission=False,training=False)


def closure():
    for p,h in PINS.items():need(base.sha(p)==h,'frozen dependency '+p)
    out=base.closure();out.update(PINS)
    for p in (Path(__file__).resolve(),ROOT/'tests'/('test_'+TAG+'.py'),ROOT/'innerdocs'/(TAG+'.md')):
        out[str(p)]=base.sha(p)
    return out


def load_array(folder,name,spec):
    need(isinstance(name,str) and Path(name).name==name and name.endswith('.npy'),'flat NPY')
    body=base.loader.bytes_read(Path(folder)/name,CAP)
    need(len(body)==spec['file_bytes'] and digest(body)==spec['file_sha256'],'NPY full bytes SHA')
    header=io.BytesIO(body);version=np.lib.format.read_magic(header)
    need(version in ((1,0),(2,0)),'bounded NPY version')
    shape,fortran,dtype=(np.lib.format.read_array_header_1_0(header) if version==(1,0)
        else np.lib.format.read_array_header_2_0(header))
    need(dtype in (np.dtype('<f4'),np.dtype('<i8')) and all(type(d) is int and d>=0 for d in shape)
        and math.prod(shape)*dtype.itemsize==len(body)-header.tell() and len(body)<=CAP,'NPY header size before allocation')
    f=io.BytesIO(body);a=np.load(f,allow_pickle=False)
    need(f.tell()==len(body) and a.dtype.str==spec['dtype'] and list(a.shape)==spec['shape'] and
        a.dtype in (np.dtype('<f4'),np.dtype('<i8')) and np.isfinite(a).all() and digest(a.tobytes())==spec['data_sha256'],
        'NPY ABI finite/no trailing/raw SHA')
    return a


def upstream():
    accepted=base.consume(PARENT_MANIFEST,PARENT_MH,PARENT_REPORT,PARENT_RH)
    need(accepted==base.read(PARENT_RECEIPT,PARENT_CH),'whole persisted parent consume equality')
    m=base.verify(PARENT_MANIFEST,PARENT_MH);r=base.read(PARENT_REPORT,PARENT_RH);slots=[]
    need([(x['root_slot'],x['condition']) for x in r['rows']]==list(base.ORDER),'fixed original four')
    for i,(row,summary,slot) in enumerate(zip(r['rows'],accepted['rows'],m['slots'],strict=True)):
        values={k:load_array(PARENT_REPORT.parent,row[k],r['arrays'][row[k]]) for k in ('z_init','high_target')}
        need(all(v.dtype==np.float32 and v.shape==(1,192) for v in values.values()),'paid f32 vector ABI')
        need(digest(values['high_target'].tobytes())==summary['high_target_sha256']==summary['low_goal_sha256'],
            'paid FIRST exact delivered goal')
        slots.append(dict(index=i,root_slot=row['root_slot'],condition=row['condition'],RGB_source=deepcopy(slot),
            buffers={k:dict(name=row[k],descriptor=r['arrays'][row[k]],values=v.tolist()) for k,v in values.items()}))
    return m,accepted,slots


def build():
    sources=closure();m,accepted,slots=upstream()
    sources.update({str(p):h for p,h in ((PARENT_MANIFEST,PARENT_MH),(PARENT_REPORT,PARENT_RH),(PARENT_RECEIPT,PARENT_CH))})
    for s in slots:
        for b in s['buffers'].values():sources[str(PARENT_REPORT.parent/b['name'])]=b['descriptor']['file_sha256']
    need(all(base.sha(p)==h for p,h in sources.items()),'complete source/input bytes')
    return dict(schema=TAG+'.prepare',protocol=protocol(),sources=sources,slots=slots,export=m['export'],
        model=m['model'],parent=dict(manifest=str(PARENT_MANIFEST),manifest_sha256=PARENT_MH,
            report=str(PARENT_REPORT),report_sha256=PARENT_RH,receipt=str(PARENT_RECEIPT),receipt_sha256=PARENT_CH),
        inherited_resources=accepted['resources'],scientific_admission=False)


def scoped(path):
    p=Path(path);need(p.is_absolute() and p.resolve()==p and p.parent==ROOT/'results' and p.name.startswith(TAG+'_'),
        'exclusive own-stem results path');return p


def write(path,body):
    p=Path(path);need(not p.exists() and len(body)<=CAP and shutil.disk_usage(p.parent).free>=FLOOR+len(body),'exclusive write cap/floor')
    with p.open('xb') as f:need(f.write(body)==len(body),'full write');f.flush();os.fsync(f.fileno())


def prepare(output):
    m=build();p=scoped(output);need(shutil.disk_usage(p.parent).free>=CAP+FLOOR,'cap+floor');p.mkdir()
    write(p/'manifest.json',encode(m));return dict(manifest=str(p/'manifest.json'),sha256=base.sha(p/'manifest.json'))


def verify(path,h):
    m=base.read(path,h);need(m==build() and base.sha(path)==h,'entire manifest reconstruction');return m


class Store(base.Store):
    # array()/put() retain original dtype/bytes and exclusive xb; own guard/path.
    def __init__(self,path,start):
        self.path=scoped(path);self.start=start;self.used=0;self.arrays={}
        need(shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'whole cap+floor');self.path.mkdir()
    def guard(self,n=0,final=False):
        need(final or time.monotonic()-self.start<=WALL,'wall budget')
        need(self.used+n<=CAP-(0 if final else RESERVE) and shutil.disk_usage(self.path).free>=FLOOR+n,'own cap/reserve/floor')


def low_shape(info,candidates):
    need(set(info)==set(INFO)|{'planner_level'} and info['planner_level']=='low','original low metadata')
    need(len(candidates.shape)==4,'candidate rank4');b,s,h,d=candidates.shape
    need(b==1 and type(s) is int and 1<=s<=64 and type(h) is int and 1<=h<=16 and d==10,'bounded variable samples/H,width10')
    need(info['z_hist'].shape==(b,s,1,192) and info['a_hist'].shape==(b,s,1,10) and
        info['z_subgoal'].shape==(b,s,192),'installed CEM expanded singlecontext')
    return b,s,h


class OwnedLow:
    """Original rollout_low equations on tensor-only owned nodes, not root override.

    Candidate S/H variable; certified context3,width10; single initial observation.
    Original action Embedder forward unsupported by owned class: explicitly use
    its original kernel1 Conv1d then the embedded MLP, as frozen RGB bridge does.
    """
    def __init__(self,t,root,trace):self.t,self.root,self.trace=t,root,trace;self.rollouts=[]
    def parameters(self):return self.root.parameters()
    def get_cost(self,info,candidates):
        b,s,h=low_shape(info,candidates);t,r,tr=self.t,self.root,self.trace
        def kernel():
            z=info['z_hist'].reshape(b*s,1,192);a=info['a_hist'].reshape(b*s,1,10);pred=[]
            for step in range(h):
                a=t.cat((a,candidates[:,:,step].reshape(b*s,1,10)),dim=1)
                def embed():
                    n=r.action_encoder.patch_embed
                    return r.action_encoder.embed(t.nn.functional.conv1d(a.transpose(1,2),n.weight,n.bias).transpose(1,2))
                emb=tr.call('action_encoder',[a],embed)
                ctx=min(3,z.shape[1],emb.shape[1]);zi,ai=z[:,-ctx:],emb[:,-ctx:]
                hidden=tr.call('low_predictor',[zi,ai],lambda:r.low_predictor(zi,ai))
                proj=tr.call('low_pred_proj',[hidden.reshape(-1,192)],lambda:r.low_pred_proj(hidden.reshape(-1,192)))
                nxt=proj.reshape(b*s,ctx,192)[:,-1:];z=t.cat((z,nxt),dim=1);pred.append(nxt[:,0])
            roll=t.stack(pred,dim=1).reshape(b,s,h,192);self.rollouts.append(tr.store.array(roll))
            return (roll[:,:,-1]-info['z_subgoal']).pow(2).sum(dim=-1)
        return tr.call('low_cost',[info[k] for k in INFO]+[candidates],kernel)


class LowBatchBridge:
    """Explicit tensor-only installed-solver shim; metadata restored after expand.

    Single environment, no tails/chunk ambiguity. Debit before validation/call;
    failures never refunded. No mutable module globals or high-schema relabel.
    """
    def __init__(self,low,info,abi):
        need(set(info)==set(INFO),'tensor-only solver info')
        self.low,self.solver_info,self.abi=low,dict(info),abi;self.calls=[]
        self.original={k:abi.fingerprint(v) for k,v in info.items()}
    def parameters(self):return self.low.parameters()
    def get_cost(self,info,candidates):
        call=dict(attempt=len(self.calls)+1,returned=False,metadata='low',candidate_rows=int(candidates.shape[0]*candidates.shape[1]))
        self.calls.append(call)
        need(len(self.calls)<=3 and sum(x['candidate_rows'] for x in self.calls)<=24,'no refund bridge budget')
        low_shape(dict(info,planner_level='low'),candidates)
        need(tuple(candidates.shape)==(1,8,2,10),'fixed production search')
        for k in INFO:
            expected=self.solver_info[k][:,None].expand(1,8,*self.solver_info[k].shape[1:])
            need(self.abi.accepts(info[k]) and self.abi.fingerprint(info[k])==self.abi.fingerprint(expected),'actual expanded same-context exact')
        call.update(expanded={k:self.abi.fingerprint(info[k]) for k in INFO},candidates=self.abi.fingerprint(candidates))
        value=self.low.get_cost(dict(info,planner_level='low'),candidates)
        call.update(cost=self.abi.fingerprint(value),returned=True);return value
    def receipt(self):
        return dict(schema=TAG+'.low_bridge',metadata='low',original=self.original,calls=self.calls,
            cost_attempts=len(self.calls),cost_returns=sum(x['returned'] for x in self.calls),
            candidate_rows_attempted=sum(x['candidate_rows'] for x in self.calls))


def search(t,CEM,Box,root,trace,row,z,target):
    device=z.device;info=dict(z_hist=z[:,None],a_hist=t.zeros((1,1,10),device=device,dtype=t.float32),z_subgoal=target)
    low=OwnedLow(t,root,trace);bridge=LowBatchBridge(low,info,base.seam.cuda_abi(t,device))
    row.update(cem=[],cem_reset=0,cem_start=0,cem_end=0)
    class Recorder:
        output_key=TAG
        def __init__(self):self.history=[]
        def reset(self):row['cem_reset']+=1;self.history=[]
        def start_batch(self):row['cem_start']+=1
        def end_solve(self):row['cem_end']+=1
        def __call__(self,**state):
            trace.store.guard();base.precision(t)
            row['cem'].append(dict(step=state['step'],arrays={k:trace.store.array(state[k]) for k in base.seam.KEYS}))
            self.history.append(dict(step=state['step']))
    try:
        solver=CEM(model=bridge,batch_size=1,num_samples=8,var_scale=1.,n_steps=3,topk=3,device=device,
            seed=1729+row['index'],callbacks=[Recorder()])
        solver.configure(action_space=Box(-1.,1.,shape=(1,2),dtype=np.float32),n_envs=1,
            config=SimpleNamespace(horizon=2,action_block=5))
        initial=t.zeros((1,2,10),device=device,dtype=t.float32);row['initial']=trace.store.array(initial)
        out=solver.solve(dict(bridge.solver_info),init_action=initial)
        row['cem_history']=out['callbacks'][TAG]
        row['outputs']={k:trace.store.array(out[k] if k=='actions' else out[k][0]) for k in ('actions','mean','var')}
        row['outputs']['costs']=trace.store.array(np.asarray(out['costs'],np.float32))
        grouped=out['actions'];row['primitive_normalized']=trace.store.array(grouped.reshape(1,10,2))
        row['first_packet_normalized']=trace.store.array(grouped[:,:1].reshape(1,5,2))
    finally:row['bridge']=bridge.receipt();row['rollouts']=low.rollouts


def inverse_transform(normalized,authority=None):
    """No authenticated original mean_/scale_ asset exists in this frozen scope.

    Deliberately no bool/caller dict/dataset-fit/common-backbone escape hatch.
    A future version must bind an independently certified authority consumer
    before implementing numeric raw delivery. This version accepts NONE.
    """
    need(False,'raw inverse unavailable: complete certified frozen original mean_/scale_, flags, dtype, units and dataset provenance required; no registered authority')


def resources(rows):
    es=[e for r in rows for e in r.get('events',[])];out={}
    for k in sorted({e['kind'] for e in es}):
        xs=[e for e in es if e['kind']==k]
        out[k]=dict(attempted=len(xs),returned=sum(e['returned'] for e in xs),
            input_rows=sum(e['layouts'][0]['shape'][0] for e in xs),
            context_tokens=sum(e['layouts'][0]['shape'][0]*(e['layouts'][0]['shape'][1] if len(e['layouts'][0]['shape'])==3 else 1) for e in xs))
    return dict(modules=out,bridge_candidate_rows=sum(r.get('bridge',{}).get('candidate_rows_attempted',0) for r in rows),
        encoder=0,projector=0,high_predictor=0,AD=0,physics=0,FLOPs=None)


def run_cuda(path,h,output,*,main_authorized=False):
    need(main_authorized is True,'MAIN only before I/O or runtime import')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','deterministic CUBLAS')
    start=time.monotonic();m=verify(path,h);store=Store(output,start);root=None
    r=dict(schema=TAG+'.report',manifest=str(Path(path).resolve()),manifest_sha256=h,protocol=protocol(),sources=m['sources'],
        inherited_resources=m['inherited_resources'],arrays=store.arrays,rows=[dict(index=s['index'],root_slot=s['root_slot'],
            condition=s['condition'],status='not_started',events=[]) for s in m['slots']],errors=[],completed=False,
        model_before=None,model_after=None,physical_release=False,scientific_admission=False)
    try:
        t,CEM,Box=base.seam.load_runtime();t.use_deterministic_algorithms(True)
        t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False;a=m['export']
        root,r['construction']=base.loader.construct_cuda(a['export_report'],a['export_report_sha256'],a['export_receipt'],a['export_receipt_sha256'],main_authorized=True)
        r['model_before']=base.paid.diag.old.weights(root)
        need(r['model_before']==r['construction']['tensor_value_hashes'],'all original425 tensors')
        device=next(root.parameters()).device;r['runtime']=dict(device=str(device),torch=str(t.__version__),cuda=t.version.cuda)
        from torch.nn.attention import sdpa_kernel,SDPBackend
        with t.inference_mode(),t.autocast('cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            r['precision_before']=base.precision(t)
            for row,slot in zip(r['rows'],m['slots'],strict=True):
                row['status']='attempted';trace=base.Trace(t,store,row['events'])
                z,target=[t.from_numpy(np.asarray(slot['buffers'][k]['values'],np.float32)).to(device) for k in ('z_init','high_target')]
                row['z_init']=store.array(z);row['target']=store.array(target)
                search(t,CEM,Box,root,trace,row,z,target)
                need(not trace.stack,'all native calls returned');row['status']='returned'
            r['precision_after']=base.precision(t)
        r['model_after']=base.paid.diag.old.weights(root)
        need(r['model_before']==r['model_after'] and verify(path,h)==m,'source/input/weights before-after unchanged')
        store.guard();r['completed']=True
    except BaseException as e:r['errors'].append(dict(type=type(e).__name__,message=str(e)[:3000]))
    finally:
        if root is not None and r['model_after'] is None:
            try:r['model_after']=base.paid.diag.old.weights(root)
            except BaseException as e:r['errors'].append(dict(type=type(e).__name__,message=str(e)[:1000]))
        r.update(resources=resources(r['rows']),elapsed_seconds=time.monotonic()-start)
        store.put('report.json',encode(r),final=True)
    need(r['completed'] and not r['errors'],'partial preserved, no retry/refund')
    return dict(report=str(store.path/'report.json'),sha256=base.sha(store.path/'report.json'))


def validate_native(row,arrays,device,z,target):
    es=row['events'];cursor=0;costs=[]
    def event(kind,parent,nargs):
        nonlocal cursor
        need(cursor<len(es),'missing native event');e=es[cursor];cursor+=1
        need(e['index']==cursor-1 and e['parent']==parent and e['kind']==kind and e['returned'] is True
            and len(e['inputs'])==len(e['layouts'])==nargs,'native chronology/kind/parent/attempt-return')
        ins=[arrays[n] for n in e['inputs']];out=arrays[e['output']]
        for a,l in zip(ins+[out],e['layouts']+[e['output_layout']],strict=True):
            need(a.dtype==np.float32,'native original f32');base.checked_layout(l,a,device)
        return e,ins,out
    for iteration in range(3):
        e,ins,value=event('low_cost',None,4);zh,ah,g,c=ins
        need(zh.shape==(1,8,1,192) and ah.shape==(1,8,1,10) and g.shape==(1,8,192)
            and c.shape==(1,8,2,10) and value.shape==(1,8),'native cost ABI')
        base.exact(zh,np.repeat(z[:,None,None],8,axis=1),'paid z -> singlecontext')
        base.exact(ah,np.zeros((1,8,1,10),np.float32),'original zero incoming history')
        base.exact(g,np.repeat(target[:,None],8,axis=1),'paid FIRST -> low goal')
        hist=zh.reshape(8,1,192);actions=ah.reshape(8,1,10);pred=[]
        for step in range(2):
            actions=np.concatenate((actions,c[:,:,step].reshape(8,1,10)),axis=1)
            _,ai,emb=event('action_encoder',e['index'],1)
            base.exact(ai[0],actions,'append outgoing candidate then embed full history')
            need(emb.shape==(8,step+2,192),'full original action embeddings')
            ctx=min(3,hist.shape[1],emb.shape[1]);_,pi,hidden=event('low_predictor',e['index'],2)
            base.exact(pi[0],hist[:,-ctx:],'low history exact');base.exact(pi[1],emb[:,-ctx:],'low outgoing action alignment')
            need(hidden.shape==(8,ctx,192),'complete native hidden')
            _,ji,proj=event('low_pred_proj',e['index'],1)
            base.exact(ji[0],hidden.reshape(-1,192),'native projection input');need(proj.shape==(8*ctx,192),'native projection rows')
            nxt=proj.reshape(8,ctx,192)[:,-1:];hist=np.concatenate((hist,nxt),axis=1);pred.append(nxt[:,0])
        roll=np.stack(pred,axis=1).reshape(1,8,2,192)
        base.exact(arrays[row['rollouts'][iteration]],roll,'full saved native rollout')
        base.check_cost(value,roll[:,:,-1],g);costs.append((ins,value))
    need(cursor==len(es),'complete census rejects unknown/extra native events');return costs


def validate_cem(row,arrays,costs,z,target):
    device=row['events'][0]['layouts'][0]['device']
    def fingerprint(saved,a,expanded=False):
        base.seam.check_fingerprint(saved,a,expanded=expanded)
        need(saved['device']==device,'bridge same native CUDA device')
    need(row['cem_reset']==row['cem_start']==row['cem_end']==1 and len(row['cem'])==3 and
        row['cem_history']==[dict(step=i) for i in range(3)],'actual installed callback lifecycle')
    b=row['bridge'];need(b['schema']==TAG+'.low_bridge' and b['metadata']=='low' and
        b['cost_attempts']==b['cost_returns']==3 and b['candidate_rows_attempted']==24 and len(b['calls'])==3,
        'full low bridge budget/metadata')
    info=dict(z_hist=z[:,None],a_hist=np.zeros((1,1,10),np.float32),z_subgoal=target)
    need(set(b['original'])==set(INFO),'complete original tensor witness')
    for k,v in info.items():fingerprint(b['original'][k],v)
    mean=np.zeros((1,2,10),np.float32);std=np.ones_like(mean);base.exact(arrays[row['initial']],mean,'full zero init')
    for i,(cb,call,(ins,value)) in enumerate(zip(row['cem'],b['calls'],costs,strict=True)):
        need(cb['step']==i and set(cb['arrays'])==set(base.seam.KEYS) and call['attempt']==i+1 and
            call['returned'] is True and call['metadata']=='low' and call['candidate_rows']==8 and set(call['expanded'])==set(INFO),
            'native CEM bridge chronology')
        x={k:arrays[n] for k,n in cb['arrays'].items()};shapes=dict(candidates=(1,8,2,10),costs=(1,8),topk_inds=(1,3),topk_vals=(1,3),
            topk_candidates=(1,3,2,10),**{k:(1,2,10) for k in ('mean','var','prev_mean','prev_var')})
        need(all(a.shape==shapes[k] and a.dtype==(np.int64 if k=='topk_inds' else np.float32) for k,a in x.items()),'actual CEM dtype/shapes')
        base.exact(x['candidates'],ins[3],'CEM -> native candidates');base.exact(x['costs'],value,'native cost -> CEM')
        for k,a in zip(INFO,ins[:3],strict=True):fingerprint(call['expanded'][k],a,expanded=True)
        fingerprint(call['candidates'],ins[3]);fingerprint(call['cost'],value)
        native=row['events'][i*7]
        for k,l in zip(INFO,native['layouts'][:3],strict=True):
            need(call['expanded'][k]['stride']==l['stride'],'actual bridge/native expanded layout')
        need(call['candidates']['stride']==native['layouts'][3]['stride'] and
            call['cost']['stride']==native['output_layout']['stride'],'actual bridge/native candidate/cost layout')
        ix=x['topk_inds'];need(((ix>=0)&(ix<8)).all() and len(set(ix[0]))==3,'unique bounded elites')
        elite=x['candidates'][np.arange(1)[:,None],ix]
        base.exact(elite,x['topk_candidates'],'gather exact');base.exact(value[np.arange(1)[:,None],ix],x['topk_vals'],'elite values exact')
        base.exact(np.sort(value,axis=1)[:,:3],x['topk_vals'],'smallest3 actual costs')
        base.exact(x['prev_mean'],mean,'mean chain');base.exact(x['prev_var'],std,'STD chain')
        base.exact(x['candidates'][:,0],mean,'mean candidate index0')
        base.seam.close_float(x['mean'],elite.astype(float).mean(axis=1),'CEM numpy mean reducer only')
        base.seam.close_float(x['var'],elite.astype(float).std(axis=1,ddof=1),'CEM numpy sampleSTD reducer only')
        mean,std=x['mean'],x['var']
    need(set(row['outputs'])=={'actions','mean','var','costs'},'full final output')
    for k,a in (('actions',mean),('mean',mean),('var',std)):base.exact(arrays[row['outputs'][k]],a,'final '+k)
    base.seam.close_float(arrays[row['outputs']['costs']],x['topk_vals'].astype(float).mean(axis=1),'CEM logging reducer only')
    base.exact(arrays[row['primitive_normalized']],mean.reshape(1,10,2),'exact grouped->primitive reshape')
    base.exact(arrays[row['first_packet_normalized']],mean[:,:1].reshape(1,5,2),'official first receding packet reshape')


def check_model(r,m):
    c=r['construction'];device=r['runtime']['device'];a=m['export']
    need(device.startswith('cuda:') and c['device']==device and c['torch']==r['runtime']['torch'] and
        c['all_parameters_frozen'] is True and c['tensor_copy_attempts']==c['tensor_copy_returns']==425 and
        r['model_before']==r['model_after']==c['tensor_value_hashes'],'original425 CUDA tensors unchanged')
    _,spec,_,export=base.loader.authorities(a['export_report'],a['export_report_sha256'],a['export_receipt'],a['export_receipt_sha256'])
    expected={e['key']:x['array']['value_sha256'] for x,e in zip(export['rows'],spec['entries'],strict=True) if x['array'] is not None}
    need(r['model_before']==expected and c['export_report_sha256']==a['export_report_sha256'] and
        c['export_receipt_sha256']==a['export_receipt_sha256'] and c['ir_sha256']==base.loader.IR_SHA,'original export authority')
    return device


def consume(path,h,report_path,rh):
    m=verify(path,h);rp=Path(report_path)
    need(digest(base.loader.bytes_read(rp,CAP))==rh,'own bounded report bytes')
    r=base.read(rp,rh)
    need(r['schema']==TAG+'.report' and r['manifest']==str(Path(path).resolve()) and r['manifest_sha256']==h and
        r['protocol']==protocol() and r['sources']==m['sources'] and r['inherited_resources']==m['inherited_resources'] and
        r['completed'] is True and not r['errors'] and r['physical_release'] is False and r['scientific_admission'] is False
        and 0<=r['elapsed_seconds']<=WALL,'complete own-schema normalized integration')
    need([(x['root_slot'],x['condition']) for x in r['rows']]==list(base.ORDER) and
        [x['index'] for x in r['rows']]==list(range(4)),'all4 no filtering')
    need(r['precision_before']==r['precision_after']==base.precision_expected(),'native precision guards')
    device=check_model(r,m);files=list(rp.parent.iterdir())
    need(all(p.is_file() and not p.is_symlink() for p in files) and {p.name for p in files}==set(r['arrays'])|{rp.name}
        and sum(p.stat().st_size for p in files)<=CAP,'whole output census/cap')
    arrays={k:load_array(rp.parent,k,s) for k,s in r['arrays'].items()};summaries=[]
    for row,slot in zip(r['rows'],m['slots'],strict=True):
        need(row['status']=='returned' and len(row['rollouts'])==3,'full returned row')
        z,target=[np.asarray(slot['buffers'][k]['values'],np.float32) for k in ('z_init','high_target')]
        base.exact(arrays[row['z_init']],z,'original paid encoded init');base.exact(arrays[row['target']],target,'original paid FIRST')
        costs=validate_native(row,arrays,device,z,target);validate_cem(row,arrays,costs,z,target)
        summaries.append(dict(index=row['index'],root_slot=row['root_slot'],condition=row['condition'],RGB_source=slot['RGB_source'],
            target_sha256=digest(target.tobytes()),primitive_normalized=arrays[row['primitive_normalized']].tolist(),
            first_packet_normalized=arrays[row['first_packet_normalized']].tolist(),raw_actions=None,
            final_mean_evaluated=False,raw_inverse_authority=None))
    used=set()
    def refs(v):
        if isinstance(v,dict):
            for x in v.values():refs(x)
        elif isinstance(v,list):
            for x in v:refs(x)
        elif isinstance(v,str) and v.endswith('.npy'):used.add(v)
    refs(r['rows']);need(used==set(arrays),'all buffers referenced no extras')
    actual=resources(r['rows']);need(actual==r['resources'] and actual['bridge_candidate_rows']==96,'independent resources')
    expected=dict(low_cost=12,action_encoder=24,low_predictor=24,low_pred_proj=24)
    need(set(actual['modules'])==set(expected),'only authorized modules')
    for k,n in expected.items():need(actual['modules'][k]['attempted']==actual['modules'][k]['returned']==n,'all4 module counts '+k)
    for k,axis,n in (('low_predictor','input_rows',192),('low_predictor','context_tokens',288),
        ('action_encoder','context_tokens',480),('low_pred_proj','input_rows',288)):
        need(actual['modules'][k][axis]==n,'resource axis '+k+'/'+axis)
    need(verify(path,h)==m and base.sha(rp)==rh,'final full source/asset closure')
    return dict(schema=TAG+'.consume',status='integrated_normalized_low_CEM_not_physical_release',manifest_sha256=h,report_sha256=rh,
        rows=summaries,resources=actual,inherited_resources=m['inherited_resources'],known_roots=2,fresh_roots=0,
        physical_release=False,official_baseline=False,full_policy=False,P1_admission=False,scientific_admission=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('manifest','manifest-sha256','output','report','report-sha256','receipt'):p.add_argument('--'+k)
    p.add_argument('--main-authorized',action='store_true');a=p.parse_args(argv)
    if a.command=='prepare':value=prepare(a.output)
    elif a.command=='verify':verify(a.manifest,a.manifest_sha256);value=dict(status='verified_not_authorized')
    elif a.command=='run':value=run_cuda(a.manifest,a.manifest_sha256,a.output,main_authorized=a.main_authorized)
    else:
        value=consume(a.manifest,a.manifest_sha256,a.report,a.report_sha256);write(scoped(a.receipt),encode(value))
    print(encode(value).decode(),end='');return 0


if __name__=='__main__':raise SystemExit(main())
