"""MAIN-only frozen P2 encode of all six native cp0/5/10/15; no new predict/physics."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import itertools
import os
from pathlib import Path
import shutil
import time

import numpy as np
from experiments.spectral_reachability import agent_p2_six_target_physical_20261029_v1 as physical
from experiments.spectral_reachability import agent_p2_three_program_rgb_20261029_v1 as oldrgb
from experiments.spectral_reachability import agent_p2_six_target_prediction_comparison_20261029_v1 as comparison
from experiments.spectral_reachability import agent_p2_six_target_rgb_roster_20261029_v1 as roster

NAME='agent_p2_six_target_rgb_20261029_v1'
AUTH='MAIN_SIX_TARGET24_RGB_HOST_CUDA_ONLY'
ROOT=physical.ROOT
bridge=oldrgb.bridge
need,sha,encode=physical.require,physical.sha,physical.raw
CAP,FINAL,FLOOR,WALL,META=256*1024**2,32*1024**2,200*1024**2,900,128*1024**2
ALLOC,RESERVED=3*1024**3,4*1024**3
PAID_REPORT_SHA='c3733a752d1e9b80929edf9726bc0eba2ad1c2ae39b6f05550652ff7b3b74dbd'
PAID_RECEIPT_SHA='6e5fa7c57721cd4765430c7379d501d19978bdb44eebe96dce94f7bc68d9a3db'
NATIVE_REPORT_SHA='ddbb339f87e0f49b1d99d32ae4f41340d72e6c19f8199fe69e6d0261f78bca2b'
NATIVE_RECEIPT_SHA='7579693032046b2427a7d5b492a90131029fe87d39147d06fe49123a3cfead13'
PHYSICAL_SOURCE_SHA='30990cedd608578679e255b867b2e1a28f12724574049ba7f415de67b90e7a32'
BOUNDARY='budget start before verify through report fsync/hash; completion acknowledgement excluded'


def read(path,h=None):
    if h is not None:need(sha(path)==h,'exact external file SHA')
    return physical.read(path,META)


def contract():
    return dict(slots=6,rows=24,pairs=24,checkpoints=list(roster.CP),known_roots=2,fresh=False,
        CUDA_only=True,batch=1,no_dedup=True,no_padding=True,retries=0,
        hidden=[1,257,192],latent=192,max_encoder_API=24,max_projector_API=24,
        new_predict=0,new_AD=0,new_physics=0,new_goal_encode=0,training=False,
        prediction_source='paid actual deployment tokens only',Q='f32 -> f64 squares -> math.fsum /192',
        wall_seconds=WALL,cap_bytes=CAP,final_reserve=FINAL,disk_floor=FLOOR,
        allocated_cap=ALLOC,reserved_cap=RESERVED,wall_boundary=BOUNDARY,
        scientific_admission=False,P1_admission=False,P2_mechanism=False,raw_deployment_authorized=False,
        equal_generation_budget=False,equal_FLOPs=False,FLOPs=None,prior_physical_time=None)


def own_files(stem):
    return [ROOT/'experiments/spectral_reachability'/(stem+'.py'),ROOT/'tests'/('test_'+stem+'.py'),ROOT/'innerdocs'/(stem+'.md')]


def source_files():
    return sorted(set(bridge.source_files()+own_files(NAME)+own_files(roster.NAME)+own_files(comparison.TAG)+own_files(oldrgb.NAME)+
        physical.own_files()+[Path(oldrgb.__file__),ROOT/'experiments/checkpoint_compat.py',
        ROOT/'experiments/gpu_preflight.py',ROOT/'experiments/spectral_reachability/preflight_pusht_d2p_p0.py']))


def build(native_output,native_report_sha,native_receipt,native_receipt_sha):
    output=Path(native_output).resolve();receipt=Path(native_receipt).resolve()
    need(sha(physical.__file__)==PHYSICAL_SOURCE_SHA,'frozen own-six physical source before authentication')
    need(native_report_sha==NATIVE_REPORT_SHA and native_receipt_sha==NATIVE_RECEIPT_SHA,'fixed actual MAIN six native authority')
    nr=read(output/'report.json',native_report_sha);external=read(receipt,native_receipt_sha)
    strict=physical.consume(output,native_report_sha)
    need(encode(strict)==encode(external),'entire original native strict rebuilt, never status-only')
    need(strict['status']==physical.RECORDED and strict['errors']==[],'clean six native authority')
    b=nr['execution_binding'];p,_=physical.verify(b['prepared'],b['prepare_sha256'])
    args=p['arguments'];need(args['replay_report_sha']==PAID_REPORT_SHA and args['replay_receipt_sha']==PAID_RECEIPT_SHA,'actual paid not normalized prediction')
    # physical.consume/verify already fully reconstruct the exact paid receipt;
    # do not launch any model, rerun inverse or issue a new authentication file.
    paid=physical.paid;pm,d=paid.verify(args['replay_manifest'],args['replay_manifest_sha'])
    pr=read(args['replay_report'],PAID_REPORT_SHA);pc=read(args['replay_receipt'],PAID_RECEIPT_SHA)
    need(pc==p['planner_authority']['receipt'] and pr['model_before']==pr['model_after']==pm['expected_state'],'paid closure/frozen425')
    folder=Path(args['replay_report']).parent
    def get_trace(i):
        row=pr['rows'][i];tr=read(folder/row['trace']['file'],row['trace']['file_sha256'])
        return tr,{k:paid.six.load_array(folder,v) for k,v in row['arrays'].items()}
    result=roster.assemble(p,strict,pr,get_trace)
    sixm=d['controller_prepare'];apref=sixm['authorities']['arbitrary_prepare'];ap=read(apref['path'],apref['sha256'])
    need(sixm['sources'].get(apref['path'])==apref['sha256'],'original arbitrary prepare already in authenticated input closure')
    goalrefs=[s['task']['coordinate_refs']['task_goal'] for s in ap['slots']]
    need(goalrefs[0]==goalrefs[1] and sixm['sources'].get(goalrefs[0]['path'])==goalrefs[0]['sha256'],'original sealed task goal authority')
    gr=read(goalrefs[0]['path'],goalrefs[0]['sha256']);gf=Path(goalrefs[0]['path']).parent
    def get_source(name):
        desc=gr['arrays'][name];path=gf/name
        need(Path(name).name==name and sixm['sources'].get(str(path))==ap['bindings'].get(str(path))==desc['file_sha256'],'exact original goal/history array source')
        return paid.six.tracker.parity.load_array(gf,name,desc)
    goals=comparison.resolve_goals(sixm,ap,gr,get_source)
    rows=result['rows']
    for row in rows:
        i=row['slot'];plan=p['plans'][i];g=goals[row['root']]
        row.update(goal192=g['value'].tolist(),goal_source=g['source'],target192=plan['controller_payload']['target'][0],
            task_sha256=plan['task_sha256'],blind_sha256=plan['blind_sha256'],full_raw_bits=plan['raw_float32_bits'],
            original_z_sha256=plan['payload_sha256']['z'],original_a_sha256=plan['payload_sha256']['a'],
            target_origin=deepcopy(plan['target_origin']),source_time=deepcopy(plan['target_origin']['row']['origin']['physical_time']
                if i%3!=2 else plan['target_origin']['plan']['source_time']),
            pixel_source=None,pixels=None,preprocessed=None,modules=[])
        if row['frame'] is not None:
            row['pixel_source']=dict(row['frame'],path=str(output/row['frame']['png']))
            bridge.load_pixels(row['pixel_source'])
    bindings=deepcopy(strict['bindings'])
    physical.native.merge(bindings,{str(receipt):native_receipt_sha,str(output/'report.json'):native_report_sha,
        **{str(f):sha(f) for f in source_files()},**bridge.rgb.MODELS})
    need(bridge.native.MODEL_TENSOR_SHA==paid.six.tracker.parity.cat.MODEL_HASH,'exact paired P2 encoder/projector model')
    bridge.verify_bindings(bindings)
    n=sum(x['pixel_source'] is not None for x in rows)
    return dict(schema=NAME+'.prepare',arguments=dict(native_output=str(output),native_report_sha=native_report_sha,
        native_receipt=str(receipt),native_receipt_sha=native_receipt_sha),contract=contract(),rows=rows,bindings=bindings,
        expected_calls={k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        policy=str(bridge.rgb.POLICY),expected_model_hash=bridge.native.MODEL_TENSOR_SHA,expected_hi_state=pm['expected_state'],
        imported_native_receipt=external,imported_paid_receipt=pc,
        imported_controller_receipt=d['imported_complete_receipt'],goal_sources=[g['source'] for g in goals])


def write_metadata(path,v):
    path=Path(path).absolute();b=encode(v)
    need(path.name.startswith(NAME+'_') and not path.is_symlink() and len(b)<=META and shutil.disk_usage(path.parent).free>=FLOOR+len(b),'own bounded exclusive metadata')
    with path.open('xb') as f:need(f.write(b)==len(b),'full metadata');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(path.parent);return dict(path=str(path),sha256=sha(path))


def verify(path,h):
    p=read(path,h);need(p==build(**p['arguments']),'entire new RGB prepare rebuilt');return p


class Budget:
    def __init__(self,clock=time.monotonic,free=lambda:shutil.disk_usage('.').free):
        self.clock,self.free=clock,free;self.started=clock();self.used=0
    def check(self,size=0,final=False):
        need(final or self.clock()-self.started<=WALL,'inclusive whole900 wall')
        need(self.used+size<=CAP-(0 if final else FINAL) and self.free()>=FLOOR+size,'256MiB/reserve32MiB/floor200MiB')


class Store(bridge.rgb.Store):
    def __init__(self,path,budget):
        need(Path(path).name.startswith(NAME+'_'),'own new RGB directory')
        super().__init__(path,budget);self.index=0;self.order=[]
    def write(self,name,raw,final=False):
        h=super().write(name,raw,final=final);self.order.append(name);return h
    def event(self,e):self.put(f'event_{self.index:04d}.json',e);self.index+=1


def process(p,rows,store,encode_one):
    need(rows==p['rows'] and len(rows)==24,'full immutable initial roster')
    for row in rows:
        if row['pixel_source'] is None:continue
        store.check();scope=dict(row=row['index'],step=row['cp_public_step'],phase='encode')
        row['status']='attempted';store.event(dict(kind='image_attempt',scope=scope))
        pix=bridge.load_pixels(row['pixel_source']);row['pixels']=store.array(f'r{row["index"]}_pixels',pix)
        z=encode_one(pix,row,scope);roster.f32(z,(192,))
        row['latent']=store.array(f'r{row["index"]}_latent',z);row['status']='encoded'
        store.event(dict(kind='image_return',scope=scope,row=deepcopy(row)))


def accounting(p,rows,modules):
    calls=bridge.counts(modules)
    return dict(module_calls=calls,encoder_samples=calls['encoder']['completed'],projector_samples=calls['projector']['completed'],
        encoded_images=sum(x['status']=='encoded' for x in rows),fixed_positions=24,
        missing_positions=sum(x['pixel_source'] is None for x in p['rows']),
        persisted_tensor_captures=sum(len(x.get('inputs',[]))+int('output' in x) for x in modules)+
            sum(x.get('preprocessed') is not None for x in rows)+sum(x.get('latent') is not None for x in rows),
        all_cpu_transfer_calls=None,FLOPs=None,new_predict=0,new_physics=0,new_AD=0)


class Trace:
    """Read-only per-module hooks, no method/global monkeypatch."""
    def __init__(self,model,t,store,expected,check):
        self.model,self.t,self.store,self.expected,self.check=model,t,store,expected,check
        self.entries=[];self.hooks=[];self.scope=None;self.pending={}
    def save(self,x,name):
        need(x.device.type=='cuda' and x.dtype==self.t.float32,'CUDA f32 module I/O only')
        self.check();self.t.cuda.synchronize();v=x.detach().cpu().contiguous().numpy().copy()
        self.check();return self.store.array(name,v)
    def __enter__(self):
        def pre(kind):
            def hook(module,args):
                self.check();need(not self.t.is_grad_enabled() and self.scope is not None,'scoped inference-only')
                seq=[e['resource'] for e in self.entries if e['scope']==self.scope]
                need(kind in ('encoder','projector') and seq==([] if kind=='encoder' else ['encoder']) and
                     sum(e['resource']==kind for e in self.entries)<self.expected[kind] and module not in self.pending,'no extra/forbidden module calls')
                e=dict(index=len(self.entries),scope=deepcopy(self.scope),resource=kind,completed=False)
                self.entries.append(e);self.pending[module]=e;self.store.event(dict(kind='module_attempt',entry=deepcopy(e)))
                need(len(args)==1,'one native module tensor input');e['inputs']=[self.save(args[0],f'm{e["index"]}_in')]
            return hook
        def post(module,args,out):
            e=self.pending.pop(module)
            if hasattr(out,'last_hidden_state'):out=out.last_hidden_state
            need(tuple(out.shape)==((1,257,192) if e['resource']=='encoder' else (1,192)),'exact native257 hidden / projected192')
            e['output']=self.save(out,f'm{e["index"]}_out');e['completed']=True
            self.store.event(dict(kind='module_return',entry=deepcopy(e)))
        try:
            for kind in ('encoder','projector','action_encoder','predictor','pred_proj'):
                module=getattr(self.model,kind);self.hooks.append(module.register_forward_pre_hook(pre(kind)))
                if kind in ('encoder','projector'):self.hooks.append(module.register_forward_hook(post))
            return self
        except BaseException:self.__exit__(None,None,None);raise
    def __exit__(self,typ,exc,tb):
        errors=[]
        for h in reversed(self.hooks):
            try:h.remove()
            except BaseException as e:errors.append(str(e))
        need(not errors,'all module hooks restored '+str(errors))


def run(prepared,mh,output,*,main_authorized=False):
    need(main_authorized is True,'MAIN authorization before model/preflight/verify')
    budget=Budget();p=verify(prepared,mh);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);errors=[];model=trace=None;before=after=None;precision=None;determinism=None;host=None;peaks={}
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=mh))
    store.put('initial_roster.json',p['rows'])
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
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
            need(peaks['allocated']<=ALLOC and peaks['reserved']<=RESERVED,'actual CUDA memory caps')
        class Guard(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                check();value=func(*args,**(kwargs or {}));check();return value
        model=load_compatible_pretrained(p['policy'],device=torch.device('cuda:0')).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model)
        need(before==p['expected_model_hash'],'original paired model before')
        trace=Trace(model,torch,store,p['expected_calls'],check);transform=p0.image_transform()
        def encode_one(pix,row,scope):
            trace.scope=scope
            try:
                x=p0._image_tensor(pix,transform=transform,device=torch.device('cuda:0'))
                row['preprocessed']=trace.save(x,f'r{row["index"]}_preprocessed')
                bridge.same(x.detach().cpu().numpy(),bridge.pixels.preprocess_pixels(pix),'exact original CUDA preprocess')
                out=model.encode({'pixels':x})['emb'];need(tuple(out.shape)==(1,1,192),'original native emb')
                return out[0,0].detach().cpu().numpy().copy()
            finally:
                row['modules']=deepcopy([e for e in trace.entries if e['scope']==scope]);trace.scope=None
        with trace,torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH),Guard():
            process(p,rows,store,encode_one)
        check()
    except BaseException as e:errors.append(bridge.native.error('encode',e))
    finally:
        if precision is not None:
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=precision
        if determinism is not None:torch.use_deterministic_algorithms(determinism[0],warn_only=determinism[1])
        try:
            if model is not None:bridge.native.frozen(model);after=bridge.native.tensor_hash(model)
            bridge.verify_bindings(p['bindings']);need(sha(prepared)==mh,'prepare unchanged');budget.check()
        except BaseException as e:errors.append(bridge.native.error('final_guards',e))
        entries=[] if trace is None else trace.entries
        complete=not errors and before==after==p['expected_model_hash'] and bridge.counts(entries)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()}
        seal=dict(rows=rows,modules=entries,counts=bridge.counts(entries),errors=errors)
        seal_sha=store.put('native_seal.json',seal,final=True)
        r=dict(schema=NAME+'.report',contract=contract(),completed=complete,prepared=str(Path(prepared).resolve()),prepare_sha256=mh,
            **seal,native_seal_sha256=seal_sha,host=host,model_before=before,model_after=after,gpu_peaks=peaks,
            resources=accounting(p,rows,entries),
            event_count=store.index,artifact_sha256=deepcopy(store.hashes),artifact_order=list(store.order),wall_before_report=budget.clock()-budget.started)
        store.put('report.json',r,final=True);elapsed=budget.clock()-budget.started
        done=dict(schema=NAME+'.completion',report_sha256=sha(store.path/'report.json'),complete=bool(complete and elapsed<=WALL),wall_inclusive=elapsed,boundary=BOUNDARY)
        store.put('completion.json',done,final=True)
    need(done['complete'],'partial failed artifacts retained, no retry '+str(errors))
    return dict(report=str(store.path/'report.json'),sha256=done['report_sha256'])


def panel(p,rows,load):
    need(len(p['rows'])==len(rows)==24,'all24 panel denominator')
    for root in (0,1):
        g=np.asarray(p['rows'][12*root]['goal192'],np.float32)
        for slot in range(3*root,3*root+3):
            for j in range(4):
                roster.same(np.asarray(p['rows'][4*slot+j]['goal192'],np.float32),g,'one original task goal per root, not tracking target')
    out=[]
    for spec,row in zip(p['rows'],rows,strict=True):
        pred=np.asarray(spec['predicted192'],np.float32);target=np.asarray(spec['target192'],np.float32);goal=np.asarray(spec['goal192'],np.float32)
        z=None if row['latent'] is None else load(row['latent'])
        if z is not None:
            roster.f32(z,(192,))
            if spec['cp_public_step']==0:roster.same(z,pred,'cp0 real encoded root equals sealed observed latent exactly')
        mse=comparison.mse64
        f=spec['frame'];native=None if f is None else physical.independent.scalars(f['state'],spec['native_goal'])
        out.append(dict(index=spec['index'],slot=spec['slot'],root=spec['root'],arm=spec['arm'],cp_public_step=spec['cp_public_step'],
            missing=z is None,model_tracking_Q=mse(pred,target),model_task_Q=mse(pred,goal),
            actual_tracking_Q=None if z is None else mse(z,target),actual_task_Q=None if z is None else mse(z,goal),
            prediction_vs_encoded_MSE=None if z is None else mse(pred,z),model_minus_actual_task_Q=None if z is None else mse(pred,goal)-mse(z,goal),
            native=native,source=spec['pixel_source'],prediction_source=spec['prediction_pointer'],
            response=spec['response'],task_id=spec['task_id'],goal_source=spec['goal_source'],raw15_sha256=spec['raw15_sha256'],
            source_time=spec['source_time'],target_origin=spec['target_origin'],
            physical_benefit_claim=False,P2_mechanism=False))
    for slot in range(6):
        base=out[4*slot]
        for row in out[4*slot:4*slot+4]:
            row['model_task_reduction_from_cp0']=base['model_task_Q']-row['model_task_Q']
            row['actual_task_reduction_from_cp0']=None if row['missing'] or base['missing'] else base['actual_task_Q']-row['actual_task_Q']
            row['native_error_gain_from_cp0']=None if row['native'] is None or base['native'] is None else {
                k:base['native'][k]-row['native'][k] for k in physical.independent.ERRORS}
            row['native_reward_gain_from_cp0']=None if row['native'] is None or base['native'] is None else row['native']['native7_reward']-base['native']['native7_reward']
    pairs=[]
    for root in (0,1):
        for left,right in itertools.combinations(range(3*root,3*root+3),2):
            for j,cp in enumerate(roster.CP):
                a,b=out[left*4+j],out[right*4+j];ok=not a['missing'] and not b['missing']
                pairs.append(dict(root=root,left_slot=left,right_slot=right,cp_public_step=cp,comparable=ok,
                    right_minus_left_actual_task_Q=None if not ok else b['actual_task_Q']-a['actual_task_Q'],
                    native_delta=None if not ok else {k:float(b['native'][k])-float(a['native'][k]) for k in physical.independent.SCALARS},
                    cross_target_tracking_loss_difference=None))
    return dict(denominator=dict(slots=6,rows=24,pairs=24),rows=out,pairs=pairs,winner=None,scientific_admission=False)


def validate(p,r,events,load):
    need(r['schema']==NAME+'.report' and r['contract']==p['contract']==contract() and r['completed'] is True and r['errors']==[], 'complete new six RGB report')
    need(r['model_before']==r['model_after']==p['expected_model_hash'],'model strongly unchanged')
    need(r['host']['device']=='cuda:0' and 0<r['gpu_peaks']['allocated']<=ALLOC and 0<r['gpu_peaks']['reserved']<=RESERVED and
         r['gpu_peaks']['guard_checks']>0,'actual bounded CUDA peaks, no CPU fallback')
    need(len(p['rows'])==len(r['rows'])==24,'all24 saved denominator')
    modules=[];want=[]
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        mutable={'status','latent','pixels','preprocessed','modules'}
        need({k:v for k,v in spec.items() if k not in mutable}=={k:v for k,v in row.items() if k not in mutable},'exact sealed input roster')
        if spec['pixel_source'] is None:need(row==spec,'missing preserved exactly');continue
        need(row['status']=='encoded','all available frames encoded exactly once')
        pix=load(row['pixels']);bridge.same(pix,bridge.load_pixels(spec['pixel_source']),'native canonical pixels')
        pre=load(row['preprocessed']);bridge.same(pre,bridge.pixels.preprocess_pixels(pix),'preprocessing exact')
        mods=row['modules'];scope=dict(row=row['index'],step=row['cp_public_step'],phase='encode')
        need([x['resource'] for x in mods]==['encoder','projector'] and all(x['scope']==scope and x['completed'] is True and len(x['inputs'])==1 for x in mods),'full original module chain')
        en,proj=mods;ein,eout,pin,pout,z=(load(d) for d in (en['inputs'][0],en['output'],proj['inputs'][0],proj['output'],row['latent']))
        need(all(v.dtype==np.float32 and np.isfinite(v).all() for v in (pre,ein,eout,pin,pout,z)),'all original f32 module IO, no dtype coercion')
        bridge.physical.saved.overlay.visual_bridge(pre,ein,eout,pin,pout,z,[1,257,192])
        roster.same(pre.reshape(1,3,224,224),ein,'preprocess to encoder bytes');roster.same(eout[:,0],pin,'full hidden CLS to projector bytes');roster.same(pout[0],z,'projector to actual latent bytes')
        want.append(dict(kind='image_attempt',scope=scope))
        for mod in mods:
            want.extend([dict(kind='module_attempt',entry={k:mod[k] for k in ('index','scope','resource')}|dict(completed=False)),dict(kind='module_return',entry=mod)])
        want.append(dict(kind='image_return',scope=scope,row=row));modules.extend(mods)
    need(modules==r['modules'] and [x['index'] for x in modules]==list(range(len(modules))) and events==want and
         r['counts']==bridge.counts(modules)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()},'all events/module calls no unknown filtered')
    need(r['resources']==accounting(p,r['rows'],modules),'all actual resource axes independently rebuilt')
    return panel(p,r['rows'],load)


def consume(output,rh):
    folder=Path(output).resolve();need(sha(folder/'report.json')==rh,'external RGB report SHA')
    fs=list(folder.iterdir());need(all(f.is_file() and not f.is_symlink() for f in fs) and sum(f.stat().st_size for f in fs)<=CAP,'whole bounded regular census')
    files={f.name:f.read_bytes() for f in fs};r=read(folder/'report.json',rh);p=verify(r['prepared'],r['prepare_sha256'])
    done=read(folder/'completion.json')
    need(done['schema']==NAME+'.completion' and done['report_sha256']==rh and done['complete'] is True and
         done['boundary']==BOUNDARY and 0<=r['wall_before_report']<=done['wall_inclusive']<=WALL,'actual inclusive900 report boundary')
    inventory=r['artifact_sha256']
    need(set(files)==set(inventory)|{'report.json','completion.json'} and all(hashlib.sha256(files[k]).hexdigest()==h for k,h in inventory.items()),'entire file inventory exact')
    order=r['artifact_order'];need(len(order)==len(set(order)) and set(order)==set(inventory) and
        order[:2]==['binding.json','initial_roster.json'] and order[-1]=='native_seal.json','full durable order and input/output seal boundaries')
    need(read(folder/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and read(folder/'initial_roster.json')==p['rows'],'real original sealed inputs')
    seal=read(folder/'native_seal.json',r['native_seal_sha256']);need(seal=={k:r[k] for k in ('rows','modules','counts','errors')},'native seal before diagnostics')
    events=[read(folder/f'event_{i:04d}.json') for i in range(r['event_count'])];used={'binding.json','initial_roster.json','native_seal.json',*(f'event_{i:04d}.json' for i in range(r['event_count']))}
    need([order.index(f'event_{i:04d}.json') for i in range(len(events))]==sorted(order.index(f'event_{i:04d}.json') for i in range(len(events))),'actual complete event persistence order')
    for i,e in enumerate(events):
        pos=order.index(f'event_{i:04d}.json')
        if e['kind']=='module_return':
            mod=e['entry'];names=[x['file'] for x in [*mod['inputs'],mod['output']]]
            attempt=next(j for j,x in enumerate(events[:i]) if x['kind']=='module_attempt' and x['entry']['index']==mod['index'])
            need(all(order.index(f'event_{attempt:04d}.json')<order.index(n)<pos for n in names),'module actual IO captured between attempt/return')
        if e['kind']=='image_return':
            row=e['row'];attempt=next(j for j,x in enumerate(events[:i]) if x['kind']=='image_attempt' and x['scope']==e['scope'])
            need(all(order.index(f'event_{attempt:04d}.json')<order.index(row[k]['file'])<pos for k in ('pixels','preprocessed','latent')),'actual image arrays captured within image scope')
    def blob(name):used.add(name);return files[name]
    out=validate(p,r,events,lambda d:bridge.rgb.load_array(d,blob))
    need(used==set(inventory),'all arrays referenced; no unknown files')
    bridge.verify_bindings(p['bindings']);need(all(sha(folder/k)==hashlib.sha256(v).hexdigest() for k,v in files.items()),'final source/array bytes unchanged')
    return dict(schema=NAME+'.consume',status='all24_saved_RGB_positions_verified_science_blocked',report_sha256=rh,
        prepare_sha256=r['prepare_sha256'],completion_sha256=sha(folder/'completion.json'),**out,counts=r['counts'],
        inherited_native_counts=p['imported_native_receipt']['counts'],inherited_paid_receipt=p['imported_paid_receipt'],
        original_controller_receipt=p['imported_controller_receipt'],bindings=p['bindings'],FLOPs=None,all_cpu_transfer_calls=None,
        resources=r['resources'],gpu_peaks=r['gpu_peaks'],whole_wall_through_report_fsync_hash=done['wall_inclusive'],
        new_predict=0,new_physics=0,new_AD=0,equal_generation_budget=False,known_roots=2,fresh=False,prior_physical_time=None)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume','protocol'))
    for k in ('native-output','native-receipt','prepared','output','rgb-output'):parser.add_argument('--'+k,type=Path)
    for k in ('native-report-sha','native-receipt-sha','prepare-sha','report-sha'):parser.add_argument('--'+k)
    parser.add_argument('--main-authorized',action='store_true');a=parser.parse_args(argv)
    if a.command=='prepare':v=write_metadata(a.output,build(a.native_output,a.native_report_sha,a.native_receipt,a.native_receipt_sha))
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);v=dict(verified=True)
    elif a.command=='run':v=run(a.prepared,a.prepare_sha,a.output,main_authorized=a.main_authorized)
    elif a.command=='consume':
        need(a.rgb_output is not None,'consume requires --rgb-output');v=write_metadata(a.output,consume(a.rgb_output,a.report_sha))
    else:v=contract()
    print(encode(v).decode())


if __name__=='__main__':main()
