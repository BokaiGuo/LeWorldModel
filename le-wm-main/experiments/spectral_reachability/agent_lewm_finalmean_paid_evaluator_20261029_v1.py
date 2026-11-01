"""Known2 delivered final means: separate paid whole/segmented CUDA evaluation."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from experiments.spectral_reachability import agent_lewm_native_cem_slice_20261029_v1 as old

NAME='agent_lewm_finalmean_paid_evaluator_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
CAP,RESERVE,FLOOR,WALL=512*1024**2,64*1024**2,200*1024**2,900
AUTH='MAIN_FINALMEAN_PAID_HOSTCUDA_ONLY'
REPORT=ROOT/'results/agent_lewm_native_cem_slice_20261029_v1_cuda_MAIN01/report.json'
REPORT_SHA='ab3313d24b6efb3539bb1b1a70417f7aa41df1602157465d1da5cd1d26e8b965'
PREPARE_SHA='f8f773572326ddd52c407d14b850584228f29c5c3a14d0f44262129a676d276c'
RECEIPT=ROOT/'results/agent_lewm_native_cem_slice_20261029_v1_consume_MAIN01.json'
RECEIPT_SHA='85af90a95df1481f1bb2491f1db0adfd307be50d0d7a1f636d3e4b0a712e5ada'
OLD_SHA='87bb5269388ced3f4dc74d41550b4939c5cf5457946c1c1b63d0e272a8c88e26'
need,raw,sha,read,exact,durable=old.need,old.raw,old.sha,old.read,old.exact,old.durable
verify_upstream=old.verify
KINDS=('action_encoder','predictor','pred_proj')
LAYOUTS=('whole','segmented')


def protocol():
    return dict(roots=[0,1],known_only=True,fresh=False,condition='off',layouts=list(LAYOUTS),
        delivery='original returned mean, no new search or selection',normalized=True,renormalize=False,
        whole='official rollout cached actual0/5/10, incoming2+mean2, one AE4/two predict3',
        segmented='AE3/predict3 then causal shift with own pred15 and AE3/predict3; no fabricated RGB',
        seal='both roots/both layouts all predictions before all four criterion calls',
        resources=dict(rollout_API=2,segmented_predict_API=4,criterion_API=4,action_encoder=6,AE_samples=6,
            AE_tokens=20,predictor=8,predictor_samples=8,predictor_context_tokens=24,pred_proj=8,pred_proj_rows=24,
            encoder=0,projector=0,AD=0,physics=0,training=0),
        criterion='official float32 sum192, independent saved float64 sum192/mean192',
        comparison='all elements bitwise/maxabs/float32 ULP, no tolerance or layout change',
        output_cap=CAP,final_reserve=RESERVE,disk_floor=FLOOR,wall_seconds=WALL,
        cuda_allocated=3*1024**3,cuda_reserved=4*1024**3,requested_SDP='MATH',kernel_certificate=False,
        old_slice_mean_evaluated=False,equal_FLOPs=False,same_domain_as_bounded_CEM=False,
        physical_release=False,scientific_admission=False)


def build():
    need(sha(old.__file__)==OLD_SHA,'frozen old source')
    r=read(REPORT,REPORT_SHA);receipt=read(RECEIPT,RECEIPT_SHA)
    need(r['schema']==old.NAME+'.report' and r['status']=='all_known2_native_normalized_CEM_recorded' and
        r['prepare_sha256']==PREPARE_SHA and r['errors']==[] and r['source_verified'] is True and
        r['protocol']==old.protocol() and r['protocol']['final_mean_evaluated'] is False,'actual original CEM complete, still unevaluated')
    p=verify_upstream(r['prepared'],PREPARE_SHA)  # saved prepare closure, NOT owner fullstrict
    need(receipt['schema']==old.NAME+'.consume' and receipt['status']=='all_known2_native_normalized_CEM_saved_verified' and
        receipt['report_sha256']==REPORT_SHA and receipt['inherited']==p['inherited'] and
        receipt['per_root_resources']==old.protocol()['per_root'] and
        receipt['total_resources']=={k:2*v for k,v in old.protocol()['per_root'].items()} and
        r['model_before']==r['model_after']==p['expected_model_hash'] and len(r['rows'])==len(receipt['rows'])==2,
        'exact original complete receipt authority, no nested fullstrict')
    slots=[];sources=dict(p['sources'])
    for i,(row,attest,s) in enumerate(zip(r['rows'],receipt['rows'],p['slots'],strict=True)):
        need(row['root']==attest['root']==s['source_slot']==i and row['status']=='normalized_mean_sealed' and
            attest['iterations']==30 and attest['candidate_count']==9000 and attest['returned_mean_evaluated'] is False and
            attest['raw_actions'] is None and attest['physical_release'] is False,'all original delivery slots')
        refs=dict(row['base_inputs'],warm=row['warm'],actions=row['actions'])
        load=lambda x:old.load_array(REPORT.parent,x,r['artifacts'])
        delivered=load(row['actions'])
        need(delivered.shape==(1,2,10) and exact(delivered,load(row['mean'])) and
            exact(delivered,load(row['callbacks'][-1]['mean'])),'actual returned action/mean/last callback bitexact')
        for rec in [*refs.values(),row['mean'],row['callbacks'][-1]['mean']]:
            load(rec);sources[str(REPORT.parent/rec['file'])]=rec['sha256']
        need(exact(load(refs['emb']),np.asarray(s['emb'],np.float32)[None]) and
            exact(load(refs['warm']),np.asarray(s['warm'],np.float32)[None,None]) and
            exact(load(refs['goal_emb']),np.asarray(s['goal_emb'],np.float32)[None,None]),'original context/warm/goal identity')
        slots.append(dict(root=i,refs=refs,inventory={x['file']:r['artifacts'][x['file']] for x in refs.values()}))
    sources.update({str(REPORT):REPORT_SHA,str(RECEIPT):RECEIPT_SHA,str(Path(r['prepared'])):PREPARE_SHA})
    for path in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),
                 ROOT/'innerdocs'/(NAME+'_protocol.md')):sources[str(path)]=sha(path)
    need(all(sha(f)==h for f,h in sources.items()),'all source/model/input bytes')
    return dict(schema=NAME+'.prepare',protocol=protocol(),slots=slots,sources=sources,
        upstream=dict(report=str(REPORT),report_sha256=REPORT_SHA,receipt=str(RECEIPT),receipt_sha256=RECEIPT_SHA,
            prepared=r['prepared'],prepare_sha256=PREPARE_SHA,trust='complete original receipt + saved closure, not rerun fullstrict'),
        policy=p['policy'],expected_model_hash=p['expected_model_hash'],
        inherited=dict(CEM=receipt['total_resources'],encoding=p['inherited']))


def own(path):
    p=Path(path).absolute();need(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and
        not any(x.is_symlink() for x in (p,*p.parents)),'exclusive own results direct child');return p


def prepare(output):
    p=own(output);v=build();b=raw(v);need(len(b)<=2*1024**2 and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'prepare bounds')
    durable(p,b);return dict(path=str(p),sha256=sha(p))


def verify(prepared,h):
    p=read(prepared,h);need(p==build(),'complete new saved prepare closure');return p


class Store(old.Store):
    """Reuse frozen codecs, but all namespace/budgets are new instance-owned guards."""
    def __init__(self,path,started=None):
        self.path=own(path);self.started=time.monotonic() if started is None else started
        need(not self.path.exists() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'exclusive512MiB+floor')
        self.path.mkdir();self.inventory={};self.events=[];self.total=0;self.tail=None;self.extra_guard=lambda:None
    def guard(self,size=0,final=False):
        need(final or time.monotonic()-self.started<WALL,'cooperative900s')
        need(self.total+size<=CAP-(0 if final else RESERVE) and shutil.disk_usage(self.path).free>=FLOOR+size,'512MiB/64reserve/200floor')
        if not final:self.extra_guard()


def schedule():
    def ae(n):return ('action_encoder',[(1,n,10)],(1,n,192))
    pr=('predictor',[(1,3,192),(1,3,192)],(1,3,192))
    proj=('pred_proj',[(3,192)],(3,192))
    return [ae(4),pr,proj,pr,proj,ae(3),pr,proj,ae(3),pr,proj]


class Trace:
    def __init__(self,model,store,row):self.model,self.store,self.row=model,store,row;self.handles=[];self.pending={}
    def __enter__(self):
        for kind in (*KINDS,'encoder','projector'):
            def pre(module,args,k=kind):
                need(k in KINDS,'no encoding/projector allowed')
                import torch
                self.store.guard();index=len(self.row['modules']);need(index<len(schedule()),'no extra native module')
                expected,shapes,_=schedule()[index]
                need(k==expected and [tuple(x.shape) for x in args]==shapes and not self.pending,'native exact schedule/input shapes')
                need(torch.is_inference_mode_enabled() and not torch.is_grad_enabled() and not torch.is_autocast_enabled('cuda') and
                    not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,'actual numerical guards')
                need(all(x.dtype==torch.float32 and x.device.type=='cuda' and not x.requires_grad for x in args),'CUDA float32 only')
                e=dict(index=index,kind=k,layout=self.row['active_layout'],returned=False,inputs=[]);self.row['modules'].append(e)
                self.store.event('module_intent',root=self.row['root'],index=index,kind_name=k)
                self.pending[k]=e;e['inputs']=[self.store.array(x) for x in args]
            def post(module,args,out,k=kind):
                import torch
                e=self.pending[k];need(tuple(out.shape)==schedule()[e['index']][2] and out.device.type=='cuda' and
                    out.dtype==torch.float32 and not out.requires_grad,'native full output shape/device/dtype')
                e['output']=self.store.array(out);e['returned']=True;self.pending.pop(k)
                self.store.event('module_return',root=self.row['root'],index=e['index'],kind_name=k)
            self.handles.extend([getattr(self.model,kind).register_forward_pre_hook(pre),getattr(self.model,kind).register_forward_hook(post)])
        return self
    def __exit__(self,*args):
        for handle in self.handles:handle.remove()
        if args[0] is None:need(not self.pending and len(self.row['modules'])==11,'complete original fixed layouts')


class NoNeural:
    """The scoring phase may use criterion tensor algebra, not any NN module."""
    def __init__(self,model):self.model=model;self.handles=[]
    def __enter__(self):
        def reject(module,args):need(False,'no neural module during criterion scoring')
        for k in (*KINDS,'encoder','projector'):
            self.handles.append(getattr(self.model,k).register_forward_pre_hook(reject))
        return self
    def __exit__(self,*args):
        for handle in self.handles:handle.remove()


def run(prepared,h,output,authorization):
    need(authorization==AUTH and not os.environ.get('PYTHONPATH'),'MAIN CUDA-only authority/no path overlay')
    started=time.monotonic();p=verify(prepared,h);store=Store(output,started)
    rows=[dict(root=i,status='not_started',modules=[],layouts={k:dict(attempted=False,returned=False) for k in LAYOUTS},
        scores={k:dict(attempted=False,returned=False) for k in LAYOUTS}) for i in (0,1)]
    r=dict(schema=NAME+'.report',prepared=str(Path(prepared).absolute()),prepare_sha256=h,protocol=protocol(),rows=rows,
        inherited=p['inherited'],upstream=p['upstream'],model_before=None,model_after=None,source_verified=False,
        all_predictions_sealed=False,errors=[],physical_release=False,scientific_admission=False)
    model=None
    try:
        store.event('started',prepare_sha256=h)
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        need(os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8','fixed requested CUBLAS')
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        r['cuda_preflight']=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        need(Path(inspect.getfile(type(model))).resolve()==old.WM,'actual official model class')
        r['model_before']=old.inputs.moving.low.tensor_hash(model);need(r['model_before']==p['expected_model_hash'],'frozen original model')
        store.extra_guard=lambda:need(torch.cuda.memory_allocated()<=3*1024**3 and torch.cuda.memory_reserved()<=4*1024**3,'GPU3/4GiB')
        outputs=[]
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            for slot,row in zip(p['slots'],rows,strict=True):
                arrays={k:old.load_array(Path(p['upstream']['report']).parent,v,slot['inventory']) for k,v in slot['refs'].items()}
                row['inputs']={k:store.array(a) for k,a in arrays.items()};store.event('delivery_sealed',root=row['root'])
                tensor=lambda x:torch.as_tensor(x,device=device)
                emb=tensor(arrays['emb']);warm=tensor(arrays['warm']);mean=tensor(arrays['actions'])
                actions=torch.cat((warm.expand(1,1,2,10),mean[:,None]),dim=2)
                with Trace(model,store,row):
                    for layout in LAYOUTS:
                        row['active_layout']=layout;e=row['layouts'][layout];e['attempted']=True
                        store.guard();store.event('layout_intent',root=row['root'],layout=layout)
                        if layout=='whole':
                            info=dict(pixels=tensor(arrays['pixels'])[:,None],emb=emb[:,None])
                            actual=model.rollout(info,actions)
                            need(actual['emb'] is info['emb'] and tuple(actual['predicted_emb'].shape)==(1,1,5,192),'whole cached official rollout')
                            pred=actual['predicted_emb'];e['action_prefix']=store.array(actual['action'])
                        else:
                            ae1=model.action_encoder(actions[:,0,:3])
                            full1=model.predict(emb,ae1);first=full1[:,-1:]
                            shifted=torch.cat((emb[:,1:],first),dim=1)
                            ae2=model.action_encoder(actions[:,0,1:])
                            full2=model.predict(shifted,ae2)
                            pred=torch.cat((emb,first,full2[:,-1:]),dim=1)[:,None]
                            e['full3']=[store.array(full1),store.array(full2)]
                        e['predicted_emb']=store.array(pred);e['returned']=True
                        outputs.append((row,layout,pred,arrays['goal_emb']))
                        store.event('layout_return',root=row['root'],layout=layout)
                del row['active_layout'];row['status']='predictions_sealed';store.event('root_sealed',root=row['root'])
            need(len(outputs)==4 and all(x['status']=='predictions_sealed' for x in rows),'all four predictions before scoring')
            r['all_predictions_sealed']=True;store.event('all_predictions_sealed')
            with NoNeural(model):
                for row,layout,pred,goal in outputs:
                    store.guard();e=dict(attempted=True,returned=False);row['scores'][layout]=e
                    store.event('criterion_intent',root=row['root'],layout=layout)
                    # No extra rollout or predict: criterion only, original cached goal.
                    cost=model.criterion(dict(predicted_emb=pred,goal_emb=tensor(goal)[:,None]))
                    need(tuple(cost.shape)==(1,1) and cost.dtype==torch.float32 and cost.device.type=='cuda' and
                        not cost.requires_grad,'official CUDA float32 scalar criterion')
                    e['cost']=store.array(cost);e['returned']=True;store.event('criterion_return',root=row['root'],layout=layout)
            for row in rows:row['status']='paid_finalmean_evaluated'
        r['model_after']=old.inputs.moving.low.tensor_hash(model);need(r['model_after']==r['model_before'],'unchanged original model')
        need(all(sha(f)==v for f,v in p['sources'].items()),'source/input/model files unchanged');r['source_verified']=True
        r['cuda_peak_allocated']=torch.cuda.max_memory_allocated();r['cuda_peak_reserved']=torch.cuda.max_memory_reserved()
        need(r['cuda_peak_allocated']<=3*1024**3 and r['cuda_peak_reserved']<=4*1024**3,'whole peak GPU3/4GiB')
        r['numerics']=dict(requested_SDP='MATH',kernel_certificate=False,observed_inference=True,observed_autocast=False,observed_TF32=False)
    except BaseException as ex:r['errors'].append(dict(type=type(ex).__name__,message=str(ex)))
    finally:
        if model is not None and r['model_after'] is None:
            try:r['model_after']=old.inputs.moving.low.tensor_hash(model)
            except BaseException as ex:r['errors'].append(dict(type=type(ex).__name__,message=str(ex)))
        r['elapsed_seconds']=time.monotonic()-started
        if r['elapsed_seconds']>=WALL:r['errors'].append(dict(type='wall_cap',message='whole900s'))
        r['status']='known2_paid_finalmean_evaluated' if not r['errors'] and r['source_verified'] and r['all_predictions_sealed'] else 'partial_or_failed'
        store.event('finished',final=True,status=r['status']);journal=store.path/'journal.jsonl'
        store.inventory['journal.jsonl']=dict(sha256=sha(journal),file_bytes=journal.stat().st_size)
        r.update(artifacts=dict(store.inventory),journal_tail=store.tail);store.put('report.json',raw(r),final=True)
    return dict(status=r['status'],path=str(store.path/'report.json'),sha256=sha(store.path/'report.json'))


def compare(a,b):
    need(a.shape==b.shape and a.dtype==b.dtype==np.float32 and a.size>0 and np.isfinite(a).all() and np.isfinite(b).all(),'finite same-shape float32 comparisons')
    def ordered(x):
        u=x.view(np.uint32).astype(np.int64)
        return np.where(u>=0x80000000,0x100000000-u,u+0x80000000)
    bits=a.view(np.uint32)!=b.view(np.uint32)
    return dict(bitwise=exact(a,b),maxabs=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))),
        max_ULP=int(np.max(np.abs(ordered(a)-ordered(b)))),bitwise_different_elements=int(bits.sum()),
        signed_zero_bit_differences=int((bits&(a==0)&(b==0)).sum()),elements=a.size,parity_upgraded=False)


def validate_row(slot,row,load,source):
    need(row['root']==slot['root'] and row['status']=='paid_finalmean_evaluated' and set(row['layouts'])==set(row['scores'])==set(LAYOUTS),'full root/layout/score slots')
    inputs={k:load(v) for k,v in row['inputs'].items()}
    need(set(inputs)==set(slot['refs']) and all(exact(inputs[k],source(slot['refs'][k])) for k in inputs),'delivered originals unchanged')
    emb=inputs['emb'];actions=np.concatenate((inputs['warm'],inputs['actions'][:,None]),axis=2)
    need(emb.shape==(1,3,192) and actions.shape==(1,1,4,10) and inputs['pixels'].shape==(1,3,3,224,224),'original cached shapes')
    modules=row['modules'];need(len(modules)==11,'all11 native dispatches per root')
    values=[]
    for i,(e,(kind,shapes,outshape)) in enumerate(zip(modules,schedule(),strict=True)):
        args=[load(x) for x in e['inputs']];out=load(e['output'])
        need(e['index']==i and e['kind']==kind and e['layout']==('whole' if i<5 else 'segmented') and e['returned'] is True and
            [x.shape for x in args]==shapes and out.shape==outshape and all(x.dtype==np.float32 for x in [*args,out]),'all native sequence/shapes/dtypes')
        values.append((args,out))
    results={}
    for layout in LAYOUTS:
        e=row['layouts'][layout];need(e['attempted'] is e['returned'] is True,'actual layouts returned')
        pred=load(e['predicted_emb']);need(pred.dtype==np.float32 and pred.shape==(1,1,5,192) and exact(pred[:,0,:3],emb),'all predicted frames/source context')
        if layout=='whole':
            need(exact(load(e['action_prefix']),actions[:,:,:3]) and exact(values[0][0][0],actions[:,0]),'whole actual four action tokens')
            ae=values[0][1]
            for step in (0,1):
                pi,po=values[1+2*step];qi,qo=values[2+2*step]
                need(exact(pi[0],pred[:,0,step:step+3]) and exact(pi[1],ae[:,step:step+3]) and
                    exact(qi[0],po.reshape(3,192)) and exact(qo.reshape(1,3,192)[:,-1],pred[:,0,3+step]),'whole causal native full3 chain')
        else:
            need(exact(values[5][0][0],actions[:,0,:3]) and exact(values[8][0][0],actions[:,0,1:]),'segmented actual three-token windows')
            need(len(e['full3'])==2,'both segmented full3 snapshots')
            for step,start in ((0,5),(1,8)):
                pi,po=values[start+1];qi,qo=values[start+2];full=load(e['full3'][step])
                need(exact(pi[0],pred[:,0,step:step+3]) and exact(pi[1],values[start][1]) and
                    exact(qi[0],po.reshape(3,192)) and exact(full,qo.reshape(1,3,192)) and exact(full[:,-1],pred[:,0,3+step]),'segmented own-causal full3 chain')
        s=row['scores'][layout];need(s['attempted'] is s['returned'] is True,'actual criterion returned')
        cost=load(s['cost']);residual=pred[:,:,-1].astype(np.float64)-inputs['goal_emb'][:,0,None].astype(np.float64)
        total=np.sum(residual**2,axis=-1,dtype=np.float64)
        need(cost.shape==(1,1) and cost.dtype==np.float32,'original criterion scalar dtype')
        old.bounded_reduction(cost,total,194)
        results[layout]=dict(sum192_float32=cost.tolist(),sum192_float64=total.tolist(),mean192_float64=(total/192).tolist())
    whole=load(row['layouts']['whole']['predicted_emb']);seg=load(row['layouts']['segmented']['predicted_emb'])
    return dict(root=row['root'],scores=results,comparisons=dict(all_future=compare(whole[:,:,3:],seg[:,:,3:]),
        frame15=compare(whole[:,:,3],seg[:,:,3]),final20=compare(whole[:,:,4],seg[:,:,4])),returned_mean_evaluated_in_new_job=True,
        old_slice_mean_evaluated=False,raw_actions=None,physical_release=False)


def expected_events(h,status):
    events=[dict(kind='started',prepare_sha256=h)]
    for root in (0,1):
        events.append(dict(kind='delivery_sealed',root=root))
        for layout,indices in (('whole',range(5)),('segmented',range(5,11))):
            events.append(dict(kind='layout_intent',root=root,layout=layout))
            for i in indices:
                for kind in ('module_intent','module_return'):
                    events.append(dict(kind=kind,root=root,index=i,kind_name=schedule()[i][0]))
            events.append(dict(kind='layout_return',root=root,layout=layout))
        events.append(dict(kind='root_sealed',root=root))
    events.append(dict(kind='all_predictions_sealed'))
    for root in (0,1):
        for layout in LAYOUTS:
            events.extend([dict(kind=k,root=root,layout=layout) for k in ('criterion_intent','criterion_return')])
    events.append(dict(kind='finished',status=status));return events


def consume(output,h):
    folder=own(output);r=read(folder/'report.json',h);p=verify(r['prepared'],r['prepare_sha256'])
    need(r['schema']==NAME+'.report' and r['protocol']==protocol() and r['status']=='known2_paid_finalmean_evaluated' and r['errors']==[] and
        r['all_predictions_sealed'] is r['source_verified'] is True and r['model_before']==r['model_after']==p['expected_model_hash'] and
        r['inherited']==p['inherited'] and r['upstream']==p['upstream'] and r['physical_release'] is r['scientific_admission'] is False and
        len(r['rows'])==2,'complete new paid job only')
    need(r['elapsed_seconds']<WALL and r['cuda_peak_allocated']<=3*1024**3 and r['cuda_peak_reserved']<=4*1024**3 and
        r['numerics']==dict(requested_SDP='MATH',kernel_certificate=False,observed_inference=True,observed_autocast=False,observed_TF32=False),'saved engineering/numerical guards')
    inv=r['artifacts'];need(set(inv)|{'report.json'}=={x.name for x in folder.iterdir()},'whole file census')
    need(sum(x['file_bytes'] for x in inv.values())+(folder/'report.json').stat().st_size<=CAP,'all output bytes')
    for name,v in inv.items():need(Path(name).name==name and not (folder/name).is_symlink() and
        (folder/name).stat().st_size==v['file_bytes'] and sha(folder/name)==v['sha256'],'every artifact bytes/SHA')
    tail=None;actual=[]
    with (folder/'journal.jsonl').open('rb') as f:
        for line in f:
            need(len(line)<=1024**2,'bounded journal');e=json.loads(line);eventsha=e.pop('event_sha256')
            need(e['sequence']==len(actual) and e['previous_sha256']==tail and hashlib.sha256(raw(e)).hexdigest()==eventsha,'durable hashchain')
            e['event_sha256']=eventsha;need(raw(e)==line,'exactly one canonical newline');tail=eventsha
            actual.append({k:v for k,v in e.items() if k not in ('sequence','previous_sha256','event_sha256')})
    need(actual==expected_events(r['prepare_sha256'],r['status']) and tail==r['journal_tail'],'all native events and seal before criterion')
    used=set()
    def load(rec):used.add(rec['file']);return old.load_array(folder,rec,inv)
    results=[]
    for slot,row in zip(p['slots'],r['rows'],strict=True):
        source=lambda rec:old.load_array(Path(p['upstream']['report']).parent,rec,slot['inventory'])
        results.append(validate_row(slot,row,load,source))
    need(used==set(inv)-{'journal.jsonl'},'all native arrays referenced')
    return dict(schema=NAME+'.consume',status='known2_paid_finalmean_saved_verified',report_sha256=h,rows=results,
        resources=protocol()['resources'],inherited=r['inherited'],upstream=r['upstream'],parity_upgraded=False,
        old_slice_mean_evaluated=False,physical_release=False,scientific_admission=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True)
    for cmd in ('verify','run'):
        a=sub.add_parser(cmd);a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True)
        if cmd=='run':a.add_argument('--output',required=True);a.add_argument('--authorization',required=True)
    a=sub.add_parser('consume');a.add_argument('--output',required=True);a.add_argument('--report-sha',required=True);a.add_argument('--receipt',required=True)
    args=parser.parse_args()
    if args.command=='prepare':v=prepare(args.output)
    elif args.command=='verify':verify(args.prepared,args.prepare_sha);v=dict(status='paid_finalmean_prepare_verified')
    elif args.command=='run':v=run(args.prepared,args.prepare_sha,args.output,args.authorization)
    else:
        v=consume(args.output,args.report_sha);b=raw(v);need(len(b)<=RESERVE,'receipt bound');p=own(args.receipt)
        need(shutil.disk_usage(p.parent).free>=FLOOR+len(b),'receipt floor');durable(p,b);v=dict(status=v['status'],path=str(p),sha256=sha(p))
    print(raw(v).decode(),end='')
    if v.get('status')=='partial_or_failed':raise SystemExit(2)


if __name__=='__main__':main()
