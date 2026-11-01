"""E1 fixed six-slot RGB roster and saved prediction arithmetic; no model execution."""
from copy import deepcopy
import argparse
import hashlib
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_e1_six_slot_physical_20261029_v1 as native
from experiments.spectral_reachability import agent_p2_six_target_rgb_20261029_v1 as shared

NAME='agent_e1_six_slot_rgb_20261029_v1'
CP=(0,5,10)
need=native.require
ROOT=native.ROOT
bridge=shared.bridge
roster=shared.roster
read=shared.read
sha=native.old.sha
raw=native.old.raw
REPORT_SHA='77cd42de14e1c7d75f8e7b5b4b3d9ad58fb87e94a77c209333ec8a5fc739c5ce'
RECEIPT_SHA='6644156f48a79a30f5118d9c56c3c3a06c3bc818e44edb55e6fe8bf4c29d38d4'
NATIVE_SHA='47e238e9d5188a5495476c429ac5bbadf120338cec6f5565360a2e177d339a19'
SHARED_SHA='5a3ab9162f527ba6e51595b53cff095e095fbbcb596e8567eb29eac88bc2168e'
CAP,FINAL,FLOOR,WALL,META=shared.CAP,shared.FINAL,shared.FLOOR,shared.WALL,shared.META
ALLOC,RESERVED=shared.ALLOC,shared.RESERVED
BOUNDARY=shared.BOUNDARY


def vec(value):
    a=np.asarray(value,dtype=np.float32)
    need(a.shape==(192,) and np.isfinite(a).all(),'finite 192 float32 vector')
    return a


def mse(x,y):
    a,b=vec(x).astype(np.float64),vec(y).astype(np.float64)
    return math.fsum((float(i)-float(j))**2 for i,j in zip(a,b,strict=True))/192


def assemble(plans,rows,tasks,replays):
    """Pure ABI assembly; authorities MUST be authenticated by future caller."""
    need(len(plans)==len(rows)==len(tasks)==len(replays)==6,'fixed six slots')
    out=[]
    for i,(root,arm) in enumerate(native.producer.core.SLOTS):
        plan,row,task,replay=plans[i],rows[i],tasks[i],replays[i]
        need((plan['root_slot'],plan['arm'])==(row['root_slot'],row['arm'])==(root,arm)
             and row['branch_index']==i and row['selected_plan']==plan,'exact slot/plan join')
        need(replay['raw']==plan['commands'] and len(replay['whole'])==len(replay['segmented'])==2,'paid raw10 and both replay paths')
        need(len(plan['commands'])==10,'ten raw actions')
        goal=vec(task['goal']);target=vec(plan['target']);rootz=vec(task['root_context']['latent_history'][-1])
        points={cp['step']:cp for cp in row['checkpoints']}
        need(len(points)==len(row['checkpoints']) and set(points)<=set(CP),'unique fixed checkpoints')
        for j,step in enumerate(CP):
            cp=points.get(step)
            if cp is not None:need(cp['phase']=='plan' and cp['step']==step and cp['state']==cp['observation_state']==cp['snapshot']['state'],'checkpoint state')
            pred=rootz if j==0 else vec(replay['whole'][j-1]['latent_history'][-1])
            seg=rootz if j==0 else vec(replay['segmented'][j-1]['latent_history'][-1])
            out.append(dict(index=3*i+j,slot=i,root=root,arm=arm,step=step,
                checkpoint=deepcopy(cp),goal192=goal.tolist(),target192=target.tolist(),
                predicted192=pred.tolist(),segmented192=seg.tolist(),selection_digest=plan['selection_digest'],
                whole_context=None if j==0 else deepcopy(replay['whole'][j-1]),
                segmented_context=None if j==0 else deepcopy(replay['segmented'][j-1]),
                task_id=plan['task_id'],commands=deepcopy(plan['commands'])))
    for root in (0,1):
        reference=vec(out[root*9]['goal192']).tobytes()
        need(all(vec(r['goal192']).tobytes()==reference for r in out[root*9:root*9+9]),'same task goal for root')
    return out


def panel(roster,actual):
    """Pure saved arithmetic; unavailable images stay null, with no selection."""
    need(len(roster)==len(actual)==18,'fixed eighteen cells')
    out=[]
    for r,z in zip(roster,actual,strict=True):
        p,g,t=r['predicted192'],r['goal192'],r['target192']
        need((z is None)==(r['checkpoint'] is None),'missing image preserved')
        if z is not None and r['step']==0:
            need(vec(z).tobytes()==vec(p).tobytes(),'cp0 native encoder exact source root latent')
        out.append(dict(index=r['index'],slot=r['slot'],root=r['root'],arm=r['arm'],step=r['step'],
            model_tracking_Q=mse(p,t),model_task_Q=mse(p,g),
            actual_tracking_Q=None if z is None else mse(z,t),actual_task_Q=None if z is None else mse(z,g),
            prediction_error=None if z is None else mse(p,z),
            whole_segmented_prediction_error=mse(p,r['segmented192']),
            whole_segmented_context_bits=None if r['step']==0 else {
                key:native.producer.f32_diff(r['whole_context'][key],r['segmented_context'][key])
                for key in ('latent_history','action_history')},
            checkpoint=deepcopy(r['checkpoint']),missing=z is None))
    return dict(rows=out,fixed_slots=6,fixed_cells=18,winner=None,FLOPs=None,scientific_admission=False)


def contract():
    return dict(slots=6,cells=18,checkpoints=list(CP),known_roots=2,fresh=False,
        CUDA_only=True,batch=1,no_dedup=True,no_padding=True,retries=0,hidden=[1,257,192],latent=192,
        max_encoder_API=18,max_projector_API=18,new_predict=0,new_AD=0,new_physics=0,new_goal_encode=0,
        training=False,prediction_source='paid raw10 whole replay; segmented separately retained',
        wall_seconds=WALL,cap_bytes=CAP,final_reserve=FINAL,disk_floor=FLOOR,
        allocated_cap=ALLOC,reserved_cap=RESERVED,wall_boundary=BOUNDARY,
        scientific_admission=False,P1_admission=False,FLOPs=None,equal_FLOPs=None)


def build(native_output,native_report_sha,native_receipt,native_receipt_sha):
    output=Path(native_output).resolve();receipt=Path(native_receipt).resolve()
    need(native_report_sha==REPORT_SHA and native_receipt_sha==RECEIPT_SHA and
         sha(native.__file__)==NATIVE_SHA and sha(shared.__file__)==SHARED_SHA,'fixed authority and source')
    report=native.old.read(output/'report.json',META)
    need(sha(output/'report.json')==REPORT_SHA and sha(receipt)==RECEIPT_SHA,'actual physical SHA')
    external=native.old.read(receipt,META)
    verified=native.consume(output,REPORT_SHA)
    need(verified==external and verified['status']=='six_slot_native_recorded','complete physical receipt rebuilt')
    p=native.verify(report['prepared'],report['prepare_sha256'])
    producer=native.producer
    controller=native.old.read(native.RECEIPT,META)
    need(sha(native.RECEIPT)==native.RECEIPT_SHA and controller['report_sha256']==native.REPORT_SHA and
         controller['status']=='all_six_saved_controller_and_replays_verified','actual complete controller receipt')
    upstream=producer.parse_canonical(producer.read_bound(native.OUTPUT/'report.json',native.REPORT_SHA,producer.RESERVE))
    payload,_=producer.read_store(native.OUTPUT,upstream)
    need(upstream['model_hash_before']==upstream['model_hash_after']==bridge.native.MODEL_TENSOR_SHA,'controller encoder model match')
    tasks=[];replays=[]
    for i,plan in enumerate(p['plans']):
        loader=producer.core.old.sampling.old
        task=loader.load_export(plan['task_export'],plan['task_sha256'])['gn_task']
        need(producer.core.old.core.task_id(task)==plan['task_id'],'original task identity')
        export=loader.read(plan['task_export'],2*1024**2)
        need(export['source_slot']==plan['root_slot'] and export['task_id']==plan['task_id'] and
             native.old.source.digest(export['task_contract'])==plan['task_contract_sha256'],'original task contract identity')
        root_cp=next((x for x in verified['rows'][i]['checkpoints'] if x['step']==0),None)
        need(root_cp is not None and root_cp['pixel_sha256']==export['task_contract']['pixel_sha256'][2],
             'native cp0 same original input pixels')
        tasks.append(task);replays.append(payload[f'slot_{i}.replay.z'])
    rows=assemble(p['plans'],verified['rows'],tasks,replays)
    for row in rows:
        cp=row['checkpoint'];row.update(status='missing' if cp is None else 'pending',latent=None,pixels=None,preprocessed=None,modules=[],pixel_source=None)
        if cp is not None:
            need(Path(cp['png']).name==cp['png'] and not (output/cp['png']).is_symlink(),'flat original frame')
            row['pixel_source']=dict(cp,path=str(output/cp['png']))
            bridge.load_pixels(row['pixel_source'])
    bindings=deepcopy(p['bindings'])
    def bind(path,expected=None):
        path=str(Path(path).resolve());h=sha(path)
        need((expected is None or h==expected) and (path not in bindings or bindings[path]==h),'source closure conflict')
        bindings[path]=h
    for path in shared.source_files()+[Path(native.__file__),Path(producer.__file__),Path(__file__),
            ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]:bind(path)
    for path,h in bridge.rgb.MODELS.items():bind(path,h)
    for path,h in [(receipt,RECEIPT_SHA),(output/'report.json',REPORT_SHA),(report['prepared'],report['prepare_sha256'])]:bind(path,h)
    for row in rows:
        if row['pixel_source'] is not None:bind(row['pixel_source']['path'],row['checkpoint']['file_sha256'])
    bridge.verify_bindings(bindings)
    n=sum(row['pixel_source'] is not None for row in rows)
    return dict(schema=NAME+'.prepare',arguments=dict(native_output=str(output),native_report_sha=REPORT_SHA,
        native_receipt=str(receipt),native_receipt_sha=RECEIPT_SHA),contract=contract(),rows=rows,bindings=bindings,
        expected_calls={k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS},
        policy=str(bridge.rgb.POLICY),expected_model_hash=bridge.native.MODEL_TENSOR_SHA,
        imported_native_receipt=external,imported_controller_receipt=controller,controller_report_sha256=native.REPORT_SHA)


def verify(path,digest):
    p=shared.read(path,digest);need(p==build(**p['arguments']) and sha(path)==digest,'full prepare reconstruction');return p


def write_metadata(path,value):
    path=Path(path).absolute();b=raw(value)
    need(path.name.startswith(NAME+'_') and not path.is_symlink() and len(b)<=META and
         shutil.disk_usage(path.parent).free>=FLOOR+len(b),'exclusive bounded metadata')
    with path.open('xb') as f:need(f.write(b)==len(b),'complete write');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(path.parent);return dict(path=str(path),sha256=sha(path))


class Store(bridge.rgb.Store):
    def __init__(self,path,budget):
        need(Path(path).name.startswith(NAME+'_'),'exclusive own output')
        super().__init__(path,budget);self.index=0;self.order=[]
    def write(self,name,data,final=False):
        h=super().write(name,data,final=final);self.order.append(name);return h
    def event(self,e):self.put(f'event_{self.index:04d}.json',e);self.index+=1


def process(p,rows,store,encode_one):
    need(rows==p['rows'] and len(rows)==18,'full sealed initial roster')
    for row in rows:
        if row['pixel_source'] is None:continue
        store.check();scope=dict(row=row['index'],step=row['step'],phase='encode')
        row['status']='attempted';store.event(dict(kind='image_attempt',scope=scope))
        pixels=bridge.load_pixels(row['pixel_source']);row['pixels']=store.array(f'r{row["index"]}_pixels',pixels)
        z=encode_one(pixels,row,scope);shared.roster.f32(z,(192,))
        row['latent']=store.array(f'r{row["index"]}_latent',z);row['status']='encoded'
        store.event(dict(kind='image_return',scope=scope,row=deepcopy(row)))


def accounting(p,rows,modules):
    calls=bridge.counts(modules)
    return dict(module_calls=calls,encoder_samples=calls['encoder']['completed'],projector_samples=calls['projector']['completed'],
        encoded_images=sum(x['status']=='encoded' for x in rows),fixed_positions=18,
        missing_positions=sum(x['pixel_source'] is None for x in p['rows']),
        persisted_tensor_captures=sum(len(x.get('inputs',[]))+int('output' in x) for x in modules)+
            sum(x.get('preprocessed') is not None for x in rows)+sum(x.get('latent') is not None for x in rows),
        all_cpu_transfer_calls=None,FLOPs=None,new_predict=0,new_physics=0,new_AD=0)


def run(prepared,mh,output,*,main_authorized=False):
    need(main_authorized is True,'MAIN authorization before model/preflight/verify')
    budget=shared.Budget();p=verify(prepared,mh);budget.check();store=Store(output,budget)
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
        trace=shared.Trace(model,torch,store,p['expected_calls'],check);transform=p0.image_transform()
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



def validate(p,r,events,load):
    need(r['schema']==NAME+'.report' and r['contract']==p['contract']==contract() and r['completed'] is True and r['errors']==[], 'complete new six RGB report')
    need(r['model_before']==r['model_after']==p['expected_model_hash'],'model strongly unchanged')
    need(r['host']['device']=='cuda:0' and 0<r['gpu_peaks']['allocated']<=ALLOC and 0<r['gpu_peaks']['reserved']<=RESERVED and
         r['gpu_peaks']['guard_checks']>0,'actual bounded CUDA peaks, no CPU fallback')
    need(len(p['rows'])==len(r['rows'])==18,'all18 saved denominator')
    modules=[];want=[]
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        mutable={'status','latent','pixels','preprocessed','modules'}
        need({k:v for k,v in spec.items() if k not in mutable}=={k:v for k,v in row.items() if k not in mutable},'exact sealed input roster')
        if spec['pixel_source'] is None:need(row==spec,'missing preserved exactly');continue
        need(row['status']=='encoded','all available frames encoded exactly once')
        pix=load(row['pixels']);bridge.same(pix,bridge.load_pixels(spec['pixel_source']),'native canonical pixels')
        pre=load(row['preprocessed']);bridge.same(pre,bridge.pixels.preprocess_pixels(pix),'preprocessing exact')
        mods=row['modules'];scope=dict(row=row['index'],step=row['step'],phase='encode')
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
    return panel(p['rows'],[None if row['latent'] is None else load(row['latent']) for row in r['rows']])



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
    return dict(schema=NAME+'.consume',status='all18_saved_RGB_positions_verified_science_blocked',report_sha256=rh,
        prepare_sha256=r['prepare_sha256'],completion_sha256=sha(folder/'completion.json'),**out,counts=r['counts'],
        inherited_native_counts=p['imported_native_receipt']['counts'],controller_report_sha256=p['controller_report_sha256'],
        inherited_controller_receipt=p['imported_controller_receipt'],
        bindings=p['bindings'],all_cpu_transfer_calls=None,
        resources=r['resources'],gpu_peaks=r['gpu_peaks'],whole_wall_through_report_fsync_hash=done['wall_inclusive'],
        new_predict=0,new_physics=0,new_AD=0,equal_generation_budget=False,known_roots=2,fresh=False,prior_physical_time=None)


def main(argv=None):
    q=argparse.ArgumentParser();q.add_argument('command',choices=('prepare','verify','run','consume','protocol'))
    for k in ('native-output','native-receipt','prepared','output','rgb-output'):q.add_argument('--'+k,type=Path)
    for k in ('native-report-sha','native-receipt-sha','prepare-sha','report-sha'):q.add_argument('--'+k)
    q.add_argument('--main-authorized',action='store_true');a=q.parse_args(argv)
    if a.command=='prepare':v=write_metadata(a.output,build(a.native_output,a.native_report_sha,a.native_receipt,a.native_receipt_sha))
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);v={'verified':True}
    elif a.command=='run':v=run(a.prepared,a.prepare_sha,a.output,main_authorized=a.main_authorized)
    elif a.command=='consume':v=write_metadata(a.output,consume(a.rgb_output,a.report_sha))
    else:v=contract()
    print(raw(v).decode())


if __name__=='__main__':main()

