"""Own-six saved raw authority and MAIN-only actual deployment-token CUDA replay."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
from experiments.spectral_reachability import agent_p2_six_target_controller_cuda_20261029_v1 as six
from experiments.spectral_reachability import agent_p2_delivered_mean_paid_replay_cuda_20261029_v1 as paid

TAG='agent_p2_six_target_paid_replay_20261029_v1'
ROOT=six.ROOT
tracker,core,low,raw=six.tracker,six.core,paid.low,paid.raw
need,sha,read,encode=six.need,six.sha,six.read,six.encode
ACTIVE=tuple(range(6))
CAP,RESERVE,ROOT_CAP,FLOOR,WALL,SLOT_WALL=768*1024**2,96*1024**2,384*1024**2,200*1024**2,1350,180
BOUNDARY='Store start before domain verification through report fsync/hash; completion acknowledgement excluded'
REFS={
 'prepare':('agent_p2_six_target_controller_cuda_20261029_v1_prepare_MAIN01.json','cccfadc20911dc1be008a047c57bca7400272f6602deb435fb0341a0c8bbc8f9'),
 'report':('agent_p2_six_target_controller_cuda_20261029_v1_cuda_MAIN01/report.json','b81f12ee3602835bc14fd2e22d7e3644b8376f7ba8b0b265948bfb18429b6934'),
 'receipt':('agent_p2_six_target_controller_cuda_20261029_v1_consume_MAIN01.json','592062bded7d6eb1716594b21fac4b5ef6255cf93d5285c2361f0ec2686c0103')}
PROTOCOLS={
 'agent_p2_six_target_deployment_protocol_20261029_v1.md':'97d81f6969e0ac52e808893be09dfacf0180ed04ab0955fe38d202563a32744f',
 'agent_p2_six_target_deployment_protocol_sources_20261029_v1.md':'4c114bf5fa6bcbf0fcbcff7177bccc0379609db8fc11979747be4cf5d4e6a8dc',
 'agent_p2_six_target_deployment_protocol_authority_20261029_v1.md':'1e39e899515de8acade1e26353ab2f081265ddf76019803a0a3732f3f7e2b98a'}


def refs():return {k:dict(path=str(ROOT/'results'/p),sha256=h) for k,(p,h) in REFS.items()}


def protocol():
    return dict(schema=TAG+'.protocol',protocol_sha256=PROTOCOLS,active=list(ACTIVE),missing=[],denominator=6,
        H=3,S=1,segments=[1,2],search=False,reselection=False,clip=False,tanh=False,
        arithmetic=['mul_float32','add_float32','sub_float32','div_float32'],
        planned=dict(predict_API=36,predictor_samples=36,action_encoder=36,low_projection=36,
                     AE_tokens=144,predictor_context_tokens=108,projection_rows=108),
        encoder=0,high=0,AD_primal=0,J=0,VJP=0,backward=0,physics=0,FLOPs=None,
        output_cap=CAP,reserve=RESERVE,per_root_cap=ROOT_CAP,floor=FLOOR,whole_wall=WALL,per_slot_wall=SLOT_WALL,
        trace_payload_cap=64*1024**2,cuda_allocated=3*1024**3,cuda_reserved=4*1024**3,
        raw_deployment_authorized=False,scientific_admission=False,original_Hi_scaler_certified=False,
        known_roots=2,fresh_roots=0,prior_physical_time=None,whole_wall_boundary=BOUNDARY,
        nested_trace_schema=paid.TAG+'.trace',nested_trace_reuse='unchanged low-level arithmetic/rollout only, NOT old fixed-four authority')


def closure():
    out=six.closure();six.merge(out,paid.closure())
    for name,h in PROTOCOLS.items():
        p=ROOT/'innerdocs'/name;need(sha(p)==h,'approved frozen deployment protocol');six.merge(out,{str(p):h})
    for folder,name in (('experiments/spectral_reachability',TAG+'.py'),('tests','test_'+TAG+'.py'),('innerdocs',TAG+'.md')):
        p=ROOT/folder/name;six.merge(out,{str(p):sha(p)})
    return out


def bound_sources(sources):need(all(sha(p)==h for p,h in sources.items()),'all complete source/input/model bytes unchanged')


def upstream():
    """MAIN saved-only verification; never constructs model or runs producer solver."""
    rr=refs();m,r,c=(read(rr[k]['path'],rr[k]['sha256']) for k in ('prepare','report','receipt'))
    need(r['manifest_sha256']==c['manifest_sha256']==rr['prepare']['sha256'] and
         str(Path(r['manifest']).resolve())==rr['prepare']['path'] and c['report_sha256']==rr['report']['sha256'],'exact complete authority triple')
    complete=Path(rr['report']['path']).parent/'completion.json';ch=sha(complete);done=read(complete,ch)
    need(c['completion_sha256']==ch and done['schema']==six.TAG+'.completion' and done['complete'] is True and
         done['report_sha256']==rr['report']['sha256'] and done['boundary']==
         'Store start before verify through report fsync/hash; completion acknowledgement excluded' and
         0<=r['whole_wall_seconds_before_report']<=done['whole_wall_through_report_fsync_hash']<=six.WALL,'upstream final fsync complete boundary')
    bound_sources(m['sources'])
    check=six.analyze_saved(m,r,Path(rr['report']['path']).parent)
    six.check_authentication(c['saved_consume_authentication'],m['full_receipts']['arbitrary'])
    check.update(manifest_sha256=rr['prepare']['sha256'],report_sha256=rr['report']['sha256'],completion_sha256=ch,
                 saved_consume_authentication=c['saved_consume_authentication'],upstream_arbitrary_receipt=m['authorities']['arbitrary_receipt'])
    need(check==c,'entire actual six receipt reconstructed, not status-only or old-four schema')
    return m,r,c,dict(path=str(complete),sha256=ch)


def domain_rows(m,r,mean,std,get,trace):
    """Exact six input/returned-mean mapping; injected readers also support RAM fixtures."""
    values=six.select_payloads(m);need(len(r['rows'])==6,'all six actual rows')
    rows=[]
    for i,row in enumerate(r['rows']):
        need((row['slot'],row['root'],row['condition'],row['arm'],row['status'])==
             (i,i//3,'off',six.ARMS[i%3],'returned'),'fixed six returned identities')
        tr=trace(i,row['trace'])
        need(tr['completed'] is True and tr['errors']==[] and tr['backend']=='host_cuda_installed_CEM' and
             tr['host_tensor_seam_checked'] is True and tr['spec']==m['protocol']['spec'],'actual native CEM full trace')
        for k,v in values[i].items():need(raw.same(get(i,row['input_files'][k]),v),'actual controller input bytes '+k)
        key=tr['selected']['mean'];mean_actions=get(i,row['arrays'][key])
        for k in ('actions','mean'):need(raw.same(mean_actions,get(i,row['arrays'][tr['native_return'][k]])),'native returned mean/actions bits')
        checks=raw.check_mean(mean_actions,mean,std)
        rows.append(dict(slot=i,root=i//3,condition='off',arm=six.ARMS[i%3],
            task_identity={k:m['slots'][i][k] for k in ('task_id','task_sha256','blind_sha256')},
            input_sha256=m['slots'][i]['input_sha256'],checks=checks,
            mean_source=dict(trace=row['trace'],key=key,array=row['arrays'][key],data_sha256=core.digest(mean_actions.tobytes()))))
    return rows


def task_mapping(m,mean,std,physical,program,delivery):
    tasks=[]
    need(len(program['plans'])==4,'original full program denominator')
    for root in (0,1):
        p=program['plans'][2*root+1];original=delivery['rows'][3*root]
        need(physical['plans'][2*root+1]['planner_plan']==p,'entire original real physical task mapping')
        for i in range(3*root,3*root+3):
            need(all(m['slots'][i][k]==original[k]==p[k] for k in ('task_id','task_sha256','blind_sha256')),'all six same original task identity')
        need(p['normalization']==dict(mean=mean.tolist(),std=std.tolist()) and sha(p['task_file'])==p['task_sha256'],'task byte identity / normalization')
        warm=np.asarray(p['root_warm_raw10'],np.float32)
        need(warm.shape==(10,2) and np.isfinite(warm).all() and (np.abs(warm)<=1).all(),'actual original warm10 raw')
        _,a=raw.forward(warm,mean,std)
        need(raw.same(a,np.asarray(p['root_context']['action_history'],np.float32).reshape(10,2)),'original real history mapping')
        for i in range(3*root,3*root+3):need(raw.same(a.reshape(1,2,10),np.asarray(m['slots'][i]['controller_payload']['a'],np.float32)),'all six real history once normalized')
        tasks.append(dict(root=root,task_file=p['task_file'],task_sha256=p['task_sha256'],task_id=p['task_id'],
            warm_raw10_bits=warm.tobytes().hex(),history_sha256=core.digest(a.tobytes()),normalization=p['normalization']))
    return tasks


def build_domain():
    m,r,c,completion=upstream();rr=refs();folder=Path(rr['report']['path']).parent
    sources=dict(m['sources']);six.merge(sources,closure());six.merge(sources,{v['path']:v['sha256'] for v in rr.values()})
    six.merge(sources,{completion['path']:completion['sha256']})
    six.merge(sources,{str(folder/name):desc['file_sha256'] for name,desc in r['files'].items()})
    for ref in (c['saved_consume_authentication'],r['runtime_authentication']):
        proof=read(ref['path'],ref['sha256']);six.merge(sources,{ref['path']:ref['sha256']})
        six.merge(sources,{proof['receipt']['path']:proof['receipt']['sha256']})
    p,h=raw.PINS['physical_prepare'];pr=dict(path=str(ROOT/p),sha256=h);physical=read(pr['path'],h);sources[pr['path']]=h
    mean,std,normalizer=raw.source_authority(m,physical,sources)
    dref=six.refs()['delivery'];delivery=read(dref['path'],dref['sha256']);sources[dref['path']]=dref['sha256']
    pref=delivery['authorities']['program_receipt'];program=read(pref['path'],pref['sha256']);sources[pref['path']]=pref['sha256']
    tasks=task_mapping(m,mean,std,physical,program,delivery)
    for task in tasks:sources[task['task_file']]=task['task_sha256']
    def member(i,d):
        need(d['file'].startswith(f'slot{i}_') and d['root']==i//3 and
             {k:d[k] for k in ('file','file_bytes','file_sha256','root')}==r['files'][d['file']],'exact slot inventory and bytes')
        sources[str(folder/d['file'])]=d['file_sha256']
    def get(i,d):member(i,d);return six.load_array(folder,d)
    def trace(i,d):member(i,d);return read(folder/d['file'],d['file_sha256'])
    rows=domain_rows(m,r,mean,std,get,trace);blocked=[x['slot'] for x in rows if x['checks']['blocked_native_domain']]
    bound_sources(sources)
    return dict(schema=TAG+'.domain',authorities=rr,upstream_completion=completion,imported_complete_receipt=c,
        controller_prepare=m,normalizer=normalizer,tasks=tasks,rows=rows,sources=sources,protocol=protocol(),
        denominator=dict(total=6,checked=6,missing=0),blocked_slot_ids=blocked,
        status='BLOCKED_native_domain' if blocked else 'saved_domain_only_requires_actual_CUDA_paid_replay',
        raw_deployment_authorized=False,scientific_admission=False,original_Hi_scaler_certified=False)


def own_output(path):
    p=Path(path).absolute();need(p.parent.resolve()==ROOT/'results' and p.name.startswith(TAG+'_') and not p.is_symlink(),'own exclusive result path');return p


def save_metadata(output,value):
    p=own_output(output);b=encode(value)
    need(not p.exists() and len(b)<=32*1024**2 and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'exclusive bounded metadata/floor')
    tracker.durable(p,b);return dict(path=str(p),sha256=sha(p))


def domain(output):return save_metadata(output,build_domain())


def verify_domain(path,h):
    d=read(path,h);need(d==build_domain(),'entire independent saved domain authority rebuilt');return d


def prepare(domain_path,dh,output):
    d=verify_domain(domain_path,dh);need(d['blocked_slot_ids']==[],'whole batch BLOCKED, no paid job or env')
    m=d['controller_prepare'];sources=dict(d['sources']);six.merge(sources,{str(Path(domain_path).resolve()):dh})
    value=dict(schema=TAG+'.prepare',domain=dict(path=str(Path(domain_path).resolve()),sha256=dh),
        protocol=protocol(),export=m['export'],expected_state=m['expected_state'],expected_construction=m['expected_construction'],sources=sources)
    return save_metadata(output,value)


def verify(path,h):
    m=read(path,h);d=verify_domain(m['domain']['path'],m['domain']['sha256']);cm=d['controller_prepare']
    need(m['schema']==TAG+'.prepare' and m['protocol']==protocol() and d['blocked_slot_ids']==[],'own six protocol / whole batch domain')
    sources=dict(d['sources']);six.merge(sources,{m['domain']['path']:m['domain']['sha256']})
    need(m==dict(schema=TAG+'.prepare',domain=m['domain'],protocol=protocol(),export=cm['export'],
        expected_state=cm['expected_state'],expected_construction=cm['expected_construction'],sources=sources),'entire own prepare exact')
    bound_sources(sources);return m,d


def payloads(d):
    need(d['schema']==TAG+'.domain' and len(d['rows'])==6 and d['denominator']==dict(total=6,checked=6,missing=0),'own six domain denominator')
    need(d['blocked_slot_ids']==[],'whole batch blocked, never select successful subset')
    values=six.select_payloads(d['controller_prepare'])
    mean=np.frombuffer(bytes.fromhex(d['normalizer']['mean_bits']),'<f4').copy()
    std=np.frombuffer(bytes.fromhex(d['normalizer']['std_bits']),'<f4').copy()
    for i,row in enumerate(d['rows']):
        need((row['slot'],row['root'],row['condition'],row['arm'])==(i,i//3,'off',six.ARMS[i%3]),'fixed six domain order')
        cm=d['controller_prepare']['slots'][i]
        need(row['task_identity']=={k:cm[k] for k in ('task_id','task_sha256','blind_sha256')} and row['input_sha256']==cm['input_sha256'],'exact task / payload bits')
        norm=np.frombuffer(bytes.fromhex(row['checks']['array_bits']['normalized']),'<f4').copy().reshape(1,3,10)
        need(core.digest(norm.tobytes())==row['mean_source']['data_sha256'] and raw.check_mean(norm,mean,std)==row['checks'],'complete scalar inverse equation')
        need(row['checks']['domain_ok'] is True,'all six native domain')
        values[i].update(mean=norm,mean_xy=mean,std=std,
            sealed_raw=np.frombuffer(bytes.fromhex(row['checks']['array_bits']['raw']),'<f4').copy().reshape(15,2))
    return values


class Store(tracker.Store):
    def __init__(self,path):
        self.path=own_output(path);self.start=time.monotonic();self.used=0;self.root_used={0:0,1:0};self.files={};self.journal=[]
        need(not self.path.exists() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'new output cap+floor');self.path.mkdir()

    def guard(self,n=0,root=None,footer=False,final=False):
        need(final or footer or time.monotonic()-self.start<=WALL,'whole1350 cooperative wall')
        need(self.used+n<=CAP-(0 if final else RESERVE),'768MiB including96 reserve')
        need(shutil.disk_usage(self.path).free>=FLOOR+n,'floor200 before write/dispatch')
        if root is not None:need(root in (0,1) and self.root_used[root]+n<=ROOT_CAP,'perroot384MiB')


def rows_for():return [dict(slot=i,root=i//3,condition='off',arm=six.ARMS[i%3],status='not_started',
    debits=dict(known=True,predict_API=0,predictor_samples=0)) for i in ACTIVE]


def seal_predictions(store,rows):
    need(len(rows)==6 and all(x['status']=='predictions_sealed' for x in rows),'whole-six barrier, blocked is not success')
    return store.put('prediction_seal.json',encode(dict(schema=TAG+'.prediction_seal',
        slots=[dict(slot=x['slot'],trace=x['trace'],arrays=x['arrays']) for x in rows])))


def scores(tr,arrays,target):
    """Post-seal descriptive fsum only, not model search or deployment selection."""
    out=[]
    for replay in tr['replays']:
        p=arrays[replay['rollout']['prediction']][0,0,-1].astype(np.float64)
        delta=p-target[0].astype(np.float64);out.append(math.fsum(float(x)*float(x) for x in delta)/192)
    return out


def finalize(store,r):
    r['whole_wall_seconds_before_report']=time.monotonic()-store.start
    r['file_bytes_before_report']=store.used;r['root_file_bytes']=dict(store.root_used)
    store.put('report.json',encode(r),final=True);p=store.path/'report.json';rh=sha(p)
    elapsed=time.monotonic()-store.start
    done=dict(schema=TAG+'.completion',report_sha256=rh,complete=r['completed'] and r['errors']==[] and elapsed<=WALL,
        whole_wall_through_report_fsync_hash=elapsed,boundary=BOUNDARY)
    store.put('completion.json',encode(done),final=True)
    return dict(report=str(p),sha256=rh,completed=done['complete'])


def run_cuda(manifest,mh,output,*,main_authorized=False):
    need(main_authorized is True,'MAIN authorization before production IO/torch')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','fixed CUBLAS workspace')
    store=Store(output);caps=tracker.GPUResourceGuard(store);graph=t=None;precision=handler=None
    r=dict(schema=TAG+'.report',manifest=str(Path(manifest).resolve()),manifest_sha256=mh,protocol=protocol(),
        completed=False,errors=[],rows=rows_for(),files=store.files,write_journal=store.journal,
        construction=None,model_before=None,model_after=None,sources_after=None,prediction_seal=None,
        raw_deployment_authorized=False,scientific_admission=False)
    def timeout(signum,frame):raise TimeoutError('whole1350s alarm; no CUDA kernel preemption')
    try:
        need(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'standalone no active alarm')
        handler=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,WALL)
        m,d=verify(manifest,mh);store.put('dispatch_manifest.json',encode(m));values=payloads(d)
        for i in ACTIVE:r['rows'][i]['inputs']={k:store.array(f'slot{i}_array_{9000+j}.npy',v,i//3) for j,(k,v) in enumerate(values[i].items())}
        r['input_seal']=store.put('input_seal.json',encode([dict(slot=x['slot'],inputs=x['inputs']) for x in r['rows']]))
        import torch as t
        need(Path(t.__file__).resolve()==ROOT/'.venv/lib/python3.12/site-packages/torch/__init__.py' and t.cuda.is_available(),'pinned HOST CUDA only')
        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.nn.attention import sdpa_kernel,SDPBackend
        precision=(t.are_deterministic_algorithms_enabled(),t.backends.cuda.matmul.allow_tf32,t.backends.cudnn.allow_tf32)
        t.use_deterministic_algorithms(True);t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False
        t.cuda.set_device(0);t.cuda.reset_peak_memory_stats(0)
        def gpu():caps.observe(int(t.cuda.max_memory_allocated(0)),int(t.cuda.max_memory_reserved(0)))
        class Bounds(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                gpu();out=func(*args,**({} if kwargs is None else kwargs));gpu();return out
        class Ops:
            mul=staticmethod(t.mul);add=staticmethod(t.add);sub=staticmethod(t.sub);div=staticmethod(t.div)
            def capture(self,x):
                need(t.is_tensor(x) and x.device==t.device('cuda:0') and not x.requires_grad,'CUDA frozen nonAD capture')
                t.cuda.synchronize();return x.detach().cpu().numpy().copy()
            def layout(self,x):return dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),device=str(x.device),requires_grad=x.requires_grad)
        ops=Ops();pending={}
        with t.enable_grad(),t.autocast('cuda',enabled=False),Bounds(),sdpa_kernel(SDPBackend.MATH):
            e=m['export'];graph,r['construction']=tracker.parity.rgb.loader.construct_cuda(e['report'],e['report_sha256'],e['receipt'],e['receipt_sha256'],main_authorized=True)
            tracker.validate_construction(r['construction'],m)
            r['construction_file']=store.put('construction.json',encode(r['construction']))
            r['model_before']=tracker.state_hashes(graph,t);need(r['model_before']==m['expected_state'],'all425 before')
            facade=core.diff.wrap(graph,t,main_authorized=True);kernels=low.cuda_kernels(t,facade,main_authorized=True)
            for i in ACTIVE:
                row=r['rows'][i];start=time.monotonic();row['status']='attempted'
                row['model_before']=tracker.state_hashes(graph,t);need(row['model_before']==m['expected_state'],'slot425 before')
                xs={k:t.from_numpy(v.copy()).to('cuda:0') for k,v in values[i].items()}
                row['debits']=dict(known=False,predict_API=None,predictor_samples=None)
                tr,arrays=paid.execute(xs,kernels,ops,'host_cuda_actual_tokens')
                row['debits']=dict(known=True,predict_API=tr['predict_API_debited'],predictor_samples=tr['predictor_samples_debited'])
                paid.persist_trace(store,row,tr,arrays)
                row['local_check']=paid.validate_trace(tr,arrays,values[i]);row['model_after']=tracker.state_hashes(graph,t)
                need(row['model_after']==row['model_before'],'slot425 after')
                row['wall_seconds']=time.monotonic()-start;need(row['wall_seconds']<=SLOT_WALL,'slot180 incl persistence')
                row['status']='blocked' if tr['blocked'] else 'predictions_sealed';pending[i]=(tr,arrays)
                need(row['status']=='predictions_sealed','whole batch blocked; no subset deployment')
            r['prediction_seal']=seal_predictions(store,r['rows'])
            for i in ACTIVE:
                row=r['rows'][i];started=time.monotonic()
                row['target_MSE_fsum']=scores(*pending[i],values[i]['target'])
                row['score_file']=store.put(f'slot{i}_scores.json',encode(row['target_MSE_fsum']),i//3)
                row['status']='returned'
                row['wall_seconds']+=time.monotonic()-started;need(row['wall_seconds']<=SLOT_WALL,'slot180 incl post-seal fsum')
            gpu()
        r['model_after']=tracker.state_hashes(graph,t);need(r['model_before']==r['model_after']==m['expected_state'],'all425 after');gpu()
        r['sources_after']={p:sha(p) for p in m['sources']};need(r['sources_after']==m['sources'],'complete sources after')
        store.guard();r['completed']=True
    except BaseException as ex:r['errors'].append(dict(type=type(ex).__name__,message=str(ex)[:3000]))
    finally:
        if handler is not None:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,handler)
        if graph is not None and r['model_after'] is None:
            try:r['model_after']=tracker.state_hashes(graph,t)
            except BaseException as ex:r['errors'].append(dict(stage='final_model',message=str(ex)[:1000]))
        if 'm' in locals() and r['sources_after'] is None:
            try:r['sources_after']={p:sha(p) for p in m['sources']}
            except BaseException as ex:r['errors'].append(dict(stage='final_sources',message=str(ex)[:1000]))
        if precision is not None:
            t.use_deterministic_algorithms(precision[0]);t.backends.cuda.matmul.allow_tf32=precision[1];t.backends.cudnn.allow_tf32=precision[2]
        r['cuda_resources']=caps.snapshot();r['debits']=six.old.totals(r['rows'])
        result=finalize(store,r)
    return result


def validate_files(r,folder):
    folder=Path(folder);files=r['files']
    need({p.name for p in folder.iterdir()}==set(files)|{'report.json','completion.json'},'exact complete file census')
    for name,desc in files.items():
        p=folder/name;need(Path(name).name==name and not p.is_symlink() and desc['file']==name and
            p.stat().st_size==desc['file_bytes'] and sha(p)==desc['file_sha256'],'all file size/hash')
    need(sum(d['file_bytes'] for d in files.values())==r['file_bytes_before_report'] and
        sum(p.stat().st_size for p in folder.iterdir())<=CAP,'all output768MiB including footer')
    need(len(r['write_journal'])==len(files) and {e['name']:e for e in r['write_journal']}==
        {n:dict(name=n,root=d['root'],bytes=d['file_bytes'],returned=True) for n,d in files.items()},'all durable writes exact')
    for root in (0,1):need(sum(d['file_bytes'] for d in files.values() if d['root']==root)==r['root_file_bytes'][str(root)]<=ROOT_CAP,'perroot exact384MiB')


def validate_saved(m,d,r,folder,*,host=True):
    """Own-six complete consumer. RAM fixtures explicitly opt out of HOST evidence."""
    need(r['schema']==TAG+'.report' and r['protocol']==protocol() and r['completed'] is True and r['errors']==[],'complete own-six report')
    need(r['scientific_admission'] is False and r['raw_deployment_authorized'] is False,'no science/physical release')
    tracker.validate_construction(r['construction'],m)
    need(r['model_before']==r['model_after']==m['expected_state'] and r['sources_after']==m['sources'],'all model/source guards')
    need(0<=r['whole_wall_seconds_before_report']<=WALL,'whole before report wall')
    gpu=r['cuda_resources'];need(gpu['dispatch_guard_calls']>0 and 0<gpu['max_allocated']<=gpu['allocated_cap']==3*1024**3 and
        0<gpu['max_reserved']<=gpu['reserved_cap']==4*1024**3,'actual GPU counters/caps')
    files=r['files'];referenced=set();order=[e['name'] for e in r['write_journal']];need(len(order)==len(set(order)),'unique durable journal')
    def member(desc):
        need({k:desc[k] for k in ('file','file_bytes','file_sha256','root')}==files[desc['file']],'exact descriptor inventory');referenced.add(desc['file'])
    def load(desc):member(desc);return six.load_array(folder,desc)
    def json_file(desc):member(desc);return read(Path(folder)/desc['file'],desc['file_sha256'])
    need(json_file(files['dispatch_manifest.json'])==m and json_file(r['construction_file'])==r['construction'],'durable dispatch/construction exact')
    values=payloads(d);need(len(r['rows'])==6,'all six rows, no metadata concatenation')
    need(json_file(r['input_seal'])==[dict(slot=x['slot'],inputs=x['inputs']) for x in r['rows']],'all input seal')
    need(order.index('dispatch_manifest.json')<order.index('input_seal.json')<order.index('construction.json'),'inputs sealed before model construction')
    checks=[];seal_rows=[];last=order.index('construction.json')
    for i,row in enumerate(r['rows']):
        need((row['slot'],row['root'],row['condition'],row['arm'],row['status'])==(i,i//3,'off',six.ARMS[i%3],'returned'),'all six actual returned')
        need(0<=row['wall_seconds']<=SLOT_WALL and row['model_before']==row['model_after']==m['expected_state'],'slot425/wall')
        need(set(row['inputs'])==set(values[i]),'all and only expected inputs')
        for desc in [*row['inputs'].values(),row['trace'],*row['arrays'].values()]:
            need(desc['root']==i//3 and desc['file'].startswith(f'slot{i}_'),'no cross-slot/original array reuse')
        for k,v in values[i].items():
            need(raw.same(load(row['inputs'][k]),v) and order.index(row['inputs'][k]['file'])<order.index('input_seal.json'),'all exact presealed inputs')
        need(order.index(row['trace']['file'])>last,'all six fresh serial traces')
        tr=json_file(row['trace']);arrays={k:load(desc) for k,desc in row['arrays'].items()}
        need(all(order.index(desc['file'])>order.index(row['trace']['file']) for desc in row['arrays'].values()),'trace debits before arrays')
        last=max(order.index(desc['file']) for desc in row['arrays'].values())
        lc=paid.validate_trace(tr,arrays,values[i],host=host)
        need(lc['blocked'] is None and row['local_check']==lc,'complete all-six trace rebuilt / blocked not admitted')
        need(row['debits']==dict(known=True,predict_API=6,predictor_samples=6) and lc['predict_API']==lc['predictor_samples']==6,'actual per-slot paid6')
        need(row['target_MSE_fsum']==scores(tr,arrays,values[i]['target']),'independent post-seal target fsum')
        need(json_file(row['score_file'])==row['target_MSE_fsum'] and row['score_file']['root']==i//3 and
            row['score_file']['file']==f'slot{i}_scores.json' and
            order.index(row['score_file']['file'])>order.index('prediction_seal.json'),'all scores persisted only after six prediction seal')
        need(row['target_MSE_fsum'][0]==row['target_MSE_fsum'][1],'whole segment objective identity')
        checks.append(dict(slot=i,root=i//3,arm=row['arm'],**lc));seal_rows.append(dict(slot=i,trace=row['trace'],arrays=row['arrays']))
    need(json_file(r['prediction_seal'])==dict(schema=TAG+'.prediction_seal',slots=seal_rows) and
        order.index('prediction_seal.json')>last,'all predictions durable before barrier/score')
    need(set(files)==referenced and r['debits']==six.old.totals(r['rows']),'all files and actual debit accounting')
    return dict(schema=TAG+'.receipt',status='all6_actual_deployment_token_replays_verified_not_physically_released',
        denominator=dict(total=6,returned=6,blocked=0,missing=0),local_checks=checks,
        predict_API=sum(x['predict_API'] for x in checks),predictor_samples=sum(x['predictor_samples'] for x in checks),
        module_resources={k:sum(x[k] for x in checks) for k in ('action_encoder','low_projection','AE_tokens','predictor_context_tokens','projection_rows')},
        original_Hi_scaler_certified=False,raw_deployment_authorized=False,scientific_admission=False,
        physical_approval_required=True,FLOPs=None,all_cpu_transfer_calls=None,evidence_backend='HOST_CUDA' if host else 'RAM_fixture_not_evidence')


def check_completion(r,done,rh):
    need(done['schema']==TAG+'.completion' and done['report_sha256']==rh and done['complete'] is True and
        done['boundary']==BOUNDARY and 0<=r['whole_wall_seconds_before_report']<=done['whole_wall_through_report_fsync_hash']<=WALL,'final report fsync/hash inclusive wall')


def consume(manifest,mh,report,rh,output):
    m,d=verify(manifest,mh);r=read(report,rh);folder=Path(report).parent
    need(r['manifest_sha256']==mh and Path(r['manifest']).resolve()==Path(manifest).resolve(),'exact paid manifest/report join')
    ch=sha(folder/'completion.json');done=read(folder/'completion.json',ch);check_completion(r,done,rh)
    validate_files(r,folder);out=validate_saved(m,d,r,folder)
    out.update(manifest_sha256=mh,report_sha256=rh,completion_sha256=ch,domain=m['domain'],
               controller_authorities=d['authorities'],whole_wall_through_report_fsync_hash=done['whole_wall_through_report_fsync_hash'])
    bound_sources(m['sources']);return save_metadata(output,out)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('protocol','domain','verify-domain','prepare','verify','run','consume'))
    for k in ('domain','manifest','report','output'):p.add_argument('--'+k)
    for k in ('domain','manifest','report'):p.add_argument('--'+k+'-sha256')
    p.add_argument('--main-authorized',action='store_true');a=p.parse_args(argv)
    if a.command=='protocol':out=protocol()
    elif a.command=='domain':out=domain(a.output)
    elif a.command=='verify-domain':verify_domain(a.domain,a.domain_sha256);out=dict(verified=True)
    elif a.command=='prepare':out=prepare(a.domain,a.domain_sha256,a.output)
    elif a.command=='verify':verify(a.manifest,a.manifest_sha256);out=dict(verified=True)
    elif a.command=='run':out=run_cuda(a.manifest,a.manifest_sha256,a.output,main_authorized=a.main_authorized)
    else:out=consume(a.manifest,a.manifest_sha256,a.report,a.report_sha256,a.output)
    print(encode(out).decode())


if __name__=='__main__':main()
