"""MAIN-only known2 normalized tracker smoke. No proposals, encoder or raw deployment."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import tempfile
import time

import numpy as np
from experiments.spectral_reachability import agent_hilewm_history3_target_tracker_20261029_v1 as core
from experiments.spectral_reachability import agent_p2_hilewm_computation_parity_20261029_v1 as parity

TAG='agent_p2_history3_tracker_cuda_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
CANONICAL=ROOT/'innerdocs/main_p2_history3_tracker_cuda_slice_protocol_20261029_v1.md'
PINS={Path(core.__file__):'0a3af7256d938dcfec0698aff1eb439cf6e6701bf4baf05811ad4868f056de54',
      Path(parity.__file__):'ed44fe11aeec738852d0d560ad658ee4b469faf8211126d014ede85b6da61170',
      CANONICAL:'5a42371438472ae786d00175eba1167146718174cd8243697ddb30ee6b9247d7'}
ACTUAL=dict(manifest='796c80ca14574b29b9920b0ef53b02cc6c3b3b47842708951e5bc0b17aadf4e1',
            report='12fe43cc4a796fffc18b98fcc867283f8534c22110d794729652ac112aedead8',
            receipt='3555696c02329062084d791c0c40958084d7352e6f7680938bade0068afe9480')
CAP,RESERVE,ROOT_CAP,ROOT_RESERVE,FLOOR,WALL=256*1024**2,32*1024**2,64*1024**2,8*1024**2,200*1024**2,900
ALLOCATED,RESERVED=3*1024**3,4*1024**3
need=core.need


def encode(x): return (json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(p): return hashlib.sha256(parity.rgb.loader.bytes_read(p,128*1024**2)).hexdigest()


def read(p,h):
    b=parity.rgb.loader.bytes_read(p,32*1024**2)
    need(hashlib.sha256(b).hexdigest()==h,'exact JSON SHA')
    def pairs(xs):
        d={}
        for k,v in xs: need(k not in d,'duplicate JSON key'); d[k]=v
        return d
    return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda _:need(False,'nonfinite JSON'))


def spec():
    return core.Spec(horizon=3,samples=8,iterations=2,topk=2,var_scale=1.,seed=0,segments=(1,2),
        initialization='full_horizon_zero_no_witness',action_domain='unclipped_normalized_gaussian',
        objective='identity192_terminal_meanMSE_float64',predict_api_cap=12,predict_sample_cap=54,
        array_bytes_cap=33554432,wall_seconds=300.).checked()


def protocol():
    return dict(schema=TAG+'.protocol',roots=[0,1],condition='off',known_only=True,fresh=False,
        spec=json.loads(encode(core.asdict(spec()))),per_root=spec().planned(),
        total_predict_API=24,total_predictor_samples=108,encoder=0,high=0,AD=0,physics=0,FLOPs=None,
        whole_wall_seconds=WALL,per_root_file_cap=ROOT_CAP,output_cap=CAP,final_reserve=RESERVE,
        floor=FLOOR,cuda_allocated_cap=ALLOCATED,cuda_reserved_cap=RESERVED,
        raw_inverse=False,scientific_admission=False,performance_claim=False)


def closure():
    out=parity.closure(); out.update(core.source_closure())
    for p,h in PINS.items(): need(sha(p)==h,'frozen input source: '+str(p)); out[str(p)]=h
    for folder,name in (('experiments/spectral_reachability',TAG+'.py'),('tests','test_'+TAG+'.py'),('innerdocs',TAG+'.md')):
        p=ROOT/folder/name; out[str(p)]=sha(p)
    for rel in ('torch/utils/_python_dispatch.py','torch/cuda/memory.py','torch/cuda/__init__.py',
                'stable_worldmodel/solver/callbacks/__init__.py','stable_worldmodel/solver/callbacks/common.py',
                'stable_worldmodel/solver/callbacks/cem.py'):
        p=ROOT/'.venv/lib/python3.12/site-packages'/rel; out[str(p)]=sha(p)
    return out


def authorities(manifest,mh,report,rh,receipt,ch):
    values=dict(manifest=(manifest,mh),report=(report,rh),receipt=(receipt,ch))
    need({k:v[1] for k,v in values.items()}==ACTUAL,'fixed actual parity triple, not caller-certified future output')
    out={k:dict(path=str(Path(p).resolve()),sha256=h) for k,(p,h) in values.items()}
    # Verify bytes before the owner consumer can perform any substantial I/O.
    for ref in out.values(): read(ref['path'],ref['sha256'])
    return out


def certify_parity(auth):
    """MAIN-only saved full owner consume, temporary receipt, no model forward."""
    p,r,c=(auth[k] for k in ('manifest','report','receipt'))
    old=read(r['path'],r['sha256'])
    need(Path(old['manifest']).resolve()==Path(p['path']) and old['manifest_sha256']==p['sha256'], 'actual report->prepare binding')
    with tempfile.TemporaryDirectory(prefix=TAG+'_upstream_',dir=ROOT/'results') as d:
        tmp=Path(d)/'receipt.json'
        parity.consume(Path(r['path']),r['sha256'],tmp)
        actual=read(tmp,sha(tmp)); saved=read(c['path'],c['sha256'])
        need(actual==saved,'complete actual parity receipt, not status-only')
    need(actual['all_values_exact'] is True and actual['raw_deployment'] is False and
         actual['scientific_admission'] is False,'fixed parity numerical acceptance only')
    return read(p['path'],p['sha256']),old,actual


def select_inputs(pm,pr,folder):
    """Pure saved-array join. Never search/select roots or reconstruct goal heuristically."""
    need([(s['root'],s['condition']) for s in pm['slots']]==[(0,'off'),(1,'off')], 'fixed known2 slots')
    need(len(pr['rows'])==2,'complete prior root denominator')
    slots=[]; arrays=[]
    for s,row in zip(pm['slots'],pr['rows'],strict=True):
        need(row['root']==s['root'] and row['status']=='returned' and len(row['pixels'])==4,'original root complete')
        def get(name): return parity.load_array(folder,name,pr['arrays'][name])
        observed=[row['pixels'][i]['paths']['hi']['output'] for i in range(3)]
        goal=row['pixels'][3]['paths']['hi']['output']
        for i,pixel in enumerate(row['pixels']):
            need(pixel['image']==i,'original RGB checkpoint order')
            core.need(core.digest(get(pixel['rgb']).tobytes())==s['pixel_sha256'][i],'source RGB identity')
            need(core.exact(get(pixel['paths']['hi']['output']),get(pixel['paths']['p2']['output'])), 'both original encoders exact')
        enc=[get(n) for n in observed]
        need(all(x.shape==(1,192) and x.dtype==np.float32 for x in enc),'all3 true encoded history slots')
        z=np.concatenate(enc,axis=0)[None]
        low=row['routes']['hi_whole']; a=get(low['a']); target=get(goal)
        need(core.exact(z,get(low['z'])) and core.exact(z,np.asarray(s['z'],np.float32)[None]),'actual encodings equal original causal history3')
        need(core.exact(a,np.asarray(s['a'],np.float32)[None]),'actual original normalized between actions')
        core.check_inputs(z,a,target,core.RAMOps())
        pointers=dict(observed=observed,history=low['z'],actions=low['a'],target=goal,
                      rgb=[x['rgb'] for x in row['pixels']])
        names=observed+[low['z'],low['a'],goal]+pointers['rgb']
        descriptors={name:pr['arrays'][name] for name in names}
        slots.append(dict(root=s['root'],condition='off',task_id=s['task_id'],task_file=s['task_file'],
                          task_sha256=s['task_sha256'],blind=s['blind'],blind_sha256=s['blind_sha256'],
                          pixel_sha256=s['pixel_sha256'],pointers=pointers,source_arrays=descriptors,
                          input_sha256={k:core.digest(x.tobytes()) for k,x in (('z',z),('a',a),('target',target))}))
        arrays.append(dict(z=z,a=a,target=target))
    return slots,arrays


def expected_state(export,expected_hashes):
    need(len(export['state_dict_paths'])==len(export['rows']),'export key denominator')
    out={}
    for key,row in zip(export['state_dict_paths'],export['rows'],strict=True):
        if key['include_in_literal_state_dict']:
            a=row['array']; need(a is not None and key['key'] not in out,'actual state entry, no None fallback')
            out[key['key']]=dict(shape=a['shape'],dtype={'<f4':'torch.float32','<i8':'torch.int64'}[a['descr']],sha256=a['value_sha256'])
    need(len(out)==425 and {k:v['sha256'] for k,v in out.items()}==expected_hashes,'full425 shape/dtype/value binding')
    return out


def build(auth):
    pm,pr,receipt=certify_parity(auth)
    slots,_=select_inputs(pm,pr,Path(auth['report']['path']).parent)
    er=read(pm['export']['report'],pm['export']['report_sha256'])
    state=expected_state(er,pm['expected_hi_hashes'])
    sources=dict(pm['sources']); sources.update(closure())
    for ref in auth.values(): sources[ref['path']]=ref['sha256']
    folder=Path(auth['report']['path']).parent
    for name,desc in pr['arrays'].items(): sources[str(folder/name)]=desc['file_sha256']
    need(all(sha(p)==h for p,h in sources.items()),'all actual sources and saved arrays unchanged')
    return dict(schema=TAG+'.prepare',protocol=protocol(),authorities=auth,upstream_receipt=receipt,
                slots=slots,export=pm['export'],expected_state=state,
                expected_construction=pr['construction'],sources=sources)


def own_output(path):
    p=Path(path)
    need(p.parent.resolve()==ROOT/'results' and p.name.startswith(TAG+'_') and not p.is_symlink(),'own exclusive results child')
    return p


def durable(path,b):
    with Path(path).open('xb') as f:
        need(f.write(b)==len(b),'complete exclusive write'); f.flush(); os.fsync(f.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def prepare(output,auth):
    m=build(auth); b=encode(m); need(len(b)<=RESERVE,'manifest metadata cap')
    p=own_output(output); need(shutil.disk_usage(p.parent).free>=FLOOR+len(b),'prepare floor')
    durable(p,b); return dict(path=str(p.resolve()),sha256=sha(p))


def verify(manifest,mh):
    m=read(manifest,mh)
    auth=m['authorities']
    need({k:v['sha256'] for k,v in auth.items()}==ACTUAL,'fixed actual triple')
    need(m==build(auth),'full prepare reconstructed including external strict receipt')
    return m


class Store:
    def __init__(self,path):
        self.path=own_output(path); self.start=time.monotonic(); self.used=0; self.root_used={0:0,1:0}
        self.files={}; self.journal=[]
        need(not self.path.exists() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'new bounded output/floor')
        self.path.mkdir()

    def guard(self,n=0,root=None,footer=False,final=False):
        need(final or time.monotonic()-self.start<=WALL,'whole job cooperative wall')
        need(self.used+n<=CAP-(0 if final else RESERVE),'whole file cap/reserve')
        need(shutil.disk_usage(self.path).free>=FLOOR+n,'floor before output/dispatch')
        if root is not None:
            need(root in (0,1) and self.root_used[root]+n<=ROOT_CAP-(0 if footer else ROOT_RESERVE),'perroot cap/reserve')

    def put(self,name,b,root=None,footer=False,final=False):
        need(Path(name).name==name and name not in self.files,'flat exclusive file')
        event=dict(name=name,root=root,bytes=len(b),returned=False)
        if not final: self.journal.append(event)
        try:
            self.guard(len(b),root,footer,final); durable(self.path/name,b)
            self.used+=len(b)
            if root is not None: self.root_used[root]+=len(b)
            desc=dict(file=name,file_bytes=len(b),file_sha256=hashlib.sha256(b).hexdigest(),root=root)
            if not final: self.files[name]=desc
            event['returned']=True
            return desc
        except BaseException as ex:
            event['error']=type(ex).__name__+': '+str(ex)[:1000]
            raise

    def array(self,name,a,root):
        need(type(a) is np.ndarray and a.dtype.str in ('<f4','<f8','<i8') and np.isfinite(a).all(),'finite exact array ABI')
        s=io.BytesIO(); np.lib.format.write_array(s,np.array(a,copy=True,order='C'),version=(1,0),allow_pickle=False)
        d=self.put(name,s.getvalue(),root)
        return dict(**d,shape=list(a.shape),dtype=a.dtype.str,data_sha256=core.digest(a.tobytes(order='C')))


def load_saved_array(folder,desc):
    name=desc['file']; need(re.fullmatch(r'root[01]_(?:input_(?:z|a|target)|array_[0-9]+)\.npy',name) is not None,'fixed flat root NPY')
    b=parity.rgb.loader.bytes_read(Path(folder)/name,ROOT_CAP)
    need(len(b)==desc['file_bytes'] and hashlib.sha256(b).hexdigest()==desc['file_sha256'],'durable file SHA/size')
    f=io.BytesIO(b); need(np.lib.format.read_magic(f)==(1,0),'NPY v1 only')
    shape,order,dtype=np.lib.format.read_array_header_1_0(f)
    need(list(shape)==desc['shape'] and dtype.str==desc['dtype'] and dtype.str in ('<f4','<f8','<i8')
         and not order and math.prod(shape)*dtype.itemsize==len(b)-f.tell(),'exact bounded NPY header/EOF')
    f.seek(0); a=np.load(f,allow_pickle=False)
    need(np.isfinite(a).all() and core.digest(a.tobytes())==desc['data_sha256'],'finite original data SHA')
    return a


def persist_core(store,row,report,arrays):
    root=row['root']; row['arrays']={}
    for key,a in arrays.items():
        need(re.fullmatch(r'array_[0-9]+',key) is not None,'core array key')
        row['arrays'][key]=store.array('root%d_%s.npy'%(root,key),a,root)
    # Core metadata stays untouched. No relabelling RAM traces into HOST.
    row['trace']=store.put('root%d_trace.json'%root,encode(report),root,footer=True)
    row['trace_completed']=report['completed']


def state_hashes(graph,t):
    values=graph.state_dict()
    need(len(values)==425 and all(not m.training for m in graph.modules()),'original complete frozen eval graph')
    need(all(not p.requires_grad and p.device==t.device('cuda:0') for p in graph.parameters()),'frozen actual CUDA parameters')
    return {k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=core.digest(v.detach().contiguous().cpu().numpy().tobytes()))
            for k,v in values.items()}


def validate_construction(c,m):
    need(c==m['expected_construction'],'entire actual parity construction receipt, not caller claims')
    need(all(m['sources'].get(p)==h for p,h in c['own_sha256'].items()),'constructor code/test/doc presealed')
    need(c['tensor_copy_attempts']==c['tensor_copy_returns']==c['state_key_count']==425 and c['forward_calls']==c['physics']==0,
         'no forward at construction;425 copies')


class GPUResourceGuard:
    """Plain counter policy testable without torch; used by actual TorchDispatchMode."""
    def __init__(self,store): self.store=store; self.calls=0; self.max_allocated=0; self.max_reserved=0; self.stages=[]
    def observe(self,allocated,reserved):
        self.calls+=1; self.max_allocated=max(self.max_allocated,allocated); self.max_reserved=max(self.max_reserved,reserved)
        self.store.guard()
        need(allocated<=ALLOCATED and reserved<=RESERVED,'CUDA allocated/reserved hard cap')
    def snapshot(self):
        return dict(dispatch_guard_calls=self.calls,max_allocated=self.max_allocated,max_reserved=self.max_reserved,
                    allocated_cap=ALLOCATED,reserved_cap=RESERVED,stages=self.stages)


def run_cuda(manifest,mh,output,*,main_authorized=False):
    need(main_authorized is True,'MAIN authorization before production I/O/torch')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','fixed CUBLAS workspace')
    store=Store(output); caps=GPUResourceGuard(store); graph=t=None; old_precision=None
    r=dict(schema=TAG+'.report',manifest=str(Path(manifest).resolve()),manifest_sha256=mh,
           completed=False,errors=[],rows=[dict(root=i,status='not_started',arrays={},local_check=None,
               debits=dict(known=True,predict_API=0,predictor_samples=0)) for i in (0,1)],
           construction=None,model_before=None,model_after=None,sources_after=None,resources=protocol(),
           files=store.files,write_journal=store.journal,raw_inverse=False,scientific_admission=False)
    old_handler=None
    def timeout(signum,frame): raise TimeoutError('whole-job900s alarm; cannot preempt an active CUDA kernel')
    try:
        need(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'standalone CLI, no existing alarm replaced')
        old_handler=signal.signal(signal.SIGALRM,timeout); signal.setitimer(signal.ITIMER_REAL,WALL)
        m=verify(manifest,mh); store.guard()
        store.put('dispatch_manifest.json',encode(m))
        pm=read(m['authorities']['manifest']['path'],m['authorities']['manifest']['sha256'])
        pr=read(m['authorities']['report']['path'],m['authorities']['report']['sha256'])
        slots,inputs=select_inputs(pm,pr,Path(m['authorities']['report']['path']).parent)
        need(slots==m['slots'],'current actual input pointers')
        for row,values in zip(r['rows'],inputs,strict=True):
            row['input_files']={k:store.array('root%d_input_%s.npy'%(row['root'],k),a,row['root']) for k,a in values.items()}
            row['source']=m['slots'][row['root']]
        import torch as t
        need(Path(t.__file__).resolve()==ROOT/'.venv/lib/python3.12/site-packages/torch/__init__.py' and t.cuda.is_available(),
             'actual pinned venv HOST CUDA; no CPU fallback')
        from torch.utils._python_dispatch import TorchDispatchMode
        old_precision=(t.are_deterministic_algorithms_enabled(),t.backends.cuda.matmul.allow_tf32,t.backends.cudnn.allow_tf32)
        t.use_deterministic_algorithms(True); t.backends.cuda.matmul.allow_tf32=False; t.backends.cudnn.allow_tf32=False
        t.cuda.set_device(0); t.cuda.reset_peak_memory_stats(0)
        def gpu_check(): caps.observe(int(t.cuda.max_memory_allocated(0)),int(t.cuda.max_memory_reserved(0)))
        class Bounds(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                gpu_check(); result=func(*args,**({} if kwargs is None else kwargs)); gpu_check(); return result
        gpu_check()
        with t.enable_grad(),t.autocast('cuda',enabled=False),Bounds():
            e=m['export']
            graph,r['construction']=parity.rgb.loader.construct_cuda(e['report'],e['report_sha256'],e['receipt'],e['receipt_sha256'],main_authorized=True)
            validate_construction(r['construction'],m)
            r['model_before']=state_hashes(graph,t); need(r['model_before']==m['expected_state'],'all original425 values/shapes/dtypes')
            caps.stages.append(dict(stage='constructed',allocated=int(t.cuda.memory_allocated()),reserved=int(t.cuda.memory_reserved())))
            for row,values in zip(r['rows'],inputs,strict=True):
                store.guard(); row['status']='attempted'; start=time.monotonic()
                row['dispatch_model_before']=state_hashes(graph,t)
                tensors={k:t.from_numpy(a.copy()).to('cuda:0') for k,a in values.items()}
                row['debits']=dict(known=False,predict_API=None,predictor_samples=None)
                cr,ca=core.track_cuda(graph,tensors['z'],tensors['a'],tensors['target'],spec=spec(),
                                     expected_state=m['expected_state'],main_authorized=True)
                row['debits']=dict(known=True,predict_API=cr['predict_API_debited'],predictor_samples=cr['predictor_samples_debited'])
                row['core_errors']=cr['errors']
                # No rerun/extra replay: preserve whatever the frozen core actually returned.
                persist_core(store,row,cr,ca)
                row['status']='partial'
                need(cr['backend']=='host_cuda_installed_CEM' and cr.get('host_tensor_seam_checked') is True,'actual HOST path only')
                row['local_check']=core.validate_saved(cr,ca)
                row['dispatch_model_after']=state_hashes(graph,t)
                need(row['dispatch_model_before']==row['dispatch_model_after']==m['expected_state'],'perroot complete original model guard')
                row['wall_seconds']=time.monotonic()-start
                need(row['wall_seconds']<=300,'root300s complete integration scope')
                gpu_check(); row['status']='returned'; del tensors,ca,cr
                caps.stages.append(dict(stage='root%d_returned'%row['root'],allocated=int(t.cuda.memory_allocated()),reserved=int(t.cuda.memory_reserved())))
        r['model_after']=state_hashes(graph,t)
        need(r['model_after']==m['expected_state'],'all original weights after known2')
        r['sources_after']={p:sha(p) for p in m['sources']}; need(r['sources_after']==m['sources'],'all presealed source/input bytes after run')
        store.guard(); r['completed']=True
    except BaseException as ex:
        r['errors'].append(dict(type=type(ex).__name__,message=str(ex)[:3000]))
    finally:
        if old_handler is not None:
            signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old_handler)
        if graph is not None and r['model_after'] is None:
            try: r['model_after']=state_hashes(graph,t)
            except BaseException as ex: r['errors'].append(dict(stage='final_model_guard',message=str(ex)[:1000]))
        if 'm' in locals() and r['sources_after'] is None:
            try: r['sources_after']={p:sha(p) for p in m['sources']}
            except BaseException as ex: r['errors'].append(dict(stage='final_source_guard',message=str(ex)[:1000]))
        if old_precision is not None:
            t.use_deterministic_algorithms(old_precision[0]); t.backends.cuda.matmul.allow_tf32=old_precision[1]; t.backends.cudnn.allow_tf32=old_precision[2]
        r['cuda_resources']=caps.snapshot(); r['whole_wall_seconds']=time.monotonic()-store.start
        r['file_bytes_before_report']=store.used; r['root_file_bytes']=store.root_used
        r['known_predict_API_debited']=sum(row['debits']['predict_API'] for row in r['rows'] if row['debits']['known'])
        r['known_predictor_samples_debited']=sum(row['debits']['predictor_samples'] for row in r['rows'] if row['debits']['known'])
        known=all(row['debits']['known'] for row in r['rows'])
        r['actual_predict_API_debited']=r['known_predict_API_debited'] if known else None
        r['actual_predictor_samples_debited']=r['known_predictor_samples_debited'] if known else None
        store.put('report.json',encode(r),final=True)
    return dict(report=str((store.path/'report.json').resolve()),sha256=sha(store.path/'report.json'),completed=r['completed'])


def analyze_saved(m,r,folder):
    """No torch, loader or neural forward; full frozen core local_check on disk arrays."""
    need(r['schema']==TAG+'.report' and r['completed'] is True and r['errors']==[],'complete known2 HOST job required')
    need(r['resources']==protocol() and r['raw_inverse'] is False and r['scientific_admission'] is False,'fixed smoke protocol/no raw/science')
    validate_construction(r['construction'],m)
    need(r['model_before']==r['model_after']==m['expected_state'] and r['sources_after']==m['sources'],'full frozen source/model closure')
    files=r['files']; names=set(files)
    validate_files(r,folder)
    def ref(d):
        need({k:d[k] for k in ('file','file_bytes','file_sha256','root')}==files[d['file']], 'root/file descriptor joined to entire inventory')
        return d
    need(read(Path(folder)/'dispatch_manifest.json',files['dispatch_manifest.json']['file_sha256'])==m,'actual predispatch durable manifest')
    pm=read(m['authorities']['manifest']['path'],m['authorities']['manifest']['sha256'])
    pr=read(m['authorities']['report']['path'],m['authorities']['report']['sha256'])
    slots,inputs=select_inputs(pm,pr,Path(m['authorities']['report']['path']).parent)
    need(slots==m['slots'] and len(r['rows'])==2,'complete source join/denominator')
    referenced={'dispatch_manifest.json'}; checks=[]; api=samples=0
    for i,(row,values) in enumerate(zip(r['rows'],inputs,strict=True)):
        need(row['root']==i and row['source']==slots[i] and row['status']=='returned' and row['trace_completed'] is True,'all2 fixed roots returned')
        need(row['dispatch_model_before']==row['dispatch_model_after']==m['expected_state'],'per dispatch weights')
        need(0<=row['wall_seconds']<=300,'root wall scope')
        for key,a in values.items():
            d=ref(row['input_files'][key]); referenced.add(d['file'])
            need(d['file']=='root%d_input_%s.npy'%(i,key) and core.exact(a,load_saved_array(folder,d)) and d['root']==i,'actual original input copy')
        td=ref(row['trace']); referenced.add(td['file']); trace=read(Path(folder)/td['file'],td['file_sha256'])
        need(td['file']=='root%d_trace.json'%i and td['root']==i,'root trace identity')
        need(trace['backend']=='host_cuda_installed_CEM' and trace.get('host_tensor_seam_checked') is True,'never promote fixture trace')
        need(trace['spec']==protocol()['spec'] and trace['model_before']==trace['model_after']==m['expected_state'],'actual fixed spec/complete model')
        need(all(m['sources'].get(p)==h for p,h in trace['sources_before'].items()),'core executable closure prebound')
        arrays={}
        for key,d in row['arrays'].items():
            ref(d)
            need(d['root']==i and d['file']=='root%d_%s.npy'%(i,key),'no crossroot reuse or snapshot dedup'); referenced.add(d['file']); arrays[key]=load_saved_array(folder,d)
        for key,a in values.items(): need(core.exact(a,arrays[trace['inputs'][key]]),'actual input delivered to tracker')
        c=core.validate_saved(trace,arrays); need(c==row['local_check'],'entire actual local_check, not status-only')
        checks.append(c); api+=trace['predict_API_debited']; samples+=trace['predictor_samples_debited']
        need(row['debits']==dict(known=True,predict_API=trace['predict_API_debited'],predictor_samples=trace['predictor_samples_debited'])
             and row['core_errors']==trace['errors']==[], 'complete actual debit record, no refunds')
        used=sum(d['file_bytes'] for d in files.values() if d['root']==i)
        need(used==r['root_file_bytes'][str(i)] and used<=ROOT_CAP,'perroot actual file cap')
    need(referenced==names,'all file references, no unconsumed metadata')
    need(api==r['actual_predict_API_debited']==24 and samples==r['actual_predictor_samples_debited']==108,'all paid known2/replay budget axes')
    need(r['known_predict_API_debited']==api and r['known_predictor_samples_debited']==samples,'known debit totals')
    gpu=r['cuda_resources']
    need(gpu['dispatch_guard_calls']>0 and 0<gpu['max_allocated']<=ALLOCATED and 0<gpu['max_reserved']<=RESERVED
         and gpu['allocated_cap']==ALLOCATED and gpu['reserved_cap']==RESERVED,'actual guarded CUDA memory peaks')
    need(0<=r['whole_wall_seconds']<=WALL,'whole900s scope')
    return dict(schema=TAG+'.receipt',status='all_known2_normalized_tracker_saved_verified',
                local_checks=checks,predict_API=api,predictor_samples=samples,
                encoder=0,high=0,AD=0,physics=0,FLOPs=None,raw_inverse=False,
                scientific_admission=False,performance_claim=False,independent_gradient_recompute=False)


def validate_files(r,folder):
    """Bounded durable census and write ledger, also usable on RAM fixtures."""
    files=r['files']; names=set(files)
    need({p.name for p in Path(folder).iterdir()}==names|{'report.json'},'exact durable file census')
    for name,d in files.items():
        need(d['file']==name and Path(name).name==name and sha(Path(folder)/name)==d['file_sha256'] and
             (Path(folder)/name).stat().st_size==d['file_bytes'],'all durable file hashes/sizes')
    need(sum(d['file_bytes'] for d in files.values())==r['file_bytes_before_report'] and
         r['file_bytes_before_report']+(Path(folder)/'report.json').stat().st_size<=CAP,'total real file budget')
    need(len(r['write_journal'])==len(files) and {e['name']:e for e in r['write_journal']}==
         {name:dict(name=name,root=d['root'],bytes=d['file_bytes'],returned=True) for name,d in files.items()},
         'all actual exclusive write attempts returned')
    return dict(files=len(files),actual_file_bytes=r['file_bytes_before_report'])


def consume(report,rh,output):
    r=read(report,rh); m=verify(r['manifest'],r['manifest_sha256'])
    c=analyze_saved(m,r,Path(report).parent); c['report_sha256']=rh; c['manifest_sha256']=r['manifest_sha256']
    p=own_output(output); b=encode(c); need(len(b)<=RESERVE and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'receipt cap/floor')
    durable(p,b); return dict(path=str(p.resolve()),sha256=sha(p))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('prepare','verify','run','consume','protocol'))
    for key in ('manifest','report','output','parity-manifest','parity-report','parity-receipt'): p.add_argument('--'+key,type=Path)
    for key in ('manifest','report','parity-manifest','parity-report','parity-receipt'): p.add_argument('--'+key+'-sha256')
    p.add_argument('--main-authorized',action='store_true'); a=p.parse_args(argv)
    if a.command=='prepare':
        auth=authorities(a.parity_manifest,a.parity_manifest_sha256,a.parity_report,a.parity_report_sha256,a.parity_receipt,a.parity_receipt_sha256)
        value=prepare(a.output,auth)
    elif a.command=='verify': value=dict(verified=bool(verify(a.manifest,a.manifest_sha256)))
    elif a.command=='run': value=run_cuda(a.manifest,a.manifest_sha256,a.output,main_authorized=a.main_authorized)
    elif a.command=='consume': value=consume(a.report,a.report_sha256,a.output)
    else: value=protocol()
    print(json.dumps(value,sort_keys=True,allow_nan=False))


if __name__=='__main__': main()
