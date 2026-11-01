"""Four fixed delivered targets, MAIN HOST CUDA only; arbitrary stays missing."""
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
import time

import numpy as np
from experiments.spectral_reachability import agent_p2_target_only_delivery_20261029_v1 as delivery

TAG='agent_p2_target_controller_cuda_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
tracker=delivery.tracker
core=tracker.core
need=tracker.need
DELIVERY_SHA='82b36fa3030f8ebb2da37bdc91ba3e8d582f129d6bc9e5c9f8e774b363ef3f62'
DELIVERY_PATH=ROOT/'results/agent_p2_target_only_delivery_20261029_v1_prepare_MAIN01.json'
CAP,RESERVE,ROOT_CAP,ROOT_RESERVE,FLOOR,WALL=1024*1024**2,128*1024**2,256*1024**2,32*1024**2,200*1024**2,1800
ALLOCATED,RESERVED=3*1024**3,4*1024**3
ACTIVE=(0,1,3,4)
ARMS=('prior_FIRST','program_first_postcondition','arbitrary')
PINS={Path(delivery.__file__):'e69b2d1e5fcd5d853004a0b3b2126c8506ffd7068a39ac867b60134d3358f4c7',
      ROOT/'tests/test_agent_p2_target_only_delivery_20261029_v1.py':'6a3630ab6c67d7d1d590a856e594eb110f9cb4a8732adf56cf499ee8c5c44ae9',
      ROOT/'innerdocs/agent_p2_target_only_delivery_20261029_v1.md':'722ca5aa5457c3f9e9708a7deffd7b79ff8a0b5506e6b4b8d64ff4a3deec740b'}
sha=tracker.sha
read=tracker.read
encode=tracker.encode


def protocol():
    return dict(schema=TAG+'.protocol',roots=[0,1],arms=list(ARMS),active_slots=list(ACTIVE),
        denominator=6,missing_slots=[2,5],condition='off',spec=json.loads(encode(core.asdict(tracker.spec()))),
        per_slot=tracker.spec().planned(),total_predict_API=48,total_predictor_samples=216,
        encoder=0,high=0,AD=0,physics=0,FLOPs=None,output_cap=CAP,footer_reserve=RESERVE,
        per_root_file_cap=ROOT_CAP,per_root_footer_reserve=ROOT_RESERVE,floor=FLOOR,wall_seconds=WALL,
        cuda_allocated_cap=ALLOCATED,cuda_reserved_cap=RESERVED,known_only=True,fresh=False,
        time_matched=False,equal_generation_budget=False,fair_baseline=False,scientific_admission=False,raw_inverse=False)


def closure():
    out=tracker.closure()
    for p,h in PINS.items():need(sha(p)==h,'frozen delivery triple');out[str(p)]=h
    for folder,name in (('experiments/spectral_reachability',TAG+'.py'),('tests','test_'+TAG+'.py'),('innerdocs',TAG+'.md')):
        p=ROOT/folder/name;out[str(p)]=sha(p)
    return out


def select_payloads(d):
    need(d['schema']==delivery.TAG+'.prepare' and len(d['rows'])==6,'complete six delivery slots')
    need(d['denominator']==dict(roots=2,arms=3,total=6,assembled=4,missing=2,dispatched=0),'fixed assembled/missing denominator')
    out={}
    for i,row in enumerate(d['rows']):
        need((row['root'],row['condition'],row['arm'])==(i//3,'off',ARMS[i%3]),'fixed slot identity/order')
        if i in ACTIVE:
            need(row['status']=='assembled_not_dispatched','all four assembled')
            values=delivery.validate_payload(row['controller_payload'])
            need(core.digest(values['target'].tobytes())==row['target_sha256'],'actual delivered target SHA')
            out[i]=values
        else:
            need(row['status']=='missing' and row['controller_payload'] is None,'arbitrary stays missing, never padded')
    return out


def build(path,h):
    need(h==DELIVERY_SHA and Path(path).resolve()==DELIVERY_PATH,'fixed actual delivery prepare')
    d=read(path,h)
    need(d==delivery.build(),'entire actual saved-only delivery reconstructed; no full strict')
    select_payloads(d)
    auth=d['authorities']['tracker_prepare'];old=read(auth['path'],auth['sha256'])
    sources=dict(old['sources'])
    for p,v in d['sources'].items():
        need(p not in sources or sources[p]==v,'consistent source closures');sources[p]=v
    sources.update(closure());sources[str(Path(path).resolve())]=h
    need(all(sha(p)==v for p,v in sources.items()),'full original425/source/input closure unchanged')
    return dict(schema=TAG+'.prepare',protocol=protocol(),delivery=dict(path=str(Path(path).resolve()),sha256=h),
        delivery_artifact=d,export=old['export'],expected_state=old['expected_state'],
        expected_construction=old['expected_construction'],sources=sources)


def own_output(p):
    p=Path(p);need(p.parent.resolve()==ROOT/'results' and p.name.startswith(TAG+'_') and not p.is_symlink(),'exclusive own results child');return p


def prepare(output,path=DELIVERY_PATH,h=DELIVERY_SHA):
    m=build(path,h);b=encode(m);p=own_output(output)
    need(len(b)<=32*1024**2 and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'prepare cap/floor')
    tracker.durable(p,b);return dict(path=str(p.resolve()),sha256=sha(p))


def verify(path,h):
    m=read(path,h);d=m['delivery'];need(m==build(d['path'],d['sha256']),'full prepare reconstructed')
    return m


class Store(tracker.Store):
    """Reuse exclusive write/NPY logic, explicitly owned caps; no global patch."""
    def __init__(self,path):
        self.path=own_output(path);self.start=time.monotonic();self.used=0;self.root_used={0:0,1:0}
        self.files={};self.journal=[]
        need(not self.path.exists() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'new output/floor')
        self.path.mkdir()

    def guard(self,n=0,root=None,footer=False,final=False):
        need(final or time.monotonic()-self.start<=WALL,'whole1800s cooperative wall')
        need(self.used+n<=CAP-(0 if final else RESERVE),'job1024MiB/128reserve')
        need(shutil.disk_usage(self.path).free>=FLOOR+n,'floor before write/dispatch')
        if root is not None:
            need(root in (0,1) and self.root_used[root]+n<=ROOT_CAP-(0 if footer else ROOT_RESERVE),'root256MiB/32reserve')


def input_rows(d):
    select_payloads(d)
    return [dict(slot=i,root=r['root'],arm=r['arm'],condition='off',
        status='not_started' if i in ACTIVE else 'missing',missing_reason=r.get('reason') if i not in ACTIVE else None,
        arrays={},local_check=None,debits=dict(known=True,predict_API=0,predictor_samples=0))
        for i,r in enumerate(d['rows'])]


def persist_inputs(store,rows,values):
    need(len(rows)==6 and tuple(values)==ACTIVE,'all four inputs sealed before construction')
    for i in ACTIVE:
        row=rows[i]
        row['input_files']={k:store.array(f'slot{i}_input_{k}.npy',a,row['root']) for k,a in values[i].items()}


def persist_trace(store,row,report,arrays):
    row['arrays']={}
    for key,a in arrays.items():
        need(re.fullmatch(r'array_[0-9]+',key) is not None,'original core array key')
        row['arrays'][key]=store.array(f"slot{row['slot']}_{key}.npy",a,row['root'])
    row['trace']=store.put(f"slot{row['slot']}_trace.json",encode(report),row['root'],footer=True)
    row['trace_completed']=report['completed']


def load_array(folder,d):
    name=d['file'];need(re.fullmatch(r'slot[0134]_(?:input_(?:z|a|target)|array_[0-9]+)\.npy',name) is not None,'flat fixed slot NPY')
    b=tracker.parity.rgb.loader.bytes_read(Path(folder)/name,ROOT_CAP)
    need(len(b)==d['file_bytes'] and hashlib.sha256(b).hexdigest()==d['file_sha256'],'NPY file bytes/SHA')
    f=io.BytesIO(b);need(np.lib.format.read_magic(f)==(1,0),'NPYv1')
    shape,order,dtype=np.lib.format.read_array_header_1_0(f)
    need(list(shape)==d['shape'] and dtype.str==d['dtype'] and dtype.str in ('<f4','<f8','<i8') and
         not order and math.prod(shape)*dtype.itemsize==len(b)-f.tell(),'finite allowed dtype/shape/exact EOF')
    f.seek(0);a=np.load(f,allow_pickle=False)
    need(np.isfinite(a).all() and core.digest(a.tobytes())==d['data_sha256'],'NPY finite/data SHA');return a


def execute_slots(store,rows,values,invoke,snapshot,expected_state):
    """Strict orchestration. invoke receives ONLY a fresh three-array dict."""
    need(all('input_files' in rows[i] for i in ACTIVE),'all four input files first')
    for i in ACTIVE:
        store.guard();row=rows[i];row['status']='attempted';start=time.monotonic()
        row['dispatch_model_before']=snapshot()
        need(row['dispatch_model_before']==expected_state,'per-slot original425 before')
        data=values[i];need(set(data)=={'z','a','target'},'only whitelisted causal input/target')
        row['debits']=dict(known=False,predict_API=None,predictor_samples=None)
        cr,ca=invoke({k:v.copy() for k,v in data.items()})
        row['debits']=dict(known=True,predict_API=cr['predict_API_debited'],predictor_samples=cr['predictor_samples_debited'])
        row['core_errors']=cr['errors'];row['status']='partial'
        persist_trace(store,row,cr,ca)
        need(cr['backend']=='host_cuda_installed_CEM' and cr.get('host_tensor_seam_checked') is True,'HOST only; no fixture promotion')
        row['local_check']=core.validate_saved(cr,ca)
        row['dispatch_model_after']=snapshot()
        need(row['dispatch_model_before']==row['dispatch_model_after']==expected_state,'per-slot original425 after')
        row['wall_seconds']=time.monotonic()-start
        need(row['wall_seconds']<=300,'complete per-slot300s integration scope')
        row['status']='returned'


def totals(rows):
    known=all(r['debits']['known'] for r in rows)
    api=sum(r['debits']['predict_API'] for r in rows if r['debits']['known'])
    samples=sum(r['debits']['predictor_samples'] for r in rows if r['debits']['known'])
    return dict(known_predict_API_debited=api,known_predictor_samples_debited=samples,
        actual_predict_API_debited=api if known else None,actual_predictor_samples_debited=samples if known else None)


def run_cuda(manifest,mh,output,*,main_authorized=False):
    need(main_authorized is True,'MAIN authorization before production I/O/torch')
    need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','fixed CUBLAS workspace')
    store=Store(output);caps=tracker.GPUResourceGuard(store);graph=t=None;old_precision=None;old_handler=None
    r=dict(schema=TAG+'.report',manifest=str(Path(manifest).resolve()),manifest_sha256=mh,completed=False,errors=[],
        rows=[dict(slot=i,root=i//3,arm=ARMS[i%3],status='not_started' if i in ACTIVE else 'missing',
                   debits=dict(known=True,predict_API=0,predictor_samples=0)) for i in range(6)],
        protocol=protocol(),construction=None,model_before=None,model_after=None,sources_after=None,
        files=store.files,write_journal=store.journal,scientific_admission=False,raw_inverse=False)
    def timeout(signum,frame):raise TimeoutError('whole1800s alarm; no active CUDA kernel preemption')
    try:
        need(signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'standalone CLI no existing alarm overwritten')
        old_handler=signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,WALL)
        m=verify(manifest,mh);store.put('dispatch_manifest.json',encode(m))
        values=select_payloads(m['delivery_artifact']);r['rows']=input_rows(m['delivery_artifact'])
        persist_inputs(store,r['rows'],values)
        # No environment/source labels or origin metadata below this boundary.
        import torch as t
        need(Path(t.__file__).resolve()==ROOT/'.venv/lib/python3.12/site-packages/torch/__init__.py' and t.cuda.is_available(),
             'actual pinned venv HOST CUDA only')
        from torch.utils._python_dispatch import TorchDispatchMode
        old_precision=(t.are_deterministic_algorithms_enabled(),t.backends.cuda.matmul.allow_tf32,t.backends.cudnn.allow_tf32)
        t.use_deterministic_algorithms(True);t.backends.cuda.matmul.allow_tf32=False;t.backends.cudnn.allow_tf32=False
        t.cuda.set_device(0);t.cuda.reset_peak_memory_stats(0)
        def gpu_check():caps.observe(int(t.cuda.max_memory_allocated(0)),int(t.cuda.max_memory_reserved(0)))
        class Bounds(TorchDispatchMode):
            def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                gpu_check();out=func(*args,**({} if kwargs is None else kwargs));gpu_check();return out
        gpu_check()
        with t.enable_grad(),t.autocast('cuda',enabled=False),Bounds():
            e=m['export']
            graph,r['construction']=tracker.parity.rgb.loader.construct_cuda(e['report'],e['report_sha256'],e['receipt'],e['receipt_sha256'],main_authorized=True)
            tracker.validate_construction(r['construction'],m)
            def snapshot():return tracker.state_hashes(graph,t)
            r['model_before']=snapshot();need(r['model_before']==m['expected_state'],'all425 before')
            def invoke(p):
                need(set(p)=={'z','a','target'},'no witness metadata in controller dispatch')
                xs={k:t.from_numpy(a.copy()).to('cuda:0') for k,a in p.items()}
                return core.track_cuda(graph,xs['z'],xs['a'],xs['target'],spec=tracker.spec(),expected_state=m['expected_state'],main_authorized=True)
            execute_slots(store,r['rows'],values,invoke,snapshot,m['expected_state']);gpu_check()
        r['model_after']=tracker.state_hashes(graph,t)
        need(r['model_before']==r['model_after']==m['expected_state'],'all425 after')
        r['sources_after']={p:sha(p) for p in m['sources']};need(r['sources_after']==m['sources'],'entire original source/input after')
        store.guard();r['completed']=True
    except BaseException as ex:r['errors'].append(dict(type=type(ex).__name__,message=str(ex)[:3000]))
    finally:
        if old_handler is not None:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        if graph is not None and r['model_after'] is None:
            try:r['model_after']=tracker.state_hashes(graph,t)
            except BaseException as ex:r['errors'].append(dict(stage='final_model_guard',message=str(ex)[:1000]))
        if 'm' in locals() and r['sources_after'] is None:
            try:r['sources_after']={p:sha(p) for p in m['sources']}
            except BaseException as ex:r['errors'].append(dict(stage='final_source_guard',message=str(ex)[:1000]))
        if old_precision is not None:
            t.use_deterministic_algorithms(old_precision[0]);t.backends.cuda.matmul.allow_tf32=old_precision[1];t.backends.cudnn.allow_tf32=old_precision[2]
        r.update(totals(r['rows']));r['cuda_resources']=caps.snapshot();r['whole_wall_seconds']=time.monotonic()-store.start
        r['file_bytes_before_report']=store.used;r['root_file_bytes']=store.root_used
        store.put('report.json',encode(r),final=True)
    return dict(report=str((store.path/'report.json').resolve()),sha256=sha(store.path/'report.json'),completed=r['completed'])


def validate_files(r,folder):
    files=r['files'];need({p.name for p in Path(folder).iterdir()}==set(files)|{'report.json'},'exact file census')
    for n,d in files.items():
        need(Path(n).name==n and d['file']==n and sha(Path(folder)/n)==d['file_sha256'] and
             (Path(folder)/n).stat().st_size==d['file_bytes'],'all file bytes/SHA')
    used=sum(d['file_bytes'] for d in files.values())
    need(used==r['file_bytes_before_report'] and used+(Path(folder)/'report.json').stat().st_size<=CAP,'entire1024MiB budget')
    need(len(r['write_journal'])==len(files) and {e['name']:e for e in r['write_journal']}==
         {n:dict(name=n,root=d['root'],bytes=d['file_bytes'],returned=True) for n,d in files.items()},'all write attempts returned')
    for root in (0,1):need(sum(d['file_bytes'] for d in files.values() if d['root']==root)==r['root_file_bytes'][str(root)]<=ROOT_CAP,'root two-slot256MiB')


def analyze_saved(m,r,folder):
    need(r['schema']==TAG+'.report' and r['completed'] is True and r['errors']==[],'complete four HOST slots')
    need(r['protocol']==protocol() and r['scientific_admission'] is False and r['raw_inverse'] is False,'fixed integration-only protocol')
    tracker.validate_construction(r['construction'],m)
    need(r['model_before']==r['model_after']==m['expected_state'] and r['sources_after']==m['sources'],'entire source/model beforeafter')
    validate_files(r,folder);files=r['files'];referenced={'dispatch_manifest.json'}
    need(read(Path(folder)/'dispatch_manifest.json',files['dispatch_manifest.json']['file_sha256'])==m,'durable predispatch manifest')
    values=select_payloads(m['delivery_artifact']);template=input_rows(m['delivery_artifact']);checks=[]
    need(len(r['rows'])==6,'full six denominator')
    def ref(d):
        need({k:d[k] for k in ('file','file_bytes','file_sha256','root')}==files[d['file']],'file descriptor inventory join')
        referenced.add(d['file'])
    for i,(row,initial) in enumerate(zip(r['rows'],template,strict=True)):
        for k in ('slot','root','arm','condition','missing_reason'):need(row[k]==initial[k],'original slot identity/missing reason')
        if i not in ACTIVE:
            need(row==initial,'arbitrary untouched missing with zero dispatch/debit');continue
        need(row['status']=='returned' and row['trace_completed'] is True and 0<=row['wall_seconds']<=300,'all four returned bounded')
        need(row['dispatch_model_before']==row['dispatch_model_after']==m['expected_state'],'per-slot425 guard')
        need(set(row['input_files'])=={'z','a','target'},'exact persisted payload allowlist')
        for k,a in values[i].items():
            d=row['input_files'][k];ref(d)
            need(d['file']==f'slot{i}_input_{k}.npy' and d['root']==row['root'] and delivery.same(a,load_array(folder,d)),'exact delivered payload persisted')
        td=row['trace'];ref(td)
        need(td['file']==f'slot{i}_trace.json' and td['root']==row['root'],'unique slot trace')
        cr=read(Path(folder)/td['file'],td['file_sha256'])
        need(cr['backend']=='host_cuda_installed_CEM' and cr.get('host_tensor_seam_checked') is True,'no RAM promotion')
        need(cr['spec']==protocol()['spec'] and cr['model_before']==cr['model_after']==m['expected_state'],'actual fixed core spec/model')
        need(all(m['sources'].get(p)==h for p,h in cr['sources_before'].items()),'core closure presealed')
        arrays={}
        for key,d in row['arrays'].items():
            ref(d);need(d['file']==f'slot{i}_{key}.npy' and d['root']==row['root'],'no cross-slot dedup/reuse')
            arrays[key]=load_array(folder,d)
        for k,a in values[i].items():need(delivery.same(a,arrays[cr['inputs'][k]]),'controller receives exact onlypayload')
        c=core.validate_saved(cr,arrays);need(c==row['local_check'],'full reconstructed local_check')
        need(row['debits']==dict(known=True,predict_API=cr['predict_API_debited'],predictor_samples=cr['predictor_samples_debited']) and
             cr['predict_API_debited']==12 and cr['predictor_samples_debited']==54 and row['core_errors']==cr['errors']==[],'all paid per-slot budget axes')
        checks.append(dict(slot=i,root=row['root'],arm=row['arm'],local_check=c))
    need(referenced==set(files),'no unconsumed files/arrays')
    need(totals(r['rows'])=={k:r[k] for k in totals(r['rows'])} and r['actual_predict_API_debited']==48 and
         r['actual_predictor_samples_debited']==216,'all four actual debits')
    gpu=r['cuda_resources'];need(gpu['dispatch_guard_calls']>0 and 0<gpu['max_allocated']<=ALLOCATED and
        0<gpu['max_reserved']<=RESERVED and gpu['allocated_cap']==ALLOCATED and gpu['reserved_cap']==RESERVED,'actual CUDA peak guards')
    need(0<=r['whole_wall_seconds']<=WALL,'whole1800 scope')
    return dict(schema=TAG+'.receipt',status='all4_target_controller_saved_verified_2_arbitrary_missing',
        denominator=dict(total=6,returned=4,missing=2),local_checks=checks,predict_API=48,predictor_samples=216,
        encoder=0,high=0,AD=0,physics=0,FLOPs=None,raw_inverse=False,scientific_admission=False,
        time_matched=False,fair_baseline=False,equal_generation_budget=False,independent_gradient_recompute=False)


def consume(report,h,output):
    r=read(report,h);m=verify(r['manifest'],r['manifest_sha256']);c=analyze_saved(m,r,Path(report).parent)
    c.update(report_sha256=h,manifest_sha256=r['manifest_sha256']);p=own_output(output);b=encode(c)
    need(len(b)<=32*1024**2 and shutil.disk_usage(p.parent).free>=FLOOR+len(b),'receipt cap/floor')
    tracker.durable(p,b);return dict(path=str(p.resolve()),sha256=sha(p))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('protocol','prepare','verify','run','consume'))
    for k in ('manifest','report','output','delivery'):p.add_argument('--'+k,type=Path)
    for k in ('manifest','report','delivery'):p.add_argument('--'+k+'-sha256')
    p.add_argument('--main-authorized',action='store_true');a=p.parse_args(argv)
    if a.command=='protocol':out=protocol()
    elif a.command=='prepare':out=prepare(a.output,a.delivery or DELIVERY_PATH,a.delivery_sha256 or DELIVERY_SHA)
    elif a.command=='verify':out=dict(verified=bool(verify(a.manifest,a.manifest_sha256)))
    elif a.command=='run':out=run_cuda(a.manifest,a.manifest_sha256,a.output,main_authorized=a.main_authorized)
    else:out=consume(a.report,a.report_sha256,a.output)
    print(json.dumps(out,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
