"""Controlled H15 HOST candidate; prepare/consume CPU-only, run requires real CUDA."""
import argparse
import os
from pathlib import Path
import resource
import shutil
import time
from copy import deepcopy

from experiments.spectral_reachability import p2_h15_host_lifecycle_20261029_v1 as life
from experiments.spectral_reachability import p2_h15_host_attestation_20261029_v1 as att
from experiments.spectral_reachability import p2_h15_bridge_lifecycle_20261029_v1 as bridge
from experiments.spectral_reachability import agent_p2_intersection_host_cuda_20261029_v2 as historical

NAME='p2_h15_controlled_host_20261029_v1'
AUTH='MAIN_APPROVED_P2_H15_HOST_CUDA'
WALL=3600
CAP=1536*1024**2
FLOOR=512*1024**2
FILE_CAP=128*1024**2
ALLOCATED=4*1024**3
RESERVED=6*1024**3
PREFLIGHT=bridge.source.prep.ROOT/'results/p2_h15_host_attestation_20261029_v1_MAIN01.json'
PREFLIGHT_SHA='6ba2be4a1c364fa79aec5909dd8fdbdfba9c49210d077069b0b697e28dbcc6f5'
need=life.need


def check_extra_sources(pins): return att.verify_sources(pins)


def dispatch_ready(value):
    need(value==dict(authorization=AUTH,preflight_verified=True,limits_installed=True),'dispatch prerequisites')
    return True


class ResourceGuard:
    def __init__(self,path,clock,memory,free,start):
        self.path=Path(path);self.clock=clock;self.memory=memory;self.free=free;self.start=start
    def check(self):
        need(self.clock()-self.start<WALL,'wall cap')
        allocated,reserved=self.memory()
        need(allocated<=ALLOCATED and reserved<=RESERVED,'CUDA peak cap')
        total=0
        for p in self.path.rglob('*'):
            need(not p.is_symlink(),'no output symlinks')
            if p.is_file():total+=p.stat().st_size
        need(total<=CAP and self.free()>=FLOOR,'disk cap/floor')


def extra_sources():
    files=[Path(__file__),Path(life.__file__),Path(att.__file__),Path(bridge.__file__)]
    return {str(p.resolve()):bridge.sha(p) for p in files}


def build():
    old=att.consume_preflight(PREFLIGHT,PREFLIGHT_SHA)
    return dict(schema=NAME+'.prepare',preflight_sha256=PREFLIGHT_SHA,
                bridge_sha256=old['bridge_sha256'],extra_sources=extra_sources(),
                limits=dict(wall_seconds=WALL,output_cap_bytes=CAP,disk_floor_bytes=FLOOR,
                            file_cap_bytes=FILE_CAP,cuda_allocated_bytes=ALLOCATED,cuda_reserved_bytes=RESERVED),
                budget_per_root=life.single.caps(),fixed_denominator=2,
                monitoring='cooperative wall/disk/CUDA peak at native boundaries; per-file RLIMIT_FSIZE',
                physics=False,training=False,scientific_admission=False)


def prepare(path):
    path=Path(path);need(path.name.startswith(NAME+'_'),'exclusive filename')
    life.single._put(path,build());return bridge.sha(path)


def verify(path,digest):
    value=bridge.source.prep.read(Path(path),digest)
    need(value==build(),'prepare source rebuild')
    return value


class LoaderStore:
    def __init__(self,path,guard):
        self.path=path;self.guard=guard;self.events=life.transport.Journal(path/'loader_events.jsonl',8*1024**2)
    def event(self,event):self.guard.check();self.events.emit('loader',event)
    def put(self,name,value):
        need(name=='loader.json','loader file');self.guard.check();life.single._put(self.path/name,value)
    def close(self):self.events.close()


class GuardedSession(att.Session):
    def __init__(self,*args,guard,**kwargs):
        self.guard=guard
        super().__init__(*args,**kwargs)
    def check(self):
        self.guard.check();super().check();self.guard.check()


def run(prepared,expected,output,authorization):
    started=time.monotonic();p=verify(prepared,expected)
    need(authorization==AUTH,'explicit authorization')
    path=Path(output).absolute();need(path.parent==bridge.source.prep.ROOT/'results' and
        path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink(),'exclusive own output')
    need(shutil.disk_usage(path.parent).free>=CAP+FLOOR,'whole output allowance and floor')
    soft,hard=resource.getrlimit(resource.RLIMIT_FSIZE)
    need(hard==resource.RLIM_INFINITY or hard>=FILE_CAP,'file limit available')
    resource.setrlimit(resource.RLIMIT_FSIZE,(FILE_CAP,hard))
    dispatch_ready(dict(authorization=authorization,preflight_verified=True,
                        limits_installed=resource.getrlimit(resource.RLIMIT_FSIZE)[0]==FILE_CAP))
    path.mkdir();fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    state=dict(schema=NAME+'.report',status='failed_partial',root_status=['not_started','not_started'],
               prepared_sha256=expected,errors=[],host=None,slots=[],attestation=None,physical_release=False,
               scientific_admission=False,resource_semantics=p['monitoring'])
    model=None;session=None;store=None;journal=None
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.gpu_preflight import require_cuda_allocation
        state['host']=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.cuda.reset_peak_memory_stats(device)
        guard=ResourceGuard(path,time.monotonic,lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)),
                            lambda:shutil.disk_usage(path).free,started)
        guard.check();store=LoaderStore(path,guard)
        host=bridge.source.prep.read(bridge.source.HOST,bridge.source.HOST_SHA)
        model,_=historical.old.loader.loaded_model(host,torch,device,store,bridge.source.HOST_SHA)
        store.close();store=None
        state['loader_sha256']=bridge.sha(path/'loader.json')
        state['loader_events_sha256']=bridge.sha(path/'loader_events.jsonl')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False),torch.autocast(device_type='cpu',enabled=False):
            before=att.model_hash(model)
            session=GuardedSession(bridge.build_bridge(),model,device,torch,att.historical_inventory(),before,guard=guard)
            check_extra_sources(p['extra_sources'])
            journal=life.transport.Journal(path/'lifecycle.jsonl',1024*1024)
            mapped=session.mapped;transitions=[]
            journal.emit('bridge',dict(sha256=life.core.digest(mapped),backend='HOST_CUDA'))
            initial=dict(model=session.model_before,precision=session.precision_before,sources=session.source_before,fixture_only=False)
            state['attestation_before']=initial;journal.emit('attestation_before',initial)
            for root in (0,1):
                guard.check();check_extra_sources(p['extra_sources']);task=mapped['roots'][root]['task']
                transitions.append(('start',root));journal.emit('start',dict(root=root,task_sha256=life.core.digest(task)))
                state['root_status']=bridge.source.lifecycle(transitions)
                try:
                    slot,digest=life._slot(path/('root'+str(root)),task,lambda t,s:life.GuardedNative(session,root,s),session.check,'HOST_CUDA')
                    life._audit_slot(path/('root'+str(root)),digest,task,'HOST_CUDA')
                    action='seal' if slot['status']=='complete' else 'fail'
                    item=dict(root=root,status=slot['status'],report_sha256=digest)
                except Exception as exc:
                    action='fail';item=dict(root=root,status='outer_failure',report_sha256=None,error=dict(type=type(exc).__name__,message=str(exc)[:2048]))
                transitions.append((action,root));state['slots'].append(item);state['root_status']=bridge.source.lifecycle(transitions)
                journal.emit(action,item)
                if action=='fail':break
            state['attestation']=session.close();check_extra_sources(p['extra_sources']);guard.check()
            journal.emit('attestation',dict(value=state['attestation'],error=None))
            state['peak']=dict(allocated=torch.cuda.max_memory_allocated(device),reserved=torch.cuda.max_memory_reserved(device))
            if state['root_status']==['sealed','sealed']:state['status']='recorded_not_admitted'
    except BaseException as exc:
        state['errors'].append(dict(type=type(exc).__name__,message=str(exc)[:2048]))
    finally:
        if store is not None:store.close()
        if journal is not None:journal.close();state['lifecycle_sha256']=bridge.sha(path/'lifecycle.jsonl')
        state['elapsed_seconds']=time.monotonic()-started
        life.single._put(path/'report.json',state)
    return dict(report=str(path/'report.json'),sha256=bridge.sha(path/'report.json'),status=state['status'])


def consume(directory,report_sha,prepared,prepare_sha):
    p=verify(prepared,prepare_sha);path=Path(directory)
    need(path.is_dir() and not path.is_symlink(),'saved output directory')
    r=bridge.source.prep.read(path/'report.json',report_sha)
    need(r['schema']==NAME+'.report' and r['prepared_sha256']==prepare_sha and
         r['physical_release'] is False and r['scientific_admission'] is False,'saved report identity')
    need(r['status']=='recorded_not_admitted' and not r['errors'] and r['root_status']==['sealed','sealed'],
         'partial execution cannot certify completion')
    host=bridge.source.prep.read(bridge.source.HOST,bridge.source.HOST_SHA)
    loader=bridge.source.prep.read(path/'loader.json',r['loader_sha256'])
    historical.old.loader.validate_loader(loader,host,bridge.source.HOST_SHA)
    need(bridge.sha(path/'loader_events.jsonl')==r['loader_events_sha256'],'loader event SHA')
    life.transport.read_journal(path/'loader_events.jsonl')
    need(bridge.sha(path/'lifecycle.jsonl')==r['lifecycle_sha256'],'lifecycle SHA')
    events=life.transport.read_journal(path/'lifecycle.jsonl');mapped=bridge.build_bridge()
    expected_events=[('bridge',dict(sha256=life.core.digest(mapped),backend='HOST_CUDA')),
                     ('attestation_before',r['attestation_before'])]
    need(len(r['slots'])==2,'fixed denominator')
    resources=[]
    for i,slot in enumerate(r['slots']):
        task=mapped['roots'][i]['task']
        need(slot['root']==i and slot['status']=='complete','complete root ordering')
        saved=life._audit_slot(path/('root'+str(i)),slot['report_sha256'],task,'HOST_CUDA')
        need(saved['status']=='complete','full native slot audit');resources.append(saved['resources'])
        expected_events.extend([('start',dict(root=i,task_sha256=life.core.digest(task))),('seal',slot)])
    a=r['attestation'];b=r['attestation_before']
    need(a is not None and a['fixture_only'] is False and a['physical_release'] is False and
         b['model']==a['model_before']==a['model_after'] and
         a['model_before']['original_state_dict']==host['expected_model_hash'] and
         b['precision']==a['precision_before']==a['precision_after']==att.native.precision_contract() and
         b['sources']==a['sources_before']==a['sources_after']==att.verify_sources(att.historical_inventory()),
         'model precision source closure')
    need(r['peak']['allocated']<=ALLOCATED and r['peak']['reserved']<=RESERVED and r['elapsed_seconds']<WALL,
         'recorded runtime limits')
    expected_events.append(('attestation',dict(value=a,error=None)))
    need([(e['kind'],e['payload']) for e in events]==expected_events,'journal sequence')
    return dict(schema=NAME+'.consume',status='saved_native_checked_not_admitted',fixed_denominator=2,
                resources=resources,physical_release=False,scientific_admission=False,
                limitations='saved traces cannot prove absence of uninstrumented calls or authenticate execution host')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run','consume'))
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--sha256');p.add_argument('--output',type=Path);p.add_argument('--authorization');p.add_argument('--report-sha256')
    a=p.parse_args()
    if a.command=='prepare':need(a.sha256 is None and a.output is None,'prepare arguments');result=dict(sha256=prepare(a.prepared))
    elif a.command=='verify':need(a.sha256 and a.output is None,'verify arguments');verify(a.prepared,a.sha256);result=dict(verified=True)
    elif a.command=='run':need(a.sha256 and a.output and a.report_sha256 is None,'run arguments');result=run(a.prepared,a.sha256,a.output,a.authorization)
    else:need(a.sha256 and a.output and a.report_sha256,'consume arguments');result=consume(a.output,a.report_sha256,a.prepared,a.sha256)
    import json
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
