"""Four-slot actual-cp5 midpoint compiler host. MAIN CUDA only."""
import argparse
from copy import deepcopy
import inspect
import os
from pathlib import Path
import shutil
import time

from experiments.spectral_reachability import agent_p1_midpoint_monotone_compiler_20261029_v1 as method
from experiments.spectral_reachability import agent_p1_midpoint_host_audit_20261029_v1 as row_audit

original=method.original
F,core,signal,transport=original.F,original.core,original.signal,original.transport
ROOT=original.ROOT
require,sha,digest,raw=original.require,original.sha,original.digest,original.raw
NAME='agent_p1_midpoint_host_cuda_20261029_v1'
AUTH='MAIN_P1_MIDPOINT_HOST_CUDA_ONLY'
PINS=((method,'4e14f0131a7d1219843653134cd9812b1e45f999f31163ac89b4abd5d6912883'),
      (row_audit,'d6e95f824be3d558831915967da814258e07df6621b7bd11321046813812009f'),
      (original,'e66bac861fce455ab6bb7a0cf12ba1644258c6c2e9cb6f2905973ac146e33567'))


def contract():
    return dict(schema=NAME+'.contract',method=method.contract(),per_slot=F.caps(26,8),total=F.caps(104,32),
                wall_seconds=1800,output_bytes=256*1024**2,final_reserve_bytes=32*1024**2,
                disk_floor_bytes=200*1024**2,cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
                fixed_denominator=4,known_roots=2,stop_on_first_failure=True,
                training=False,physical_release=False,P1_admission=False,freshness=False,total_FLOPs=None)


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build(midpoint_output,midpoint_report_sha,midpoint_receipt,midpoint_receipt_sha):
    for module,h in PINS:require(sha(module.__file__)==h,'frozen method/source pin')
    p=original.build(midpoint_output,midpoint_report_sha,midpoint_receipt,midpoint_receipt_sha)
    bindings=deepcopy(p['bindings'])
    for module,h in PINS:bindings[str(Path(module.__file__).resolve())]=h
    for module,_ in PINS[:2]:
        stem=module.__name__.rsplit('.',1)[-1]
        for f in (ROOT/'tests'/('test_'+stem+'.py'),ROOT/'innerdocs'/(stem+'.md')):
            bindings[str(f)]=sha(f)
    for f in own_files():bindings[str(f)]=sha(f)
    F.check_bindings(bindings)
    return dict(schema=NAME+'.prepare',arguments=p['arguments'],contract=contract(),original_prepare=p,
                bindings=bindings,production_ready=p['production_ready'],P1_admission=False,physical_release=False)


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'own prepare SHA')
    p=signal.read(path,64*1024**2)
    require(raw(p)==raw(build(**p['arguments'])) and sha(path)==expected,'complete source/authority rebuild')
    return p


def write(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and path.resolve()==path
            and not path.exists() and not path.is_symlink() and len(payload)<=64*1024**2,'exclusive bounded metadata')
    require(shutil.disk_usage(path.parent).free>=contract()['disk_floor_bytes']+len(payload),'disk floor')
    with path.open('xb') as f:
        require(f.write(payload)==len(payload),'complete write');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(path.parent)
    return dict(path=str(path),sha256=sha(path))


class Store(F.Store):
    def __init__(self,path,started):
        self.path=Path(path).absolute();self.started=started;self.finalizing=False
        require(self.path.parent==ROOT/'results' and self.path.name.startswith(NAME+'_') and self.path.resolve()==self.path
                and not self.path.exists() and shutil.disk_usage(self.path.parent).free>=contract()['output_bytes']+contract()['disk_floor_bytes'], 'exclusive output/floor')
        self.path.mkdir();signal.fsync_dir(self.path.parent)
    def check(self,size=4096):
        require(self.finalizing or time.monotonic()-self.started<contract()['wall_seconds'],'wall watchdog')
        used=sum(f.stat().st_size for f in self.path.rglob('*') if f.is_file())
        require(used+size<=contract()['output_bytes']-(0 if self.finalizing else contract()['final_reserve_bytes'])
                and shutil.disk_usage(self.path).free>=contract()['disk_floor_bytes']+size,'output/final reserve/floor')


def roster():return [dict(index=i,status='not_started',result=None,resources=None) for i in range(4)]


def dispatch(p,store,rows,check,factory):
    specs=p['original_prepare']['rows']
    require(p['production_ready'] is True and len(specs)==len(rows)==4,'all four ready')
    for spec,row in zip(specs,rows,strict=True):
        require(spec['index']==row['index'] and spec['status']=='ready','slot order')
        row['status']='attempted';ns=f'r{row["index"]}'
        journal=transport.NativeJournal(store,ns,check);ledger=core.Ledger(F.caps(26,8),sink=journal.event)
        try:
            b=factory(spec['request']['task'],journal,ledger)
            with b.observe():row['result']=method.compile_slot(spec,b,journal)
            require(ledger.attempted==ledger.completed==F.caps(26,8),'all26 native calls and 8 reverse')
            row['status']='sealed'
        finally:
            row['resources']=ledger.receipt();journal.close()
            row.update(journal_file=ns+'/events.jsonl',journal_sha256=sha(journal.path),journal_tail=journal.tail)
            if row['status']!='sealed':row['status']='failed_partial'
            store.put(ns+'_result.json',row)


def original_prefix_view(receipt,events):
    """Derived prefix view, never a receipt: whole ledger is checked first."""
    transport.audit_resources(receipt,events)
    boundary=next(i for i,e in enumerate(events) if e['kind']=='compile_return')
    prefix=events[:boundary+1]
    ops=[deepcopy(e) for e in receipt['events'] if int(e['scope']['forward'][1:])<18]
    require([e['index'] for e in ops]==list(range(len(ops))) and all(e['status']=='returned' for e in ops),'actual complete original ledger prefix')
    inclusive={}
    for e in ops:inclusive[e['resource']]=inclusive.get(e['resource'],0)+e['ended_ns']-e['started_ns']
    view=deepcopy(receipt);view.update(caps=F.caps(18,8),attempted=F.caps(18,8),completed=F.caps(18,8),events=ops,inclusive_operation_wall_ns=inclusive)
    return view,prefix


def audit_row(spec,row,events):
    result=row['result'];receipt=row['resources']
    out=row_audit.audit(spec,result,receipt,events)
    view,prefix=original_prefix_view(receipt,events)
    original.audit_row(spec,result['original'],view,prefix)
    return out


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit HOST CUDA authorization')
    started=time.monotonic();p=verify(prepared,expected);require(p['production_ready'] is True,'complete actual cp5')
    store=Store(output,started);rows=roster();errors=[];model=before=after=host=pb=pa=peaks=None;flags=None;source_ok=False
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def check():
            store.check();original.precision(torch)
            require(torch.cuda.memory_allocated(device)<=contract()['cuda_allocated_bytes'] and torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'GPU caps')
        with torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            check();model=load_compatible_pretrained(p['original_prepare']['policy'],device=device).eval().requires_grad_(False);check()
            before=signal.model_hash(model);require(before['original_state_dict']==p['original_prepare']['expected_model_hash'],'original model')
            for module in model.modules():require(str(Path(inspect.getfile(type(module))).resolve()) in p['bindings'],'loaded class source sealed')
            pb=original.precision(torch)
            dispatch(p,store,rows,check,lambda task,journal,ledger:original.Backend(model,device,task,journal,ledger,check))
            pa=original.precision(torch);torch.cuda.synchronize(device);check()
    except BaseException as exc:errors.append(dict(stage='run',error=repr(exc)))
    finally:
        store.finalizing=True
        try:
            if model is not None:
                original.midpoint.bridge.native.frozen(model);after=signal.model_hash(model);require(before==after,'model unchanged')
                peaks=[torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)]
            F.check_bindings(p['bindings']);require(sha(prepared)==expected,'source and prepare unchanged');source_ok=True
        except BaseException as exc:errors.append(dict(stage='closure',error=repr(exc)))
        wall=time.monotonic()-started
        if wall>=contract()['wall_seconds']:errors.append(dict(stage='wall',error='whole-stage cap'))
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        complete=not errors and source_ok and before is not None and before==after and all(r['status']=='sealed' for r in rows)
        artifacts={str(f.relative_to(store.path)):sha(f) for f in store.path.rglob('*') if f.is_file()}
        report=dict(schema=NAME+'.report',status='complete_model_only' if complete else 'partial_failclosed',contract=contract(),
                    prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,errors=errors,fixed_denominator=4,
                    source_verified=source_ok,source_manifest_sha256=digest(p['bindings']),model_before=before,model_after=after,
                    host=host,precision_before=pb,precision_after=pa,CUDA_peak_bytes=peaks,wall_seconds=wall,artifact_sha256=artifacts,
                    P1_admission=False,freshness=False,physical_release=False)
        store.put('report.json',report,final=True)
    return dict(status=report['status'],report_sha256=sha(store.path/'report.json'),output=str(store.path),errors=errors)


def consume(output,expected):
    output=Path(output).absolute();core.hash_value(expected);require(sha(output/'report.json')==expected,'report hash')
    r=signal.read(output/'report.json',64*1024**2);p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and raw(r['contract'])==raw(contract()) and len(r['rows'])==r['fixed_denominator']==4
            and all(r[k] is False for k in ('P1_admission','freshness','physical_release')),'report contract/gates')
    require(r['status'] in ('complete_model_only','partial_failclosed') and type(r['errors']) is list,'known report disposition')
    require(signal.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'execution binding')
    files={str(f.relative_to(output)) for f in output.rglob('*') if f.is_file()}
    require(files==set(r['artifact_sha256'])|{'report.json'} and sum(f.stat().st_size for f in output.rglob('*') if f.is_file())<=contract()['output_bytes'],'complete file census')
    for name,h in r['artifact_sha256'].items():
        path=Path(name);require(not path.is_absolute() and all(v not in ('.','..') for v in path.parts)
             and (output/path).resolve()==output/path and sha(output/path)==h,'artifact hash/path')
    receipts=[];audits=[];stopped=False
    for i,(spec,row) in enumerate(zip(p['original_prepare']['rows'],r['rows'],strict=True)):
        require(type(row['index']) is int and row['index']==i==spec['index'],'four-slot ordering')
        if row['resources'] is None:
            require(row['status']=='not_started' and row['result'] is None,'unstarted null');stopped=True;audits.append(None);continue
        require(not stopped,'no slot after failure or unreached slot')
        require(signal.read(output/f'r{i}_result.json',32*1024**2)==row,'persisted slot')
        events=F.load_journal(output,row);transport.audit_resources(row['resources'],events);receipts.append(row['resources'])
        if row['status']=='sealed':audits.append(audit_row(spec,row,events))
        else:
            require(row['status']=='failed_partial','failed partial retained without release');stopped=True;audits.append(None)
    counts={kind:{k:sum(x[kind][k] for x in receipts) for k in core.AXES} for kind in ('attempted','completed')}
    complete=r['status']=='complete_model_only' and not r['errors'] and all(x is not None for x in audits)
    require((r['status']=='complete_model_only')==complete,'complete status must be fully evidenced')
    if complete:
        require(counts['attempted']==counts['completed']==F.caps(104,32),'full combined resource ledger')
        require(r['source_verified'] is True and r['model_before']==r['model_after'] and r['model_before']['original_state_dict']==p['original_prepare']['expected_model_hash']
                and r['source_manifest_sha256']==digest(p['bindings']),'model and source closure')
        precision=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)
        require(r['precision_before']==r['precision_after']==precision,'precision closure')
        require(0<=original.scalar(r['wall_seconds'],'wall')<contract()['wall_seconds'] and len(r['CUDA_peak_bytes'])==2
                and all(type(x) is int and 0<=x<=contract()[k] for x,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)),'wall/GPU caps')
    F.check_bindings(p['bindings']);require(sha(output/'report.json')==expected,'final report identity')
    return dict(schema=NAME+'.consume',status='verified_model_only' if complete else 'partial_failclosed',complete=complete,
                report_sha256=expected,fixed_denominator=4,known_roots=2,rows=r['rows'],audits=audits,actual_new_compute=counts,
                inherited_resources=p['original_prepare']['inherited_resources'],P1_admission=False,freshness=False,physical_release=False,total_FLOPs=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for name in ('midpoint-output','midpoint-report-sha','midpoint-receipt','midpoint-receipt-sha','output','prepared','prepare-sha','report-sha','receipt','authorization'):parser.add_argument('--'+name)
    a=parser.parse_args()
    if a.command=='prepare':result=write(a.output,build(a.midpoint_output,a.midpoint_report_sha,a.midpoint_receipt,a.midpoint_receipt_sha))
    elif a.command=='verify':result=dict(verified=True,production_ready=verify(a.prepared,a.prepare_sha)['production_ready'])
    elif a.command=='run':result=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:result=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(result).decode(),flush=True)


if __name__=='__main__':main()
