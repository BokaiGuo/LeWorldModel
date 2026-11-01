"""Four-slot, eight-arm real two-packet host; MAIN CUDA only."""
import argparse
from copy import deepcopy
import inspect
import os
from pathlib import Path
import shutil
import time
from experiments.spectral_reachability import agent_p1_two_packet_uphill_generator_20261029_v1 as method
from experiments.spectral_reachability import agent_p1_two_packet_winner_authority_20261029_v1 as authority
from experiments.spectral_reachability import agent_p1_two_packet_native_audit_20261029_v1 as audit
from experiments.spectral_reachability import agent_nonlinear_suffix_observed_recompile_20261028_v2 as old
from experiments.spectral_reachability import p1_scalar_objective_audit_20261029 as scalar_audit

NAME='agent_p1_two_packet_host_cuda_20261029_v1'
AUTH='MAIN_P1_TWO_PACKET_HOST_CUDA_ONLY'
ROOT=method.F.ROOT
F,core,signal,transport=method.F,method.core,method.F.signal,method.F.transport
require,sha,digest,raw=F.require,F.sha,F.digest,F.raw
PINS=((method,'ec5c8698858f4c41cafa24ebde4170c94a3e50325c6f6b9922791a3e0754b6d4'),
      (authority,'a910df6135edd00b1dfce8df8a7bf7e93c1712c6cddde5381945c821d8e83054'),
      (audit,'6838bde770d514acc9dbae6159f01ef846a9e2acd6cdea2a0669a607c4b9d7d6'),
      (F,'0a3e475af9271a0769a3829b4e714fa4e2d6f00eaed38ec8c3ea5d7b9a163fbc'),
      (old,'e66bac861fce455ab6bb7a0cf12ba1644258c6c2e9cb6f2905973ac146e33567'))
SCALAR_SOURCE_SHA='5ed4a9b66563e7e90c3bb74294931028c37c2e2b81fc71783833f7aade63340e'
SCALAR_TEST_SHA='e8898d5b9d3c43a858cc1aff22457973483f34b774cefb8bd97a5d92a150cecf'


def contract():
    return dict(schema=NAME+'.contract',method=method.contract(),per_arm=F.caps(36,16),total=F.caps(288,128),
        fixed_slots=4,fixed_arms=8,fixed_candidate_denominator=72,stop_on_first_failure=True,
        output_bytes=512*1024**2,final_reserve_bytes=64*1024**2,disk_floor_bytes=200*1024**2,
        wall_seconds=3600,cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        training=False,physics=False,physical_release=False,P1_admission=False,freshness=False,FLOPs=None)


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build(source_output,source_report_sha):
    for module,h in PINS:require(sha(module.__file__)==h,'frozen method/authority/audit source')
    scalar_source=Path(scalar_audit.__file__).resolve();scalar_test=ROOT/'tests/test_p1_scalar_objective_audit_20261029.py'
    require(sha(scalar_source)==SCALAR_SOURCE_SHA and sha(scalar_test)==SCALAR_TEST_SHA,'independent scalar audit source/test pin')
    require(Path(source_output).is_absolute() and Path(source_output).resolve()==Path(source_output),'absolute actual source')
    a=authority.material(source_output,source_report_sha);bindings=deepcopy(a['bindings'])
    bindings[str(scalar_source)]=SCALAR_SOURCE_SHA;bindings[str(scalar_test)]=SCALAR_TEST_SHA
    for module,h in PINS:
        path=Path(module.__file__).resolve();bindings[str(path)]=h;stem=module.__name__.rsplit('.',1)[-1]
        for f in (ROOT/'tests'/('test_'+stem+'.py'),ROOT/'innerdocs'/(stem+'.md')):bindings[str(f)]=sha(f)
    for f in own_files():bindings[str(f)]=sha(f)
    for v in F.installed_class_sources().values():
        require(v['path'] not in bindings or bindings[v['path']]==v['sha256'],'installed class conflict');bindings[v['path']]=v['sha256']
    F.check_bindings(bindings)
    return dict(schema=NAME+'.prepare',arguments=dict(source_output=source_output,source_report_sha=source_report_sha),
        contract=contract(),authority=a,bindings=bindings,production_ready=True,P1_admission=False,physical_release=False)


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'actual own prepare SHA');p=signal.read(path,128*1024**2)
    require(raw(p)==raw(build(**p['arguments'])) and sha(path)==expected,'full original strict and source rebuild');return p


def write(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and path.resolve()==path
        and not path.exists() and not path.is_symlink() and len(payload)<=128*1024**2,'exclusive bounded metadata')
    require(shutil.disk_usage(path.parent).free>=contract()['disk_floor_bytes']+len(payload),'disk floor')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path))


class Store(F.Store):
    def __init__(self,path,started):
        self.path=Path(path).absolute();self.started=started;self.finalizing=False
        require(self.path.parent==ROOT/'results' and self.path.name.startswith(NAME+'_') and self.path.resolve()==self.path
            and not self.path.exists() and shutil.disk_usage(self.path.parent).free>=contract()['output_bytes']+contract()['disk_floor_bytes'],'own output/floor')
        self.path.mkdir();signal.fsync_dir(self.path.parent)
    def check(self,size=4096):
        require(self.finalizing or time.monotonic()-self.started<contract()['wall_seconds'],'whole watchdog')
        used=sum(f.stat().st_size for f in self.path.rglob('*') if f.is_file())
        require(used+size<=contract()['output_bytes']-(0 if self.finalizing else contract()['final_reserve_bytes'])
            and shutil.disk_usage(self.path).free>=contract()['disk_floor_bytes']+size,'output cap/reserve/floor')


def roster():
    return [dict(index=i,arms=[dict(arm=a,status='not_started',result=None,resources=None) for a in method.ARMS]) for i in range(4)]


def dispatch(p,store,rows,check,factory):
    specs=p['authority']['rows'];require(p['production_ready'] is True and len(specs)==len(rows)==4,'fixed four slots')
    for i,(spec,row) in enumerate(zip(specs,rows,strict=True)):
        q=spec['request'];method.request(q);require(q['slot']==row['index']==i,'fixed slot identity')
        for j,(arm,entry) in enumerate(zip(method.ARMS,row['arms'],strict=True)):
            require(entry['arm']==arm,'fixed arm order');entry['status']='attempted';ns=f'r{i}_a{j}'
            try:
                journal=transport.NativeJournal(store,ns,check)
            except BaseException as exc:
                entry.update(status='failed_before_journal',error=repr(exc))
                store.put(ns+'_result.json',entry)
                raise
            ledger=core.Ledger(F.caps(36,16),sink=journal.event)
            try:
                b=factory(q['task'],journal,ledger)
                with b.observe():entry['result']=method.run_arm(q,arm,audit.LabelAdapter(b),journal.event)
                require(ledger.attempted==ledger.completed==F.caps(36,16),'complete paid arm')
                if j:
                    first=row['arms'][0]['result'];second=entry['result']
                    require(first['initial_Q']==second['initial_Q'] and first['original_winner_terminal_Q']==second['original_winner_terminal_Q']
                        and raw(first['iterates'][0])==raw(second['iterates'][0]),'same original winner and context across arms')
                entry['status']='sealed'
            finally:
                entry['resources']=ledger.receipt();journal.close()
                entry.update(journal_file=ns+'/events.jsonl',journal_sha256=sha(journal.path),journal_tail=journal.tail)
                if entry['status']!='sealed':entry['status']='failed_partial'
                store.put(ns+'_result.json',entry)


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit CUDA authorization')
    started=time.monotonic();p=verify(prepared,expected);store=Store(output,started);rows=roster()
    errors=[];model=before=after=host=pb=pa=peaks=None;flags=None;source_ok=False
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
            store.check();old.precision(torch)
            require(torch.cuda.memory_allocated(device)<=contract()['cuda_allocated_bytes'] and torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'GPU cap')
        with torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            check();model=load_compatible_pretrained(p['authority']['policy'],device=device).eval().requires_grad_(False);check()
            before=signal.model_hash(model);require(before['original_state_dict']==p['authority']['expected_model_hash'],'frozen checkpoint identity')
            for module in model.modules():require(str(Path(inspect.getfile(type(module))).resolve()) in p['bindings'],'installed model class source')
            pb=old.precision(torch)
            dispatch(p,store,rows,check,lambda task,journal,ledger:old.Backend(model,device,task,journal,ledger,check))
            pa=old.precision(torch);torch.cuda.synchronize(device);check()
    except BaseException as exc:errors.append(dict(stage='run',error=repr(exc)))
    finally:
        store.finalizing=True
        try:
            if model is not None:
                old.midpoint.bridge.native.frozen(model);after=signal.model_hash(model);require(before==after,'all model state unchanged')
                peaks=[torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)]
            F.check_bindings(p['bindings']);require(sha(prepared)==expected,'source and prepare closure');source_ok=True
        except BaseException as exc:errors.append(dict(stage='closure',error=repr(exc)))
        wall=time.monotonic()-started
        if wall>=contract()['wall_seconds']:errors.append(dict(stage='wall',error='whole-stage cap'))
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        complete=not errors and source_ok and before is not None and before==after and all(a['status']=='sealed' for r in rows for a in r['arms'])
        artifacts={str(f.relative_to(store.path)):sha(f) for f in store.path.rglob('*') if f.is_file()}
        report=dict(schema=NAME+'.report',status='complete_model_only' if complete else 'partial_failclosed',contract=contract(),
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,errors=errors,fixed_denominator=4,arm_denominator=8,
            candidate_denominator=72,source_verified=source_ok,source_manifest_sha256=digest(p['bindings']),model_before=before,model_after=after,
            host=host,precision_before=pb,precision_after=pa,CUDA_peak_bytes=peaks,wall_seconds=wall,artifact_sha256=artifacts,
            P1_admission=False,freshness=False,physical_release=False)
        store.put('report.json',report,final=True)
    return dict(status=report['status'],report_sha256=sha(store.path/'report.json'),output=str(store.path),errors=errors)


def consume(output,expected):
    output=Path(output).absolute();core.hash_value(expected);require(sha(output/'report.json')==expected,'report SHA')
    r=signal.read(output/'report.json',128*1024**2);p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and raw(r['contract'])==raw(contract()) and len(r['rows'])==r['fixed_denominator']==4
        and r['arm_denominator']==8 and r['candidate_denominator']==72 and all(r[k] is False for k in ('P1_admission','freshness','physical_release')),'full denominators and gates')
    require(r['status'] in ('complete_model_only','partial_failclosed') and type(r['errors']) is list,'known disposition')
    require(signal.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'actual execution binding')
    files={str(f.relative_to(output)) for f in output.rglob('*') if f.is_file()}
    require(files==set(r['artifact_sha256'])|{'report.json'} and sum(f.stat().st_size for f in output.rglob('*') if f.is_file())<=contract()['output_bytes'],'complete file census')
    for name,h in r['artifact_sha256'].items():
        path=Path(name);require(not path.is_absolute() and all(x not in ('.','..') for x in path.parts)
            and (output/path).resolve()==output/path and sha(output/path)==h,'exact artifact SHA/path')
    receipts=[];audits=[];stopped=False
    for i,(spec,row) in enumerate(zip(p['authority']['rows'],r['rows'],strict=True)):
        require(type(row['index']) is int and row['index']==i==spec['request']['slot'] and len(row['arms'])==2,'fixed slot')
        aa=[]
        for j,(arm,entry) in enumerate(zip(method.ARMS,row['arms'],strict=True)):
            require(entry['arm']==arm,'fixed arm')
            if entry['resources'] is None:
                require(entry['result'] is None,'no result without ledger')
                if entry['status']=='failed_before_journal':
                    require(not stopped and signal.read(output/f'r{i}_a{j}_result.json',32*1024**2)==entry
                        and type(entry.get('error')) is str,'durable journal creation failure')
                else:require(entry['status']=='not_started','unreached null')
                stopped=True;aa.append(None);continue
            require(not stopped and signal.read(output/f'r{i}_a{j}_result.json',32*1024**2)==entry,'durable ordered arm')
            events=F.load_journal(output,entry);transport.audit_resources(entry['resources'],events);receipts.append(entry['resources'])
            if entry['status']=='sealed':
                checked=audit.audit(spec['request'],entry['result'],entry['resources'],events)
                checked['scalar_objective']=scalar_audit.audit_arm(spec['request']['task'],entry['result'])
                aa.append(checked)
            else:require(entry['status']=='failed_partial','failed arm retained');stopped=True;aa.append(None)
        if all(v is not None for v in aa):
            x,y=(e['result'] for e in row['arms'])
            require(x['initial_Q']==y['initial_Q'] and x['original_winner_terminal_Q']==y['original_winner_terminal_Q']
                and raw(x['iterates'][0])==raw(y['iterates'][0]),'same-context same-winner paired controls')
        audits.append(aa)
    counts={kind:{k:sum(x[kind][k] for x in receipts) for k in core.AXES} for kind in ('attempted','completed')}
    complete=r['status']=='complete_model_only' and not r['errors'] and all(v is not None for aa in audits for v in aa)
    require((r['status']=='complete_model_only')==complete,'complete disposition is evidenced')
    if complete:
        require(counts['attempted']==counts['completed']==F.caps(288,128),'all paid native resources')
        require(r['source_verified'] is True and r['model_before']==r['model_after'] and r['model_before']['original_state_dict']==p['authority']['expected_model_hash']
            and r['source_manifest_sha256']==digest(p['bindings']),'frozen model/source closure')
        precision=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)
        require(r['precision_before']==r['precision_after']==precision,'precision closure')
        require(0<=old.scalar(r['wall_seconds'],'wall')<contract()['wall_seconds'] and len(r['CUDA_peak_bytes'])==2
            and all(type(x) is int and 0<=x<=contract()[k] for x,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)),'actual wall/GPU caps')
    F.check_bindings(p['bindings']);require(sha(output/'report.json')==expected,'final report unchanged')
    return dict(schema=NAME+'.consume',status='verified_model_only' if complete else 'partial_failclosed',complete=complete,
        report_sha256=expected,fixed_denominator=4,arm_denominator=8,candidate_denominator=72,rows=r['rows'],audits=audits,
        actual_new_compute=counts,source_report_sha256=p['arguments']['source_report_sha'],P1_admission=False,freshness=False,physical_release=False,FLOPs=None)


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('source-output','source-report-sha','output','prepared','prepare-sha','report-sha','receipt','authorization'):q.add_argument('--'+k)
    a=q.parse_args()
    if a.command=='prepare':v=write(a.output,build(a.source_output,a.source_report_sha))
    elif a.command=='verify':v=dict(verified=True,production_ready=verify(a.prepared,a.prepare_sha)['production_ready'])
    elif a.command=='run':v=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),flush=True)


if __name__=='__main__':main()
