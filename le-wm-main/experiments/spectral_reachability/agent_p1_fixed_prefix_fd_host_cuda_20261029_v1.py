"""Fixed 64-group FD host; MAIN-authorized CUDA and saved-only consume."""
import argparse
from copy import deepcopy
import inspect
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p1_fixed_prefix_fd_20261029_v1 as fd
from experiments.spectral_reachability import agent_p1_fixed_prefix_fd_host_component_20261029_v1 as audit
from experiments.spectral_reachability import agent_p1_fixed_prefix_saved_handoff_20261029_v1 as authority
from experiments.spectral_reachability import agent_p1_fixed_prefix_host_cuda_20261029_v1 as previous

NAME='agent_p1_fixed_prefix_fd_host_cuda_20261029_v1'
AUTH='MAIN_P1_FIXED_PREFIX_FD_HOST_CUDA_ONLY'
ROOT=previous.ROOT
F,core,signal,transport,old=previous.F,previous.core,previous.signal,previous.transport,previous.old
require,sha,raw,digest=F.require,F.sha,F.raw,F.digest
PINS=((fd,'5647d825986f37f701b30e0d4f821cf9f96efe698e98782cbba4d5d9591a68f2'),
      (audit,'af08ab2e534b66cc8e63bc7c178dc279454d1165e12907b723d93f0370be661a'),
      (authority,'0b9442da65a96e9a8aa1eb467d30ba660747122cbc977144e34f1e4fb887a106'),
      (previous,'fa11dd40ce0e2c437306101fcd707b5d3ae5c2a131ecbf84529df85bf6cb9e94'))


def contract():
    return dict(schema=NAME+'.contract',method=fd.contract(),group_denominator=64,pair_denominator=2560,
        max_new_compute=F.caps(5120,0),inherited_source=F.caps(1216,512),stop_on_first_failure=True,
        output_bytes=8*1024**3,final_reserve_bytes=512*1024**2,disk_floor_bytes=200*1024**2,
        wall_seconds=14400,cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        wall_scope='prepare verification through pre-report sample; report fsync excluded',
        cap_is_fit_guarantee=False,FLOPs=None,training=False,physics=False,P1_admission=False,freshness=False,physical_release=False)


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def invalid_causes(q,result):
    """Independently expose every applicable reason without changing the frozen kernel."""
    pairs=fd.pairs(q,result); x=np.asarray(result['iterates'][0]['logits'],np.float32)
    a=F.chart(x).reshape(10);out=[]
    for i,p in enumerate(pairs):
        if p['valid']:continue
        base=a if p['space']=='raw' else x;j=p['coordinate'];h=p['requested_step']
        hi=base.copy();lo=base.copy();hi[j]=np.float32(float(base[j])+h);lo[j]=np.float32(float(base[j])-h)
        reasons=[]
        if p['plus_step']<=0 or p['minus_step']<=0:reasons.append('input_float32_collapse')
        if p['space']=='raw' and (float(lo[j]) < -1 or float(hi[j]) > 1):reasons.append('raw_domain')
        if p['space']=='chart_logit' and (p['plus_norm']>F.CAP or p['minus_norm']>F.CAP):reasons.append('chart_ball_boundary')
        plus=hi.reshape(5,2) if p['space']=='raw' else F.chart(hi)
        minus=lo.reshape(5,2) if p['space']=='raw' else F.chart(lo)
        if F.bits(plus)==F.bits(minus):reasons.append('output_float32_collapse')
        require(reasons,'every invalid pair has explicit reason')
        out.append(dict(pair=i,space=p['space'],coordinate=j,requested_step=h,reasons=reasons))
    return out


def build(host_output,host_report_sha,host_prepared,host_prepare_sha,host_receipt,host_receipt_sha):
    for module,h in PINS:require(sha(module.__file__)==h,'frozen source pin')
    verified=authority.build(host_output,host_report_sha,host_prepared,host_prepare_sha,host_receipt,host_receipt_sha)
    p=signal.read(host_prepared,256*1024**2); receipt=signal.read(host_receipt,256*1024**2)
    require(len(p['groups'])==len(receipt['rows'])==64 and receipt['complete'] is True,'actual complete upstream')
    specs=[]
    for i,(q,row) in enumerate(zip(p['groups'],receipt['rows'],strict=True)):
        require(q['index']==row['index']==i and row['status']=='sealed','fixed source identities')
        count=sum(x['valid'] for x in fd.pairs(q,row['result']))
        specs.append(dict(index=i,request=deepcopy(q),source_result=deepcopy(row['result']),valid_pairs=count,invalid_causes=invalid_causes(q,row['result']),
                          resource_caps=F.caps(2*count,0)))
    bindings=deepcopy(verified['bindings'])
    for module,h in PINS:
        bindings[str(Path(module.__file__).resolve())]=h; stem=module.__name__.rsplit('.',1)[-1]
        for f in (ROOT/'tests'/('test_'+stem+'.py'),ROOT/'innerdocs'/(stem+'.md')):bindings[str(f)]=sha(f)
    for f in own_files():bindings[str(f)]=sha(f)
    F.check_bindings(bindings)
    args=dict(host_output=host_output,host_report_sha=host_report_sha,host_prepared=host_prepared,
              host_prepare_sha=host_prepare_sha,host_receipt=host_receipt,host_receipt_sha=host_receipt_sha)
    return dict(schema=NAME+'.prepare',arguments=args,contract=contract(),specs=specs,bindings=bindings,
        policy=p['policy'],expected_model_hash=p['expected_model_hash'],planned_valid_pairs=sum(x['valid_pairs'] for x in specs),
        P1_admission=False,freshness=False,physical_release=False)


def verify(path,expected):
    core.hash_value(expected); require(sha(path)==expected,'prepare SHA')
    p=signal.read(path,256*1024**2)
    require(raw(p)==raw(build(**p['arguments'])) and sha(path)==expected,'full source strict and prepare rebuild')
    return p


def write(path,value):
    path=Path(path).absolute(); data=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and path.resolve()==path and
            not path.exists() and not path.is_symlink() and len(data)<=256*1024**2,'exclusive metadata')
    require(shutil.disk_usage(path.parent).free>=contract()['disk_floor_bytes']+len(data),'disk floor')
    with path.open('xb') as f:require(f.write(data)==len(data),'full write');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path))


class Store(F.Store):
    def __init__(self,path,started):
        self.path=Path(path).absolute();self.started=started;self.finalizing=False
        require(self.path.parent==ROOT/'results' and self.path.name.startswith(NAME+'_') and self.path.resolve()==self.path
            and not self.path.exists() and shutil.disk_usage(self.path.parent).free>=contract()['output_bytes']+contract()['disk_floor_bytes'],'exclusive output/floor')
        self.path.mkdir();signal.fsync_dir(self.path.parent)
    def check(self,size=4096):
        require(self.finalizing or time.monotonic()-self.started<contract()['wall_seconds'],'wall watchdog')
        used=sum(f.stat().st_size for f in self.path.rglob('*') if f.is_file())
        require(used+size<=contract()['output_bytes']-(0 if self.finalizing else contract()['final_reserve_bytes'])
            and shutil.disk_usage(self.path).free>=contract()['disk_floor_bytes']+size,'output/final reserve/disk floor')


def roster():return [dict(index=i,status='not_started',result=None,resources=None) for i in range(64)]


def dispatch(p,store,rows,check,factory):
    require(len(p['specs'])==len(rows)==64,'all fixed groups')
    for i,(spec,row) in enumerate(zip(p['specs'],rows,strict=True)):
        require(spec['index']==row['index']==spec['request']['index']==i,'group identity')
        q=spec['request'];source=spec['source_result'];valid=sum(v['valid'] for v in fd.pairs(q,source))
        require(valid==spec['valid_pairs'] and spec['resource_caps']==F.caps(2*valid,0),'presealed valid pair count')
        require(spec['invalid_causes']==invalid_causes(q,source),'presealed explicit null reasons')
        row['status']='attempted'; ns=f'g{i:02}'
        try:journal=transport.NativeJournal(store,ns,check)
        except BaseException as exc:
            row.update(status='failed_before_journal',error=repr(exc));store.put(ns+'_result.json',row);raise
        ledger=core.Ledger(spec['resource_caps'],sink=journal.event)
        try:
            b=factory(q['task'],journal,ledger)
            with b.observe():row['result']=fd.evaluate(q,source,b,journal.event)
            require(ledger.attempted==ledger.completed==spec['resource_caps'],'full paid group')
            row['status']='sealed'
        finally:
            row['resources']=ledger.receipt();journal.close()
            row.update(journal_file=ns+'/events.jsonl',journal_sha256=sha(journal.path),journal_tail=journal.tail)
            if row['status']!='sealed':row['status']='failed_partial'
            store.put(ns+'_result.json',row)


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit CUDA authorization')
    started=time.monotonic();p=verify(prepared,expected);store=Store(output,started);rows=roster()
    errors=[];model=before=after=host=pb=pa=peaks=flags=None;source_ok=False
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
        def check():
            store.check();old.precision(torch)
            require(torch.cuda.memory_allocated(device)<=contract()['cuda_allocated_bytes'] and
                    torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'GPU cap')
        with torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            check();model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False);check()
            before=signal.model_hash(model);require(before['original_state_dict']==p['expected_model_hash'],'frozen checkpoint')
            for module in model.modules():require(str(Path(inspect.getfile(type(module))).resolve()) in p['bindings'],'class source')
            pb=old.precision(torch)
            dispatch(p,store,rows,check,lambda task,journal,ledger:old.Backend(model,device,task,journal,ledger,check))
            pa=old.precision(torch);torch.cuda.synchronize(device);check()
    except BaseException as exc:errors.append(dict(stage='run',error=repr(exc)))
    finally:
        store.finalizing=True
        try:
            if model is not None:
                old.midpoint.bridge.native.frozen(model);after=signal.model_hash(model);require(before==after,'model unchanged')
                peaks=[torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)]
            F.check_bindings(p['bindings']);require(sha(prepared)==expected,'source closure');source_ok=True
        except BaseException as exc:errors.append(dict(stage='closure',error=repr(exc)))
        wall=time.monotonic()-started
        if wall>=contract()['wall_seconds']:errors.append(dict(stage='wall',error='whole cap'))
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        complete=not errors and source_ok and before is not None and before==after and all(x['status']=='sealed' for x in rows)
        artifacts={str(f.relative_to(store.path)):sha(f) for f in store.path.rglob('*') if f.is_file()}
        report=dict(schema=NAME+'.report',status='complete_model_only' if complete else 'partial_failclosed',contract=contract(),
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,errors=errors,group_denominator=64,
            pair_denominator=2560,source_verified=source_ok,source_manifest_sha256=digest(p['bindings']),
            model_before=before,model_after=after,host=host,precision_before=pb,precision_after=pa,
            CUDA_peak_bytes=peaks,wall_seconds=wall,artifact_sha256=artifacts,P1_admission=False,freshness=False,physical_release=False)
        store.put('report.json',report,final=True)
    return dict(status=report['status'],report_sha256=sha(store.path/'report.json'),output=str(store.path),errors=errors)


def consume(output,expected):
    output=Path(output).absolute();core.hash_value(expected);require(sha(output/'report.json')==expected,'report SHA')
    r=signal.read(output/'report.json',256*1024**2);p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and raw(r['contract'])==raw(contract()) and len(r['rows'])==r['group_denominator']==64
        and r['pair_denominator']==2560 and all(r[k] is False for k in ('P1_admission','freshness','physical_release')),'denominators/gates')
    require(r['status'] in ('complete_model_only','partial_failclosed') and type(r['errors']) is list
        and (r['status']!='partial_failclosed' or bool(r['errors'])),'disposition')
    require(signal.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'binding')
    files={str(f.relative_to(output)) for f in output.rglob('*') if f.is_file()}
    require(files==set(r['artifact_sha256'])|{'report.json'} and sum(f.stat().st_size for f in output.rglob('*') if f.is_file())<=contract()['output_bytes'],'file census')
    for name,h in r['artifact_sha256'].items():
        path=Path(name);require(not path.is_absolute() and all(x not in ('.','..') for x in path.parts)
            and (output/path).resolve()==output/path and sha(output/path)==h,'artifact path/SHA')
    receipts=[];audits=[];stopped=False
    for i,(spec,row) in enumerate(zip(p['specs'],r['rows'],strict=True)):
        require(spec['index']==row['index']==i,'ordered groups')
        if row['resources'] is None:
            require(row['result'] is None,'no result without ledger')
            if row['status']=='failed_before_journal':
                require(not stopped and type(row.get('error')) is str and signal.read(output/f'g{i:02}_result.json',32*1024**2)==row,'journal failure')
            else:require(row['status']=='not_started','unreached group')
            stopped=True;audits.append(None);continue
        require(not stopped and signal.read(output/f'g{i:02}_result.json',32*1024**2)==row and row['journal_file']==f'g{i:02}/events.jsonl','ordered durable group')
        require(row['resources']['caps']==spec['resource_caps'] and spec['invalid_causes']==invalid_causes(spec['request'],spec['source_result']),'sealed pair budget and null reasons')
        events=F.load_journal(output,row);transport.audit_resources(row['resources'],events);receipts.append(row['resources'])
        if row['status']=='sealed':audits.append(audit.audit_group(spec['request'],spec['source_result'],row['result'],row['resources'],events))
        else:require(row['status']=='failed_partial','partial group');stopped=True;audits.append(None)
    counts={kind:{k:sum(x[kind][k] for x in receipts) for k in core.AXES} for kind in ('attempted','completed')}
    complete=r['status']=='complete_model_only' and not r['errors'] and all(x is not None for x in audits)
    require((r['status']=='complete_model_only')==complete,'complete evidence')
    if complete:
        require(counts['attempted']==counts['completed']==F.caps(2*p['planned_valid_pairs'],0),'all valid pairs paid')
        require(r['source_verified'] is True and r['model_before']==r['model_after'] and
            r['model_before']['original_state_dict']==p['expected_model_hash'] and r['source_manifest_sha256']==digest(p['bindings']),'source/model closure')
        precision=dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False)
        require(r['precision_before']==r['precision_after']==precision,'precision')
        require(0<=old.scalar(r['wall_seconds'],'wall')<contract()['wall_seconds'] and len(r['CUDA_peak_bytes'])==2 and
            all(type(x) is int and 0<=x<=contract()[k] for x,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)),'wall/GPU')
    F.check_bindings(p['bindings']);require(sha(output/'report.json')==expected,'report unchanged')
    return dict(schema=NAME+'.consume',status='verified_model_only' if complete else 'partial_failclosed',complete=complete,
        report_sha256=expected,group_denominator=64,pair_denominator=2560,rows=r['rows'],audits=audits,
        actual_new_compute=counts,invalid_causes=[x['invalid_causes'] for x in p['specs']],P1_admission=False,freshness=False,physical_release=False,FLOPs=None)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('host-output','host-report-sha','host-prepared','host-prepare-sha','host-receipt','host-receipt-sha',
              'output','prepared','prepare-sha','report-sha','receipt','authorization'):ap.add_argument('--'+k)
    a=ap.parse_args()
    if a.command=='prepare':v=write(a.output,build(a.host_output,a.host_report_sha,a.host_prepared,a.host_prepare_sha,a.host_receipt,a.host_receipt_sha))
    elif a.command=='verify':verify(a.prepared,a.prepare_sha);v=dict(verified=True)
    elif a.command=='run':v=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),flush=True)


if __name__=='__main__':main()
