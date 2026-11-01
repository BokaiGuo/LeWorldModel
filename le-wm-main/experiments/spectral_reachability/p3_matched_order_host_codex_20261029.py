"""Bounded HOST CUDA runner and saved native-IO consumer for matched-order P3.

Only `run --authorize-host-cuda` imports torch. Never dispatches physics.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from experiments.spectral_reachability import p3_matched_order_codex_20261029 as design
from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as prior

core, signal = design.core, prior.signal
ROOT=prior.ROOT
NAME='p3_matched_order_host_codex_20261029'
UPSTREAM=ROOT/'results/agent_p3_parent_native_cuda_20261029_v1_prepare_MAIN01.json'
UPSTREAM_SHA='deb87295f3984929f4734ece0b73770220ec59ac3b621a97e85e2bade60e3599'
PRIOR_SHA='eeea5710d4171b7c43f99d1f2b5ad2412e03a25ba14e8fc5589ea5ef73f57b3b'
DESIGN_SHA='c7a10e794bbcad09b033bdc52ae7eb2b6c64c2b5722153bcb6cdb40522504a74'
need=core.require


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def roster(): return [dict(slot=2*r+i,root=r,policy=policy) for r in (0,1) for i,policy in enumerate(('fixed_order','signal_order'))]
def limits():
    return dict(wall_seconds=3600,output_bytes=1024**3,final_reserve_bytes=256*1024**2,
        disk_floor_bytes=200*1024**2,cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3)


def check_sources(sources):
    need(type(sources) is dict and sources,'source manifest')
    for name,h in sources.items():
        p=Path(name);need(p.is_absolute() and p.is_file() and not p.is_symlink() and sha(p)==h,'source drift: '+name)


def build():
    up=design.read_pinned(UPSTREAM,UPSTREAM_SHA)
    need(sha(prior.__file__)==PRIOR_SHA and sha(design.__file__)==DESIGN_SHA,'frozen implementation drift')
    need(up['schema']==prior.NAME+'.prepare' and len(up['tasks'])==2,'known task authority')
    sources=dict(up['sources'])
    for mod in (design,design.shared,design.frozen,design.live,design.signal,design.parent,prior,
                design.shared.bridge,design.shared.moving,design.shared.old,design.shared.native,
                design.shared.harden,design.shared.frames):
        f=Path(mod.__file__).resolve();h=sha(f);need(str(f) not in sources or sources[str(f)]==h,'upstream source mismatch');sources[str(f)]=h
    sources[str(Path(__file__).resolve())]=sha(__file__)
    check_sources(sources)
    tasks=[core.validate_task(t) for t in up['tasks']]
    return dict(schema=NAME+'.prepare',upstream_sha256=UPSTREAM_SHA,tasks=tasks,policy=up['policy'],
        expected_model_hash=up['expected_model_hash'],sources=sources,roster=roster(),limits=limits(),
        contract=design.contract(),per_unit_cap=design.expected(),precision=prior.precision_contract(),
        science=False,equal_FLOPs=False,physical_authorized=False)


def exclusive(path):
    p=Path(path).absolute();need(p.parent==ROOT/'results' and p.name.startswith(NAME) and p.resolve()==p and not p.exists(),'exclusive own output')
    return p


def prepare(path):
    p=exclusive(path);value=build();payload=core.canonical(value)
    need(len(payload)<=16*1024**2 and shutil.disk_usage(p.parent).free>=limits()['disk_floor_bytes']+len(payload),'prepare storage')
    with p.open('xb') as f:need(f.write(payload)==len(payload),'prepare write');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(p.parent);return sha(p)


def verify_prepare(path,expected):
    p=design.read_pinned(Path(path),expected);need(p==build(),'prepare/source reconstruction');return p


def execute(p,factory,store,*,check=lambda:None,permit_ram=False):
    rows=[dict(**identity,status='not_started',result=None) for identity in roster()]
    store.event(dict(kind='denominator',rows=deepcopy(rows)))
    failed=False
    for row in rows:
        if failed: row['status']='blocked_after_failure';continue
        sink=prior.ArmSink(store,row['slot']);row['status']='attempted'
        store.event(dict(kind='arm_attempt',identity=roster()[row['slot']]))
        try:
            check()
            def make(task,s):return prior.RecordingBackend(factory(task,s),task,s,check)
            result=design.run_unit(p['tasks'][row['root']],row['policy'],make,sink,permit_ram=permit_ram)
            row.update(status=result['status'],result=result);check()
            failed=result['status']!='complete_measured_axes_only'
        except Exception as exc:
            row.update(status='outer_failure',error=type(exc).__name__+': '+str(exc));failed=True
        store.event(dict(kind='arm_return',row=deepcopy(row)))
    return rows


def run(prepared,prepare_sha,output,authorized):
    need(authorized is True,'explicit coordinated HOST CUDA authorization required')
    start=time.monotonic();p=verify_prepare(prepared,prepare_sha);out=exclusive(output)
    need(shutil.disk_usage(out.parent).free>=p['limits']['output_bytes']+p['limits']['disk_floor_bytes'],'full output allowance and floor')
    store=signal.Store(out,p['limits']);store.started=start
    rows=[dict(**r,status='blocked_before_model',result=None) for r in roster()]
    errors=[];model=None;before=after=classes=classes_after=host=None;precision=[];source_ok=False
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=prepare_sha))
        store.event(dict(kind='source_before',sha256=core.digest(p['sources'])))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        store.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
        store.check();model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        before=signal.model_hash(model);need(before['original_state_dict']==p['expected_model_hash'],'model identity')
        classes=prior.model_sources(model,p);store.event(dict(kind='model_before',hashes=before,classes=classes))
        forwards=[(m,'forward' in m.__dict__,m.__dict__.get('forward')) for m in model.modules()]
        with torch.autocast(device_type='cuda',enabled=False),torch.autocast(device_type='cpu',enabled=False),sdpa_kernel(SDPBackend.MATH):
            def guard():store.check();prior.checked_precision(torch)
            precision.append(prior.checked_precision(torch));store.event(dict(kind='precision_before',value=precision[-1]))
            rows=execute(p,lambda t,s:prior.NativeTree(model,device,t,s),store,check=guard)
            precision.append(prior.checked_precision(torch));store.event(dict(kind='precision_after',value=precision[-1]))
        torch.cuda.synchronize(device);store.check()
        need(all(('forward' in m.__dict__)==had and m.__dict__.get('forward') is old for m,had,old in forwards),'restored hooks')
    except BaseException as exc:errors.append(dict(stage='run',type=type(exc).__name__,message=str(exc)))
    finally:
        if model is not None:
            try:
                after=signal.model_hash(model);classes_after=prior.model_sources(model,p)
                need(before==after and classes==classes_after and all(not m.training for m in model.modules()) and all(not q.requires_grad for q in model.parameters()),'frozen model after')
                store.event(dict(kind='model_after',hashes=after,classes=classes_after))
            except BaseException as exc:errors.append(dict(stage='model_after',message=str(exc)))
        try:
            verify_prepare(prepared,prepare_sha);store.check();source_ok=True
            store.event(dict(kind='source_after',sha256=core.digest(p['sources'])))
        except BaseException as exc:errors.append(dict(stage='source_or_limits',message=str(exc)))
        store.close()
        good=not errors and all(r['status']=='complete_measured_axes_only' for r in rows)
        report=dict(schema=NAME+'.report',status='complete_engineering_only' if good else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=prepare_sha,rows=rows,errors=errors,
            model_before=before,model_after=after,classes_before=classes,classes_after=classes_after,
            precision=precision,host=host,sources_verified=source_ok,wall_seconds=store.clock()-start,
            CUDA_peak_bytes=store.memory() if store.memory else None,journal_tail=store.tail,
            artifact_sha256={n:sha(out/n) for n in ('binding.json','events.jsonl') if (out/n).exists()},
            science=False,equal_FLOPs=False,physical_authorized=False)
        store.put('report.json',report)
    return dict(output=str(out),report_sha256=sha(out/'report.json'),status=report['status'])


def audit_arm(task,row,wrapped):
    result=row['result'];need(row['status']=='complete_measured_axes_only' and result['backend_kind']=='HOST_CUDA','complete native unit')
    events=[x['payload'] for x in wrapped]
    need(events[0]['kind']==design.NAME+'.start' and events[-1]==dict(kind=design.NAME+'.final',result=result),'unit journal bookends')
    prior.resource_census(result['resource_receipt'],events)
    calls=[];ids=[]
    for e in wrapped:
        cid=e['call'];kind=e['payload']['kind']
        if cid is None:
            need(kind in prior.RESOURCE|{design.NAME+'.start',design.NAME+'.final'},'unexpected unscoped event');continue
        if not ids or ids[-1]!=cid:
            need(cid==len(ids),'native call sequence');ids.append(cid);calls.append([])
        calls[-1].append(e['payload'])
    need(calls and all(prior.audit_native(call) for call in calls),'full native invocation audit')
    state=dict(calls=calls,index=0);sink=type('Sink',(),{'event':lambda self,e:None})()
    replay=design.run_unit(task,row['policy'],lambda t,s:prior.SavedBackend(t,state),sink)
    need(state['index']==len(calls) and prior.stripped(replay)==prior.stripped(result),'complete saved policy and native replay')
    design.verify_unit(task,result)
    return dict(native_calls=len(calls),completed=result['resource_receipt']['completed'])


def consume(output,report_sha,prepared,prepare_sha):
    p=verify_prepare(prepared,prepare_sha);out=Path(output).absolute()
    report=design.read_pinned(out/'report.json',report_sha)
    need(report['schema']==NAME+'.report' and report['status']=='complete_engineering_only' and not report['errors'], 'complete report')
    need(report['prepared']==str(Path(prepared).absolute()) and report['prepare_sha256']==prepare_sha and report['sources_verified'],'prepare binding')
    need(report['model_before']==report['model_after'] and report['model_before']['original_state_dict']==p['expected_model_hash'] and report['classes_before']==report['classes_after'],'model identity')
    need(report['precision']==[p['precision'],p['precision']] and report['wall_seconds']<=p['limits']['wall_seconds'],'precision/wall')
    host=report['host'];need(type(host) is dict and host.get('device')=='cuda:0' and host.get('allocation_sum')==4.0 and bool(host.get('visible_nvidia_nodes')),'CUDA allocation evidence')
    peak=report['CUDA_peak_bytes'];need(type(peak) is list and len(peak)==2 and all(type(x) is int and x>=0 for x in peak) and peak[0]<=p['limits']['cuda_allocated_bytes'] and peak[1]<=p['limits']['cuda_reserved_bytes'],'peak CUDA bounds')
    for n,h in report['artifact_sha256'].items():need(n in ('binding.json','events.jsonl') and sha(out/n)==h,'artifact hash')
    need(set(report['artifact_sha256'])=={'binding.json','events.jsonl'},'complete artifacts')
    need(json.loads((out/'binding.json').read_bytes())==dict(prepared=str(Path(prepared).absolute()),prepare_sha256=prepare_sha),'execution binding')
    events,tail=signal.journal(out/'events.jsonl');need(tail==report['journal_tail'],'journal tail')
    rows=report['rows'];need(len(rows)==4 and all({k:r[k] for k in ('slot','root','policy')}==ident for r,ident in zip(rows,roster(),strict=True)),'fixed roster')
    top=[e for e in events if e['kind']!='arm_event']
    expected_top=[dict(kind='source_before',sha256=core.digest(p['sources'])),
        dict(kind='model_before',hashes=report['model_before'],classes=report['classes_before']),
        dict(kind='precision_before',value=p['precision']),
        dict(kind='denominator',rows=[dict(**r,status='not_started',result=None) for r in roster()])]
    for ident,row in zip(roster(),rows,strict=True):
        expected_top += [dict(kind='arm_attempt',identity=ident),dict(kind='arm_return',row=row)]
    expected_top += [dict(kind='precision_after',value=p['precision']),
        dict(kind='model_after',hashes=report['model_after'],classes=report['classes_after']),
        dict(kind='source_after',sha256=core.digest(p['sources']))]
    need(top==expected_top,'complete ordered top journal')
    need(all(e['kind']!='arm_event' or type(e['slot']) is int and 0<=e['slot']<4 for e in events),'no foreign arm')
    active=None
    for e in events:
        if e['kind']=='arm_attempt':need(active is None,'nested arms');active=e['identity']['slot']
        elif e['kind']=='arm_event':need(active==e['slot'],'arm event outside active slot')
        elif e['kind']=='arm_return':need(active==e['row']['slot'],'arm return identity');active=None
    need(active is None,'unclosed arm')
    audits=[]
    for row in rows:
        wrapped=[e for e in events if e['kind']=='arm_event' and e['slot']==row['slot']]
        audits.append(audit_arm(p['tasks'][row['root']],row,wrapped))
    pairs=[design.compare(rows[0]['result'],rows[1]['result']),design.compare(rows[2]['result'],rows[3]['result'])]
    need(all(x['matched_measured_axes'] and x['matched_hook_shape_multiset'] for x in pairs),'measured vector and shape mismatch')
    return dict(schema=NAME+'.consume',status='saved_native_IO_engineering_only',report_sha256=report_sha,
        prepare_sha256=prepare_sha,units=audits,pairs=pairs,scientific_pass=False,equal_FLOPs=False,
        limitation='saved IO closure and hooks, not hidden-call exclusion or equal total compute')


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('prepare');a.add_argument('--output',required=True)
    a=sub.add_parser('run');a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True);a.add_argument('--output',required=True);a.add_argument('--authorize-host-cuda',action='store_true')
    a=sub.add_parser('consume');a.add_argument('--prepared',required=True);a.add_argument('--prepare-sha',required=True);a.add_argument('--output',required=True);a.add_argument('--report-sha',required=True);a.add_argument('--receipt',required=True)
    args=ap.parse_args()
    if args.cmd=='prepare': print(json.dumps(dict(prepare_sha256=prepare(args.output))))
    elif args.cmd=='run':
        result=run(args.prepared,args.prepare_sha,args.output,args.authorize_host_cuda);print(json.dumps(result))
        if result['status']!='complete_engineering_only':raise SystemExit(2)
    else:
        result=consume(args.output,args.report_sha,args.prepared,args.prepare_sha);path=exclusive(args.receipt);raw=core.canonical(result)
        with path.open('xb') as f:need(f.write(raw)==len(raw),'receipt write');f.flush();os.fsync(f.fileno())
        signal.fsync_dir(path.parent);print(json.dumps(dict(receipt_sha256=sha(path))))


if __name__=='__main__':main()
