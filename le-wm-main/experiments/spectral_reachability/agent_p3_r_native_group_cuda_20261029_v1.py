"""MAIN-only native zero+spectral8 diagnostic, not an equal-cap tree or plan."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import time

from experiments.spectral_reachability import agent_p3_r_native_orchestrator_20261029_v1 as tree

old,core,parent,signal,live=tree.old,tree.core,tree.parent,tree.signal,tree.live
require,canonical,digest,sha=core.require,core.canonical,core.digest,core.sha
ROOT=core.ROOT
NAME='agent_p3_r_native_group_cuda_20261029_v1'
PROTOCOL=ROOT/'innerdocs/main_p3_r_native_group_diagnostic_protocol_20261029_v1.md'
PROTOCOL_SHA='b8c738587bc51cd0db1ce7b5883924a82ea1167d572e20389ca6929522faf505'
TREE_PINS={
    Path(tree.__file__).resolve():'da00e09c89ed8b5be5399952c2ec328a98e036db08f764b61342d69a18f5e6a6',
    ROOT/'tests/test_agent_p3_r_native_orchestrator_20261029_v1.py':'3294e079f0dbf10ebbfd75617914ce6bc7e0a1e3ae632907f7c5d38c3be37f39',
    ROOT/'innerdocs/agent_p3_r_native_orchestrator_20261029_v1.md':'a02a2ce407cac69ad225c1ec3822b8a2c3dc3b9f9e6d478d9a4445cf1e78496c',
}
FLAGS=dict(tree.FLAGS,model_plan_available=False,equal_cap_tree=False,actual_cp5_start=False)
ROLE='original_saved_warmup_control0_NOT_actual_cp5'


def roster():return [dict(slot=2*r+i,root=r,horizon=h) for r in range(2) for i,h in enumerate((1,3))]


def signal_cost(h):
    require(type(h) is int and h in (1,3),'fixed H1/H3')
    v=live.predict_axes(2*h)
    v.update(signal_compile_calls=1,jacobian_API=1,VJP_directions=192,grad_API=192,raw_backward_callbacks=192)
    return v


def caps(h):return tree.sum_axes(signal_cost(h),tree.group_cost(h))


def profile():
    return dict(wall_seconds=1800,output_bytes=2*1024**3,final_reserve_bytes=256*1024**2,
        group_output_bytes=512*1024**2,disk_floor_bytes=200*1024**2,
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        resource_caps={str(h):caps(h) for h in (1,3)})


def rules():
    return dict(protocol_sha256=PROTOCOL_SHA,bank_protocol_sha256=tree.spectral.PROTOCOL_SHA,
        root_context_role=ROLE,all9_paid=True,all8_preserved=True,all_eligible_preserved=True,
        final_replay_calls=0,plan_release=False,tree_execution=False,
        failure='stop later groups; preserve four slots and charged prefix; no retry',
        scalar_backward_calls=None,total_FLOPs=None,**FLAGS)


def group(task,h,factory,sink,*,permit_ram=False,guard=lambda:None,capsule,saved_calls=None,clock=time.monotonic_ns):
    task=core.validate_task(task);tree.components.post_task(task,[],[])
    kind='RAM_FIXTURE' if permit_ram else 'HOST_CUDA'
    ledger=core.Ledger(caps(h),clock=clock)
    s=tree.Session(task,factory,sink,ledger,kind,guard,capsule,saved_calls)
    ledger.sink=s.event
    a=parent.ParentNativeAdapter(task,lambda t,_:tree.SignalBackend(s,t),
        type('Sink',(),{'event':lambda self,e:s.event(e)})(),permit_ram=permit_ram)
    r=dict(schema=NAME+'.group',backend_kind=kind,horizon=h,root_task_sha256=digest(task),
        root_context_role=ROLE,parent_signal=None,R_groups=[],status='not_started',errors=[],
        fixed_denominator=dict(groups=1,edges=8,zero_controls=1),**FLAGS)
    try:
        guard();ledger.scope=dict(phase='parent_signal',horizon=h,parent_context_sha256=digest(task['root_context']))
        req=core.request(task,parent.root_node(task),operation='compile_finite_horizon_J',horizon=h)
        r['status']='parent_signal_attempted'
        r['parent_signal']=ledger.call('signal_compile',lambda:a.signal(req,ledger))
        require(ledger.attempted==signal_cost(h),'actual parent signal complete cost')
        ledger.scope=dict(phase='R_group',horizon=h,parent_context_sha256=digest(task['root_context']))
        decision=dict(extension_type='reconfiguration',candidates=8,horizon=h)
        g=tree.evaluate_R(task,parent.root_node(task),decision,a.maps[-1],s,tree.empty_axes(),s.event,
            on_start=r['R_groups'].append)
        require(g['status'] in ('no_witness','witness_found'),'all9 diagnostic settled')
        require(ledger.attempted==ledger.completed==caps(h),'exact static full cost, no refunds')
        guard();r['status']='complete_native_R_group_not_admitted'
    except Exception as exc:
        r.update(status='partial_runtime_failure',errors=[dict(type=type(exc).__name__,message=str(exc))])
        for g in r['R_groups']:
            if g['status']=='running':
                g['status']='partial_runtime_failure'
                for row in [g['control']]+g['rows']:
                    if row['status']=='attempted':row['status']='failed'
    r.update(signal_mappings=deepcopy(a.maps),native_calls=deepcopy(s.records),resource_receipt=ledger.receipt())
    core.audit_ledger(r['resource_receipt'])
    return r


def execute(p,factory,store,*,permit_ram=False,check=lambda:None):
    rows=[dict(**x,status='not_requested',result=None) for x in roster()];failed=False
    store.event(dict(kind='denominator',rows=deepcopy(rows)))
    for row in rows:
        if failed:row['status']='blocked_after_runtime_failure';continue
        slot=row['slot'];begin=getattr(store,'bytes',0)
        store.event(dict(kind='group_attempt',identity=roster()[slot]))
        sink=type('Sink',(),{'event':lambda self,e:store.event(dict(kind='group_event',slot=slot,payload=deepcopy(e)))})()
        try:
            check();r=group(p['tasks'][row['root']],row['horizon'],factory,sink,permit_ram=permit_ram,guard=check,
                capsule=lambda cid,call:store.capsule(slot,cid,call))
            row.update(status=r['status'],result=r);failed=bool(r['errors'])
            store.event(dict(kind='group_return',row=deepcopy(row)))
            require(getattr(store,'bytes',0)-begin+len(canonical(row))<=p['profile']['group_output_bytes'],'512MiB group including final row reserve')
            check()
        except Exception as exc:
            row.update(status='outer_runtime_failure',error=dict(type=type(exc).__name__,message=str(exc)));failed=True
            store.event(dict(kind='group_outer_failure',row=deepcopy(row)))
    return rows


class Store(tree.Store):
    def __init__(self,output,limits):
        super().__init__(output,limits);self.group_start=None
    def event(self,value):
        if value['kind']=='group_attempt':self.group_start=self.bytes
        if self.group_start is not None:
            require(self.bytes-self.group_start+len(canonical(value))+1024<=self.limits['group_output_bytes'],'live512MiB group cap')
        super().event(value)
        if value['kind'] in ('group_return','group_outer_failure'):self.group_start=None


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def build():
    require(sha(PROTOCOL)==PROTOCOL_SHA,'frozen MAIN group diagnostic protocol')
    require(all(sha(p)==h for p,h in TREE_PINS.items()),'frozen full tree components')
    prior=tree.build();sources=dict(prior['sources'])
    sources.update({str(p.resolve()):sha(p) for p in own_files()+[PROTOCOL]})
    return dict(schema=NAME+'.prepare',tasks=prior['tasks'],policy=prior['policy'],expected_model_hash=prior['expected_model_hash'],
        authority=prior['authority'],sources=sources,roster=roster(),profile=profile(),precision=old.precision_contract(),rules=rules(),**FLAGS)


def exclusive(path):
    p=Path(path).absolute()
    require(p.parent==ROOT/'results' and p.name.startswith(NAME) and p.resolve()==p and not p.exists() and not p.is_symlink(),'new exclusive group output')
    return p


def write_new(path,value,cap=16*1024**2):
    p=exclusive(path);raw=canonical(value)
    require(len(raw)<=cap and shutil.disk_usage(p.parent).free>=profile()['disk_floor_bytes']+len(raw),'bounded metadata/floor')
    with p.open('xb') as f:require(f.write(raw)==len(raw),'full metadata');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(p.parent);return dict(path=str(p),sha256=sha(p))


def prepare(output):return write_new(output,build())


def verify(prepared,expected):
    core.hash_value(expected);p=old.regular(prepared,16*1024**2)
    require(sha(p)==expected,'actual prepare SHA');v=signal.read(p,16*1024**2)
    require(v==build() and sha(p)==expected,'full source/task/protocol static caps rebuild')
    return v


def run(prepared,expected,output,*,authorization):
    require(authorization=='MAIN_APPROVED_HOST_CUDA','MAIN source audit and HOST CUDA authorization')
    started=time.monotonic();p=verify(prepared,expected);output=exclusive(output)
    require(shutil.disk_usage(output.parent).free>=p['profile']['output_bytes']+p['profile']['disk_floor_bytes'],'2GiB allowance plus floor')
    store=Store(output,p['profile']);store.started=started
    rows=[dict(**x,status='blocked_before_model',result=None) for x in roster()]
    model=None;errors=[];before=after=classes=classes_after=host=None;guards=[];source_ok=False
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        store.event(dict(kind='source_before',sha256=digest(p['sources'])))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        store.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
        store.check();model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        before=signal.model_hash(model);require(before['original_state_dict']==p['expected_model_hash'],'original frozen model')
        classes=old.model_sources(model,p);store.event(dict(kind='model_before',hashes=before,classes=classes))
        forwards=[(m,'forward' in m.__dict__,m.__dict__.get('forward')) for m in model.modules()]
        with torch.autocast(device_type='cuda',enabled=False),torch.autocast(device_type='cpu',enabled=False),sdpa_kernel(SDPBackend.MATH):
            def guard():store.check();old.checked_precision(torch)
            guards.append(old.checked_precision(torch));store.event(dict(kind='precision_before',value=guards[-1]))
            rows=execute(p,lambda t,s:old.NativeTree(model,device,t,s),store,check=guard)
            guards.append(old.checked_precision(torch));store.event(dict(kind='precision_after',value=guards[-1]))
        torch.cuda.synchronize(device)
        require(all(('forward' in m.__dict__)==had and m.__dict__.get('forward') is v for m,had,v in forwards),'instance hooks restored')
    except BaseException as exc:errors.append(dict(stage='run',type=type(exc).__name__,message=str(exc)))
    finally:
        if model is not None:
            try:
                after=signal.model_hash(model);classes_after=old.model_sources(model,p)
                require(before==after and classes==classes_after and all(not m.training for m in model.modules())
                    and all(not x.requires_grad for x in model.parameters()),'frozen model/classes beforeafter')
                store.event(dict(kind='model_after',hashes=after,classes=classes_after))
            except BaseException as exc:errors.append(dict(stage='model_final',message=str(exc)))
        try:
            verify(prepared,expected);store.check();source_ok=True;store.event(dict(kind='source_after',sha256=digest(p['sources'])))
        except BaseException as exc:errors.append(dict(stage='source_final',message=str(exc)))
        store.close();complete=all(x['status']=='complete_native_R_group_not_admitted' for x in rows)
        report=dict(schema=NAME+'.report',status='four_groups_complete_not_admitted' if complete and not errors else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,errors=errors,model_before=before,model_after=after,
            classes_before=classes,classes_after=classes_after,host=host,precision=guards,sources_verified=source_ok,
            source_manifest_sha256=digest(p['sources']),wall_seconds=store.clock()-store.started,
            CUDA_peak_bytes=store.memory() if store.memory else None,journal_tail=store.tail,
            artifact_sha256={n:sha(output/n) for n in ('binding.json','events.jsonl') if (output/n).exists()}|store.capsule_hashes,**FLAGS)
        store.put('report.json',report)
    require(report['status']=='four_groups_complete_not_admitted','partial saved; no retry '+repr(errors))
    return dict(output=str(output),report_sha256=sha(output/'report.json'))


def validate_group(task,h,r,events,*,permit_fixture=False,capsule_reader):
    require(r['backend_kind']=='HOST_CUDA' or permit_fixture and r['backend_kind']=='RAM_FIXTURE','no RAM production')
    require(all(e['kind']=='io' for e in events),'only own IO envelope')
    allowed=old.RESOURCE|old.CALL_ARRAYS|{'direction','native_call_attempt','native_call_return','native_call_failure',
        parent.NAME+'.signal_map','bank_event','R_attempt','R_item_attempt','R_item_return','R_return'}
    require(all(e['payload']['kind'] in allowed for e in events),'unknown group event')
    require(r['resource_receipt']['caps']==caps(h),'exact static group caps')
    payloads=[e['payload'] for e in events];old.resource_census(r['resource_receipt'],payloads)
    calls=r['native_calls'];require([x['call_id'] for x in calls]==list(range(len(calls))),'all native calls ordered')
    require(all(e['call'] is None or type(e['call']) is int and 0<=e['call']<len(calls) for e in events),'no orphan call')
    for c in calls:
        require([e['payload'] for e in events if e['call']==c['call_id']]==c['transcript'],'full native journal equality')
        old.audit_native(c['transcript'])
    sink=type('Sink',(),{'event':lambda self,e:None})()
    replay=group(task,h,None,sink,permit_ram=permit_fixture,guard=lambda:None,capsule=capsule_reader,saved_calls=calls,
        clock=tree.replay_clock(r['resource_receipt'],payloads))
    require(replay==r,'complete saved native/bank/ridge/eligibility algebra')
    complete=r['status']=='complete_native_R_group_not_admitted'
    if complete:
        require(len(calls)==28 and r['resource_receipt']['attempted']==r['resource_receipt']['completed']==caps(h),'all28 native calls/full9 costs')
    return dict(horizon=h,status=r['status'],complete=complete,eligible_indices=r['R_groups'][0]['eligible_indices'] if complete else None,
        no_witness=not r['R_groups'][0]['eligible_indices'] if complete else None,native_calls=len(calls),resources=r['resource_receipt'],
        AD_primal_predict_API=sum(c['transcript'][0]['request']['horizon'] for c in calls if c['transcript'][0]['method']=='signal') if complete else None,
        Jacobian_elements=sum(192*10*c['transcript'][0]['request']['horizon'] for c in calls if c['transcript'][0]['method']=='signal') if complete else None,
        scalar_backward_calls=None,total_FLOPs=None,**FLAGS)


def validate_rows(p,rows,events,*,permit_fixture=False,capsule_reader):
    require(len(rows)==4 and [(r['slot'],r['root'],r['horizon']) for r in rows]==[(r['slot'],r['root'],r['horizon']) for r in roster()],'fixed known2xH1H3 four groups')
    top={'denominator','group_attempt','group_return','group_outer_failure','source_before','source_after','model_before','model_after','precision_before','precision_after'}
    require(all(e['kind'] in top|{'group_event'} for e in events),'unknown top event')
    require(all(type(e['slot']) is int and 0<=e['slot']<4 for e in events if e['kind']=='group_event'),'orphan group')
    initial=[dict(**x,status='not_requested',result=None) for x in roster()]
    require([e['rows'] for e in events if e['kind']=='denominator']==[initial],'initial four denominator')
    failed=False;position=-1;out=[]
    for row,identity in zip(rows,roster(),strict=True):
        es=[e['payload'] for e in events if e['kind']=='group_event' and e['slot']==row['slot']]
        starts=[i for i,e in enumerate(events) if e['kind']=='group_attempt' and e['identity']==identity]
        ends=[i for i,e in enumerate(events) if e['kind']=='group_return' and e['row']==row]
        if row['result'] is None:
            require(failed and row['status']=='blocked_after_runtime_failure' and not starts and not ends and not es,'failstop later groups');continue
        require(not failed and row['status']==row['result']['status'] and len(starts)==len(ends)==1 and position<starts[0]<ends[0],'exact group dispatch')
        require(all(starts[0]<i<ends[0] for i,e in enumerate(events) if e['kind']=='group_event' and e['slot']==row['slot']),'all work within own group')
        position=ends[0]
        out.append(validate_group(p['tasks'][row['root']],row['horizon'],row['result'],es,permit_fixture=permit_fixture,
            capsule_reader=lambda cid,call,slot=row['slot']:capsule_reader(slot,cid,call)))
        failed=bool(row['result']['errors'])
    require(len([e for e in events if e['kind']=='group_attempt'])==len(out)==len([e for e in events if e['kind']=='group_return']),'no extra dispatch')
    require(not any(e['kind']=='group_outer_failure' for e in events),'outer failures retained but not certified')
    return out


def journal(output,r,group_sizes=None):
    path=old.regular(output/'events.jsonl',profile()['output_bytes']);events=[];tail='0'*64;active=None
    with path.open('rb') as f:
        for i,line in enumerate(f):
            require(line.endswith(b'\n') and len(line)<=128*1024**2,'complete bounded JSONL record')
            x=json.loads(line,object_pairs_hook=signal.unique_pairs);core.keys(x,('index','previous','payload','sha256'))
            h=x.pop('sha256');require(type(x['index']) is int and x['index']==i and x['previous']==tail and digest(x)==h,'actual journal hashchain')
            tail=h;payload=x['payload'];events.append(payload)
            if group_sizes is not None:
                if payload['kind']=='group_attempt':active=payload['identity']['slot']
                if active is not None:group_sizes[active]=group_sizes.get(active,0)+len(line)
                if payload['kind'] in ('group_return','group_outer_failure'):active=None
    require(tail==r['journal_tail'],'final journal tail');return events


def consume(output,report_sha,prepared,prepare_sha):
    output=Path(output).absolute();require(output.resolve()==output and not output.is_symlink(),'exact own output')
    core.hash_value(report_sha);rp=old.regular(output/'report.json',profile()['final_reserve_bytes'])
    require(sha(rp)==report_sha,'actual group report SHA');r=signal.read(rp,profile()['final_reserve_bytes']);p=verify(prepared,prepare_sha)
    require(r['schema']==NAME+'.report' and r['prepared']==str(Path(prepared).absolute()) and r['prepare_sha256']==prepare_sha,'own schema authority')
    require(not r['errors'] and all(r[k] is False for k in FLAGS) and r['sources_verified'] is True
        and r['source_manifest_sha256']==digest(p['sources']),'complete run source closure')
    require(r['model_before']==r['model_after'] and r['model_before']['original_state_dict']==p['expected_model_hash'],'frozen model exact')
    require(r['classes_before']==r['classes_after'] and r['classes_before'],'class source closure')
    for c in r['classes_before']:require(p['sources'].get(c['path'])==c['sha256'],'installed classes pinned')
    require(r['precision']==[old.precision_contract()]*2 and r['host']['device']=='cuda:0' and r['host']['allocation_sum']==4.
        and r['host']['visible_nvidia_nodes'],'actual CUDA and numerical guards')
    names=set(r['artifact_sha256']);require({'binding.json','events.jsonl'}<=names and all(n in ('binding.json','events.jsonl')
        or n.startswith('signal_s') and n.endswith('.json') and Path(n).name==n for n in names),'explicit artifacts only')
    require({x.name for x in output.iterdir()}==names|{'report.json'},'full artifact census')
    for n,h in r['artifact_sha256'].items():require(sha(old.regular(output/n,profile()['output_bytes']))==h,'all saved bytes')
    require(signal.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=prepare_sha),'persisted binding')
    group_sizes={};es=journal(output,r,group_sizes)
    for phase in ('before','after'):
        require([e for e in es if e['kind']=='model_'+phase]==[dict(kind='model_'+phase,hashes=r['model_'+phase],classes=r['classes_'+phase])],'model journal')
        require([e for e in es if e['kind']=='source_'+phase]==[dict(kind='source_'+phase,sha256=digest(p['sources']))],'source journal')
        require([e['value'] for e in es if e['kind']=='precision_'+phase]==[old.precision_contract()],'precision journal')
    seen=set()
    def read_capsule(slot,cid,call):
        n=f'signal_s{slot}_{cid:04}.json';require(n in names,'actual saved parent signal capsule')
        require(signal.read(old.regular(output/n,64*1024**2),64*1024**2)==call and sha(output/n)==digest(call),'exact source capsule')
        seen.add(n);return dict(path=str(output/n),sha256=sha(output/n))
    summaries=validate_rows(p,r['rows'],es,capsule_reader=read_capsule)
    require(seen==names-{'binding.json','events.jsonl'},'no dangling source capsules')
    require(r['status']=='four_groups_complete_not_admitted' and len(summaries)==4 and all(x['complete'] for x in summaries),'all four complete required')
    require(0<=r['wall_seconds']<=p['profile']['wall_seconds'] and sum(x.stat().st_size for x in output.iterdir())<=p['profile']['output_bytes'],'wall/output limit')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes'])==2 and all(type(v) is int and 0<=v<=p['profile'][k]
        for v,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'))),'CUDA memory limits')
    for row in r['rows']:
        # Exact actual JSONL bytes (including envelopes/hashchain) plus final row
        # and independently persisted parent source capsule, not just payloads.
        used=group_sizes[row['slot']]+len(canonical(row))
        used+=sum((output/n).stat().st_size for n in names if n.startswith('signal_s'+str(row['slot'])+'_'))
        require(used<=p['profile']['group_output_bytes'],'per-group saved output bound')
    require(sha(rp)==report_sha and all(sha(f)==h for f,h in p['sources'].items()),'unchanged final closure')
    return dict(schema=NAME+'.consume',status='all4_native_R_groups_verified_not_admitted',prepared=r['prepared'],
        prepare_sha256=prepare_sha,report_sha256=report_sha,denominator=4,arms=summaries,
        protocol_sha256=PROTOCOL_SHA,root_context_role=ROLE,**FLAGS)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare');q.add_argument('--output',required=True)
    for cmd in ('verify','run','consume'):
        q=sub.add_parser(cmd);q.add_argument('--prepared',required=True);q.add_argument('--prepare-sha',required=True)
        if cmd in ('run','consume'):q.add_argument('--output',required=True)
        if cmd=='run':q.add_argument('--authorization',required=True)
        if cmd=='consume':q.add_argument('--report-sha',required=True);q.add_argument('--receipt',required=True)
    a=p.parse_args(argv)
    if a.cmd=='prepare':out=prepare(a.output)
    elif a.cmd=='verify':out=dict(verified=True,schema=verify(a.prepared,a.prepare_sha)['schema'])
    elif a.cmd=='run':out=run(a.prepared,a.prepare_sha,a.output,authorization=a.authorization)
    else:out=write_new(a.receipt,consume(a.output,a.report_sha,a.prepared,a.prepare_sha),128*1024**2)
    print(canonical(out).decode())


if __name__=='__main__':main()
