"""Actual-cp5 spectral8 model delivery. CUDA only after explicit MAIN approval."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import agent_p3_r_native_group_cuda_20261029_v1 as base
from experiments.spectral_reachability import agent_p3_actual_cp5_task_builder_20261029_v1 as resolver

tree,old,core,parent,signal,live=base.tree,base.old,base.core,base.parent,base.signal,base.live
require,canonical,digest,sha=core.require,core.canonical,core.digest,core.sha
ROOT=core.ROOT
NAME='agent_p3_actual_cp5_r_delivery_20261029_v1'
PROTOCOL=ROOT/'innerdocs/main_p3_actual_cp5_R_delivery_protocol_20261029_v1.md'
PROTOCOL_SHA='deed88fd70073c1aaca96d32c81968b7f02a7a3841e3d8a7be00d4e4d573275f'
CP5=ROOT/'results/agent_p3_actual_cp5_task_builder_20261029_v1_prepare_MAIN01.json'
CP5_SHA='ce6670d84f6f0f712bdfe5e1b7471be65b0441338c18010f79b3967ef28397c7'
PINS={
    Path(base.__file__).resolve():'2fc119a6f7bc424825cf81d289e278b48922297fe50c1b5015161e713afbcb6c',
    ROOT/'tests'/('test_'+base.NAME+'.py'):'5d8fe93b3878c4dcb103b3fdf0080e1a4be898266d8fa248360f46d01acd484f',
    ROOT/'innerdocs'/(base.NAME+'.md'):'96d45f8cd89372d2d3bda20506fb57307c7e893b38b8604216429fd871544dfe',
    Path(resolver.__file__).resolve():'703f5f552c1911159b2144f72711c542ca48c408ed4e306b1fc17b019df889f1',
    ROOT/'tests'/('test_'+resolver.NAME+'.py'):'e25d84eb495bd27d3cf684578291f8e83947f224a2e9f2aad5e199608932572d',
    ROOT/'innerdocs'/(resolver.NAME+'.md'):'4e6893127778141812f4e0739b3d7e4bc6fe608e77209595fce100e8f7ad1ed2',
}
FLAGS=dict(tree.FLAGS,execution_authorized=False,equal_cap_tree=False,full_tree_executed=False)
ROLE='actual_observed_abs5_10_15_control_minus5_0_5'
SETTLED=('model_delivery_verified','no_witness')


def roster():return base.roster()


def verification_cost(h):
    v=live.predict_axes(2*h);v['replay_calls']=2
    return v


def caps(h):return tree.sum_axes(base.caps(h),verification_cost(h))


def profile():return dict(base.profile(),resource_caps={str(h):caps(h) for h in (1,3)})


def rules():
    return dict(protocol_sha256=PROTOCOL_SHA,cp5_tasks_sha256=CP5_SHA,root_context_role=ROLE,
        actual_cp5_start=True,selection=['immediate','candidate_index'],all_eligible_preserved=True,
        segmented_raw_steps={'1':[5],'3':[5,10]},commit_raw_steps=5,suffix_is_plan=False,
        no_witness='no delivery; no fallback; continue other fixed slots',
        runtime_failure='preserve charged prefix and four slots; block later slots; no retry',
        scalar_backward_calls=None,total_FLOPs=None,**FLAGS)


def raw_hash(raw):
    a=core.finite(raw,(len(raw),2),True) if raw else np.empty((0,2),np.float32)
    return hashlib.sha256(np.asarray(a,np.dtype('<f4')).tobytes(order='C')).hexdigest()


def ranked(g):
    indices=[r['index'] for r in g['rows'] if r['eligible']]
    require(indices==g['eligible_indices'],'all eligible retained before ordering')
    return sorted(indices,key=lambda i:(g['rows'][i]['immediate'],i))


def segmented_packets(actions):
    """Explicit shared RAM/HOST schedule: H3 reset only after its first5."""
    a=core.finite(actions,(len(actions),2),True)
    require(len(actions) in (5,15) and np.max(np.abs(a))<=1,'delivery H1/H3 packets')
    return [(i,p.copy(),i==1) for i,p in enumerate(a.astype(np.float32).reshape(-1,5,2))]


def input_pair(task,provenance):
    task=core.validate_task(task);p=deepcopy(provenance)
    original=core.validate_task(p['original_task'])
    require(p['original_task_sha256']==digest(original) and p['local_task_sha256']==digest(task),'base/local cp5 task double binding')
    require(p['actual_cp5_absolute_steps']==[5,10,15] and p['actual_cp5_control_steps']==[-5,0,5]
        and p['original_observed_absolute_steps']==[0,5,10] and p['original_control_steps']==[-10,-5,0],'real cp5 chronology')
    for key in ('goal','projection','action_mean','action_std'):
        require(canonical(task[key])==canonical(original[key]),'frozen original goal/Q/normalizer')
    for key in ('root_sha256','blind_sha256','model_sha256','goal_sha256','projection_sha256','normalizer_sha256','action_domain_sha256'):
        require(task['binding'][key]==original['binding'][key],'original root authority '+key)
    require(task['binding']['task_id']!=original['binding']['task_id'],'rebased cp5 task, not relabelled control0')
    raw=core.finite(p['raw10'],(10,2),True);prefix=core.finite(p['prefix5'],(5,2),True)
    require(np.max(np.abs(raw))<=1 and raw_hash(raw[5:].tolist())==raw_hash(prefix.tolist()),'actual prefix raw bits')
    expected=((raw.astype(np.float32)-np.asarray(task['action_mean'],np.float32))/np.asarray(task['action_std'],np.float32)).reshape(2,10)
    require(expected.tobytes()==np.asarray(task['root_context']['action_history'],np.float32).tobytes(),'actual raw10 once-normalized history bits')
    require(canonical(task['root_context']['latent_history'][:2])==canonical(original['root_context']['latent_history'][1:]),'actual cp5 older two latent shift')
    require(len(p['branch_witnesses'])==4 and len({x['episode_id'] for x in p['branch_witnesses']})==4,'all four source branches')
    return task,p


class NativeDelivery(old.NativeTree):
    """Same native equations; H3 segmented replay has the explicit 5+10 split."""
    def replay(self,parent_context,actions,segmented,ledger):
        if not segmented:return super().replay(parent_context,actions,False,ledger)
        core.context(parent_context);a=core.finite(actions,(len(actions),2),True)
        require(parent_context==self.task['root_context'] and len(actions) in (5,15) and np.max(np.abs(a))<=1,'H1/H3 delivery raw edge')
        import torch
        self.frozen();self.ledger=ledger;self.phase='segmented'
        state=torch.tensor(parent_context['latent_history'],dtype=torch.float32,device=self.device)[None]
        history=torch.tensor(parent_context['action_history'],dtype=torch.float32,device=self.device)[None]
        mean=torch.tensor(self.task['action_mean'],dtype=torch.float32,device=self.device)
        std=torch.tensor(self.task['action_std'],dtype=torch.float32,device=self.device);path=[]
        with torch.no_grad(),self.observe():
            for step,packet,reset in segmented_packets(actions):
                self.step=step
                if reset: # H3: reconstitute after5, then execute the remaining10 continuously.
                    state=torch.tensor(path[-1]['latent_history'],dtype=torch.float32,device=self.device)[None]
                    history=torch.tensor(path[-1]['action_history'],dtype=torch.float32,device=self.device)[None]
                raw=torch.tensor(packet,dtype=torch.float32,device=self.device)
                token=((raw-mean)/std).reshape(10);outgoing=torch.cat((history,token[None,None]),dim=1)
                self.arrays('program_packet',latent=state,history=history,raw=raw,mean=mean,std=std,token=token,outgoing=outgoing)
                encoded=self.model.action_encoder(outgoing)
                predicted=self.call('predict',lambda:self.model.predict(state,encoded),dict(latent=state,act_emb=encoded),shape=list(state.shape))
                require(tuple(predicted.shape)==(1,3,192),'native full3 predictor ABI')
                state=torch.cat((state[:,1:],predicted[:,-1:]),dim=1);history=outgoing[:,1:]
                self.arrays('program_boundary',latent=state,history=history)
                path.append(dict(parent_context,latent_history=state[0].cpu().tolist(),action_history=history[0].cpu().tolist()))
        self.frozen();core.checked_path(self.task,parent_context,actions,path)
        return path


def delivery(task,pro,h,g,order,verification,source_sha):
    selected=g['rows'][order[0]];raw=selected['actions']
    return dict(schema=NAME+'.delivery',model_delivery_verified=True,candidate_index=selected['index'],
        eligible_order=order,selection_key=[selected['immediate'],selected['index']],task=deepcopy(task),
        task_sha256=digest(task),base_task=deepcopy(pro['original_task']),base_task_sha256=pro['original_task_sha256'],
        provenance_sha256=digest(pro),source_closure_sha256=source_sha,root_context_role=ROLE,
        raw_edge=deepcopy(raw),raw_edge_float32_sha256=raw_hash(raw),path=deepcopy(selected['path']),
        commit_raw5=deepcopy(raw[:5]),commit_float32_sha256=raw_hash(raw[:5]),
        lookahead_raw=deepcopy(raw[5:]),lookahead_float32_sha256=raw_hash(raw[5:]),
        control_start=5,commit_control_steps=[6,7,8,9,10],prediction_end_control=5+5*h,
        prediction_end_absolute=15+5*h,verification=deepcopy(verification),
        suffix_in_plan=False,**FLAGS)


def group(task,pro,h,factory,sink,*,source_sha,permit_ram=False,guard=lambda:None,capsule,saved_calls=None,clock=time.monotonic_ns):
    task,pro=input_pair(task,pro);base.signal_cost(h);core.hash_value(source_sha)
    kind='RAM_FIXTURE' if permit_ram else 'HOST_CUDA';ledger=core.Ledger(caps(h),clock=clock)
    s=tree.Session(task,factory,sink,ledger,kind,guard,capsule,saved_calls);ledger.sink=s.event
    adapter=parent.ParentNativeAdapter(task,lambda t,_:tree.SignalBackend(s,t),
        type('Sink',(),{'event':lambda self,e:s.event(e)})(),permit_ram=permit_ram)
    r=dict(schema=NAME+'.group',backend_kind=kind,horizon=h,task_sha256=digest(task),provenance_sha256=digest(pro),
        source_closure_sha256=source_sha,root_context_role=ROLE,actual_cp5_start=True,parent_signal=None,
        R_groups=[],eligible_order=None,delivery=None,model_delivery_verified=False,status='not_started',errors=[],
        verification=[dict(mode=k,segments=[5*h] if k=='whole' else ([5] if h==1 else [5,10]),status='not_requested',call_id=None,path=None)
            for k in ('whole','segmented')],**FLAGS)
    try:
        guard();ledger.scope=dict(phase='parent_signal',horizon=h,parent_context_sha256=digest(task['root_context']))
        req=core.request(task,parent.root_node(task),operation='compile_finite_horizon_J',horizon=h)
        r['parent_signal']=ledger.call('signal_compile',lambda:adapter.signal(req,ledger))
        require(ledger.attempted==base.signal_cost(h),'parent signal exact cost')
        ledger.scope=dict(phase='R_group',horizon=h,parent_context_sha256=digest(task['root_context']))
        g=tree.evaluate_R(task,parent.root_node(task),dict(extension_type='reconfiguration',candidates=8,horizon=h),
            adapter.maps[-1],s,verification_cost(h),s.event,on_start=r['R_groups'].append)
        require(ledger.attempted==ledger.completed==base.caps(h),'complete paid9 before selection')
        order=ranked(g);r['eligible_order']=order
        s.event(dict(kind='delivery_selection',eligible_order=order,selection_key=['immediate','candidate_index']))
        if not order:r['status']='no_witness'
        else:
            selected=g['rows'][order[0]]
            for v in r['verification']:
                v.update(status='attempted',call_id=len(s.records))
                ledger.scope=dict(phase='delivery_verification',horizon=h,mode=v['mode'],segments=v['segments'])
                s.event(dict(kind='delivery_verification_attempt',verification=deepcopy(v)))
                path=ledger.call('replay',lambda v=v:s.replay(task,selected['actions'],v['mode']=='segmented'))
                require(canonical(path)==canonical(selected['path']),'full edge whole/5+10 segmented exact context bits')
                v.update(status='returned',path=deepcopy(path));s.event(dict(kind='delivery_verification_return',verification=deepcopy(v)))
            require(ledger.attempted==ledger.completed==caps(h),'full independent validation charges')
            guard();r.update(status='model_delivery_verified',model_delivery_verified=True,
                delivery=delivery(task,pro,h,g,order,r['verification'],source_sha))
            s.event(dict(kind='model_delivery',delivery=deepcopy(r['delivery'])))
    except Exception as exc:
        r.update(status='partial_runtime_failure',errors=[dict(type=type(exc).__name__,message=str(exc))],delivery=None,model_delivery_verified=False)
        for g in r['R_groups']:
            if g['status']=='running':
                g['status']='partial_runtime_failure'
                for item in [g['control']]+g['rows']:
                    if item['status']=='attempted':item['status']='failed'
        for v in r['verification']:
            if v['status']=='attempted':v['status']='failed'
    r.update(signal_mappings=deepcopy(adapter.maps),native_calls=deepcopy(s.records),resource_receipt=ledger.receipt())
    core.audit_ledger(r['resource_receipt']);return r


def execute(p,factory,store,*,permit_ram=False,check=lambda:None):
    rows=[dict(**x,status='not_requested',result=None) for x in roster()];failed=False
    store.event(dict(kind='denominator',rows=deepcopy(rows)))
    for row in rows:
        if failed:row['status']='blocked_after_runtime_failure';continue
        slot=row['slot'];begin=getattr(store,'bytes',0);store.event(dict(kind='group_attempt',identity=roster()[slot]))
        sink=type('Sink',(),{'event':lambda self,e:store.event(dict(kind='group_event',slot=slot,payload=deepcopy(e)))})()
        try:
            check();r=group(p['tasks'][row['root']],p['provenance'][row['root']],row['horizon'],factory,sink,
                source_sha=digest(p['sources']),permit_ram=permit_ram,guard=check,capsule=lambda cid,call:store.capsule(slot,cid,call))
            row.update(status=r['status'],result=r);failed=bool(r['errors'])
            store.event(dict(kind='group_return',row=deepcopy(row)))
            require(getattr(store,'bytes',0)-begin+len(canonical(row))<=p['profile']['group_output_bytes'],'512MiB full group output cap');check()
        except Exception as exc:
            row.update(status='outer_runtime_failure',error=dict(type=type(exc).__name__,message=str(exc)));failed=True
            store.event(dict(kind='group_outer_failure',row=deepcopy(row)))
    return rows


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def material(prior,cp):
    require(len(prior['tasks'])==len(cp['tasks'])==len(cp['provenance'])==2,'all known2 tasks')
    for root,(t,p,b) in enumerate(zip(cp['tasks'],cp['provenance'],prior['tasks'],strict=True)):
        require(type(p['source_slot']) is int and p['source_slot']==root,'fixed root ordering')
        input_pair(t,p);require(canonical(p['original_task'])==canonical(b),'exact original root source, not different cohort')
    return deepcopy(cp['tasks']),deepcopy(cp['provenance'])


def build():
    require(sha(PROTOCOL)==PROTOCOL_SHA and all(sha(p)==h for p,h in PINS.items()),'sealed protocol and component triples')
    cp=resolver.read_pinned(CP5,CP5_SHA,resolver.OUTPUT_CAP)
    rebuilt=resolver.resolve_actual_cp5_tasks(cp['authorities'])
    require(canonical(rebuilt)==canonical(cp) and sha(CP5)==CP5_SHA,'full frozen actual-cp5 resolver output reconstruction')
    prior=base.build();tasks,pro=material(prior,cp);sources=dict(prior['sources'])
    sources.update(cp['bindings']);sources.update({str(p.resolve()):sha(p) for p in own_files()+[PROTOCOL,CP5]})
    require(all(sha(p)==h for p,h in sources.items()),'complete current source/input closure')
    return dict(schema=NAME+'.prepare',tasks=tasks,provenance=pro,cp5_tasks=dict(path=str(CP5),sha256=CP5_SHA),
        policy=prior['policy'],expected_model_hash=prior['expected_model_hash'],authority=prior['authority'],sources=sources,
        roster=roster(),profile=profile(),precision=old.precision_contract(),rules=rules(),**FLAGS)


def exclusive(path):
    p=Path(path).absolute()
    require(p.parent==ROOT/'results' and p.name.startswith(NAME) and p.resolve()==p and not p.exists() and not p.is_symlink(),'new exclusive delivery output')
    return p


def write_new(path,value,cap=16*1024**2):
    p=exclusive(path);raw=canonical(value)
    require(len(raw)<=cap and shutil.disk_usage(p.parent).free>=profile()['disk_floor_bytes']+len(raw),'metadata cap/floor')
    with p.open('xb') as f:require(f.write(raw)==len(raw),'complete metadata');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(p.parent);return dict(path=str(p),sha256=sha(p))


def prepare(output):return write_new(output,build())


def verify(prepared,expected):
    core.hash_value(expected);p=old.regular(prepared,16*1024**2)
    require(sha(p)==expected,'actual delivery prepare SHA');v=signal.read(p,16*1024**2)
    require(v==build() and sha(p)==expected,'full delivery source/cp5/budget reconstruction');return v


def run(prepared,expected,output,*,authorization):
    require(authorization=='MAIN_APPROVED_HOST_CUDA','MAIN source audit and HOST CUDA authorization')
    started=time.monotonic();p=verify(prepared,expected);output=exclusive(output)
    require(shutil.disk_usage(output.parent).free>=p['profile']['output_bytes']+p['profile']['disk_floor_bytes'],'2GiB output allowance plus floor')
    store=base.Store(output,p['profile']);store.started=started
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
            rows=execute(p,lambda t,s:NativeDelivery(model,device,t,s),store,check=guard)
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
        store.close();settled=all(x['status'] in SETTLED for x in rows)
        report=dict(schema=NAME+'.report',status='four_slots_settled_not_admitted' if settled and not errors else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,errors=errors,model_before=before,model_after=after,
            classes_before=classes,classes_after=classes_after,host=host,precision=guards,sources_verified=source_ok,
            source_manifest_sha256=digest(p['sources']),wall_seconds=store.clock()-store.started,
            CUDA_peak_bytes=store.memory() if store.memory else None,journal_tail=store.tail,
            artifact_sha256={n:sha(output/n) for n in ('binding.json','events.jsonl') if (output/n).exists()}|store.capsule_hashes,**FLAGS)
        store.put('report.json',report)
    require(report['status']=='four_slots_settled_not_admitted','partial saved; no retry '+repr(errors))
    return dict(output=str(output),report_sha256=sha(output/'report.json'))


def validate_group(task,pro,h,r,events,*,source_sha,permit_fixture=False,capsule_reader):
    require(r['backend_kind']=='HOST_CUDA' or permit_fixture and r['backend_kind']=='RAM_FIXTURE','no RAM production')
    require(all(e['kind']=='io' for e in events),'only own IO envelope')
    allowed=old.RESOURCE|old.CALL_ARRAYS|{'direction','native_call_attempt','native_call_return','native_call_failure',
        parent.NAME+'.signal_map','bank_event','R_attempt','R_item_attempt','R_item_return','R_return',
        'delivery_selection','delivery_verification_attempt','delivery_verification_return','model_delivery'}
    require(all(e['payload']['kind'] in allowed for e in events),'unknown delivery event')
    require(r['resource_receipt']['caps']==caps(h),'exact independent integration caps')
    payloads=[e['payload'] for e in events];old.resource_census(r['resource_receipt'],payloads)
    calls=r['native_calls'];require([x['call_id'] for x in calls]==list(range(len(calls))),'native invocation census')
    require(all(e['call'] is None or type(e['call']) is int and 0<=e['call']<len(calls) for e in events),'no orphan call')
    for c in calls:
        require([e['payload'] for e in events if e['call']==c['call_id']]==c['transcript'],'complete native journal binding')
        old.audit_native(c['transcript'])
    replay_events=[];sink=type('Sink',(),{'event':lambda self,e:replay_events.append(deepcopy(e))})()
    replay=group(task,pro,h,None,sink,source_sha=source_sha,permit_ram=permit_fixture,capsule=capsule_reader,saved_calls=calls,
        clock=tree.replay_clock(r['resource_receipt'],payloads))
    require(canonical(replay)==canonical(r),'exact saved full9/order/edge/commit/verification algebra')
    # Session preserves call transcripts but inert replay does not re-emit their
    # bodies. Compare every non-call event independently (including selection).
    require([e for e in replay_events if e['call'] is None]==[e for e in events if e['call'] is None],'all orchestration events exact census/order')
    complete=r['status'] in SETTLED;has_delivery=r['status']=='model_delivery_verified'
    if complete:
        require(len(calls)==(30 if has_delivery else 28),'no hidden/unused native invocations')
        expected=caps(h) if has_delivery else base.caps(h)
        require(r['resource_receipt']['attempted']==r['resource_receipt']['completed']==expected,'exact witness/no-witness costs')
    return dict(horizon=h,status=r['status'],complete=complete,model_delivery_verified=has_delivery,
        eligible_order=r['eligible_order'] if complete else None,no_witness=not has_delivery if complete else None,
        delivery=deepcopy(r['delivery']),native_calls=len(calls),resources=r['resource_receipt'],
        AD_primal_predict_API=h+9 if complete else None,Jacobian_elements=192*10*(h+9) if complete else None,
        scalar_backward_calls=None,total_FLOPs=None,**FLAGS)


def validate_rows(p,rows,events,*,permit_fixture=False,capsule_reader):
    require(len(rows)==4 and [(r['slot'],r['root'],r['horizon']) for r in rows]==[(r['slot'],r['root'],r['horizon']) for r in roster()],'all four independent fixed slots')
    top={'denominator','group_attempt','group_return','group_outer_failure','source_before','source_after','model_before','model_after','precision_before','precision_after'}
    require(all(e['kind'] in top|{'group_event'} for e in events),'unknown top event')
    require(all(type(e['slot']) is int and 0<=e['slot']<4 for e in events if e['kind']=='group_event'),'orphan slot')
    initial=[dict(**x,status='not_requested',result=None) for x in roster()]
    require([e['rows'] for e in events if e['kind']=='denominator']==[initial],'initial denominator')
    failed=False;position=-1;out=[]
    for row,identity in zip(rows,roster(),strict=True):
        es=[e['payload'] for e in events if e['kind']=='group_event' and e['slot']==row['slot']]
        starts=[i for i,e in enumerate(events) if e['kind']=='group_attempt' and e['identity']==identity]
        ends=[i for i,e in enumerate(events) if e['kind']=='group_return' and e['row']==row]
        if row['result'] is None:
            require(failed and row['status']=='blocked_after_runtime_failure' and not starts and not ends and not es,'runtime-only stop of later slots');continue
        require(not failed and row['status']==row['result']['status'] and len(starts)==len(ends)==1 and position<starts[0]<ends[0],'exact ordered dispatch')
        require(all(starts[0]<i<ends[0] for i,e in enumerate(events) if e['kind']=='group_event' and e['slot']==row['slot']),'native events within own dispatch')
        position=ends[0]
        summary=validate_group(p['tasks'][row['root']],p['provenance'][row['root']],row['horizon'],row['result'],es,
            source_sha=digest(p['sources']),permit_fixture=permit_fixture,capsule_reader=lambda cid,call,slot=row['slot']:capsule_reader(slot,cid,call))
        out.append(dict(slot=row['slot'],root=row['root'],**summary))
        failed=bool(row['result']['errors'])
    require(len([e for e in events if e['kind']=='group_attempt'])==len(out)==len([e for e in events if e['kind']=='group_return']),'no extra dispatch')
    require(not any(e['kind']=='group_outer_failure' for e in events),'outer failure retained not certified');return out


def consume(output,report_sha,prepared,prepare_sha):
    output=Path(output).absolute();require(output.resolve()==output and not output.is_symlink(),'exact own output')
    core.hash_value(report_sha);rp=old.regular(output/'report.json',profile()['final_reserve_bytes'])
    require(sha(rp)==report_sha,'actual report SHA');r=signal.read(rp,profile()['final_reserve_bytes']);p=verify(prepared,prepare_sha)
    require(r['schema']==NAME+'.report' and r['prepared']==str(Path(prepared).absolute()) and r['prepare_sha256']==prepare_sha,'own authority schema')
    require(not r['errors'] and all(r[k] is False for k in FLAGS) and r['sources_verified'] is True
        and r['source_manifest_sha256']==digest(p['sources']),'actual source/run closure')
    require(r['model_before']==r['model_after'] and r['model_before']['original_state_dict']==p['expected_model_hash'],'frozen model')
    require(r['classes_before']==r['classes_after'] and r['classes_before'],'installed model class closure')
    for c in r['classes_before']:require(p['sources'].get(c['path'])==c['sha256'],'actual class source pin')
    require(r['precision']==[old.precision_contract()]*2 and r['host']['device']=='cuda:0' and r['host']['allocation_sum']==4.
        and r['host']['visible_nvidia_nodes'],'HOST CUDA and numerical guards')
    names=set(r['artifact_sha256']);require({'binding.json','events.jsonl'}<=names and all(n in ('binding.json','events.jsonl')
        or n.startswith('signal_s') and n.endswith('.json') and Path(n).name==n for n in names),'explicit artifacts only')
    require({x.name for x in output.iterdir()}==names|{'report.json'},'exact full artifact census')
    for n,h in r['artifact_sha256'].items():require(sha(old.regular(output/n,profile()['output_bytes']))==h,'all saved artifact bytes')
    require(signal.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=prepare_sha),'persisted binding')
    sizes={};es=base.journal(output,r,sizes)
    for phase in ('before','after'):
        require([e for e in es if e['kind']=='model_'+phase]==[dict(kind='model_'+phase,hashes=r['model_'+phase],classes=r['classes_'+phase])],'model journal')
        require([e for e in es if e['kind']=='source_'+phase]==[dict(kind='source_'+phase,sha256=digest(p['sources']))],'source journal')
        require([e['value'] for e in es if e['kind']=='precision_'+phase]==[old.precision_contract()],'precision journal')
    seen=set()
    def capsule_reader(slot,cid,call):
        n=f'signal_s{slot}_{cid:04}.json';require(n in names,'actual parent signal capsule')
        require(signal.read(old.regular(output/n,64*1024**2),64*1024**2)==call and sha(output/n)==digest(call),'exact source signal capsule bytes')
        seen.add(n);return dict(path=str(output/n),sha256=sha(output/n))
    summaries=validate_rows(p,r['rows'],es,capsule_reader=capsule_reader)
    require(seen==names-{'binding.json','events.jsonl'},'no dangling capsules')
    require(r['status']=='four_slots_settled_not_admitted' and len(summaries)==4 and all(x['complete'] for x in summaries),'all four settled including no_witness')
    require(0<=r['wall_seconds']<=p['profile']['wall_seconds'] and sum(x.stat().st_size for x in output.iterdir())<=p['profile']['output_bytes'],'wall/total output caps')
    require(type(r['CUDA_peak_bytes']) is list and len(r['CUDA_peak_bytes'])==2 and all(type(v) is int and 0<=v<=p['profile'][k]
        for v,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'))),'CUDA peak caps')
    for row in r['rows']:
        used=sizes[row['slot']]+len(canonical(row))+sum((output/n).stat().st_size for n in names if n.startswith('signal_s'+str(row['slot'])+'_'))
        require(used<=p['profile']['group_output_bytes'],'512MiB full group bound')
    require(sha(rp)==report_sha and all(sha(f)==h for f,h in p['sources'].items()),'unchanged final source/input bytes')
    return dict(schema=NAME+'.consume',status='actual_cp5_R_model_deliveries_verified_NOT_physical',prepared=r['prepared'],
        prepare_sha256=prepare_sha,report_sha256=report_sha,cp5_tasks=dict(path=str(CP5),sha256=CP5_SHA),
        protocol_sha256=PROTOCOL_SHA,denominator=4,arms=summaries,root_context_role=ROLE,**FLAGS)


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
