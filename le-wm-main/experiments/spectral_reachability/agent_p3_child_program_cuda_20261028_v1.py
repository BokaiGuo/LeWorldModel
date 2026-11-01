"""Paid-child progress compiler and independent replays; MAIN-only CUDA run.

Prepare/plan/verify/consume use saved arrays only. No physical execution, root J
substitution, policy changes, quota shrinking, or full-tree/scientific admission.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import os
from pathlib import Path
import time

import numpy as np

from experiments.spectral_reachability import agent_p3_child_signal_cuda_20261028_v2 as paid
from experiments.spectral_reachability import agent_p3_native_program_20261028_v1 as original

NAME='agent_p3_child_program_cuda_20261028_v1'
ROOT=paid.ROOT
contract,core,base=paid.contract,paid.core,paid.base
require,canonical,digest,sha=paid.require,paid.canonical,paid.digest,paid.sha
FLAGS=dict(paid.FLAGS)
PAID_SHAS=('1a7ea154b09c0caf59b5bc4c4d8b4b333493aaac4b6dc1a1afe1363f9ecc2670',
    '9814e24cafd2be4ccf60d39d7b4721fee2f1225c228d004d8e6ec57676634487',
    'fcb962c30b9e26bcb41588668113003c8c942cd0c719a6f7fe984f7116f05786')
ORIGINAL_SHA='c616c95f463b07bd0356c3855aba96ec7d0fbc492060ab4304854d7be790f44f'
READY='ready_progress'
BLOCKED='blocked_whole_node_search_budget'
UNSUPPORTED='blocked_undefined_reconfiguration'
FINAL='child_extension_saved_not_admitted'
SIGNAL_OUTPUT=ROOT/'results/agent_p3_child_signal_cuda_20261028_v2_cuda_MAIN02'
SIGNAL_REPORT_SHA='cccb96d7d57dcf041021f8c43ab1f027a1012732ec56068ef1fb85348829bf5a'
SIGNAL_RECEIPT=ROOT/'results/consume_child_signal_MAIN_20261028_v1_MAIN02.json'
SIGNAL_RECEIPT_SHA='efccb0ad57ff76e126e7027e9518b1c24b092c806a71a0ed55dcd96fa7ac90b5'


def inputs_template():
    return dict(signal_output=str(SIGNAL_OUTPUT),report_sha256=SIGNAL_REPORT_SHA,
        signal_receipt=str(SIGNAL_RECEIPT),receipt_sha256=SIGNAL_RECEIPT_SHA)


def own_files():
    return [Path(__file__).resolve(),ROOT/f'tests/{NAME}_test.py',ROOT/f'innerdocs/{NAME}.md']


def pins():
    files={str(p):h for p,h in zip(paid.own_files(),PAID_SHAS,strict=True)}
    files[str(Path(original.__file__).resolve())]=ORIGINAL_SHA
    contract.check_sources(files);return files


def rule():
    return dict(schema=NAME+'.rule',allocation=contract.rule(),search_predict_cap=24,
        selected_prefix_predict_reserve=1,fraction_rule='i/(original_count-1); count is unchanged 4 or 8',
        generator='original bounded_ridge Q@same-child J / max(1,max(abs(direction))); cast once float32',
        damping=original.rule()['damping'],joint_raw_l2_cap=original.rule()['joint_raw_l2_cap'],
        reconfiguration='undefined in frozen native generator: retain node and refuse, never relabel progress',
        selection='original Q terminal mean-square then original candidate index, AFTER all candidates',
        replay='each candidate whole H then independent segmented H, every boundary exact; selected raw5 independently replayed',
        max_local_extension_depth=1,physical_prefix_is_not_model_composition=True,
        future_descendants_require_new_paid_signals=True,no_padding=True,no_retry=True,outcome_input=False,
        P1_or_C3=False,equal_FLOPs_or_wall=False)


def eligibility(decision):
    h,n=decision['horizon'],decision['candidates']
    require(type(h) is int and h in (1,3) and type(n) is int and n in (4,8),'original H1/H3 and 4/8 quota')
    if 2*h*n>24:return BLOCKED
    if decision['extension_type']!='progress' or decision['program_type']!='progress':return UNSUPPORTED
    require(decision['continue_expanding'] is True,'original local depth0 expansion')
    return READY


def branches_from(child_contract,receipt):
    tasks=child_contract['tasks'];children=child_contract['children']
    require(len(tasks)==2 and len(children)==24 and len(receipt['rows'])==24 and
        receipt['fixed_denominator']==48 and receipt['child_denominator']==24,'full paid 24child/48method denominator')
    require([(c['episode_id'],c['boundary']) for c in children]==[(e,b) for e in contract.EPISODES for b in contract.STEPS],
        'fixed original child order')
    out=[]
    for child,row in zip(children,receipt['rows'],strict=True):
        child=contract.validate_child(child,tasks)
        require(row['child_id']==child['child_id'],'same paid child identity/order')
        for mode in contract.MODES:
            branch=dict(child_id=child['child_id'],root_slot=child['root_slot'],mode=mode,
                child_sha256=digest(child),parent_context_sha256=child['context_sha256'],
                status='not_reached',decision=None,queue_sha256=None,response=None,
                signal_group=None,signal_charge=None,signal_receipt_sha256=None,
                reserved_search_predict_API=None,search_predict_cap=24,selected_prefix_predict_reserve=1)
            if child['status']=='not_reached':
                require(row==dict(child_id=child['child_id'],status='not_reached',result=None,error=None),'unreached unknown')
                out.append(branch);continue
            require(row['status']=='returned' and row['error'] is None,'complete paid signal only, no partial child')
            result=row['result'];require([x['mode'] for x in result['rows']]==list(contract.MODES),'fixed/adaptive paid order')
            item=result['rows'][contract.MODES.index(mode)];d=item['decision'];status=eligibility(d)
            require(item['status']==(BLOCKED if 2*d['horizon']*d['candidates']>24 else 'pending_reviewed_child_compiler') and
                item['reserved_search_predict_API']==2*d['horizon']*d['candidates'] and
                item['search_predict_cap']==24 and item['selected_prefix_predict_reserve']==1,'original search allocation')
            acquisitions=[a for a in result['acquisitions'] if a['group']==item['signal_group']]
            require(len(acquisitions)==1 and acquisitions[0]['status']=='returned','own actual acquisition')
            a=acquisitions[0];responses=a['signal']['responses']
            signals=core.FrozenSignals(**a['signal']['scheduling_signal'])
            require(canonical(d)==canonical(core.choose(tasks[child['root_slot']],dict(context=child['context'],depth=0),signals,mode)),
                'unchanged original same-child chooser, no post-signal quota adjustment')
            require([r['horizon'] for r in responses]==[1,3],'own paid H1/H3 pair')
            response=next(r for r in responses if r['horizon']==d['horizon'])
            require(response['binding_sha256']==digest(tasks[child['root_slot']]['binding']) and
                response['parent_context_sha256']==child['context_sha256'] and response['producer_sha256']==sha(paid.__file__),
                'same observed child response, never root/stale/cross-mode substitution')
            q=item['queue'];require(len(q)==d['candidates'] and all(req['candidate_index']==k and
                req['parent_context']==child['context'] and req['binding']==tasks[child['root_slot']]['binding'] and
                req['child_id']==child['child_id'] and req['actual_executed_prefix']==child['executed_prefix'] and
                req['local_response']==responses for k,req in enumerate(q)),'unchanged full paid queue')
            branch.update(status=status,decision=deepcopy(d),queue_sha256=digest(q),response=deepcopy(response),
                signal_group=item['signal_group'],signal_charge=deepcopy(item['signal_charge']),
                signal_receipt_sha256=digest(a),reserved_search_predict_API=2*d['horizon']*d['candidates'])
            out.append(branch)
    require(len(out)==48,'never drop refused/not reached methods');return out


def caps(branch):
    result=dict.fromkeys(core.AXES,0)
    if branch['status']!=READY:return result
    h,n=(branch['decision'][k] for k in ('horizon','candidates'));calls=2*h*n+1
    result.update(predict_API=calls,predict_samples=calls,predict_context_tokens=3*calls,
        action_encoder_calls=calls,action_encoder_samples=calls,action_encoder_context_tokens=3*calls,
        predictor_calls=calls,predictor_samples=calls,predictor_context_tokens=3*calls,
        pred_proj_calls=calls,pred_proj_rows=3*calls,array_SVD_calls=1,solve_calls=1,
        program_compile_calls=1,replay_calls=2*n+1,metric_calls=n)
    return result


def profile(branches):
    totals={axis:sum(caps(b)[axis] for b in branches) for axis in core.AXES}
    return dict(resource_caps=totals,output_bytes=512*1024**2,final_reserve_bytes=256*1024**2,
        wall_seconds=900,disk_floor_bytes=200*1024**2,cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        TF32=False,SDP='MATH',CPU_neural=False,physics_steps=0,no_retry=True,no_padding=True,
        wall_scope='before verify through final guards and journal close; final report write excluded',
        no_new_J=True,paid_signal_cost='historical actual independent/shared charge retained, NOT free',
        search_cap_per_method=24,selected_prefix_cap_per_method=1)


def exclusive(path):
    p=contract.absolute(path)
    require(p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and not p.exists(),'NEW exclusive own result')
    return p


def write_new(path,value):
    path=exclusive(path);raw=canonical(value);require(len(raw)<=256*1024**2,'bounded complete metadata')
    with path.open('xb') as f:
        require(f.write(raw)==len(raw),'complete metadata write');f.flush();os.fsync(f.fileno())
    base.fsync_dir(path.parent);return dict(output=str(path),sha256=sha(path),bytes=len(raw))


def material(inputs):
    core.keys(inputs,('signal_output','report_sha256','signal_receipt','receipt_sha256'))
    require(inputs==inputs_template(),'fixed actual MAIN02 report/complete strict receipt, no cohort replacement')
    pins();output=contract.absolute(inputs['signal_output']);saved=contract.absolute(inputs['signal_receipt'])
    for k in ('report_sha256','receipt_sha256'):core.hash_value(inputs[k])
    require(sha(saved)==inputs['receipt_sha256'],'external actual persisted paid receipt SHA')
    # Mandatory complete producer consumer BEFORE projecting any decisions/J.
    actual=paid.consume(str(output),inputs['report_sha256'])
    receipt=base.read(saved,256*1024**2)
    require(canonical(receipt)==canonical(actual),'persisted receipt is complete strict v2.consume, not prepare/status summary')
    report=base.read(output/'report.json',256*1024**2)
    prepared_path=contract.absolute(report['prepared'])
    require(sha(prepared_path)==report['prepare_sha256'],'actual paid prepare before projection')
    p=base.read(prepared_path,64*1024**2)
    require(p['schema']==paid.NAME+'.prepare' and receipt['schema']==paid.NAME+'.consume','v2 real paid schema only')
    c=p['child_contract'];branches=branches_from(c,receipt)
    sources=deepcopy(p['sources']);contract.merge(sources,pins())
    extra=[*own_files(),Path(original.generator.__file__).resolve(),
        ROOT/'experiments/spectral_reachability/pusht_full_packet_chart_20261027.py',
        saved,contract.absolute(report['prepared']),*(output/n for n in ('report.json','binding.json',paid.JOURNAL))]
    contract.merge(sources,{str(f):sha(f) for f in extra});contract.check_sources(sources)
    require(sha(saved)==inputs['receipt_sha256'] and sha(output/'report.json')==inputs['report_sha256'] and
        sha(prepared_path)==report['prepare_sha256'],'paid inputs unchanged')
    return dict(schema=NAME+'.prepare',inputs=deepcopy(inputs),tasks=deepcopy(c['tasks']),children=deepcopy(c['children']),
        branches=branches,rule=rule(),profile=profile(branches),sources=sources,policy=p['policy'],
        expected_model_hash=p['expected_model_hash'],historical_signal_resources=receipt['resources'],
        historical_signal_sharing=report['sharing'],paid_saved_consumer_array_SVD_calls=receipt['saved_verification']['array_SVD_calls'],
        fixed_denominator=48,child_denominator=24,independent_roots=2,local_extension_depth=1,
        future_child_signals_reused=False,physical_dispatch=False,status='child_compiler_prepared_not_admitted',**FLAGS)


def prepare_value(path,expected):
    path=contract.absolute(path);core.hash_value(expected)
    require(sha(path)==expected,'external actual input manifest SHA')
    inputs=base.read(path,1024**2);value=material(inputs)
    value['input_manifest']=dict(path=str(path),sha256=expected)
    contract.merge(value['sources'],{str(path):expected})
    require(sha(path)==expected,'input manifest unchanged');return value


def verify(path,expected):
    path=contract.absolute(path);core.hash_value(expected);require(sha(path)==expected,'external actual compiler prepare SHA')
    p=base.read(path,64*1024**2);contract.check_sources(p['sources'])
    manifest=p['input_manifest'];core.keys(manifest,('path','sha256'))
    require(canonical(p)==canonical(prepare_value(manifest['path'],manifest['sha256'])) and sha(path)==expected,
        'full paid inputs/source/allocation rebuild')
    return p


def plan(p):
    require(p['schema']==NAME+'.prepare' and p['rule']==rule() and len(p['branches'])==48 and
        all(p[k] is False for k in FLAGS),'complete closed compiler prepare')
    counts={status:sum(b['status']==status for b in p['branches']) for status in ('not_reached',BLOCKED,UNSUPPORTED,READY)}
    require(sum(counts.values())==48,'full unfiltered refusal denominator')
    return dict(schema=NAME+'.plan',fixed_denominator=48,child_denominator=24,independent_roots=2,
        counts=counts,all_observed_refused=(counts[READY]==0),branches=deepcopy(p['branches']),profile=p['profile'],
        missing=['reviewed MAIN HOST CUDA execution and strict consume','undefined reconfiguration generator',
            'fresh descendant paid signals/full composition tree','live physical bridge/P1/applicable P2/fresh cohort/C3'],**FLAGS)


def generate(task,child,branch,ledger):
    task=core.validate_task(task);child=contract.validate_child(child,[task,task])
    # validate_child checks the supplied slot task without altering original root binding.
    require(branch['status']==READY and branch['child_id']==child['child_id'] and branch['child_sha256']==digest(child) and
        branch['parent_context_sha256']==child['context_sha256'] and eligibility(branch['decision'])==READY,'eligible same-child progress only')
    r=branch['response'];h,n=(branch['decision'][k] for k in ('horizon','candidates'))
    core.keys(r,('binding_sha256','parent_context_sha256','horizon','nominal','jacobian','producer_sha256'))
    require(type(r['horizon']) is int and r['horizon']==h and r['binding_sha256']==digest(task['binding']) and
        r['parent_context_sha256']==child['context_sha256'] and r['producer_sha256']==sha(paid.__file__),'same paid child J, not root J')
    j=core.finite(r['jacobian'],(192,10*h),True);nominal=core.finite(r['nominal'],(192,),True)
    q=np.asarray(task['projection'],np.float64);residual=q@(np.asarray(task['goal'])-nominal)
    solve=ledger.call('solve',lambda:ledger.call('array_svd',lambda:original.generator.bounded_ridge(q@j,residual,
        damping=rule()['damping'],cap=rule()['joint_raw_l2_cap'])))
    direction=np.asarray(solve['params'],np.float64);divisor=max(1.,float(np.max(np.abs(direction))))
    fractions=[i/(n-1) for i in range(n)]
    return dict(solve=solve,divisor=divisor,fractions=fractions,
        actions=[(f*direction/divisor).astype(np.float32).reshape(5*h,2).tolist() for f in fractions],
        response_sha256=digest(r),child_sha256=digest(child),rule=rule())


def initial_rows(branches):
    return [dict(child_id=b['child_id'],root_slot=b['root_slot'],mode=b['mode'],branch_sha256=digest(b),
        status='not_started' if b['status']==READY else b['status'],bank=None,candidates=None,selected=None,
        resource_receipt=None,error=None) for b in branches]


def scope(branch,index,phase,horizon):
    return dict(child_id=branch['child_id'],root_slot=branch['root_slot'],mode=branch['mode'],index=index,
        phase=phase,horizon=horizon,parent_context_sha256=branch['parent_context_sha256'])


def execute_branch(task,child,branch,row,factory,sink,*,permit_ram=False,authorization=None):
    require(branch['status']==READY and row['status']=='not_started','no dispatch of refused/unreached nodes')
    require(permit_ram is True or authorization=='MAIN_APPROVED_HOST_CHILD_PROGRAM_CUDA','authorization BEFORE backend factory')
    ledger=core.Ledger(caps(branch),sink=sink.event);row['status']='attempted'
    try:
        backend=factory(child,branch,sink)
        require((permit_ram is True and backend.kind=='RAM_FIXTURE') or
            (authorization=='MAIN_APPROVED_HOST_CHILD_PROGRAM_CUDA' and backend.kind=='HOST_CUDA' and
            callable(getattr(backend,'event_sink',None))),'explicit RAM or durable MAIN CUDA backend only')
        h,n=(branch['decision'][k] for k in ('horizon','candidates'));parent=child['context']
        ledger.scope=scope(branch,None,'compile',h)
        row['bank']=ledger.call('program_compile',lambda:generate(task,child,branch,ledger))
        sink.event(dict(kind='bank',branch_sha256=digest(branch),bank=deepcopy(row['bank'])))
        row['candidates']=[]
        for k,actions in enumerate(row['bank']['actions']):
            candidate=dict(index=k,actions=deepcopy(actions),whole=None,segmented=None,cost=None)
            row['candidates'].append(candidate)
            for phase in ('whole','segmented'):
                ledger.scope=scope(branch,k,phase,h)
                candidate[phase]=ledger.call('replay',lambda phase=phase:backend.replay(parent,actions,phase,ledger))
                core.checked_path(task,parent,actions,candidate[phase])
            require(candidate['whole']==candidate['segmented'],'all whole/segmented boundaries exact')
            ledger.scope=scope(branch,k,'metric',h)
            candidate['cost']=ledger.call('metric',lambda:core.model_cost(task,candidate['whole'][-1]))
        require(len(row['candidates'])==n,'full original candidate quota before selection')
        w=min(row['candidates'],key=lambda x:(x['cost'],x['index']))
        ledger.scope=scope(branch,w['index'],'selected_prefix',1);raw5=w['actions'][:5]
        prefix=ledger.call('replay',lambda:backend.replay(parent,raw5,'selected_prefix',ledger))
        core.checked_path(task,parent,raw5,prefix)
        require(prefix==w['whole'][:1],'independent selected prefix equals sealed first candidate boundary')
        row['selected']=dict(index=w['index'],terminal_cost=w['cost'],raw5=deepcopy(raw5),prefix=prefix,
            terminal_context_sha256=digest(w['whole'][-1]),selection_digest=digest(row['candidates']),
            parent_context_sha256=child['context_sha256'],physical_release=False)
        require(ledger.attempted==ledger.completed==caps(branch),'exact actual search plus selected-prefix costs')
        row['status']=FINAL
    except BaseException as exc:
        row.update(status='failed',error=type(exc).__name__+': '+str(exc));raise
    finally:row['resource_receipt']=ledger.receipt()


class NativeProgram(base.NativeSignal):
    def __init__(self,model,device,task,child,store,tasks):
        self.child=contract.validate_child(child,tasks)
        require(self.child['status']=='observed_child' and self.child['ancestry']['task_binding_sha256']==digest(task['binding']),
            'exact observed child and original task')
        super().__init__(model,device,task,store)

    def replay(self,parent,actions,phase,ledger):
        require(getattr(self.device,'type',None)=='cuda' and parent==self.child['context'],'exact child CUDA context before torch')
        require(phase in ('whole','segmented','selected_prefix'),'explicit independent replay phase')
        raw=core.finite(actions,(len(actions),2),True)
        require(len(raw) in (5,15) and np.max(np.abs(raw))<=1 and
            (phase!='selected_prefix' or len(raw)==5),'literal original H packets without padding')
        import torch
        self.frozen();self.ledger=ledger;self.phase=phase
        state=torch.tensor(parent['latent_history'],dtype=torch.float32,device=self.device)[None]
        history=torch.tensor(parent['action_history'],dtype=torch.float32,device=self.device)[None]
        mean=torch.tensor(self.task['action_mean'],dtype=torch.float32,device=self.device)
        std=torch.tensor(self.task['action_std'],dtype=torch.float32,device=self.device);path=[]
        with torch.no_grad(),self.observe():
            for step,packet in enumerate(raw.astype(np.float32).reshape(-1,5,2)):
                self.step=step
                if phase=='segmented' and path:
                    state=torch.tensor(path[-1]['latent_history'],dtype=torch.float32,device=self.device)[None]
                    history=torch.tensor(path[-1]['action_history'],dtype=torch.float32,device=self.device)[None]
                value=torch.tensor(packet,dtype=torch.float32,device=self.device)
                token=((value-mean)/std).reshape(10);outgoing=torch.cat((history,token[None,None]),dim=1)
                self.arrays('program_packet',latent=state,history=history,raw=value,mean=mean,std=std,token=token,outgoing=outgoing)
                encoded=self.model.action_encoder(outgoing)
                predicted=self.call('predict',lambda:self.model.predict(state,encoded),dict(latent=state,act_emb=encoded),shape=list(state.shape))
                require(tuple(predicted.shape)==(1,3,192),'actual full native prediction ABI')
                state=torch.cat((state[:,1:],predicted[:,-1:]),dim=1);history=outgoing[:,1:]
                self.arrays('program_boundary',latent=state,history=history)
                path.append(dict(parent,latent_history=state[0].cpu().tolist(),action_history=history[0].cpu().tolist()))
        self.frozen();core.checked_path(self.task,parent,actions,path);return path


class Tagged:
    def __init__(self,store,branch):self.store,self.group=store,branch['child_id']+':'+branch['mode']
    def event(self,e):self.store.event(dict(kind='compiler_trace',group=self.group,payload=deepcopy(e)))


def totals(rows):
    out={k:dict.fromkeys(core.AXES,0) for k in ('attempted','completed')}
    for row in rows:
        r=row['resource_receipt']
        if r is None:continue  # explicitly never dispatched, not an imputed physical/model outcome
        core.audit_ledger(r)
        for kind in out:
            for axis in core.AXES:out[kind][axis]+=r[kind][axis]
    return dict(**out,FLOPs=None,cuda_ns=None,backward_engine_calls=None,physics_steps=0,nested_wall_not_added=True)


def run(prepared,expected,output,authorization):
    require(authorization=='MAIN_APPROVED_HOST_CHILD_PROGRAM_CUDA','MAIN authorization before verify/model import')
    output=exclusive(output);started=time.monotonic();p=verify(prepared,expected)
    store=paid.Store(output,p['profile']);store.started=started;rows=initial_rows(p['branches'])
    errors=[];before=after=host=None;model=None;sources_ok=False;cleanup=False
    active=any(b['status']==READY for b in p['branches'])
    try:
        store.put('binding.json',dict(prepared=str(prepared),prepare_sha256=expected))
        store.event(dict(kind='denominator',rows=deepcopy(rows),fixed_denominator=48,child_denominator=24))
        if active:
            os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
            import torch
            from torch.nn.attention import SDPBackend,sdpa_kernel
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            store.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
            store.check();model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
            before=base.model_hash(model);require(before['original_state_dict']==p['expected_model_hash'],'exact frozen model')
            for module in model.modules():
                path=str(Path(inspect.getfile(type(module))).resolve())
                require(path in p['sources'] and sha(path)==p['sources'][path],'sealed loaded native class source')
        store.event(dict(kind='model_before',hashes=before,native_dispatch=active))
        if active:
            def factory(child,branch,sink):return NativeProgram(model,device,p['tasks'][child['root_slot']],child,sink,p['tasks'])
            with torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
                for i,(branch,row) in enumerate(zip(p['branches'],rows,strict=True)):
                    if branch['status']!=READY:continue
                    store.check();child=p['children'][i//2];sink=Tagged(store,branch)
                    execute_branch(p['tasks'][child['root_slot']],child,branch,row,factory,sink,authorization=authorization)
                    store.event(dict(kind='branch_return',row=deepcopy(row)))
            torch.cuda.synchronize(device)
        require(totals(rows)['attempted']==totals(rows)['completed']==p['profile']['resource_caps'],'full eligible resources')
        store.check();cleanup=True
    except BaseException as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)))
    finally:
        for row in rows:
            if row['status']=='not_started':row['status']='blocked_not_dispatched'
            elif row['status']=='attempted':row['status']='failed'
        if model is not None:
            try:
                after=base.model_hash(model);require(before==after,'model unchanged including all buffers')
                require(all(not m.training for m in model.modules()) and all(not v.requires_grad for v in model.parameters()),'frozen eval')
            except BaseException as exc:errors.append(dict(stage='model_final',message=str(exc)))
        try:store.event(dict(kind='model_after',hashes=after,native_dispatch=active))
        except BaseException as exc:errors.append(dict(stage='model_journal',message=str(exc)))
        try:
            contract.check_sources(p['sources']);pins();sources_ok=True
            store.event(dict(kind='sources_final',sha256=digest(p['sources'])));store.check()
        except BaseException as exc:errors.append(dict(stage='source_or_limits',message=str(exc)))
        try:store.close()
        except BaseException as exc:errors.append(dict(stage='close',message=str(exc)))
        complete=not errors and sources_ok and cleanup and all(row['status']==(FINAL if b['status']==READY else b['status'])
            for b,row in zip(p['branches'],rows,strict=True))
        report=dict(schema=NAME+'.report',status='child_panel_saved_not_admitted' if complete else 'partial_or_failed',
            prepared=str(prepared),prepare_sha256=expected,rows=rows,resources=totals(rows),errors=errors,
            model_before=before,model_after=after,native_dispatch=active,host=host,sources_verified=sources_ok,cleanup_ok=cleanup,
            source_manifest_sha256=digest(p['sources']),fixed_denominator=48,child_denominator=24,
            wall_seconds=store.clock()-store.started,CUDA_peak_bytes=store.memory() if store.memory else None,
            journal_tail=store.tail,artifact_sha256={n:sha(output/n) for n in ('binding.json',paid.JOURNAL)},**FLAGS)
        store.put('report.json',report)
    require(complete,'failed engineering evidence retained, no release: '+repr(errors))
    return dict(output=str(output),report_sha256=sha(output/'report.json'),counts=plan(p)['counts'],resources=report['resources'])


def validate_arrays(task,child,branch,row,events):
    """Saved full native wiring and paid-entry audit; no model/AD recomputation."""
    receipt=row['resource_receipt'];core.audit_ledger(receipt)
    require(receipt['caps']==receipt['attempted']==receipt['completed']==caps(branch),'exact new same-unit costs')
    allowed={'resource_attempt','resource_return','io_input','io_return','program_packet','program_boundary','bank'}
    require(all(e['kind'] in allowed for e in events),'closed full compiler trace vocabulary')
    expected=[];operations=receipt['events'];stack=[];linked={};top=[]
    for op in operations:
        core.keys(op,('index','scope','resource','shape','directions','delta','status','started_ns','ended_ns','error'))
        require(op['status']=='returned' and op['error'] is None,'all actual entries returned')
        expected.extend([dict(kind='resource_attempt',entry=dict(op,status='attempted',ended_ns=None)),dict(kind='resource_return',entry=op)])
    actual=[e for e in events if e['kind'].startswith('resource_')]
    require(sorted(map(canonical,actual))==sorted(map(canonical,expected)),'full attempt/return journal bijection')
    for event in events:
        kind=event['kind']
        if kind.startswith('resource_'):
            core.keys(event,('kind','entry'));op=event['entry']
            if kind=='resource_attempt':
                parent=operations[stack[-1]]['resource'] if stack else None
                expected_parent={'program_compile':None,'solve':'program_compile','array_svd':'solve',
                    'replay':None,'metric':None,'action_encoder':'replay','predict':'replay',
                    'predictor':'predict','pred_proj':'predict'}
                require(op['resource'] in expected_parent and parent==expected_parent[op['resource']],
                    'actual native module/compiler nesting')
                if not stack:top.append((op['resource'],op['scope']))
                stack.append(op['index'])
            else:require(stack and stack.pop()==op['index'],'actual nested native order')
        elif kind=='bank':
            core.keys(event,('kind','branch_sha256','bank'));require(not stack,'bank sealed after compile return')
        else:
            core.keys(event,('kind','scope','phase','step','operation_index','arrays'))
            require(stack and event['scope']==operations[stack[-1]]['scope'] and
                event['phase']==event['scope']['phase'],'IO inside same paid dispatch')
            for blob in event['arrays'].values():require(base.unblob(blob).dtype==np.float32,'full native float32, no repair')
            if kind.startswith('io_'):
                index=event['operation_index'];require(type(index) is int and stack[-1]==index,'literal actual IO index')
                op=operations[index];resource=op['resource']
                require(resource in ('predict','action_encoder','predictor','pred_proj'),'only allowed native module IO')
                names={'predict':('latent','act_emb'),'action_encoder':('0',),'predictor':('0','1'),'pred_proj':('0',)}[resource]
                core.keys(event['arrays'],(resource+':output',) if kind=='io_return' else tuple(resource+':'+n for n in names))
                if kind=='io_input':require(event['arrays'][resource+':'+names[0]]['shape']==op['shape'],'actual charged shape')
                linked.setdefault(index,[]).append(kind)
            else:require(event['operation_index'] is None,'packet/boundary cannot borrow module operation')
    require(not stack,'no dangling native calls')
    for op in operations:
        if op['resource'] in ('predict','action_encoder','predictor','pred_proj'):
            require(linked.get(op['index'])==['io_input','io_return'],'all actual module IO present')
    paths=[];h=branch['decision']['horizon']
    for candidate in row['candidates']:
        for phase in ('whole','segmented'):
            paths.append((candidate['index'],phase,h,candidate['actions'],candidate[phase]))
    selected=row['selected'];paths.append((selected['index'],'selected_prefix',1,selected['raw5'],selected['prefix']))
    expected_scopes=[scope(branch,None,'compile',h)]
    for candidate in row['candidates']:
        expected_scopes.extend(scope(branch,candidate['index'],phase,h) for phase in ('whole','segmented','metric'))
    expected_scopes.append(scope(branch,selected['index'],'selected_prefix',1))
    require(all(canonical(op['scope']) in list(map(canonical,expected_scopes)) for op in operations),
        'own literal child/mode/candidate scope only')
    top_expected=[('program_compile',scope(branch,None,'compile',h))]
    for k in range(len(row['candidates'])):
        top_expected.extend((resource,scope(branch,k,phase,h)) for resource,phase in
            (('replay','whole'),('replay','segmented'),('metric','metric')))
    top_expected.append(('replay',scope(branch,selected['index'],'selected_prefix',1)))
    require(canonical(top)==canonical(top_expected),'compile/search/all metrics before selected-prefix order')
    require([op['scope'] for op in operations if op['resource']=='program_compile']==[expected_scopes[0]] and
        [op['scope'] for op in operations if op['resource']=='solve']==[expected_scopes[0]] and
        [op['scope'] for op in operations if op['resource']=='array_svd']==[expected_scopes[0]],'one own ridge compilation')
    require([op['scope'] for op in operations if op['resource']=='replay']==[scope(branch,k,phase,horizon) for k,phase,horizon,actions,path in paths] and
        [op['scope'] for op in operations if op['resource']=='metric']==[scope(branch,k,'metric',h) for k in range(len(row['candidates']))],
        'full original candidate/replay/metric and selected-prefix order')
    array_count=0
    for k,phase,horizon,actions,path in paths:
        s=scope(branch,k,phase,horizon);local=[e for e in events if e.get('scope')==s]
        packets=[e for e in local if e['kind']=='program_packet'];boundaries=[e for e in local if e['kind']=='program_boundary']
        require(len(packets)==len(boundaries)==horizon and len([e for e in local if e['kind']=='io_input'])==4*horizon,
            'exact all native packet/module boundaries')
        before=child['context'];core.checked_path(task,before,actions,path)
        for step,(pe,be,after) in enumerate(zip(packets,boundaries,path,strict=True)):
            require(type(pe['step']) is type(be['step']) is int and pe['step']==be['step']==step,'ordered literal steps')
            packet={key:base.unblob(v) for key,v in pe['arrays'].items()};boundary={key:base.unblob(v) for key,v in be['arrays'].items()}
            core.keys(packet,('latent','history','raw','mean','std','token','outgoing'));core.keys(boundary,('latent','history'))
            raw=np.asarray(actions[5*step:5*step+5],np.float32)
            latent=np.asarray(before['latent_history'],np.float32)[None];hist=np.asarray(before['action_history'],np.float32)[None]
            mean,std=(np.asarray(task[key],np.float32) for key in ('action_mean','action_std'))
            token=((raw-mean)/std).reshape(10);outgoing=np.concatenate((hist,token[None,None]),axis=1)
            for key,value in dict(latent=latent,history=hist,raw=raw,mean=mean,std=std,token=token,outgoing=outgoing).items():
                require(np.array_equal(packet[key],value),'exact real child/raw/normalizer packet: '+key)
            def io(resource,returned=False):
                found=[e for e in local if e['kind']==('io_return' if returned else 'io_input') and e['step']==step and
                    any(key.startswith(resource+':') for key in e['arrays'])]
                require(len(found)==1,'one full native module per packet')
                return {key:base.unblob(v) for key,v in found[0]['arrays'].items()}
            ai,ao=io('action_encoder'),io('action_encoder',True);pi,po=io('predict'),io('predict',True)
            ni,no=io('predictor'),io('predictor',True);qi,qo=io('pred_proj'),io('pred_proj',True)
            require(np.array_equal(ai['action_encoder:0'],outgoing) and np.array_equal(pi['predict:latent'],latent) and
                np.array_equal(pi['predict:act_emb'],ao['action_encoder:output']) and np.array_equal(ni['predictor:0'],latent) and
                np.array_equal(ni['predictor:1'],ao['action_encoder:output']) and
                np.array_equal(qi['pred_proj:0'],no['predictor:output'].reshape(3,192)) and
                np.array_equal(qo['pred_proj:output'].reshape(1,3,192),po['predict:output']),'full native AE/predict/predictor/pred_proj closure')
            require(np.array_equal(boundary['latent'],np.concatenate((latent[:,1:],po['predict:output'][:,-1:]),axis=1)) and
                np.array_equal(boundary['history'],outgoing[:,1:]) and np.array_equal(boundary['latent'][0],after['latent_history']) and
                np.array_equal(boundary['history'][0],after['action_history']),'exact native endpoint; no graph/root substitution')
            before=after
        array_count+=sum('arrays' in e for e in local)
    require(array_count==sum('arrays' in e for e in events),'no extra/unknown native arrays')


def validate_branch(task,child,branch,row,events):
    expected=initial_rows([branch])[0]
    core.keys(row,tuple(expected))
    require(all(canonical(row[k])==canonical(expected[k]) for k in ('child_id','root_slot','mode','branch_sha256')) and row['status']==FINAL and
        row['error'] is None,'complete exact original branch')
    ledger=core.Ledger(caps(branch));ledger.scope=scope(branch,None,'compile',branch['decision']['horizon'])
    bank=ledger.call('program_compile',lambda:generate(task,child,branch,ledger))
    require(canonical(row['bank'])==canonical(bank) and [e for e in events if e['kind']=='bank']==[
        dict(kind='bank',branch_sha256=digest(branch),bank=bank)],'independent saved same-child ridge/actions')
    n=branch['decision']['candidates'];require(len(row['candidates'])==n,'full original candidate denominator')
    for k,candidate in enumerate(row['candidates']):
        core.keys(candidate,('index','actions','whole','segmented','cost'))
        require(type(candidate['index']) is int and candidate['index']==k and candidate['actions']==bank['actions'][k] and
            candidate['whole']==candidate['segmented'],'all original generated actions and independent replay equality')
        endpoint=core.checked_path(task,child['context'],candidate['actions'],candidate['whole'])
        require(type(candidate['cost']) is float and candidate['cost']==core.model_cost(task,endpoint),'original Q terminal MSE, not native success')
    w=min(row['candidates'],key=lambda x:(x['cost'],x['index']))
    selected=dict(index=w['index'],terminal_cost=w['cost'],raw5=w['actions'][:5],prefix=w['whole'][:1],
        terminal_context_sha256=digest(w['whole'][-1]),selection_digest=digest(row['candidates']),
        parent_context_sha256=child['context_sha256'],physical_release=False)
    require(canonical(row['selected'])==canonical(selected),'stable selection AFTER full search and independent exact selected prefix')
    validate_arrays(task,child,branch,row,events)


def consume(output,expected):
    output=contract.absolute(output);core.hash_value(expected);require(sha(output/'report.json')==expected,'external compiler report SHA')
    r=base.read(output/'report.json',256*1024**2);p=verify(r['prepared'],r['prepare_sha256']);active=any(b['status']==READY for b in p['branches'])
    core.keys(r,('schema','status','prepared','prepare_sha256','rows','resources','errors','model_before','model_after',
        'native_dispatch','host','sources_verified','cleanup_ok','source_manifest_sha256','fixed_denominator','child_denominator',
        'wall_seconds','CUDA_peak_bytes','journal_tail','artifact_sha256',*FLAGS))
    require(r['schema']==NAME+'.report' and r['status']=='child_panel_saved_not_admitted' and r['errors']==[] and
        r['fixed_denominator']==48 and r['child_denominator']==24 and all(r[k] is False for k in FLAGS) and
        r['sources_verified'] is r['cleanup_ok'] is True and r['source_manifest_sha256']==digest(p['sources']) and
        r['native_dispatch'] is active,'complete frozen source/closed gates/denominator')
    require({f.name for f in output.iterdir()}=={'report.json','binding.json',paid.JOURNAL} and
        set(r['artifact_sha256'])=={'binding.json',paid.JOURNAL},'exact native artifact set')
    require(sum(f.stat().st_size for f in output.iterdir())<=p['profile']['output_bytes'] and
        (output/paid.JOURNAL).stat().st_size+(output/'binding.json').stat().st_size<=p['profile']['output_bytes']-p['profile']['final_reserve_bytes'],
        'actual compressed storage partitions')
    require(base.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'actual prepare execution edge')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'actual artifact SHA')
    require(len(r['rows'])==48,'all48 methods incl refused/unreached')
    initial=initial_rows(p['branches']);ready=[]
    for i,(b,row) in enumerate(zip(p['branches'],r['rows'],strict=True)):
        if b['status']!=READY:require(canonical(row)==canonical(initial[i]),'refused/unreached unchanged, unknown not zero')
        else:ready.append((b,row,p['children'][i//2]))
    require(r['model_before']==r['model_after'],'unchanged original frozen model')
    if active:
        for hashes in (r['model_before'],r['model_after']):
            core.keys(hashes,('original_state_dict','all_buffers'))
            for h in hashes.values():core.hash_value(h)
        require(r['model_before']['original_state_dict']==p['expected_model_hash'],'bound original model state_dict')
        core.keys(r['host'],('device','index','name','allocation_sum','torch_version','torch_cuda','visible_nvidia_nodes'))
        require(r['host']['device']=='cuda:0' and type(r['host']['index']) is int and r['host']['index']==0 and
            r['host']['allocation_sum']==4. and all(type(r['host'][k]) is str and r['host'][k] for k in ('name','torch_version','torch_cuda')) and
            type(r['host']['visible_nvidia_nodes']) is list and r['host']['visible_nvidia_nodes'],'HOST metadata, not hardware attestation')
        require(type(r['CUDA_peak_bytes']) in (list,tuple) and len(r['CUDA_peak_bytes'])==2 and
            all(type(v) is int and 0<=v<=p['profile'][key] for v,key in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'))),
            'actual bounded CUDA peak')
    else:require(r['model_before'] is r['model_after'] is r['host'] is r['CUDA_peak_bytes'] is None,'all refused: no fictitious CUDA work')
    stage='denominator';position=0;trace=[];tail='0'*64
    for e,tail in paid.iter_journal(output/paid.JOURNAL):
        kind=e['kind']
        if stage=='denominator':
            require(e==dict(kind='denominator',rows=initial,fixed_denominator=48,child_denominator=24),'denominator before model dispatch')
            stage='before';continue
        if stage=='before':
            require(e==dict(kind='model_before',hashes=r['model_before'],native_dispatch=active),'exact pre-model guard')
            stage='work';continue
        if kind=='compiler_trace':
            core.keys(e,('kind','group','payload'));require(stage=='work' and position<len(ready),'guarded current child trace')
            branch=ready[position][0];require(e['group']==branch['child_id']+':'+branch['mode'],'no cross-child or cross-mode trace')
            trace.append(e['payload'])
        elif kind=='branch_return':
            core.keys(e,('kind','row'));require(stage=='work' and position<len(ready),'exact branch order')
            branch,row,child=ready[position];require(e['row']==row,'durable full branch before release')
            validate_branch(p['tasks'][child['root_slot']],child,branch,row,trace);trace=[];position+=1
        elif kind=='model_after':
            require(stage=='work' and position==len(ready) and not trace and
                e==dict(kind='model_after',hashes=r['model_after'],native_dispatch=active),'all work within model guards')
            stage='sources'
        elif kind=='sources_final':
            require(stage=='sources' and e==dict(kind='sources_final',sha256=digest(p['sources'])),'exact final source guard');stage='done'
        else:raise ValueError('unknown/duplicate/out of order journal record')
    require(stage=='done' and tail==r['journal_tail'],'complete footer/hash chain/terminal guards')
    total=totals(r['rows']);require(r['resources']==total and total['attempted']==total['completed']==p['profile']['resource_caps'],'all actual resources, no caps-as-actual')
    require(type(r['wall_seconds']) in (int,float) and 0<=r['wall_seconds']<=p['profile']['wall_seconds'],'actual wall scope')
    contract.check_sources(p['sources']);pins()
    require(sha(output/'report.json')==expected and all(sha(output/n)==h for n,h in r['artifact_sha256'].items()),'final artifact drift')
    return dict(schema=NAME+'.consume',status='saved_child_compiler_verified_not_admitted',report_sha256=expected,
        prepare_sha256=r['prepare_sha256'],fixed_denominator=48,child_denominator=24,independent_roots=2,
        refusal_census=plan(p)['counts'],rows=r['rows'],new_resources=total,historical_signal_resources=p['historical_signal_resources'],
        saved_verification=dict(upstream_array_SVD_calls=p['paid_saved_consumer_array_SVD_calls'],
            compiler_array_SVD_calls=len(ready),outside_native_ledger=True,neural_calls=0,physics_calls=0),
        physical_benefit=None,full_tree=False,new_descendant_J=False,**FLAGS)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('inputs','prepare','verify','plan','run','consume'))
    parser.add_argument('--input');parser.add_argument('--prepared');parser.add_argument('--sha256')
    parser.add_argument('--output');parser.add_argument('--authorization');parser.add_argument('--receipt')
    a=parser.parse_args(argv);require(a.receipt is None or a.command=='consume','receipt only for strict consumption')
    require(a.command=='inputs' or a.sha256 is not None,'actual external SHA required')
    if a.command=='inputs':result=write_new(a.output,inputs_template())
    elif a.command=='prepare':
        exclusive(a.output);result=write_new(a.output,prepare_value(a.input,a.sha256))
    elif a.command in ('verify','plan'):
        p=verify(a.prepared,a.sha256);result=plan(p)
        if a.output:result=write_new(a.output,result)
    elif a.command=='run':result=run(a.prepared,a.sha256,a.output,a.authorization)
    else:
        if a.receipt:exclusive(a.receipt)
        result=consume(a.output,a.sha256)
        if a.receipt:result=write_new(a.receipt,result)
    print(canonical(result).decode());return 0


if __name__=='__main__':raise SystemExit(main())
