"""New-schema real 12/4 generator, independent replay, full native array validation.

Literal compiler/validator implementation derived from the frozen child compiler;
only allocation reads execution_quota. Never rewrites the original decision.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_p3_child_program_cuda_20261028_v1 as frozen
paid,original,core,contract,base=frozen.paid,frozen.original,frozen.core,frozen.contract,frozen.base
require,canonical,digest,sha=frozen.require,frozen.canonical,frozen.digest,frozen.sha
READY='paired_quota_ready'
FINAL='paired_quota_compiled_not_admitted'
NativeProgram=frozen.NativeProgram

def rule():
    return dict(schema='agent_p3_paired_quota_runtime_20261028_v1.generator',
        search_predict_cap=24,selected_prefix_predict_reserve=1,
        damping=frozen.rule()['damping'],joint_raw_l2_cap=frozen.rule()['joint_raw_l2_cap'],
        fractions='i/(execution_count-1); all candidates independently evaluated; no dedup',
        selection='original Q terminal mean-square then stable index',
        original_decision_unchanged=True)

def quota(decision):
    h,n=decision['horizon'],decision['candidates']
    require(type(h) is int and h in (1,3) and type(n) is int and n in (4,8),
        'original H1/H3 4/8 decision')
    count=24//(2*h)
    require(2*h*count==24,'exact search quota without rounding')
    reason=None
    if count<n:reason='execution_count_below_original_request'
    elif decision['extension_type']!='progress' or decision['program_type']!='progress':
        reason='undefined_reconfiguration'
    elif decision['continue_expanding'] is not True:reason='original_expansion_refused'
    return dict(horizon=h,candidates=count,search_predict_API=24,
        selected_prefix_predict_API=1,signal_predict_API=8,total_predict_API=33,
        original_requested_candidates=n,allowed=reason is None,reason=reason)

def caps(branch):
    result=dict.fromkeys(core.AXES,0)
    if branch['status']!=READY:return result
    h,n=(branch['execution_quota'][k] for k in ('horizon','candidates'));calls=2*h*n+1
    result.update(predict_API=calls,predict_samples=calls,predict_context_tokens=3*calls,
        action_encoder_calls=calls,action_encoder_samples=calls,action_encoder_context_tokens=3*calls,
        predictor_calls=calls,predictor_samples=calls,predictor_context_tokens=3*calls,
        pred_proj_calls=calls,pred_proj_rows=3*calls,array_SVD_calls=1,solve_calls=1,
        program_compile_calls=1,replay_calls=2*n+1,metric_calls=n)
    return result



def generate(task,child,branch,ledger):
    task=core.validate_task(task);child=contract.validate_child(child,[task,task])
    # validate_child checks the supplied slot task without altering original root binding.
    require(branch['status']==READY and branch['child_id']==child['child_id'] and branch['child_sha256']==digest(child) and
        branch['parent_context_sha256']==child['context_sha256'] and branch['execution_quota']==quota(branch['decision']) and branch['execution_quota']['allowed'] is True,'eligible same-child progress only')
    r=branch['response'];h,n=(branch['execution_quota'][k] for k in ('horizon','candidates'))
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
        h,n=(branch['execution_quota'][k] for k in ('horizon','candidates'));parent=child['context']
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
    paths=[];h=branch['execution_quota']['horizon']
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
    ledger=core.Ledger(caps(branch));ledger.scope=scope(branch,None,'compile',branch['execution_quota']['horizon'])
    bank=ledger.call('program_compile',lambda:generate(task,child,branch,ledger))
    require(canonical(row['bank'])==canonical(bank) and [e for e in events if e['kind']=='bank']==[
        dict(kind='bank',branch_sha256=digest(branch),bank=bank)],'independent saved same-child ridge/actions')
    n=branch['execution_quota']['candidates'];require(len(row['candidates'])==n,'full original candidate denominator')
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


