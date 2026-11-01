"""Two useful horizon blocks with matched measured axes; injectable native backend.

This is an experimental model-only primitive, not a HOST launcher or receipt
authority. The second block consumes the first block's predicted context.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p3_shared_panel_20261029_v1 as shared
from experiments.spectral_reachability import agent_p3_parent_native_adapter_20261029_v1 as parent

core, signal, frozen, live = shared.core, shared.signal, shared.frozen, shared.live
NAME = 'p3_matched_order_codex_20261029'


def contract():
    return dict(schema=NAME+'.contract', policies=['fixed_order','signal_order'], horizons=[1,3],
        blocks=2, candidates=4, bank='endpoints', replay='whole+segmented per candidate plus independent winner',
        selection='signal_order uses original progress/continue horizon at first parent; remaining quota forced',
        second_context='predicted selected full path, not real observation',
        precision='inherited native backend; shape equality is not FLOPs equality',
        no_padding=True, no_retry=True, full_P3=False, equal_FLOPs=False, scientific_pass=False)


def expected():
    # Per block: signal H1+H3 = 8 predict, selected panel = 9*H.
    c=live.predict_axes(52)
    c.update(jacobian_API=4,VJP_directions=768,grad_API=768,raw_backward_callbacks=768,
        array_SVD_calls=6,solve_calls=2,signal_compile_calls=4,program_compile_calls=2,
        replay_calls=18,metric_calls=8,schedule_decisions=2)
    return c


def choose(policy, decision, remaining, stage):
    core.require(policy in ('fixed_order','signal_order') and stage in (0,1), 'fixed policy/stage')
    core.require(type(remaining) is list and (remaining==[1,3] if stage==0 else len(remaining)==1)
                 and all(h in (1,3) for h in remaining), 'quota state')
    if policy=='signal_order' and stage==0:
        core.require(decision['continue_expanding'] is True and decision['program_type']=='progress' and
                     decision['extension_type']=='progress' and decision['horizon'] in remaining,
                     'unsupported original first decision')
        return decision['horizon']
    return remaining[0]


def run_unit(root, policy, factory, store, *, permit_ram=False):
    """Run two blocks using native signal/replay callbacks and actual ledger hooks.

    Caller must pre-seal sources, tasks, policy and hardware limits and persist
    events. A backend kind string is not a provenance or CUDA certification.
    Exceptions retain paid attempts in the final store event; no result is released.
    """
    root=core.validate_task(root); core.require(policy in ('fixed_order','signal_order'), 'policy')
    ledger=core.Ledger(expected(),sink=store.event); remaining=[1,3]; context=deepcopy(root['root_context'])
    result=dict(schema=NAME+'.unit',contract=contract(),root_sha256=core.digest(root),policy=policy,
                backend_kind=None,status='attempted',blocks=[],error=None,resource_receipt=None)
    store.event(dict(kind=NAME+'.start',root_sha256=result['root_sha256'],policy=policy,contract=result['contract']))
    try:
        for stage in range(2):
            task=parent.parent_task(root,context); backend=factory(task,store); kind=getattr(backend,'kind',None)
            core.require(kind=='HOST_CUDA' or (permit_ram is True and kind=='RAM_FIXTURE'), 'HOST CUDA or explicit fixture')
            core.require(result['backend_kind'] in (None,kind),'backend kind switch');result['backend_kind']=kind
            block=dict(stage=stage,parent_context_sha256=core.digest(context),responses=[],candidates=[],status='attempted')
            result['blocks'].append(block)
            for h in (1,3):
                ledger.scope=dict(stage=stage,phase='signal',horizon=h)
                req=core.request(task,dict(context=context),operation='compile_finite_horizon_J',horizon=h)
                response=ledger.call('signal_compile',lambda req=req:backend.signal(req,ledger))
                core.require(response['binding_sha256']==req['binding_sha256'] and response['parent_context_sha256']==req['parent_context_sha256'] and response['horizon']==h,'signal binding')
                block['responses'].append(response)
            ledger.scope=dict(stage=stage,phase='diagnostic')
            diagnostic=signal.spectra(task,block['responses'],svd=lambda a:ledger.call('array_svd',lambda:np.linalg.svd(a,full_matrices=False)))
            node,signals,_=signal.runtime_signal(task,block['responses'],diagnostic)
            ledger.scope=dict(stage=stage,phase='schedule')
            decision=ledger.call('schedule',lambda:core.choose(task,node,signals,'adaptive'))
            h=choose(policy,decision,remaining,stage); remaining.remove(h)
            block.update(decision=decision,selected_horizon=h,remaining=deepcopy(remaining),diagnostic=diagnostic)
            ledger.scope=dict(stage=stage,phase='program',horizon=h)
            bank=ledger.call('program_compile',lambda:frozen.generate(task,block['responses'][0 if h==1 else 1],'endpoints',ledger))
            block['bank']=bank
            for index,actions in enumerate(bank['actions']):
                ledger.scope=dict(stage=stage,phase='candidate',horizon=h,index=index)
                whole=ledger.call('replay',lambda:backend.replay(context,actions,False,ledger))
                segmented=ledger.call('replay',lambda:backend.replay(context,actions,True,ledger))
                core.require(whole==segmented,'candidate replay mismatch')
                terminal=core.checked_path(task,context,actions,whole)
                cost=ledger.call('metric',lambda:core.model_cost(task,terminal))
                block['candidates'].append(dict(index=index,actions=actions,whole=whole,segmented=segmented,cost=cost))
            core.require(len(block['candidates'])==4,'complete candidate queue')
            winner=min(block['candidates'],key=lambda x:(x['cost'],x['index']))
            ledger.scope=dict(stage=stage,phase='winner',horizon=h,index=winner['index'])
            path=ledger.call('replay',lambda:backend.replay(context,winner['actions'],False,ledger))
            core.require(path==winner['whole'],'winner replay mismatch')
            context=core.checked_path(task,context,winner['actions'],path)
            block.update(status='returned',winner_index=winner['index'],selected_path=path,output_context_sha256=core.digest(context))
        core.require(not remaining and ledger.attempted==ledger.completed==expected(),'exact measured complete vector')
        result['status']='complete_measured_axes_only'
    except Exception as exc:
        result.update(status='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        result['resource_receipt']=ledger.receipt()
        store.event(dict(kind=NAME+'.final',result=deepcopy(result)))
    return result


def verify_unit(root, r):
    """Saved arithmetic and parent-chain verification, not native IO provenance."""
    root=core.validate_task(root)
    core.require(r['contract']==contract() and r['root_sha256']==core.digest(root) and
                 r['status']=='complete_measured_axes_only' and len(r['blocks'])==2, 'unit binding/status')
    core.audit_ledger(r['resource_receipt'])
    core.require(r['resource_receipt']['attempted']==r['resource_receipt']['completed']==expected(), 'unit budget')
    remaining=[1,3]; context=deepcopy(root['root_context'])
    for i,block in enumerate(r['blocks']):
        task=parent.parent_task(root,context)
        core.require(block['stage']==i and block['status']=='returned' and block['parent_context_sha256']==core.digest(context), 'parent identity')
        responses=block['responses']; core.require(len(responses)==2 and [v['horizon'] for v in responses]==[1,3], 'signals')
        for response in responses:
            core.require(response['binding_sha256']==core.digest(task['binding']) and response['parent_context_sha256']==core.digest(context),'signal binding')
        diagnostic=signal.spectra(task,responses)
        core.require(block['diagnostic']==diagnostic, 'signal diagnostic')
        node,signals,_=signal.runtime_signal(task,responses,diagnostic)
        decision=core.choose(task,node,signals,'adaptive')
        core.require(decision==block['decision'], 'decision')
        h=choose(r['policy'],decision,remaining,i);remaining.remove(h)
        core.require(h==block['selected_horizon'] and remaining==block['remaining'], 'quota')
        dummy=core.Ledger(expected());dummy.scope=dict(phase='saved_recompute')
        bank=frozen.generate(task,responses[0 if h==1 else 1],'endpoints',dummy)
        core.require(block['bank']==bank and len(block['candidates'])==4, 'bank')
        for j,c in enumerate(block['candidates']):
            core.require(c['index']==j and c['actions']==bank['actions'][j] and c['whole']==c['segmented'], 'candidate')
            terminal=core.checked_path(task,context,c['actions'],c['whole'])
            core.require(c['cost']==core.model_cost(task,terminal), 'candidate cost')
        winner=min(block['candidates'],key=lambda x:(x['cost'],x['index']))
        core.require(block['winner_index']==winner['index'] and block['selected_path']==winner['whole'], 'winner')
        context=core.checked_path(task,context,winner['actions'],block['selected_path'])
        core.require(block['output_context_sha256']==core.digest(context), 'output context')
    return True


def compare(a,b):
    core.require(a['root_sha256']==b['root_sha256'] and a['contract']==b['contract']==contract() and
                 (a['policy'],b['policy'])==('fixed_order','signal_order'),'pair identity')
    if a['status']!='complete_measured_axes_only' or b['status']!='complete_measured_axes_only':
        return dict(matched_measured_axes=None,reason='incomplete or failed unit',equal_FLOPs=False)
    for r in (a,b):
        core.audit_ledger(r['resource_receipt'])
        core.require(sorted(x['selected_horizon'] for x in r['blocks'])==[1,3] and
            r['resource_receipt']['attempted']==r['resource_receipt']['completed']==expected(),'complete pair accounting')
    def shapes(r):
        return Counter(core.canonical([e['resource'],e['shape'],e['directions'],e['status']]) for e in r['resource_receipt']['events'])
    return dict(matched_measured_axes=True,matched_hook_shape_multiset=shapes(a)==shapes(b),axes=list(core.AXES),equal_FLOPs=False,
                scope='hook counts only; no provenance, timing, operator FLOPs or hidden-call certification')


def read_pinned(path, sha):
    core.hash_value(sha); p=Path(path)
    core.require(p.is_file() and not p.is_symlink(), 'regular input')
    data=p.read_bytes();core.require(hashlib.sha256(data).hexdigest()==sha, 'input SHA')
    def unique(items):
        d={}
        for k,v in items:
            core.require(k not in d,'duplicate key');d[k]=v
        return d
    return json.loads(data,object_pairs_hook=unique,parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))


def main():
    p=argparse.ArgumentParser(description='CPU-only saved arithmetic consumer; never dispatches model or physics')
    for field in ('root','fixed','signal'):
        p.add_argument('--'+field,required=True);p.add_argument('--'+field+'-sha',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();root=read_pinned(a.root,a.root_sha);fixed=read_pinned(a.fixed,a.fixed_sha);dynamic=read_pinned(a.signal,a.signal_sha)
    verify_unit(root,fixed);verify_unit(root,dynamic);result=compare(fixed,dynamic)
    out=dict(schema=NAME+'.consume',input_sha256=dict(root=a.root_sha,fixed=a.fixed_sha,signal=a.signal_sha),
        accounting=result,backend_kinds=[fixed['backend_kind'],dynamic['backend_kind']],
        native_IO_certified=False,source_provenance_certified=False,scientific_pass=False,
        limitation='saved arithmetic and hook metadata only; no native transcript or tensor certification')
    data=core.canonical(out)+b'\n';path=Path(a.output)
    with path.open('xb') as f: core.require(f.write(data)==len(data),'short write');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    print(hashlib.sha256(data).hexdigest())


if __name__=='__main__': main()
