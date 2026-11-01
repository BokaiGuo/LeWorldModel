"""Pure injected-callback parent-native core. No model loader, runner or physics.

Root authority and full parent context remain separate. Only progress edges
are generated; R is explicitly refused, not relabelled or replaced by H1.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import agent_p3_fullcommit_live_20261028_v1 as progress

NAME='agent_p3_parent_native_adapter_20261029_v1'
require,digest,sha=core.require,core.digest,core.sha
FLAGS=dict(P1_admission=False,P3_admission=False,scientific_admission=False,freshness=False,
    physical_release=False,full_policy_executed=False,full_C3_demonstrated=False,
    native_execution_certified=False,adaptive_width8_executed=False,reconfiguration_executed=False,
    all_program_types_covered=False,equal_FLOPs=False)


class UnsupportedType(ValueError):
    """The original decision is retained; no surrogate extension is dispatched."""


def root_node(task):
    return dict(context=deepcopy(task['root_context']),depth=0,indices=[],actions=[],path=[],edges=[])


def parent_task(root, context):
    """Explicit local view, NOT a new root certificate or an old producer receipt."""
    t=core.validate_task(root);c=core.context(context)
    require(c['latent_dim']==t['root_context']['latent_dim'],'same root latent ABI')
    t['root_context']=deepcopy(c);t['binding']['context_sha256']=digest(c)
    return core.validate_task(t)


def validate_parent(root,node):
    core.keys(node,('context','depth','indices','actions','path','edges'))
    core.integer(node['depth']);core.context(node['context'])
    require(node['depth']==len(node['indices'])==len(node['edges']) and node['depth']<=core.policy(root).max_depth,
        'complete depth/index/edge identity')
    for i in node['indices']:core.integer(i)
    c=deepcopy(root['root_context']);actions=[];path=[]
    for edge in node['edges']:
        core.keys(edge,('kind','input_context_sha256','output_context_sha256','actions_sha256','producer_sha256','actions','path'))
        require(edge['kind']=='progress' and edge['input_context_sha256']==digest(c) and
            edge['actions_sha256']==digest(edge['actions']),'exact progress prefix identity')
        core.hash_value(edge['producer_sha256'])
        c=core.checked_path(root,c,edge['actions'],edge['path'])
        require(edge['output_context_sha256']==digest(c),'exact full output context')
        actions+=deepcopy(edge['actions']);path+=deepcopy(edge['path'])
    require(node['actions']==actions and node['path']==path and node['context']==c,'whole raw/history prefix chain')
    return deepcopy(node)


def generation_minimum(h,count):
    c=progress.predict_axes(h*count)
    c.update(program_compile_calls=count,metric_calls=count,solve_calls=1,array_SVD_calls=1)
    return c


def atomic_queue(task,node,decision,signal_receipt,used,limit,ledger):
    """Reserve identities atomically; resource checks do NOT invent paid attempts."""
    core.integer(used);core.integer(limit,1)
    require(limit<=core.policy(task).max_expansions and used<=limit,'bounded original expansion limit')
    validate_parent(task,node)
    require(type(decision['candidates']) is int and decision['candidates'] in (4,8) and
        type(decision['horizon']) is int and decision['horizon'] in (1,3),'original width and horizon')
    reqs=[core.request(task,node,operation='compile_program_extension',**decision,candidate_index=i,
        prefix_edges=deepcopy(node['edges']),prefix_actions=deepcopy(node['actions']),
        signal_sha256=digest(signal_receipt),local_response=deepcopy(signal_receipt['responses']))
        for i in range(decision['candidates'])] if decision['continue_expanding'] else []
    out=dict(schema=NAME+'.queue',parent_context_sha256=digest(node['context']),depth=node['depth'],
        decision=deepcopy(decision),requests=reqs,status='stopped_by_original_child_policy',
        used_before=used,minimum=None,verification_reserve=None,remaining_before={k:ledger.caps[k]-ledger.attempted[k] for k in core.AXES})
    if not reqs:return out
    if used+len(reqs)>limit:out['status']='expansion_cap_before_node_queue';return out
    if decision['extension_type']!='progress' or decision['program_type'] not in ('progress','composition'):
        out['status']='refused_unsupported_original_type';return out
    c=generation_minimum(decision['horizon'],len(reqs))
    reserve=progress.predict_axes(2*(len(node['actions'])//5+decision['horizon']));reserve['replay_calls']=2
    out.update(minimum=c,verification_reserve=reserve)
    if any(c[k]+reserve[k]>out['remaining_before'][k] for k in c):
        out['status']='resource_cap_before_node_queue';return out
    out['status']='queued';return out


class ParentNativeAdapter:
    """Explicit adapter; native callbacks receive ONLY their parent-bound task.

    factory(parent_task, sink) -> object with signal(req,ledger) and
    replay(parent_context, raw, segmented, ledger). No globals are replaced.
    kind is HOST_CUDA or explicitly permitted RAM_FIXTURE; kind is not a cert.
    """
    def __init__(self,task,factory,sink,*,permit_ram=False):
        self.root=core.validate_task(task);self.root_sha=digest(self.root)
        self.factory=factory;self.sink=sink;self.permit_ram=permit_ram
        self._backends={};self._signals={};self._banks={};self.maps=[];self.kind=None

    def _parent(self,context):
        require(digest(self.root)==self.root_sha,'immutable root authority')
        t=parent_task(self.root,context);key=digest(context)
        if key not in self._backends:
            backend=self.factory(deepcopy(t),self.sink);kind=getattr(backend,'kind',None)
            require(kind=='HOST_CUDA' or (self.permit_ram is True and kind=='RAM_FIXTURE'),
                'explicit injected HOST CUDA callback or RAM fixture only')
            require(self.kind is None or self.kind==kind,'no backend-kind switch across parents')
            self.kind=kind;self._backends[key]=(deepcopy(t),backend)
        saved,backend=self._backends[key];require(saved==t,'complete context hash collision or stale parent view')
        return t,backend

    def _request(self,req,keys):
        core.keys(req,keys)
        require(req['binding']==self.root['binding'] and req['binding_sha256']==digest(self.root['binding'])
            and req['parent_context_sha256']==digest(core.context(req['parent_context'])),'outer original root/parent identities')
        return self._parent(req['parent_context'])

    def signal(self,req,ledger):
        task,backend=self._request(req,('binding','binding_sha256','parent_context','parent_context_sha256','operation','horizon'))
        require(req['operation']=='compile_finite_horizon_J' and type(req['horizon']) is int and req['horizon'] in (1,3), 'original signal request')
        nr=core.request(task,dict(context=task['root_context']),operation=req['operation'],horizon=req['horizon'])
        response=deepcopy(backend.signal(deepcopy(nr),ledger))
        core.keys(response,('binding_sha256','parent_context_sha256','horizon','nominal','jacobian','producer_sha256'))
        require(response['binding_sha256']==nr['binding_sha256'] and response['parent_context_sha256']==nr['parent_context_sha256']
            and type(response['horizon']) is int and response['horizon']==req['horizon'],'actual native parent signal identity')
        core.hash_value(response['producer_sha256']);core.finite(response['nominal'],(192,),True)
        core.finite(response['jacobian'],(192,10*req['horizon']),True)
        outward=dict(response,binding_sha256=req['binding_sha256'],producer_sha256=sha(__file__))
        mapping=dict(schema=NAME+'.signal_map',root_task_sha256=self.root_sha,root_binding=deepcopy(self.root['binding']),
            parent_task=task,outer_request=deepcopy(req),native_request=nr,native_response=response,outward_response=outward)
        mapping['mapping_sha256']=digest(mapping)
        self.maps.append(deepcopy(mapping));self.sink.event(dict(kind=NAME+'.signal_map',mapping=deepcopy(mapping)))
        self._signals[digest(req['parent_context']),req['horizon']]=deepcopy(mapping)
        return deepcopy(outward)

    def generate(self,req,ledger):
        names=('binding','binding_sha256','parent_context','parent_context_sha256','operation','program_type','extension_type',
            'horizon','candidates','program_depth_limit','continue_expanding','decision_source','inherited_engineering_thresholds',
            'no_optimality_claim','candidate_index','prefix_edges','prefix_actions','signal_sha256','local_response')
        task,backend=self._request(req,names)
        require(req['operation']=='compile_program_extension','actual extension request')
        if req['extension_type']!='progress' or req['program_type'] not in ('progress','composition'):
            raise UnsupportedType('no reconfiguration generator; original type retained')
        n,h,i=req['candidates'],req['horizon'],req['candidate_index']
        require(type(n) is int and n in (4,8) and type(i) is int and 0<=i<n and type(h) is int and h in (1,3), 'exact indexed4/8 queue')
        core.hash_value(req['signal_sha256'])
        require([r['horizon'] for r in req['local_response']]==[1,3],'complete paid H1/H3 at same parent')
        for response in req['local_response']:
            key=(req['parent_context_sha256'],response['horizon'])
            require(key in self._signals and self._signals[key]['outward_response']==response,'no root/stale J at child')
        response=self._signals[req['parent_context_sha256'],h]['native_response']
        # Immutable original root chain, not merely two hashes or a terminal192.
        prefix=root_node(self.root)
        prefix['edges']=deepcopy(req['prefix_edges']);prefix['depth']=len(prefix['edges']);prefix['indices']=[0]*prefix['depth']
        prefix['actions']=deepcopy(req['prefix_actions']);prefix['path']=[c for e in prefix['edges'] for c in deepcopy(e['path'])]
        prefix['context']=deepcopy(req['parent_context']);validate_parent(self.root,prefix)
        cache=(req['parent_context_sha256'],h,n,digest(response))
        if cache not in self._banks:self._banks[cache]=progress.generate(task,response,n,ledger)
        bank=self._banks[cache];actions=deepcopy(bank['actions'][i])
        path=backend.replay(deepcopy(task['root_context']),deepcopy(actions),False,ledger)
        core.checked_path(task,task['root_context'],actions,path)
        v=dict(binding_sha256=req['binding_sha256'],parent_context_sha256=req['parent_context_sha256'],
            extension_type='progress',actions=actions,path=deepcopy(path),producer_sha256=sha(__file__))
        self.sink.event(dict(kind=NAME+'.generated',root_task_sha256=self.root_sha,parent_task_sha256=digest(task),
            request=deepcopy(req),bank=deepcopy(bank),native_response=deepcopy(response),result=deepcopy(v)))
        return v

    def replay(self,req,ledger):
        task,backend=self._request(req,('binding','binding_sha256','parent_context','parent_context_sha256',
            'operation','mode','actions','edges'))
        require(req['operation']=='verify_program' and req['mode'] in ('whole','segmented')
            and req['parent_context']==self.root['root_context'],'whole program verified from original root')
        node=root_node(self.root);node.update(edges=deepcopy(req['edges']),depth=len(req['edges']),
            indices=[0]*len(req['edges']),actions=deepcopy(req['actions']),
            path=[c for e in req['edges'] for c in deepcopy(e['path'])])
        if node['path']:node['context']=deepcopy(node['path'][-1])
        validate_parent(self.root,node)
        path=backend.replay(deepcopy(task['root_context']),deepcopy(req['actions']),req['mode']=='segmented',ledger)
        core.checked_path(task,task['root_context'],req['actions'],path)
        require(path==node['path'],'whole/segmented exact complete program path')
        return deepcopy(path)


def default_caps(mode):
    require(mode in ('fixed','adaptive'),'original method')
    c=progress.predict_axes(114 if mode=='fixed' else 562)
    c.update(jacobian_API=18 if mode=='fixed' else 34,VJP_directions=3456 if mode=='fixed' else 6528,
        grad_API=3456 if mode=='fixed' else 6528,raw_backward_callbacks=3456 if mode=='fixed' else 6528,
        signal_compile_calls=18 if mode=='fixed' else 34,array_SVD_calls=27 if mode=='fixed' else 51,
        solve_calls=9 if mode=='fixed' else 17,program_compile_calls=36 if mode=='fixed' else 136,
        metric_calls=36 if mode=='fixed' else 136,schedule_decisions=9 if mode=='fixed' else 17,replay_calls=2)
    return c


def run_core(task,factory,sink,*,mode,caps=None,max_requests=None,permit_ram=False):
    """Pure callback orchestration; no production loading, saved strict or physical release."""
    task=core.validate_task(task);require(len(task['goal'])==192,'native192 task')
    require(mode in ('fixed','adaptive'),'original method');policy=core.policy(task)
    limit=policy.max_expansions if max_requests is None else max_requests
    core.integer(limit,1);require(limit<=policy.max_expansions,'engineering cap never exceeds original144')
    ledger=core.Ledger(default_caps(mode) if caps is None else caps,sink=sink.event)
    adapter=ParentNativeAdapter(task,factory,sink,permit_ram=permit_ram)
    rows=[dict(slot=i,status='not_requested',request=None,payload=None,node=None,error=None) for i in range(limit)]
    r=dict(schema=NAME+'.runtime',root_task_sha256=digest(task),root_binding=deepcopy(task['binding']),mode=mode,
        status='running',jobs=rows,signals=[],queues=[],replays=[dict(mode=x,status='not_requested',path=None) for x in ('whole','segmented')],
        diagnostic_best=None,model_plan=None,errors=[],fixed_denominator=dict(expansion_slots=limit,max_original_expansions=144),**FLAGS)
    frontier=[root_node(task)];best=None;used=0;stopped=False
    def rank(n):return (n['cost'],n['depth'],n['indices'])
    try:
        for _ in range(policy.max_depth):
            children=[]
            for parent in frontier:
                validate_parent(task,parent)
                ledger.scope=dict(mode=mode,phase='signal',parent_context_sha256=digest(parent['context']))
                signals,receipt=core.local_signal(task,parent,adapter,ledger,policy);r['signals'].append(receipt)
                ledger.scope=dict(mode=mode,phase='schedule',parent_context_sha256=digest(parent['context']))
                decision=ledger.call('schedule',lambda:core.choose(task,parent,signals,mode))
                queue=atomic_queue(task,parent,decision,receipt,used,limit,ledger);r['queues'].append(queue)
                if queue['status']=='stopped_by_original_child_policy':continue
                if queue['status']!='queued':
                    r['status']=queue['status'];stopped=True
                    if queue['status']=='refused_unsupported_original_type':
                        for row,req in zip(rows[used:used+len(queue['requests'])],queue['requests'],strict=True):
                            row.update(status='refused_unsupported_original_type',request=deepcopy(req))
                        used+=len(queue['requests'])
                    break
                assigned=rows[used:used+len(queue['requests'])]
                for row,req in zip(assigned,queue['requests'],strict=True):row.update(status='queued',request=deepcopy(req))
                used+=len(assigned)
                for row,req in zip(assigned,queue['requests'],strict=True):
                    row['status']='attempted';ledger.scope=dict(mode=mode,phase='program',slot=row['slot'],parent_context_sha256=digest(parent['context']))
                    before=ledger.attempted['predict_API']
                    v=ledger.call('program_compile',lambda req=req:adapter.generate(deepcopy(req),ledger))
                    terminal=core.checked_path(task,parent['context'],v['actions'],v['path'])
                    require(len(v['actions'])==5*decision['horizon'] and ledger.attempted['predict_API']-before>=decision['horizon'],
                        'real full extension prediction callbacks')
                    edge=dict(kind=v['extension_type'],input_context_sha256=digest(parent['context']),output_context_sha256=digest(terminal),
                        actions_sha256=digest(v['actions']),producer_sha256=v['producer_sha256'],actions=deepcopy(v['actions']),path=deepcopy(v['path']))
                    node=dict(context=terminal,depth=parent['depth']+1,indices=parent['indices']+[req['candidate_index']],
                        actions=parent['actions']+deepcopy(v['actions']),path=parent['path']+deepcopy(v['path']),edges=parent['edges']+[edge])
                    validate_parent(task,node);node['cost']=ledger.call('metric',lambda:core.model_cost(task,terminal))
                    row.update(status='returned',payload=v,node=deepcopy(node));children.append(node)
                    if best is None or rank(node)<rank(best):best=deepcopy(node)
            if stopped or not children:break
            children.sort(key=rank);frontier=children[:policy.base_beam*(policy.expansion_factor if mode=='adaptive' else 1)]
            for node in frontier:node.pop('cost')  # cost is rank metadata, not parent identity
        if not stopped:
            require(best is not None,'nonempty witnessed progress program')
            for slot in r['replays']:
                slot['status']='attempted';ledger.scope=dict(mode=mode,phase='verification',replay_mode=slot['mode'])
                req=core.request(task,root_node(task),operation='verify_program',mode=slot['mode'],actions=best['actions'],edges=best['edges'])
                path=ledger.call('replay',lambda req=req:adapter.replay(req,ledger))
                require(path==best['path'],'entire best raw chain independently replayed');slot.update(status='returned',path=path)
            require(not ledger.failed and ledger.attempted==ledger.completed,'all paid attempts returned')
            r.update(status='callback_progress_complete_NOT_certified',model_plan=deepcopy(best))
    except Exception as exc:
        r.update(status='failed_or_budget_exhausted');r['errors'].append(dict(type=type(exc).__name__,message=str(exc)))
        for row in rows:
            if row['status']=='attempted':row.update(status='failed',error=str(exc))
            elif row['status']=='queued':row.update(status='blocked_after_failure',error=str(exc))
        for slot in r['replays']:
            if slot['status']=='attempted':slot['status']='failed'
    r.update(diagnostic_best=best,resource_receipt=ledger.receipt(),signal_mappings=deepcopy(adapter.maps),
        backend_kind=adapter.kind,realized_progress_depth=max([x['node']['depth'] for x in rows if x['node'] is not None],default=0),
        realized_widths=[len(q['requests']) for q in r['queues'] if q['status']=='queued'],
        scalar_backward_calls=None,AD_primal_predict_API=None,Jacobian_elements=None)
    core.audit_ledger(r['resource_receipt'])
    return deepcopy(r)
