"""Paid observed-child H1/H3 native adapter; only MAIN-authorized run loads CUDA.

No model execution during import/prepare/verify/consume. No physics, training,
downloads, root J substitution, candidate shrinking, or scientific admission.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import inspect
import os
from pathlib import Path
import time
import shutil
import zlib

import numpy as np

from experiments.spectral_reachability import agent_p3_child_scheduler_contract_20261028_v1 as contract

base, core = contract.signal, contract.core
ROOT = contract.ROOT
NAME = 'agent_p3_child_signal_cuda_20261028_v2'
FLAGS = dict(contract.FLAGS)
require, digest, canonical, sha = contract.require, contract.digest, contract.canonical, contract.sha
CONTRACT_SHAS = ('33fd52042dae14f46c2e0656f21c0b2749e665ec1820ca43199d711f4fa516ca',
    '496075344f51830497d6f5dd9e98273ed8ca8e0650348fec4e64b62598146ce8',
    '6979c9f0d2c1541b8568fd3ae4bab686d8656b561b9ab0424de65689d71d1bc2')
BASE_SHA = '6ce7722eca40cdd5f8958ba9590d8dc3d76d38bd10895d5ca494c1db3b6d100b'
CAP = 512*1024**2
JOURNAL = 'events.jsonl.gz'
DECODED_CAP = 1024*1024**2
LINE_CAP = 64*1024**2
V1_SHAS = ('7c6d741a8973fa42ac794ca85512167b81292c35ed679b6e291c5a37ea83cee7',
    'da50b5265ba2b56a09d7da1afe1cc3c491af1cd9297e9853f29af0eb4e023b2f',
    'fa2d58b3ba42e765f44ad16f78697daaacef1bf574c39c2a0b59abb7310a7a58')
REFERENCE = ROOT/'results/agent_p3_child_signal_cuda_20261028_v1_cuda_MAIN01'
REFERENCE_SHA = 'c0279c6a48d86bb46ed90b1d098bd436c76a9e85daa00893d34a119afa574641'
REFERENCE_JOURNAL_SHA = '41ee38987939097eff34f14bafc15dffecd488c7a689f0fb3be56a86f19a690e'


def capacity_receipt():
    """Saved engineering reference, not a successful signal/physics receipt."""
    require(sha(REFERENCE/'report.json')==REFERENCE_SHA, 'sealed failed v1 report')
    r=base.read(REFERENCE/'report.json',256*1024**2)
    require(r['errors']==[dict(message='output engineering cap',type='ValueError')] and
        sha(REFERENCE/'events.jsonl')==REFERENCE_JOURNAL_SHA and
        r['artifact_sha256']['events.jsonl']==REFERENCE_JOURNAL_SHA, 'sealed failed storage reference')
    row_max=max(len(canonical(row)) for row in r['rows'])
    require(row_max==4671473 and (REFERENCE/'events.jsonl').stat().st_size==267806913,
        'actual saved reference dimensions')
    # Measurements: isolate each group as one streaming gzip member, level6,
    # Z_SYNC_FLUSH per original line, Z_FINISH per group. Other events independently gzip.
    journal_estimate=2*(46*564813+23*1200584+1123)+16*1024**2
    report_estimate=2*23*row_max+16*1024**2
    require(journal_estimate<256*1024**2 and report_estimate<256*1024**2, 'full panel engineering forecast')
    return dict(reference_report=str(REFERENCE/'report.json'),reference_sha256=REFERENCE_SHA,
        reference_journal_sha256=REFERENCE_JOURNAL_SHA,reference_journal_bytes=267806913,
        measured_events=152326,measured_groups=38,measured_journal_child_returns=18,
        report_returned_children=19,group_max_raw_bytes=5105623,group_max_sync_gzip_bytes=564813,
        child_return_max_raw_bytes=4671684,child_return_max_gzip_bytes=1200584,
        report_max_row_bytes=row_max,misc_gzip_bytes=1123,projected_acquisitions=46,projected_returns=23,
        multiplier=2,allowance_per_partition_bytes=16*1024**2,
        estimated_journal_bytes=journal_estimate,estimated_report_bytes=report_estimate,
        empirical_not_worst_case_guarantee=True,actual_cap_still_failclosed=True,
        storage_only=True,computation_discount=False,scientific_evidence=False)

verify_contract = contract.verify


def own_files():
    return [Path(__file__).resolve(), ROOT/f'tests/{NAME}_test.py', ROOT/f'innerdocs/{NAME}.md']


def pins():
    expected = {str(p):h for p,h in zip(contract.own_files(),CONTRACT_SHAS,strict=True)}
    expected[str(Path(base.__file__).resolve())] = BASE_SHA
    v1_paths = [ROOT/'experiments/spectral_reachability/agent_p3_child_signal_cuda_20261028_v1.py',
        ROOT/'tests/agent_p3_child_signal_cuda_20261028_v1_test.py',
        ROOT/'innerdocs/agent_p3_child_signal_cuda_20261028_v1.md']
    expected.update({str(p):h for p,h in zip(v1_paths,V1_SHAS,strict=True)})
    contract.check_sources(expected)
    return expected


def roster(prepared):
    return [dict(child_id=c['child_id'],status='not_reached' if c['status']=='not_reached' else 'not_started',
        result=None,error=None) for c in prepared['children']]


def profile(prepared):
    observed=sum(c['status']=='observed_child' for c in prepared['children'])
    sharing=prepared['inputs']['sharing']; require(sharing in ('independent','shared_equal'),'sealed sharing')
    groups=observed*(2 if sharing=='independent' else 1); schedules=1 if sharing=='independent' else 2
    caps={axis:groups*count for axis,count in contract.signal_caps(schedules).items()}
    return dict(resource_caps=caps,acquisitions=groups,observed_children=observed,child_slots=24,method_slots=48,
        wall_seconds=900,output_bytes=CAP,final_reserve_bytes=256*1024**2,disk_floor_bytes=200*1024**2,
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        TF32=False,SDP='MATH',CPU_neural=False,physics_steps=0,
        journal_encoding='single gzip member level6; sync flush and fsync EACH unchanged canonical chained JSON event',
        decoded_journal_cap_bytes=DECODED_CAP,journal_line_cap_bytes=LINE_CAP,
        acquisition='native192 H1/H3: 8API/8samples/2J/384 scalar VJP plus 2 saved-array SVD; schedules paid separately',
        whole_job_wall_scope='before prepare verify through final source checks, journal close; final report write excluded',
        no_padding=True,no_retry=True,engineering_limits_only=True)


def material(prepared, expected):
    sealed=pins(); p=verify_contract(str(contract.absolute(prepared)),expected)
    require(p['schema']==contract.NAME+'.prepare' and len(p['children'])==24 and len(p['jobs'])==48 and
        p['rule']==contract.rule() and all(p[k] is False for k in FLAGS),'complete frozen known-child contract')
    edge=p['inputs']['p4']; reader=contract.rolling.Reader()
    original=reader.json(edge['prepared'],edge['prepare_sha256']); mb=original['modelbinding']
    paths={Path(path).parent for path in mb['model_files']}
    require(len(paths)==1,'single exact original policy directory')
    policy=paths.pop(); require(set(mb['model_files'])=={str(policy/'weights.pt'),str(policy/'config.json')},'only bound native weights/config')
    require(all(t['binding']['model_sha256']==mb['model_files'][str(policy/'weights.pt')] for t in p['tasks']),
        'task model SHA is original weights FILE hash, not state_dict hash')
    sources=deepcopy(p['sources']); contract.merge(sources,sealed)
    contract.merge(sources,mb['model_files']); contract.merge(sources,mb['source_closure'])
    contract.merge(sources,{str(f):sha(f) for f in own_files()})
    contract.merge(sources,{str(contract.absolute(prepared)):expected,
        str(REFERENCE/'report.json'):REFERENCE_SHA,str(REFERENCE/'events.jsonl'):REFERENCE_JOURNAL_SHA})
    contract.check_sources(sources)
    return dict(schema=NAME+'.prepare',contract_prepared=str(prepared),contract_sha256=expected,
        capacity_forecast=capacity_receipt(),child_contract=p,policy=str(policy),expected_model_hash=mb['tensor_sha256'],sources=sources,
        profile=profile(p),denominator=roster(p),methods=['fixed','adaptive'],
        status='known_root_child_signal_prepared_not_admitted',physical_dispatch=False,
        inherited_cost='P4 RGB encoding and P3 original signals remain historical, never new child payment',**FLAGS)


def verify(path,expected):
    core.hash_value(expected); path=contract.absolute(path)
    require(sha(path)==expected,'external actual prepare SHA')
    p=base.read(path,64*1024**2); contract.check_sources(p['sources'])
    require(canonical(p)==canonical(material(p['contract_prepared'],p['contract_sha256'])) and sha(path)==expected,
        'full child source/input/profile rebuild')
    return p


class NativeChildSignal(base.NativeSignal):
    """Reuses native full-buffer observers, not old root-only request validator."""
    kind='HOST_CUDA'
    def __init__(self,model,device,task,child,store,tasks):
        self.child=contract.validate_child(child,tasks)
        require(self.child['status']=='observed_child','cannot encode an unreached/future child')
        super().__init__(model,device,task,store)
        require(digest(self.task['binding'])==self.child['ancestry']['task_binding_sha256'],'exact root task binding')
        self.source_sha256=sha(__file__)

    def checked_request(self,req):
        core.keys(req,('binding','binding_sha256','parent_context','parent_context_sha256','operation','horizon'))
        h=req['horizon']; require(type(h) is int and h in (1,3),'fixed child H1/H3')
        require(req==core.request(self.task,dict(context=self.child['context']),operation='compile_finite_horizon_J',horizon=h),
            'exact observed-child task/context; no stale root or outcome inputs')
        return h

    def signal(self,req,ledger):
        h=self.checked_request(req)
        # Validation and CUDA construction guard precede any torch import.
        require(getattr(self.device,'type',None)=='cuda','HOST CUDA only; no CPU fallback')
        import torch
        self.frozen(); self.ledger=ledger; self.step=None
        original_scope=deepcopy(ledger.scope); ledger.scope=dict(original_scope,horizon=h)
        try:
            emb=torch.tensor(req['parent_context']['latent_history'],dtype=torch.float32,device=self.device)[None]
            hist=torch.tensor(req['parent_context']['action_history'],dtype=torch.float32,device=self.device)[None]
            mean=torch.tensor(self.task['action_mean'],dtype=torch.float32,device=self.device)
            std=torch.tensor(self.task['action_std'],dtype=torch.float32,device=self.device)
            def terminal(raw_flat,phase):
                self.phase=phase; state,history=emb,hist
                self.arrays('origin',latent=state,history=history,raw_flat=raw_flat,mean=mean,std=std)
                for i,raw in enumerate(raw_flat.reshape(h,5,2)):
                    self.step=i; token=((raw-mean)/std).reshape(10)
                    outgoing=torch.cat((history,token[None,None]),dim=1)
                    self.arrays('packet',raw=raw,token=token,outgoing=outgoing)
                    encoded=self.model.action_encoder(outgoing)
                    predicted=self.call('predict',lambda:self.model.predict(state,encoded),dict(latent=state,act_emb=encoded),shape=list(state.shape))
                    require(tuple(predicted.shape)==(1,3,192),'native full prediction chain')
                    state=torch.cat((state[:,1:],predicted[:,-1:]),dim=1);history=outgoing[:,1:]
                    self.arrays('boundary',latent=state,history=history)
                return state[0,-1]
            with self.observe():
                zero=emb.new_zeros(10*h)  # explicit nominal raw, NEVER zero history.
                with torch.no_grad():nominal=terminal(zero,'nominal')
                self.arrays('nominal',value=nominal)
                def jacobian():
                    with torch.enable_grad():
                        raw=zero.detach().clone().requires_grad_(True)
                        handle=raw.register_hook(lambda value:ledger.call('raw_backward',lambda:self._raw_return(value)))
                        try:
                            ad=terminal(raw,'AD');self.step=None;self.arrays('AD_terminal',value=ad)
                            require(torch.equal(ad,nominal),'independent nominal/AD exact at same raw input')
                            rows=[]
                            for direction in range(192):
                                basis=torch.zeros_like(ad);basis[direction]=1
                                self.store.event(dict(kind='direction',scope=deepcopy(ledger.scope),direction=direction))
                                def grad():
                                    return self.call('grad',lambda:torch.autograd.grad(ad,raw,grad_outputs=basis,
                                        retain_graph=direction<191,create_graph=False,allow_unused=False)[0],dict(output=ad,raw=raw,basis=basis))
                                rows.append(ledger.call('vjp',grad,directions=1))
                            return torch.stack(rows)
                        finally:handle.remove()
                j=ledger.call('jacobian',jacobian);self.arrays('jacobian',value=j)
            self.frozen()
            return dict(binding_sha256=req['binding_sha256'],parent_context_sha256=req['parent_context_sha256'],
                horizon=h,nominal=nominal.detach().cpu().tolist(),jacobian=j.detach().cpu().tolist(),producer_sha256=self.source_sha256)
        finally:ledger.scope=original_scope


class Tagged:
    def __init__(self,store,group):self.store,self.group=store,group
    def event(self,value):self.store.event(dict(kind='child_trace',group=self.group,payload=deepcopy(value)))


def exclusive(output):
    path=contract.absolute(output)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists(),'NEW exclusive child output before loading')
    return path


def write_new(output,value):
    path=exclusive(output); content=canonical(value);require(len(content)<=64*1024**2,'bounded metadata')
    with path.open('xb') as f:
        require(f.write(content)==len(content),'complete metadata write');f.flush();os.fsync(f.fileno())
    base.fsync_dir(path.parent);return dict(output=str(path),sha256=sha(path),bytes=len(content))


def total_resources(rows):
    attempted=dict.fromkeys(core.AXES,0);completed=deepcopy(attempted)
    for row in rows:
        if row['result'] is None:continue
        for a in row['result']['acquisitions']:
            r=a['resource_receipt']
            if r is None:continue
            core.audit_ledger(r)
            for axis in core.AXES:attempted[axis]+=r['attempted'][axis];completed[axis]+=r['completed'][axis]
    return dict(attempted=attempted,completed=completed,backward_engine_calls=None,cuda_ns=None,FLOPs=None,
        inclusive_wall_not_added=True,physics_steps=0)


def run(prepared,expected,output,authorization):
    require(authorization=='MAIN_APPROVED_HOST_CHILD_CUDA','MAIN approval required BEFORE verify/model import')
    output=exclusive(output);started=time.monotonic();p=verify(prepared,expected);c=p['child_contract']
    store=Store(output,p['profile']);store.started=started;rows=roster(c);errors=[]
    model=None;before=after=host=None;sources_ok=cleanup=False
    try:
        store.put('binding.json',dict(prepared=str(prepared),prepare_sha256=expected))
        store.event(dict(kind='denominator',rows=deepcopy(rows),methods=48,source_manifest_sha256=digest(p['sources'])))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        store.memory=lambda:(torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device))
        store.check();model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        before=base.model_hash(model);require(before['original_state_dict']==p['expected_model_hash'],'original state_dict identity, not weights file SHA')
        store.event(dict(kind='model_before',hashes=before))
        for module in model.modules():
            source=str(Path(inspect.getfile(type(module))).resolve())
            require(source in p['sources'] and sha(source)==p['sources'][source],'loaded native class within exact sealed sources')
        with torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            for row,child in zip(rows,c['children'],strict=True):
                if child['status']=='not_reached':continue
                store.check();row['status']='attempted'
                def factory(actual,group):
                    require(actual==child,'exact child dispatch')
                    return NativeChildSignal(model,device,c['tasks'][child['root_slot']],child,
                        Tagged(store,child['child_id']+':'+group),c['tasks'])
                result=contract.plan_child(child,c['tasks'],factory,sharing=c['inputs']['sharing'],authorization='MAIN_APPROVED_CHILD_SIGNAL')
                row['result']=result
                require(all(a['status']=='returned' for a in result['acquisitions']),'paid child signal/schedule failure; no retry or release')
                row['status']='returned';store.event(dict(kind='child_return',row=deepcopy(row)))
        torch.cuda.synchronize(device);store.check();cleanup=True
        counts=total_resources(rows)
        require(counts['attempted']==counts['completed']==p['profile']['resource_caps'],'full paid native budget, no missing rows or hidden discounts')
    except BaseException as exc:errors.append(dict(type=type(exc).__name__,message=str(exc)))
    finally:
        for row in rows:
            if row['status']=='attempted':row['status']='failed'
            elif row['status']=='not_started':row['status']='blocked_not_dispatched'
        if model is not None:
            try:
                after=base.model_hash(model);require(before==after and all(not m.training for m in model.modules()) and
                    all(not v.requires_grad for v in model.parameters()),'frozen model/buffers before/after')
                store.event(dict(kind='model_after',hashes=after))
            except BaseException as exc:errors.append(dict(stage='model_final',message=str(exc)))
        try:
            contract.check_sources(p['sources']);pins();sources_ok=True;store.event(dict(kind='sources_after',sha256=digest(p['sources'])))
            store.check()
        except BaseException as exc:errors.append(dict(stage='source_or_limits',message=str(exc)))
        try:store.close()
        except BaseException as exc:errors.append(dict(stage='close',message=str(exc)))
        complete=not errors and cleanup and sources_ok and before is not None and before==after and all(r['status'] in ('returned','not_reached') for r in rows)
        report=dict(schema=NAME+'.report',status='paid_child_signals_saved_not_admitted' if complete else 'partial_or_failed',
            prepared=str(prepared),prepare_sha256=expected,rows=rows,fixed_denominator=48,child_denominator=24,
            sharing=c['inputs']['sharing'],resources=total_resources(rows),errors=errors,model_before=before,model_after=after,
            sources_verified=sources_ok,cleanup_ok=cleanup,source_manifest_sha256=digest(p['sources']),host=host,
            wall_seconds=store.clock()-store.started,CUDA_peak_bytes=store.memory() if store.memory else None,
            journal_tail=store.tail,artifact_sha256={n:sha(output/n) for n in (JOURNAL,'binding.json') if (output/n).is_file()},**FLAGS)
        store.put('report.json',report)
    require(complete,'partial/failed child signal evidence retained, not released: '+repr(errors))
    return dict(output=str(output),report_sha256=sha(output/'report.json'),resources=report['resources'])


def recompute_signal(task,child,responses):
    node=dict(context=child['context'],depth=0);p=core.policy(task,node);r=np.asarray(task['projection'],float)@(
        np.asarray(task['goal'])-np.asarray(responses[0]['nominal']))
    energy=float(r@r);require(np.isfinite(energy),'finite child residual');details=[];chi=[]
    for h,response in zip((1,3),responses,strict=True):
        core.keys(response,('binding_sha256','parent_context_sha256','horizon','nominal','jacobian','producer_sha256'))
        require(type(response['horizon']) is int and response['horizon']==h and response['binding_sha256']==digest(task['binding']) and
            response['parent_context_sha256']==child['context_sha256'] and response['producer_sha256']==sha(__file__),'exact child response identity')
        core.finite(response['nominal'],(192,),True);j=core.finite(response['jacobian'],(192,10*h),True)
        u,s,_=np.linalg.svd(np.asarray(task['projection'],float)@j,full_matrices=False)
        rank=int(np.count_nonzero(s>s[0]*p.svd_relative_cutoff)) if len(s) and s[0]>0 else 0
        parallel=u[:,:rank]@(u[:,:rank].T@r);perp=r-parallel
        value=0. if energy==0 else float(np.clip((perp@perp)/energy,0,1));chi.append(value)
        details.append(dict(horizon=h,rank=rank,singular_values=s.tolist(),unreachable_fraction_same_reference_residual=value))
    signed=chi[0]-chi[1];signals=core.FrozenSignals(p.source_seal,chi[0],max(0.,signed))
    return signals,dict(schema=core.NAME+'.signal',responses=deepcopy(responses),spectra=details,
        reference='short-horizon nominal residual for BOTH spaces; not best-cost stagnation or physical labels',
        residual=r.tolist(),residual_energy=energy,signed_horizon_coverage_gain=signed,
        scheduling_signal=asdict(signals),no_control_or_safety_guarantee=True)


def validate_trace(task,child,acquisition,events,schedules):
    """Independent saved-array/paid-hook checks, not a neural replay or AD proof."""
    r=acquisition['resource_receipt'];core.audit_ledger(r)
    require(acquisition['status']=='returned' and acquisition['error'] is None and acquisition['backend_kind']=='HOST_CUDA' and
        r['attempted']==r['completed']==r['caps']==contract.signal_caps(schedules),'exact paid acquisition vector')
    allowed={'resource_attempt','resource_return','io_input','io_return','origin','packet','boundary','nominal','AD_terminal','jacobian','direction','raw_backward'}
    require(all(e['kind'] in allowed for e in events),'strict native child event whitelist')
    resources=[e for e in events if e['kind'].startswith('resource_')];stack=[];expected=[]
    for op in r['events']:
        require(op['status']=='returned' and op['scope']['child_id']==child['child_id'] and
            op['scope']['parent_context_sha256']==child['context_sha256'] and
            op['scope']['group']==acquisition['group'].rsplit(':',1)[-1],'all actual operations charged to SAME child/group')
        if op['resource']=='vjp':require(op['directions']==1,'original scalar VJP calls, not vectorized/grouped derivatives')
        expected.extend([dict(kind='resource_attempt',entry=dict(op,status='attempted',ended_ns=None)),dict(kind='resource_return',entry=op)])
    require(sorted(map(canonical,resources))==sorted(map(canonical,expected)),'full journal vs actual ledger')
    for e in events:
        if e['kind'] in ('resource_attempt','resource_return'):
            core.keys(e,('kind','entry'))
            if e['kind']=='resource_attempt':stack.append(e['entry']['index'])
            else:require(stack and stack.pop()==e['entry']['index'],'nested real attempt/return order')
        else:
            require(bool(stack),'native buffers/directions occur inside paid dispatch')
            if e.get('operation_index') is not None:
                require(stack[-1]==e['operation_index'],'native IO before actual return of same operation')
    require(not stack,'complete native operation stack')
    linked={}
    for e in events:
        if 'arrays' not in e:
            if e['kind']=='direction':
                core.keys(e,('kind','scope','direction'))
                require(type(e['direction']) is int and 0<=e['direction']<192 and
                    type(e['scope'].get('horizon')) is int and e['scope']['horizon'] in (1,3) and
                    e['scope']['child_id']==child['child_id'] and e['scope']['parent_context_sha256']==child['context_sha256'] and
                    e['scope']['group']==acquisition['group'].rsplit(':',1)[-1],'every direction has exact own child/H/group')
            continue
        core.keys(e,('kind','scope','phase','step','operation_index','arrays'))
        require(e['phase'] in ('nominal','AD') and type(e['scope'].get('horizon')) is int and e['scope']['horizon'] in (1,3) and
            e['scope']['child_id']==child['child_id'] and e['scope']['parent_context_sha256']==child['context_sha256'],'own H-scoped full arrays')
        require(e['scope']['group']==acquisition['group'].rsplit(':',1)[-1],'own acquisition arrays only')
        require(e['arrays'] and all(base.unblob(v).dtype==np.float32 for v in e['arrays'].values()),'original full float32 arrays')
        idx=e['operation_index']
        if e['kind'] in ('io_input','io_return','raw_backward'):
            require(type(idx) is int and 0<=idx<len(r['events']),'exact linked actual IO')
            op=r['events'][idx];require(op['scope']==e['scope'],'IO operation scope')
            linked.setdefault(idx,[]).append(e['kind'])
            if e['kind']=='io_input':
                names={'predict':('latent','act_emb'),'action_encoder':('0',),'predictor':('0','1'),'pred_proj':('0',),'grad':('output','raw','basis')}[op['resource']]
                core.keys(e['arrays'],tuple(op['resource']+':'+n for n in names))
                if op['resource']!='grad':require(e['arrays'][op['resource']+':'+names[0]]['shape']==op['shape'],'charged sample/shape identity')
            elif e['kind']=='io_return':core.keys(e['arrays'],(op['resource']+':output',))
            else:require(op['resource']=='raw_backward','raw gradient hook resource');core.keys(e['arrays'],('value',))
        else:
            require(idx is None,'non IO must not borrow an operation')
            core.keys(e['arrays'],{'origin':('latent','history','raw_flat','mean','std'),'packet':('raw','token','outgoing'),
                'boundary':('latent','history'),'nominal':('value',),'AD_terminal':('value',),'jacobian':('value',)}[e['kind']])
    for op in r['events']:
        if op['resource'] in ('predict','action_encoder','predictor','pred_proj','grad'):
            require(linked.get(op['index'])==['io_input','io_return'],'every API/module/grad has complete full IO')
        elif op['resource']=='raw_backward':require(linked.get(op['index'])==['raw_backward'],'every raw callback saved')
    responses=acquisition['signal']['responses']
    for h,response in zip((1,3),responses,strict=True):
        local=[e for e in events if e.get('scope',{}).get('horizon')==h]
        def arrays(kind,phase=None):
            return [({k:base.unblob(v) for k,v in e['arrays'].items()},e) for e in local if e['kind']==kind and (phase is None or e['phase']==phase)]
        require(len(arrays('io_input'))==len(arrays('io_return'))==8*h+192,'all and only charged native IO')
        ns,js,ads=arrays('nominal'),arrays('jacobian'),arrays('AD_terminal')
        require(len(ns)==len(js)==len(ads)==1,'one nominal/J/AD for each H')
        n,j=ns[0][0]['value'],js[0][0]['value']
        require(n.shape==(192,) and j.shape==(192,10*h) and np.array_equal(n,response['nominal']) and
            np.array_equal(j,response['jacobian']) and np.array_equal(n,ads[0][0]['value']),'exact native nominal/J/AD/response')
        for phase in ('nominal','AD'):
            origins,packets,boundaries=arrays('origin',phase),arrays('packet',phase),arrays('boundary',phase)
            require(len(origins)==1 and len(packets)==len(boundaries)==h,'complete H-packet trajectory')
            o=origins[0][0];latent,hist=o['latent'],o['history']
            require(np.array_equal(latent,np.asarray(child['context']['latent_history'],np.float32)[None]) and
                np.array_equal(hist,np.asarray(child['context']['action_history'],np.float32)[None]) and
                np.array_equal(o['raw_flat'],np.zeros(10*h,np.float32)) and np.array_equal(o['mean'],task['action_mean']) and
                np.array_equal(o['std'],task['action_std']),'actual child full latent/nonzero history and original normalizer; not stale root')
            inputs,outputs=arrays('io_input',phase),arrays('io_return',phase)
            for i,((packet,pe),(after,be)) in enumerate(zip(packets,boundaries,strict=True)):
                require(type(pe['step']) is type(be['step']) is int and pe['step']==be['step']==i and
                    np.array_equal(packet['raw'],o['raw_flat'].reshape(h,5,2)[i]),'literal raw nominal packet')
                token=((packet['raw']-o['mean'])/o['std']).reshape(10);outgoing=np.concatenate((hist,token[None,None]),axis=1)
                require(np.array_equal(packet['token'],token) and np.array_equal(packet['outgoing'],outgoing),'original normalization no /100 or clipping')
                def io(resource,returned=False):
                    found=[a for a,e in (outputs if returned else inputs) if e['step']==i and any(k.startswith(resource+':') for k in a)]
                    require(len(found)==1,'one full native module IO per packet');return found[0]
                ai,ao=io('action_encoder'),io('action_encoder',True);pi,po=io('predict'),io('predict',True)
                ni,no=io('predictor'),io('predictor',True);qi,qo=io('pred_proj'),io('pred_proj',True)
                require(np.array_equal(ai['action_encoder:0'],outgoing) and np.array_equal(pi['predict:latent'],latent) and
                    np.array_equal(pi['predict:act_emb'],ao['action_encoder:output']) and np.array_equal(ni['predictor:0'],latent) and
                    np.array_equal(ni['predictor:1'],ao['action_encoder:output']) and
                    np.array_equal(qi['pred_proj:0'],no['predictor:output'].reshape(3,192)) and
                    np.array_equal(qo['pred_proj:output'].reshape(1,3,192),po['predict:output']),'full AE/predict/predictor/pred_proj wiring')
                latent=np.concatenate((latent[:,1:],po['predict:output'][:,-1:]),axis=1);hist=outgoing[:,1:]
                require(np.array_equal(after['latent'],latent) and np.array_equal(after['history'],hist),'exact parent-child context history closure')
            require(np.array_equal(latent[0,-1],n),'saved path terminal')
        require([e['direction'] for e in local if e['kind']=='direction']==list(range(192)),'all 192 literal basis directions')
        gi=[a for a,e in arrays('io_input','AD') if 'grad:basis' in a]
        go=[a['grad:output'] for a,e in arrays('io_return','AD') if 'grad:output' in a]
        raw=[a['value'] for a,e in arrays('raw_backward','AD')]
        require(len(gi)==len(go)==len(raw)==192,'all raw derivative witnesses')
        for i,(a,b,c) in enumerate(zip(gi,go,raw,strict=True)):
            require(np.array_equal(a['grad:basis'],np.eye(192,dtype=np.float32)[i]) and np.array_equal(a['grad:raw'],np.zeros(10*h,np.float32)) and
                np.array_equal(a['grad:output'],n) and np.array_equal(b,c) and np.array_equal(b,j[i]),'raw coordinate scalar VJP/J row identity')
    signals,recomputed=recompute_signal(task,child,responses)
    require(canonical(recomputed)==canonical(acquisition['signal']),'independent full SVD/chi/signed gain recomputation')
    return signals


def validate_child_result(child,tasks,result,traces,sharing):
    child=contract.validate_child(child,tasks);task=core.validate_task(tasks[child['root_slot']])
    require(result['schema']==contract.NAME+'.child_plan' and result['child_id']==child['child_id'] and result['child_sha256']==digest(child) and
        result['fixed_denominator']==2 and result['sharing']==sharing and result['status']=='child_plan_not_admitted' and
        all(result[k] is False for k in FLAGS),'exact child result/closed gates')
    groups=[('shared',('fixed','adaptive'))] if sharing=='shared_equal' else [(m,(m,)) for m in ('fixed','adaptive')]
    require([a['group'] for a in result['acquisitions']]==[child['child_id']+':'+g for g,modes in groups] and
        [r['mode'] for r in result['rows']]==['fixed','adaptive'],'complete ordered groups/methods')
    node=dict(context=child['context'],depth=0)
    for acquisition,(group,modes) in zip(result['acquisitions'],groups,strict=True):
        sig=validate_trace(task,child,acquisition,traces[acquisition['group']],len(modes))
        require([op['scope'].get('mode') for op in acquisition['resource_receipt']['events'] if op['resource']=='schedule']==list(modes),
            'one paid schedule in exact method order')
        for mode in modes:
            row=result['rows'][('fixed','adaptive').index(mode)];decision=core.choose(task,node,sig,mode);h=decision['horizon'];count=decision['candidates']
            require(row['decision']==decision and row['status']==('pending_reviewed_child_compiler' if 2*h*count<=24 else 'blocked_whole_node_search_budget') and
                row['signal_group']==acquisition['group'] and row['signal_charge']==dict(numerator=1,denominator=len(modes)) and
                row['reserved_search_predict_API']==2*h*count and row['search_predict_cap']==24 and row['selected_prefix_predict_reserve']==1 and
                row['commands'] is row['composition_witness'] is row['error'] is None and row['generation_dispatch_allowed'] is False,'unchanged original allocation, budget refusals retained')
            queue=[core.request(task,node,operation='compile_child_program_extension',**decision,child_id=child['child_id'],child_sha256=digest(child),
                candidate_index=k,prefix_actions=[],prefix_edges=[],actual_executed_prefix=deepcopy(child['executed_prefix']),
                signal_sha256=digest(acquisition['signal']),local_response=deepcopy(acquisition['signal']['responses'])) for k in range(count)]
            require(canonical(queue)==canonical(row['queue']),'all original requests no shrink/future/outcome or stale signal')


class Store(base.Store):
    """Transactional compressor copy: failed reservations never advance compressor/chain.
    
    Only physical encoding differs. Every accepted canonical event is sync-flushed
    and fsynced. Thirty-two finish bytes are reserved before each dispatch/write
    (deflate end block plus gzip trailer; empty stream also fits).
    """
    def __init__(self,output,limits,*,clock=time.monotonic,free=None):
        self.output=Path(output);self.limits=limits;self.clock=clock;self.started=clock()
        self.free=free or (lambda:shutil.disk_usage(self.output.parent).free)
        self.memory=None;self.bytes=0;self.index=0;self.tail='0'*64;self.closed=False
        self.logical_bytes=0;self.compressor=zlib.compressobj(6,zlib.DEFLATED,31)
        self.output.mkdir(exist_ok=False);base.fsync_dir(self.output.parent)
        self.stream=(self.output/JOURNAL).open('xb');base.fsync_dir(self.output)

    def check(self,size=0,final=False):
        super().check(size+(0 if self.closed else 32),final)

    def event(self,value):
        require(not self.closed,'journal open')
        record=dict(index=self.index,previous=self.tail,payload=deepcopy(value))
        record['sha256']=digest(record);raw=canonical(record)+b'\n'
        require(len(raw)<=LINE_CAP and self.logical_bytes+len(raw)<=DECODED_CAP,'decoded journal engineering cap')
        candidate=self.compressor.copy()
        encoded=candidate.compress(raw)+candidate.flush(zlib.Z_SYNC_FLUSH)
        self.check(len(encoded))
        require(self.stream.write(encoded)==len(encoded),'short compressed journal write')
        self.stream.flush();os.fsync(self.stream.fileno())
        self.compressor=candidate;self.bytes+=len(encoded);self.logical_bytes+=len(raw)
        self.tail=record['sha256'];self.index+=1

    def put(self,name,value):
        if name=='report.json':
            require(self.closed,'close gzip before terminal report')
            require(len(canonical(value))<=self.limits['final_reserve_bytes'],'bounded full uncompressed report partition')
        return super().put(name,value)

    def close(self):
        if not self.closed:
            footer=self.compressor.copy().flush(zlib.Z_FINISH)
            # No wall/memory retry in cleanup; only finish already reserved bytes.
            require(self.bytes+len(footer)<=self.limits['output_bytes']-self.limits['final_reserve_bytes'],
                'reserved gzip footer cap')
            require(self.free()>=self.limits['disk_floor_bytes']+len(footer),'gzip footer disk floor')
            require(self.stream.write(footer)==len(footer),'short gzip footer')
            self.bytes+=len(footer);self.stream.flush();os.fsync(self.stream.fileno())
            self.stream.close();self.closed=True;base.fsync_dir(self.output)


def decoded_lines(path,*,decoded_cap=DECODED_CAP,line_cap=LINE_CAP):
    """Single complete gzip member ONLY; bounded incremental inflate and line assembly."""
    path=contract.absolute(path)
    require(path.stat().st_size<=CAP,'bounded compressed journal')
    decoder=zlib.decompressobj(31);pending=b'';total=0
    with path.open('rb') as stream:
        while True:
            chunk=stream.read(64*1024)
            if not chunk:break
            require(not decoder.eof,'trailing bytes or concatenated gzip member')
            while chunk:
                try:out=decoder.decompress(chunk,min(1024**2,decoded_cap-total+1))
                except zlib.error as exc:raise ValueError('corrupt gzip journal') from exc
                total+=len(out);require(total<=decoded_cap,'decoded journal engineering cap')
                pending+=out
                while b'\n' in pending:
                    line,pending=pending.split(b'\n',1)
                    require(len(line)+1<=line_cap,'bounded complete journal line')
                    yield line+b'\n'
                require(len(pending)<line_cap,'bounded complete journal line')
                chunk=decoder.unconsumed_tail
                require(not decoder.unused_data,'trailing bytes or concatenated gzip member')
    require(decoder.eof and not pending,'complete gzip footer and complete journal lines')


def iter_journal(path,**limits):
    tail='0'*64
    for index,line in enumerate(decoded_lines(path,**limits)):
        r=contract.rolling.decode(line);core.keys(r,('index','previous','payload','sha256'))
        body={k:r[k] for k in ('index','previous','payload')}
        require(type(r['index']) is int and r['index']==index and r['previous']==tail and digest(body)==r['sha256'],
            'complete event hash chain')
        require(canonical(r)+b'\n'==line,'original canonical event bytes')
        tail=r['sha256'];yield r['payload'],tail


def journal(path):
    """RAM compatibility helper; production consume uses bounded child-at-a-time iteration."""
    records=[];tail='0'*64
    for e,tail in iter_journal(path):records.append(e)
    return records,tail


def validate_stream(output,p,r):
    """Retain just one child's complete trace; all original per-array audits unchanged."""
    c=p['child_contract'];observed=[(row,ch) for row,ch in zip(r['rows'],c['children'],strict=True)
        if ch['status']!='not_reached']
    position=0;stage='denominator';traces={};tail='0'*64;before=after=sources=0
    expected_groups=[];seen_groups=set()
    for e,tail in iter_journal(output/JOURNAL):
        kind=e['kind']
        require(kind in {'denominator','model_before','child_trace','child_return','model_after','sources_after'},
            'closed child journal vocabulary')
        if stage=='denominator':
            require(e==dict(kind='denominator',rows=roster(c),methods=48,source_manifest_sha256=digest(p['sources'])),
                'full denominator durable before work')
            stage='before';continue
        if stage=='before':
            require(e==dict(kind='model_before',hashes=r['model_before']),'model guard before all work')
            before+=1;stage='work';continue
        if kind=='child_trace':
            require(stage=='work' and position<len(observed),'trace within model guards')
            core.keys(e,('kind','group','payload'))
            groups=[a['group'] for a in observed[position][0]['result']['acquisitions']]
            require(e['group'] in groups,'exact current child acquisition, no cross child')
            # Actual acquisitions are contiguous; may not return to an earlier group.
            if e['group'] not in traces:
                require(e['group']==groups[len(traces)],'actual original acquisition order')
                traces[e['group']]=[]
            require(e['group']==list(traces)[-1],'no interleaved/backtracked acquisition')
            traces[e['group']].append(e['payload'])
        elif kind=='child_return':
            require(stage=='work' and position<len(observed),'exact child return position')
            core.keys(e,('kind','row'));row,child=observed[position]
            require(e['row']==row,'one exact durable returned child')
            groups=[a['group'] for a in row['result']['acquisitions']]
            require(list(traces)==groups and not seen_groups.intersection(groups),'all and only own durable traces')
            validate_child_result(child,c['tasks'],row['result'],traces,r['sharing'])
            seen_groups.update(groups);expected_groups.extend(groups)
            traces={};position+=1
        elif kind=='model_after':
            require(stage=='work' and position==len(observed) and not traces and
                e==dict(kind='model_after',hashes=r['model_after']),'all work before model final guard')
            after+=1;stage='sources'
        elif kind=='sources_after':
            require(stage=='sources' and e==dict(kind='sources_after',sha256=digest(p['sources'])),
                'exact final source closure journal')
            sources+=1;stage='done'
        else:raise ValueError('duplicate/out of order journal guard')
    require(stage=='done' and before==after==sources==1 and tail==r['journal_tail'] and
        len(expected_groups)==p['profile']['acquisitions'],'complete gzip/chain/child panel terminal')
    return sum(x['status']=='blocked_whole_node_search_budget' for row,ch in observed for x in row['result']['rows'])


def consume(output,expected):
    output=contract.absolute(output);core.hash_value(expected)
    require(sha(output/'report.json')==expected,'external actual child report SHA')
    r=base.read(output/'report.json',256*1024**2);p=verify(r['prepared'],r['prepare_sha256']);c=p['child_contract']
    require(r['schema']==NAME+'.report' and r['status']=='paid_child_signals_saved_not_admitted' and r['errors']==[] and
        all(r[k] is False for k in FLAGS) and r['sources_verified'] is r['cleanup_ok'] is True and
        r['fixed_denominator']==48 and r['child_denominator']==24 and r['sharing']==c['inputs']['sharing'],'terminal complete known-child result only')
    require({f.name for f in output.iterdir()}=={'report.json','binding.json',JOURNAL} and
        set(r['artifact_sha256'])=={'binding.json',JOURNAL},'exact full artifacts')
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'actual child artifact SHA')
    require(base.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'execution prepare binding')
    require(sum(f.stat().st_size for f in output.iterdir())<=CAP and
        (output/JOURNAL).stat().st_size+(output/'binding.json').stat().st_size<=CAP-p['profile']['final_reserve_bytes'],
        'actual physical output partitions')
    require(r['model_before']==r['model_after'] and r['model_before']['original_state_dict']==p['expected_model_hash'],
        'actual unchanged frozen model receipts')
    require(r['source_manifest_sha256']==digest(p['sources']),'source manifest')
    for hashes in (r['model_before'],r['model_after']):
        core.keys(hashes,('original_state_dict','all_buffers'))
        for h in hashes.values():core.hash_value(h)
    require(len(r['rows'])==24 and [row['child_id'] for row in r['rows']]==[ch['child_id'] for ch in c['children']],
        'no missing child slots')
    for row,child in zip(r['rows'],c['children'],strict=True):
        if child['status']=='not_reached':
            require(row==dict(child_id=child['child_id'],status='not_reached',result=None,error=None),'unreached unknown, no padding')
        else:require(row['status']=='returned' and row['error'] is None,'all observed child signals return')
    refusals=validate_stream(output,p,r)
    total=total_resources(r['rows']);require(total==r['resources'] and total['attempted']==total['completed']==p['profile']['resource_caps'],'actual global same-unit ledger')
    require(type(r['wall_seconds']) in (int,float) and 0<=r['wall_seconds']<=p['profile']['wall_seconds'] and
        type(r['CUDA_peak_bytes']) in (list,tuple) and len(r['CUDA_peak_bytes'])==2 and all(type(v) is int and 0<=v<=p['profile'][k]
            for v,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'))),'actual wall/memory bounds')
    core.keys(r['host'],('device','index','name','allocation_sum','torch_version','torch_cuda','visible_nvidia_nodes'))
    require(r['host']['device']=='cuda:0' and type(r['host']['index']) is int and r['host']['index']==0 and
        type(r['host']['name']) is str and bool(r['host']['name']) and type(r['host']['torch_version']) is str and
        type(r['host']['torch_cuda']) is str and r['host']['allocation_sum']==4. and
        type(r['host']['visible_nvidia_nodes']) is list and r['host']['visible_nvidia_nodes'],'HOST metadata only, not GPU authenticity proof')
    contract.check_sources(p['sources']);pins()
    for name,h in r['artifact_sha256'].items():require(sha(output/name)==h,'final artifact drift')
    require(sha(output/'report.json')==expected,'final report drift')
    budget_by_mode={}
    for mode in ('fixed','adaptive'):
        measured=[x for row in r['rows'] if row['result'] is not None for x in row['result']['rows'] if x['mode']==mode]
        count=sum(x['status']=='blocked_whole_node_search_budget' for x in measured)
        budget_by_mode[mode]=dict(denominator=24,observed=len(measured),not_reached=24-len(measured),refused=count,
            all_observed_refused=(count==len(measured)) if measured else None,
            reason='original 2*horizon*candidates exceeds search cap24; no quota shrinking')
    return dict(schema=NAME+'.consume',status='paid_child_arrays_verified_not_admitted',report_sha256=expected,
        prepare_sha256=r['prepare_sha256'],fixed_denominator=48,child_denominator=24,observed_children=p['profile']['observed_children'],
        not_reached_methods=2*(24-p['profile']['observed_children']),measured_search_budget_refusals=refusals,
        all_observed_methods_budget_refused=(refusals==2*p['profile']['observed_children']) if p['profile']['observed_children'] else None,
        budget_refusals_by_mode=budget_by_mode,resources=total,rows=r['rows'],saved_verification=dict(array_SVD_calls=2*p['profile']['acquisitions'],neural_calls=0,physics_calls=0),
        physical_benefit=None,independent_roots=2,**FLAGS)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('prepare','verify','run','consume'))
    parser.add_argument('--prepared');parser.add_argument('--sha256',required=True);parser.add_argument('--output');parser.add_argument('--authorization')
    parser.add_argument('--receipt',help='consume only: NEW exclusive persistent strict receipt')
    a=parser.parse_args(argv)
    require(a.receipt is None or a.command=='consume','receipt only for saved consumption')
    if a.receipt:exclusive(a.receipt)
    if a.command=='prepare':
        require(a.output is not None,'NEW prepare output required');exclusive(a.output)
        result=write_new(a.output,material(a.prepared,a.sha256))
    elif a.command=='verify':
        p=verify(a.prepared,a.sha256);result=dict(status=p['status'],profile=p['profile'],**FLAGS)
    elif a.command=='run':result=run(a.prepared,a.sha256,a.output,a.authorization)
    else:
        result=consume(a.output,a.sha256)
        if a.receipt:result=write_new(a.receipt,result)
    print(canonical(result).decode());return 0


if __name__=='__main__':raise SystemExit(main())

