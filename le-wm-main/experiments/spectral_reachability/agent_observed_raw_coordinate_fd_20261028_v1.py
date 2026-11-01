"""All4 actual-cp5 iterate0, ten raw coordinates, 164 forward-only probes.

MAIN HOST CUDA only. Ambient actuator diagnostic, never a deployed selector.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import inspect
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_nonlinear_suffix_observed_recompile_20261028_v2 as source
from experiments.spectral_reachability import agent_nonlinear_suffix_scalar_fd_20261028_v1 as scalar_fd

NAME='agent_observed_raw_coordinate_fd_20261028_v1'
AUTH='MAIN_OBSERVED_RAW164_FD_HOST_CUDA_ONLY'
ROOT=source.ROOT
F,core,signal,transport=source.F,source.core,source.signal,source.transport
require,sha,digest,raw=source.require,source.sha,source.digest,source.raw
STEPS=(.01,.001)
PINS=((source,'e66bac861fce455ab6bb7a0cf12ba1644258c6c2e9cb6f2905973ac146e33567'),
    (scalar_fd,'a56e403c0d5b7a904b502c8044aaa4c575b667631fb98992716f1e13acd79639'))
PROTOCOL=ROOT.parent/'docs/D2P_RAW_COORDINATE_CAUSAL_PROTOCOL_MAIN_20261028_v1.md'


def contract():
    return dict(schema=NAME+'.contract',slots=4,known_roots=2,fresh_roots=0,coordinates=10,steps=list(STEPS),signs=[-1,1],
        baseline_per_slot=1,signed_per_slot=40,forwards_per_slot=41,forwards=164,central_pairs=80,
        order='slot, baseline, h(.01,.001), time-major coordinate0..9, sign(-1,+1)',
        anchor='observed_recompile_v2 original incumbent/iterate0 ONLY, actual cp5 context unchanged',
        perturbation='float32(float64(anchor_coordinate)+sign*h); every other raw bit unchanged',
        domain='all164 rawbox[-1,1] legal before any execution; never clamp/drop/replace',
        geometry='ambient raw actuator diagnostic; may exceed original logit L2 sphere; NOT deployable candidates',
        primary_FD='(plus-minus)/actual_float32_raw_spacing, both192-vector and native float32 mean goalQ',
        secondary_FD='nominal2h and original selector float64 meanQ separate; no best_h/threshold choice',
        max_new_compute=F.caps(164,0),new_AD=0,new_encoder=0,new_physics=0,new_training=0,new_selection=0,
        wall_seconds=1800,output_bytes=128*1024**2,final_reserve_bytes=16*1024**2,disk_floor_bytes=200*1024**2,
        cuda_allocated_bytes=3*1024**3,cuda_reserved_bytes=4*1024**3,
        CUDA_only=True,TF32=False,autocast=False,SDP='MATH',retries=0,total_FLOPs=None,equal_FLOPs=False,
        AD_correctness_certified=False,P1_admission=False,freshness=False,scientific_admission=False,physical_release=False)


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
def same(a,b):return raw(a)==raw(b)


def packet(value,shape=(5,2)):
    source.typed_tree(value)
    return scalar_fd.packet(value,shape)


def perturb(anchor,coord,h,sign):
    a=packet(anchor);require(type(coord) is int and 0<=coord<10 and type(h) is float and h in STEPS
        and type(sign) is int and sign in (-1,1),'only fixed literal coordinate/h/sign')
    b=a.copy();b.flat[coord]=np.float32(float(a.flat[coord])+sign*h)
    delta=float(b.flat[coord])-float(a.flat[coord])
    reason='outside_rawbox' if np.any(abs(b)>1) else ('collapsed_float32_step' if delta==0 else None)
    return b,delta,reason


def plan_jobs(prepared,report,strict_receipt,authority):
    """PURE manifest: no file reads, no model/physics, no fresh certificate.

    Production caller MUST first reconstruct source.consume and its saved receipt.
    Three explicit authority paths/SHA pairs remain attached to every probe.
    """
    p,r,a=prepared,report,strict_receipt
    core.keys(authority,('prepared','prepare_sha256','output','report_sha256','receipt','receipt_sha256'))
    for k in ('prepare_sha256','report_sha256','receipt_sha256'):core.hash_value(authority[k])
    for k in ('prepared','output','receipt'):require(type(authority[k]) is str and Path(authority[k]).is_absolute(),'literal absolute authority')
    require(p['schema']==source.NAME+'.prepare' and same(p['contract'],source.contract()) and p['production_ready'] is True
        and r['schema']==source.NAME+'.report' and same(r['contract'],p['contract'])
        and r['status']=='observed_suffix4_recompiled_not_admitted' and not r['errors']
        and r['prepared']==authority['prepared'] and r['prepare_sha256']==authority['prepare_sha256']
        and a['schema']==source.NAME+'.consume' and a['status']=='observed_suffix4_verified_science_blocked'
        and a['complete'] is True and a['report_sha256']==authority['report_sha256'],
        'complete exact observed-v2 authority, NOT imagined or selected replacement')
    require(len(p['rows'])==len(r['rows'])==len(a['rows'])==len(a['plans'])==4 and a['fixed_denominator']==4
        and a['known_roots']==2 and r['known_roots']==2 and same(a['rows'],r['rows'])
        and all(a[k] is False for k in ('P1_admission','scientific_admission','freshness','physical_release')),
        'all4 accepted rows/denominator/gates')
    require(a['actual_new_compute']['attempted']==a['actual_new_compute']['completed']==F.caps(72,32),'original full72/32 receipt')
    jobs=[]
    for i,(spec,row,plan) in enumerate(zip(p['rows'],r['rows'],a['plans'],strict=True)):
        q=spec['request'];source.validate_request(q)
        require(type(row['index']) is int and row['index']==spec['index']==q['slot']==i and row['status']=='sealed'
            and same(plan,row['result']['plan']),'exact original sealed slot')
        nr=row['result']['numeric'];source.strict_numeric(q,nr,row['result']['numeric_events'])
        v=nr['iterates'][0];d=nr['gradients'][0];anchor=packet(q['incumbent']['commands'])
        require(F.bits(anchor)==v['raw_bits'] and same(v['commands'],q['incumbent']['commands']), 'literal original iterate0, not selected final')
        packet(d['raw_gradient']);packet(q['task']['goal'],(192,))
        prefix=packet(q['incumbent']['prefix'])
        require(np.all(abs(anchor)<=1) and np.all(abs(prefix)<=1),'original rawbox legal anchor/prefix')
        provenance=dict(authority=deepcopy(authority),source_sha256=PINS[0][1],parent_selection_digest=spec['parent_selection_digest'],
            parent_plan_sha256=spec['parent_plan_sha256'],midpoint_row_binding_sha256=spec['midpoint_row_binding_sha256'],
            request_sha256=digest(q),numeric_witness_sha256=digest(nr),iterate=0,gradient_index=0,
            original_ridge_domain=deepcopy(q['incumbent']['original_ridge_domain']),
            selected_incumbent_domain=q['incumbent']['selected_domain'],incumbent_logits=deepcopy(q['incumbent']['logits']))
        schedule=[(None,None,0)]+[(c,h,s) for h in STEPS for c in range(10) for s in (-1,1)]
        for local,(coord,h,sign) in enumerate(schedule):
            b,delta,reason=(anchor.copy(),0.,None) if coord is None else perturb(anchor.tolist(),coord,h,sign)
            pair_spacing=None
            if coord is not None:
                minus=perturb(anchor.tolist(),coord,h,-1)[0];plus=perturb(anchor.tolist(),coord,h,1)[0]
                pair_spacing=float(plus.flat[coord])-float(minus.flat[coord])
            job=dict(slot=i,index=len(jobs),local_index=local,id=f's{i}_baseline' if coord is None else f's{i}_h{STEPS.index(h)}_c{coord}_{"minus" if sign<0 else "plus"}',
                root_slot=q['root_slot'],condition=q['condition'],coord=coord,h=h,sign=sign,
                raw_suffix5=b.tolist(),raw_bits=F.bits(b),prefix5=prefix.tolist(),prefix_bits=F.bits(prefix),full10=prefix.tolist()+b.tolist(),
                anchor_raw_suffix5=anchor.tolist(),actual_delta=delta,nominal_delta=0. if coord is None else sign*h,
                actual_pair_spacing=pair_spacing,nominal_pair_spacing=None if h is None else 2*h,
                legal=reason is None,missing_reason=reason,task_id=q['task']['binding']['task_id'],task=deepcopy(q['task']),
                observed_context=deepcopy(q['observed_context']),observed_raw10=deepcopy(q['observed_raw10']),
                root_raw10=deepcopy(q['root_raw10']),original_AD_raw_gradient=deepcopy(d['raw_gradient']),
                baseline_reference_id=f's{i}_baseline',
                original_AD_context=deepcopy(d['context']) if coord is None else None,original_AD_native_loss=d['native_float32_loss'],
                original_iterate0_context=deepcopy(v['context']) if coord is None else None,original_iterate0_Q=v['Q'],
                provenance=provenance,deployment_allowed=False,physical_release=False)
            job['job_sha256']=digest(job);jobs.append(job)
    require(len(jobs)==164,'fixed164, no filtering');return jobs


def points_from_jobs(jobs):
    require(len(jobs)==164,'complete manifest denominator')
    return [dict(index=i,root_slot=jobs[41*i]['root_slot'],condition=jobs[41*i]['condition'],
        task=deepcopy(jobs[41*i]['task']),context=deepcopy(jobs[41*i]['observed_context']),
        context_sha256=digest(jobs[41*i]['observed_context']),raw5=deepcopy(jobs[41*i]['anchor_raw_suffix5']),
        goal=deepcopy(jobs[41*i]['task']['goal']),raw_gradient=deepcopy(jobs[41*i]['original_AD_raw_gradient'])) for i in range(4)]


def build(observed_output,observed_report_sha,observed_prepared,observed_prepare_sha,observed_receipt,observed_receipt_sha):
    for module,h in PINS:require(sha(module.__file__)==h,'frozen source pin')
    source.check_pins()
    for f in (observed_output,observed_prepared,observed_receipt):require(Path(f).is_absolute() and Path(f).resolve()==Path(f),'absolute real paths')
    output=Path(observed_output)
    for f,h in ((output/'report.json',observed_report_sha),(observed_prepared,observed_prepare_sha),(observed_receipt,observed_receipt_sha)):
        core.hash_value(h);require(sha(f)==h,'explicit actual observed prepare/report/strict SHA')
    accepted=source.consume(output,observed_report_sha)
    require(same(accepted,signal.read(observed_receipt,32*1024**2)),'entire original strict rebuilt, NOT status-only')
    p=signal.read(observed_prepared,32*1024**2);r=signal.read(output/'report.json',32*1024**2)
    authority=dict(prepared=observed_prepared,prepare_sha256=observed_prepare_sha,output=observed_output,
        report_sha256=observed_report_sha,receipt=observed_receipt,receipt_sha256=observed_receipt_sha)
    jobs=plan_jobs(p,r,accepted,authority);bindings=deepcopy(p['bindings'])
    def merge(items):
        for f,h in items.items():require(f not in bindings or bindings[f]==h,'source/artifact collision');bindings[f]=h
    merge({str(output/'report.json'):observed_report_sha,observed_prepared:observed_prepare_sha,observed_receipt:observed_receipt_sha})
    merge({str(output/f):h for f,h in r['artifact_sha256'].items()})
    merge({str(Path(module.__file__).resolve()):h for module,h in PINS})
    merge({str(f):sha(f) for f in own_files()+[PROTOCOL]});F.check_bindings(bindings)
    ready=all(j['legal'] for j in jobs)
    return dict(schema=NAME+'.prepare',arguments=dict(observed_output=observed_output,observed_report_sha=observed_report_sha,
        observed_prepared=observed_prepared,observed_prepare_sha=observed_prepare_sha,observed_receipt=observed_receipt,observed_receipt_sha=observed_receipt_sha),
        contract=contract(),authority=authority,full_upstream_strict_rebuilt_here=True,bindings=bindings,jobs=jobs,points=points_from_jobs(jobs),
        manifest_sha256=digest(jobs),production_ready=ready,status='ready_ambient_diagnostic_only' if ready else 'blocked_rawbox_manifest_retained164',
        planned_compute=F.caps(164,0),actual_new_compute=None,inherited_compute=accepted['actual_new_compute'],
        inherited_resources=accepted['inherited_resources'],policy=p['policy'],expected_model_hash=p['expected_model_hash'],
        original_model_before=r['model_before'],original_model_after=r['model_after'])


def verify(prepared,expected):
    core.hash_value(expected);require(sha(prepared)==expected,'own prepare hash');p=signal.read(prepared,32*1024**2)
    require(same(p,build(**p['arguments'])) and sha(prepared)==expected,'whole actual authority/source/manifest rebuilt');return p


def write(path,value):
    path=Path(path).absolute();data=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and path.resolve()==path and not path.exists()
        and len(data)<=32*1024**2 and shutil.disk_usage(path.parent).free>=contract()['disk_floor_bytes']+len(data),'exclusive bounded metadata')
    with path.open('xb') as f:require(f.write(data)==len(data),'full metadata');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


class Store(scalar_fd.Store):
    # Policy-free namespace/check/write ABI reuse. Limits EXACTLY old scalar FD.
    def __init__(self,path,started):
        self.path=Path(path).absolute();self.started=started;self.finalizing=False
        require(self.path.parent==ROOT/'results' and self.path.name.startswith(NAME+'_') and self.path.resolve()==self.path
            and not self.path.exists() and shutil.disk_usage(self.path.parent).free>=contract()['output_bytes']+contract()['disk_floor_bytes'],'own exclusive original128MiB+200MiB')
        for k in ('wall_seconds','output_bytes','final_reserve_bytes','disk_floor_bytes'):
            require(contract()[k]==scalar_fd.contract()[k],'unchanged inherited Store resource contract')
        self.path.mkdir();signal.fsync_dir(self.path.parent)


def new_row(i):return dict(index=i,status='not_started',values=[],resources=None,journal_file=None,journal_sha256=None,journal_tail=None)


def evaluate(point,jobs,backend,row,journal):
    require(len(jobs)==41 and all(j['legal'] and j['slot']==point['index'] for j in jobs),'all41 legal before first forward')
    row['status']='attempted';row['values']=[dict(local_index=k,job_id=j['id'],status='not_started',context=None,objective=None) for k,j in enumerate(jobs)]
    for j,value in zip(jobs,row['values'],strict=True):
        journal.event(dict(kind='request_intent',job_id=j['id'],job_sha256=j['job_sha256'],context_sha256=point['context_sha256']))
        value['status']='entered'
        after=backend.step(deepcopy(point['context']),j['raw_suffix5'],j['id'])
        arrays=backend.objective(after);diagnostic=scalar_fd.check_objective(arrays,after['latent_history'][-1],point['goal'])
        if j['local_index']==0:
            require(same(after,j['original_iterate0_context']) and diagnostic['float64_endpoint_Q']==j['original_iterate0_Q'],
                'baseline MUST exactly reproduce actual observed original iterate0 full context and Q')
        value.update(status='returned',context=after,objective=dict(arrays=arrays,diagnostic=diagnostic))
        journal.event(dict(kind='request_return',job_id=j['id'],value_sha256=digest(value),objective=deepcopy(value['objective'])))
    row['status']='recorded'


def vector_metrics(v,g):
    v,g=np.asarray(v,float),np.asarray(g,float);e=v-g;nv,ng=float(np.linalg.norm(v)),float(np.linalg.norm(g))
    return dict(L2_error=float(np.linalg.norm(e)),Linf_error=float(abs(e).max()),relative_L2=None if ng==0 else float(np.linalg.norm(e))/ng,
        cosine=None if nv*ng==0 else float(v@g)/(nv*ng),FD_L2=nv,AD_L2=ng)


def summarize(point,jobs,values):
    require(len(jobs)==len(values)==41,'full41 summary')
    ad=packet(point['raw_gradient']).astype(float).reshape(10);steps=[]
    for k,h in enumerate(STEPS):
        pairs=[]
        for c in range(10):
            n=1+20*k+2*c;a,b=jobs[n:n+2];minus,plus=values[n:n+2]
            span=a['actual_pair_spacing'];require(type(span) is float and span>0 and span==b['actual_pair_spacing'],'actual raw span')
            qminus,qplus=(v['objective']['diagnostic'] for v in (minus,plus))
            dq=qplus['native_float32_Q']-qminus['native_float32_Q']
            df=qplus['float64_endpoint_Q']-qminus['float64_endpoint_Q']
            zm,zp=(packet(v['context']['latent_history'][-1],(192,)).astype(float) for v in (minus,plus))
            fd=dq/span
            pairs.append(dict(coord=c,minus_job_id=a['id'],plus_job_id=b['id'],nominal_spacing=2*h,actual_float32_spacing=span,
                native_Q_minus=qminus['native_float32_Q'],native_Q_plus=qplus['native_float32_Q'],FD_native_Q_actual_spacing=fd,
                FD_native_Q_nominal_spacing=dq/(2*h),FD_float64_Q_actual_spacing=df/span,
                endpoint192_derivative_actual_spacing=((zp-zm)/span).tolist(),
                endpoint192_derivative_nominal_spacing=((zp-zm)/(2*h)).tolist(),AD_raw_gradient=float(ad[c]),
                signed_error=fd-float(ad[c]),abs_error=abs(fd-float(ad[c]))))
        steps.append(dict(h=h,pairs=pairs,metrics_actual_spacing=vector_metrics([v['FD_native_Q_actual_spacing'] for v in pairs],ad),
            metrics_nominal_spacing=vector_metrics([v['FD_native_Q_nominal_spacing'] for v in pairs],ad)))
    base=values[0];j=jobs[0];actual=packet(base['context']['latent_history'][-1],(192,))
    old=packet(j['original_iterate0_context']['latent_history'][-1],(192,));oldad=packet(j['original_AD_context']['latent_history'][-1],(192,))
    return dict(index=point['index'],root_slot=point['root_slot'],condition=point['condition'],steps=steps,pair_denominator=20,
        baseline=dict(endpoint192=actual.tolist(),original_iterate0_endpoint192=old.tolist(),original_AD_endpoint192=oldad.tolist(),
            exact_original_iterate0=actual.tobytes()==old.tobytes(),exact_original_AD=actual.tobytes()==oldad.tobytes(),
            native_mean_Q=base['objective']['diagnostic']['native_float32_Q'],original_AD_native_loss=j['original_AD_native_loss'],
            original_iterate0_Q=j['original_iterate0_Q'],baseline_native_loss_equal=base['objective']['diagnostic']['native_float32_Q']==j['original_AD_native_loss']),
        original_AD_raw_gradient=point['raw_gradient'],AD_correctness_certified=False,best_h_selected=False,physical_release=False)


def audit_row(point,jobs,row,events):
    require(row['status']=='recorded' and len(row['values'])==41,'complete41 actual returns')
    controls=[e for e in events if e['kind'] in ('request_intent','request_return')]
    native=[e for e in events if e['kind'] not in ('request_intent','request_return')]
    allowed={'resource_attempt','resource_return','forward_started','forward_finished','program_packet','program_boundary','io_input','io_return'}
    require(all(e['kind'] in allowed for e in native),'no unknown/AD event filtered')
    starts,ends,_=F.audit_native(point['task'],row['resources'],native,[(j['id'],point['context'],j['raw_suffix5']) for j in jobs])
    positions={id(e):i for i,e in enumerate(events)};want=[]
    for k,(j,value,st,en) in enumerate(zip(jobs,row['values'],starts,ends,strict=True)):
        core.keys(value,('local_index','job_id','status','context','objective'))
        require(type(value['local_index']) is int and value['local_index']==k and value['job_id']==j['id']
            and value['status']=='returned' and same(value['context'],en['context']),'actual return/index/context')
        require(same(st['input'],point['context']) and same(st['commands'],j['raw_suffix5']),'bit-exact original observed input and probe raw')
        scalar_fd.audit_native_bits(point,dict(commands=j['raw_suffix5']),st['phase'],native,en)
        check=scalar_fd.check_objective(value['objective']['arrays'],en['context']['latent_history'][-1],point['goal'])
        require(same(check,value['objective']['diagnostic']),'native objective recomputed')
        if k==0:require(same(value['context'],j['original_iterate0_context']) and check['float64_endpoint_Q']==j['original_iterate0_Q'],'strict baseline original iterate0 equality')
        want.extend([dict(kind='request_intent',job_id=j['id'],job_sha256=j['job_sha256'],context_sha256=point['context_sha256']),
            dict(kind='request_return',job_id=j['id'],value_sha256=digest(value),objective=value['objective'])])
        require(positions[id(controls[2*k])]<positions[id(st)]<positions[id(en)]<positions[id(controls[2*k+1])]
            and (k==0 or positions[id(controls[2*k-1])]<positions[id(controls[2*k])]),'durable intent/native/return sequentially before next probe')
    require(same(controls,want) and events[0]==want[0] and events[-1]==want[-1],'full lifecycle exact ABI')
    return summarize(point,jobs,row['values'])


def dispatch(p,store,rows,check,factory):
    require(p['production_ready'] is True and len(p['jobs'])==164 and all(j['legal'] for j in p['jobs']),'whole164 rawbox preflight, no partial legal subset')
    for i,(point,row) in enumerate(zip(p['points'],rows,strict=True)):
        journal=transport.NativeJournal(store,f'p{i}_native',check);ledger=core.Ledger(F.caps(41,0),sink=journal.event)
        row['status']='attempted'
        try:
            backend=factory(point['task'],journal,ledger)
            with backend.observe():evaluate(point,p['jobs'][41*i:41*(i+1)],backend,row,journal)
            require(ledger.attempted==ledger.completed==F.caps(41,0),'actual41 forward/zeroAD')
        finally:
            if row['status']!='recorded':row['status']='failed_partial'
            row['resources']=ledger.receipt();journal.close()
            row.update(journal_file=f'p{i}_native/events.jsonl',journal_sha256=sha(journal.path),journal_tail=journal.tail)
            store.put(f'p{i}_row.json',row,final=True)


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN HOST CUDA authorization before I/O');started=time.monotonic();p=verify(prepared,expected)
    require(p['production_ready'] is True and all(j['legal'] for j in p['jobs']),'all raw probes legal before model load/output')
    store=Store(output,started);rows=[new_row(i) for i in range(4)];errors=[];model=None;before=after=flags=host=pb=pa=peaks=None;sources_ok=False
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
            store.check();source.precision(torch)
            require(not torch.is_grad_enabled(),'FD no autograd graph')
            require(torch.cuda.memory_allocated(device)<=contract()['cuda_allocated_bytes']
                and torch.cuda.memory_reserved(device)<=contract()['cuda_reserved_bytes'],'unchanged original CUDA caps')
        with torch.no_grad(),torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            check();model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False);check()
            before=signal.model_hash(model);require(before==p['original_model_before']==p['original_model_after']
                and before['original_state_dict']==p['expected_model_hash'],'original model+all buffers')
            for module in model.modules():require(str(Path(inspect.getfile(type(module))).resolve()) in p['bindings'],'all loaded classes presealed')
            pb=source.precision(torch);dispatch(p,store,rows,check,lambda task,j,l:scalar_fd.Backend(model,device,task,j,l,check));pa=source.precision(torch)
            torch.cuda.synchronize(device);check()
    except BaseException as exc:errors.append(dict(stage='run',error=repr(exc)))
    finally:
        store.finalizing=True
        try:
            if model is not None:
                source.midpoint.bridge.native.frozen(model);after=signal.model_hash(model);require(before==after,'model unchanged')
                peaks=[torch.cuda.max_memory_allocated(device),torch.cuda.max_memory_reserved(device)]
            F.check_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');sources_ok=True
        except BaseException as exc:errors.append(dict(stage='closure',error=repr(exc)))
        wall=time.monotonic()-started
        if wall>=contract()['wall_seconds']:errors.append(dict(stage='wall',error='whole1800s including verify exceeded'))
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        good=not errors and sources_ok and before is not None and before==after and all(r['status']=='recorded' for r in rows)
        report=dict(schema=NAME+'.report',status='observed164_recorded_diagnostic_only' if good else 'partial_failclosed',contract=contract(),
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,rows=rows,errors=errors,host=host,fixed_denominator=164,
            source_verified=sources_ok,source_manifest_sha256=digest(p['bindings']),model_before=before,model_after=after,
            precision_before=pb,precision_after=pa,CUDA_peak_bytes=peaks,wall_seconds=wall,
            artifact_sha256={str(f.relative_to(store.path)):sha(f) for f in store.path.rglob('*') if f.is_file()},
            AD_correctness_certified=False,P1_admission=False,freshness=False,scientific_admission=False,physical_release=False)
        store.put('report.json',report,final=True)
    return dict(output=str(store.path),report_sha256=sha(store.path/'report.json'),status=report['status'],errors=errors)


def consume(output,expected):
    output=Path(output).absolute();core.hash_value(expected);require(output.resolve()==output and sha(output/'report.json')==expected,'own report SHA')
    r=signal.read(output/'report.json',16*1024**2);p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and same(r['contract'],contract()) and len(r['rows'])==4 and r['fixed_denominator']==164
        and all(r[k] is False for k in ('AD_correctness_certified','P1_admission','freshness','scientific_admission','physical_release')),'full164/no admission')
    require(signal.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']),'real execution binding')
    files={str(f.relative_to(output)) for f in output.rglob('*') if f.is_file()}
    require(files==set(r['artifact_sha256'])|{'report.json'} and sum(f.stat().st_size for f in output.rglob('*') if f.is_file())<=contract()['output_bytes'],'whole output census/cap')
    for name,h in r['artifact_sha256'].items():
        f=Path(name);require(not f.is_absolute() and all(v not in ('.','..') for v in f.parts)
            and (output/f).resolve()==output/f and sha(output/f)==h,'all artifacts path/hash')
    panels=[];receipts=[];complete=r['status']=='observed164_recorded_diagnostic_only' and not r['errors']
    for i,(point,row) in enumerate(zip(p['points'],r['rows'],strict=True)):
        require(type(row['index']) is int and row['index']==i,'all4 exact positions')
        if row['resources'] is None:require(row==new_row(i),'untouched unknowns');complete=False;continue
        require(row['journal_file']==f'p{i}_native/events.jsonl' and same(signal.read(output/f'p{i}_row.json',8*1024**2),row),'exact durable row/journal routing')
        es=F.load_journal(output,row);transport.audit_resources(row['resources'],es);receipts.append(row['resources'])
        if row['status']=='recorded':panels.append(audit_row(point,p['jobs'][41*i:41*(i+1)],row,es))
        else:require(row['status']=='failed_partial','unknown status rejected');complete=False
    counts={kind:{k:sum(v[kind][k] for v in receipts) for k in core.AXES} for kind in ('attempted','completed')}
    if complete:
        require(len(panels)==4 and counts['attempted']==counts['completed']==F.caps(164,0),'actual164 / zero AD')
        require(r['source_verified'] is True and r['source_manifest_sha256']==digest(p['bindings'])
            and r['model_before']==r['model_after']==p['original_model_before']==p['original_model_after'],'all model/source seals')
        require(r['precision_before']==r['precision_after']==dict(matmul_TF32=False,cudnn_TF32=False,autocast=False,math=True,flash=False,efficient=False,cudnn_sdp=False),'original actual precision')
        source.scalar(r['wall_seconds'],'wall');require(r['wall_seconds']<contract()['wall_seconds'] and len(r['CUDA_peak_bytes'])==2
            and all(type(v) is int and 0<=v<=contract()[k] for v,k in zip(r['CUDA_peak_bytes'],('cuda_allocated_bytes','cuda_reserved_bytes'),strict=True)),'wall/CUDA caps')
    F.check_bindings(p['bindings']);require(sha(output/'report.json')==expected,'report final seal')
    return dict(schema=NAME+'.consume',status='observed164_verified_diagnostic_only' if complete else 'partial_failclosed',complete=complete,
        report_sha256=expected,authority=p['authority'],manifest_sha256=p['manifest_sha256'],fixed_denominator=164,pair_denominator=80,
        known_roots=2,panels=panels,actual_new_compute=counts,inherited_compute=p['inherited_compute'],inherited_resources=p['inherited_resources'],
        total_FLOPs=None,AD_correctness_certified=False,P1_admission=False,freshness=False,scientific_admission=False,physical_release=False)


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('observed-output','observed-report-sha','observed-prepared','observed-prepare-sha','observed-receipt','observed-receipt-sha',
        'prepared','prepare-sha','output','report-sha','receipt','authorization'):q.add_argument('--'+k)
    a=q.parse_args()
    if a.command=='prepare':v=write(a.output,build(a.observed_output,a.observed_report_sha,a.observed_prepared,a.observed_prepare_sha,a.observed_receipt,a.observed_receipt_sha))
    elif a.command=='verify':v=dict(verified=True,production_ready=verify(a.prepared,a.prepare_sha)['production_ready'])
    elif a.command=='run':v=run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else:require(a.receipt,'own receipt path');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),flush=True)


if __name__=='__main__':main()
