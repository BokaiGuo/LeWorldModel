"""Prepare without torch; explicit reviewed --execute-cuda uses CudaReplay only.

All sources, inputs and allocation formula are sealed BEFORE model loading.
No simulator imports, physical execution, outcome-driven tuning, or CPU neural
fallback. Current four roots remain calibration even if every runtime check passes.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.spectral_reachability import p3_fixed_horizon_budget_schedule_20261027_v2 as scheduler
from experiments.spectral_reachability import p3_quality_cost_audit_20261027_v2 as io
from experiments.spectral_reachability.d2p_adaptive_budget_v2 import PredictionBudget
from experiments.spectral_reachability.d2p_planning_compute_ledger import Binding
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

NAME='p3_fixed_horizon_budget_schedule_20261027_v2'
COMPILATION='results/pusht_d2p_metric_program_preflight4_v1_20261026_cuda/report.json'
COMPILATION_SHA='d5f544689a9d2b5c6807b2454c1a98b9db6d88bb374badc457bdcc6defa87c8a'
UPSTREAM='results/P3_ADAPTIVE_RUNTIME_20261027_cuda_v1/P3_ADAPTIVE_RUNTIME_protocol_20261027.json'
UPSTREAM_ID='a18656bd92f5db33bb95227d8d1e643fcd30096856151641c25648a6d355e5c0'
METHODS=('uniform','dynamic')
require=scheduler.require


def write(path,value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def contract():
    return dict(horizon=3,raw_steps_per_token=5,bank_size=16,planning_samples=144,
        verification_samples=3,predict_sample_cap=147,batch=1,
        uniform=[4,4],dynamic='K2=2+floor(4*chi+0.5); K3=8-K2',svd_relative_cutoff=1e-6,
        residual='r=(goal_vectors[0]-root_latent[-1])@Q.T',
        response='D=(all_16_fresh_H1_latents-root_latent[-1])@Q.T; no centering',
        chi='||r-Proj_rowspan(D)(r)||^2/||r||^2; nonzero residual and response rank required',
        metric='mean(((terminal_latent-goal_vectors[0])@Q.T)^2)',
        selection='H3 only; cost then lexicographic primitive path; no shallow winner',
        feedback='cached initial latent/action history; no environment observations or replanning',
        excluded_costs='upstream compilation/checkpoint loading/source hashing; no total pipeline comparison',
        evidence_scope='four existing calibration roots; no physical quality or P3 admission')


def packet(row):
    """Deliberately project source onto inputs, excluding old candidate outcomes."""
    c=row['compiled']; initial=c['input_context']
    receipts=c['receipts']
    require([r['candidate_id'] for r in receipts]==[f'prefix_{i}' for i in range(16)],'ordered complete raw bank')
    for r in receipts:
        a=np.asarray(r['program_live_commands'])
        require(a.shape==(2,2) and a.dtype.kind in 'fi' and np.isfinite(a).all() and np.all(np.abs(a)<=1),
                'source prefix/suffix commands must be finite and bounded')
    p=dict(root_sha256=row['root_sha256'],root_context=dict(
        layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=initial['emb'][0],action_history=initial['actions'][0][1:]),
        # Exactly the old runtime raw bank: repeat PREFIX command five times.
        # The compiler's second command is its suffix, not part of this bank.
        primitive_bank=[np.repeat(np.asarray(r['program_live_commands'])[:1],5,axis=0).tolist() for r in receipts],
        goal=c['goal_vectors'][0],projection=c['scoring']['metric']['projection'])
    scheduler._validated_context(p['root_context'])
    scheduler.geometry_inputs(p['root_context'],p['goal'],p['projection'])
    p['input_sha256']=io.sha(io.canonical(p).encode())
    return p


def prepare(root,output):
    root=Path(root).resolve(); output=Path(output).absolute()
    require(not output.exists(),'exclusive NEW prepare output required')
    evidence=io.Evidence(root)
    source=evidence.json(root/COMPILATION,COMPILATION_SHA)
    upstream=evidence.json(root/UPSTREAM)
    require(io.sha(io.canonical(upstream['protocol']).encode())==upstream['protocol_id']==UPSTREAM_ID,
            'frozen upstream protocol content identity')
    require(source['status']=='compiled_calibration' and all(v is True for v in source['gates'].values()),
            'complete calibration compilation required')
    evidence.seal(source['source_seal'])
    sources={Path(p) for p in upstream['protocol']['sources']}
    for p,h in upstream['protocol']['sources'].items(): evidence.raw(Path(p),h)
    for p,h in upstream['protocol']['bindings'].items():
        if Path(p).is_relative_to(root.parent): evidence.raw(Path(p),h)
    packets=[packet(r) for r in source['rows']]
    roots=source['protocol']['roots']
    require(len(roots)==len(set(roots))==4 and roots==upstream['protocol']['roots'] and
            [p['root_sha256'] for p in packets]==roots,'exact four calibration root identities')
    module_root=root/'experiments/spectral_reachability'
    sources.update([Path(__file__).resolve(),Path(scheduler.__file__).resolve(),Path(io.__file__).resolve(),
        root/'tests'/('test_'+NAME+'.py'),root/'tests'/'test_p3_fixed_horizon_budget_schedule_cuda_20261027_v2.py',
        root/'docs'/(NAME+'.md'),module_root/'d2p_prerun_source_seal.py'])
    for p in sources: evidence.raw(p)
    body=dict(schema=NAME+'.protocol',contract=contract(),roots=roots,methods=list(METHODS),
        arm_order=[list(METHODS if i%2==0 else METHODS[::-1]) for i in range(4)],
        model_bindings={p:h for p,h in upstream['protocol']['bindings'].items() if not Path(p).is_relative_to(root.parent)},
        inputs_sha256=dict(sorted(evidence.hashes.items())),
        sources_sha256={str(p):evidence.hashes[str(p)] for p in sorted(sources)},packets=packets,
        expected_predict_samples=1176,expected_action_encoder_samples=1176,
        expected_processed_context_tokens=3528,expected_svd_calls=8,
        new_independent_roots=0,P1_admission=False,P3_admission=False)
    identity=io.sha(io.canonical(body).encode())
    evidence.recheck()
    output.mkdir(exist_ok=False)
    protocol_path=output/'protocol.json'
    write(protocol_path,dict(protocol=body,protocol_id=identity))
    seal=create_seal(files=sorted(sources),args_json=io.canonical(dict(
        protocol_path=str(protocol_path),protocol_sha256=io.sha(protocol_path.read_bytes()))),
        output_dir=output/'source_seal')
    evidence.recheck()
    receipt=dict(status='prepared_not_executed',protocol_id=identity,
        protocol_path=str(protocol_path),protocol_sha256=io.sha(protocol_path.read_bytes()),source_seal=seal,
        new_predict_samples=0,new_physical_steps=0,model_loaded=False,torch_imported=False,
        calibration_roots=4,new_independent_roots=0,scheduled_root_arms=8,
        expected_successful_predict_samples=1176,P3_admission=False,
        external_model_bindings_not_rehashed=body['model_bindings'])
    write(output/'prepare.json',receipt)
    print(io.canonical(receipt))
    return receipt


def validate_prepared(prepared,expected_sha):
    prepared=Path(prepared).absolute()
    receipt=io.decode((prepared/'prepare.json').read_bytes())
    require(io.sha((prepared/'prepare.json').read_bytes())==expected_sha,'prepare receipt SHA256 mismatch')
    require(receipt['status']=='prepared_not_executed' and receipt['model_loaded'] is False,'prepared-only receipt')
    protocol_path=prepared/'protocol.json'
    require(str(protocol_path)==receipt['protocol_path'] and
            io.sha(protocol_path.read_bytes())==receipt['protocol_sha256'],'prepared protocol byte binding')
    envelope=io.decode(protocol_path.read_bytes()); body=envelope['protocol']
    require(io.sha(io.canonical(body).encode())==envelope['protocol_id']==receipt['protocol_id'], 'protocol content SHA')
    require(io.canonical(body['contract'])==io.canonical(contract()),'fixed reviewed allocation/contract changed')
    require(body['methods']==list(METHODS) and len(body['roots'])==len(set(body['roots']))==4 and
            [p['root_sha256'] for p in body['packets']]==body['roots'],'prepared root census')
    require(body['arm_order']==[list(METHODS if i%2==0 else METHODS[::-1]) for i in range(4)],'frozen arm order')
    for p in body['packets']:
        require(io.sha(io.canonical({k:v for k,v in p.items() if k!='input_sha256'}).encode())==p['input_sha256'],
                'packet input identity')
    for p,h in body['inputs_sha256'].items():
        require(io.sha(Path(p).read_bytes())==h,'input/source byte changed: '+p)
    seal=receipt['source_seal']
    verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    return body,receipt['protocol_id']


class Work:
    def __init__(self):
        self.action_calls=self.action_samples=self.context_tokens=self.visual_samples=0

    def action_entry(self,module,args):
        b,t=args[0].shape[:2]
        self.action_calls+=1; self.action_samples+=b; self.context_tokens+=b*t

    def visual_entry(self,module,args):
        self.visual_samples+=args[0].shape[0]


def run_arm(*,model,replay,p,binding,mode,event_sink=None):
    """Same function is covered by metadata-only tests; CUDA wrapper supplies model."""
    work=Work(); hooks=[]; search=None; failure=None; exact=False
    method=NAME+'_'+mode
    with PredictionBudget(model,binding=binding,method=method,sample_limit=147) as budget:
        try:
            hooks.append(model.action_encoder.register_forward_pre_hook(work.action_entry))
            hooks.append(model.encoder.register_forward_pre_hook(work.visual_entry))
            with budget.phase('planning',method=method,horizon_tokens=3):
                search=scheduler.run_search(root_context=p['root_context'],primitive_bank=p['primitive_bank'],
                    replay=replay,goal=p['goal'],projection=p['projection'],mode=mode,event_sink=event_sink)
                require(search['status']=='completed','incomplete planning; no verification or substitute winner')
            require(budget.receipt()['attempted_predict_samples']==144,'exact B1 planning predictions required')
            with budget.phase('evaluation',method=method,horizon_tokens=3):
                actual=replay(deepcopy(p['root_context']),np.asarray(search['best_plan']['actions']))
                exact=io.canonical(actual)==io.canonical(search['best_plan']['context'])
                require(exact,'H3 whole-path verification differs')
        except Exception as exc:
            failure=dict(type=type(exc).__name__,message=str(exc))
        finally:
            for hook in hooks: hook.remove()
        receipt=budget.receipt()
    complete=(failure is None and exact and receipt['attempted_predict_samples']==147 and
        len(receipt['phases'])==2 and all(p['completed'] and p['failed_predict_calls']==0 for p in receipt['phases']) and
        [p['predict_samples'] for p in receipt['phases']]==[144,3] and receipt['blocked_calls']==0 and
        work.action_samples==work.action_calls==147 and work.context_tokens==441 and work.visual_samples==0)
    if not complete and failure is None: failure=dict(type='AccountingMismatch',message='actual resource coordinates differ')
    return dict(root_sha256=p['root_sha256'],mode=mode,status='completed' if complete else 'incomplete',
        input_sha256=p['input_sha256'],binding=receipt['binding'],search=search,prediction_receipt=receipt,
        whole_h3_replay_exact=exact,failure=failure,action_encoder_calls=work.action_calls,
        action_encoder_samples=work.action_samples,processed_action_context_tokens=work.context_tokens,
        visual_encoder_samples=work.visual_samples,runtime_action_backward_calls=0,runtime_action_vjp_rows=0,
        runtime_input_backward_calls=0,runtime_input_vjp_rows=0,
        derivative_scope='frozen no-grad CudaReplay code policy, not autograd invocation instrumentation',
        dense_jacobian_materializations=None,new_physical_steps=0,P3_admission=False)


def compare(rows,roots):
    index=io.index(rows,('root_sha256','mode'))
    require(set(index)=={(r,m) for r in roots for m in METHODS},'all root arms including failures required')
    pairs=[]
    for root in roots:
        u,d=(index[root,m] for m in METHODS)
        complete=all(r['status']=='completed' for r in (u,d))
        same_signal=complete and io.canonical(u['search']['signal'])==io.canonical(d['search']['signal'])
        same_input=(u.get('binding') is not None and d.get('binding') is not None and
                    u['input_sha256']==d['input_sha256'] and u['binding']==d['binding'])
        matched=complete and same_signal and same_input
        pairs.append(dict(root_sha256=root,classification='matched_full_H3' if matched else 'unmatched',
            same_H1_chi=same_signal,same_input_and_feedback=same_input,
            uniform_allocation=u['search']['allocation'] if u['search'] else None,
            dynamic_allocation=d['search']['allocation'] if d['search'] else None,
            dynamic_minus_uniform_model_cost=(d['search']['best_plan']['cost']-u['search']['best_plan']['cost']) if matched else None,
            quality_scope='saved model metric only; not physical quality or efficacy'))
    return pairs


def execute_cuda(*,prepared,prepare_sha256,output):
    """Only this explicitly selected branch may import torch or load a model."""
    output=Path(output).absolute()
    require(not output.exists(),'exclusive NEW execution output required')
    body,identity=validate_prepared(prepared,prepare_sha256)
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.pusht_composition_cuda_replay_v2 import CudaReplay
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    import torch
    root=Path(__file__).resolve().parents[2]
    policy=Path(next(p for p in body['model_bindings'] if Path(p).name=='weights.pt')).parent
    # Full weight/config byte validation is deferred to this explicit CUDA launch.
    checked.validate_inputs(root/COMPILATION,policy)
    for p,h in body['model_bindings'].items(): require(checked.digest(p)==h,'external model binding changed')
    validate_prepared(prepared,prepare_sha256)
    output.mkdir(exist_ok=False)
    write(output/'execution_protocol.json',dict(protocol_id=identity,prepare_sha256=prepare_sha256))
    rows=[]; host=None; failure=None; before=after=None
    started=perf_counter()
    try:
        host=require_cuda_allocation('cuda')
        device=torch.device('cuda:0')
        model=load_compatible_pretrained(str(policy),device=device).eval().requires_grad_(False)
        replay=CudaReplay(model,device)
        before=checked.tensor_state_hash(model)
        with torch.no_grad():
            for i,p in enumerate(body['packets']):
                binding=Binding(identity,'pusht_fixed_h3_cached_latent_calibration',p['root_sha256'],
                    checked.OFFICIAL_SHA+':'+checked.CONFIG_SHA)
                for mode in body['arm_order'][i]:
                    journal_path=output/(p['root_sha256']+'_'+mode+'.events.jsonl')
                    with journal_path.open('x') as stream:
                        def journal(event):
                            stream.write(io.canonical(event)+'\n'); stream.flush(); os.fsync(stream.fileno())
                        torch.cuda.synchronize()
                        row=run_arm(model=model,replay=replay,p=p,binding=binding,mode=mode,event_sink=journal)
                        torch.cuda.synchronize()
                    row.update(journal_path=str(journal_path),journal_sha256=checked.digest(journal_path))
                    rows.append(row)
                    write(output/(p['root_sha256']+'_'+mode+'.json'),row)
        after=checked.tensor_state_hash(model)
    except (Exception,KeyboardInterrupt) as exc:
        failure=dict(type=type(exc).__name__,message=str(exc))
    # Persist every predeclared slot, even after setup/interruption failure.
    seen={(r['root_sha256'],r['mode']) for r in rows}
    for p in body['packets']:
        for mode in METHODS:
            if (p['root_sha256'],mode) not in seen:
                journal_path=output/(p['root_sha256']+'_'+mode+'.events.jsonl')
                rows.append(dict(root_sha256=p['root_sha256'],mode=mode,status='blocked',
                    input_sha256=p['input_sha256'],binding=None,search=None,
                    prediction_receipt=None,failure=failure or {'message':'arm interrupted before persistence'},
                    actual_predict_samples=None,unknown_attempted_work=True,
                    journal_path=str(journal_path) if journal_path.exists() else None,
                    journal_sha256=checked.digest(journal_path) if journal_path.exists() else None))
    source_failure=None
    try: validate_prepared(prepared,prepare_sha256)
    except Exception as exc: source_failure=str(exc)
    pairs=compare(rows,body['roots'])
    integrity=(failure is None and source_failure is None and before is not None and before==after and
               all(p['classification']=='matched_full_H3' for p in pairs))
    report=dict(status='calibration_recorded' if integrity else 'incomplete',rows=rows,pairs=pairs,
        protocol_id=identity,host=host,failure=failure,source_failure=source_failure,model_unchanged=before is not None and before==after,
        elapsed_seconds_including_setup=perf_counter()-started,
        calibration_roots=4,new_independent_roots=0,P1_admission=False,P3_admission=False,
        physical_quality_measured=False,total_flops_equality=False,
        known_predict_samples=sum(r['prediction_receipt']['attempted_predict_samples'] for r in rows if r['prediction_receipt']),
        unknown_attempted_work=any(r.get('unknown_attempted_work',False) for r in rows),
        new_physical_steps=0)
    write(output/'report.json',report)
    if not integrity: raise RuntimeError('incomplete calibration; retain all receipts, no retry/padding')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true')
    action.add_argument('--execute-cuda',action='store_true')
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--prepare-sha256')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.prepare:
        prepare(args.project_root,args.output)
    else:
        require(args.prepared is not None and args.prepare_sha256 is not None,'reviewed prepare path and SHA required')
        execute_cuda(prepared=args.prepared,prepare_sha256=args.prepare_sha256,output=args.output)
