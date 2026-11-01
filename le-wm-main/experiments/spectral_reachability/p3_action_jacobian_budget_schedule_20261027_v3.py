"""Action-Jacobian semantic repair of fixed-H3 scheduling; calibration only.

No H1-displacement signal, no outcome inputs, no formula retuning. Prepare and
array tests are model-free. Only reviewed --execute-cuda loads a neural model.
Inherited derivatives have UNKNOWN attributed upstream cost, not zero cost.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.spectral_reachability import p3_fixed_horizon_budget_schedule_cuda_20261027_v2 as old
from experiments.spectral_reachability import p3_fixed_horizon_physical_consumer_20261027_v2 as physical
from experiments.spectral_reachability.d2p_adaptive_budget_v2 import PredictionBudget
from experiments.spectral_reachability.d2p_planning_compute_ledger import Binding
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

NAME='p3_action_jacobian_budget_schedule_20261027_v3'
OLD_PREPARED='results/p3_fixed_horizon_budget_schedule_20261027_v2_prepared'
OLD_PREPARE_SHA='c4b3a2f5a8bca999707fe2c51ba41223d025b42b0978129cf0e801e79ebbc46d'
JACOBIAN_REPORT='results/pusht_full_packet_compiler_20261027_cuda_v1/report.json'
JACOBIAN_SHA='05eb37c9b1fc3fc4976da702239b13dce018eb9bb4067660687ada4e6d3988dc'
METHODS=old.METHODS
io=old.io
require=old.require
write=old.write
allocate=old.scheduler.allocate


def inherited_cost():
    return dict(status='inherited_unknown_cost',predict_samples=None,action_encoder_samples=None,
        backward_api_calls=None,vjp_directions=None,wall_time_seconds=None,
        scope='root H1 nominal and 192x10 action-Jacobian acquisition; not newly executed or free; '
              'no reliable root-only attributed receipt; shared input, never multiply upstream cost by arms')


def array(value,shape):
    a=np.asarray(value)
    require(a.shape==shape and a.dtype.kind in 'fi' and np.isfinite(a).all(),'finite real aligned arrays required')
    return a.astype(float)


def jacobian_signal(jacobian,nominal,goal,projection):
    """Squared unreachable residual in col(QJ), NOT rowspan or drift span."""
    j=array(jacobian,(192,10));n=array(nominal,(192,));g=array(goal,(192,));q=array(projection,(192,192))
    require(np.allclose(q,q.T,rtol=0,atol=1e-8) and np.allclose(q@q,q,rtol=0,atol=1e-8),
            'original Q must be symmetric idempotent')
    r=q@(g-n);k=q@j;norm2=float(r@r)
    require(math.isfinite(norm2) and norm2>0,'nonzero finite nominal residual required')
    u,s,_=np.linalg.svd(k,full_matrices=False)
    rank=int(np.sum(s>s[0]*1e-6)) if s[0]>0 else 0
    require(rank>0,'nonzero action image rank required; no uniform fallback')
    parallel=u[:,:rank]@(u[:,:rank].T@r)
    chi=float(np.clip(np.sum((r-parallel)**2)/norm2,0,1))
    require(math.isfinite(chi),'finite chi required')
    return dict(chi=chi,response_rank=rank,singular_values=s.tolist(),relative_svd_cutoff=1e-6,
        residual=r.tolist(),residual_squared_norm=norm2,projected_jacobian=k.tolist(),
        space='original frozen Q latent coordinates; r=Q(goal0-root_H1_zero_nominal); K=QJ; col(K)',
        jacobian_coordinates='d H1 terminal / d ten time-major atanh(raw) coordinates at zero; tanh derivative=1',
        scope='root-local action-Jacobian reachable image; not H3 controllability, not physical geometry')


def inherit_packets(body,report,compilation):
    """Binding-only projection: never use selected programs or physical outcomes."""
    require(report['schema']=='pusht_full_packet_compiler_20261027.v1' and report['status']=='plans_sealed' and
            report['training'] is False and report['new_physical_steps']==0 and
            report['all_plans_before_physical_execution'] is True and
            bool(report['gates']) and all(v is True for v in report['gates'].values()),'sealed derivative integrity')
    protocol=report['protocol']
    compilation_path=str(Path(__file__).resolve().parents[2]/old.COMPILATION)
    require(protocol['compilation']==compilation_path and
            protocol['bindings'][compilation_path]==old.COMPILATION_SHA,'same exact compilation binding required')
    require({k:protocol['bindings'].get(k) for k in body['model_bindings']}==body['model_bindings'],
            'same frozen model/config bindings required')
    roots=body['roots']
    require(protocol['roots']==roots and len(roots)==len(set(roots))==4,'same exact calibration roots')
    plans=io.index(report['plans'],('root_sha256',));rows=io.index(compilation['rows'],('root_sha256',))
    require(set(plans)==set(rows)=={(r,) for r in roots},'complete duplicate-free root Jacobian census')
    require(len(body['packets'])==4,'all original task packets required')
    packets=[]
    for i,p in enumerate(body['packets']):
        r=p['root_sha256'];require(r==roots[i],'root order binding')
        expected=old.packet(rows[r,])
        require(io.canonical(expected)==io.canonical(p),'same source root context, goal, Q, raw bank and packet SHA')
        plan=plans[r,]
        require(type(plan['root_index']) is int and plan['root_index']==i,'source root index binding')
        require(0<=plan['chain_rule_max_error']<=5e-5 and 0<=plan['full_fd_relative_error']<=.1,
                'inherited finite-difference and chain-rule integrity')
        j=array(plan['root_jacobian_full'],(192,10));n=array(plan['root_nominal_terminal'],(192,))
        jacobian_signal(j,n,p['goal'],p['projection'])
        new=deepcopy(p)
        new.update(inherited_jacobian=j.tolist(),inherited_nominal=n.tolist(),
            inherited_source=dict(report_sha256=JACOBIAN_SHA,root_sha256=r,root_index=i,
                original_packet_sha256=p['input_sha256'],nominal_logits=[0.]*10,
                derivative='root H1 terminal, ten time-major full packet logits; same frozen transition'))
        new['input_sha256']=io.sha(io.canonical({k:v for k,v in new.items() if k!='input_sha256'}).encode())
        packets.append(new)
    return packets


def contract():
    c=old.contract()
    c.update(residual='r=Q@(goal0-root_H1_zero_nominal)',response='K=Q@root_action_J_192x10',
        chi='||r-U_rank U_rank.T r||^2/||r||^2; U from SVD(K), relative cutoff 1e-6',
        signal_timing='independently recompute one inherited-array SVD per arm after all 16 fresh H1 edges',
        signal_semantics='local action derivative, not H1 finite displacement or H3/physical controllability',
        inherited_cost=inherited_cost(),new_jacobian_materializations=0,
        total_pipeline_cost_equivalent=False)
    return c


def prepare(root,output):
    root=Path(root).resolve();output=Path(output).absolute()
    require(not output.exists(),'exclusive NEW prepare output required')
    previous,_=old.validate_prepared(root/OLD_PREPARED,OLD_PREPARE_SHA)
    evidence=io.Evidence(root)
    evidence.raw(root/OLD_PREPARED/'prepare.json',OLD_PREPARE_SHA)
    evidence.raw(root/OLD_PREPARED/'protocol.json')
    # Preserve the entire validated predecessor dependency surface, not outcomes.
    for p,h in previous['inputs_sha256'].items():evidence.raw(Path(p),h)
    report=evidence.json(root/JACOBIAN_REPORT,JACOBIAN_SHA)
    compilation=evidence.json(root/old.COMPILATION,old.COMPILATION_SHA)
    sources={Path(p) for p in previous['sources_sha256']}
    for seal in report['source_seals']:
        evidence.seal(seal)
        manifest=evidence.json(Path(seal['output_dir'])/'manifest.json',seal['manifest_sha256'])
        sources.update(Path(f['source']) for f in manifest['files'])
    for p,h in report['protocol']['bindings'].items():
        if Path(p).is_relative_to(root.parent):evidence.raw(Path(p),h)
    for plan in report['plans']:
        saved=evidence.json((root/JACOBIAN_REPORT).parent/f'root_{plan["root_index"]}.plan.json')
        require(io.canonical(saved)==io.canonical(plan),'inherited plan/report bytes disagree')
    packets=inherit_packets(previous,report,compilation)
    sources.update([Path(__file__).resolve(),root/'tests'/('test_'+NAME+'.py'),root/'docs'/(NAME+'.md'),
        root/'tests/test_p3_fixed_horizon_budget_schedule_cuda_20261027_v2.py'])
    sources=physical.source_closure(root,sources)
    for path in sources:evidence.raw(path)
    body=dict(schema=NAME+'.protocol',contract=contract(),roots=previous['roots'],methods=list(METHODS),
        arm_order=previous['arm_order'],packets=packets,model_bindings=previous['model_bindings'],
        inputs_sha256=dict(sorted(evidence.hashes.items())),sources_sha256={str(p):evidence.hashes[str(p)] for p in sources},
        inherited_cost=inherited_cost(),inherited_arrays=4,inherited_report_sha256=JACOBIAN_SHA,
        total_pipeline_cost_equivalent=False,expected_predict_samples=1176,expected_action_encoder_samples=1176,
        expected_processed_context_tokens=3528,expected_svd_calls=8,prepare_numpy_svd_calls=4,new_jacobian_materializations=0,
        calibration_roots=4,new_independent_roots=0,P1_admission=False,P3_admission=False)
    identity=io.sha(io.canonical(body).encode());evidence.recheck()
    output.mkdir(exist_ok=False);path=output/'protocol.json'
    write(path,dict(protocol=body,protocol_id=identity));protocol_sha=io.sha(path.read_bytes())
    seals=[create_seal(files=sources[i:i+40],args_json=io.canonical(dict(protocol_path=str(path),
        protocol_sha256=protocol_sha)),output_dir=output/f'source_seal_{i//40}') for i in range(0,len(sources),40)]
    evidence.recheck()
    receipt=dict(status='prepared_not_executed',protocol_id=identity,protocol_path=str(path),protocol_sha256=protocol_sha,
        source_seals=seals,source_count=len(sources),new_predict_samples=0,new_physical_steps=0,model_loaded=False,
        calibration_roots=4,new_independent_roots=0,P3_admission=False,expected_predict_samples=1176,
        new_jacobian_materializations=0,prepare_numpy_svd_calls=4,inherited_cost=inherited_cost(),
        external_model_bindings_not_rehashed=body['model_bindings'])
    write(output/'prepare.json',receipt);print(io.canonical(receipt));return receipt


def validate_prepared(prepared,expected_sha):
    prepared=Path(prepared).absolute();path=prepared/'protocol.json'
    raw=(prepared/'prepare.json').read_bytes()
    require(io.sha(raw)==expected_sha,'prepare receipt SHA256 mismatch')
    receipt=io.decode(raw)
    require(receipt['status']=='prepared_not_executed' and receipt['model_loaded'] is False and
            receipt['protocol_path']==str(path) and io.sha(path.read_bytes())==receipt['protocol_sha256'],
            'prepared protocol byte binding')
    envelope=io.decode(path.read_bytes());body=envelope['protocol'];identity=envelope['protocol_id']
    require(io.sha(io.canonical(body).encode())==identity==receipt['protocol_id'],'protocol content SHA')
    require(body['schema']==NAME+'.protocol' and io.canonical(body['contract'])==io.canonical(contract()),'v3 fixed contract')
    require(body['methods']==list(METHODS) and len(body['roots'])==len(set(body['roots']))==4 and
        [p['root_sha256'] for p in body['packets']]==body['roots'],'prepared four-root packet census')
    require(body['arm_order']==[list(METHODS if i%2==0 else METHODS[::-1]) for i in range(4)],'arm order')
    for p in body['packets']:
        require(io.sha(io.canonical({k:v for k,v in p.items() if k!='input_sha256'}).encode())==p['input_sha256'],'packet content SHA')
    for p,h in body['inputs_sha256'].items():require(io.sha(Path(p).read_bytes())==h,'input/source changed: '+p)
    for seal in receipt['source_seals']:verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    return body,identity


def run_search(*,p,replay,mode,event_sink=None):
    """Same H3 fullbank enumeration and stable ties; only signal definition differs."""
    require(mode in METHODS,'explicit scheduler mode')
    root=old.scheduler._validated_context(p['root_context'])
    bank=old.scheduler._bank(p['primitive_bank'])
    require(len(bank)==16 and all(a.shape==(5,2) for a in bank),'full sixteen five-step bank')
    goal=array(p['goal'],(192,));q=array(p['projection'],(192,192))
    # Validate derivative arrays before spending predictions; SVD counted only below.
    array(p['inherited_jacobian'],(192,10));array(p['inherited_nominal'],(192,))
    result=dict(schema=NAME+'.search',status='incomplete',mode=mode,fixed_horizon_tokens=3,
        planning_cap=144,verification_reserve=3,events=[],layers=[],signal=None,svd_calls=0,
        allocation=None,best_plan=None,planning_edge_attempts=0,failure=None,P3_admission=False)
    frontier=[dict(context=root,actions=[],primitive_indices=[],depth=0)]
    def emit(event):
        if event_sink is not None:event_sink(deepcopy(event))
    for depth in (1,2,3):
        children=[]
        layer=dict(depth=depth,parents=[n['primitive_indices'] for n in frontier],scheduled_edges=16*len(frontier),
            attempted_edges=0,complete=False);result['layers'].append(layer)
        for parent in frontier:
            for i,primitive in enumerate(bank):
                require(result['planning_edge_attempts']<144,'planning reserve cannot consume verification')
                path=parent['primitive_indices']+[i]
                event=dict(kind='edge',depth=depth,parent_path=parent['primitive_indices'],primitive_index=i,
                    path=path,succeeded=False,replay_completed=False)
                result['events'].append(event);result['planning_edge_attempts']+=1;layer['attempted_edges']+=1
                emit(dict(kind='attempt',depth=depth,path=path))
                try:
                    child=old.scheduler._validated_context(replay(deepcopy(parent['context']),primitive.copy()),root)
                    event['replay_completed']=True
                    residual=(np.asarray(child['latent_history'][-1])-goal)@q.T
                    cost=float(np.mean(residual*residual));require(math.isfinite(cost) and cost>=0,'finite metric cost')
                    event.update(succeeded=True,cost=cost,context=deepcopy(child))
                    children.append(dict(context=child,actions=parent['actions']+primitive.tolist(),
                        primitive_indices=path,depth=depth,cost=cost))
                except Exception as exc:event.update(error_type=type(exc).__name__,error=str(exc))
                emit(event)
        layer['complete']=len(children)==layer['scheduled_edges']
        if not layer['complete']:
            result['failure']=dict(stage='layer',depth=depth,message='incomplete full bank; no reschedule or winner')
            return result
        children.sort(key=lambda n:(n['cost'],n['primitive_indices']))
        if depth==1:
            try:
                result['svd_calls']+=1
                result['signal']=jacobian_signal(p['inherited_jacobian'],p['inherited_nominal'],goal,q)
                result['allocation']=list(allocate(result['signal']['chi'],mode))
                emit(dict(kind='allocation',signal=result['signal'],allocation=result['allocation']))
            except Exception as exc:
                result.update(status='signal_unavailable',failure=dict(stage='signal',type=type(exc).__name__,message=str(exc)))
                return result
        if depth<3:frontier=children[:result['allocation'][depth-1]]
        else:result['best_plan']=deepcopy(children[0])
    require(result['planning_edge_attempts']==144,'exact 144 planning edges')
    result['status']='completed';return result


def run_arm(*,model,replay,p,binding,mode,event_sink=None):
    work=old.Work();hooks=[];search=None;failure=None;exact=False;method=NAME+'_'+mode
    with PredictionBudget(model,binding=binding,method=method,sample_limit=147) as budget:
        try:
            hooks.append(model.action_encoder.register_forward_pre_hook(work.action_entry))
            hooks.append(model.encoder.register_forward_pre_hook(work.visual_entry))
            with budget.phase('planning',method=method,horizon_tokens=3):
                search=run_search(p=p,replay=replay,mode=mode,event_sink=event_sink)
                require(search['status']=='completed','incomplete planning; no verification/substitute')
            require(budget.receipt()['attempted_predict_samples']==144,'exact B1 planning samples')
            with budget.phase('evaluation',method=method,horizon_tokens=3):
                actual=replay(deepcopy(p['root_context']),np.asarray(search['best_plan']['actions']))
                exact=io.canonical(actual)==io.canonical(search['best_plan']['context'])
                require(exact,'H3 whole-path verification differs')
        except Exception as exc:failure=dict(type=type(exc).__name__,message=str(exc))
        finally:
            for hook in hooks:hook.remove()
        receipt=budget.receipt()
    complete=(failure is None and exact and receipt['attempted_predict_samples']==147 and
        len(receipt['phases'])==2 and all(x['completed'] and x['failed_predict_calls']==0 for x in receipt['phases']) and
        [x['predict_samples'] for x in receipt['phases']]==[144,3] and receipt['blocked_calls']==0 and
        work.action_samples==work.action_calls==147 and work.context_tokens==441 and work.visual_samples==0)
    if not complete and failure is None:failure=dict(type='AccountingMismatch',message='actual resource coordinates differ')
    return dict(root_sha256=p['root_sha256'],mode=mode,status='completed' if complete else 'incomplete',
        input_sha256=p['input_sha256'],binding=receipt['binding'],search=search,prediction_receipt=receipt,
        whole_h3_replay_exact=exact,failure=failure,action_encoder_calls=work.action_calls,
        action_encoder_samples=work.action_samples,processed_action_context_tokens=work.context_tokens,
        visual_encoder_samples=work.visual_samples,runtime_action_backward_calls=0,runtime_action_vjp_rows=0,
        runtime_input_backward_calls=0,runtime_input_vjp_rows=0,new_jacobian_materializations=0,
        derivative_scope='new no-grad CudaReplay policy, not autograd instrumentation; inherited work separate',
        inherited_cost=inherited_cost(),total_pipeline_cost_equivalent=False,new_physical_steps=0,P3_admission=False)


def compare(rows,roots):
    pairs=old.compare(rows,roots)
    for pair in pairs:
        pair['same_action_jacobian_chi']=pair.pop('same_H1_chi')
        pair['total_pipeline_cost_equivalent']=False
    return pairs


def execute_cuda(*,prepared,prepare_sha256,output):
    """Explicit HOST CUDA only. No physics or automatic fallback/retry."""
    output=Path(output).absolute();require(not output.exists(),'exclusive NEW execution output required')
    body,identity=validate_prepared(prepared,prepare_sha256)
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.pusht_composition_cuda_replay_v2 import CudaReplay
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    import torch
    for p,h in body['model_bindings'].items():require(checked.digest(p)==h,'external model bytes changed')
    policy=Path(next(p for p in body['model_bindings'] if Path(p).name=='weights.pt')).parent
    validate_prepared(prepared,prepare_sha256);output.mkdir(exist_ok=False)
    write(output/'execution_protocol.json',dict(protocol_id=identity,prepare_sha256=prepare_sha256))
    rows=[];host=None;failure=None;before=after=None;started=perf_counter()
    try:
        host=require_cuda_allocation('cuda');device=torch.device('cuda:0')
        model=load_compatible_pretrained(str(policy),device=device).eval().requires_grad_(False)
        replay=CudaReplay(model,device);before=checked.tensor_state_hash(model)
        with torch.no_grad():
            for i,p in enumerate(body['packets']):
                binding=Binding(identity,'pusht_fixed_h3_cached_latent_calibration',p['root_sha256'],checked.OFFICIAL_SHA+':'+checked.CONFIG_SHA)
                for mode in body['arm_order'][i]:
                    path=output/(p['root_sha256']+'_'+mode+'.events.jsonl')
                    with path.open('x') as stream:
                        def journal(event):
                            stream.write(io.canonical(event)+'\n');stream.flush();os.fsync(stream.fileno())
                        torch.cuda.synchronize()
                        row=run_arm(model=model,replay=replay,p=p,binding=binding,mode=mode,event_sink=journal)
                        torch.cuda.synchronize()
                    row.update(journal_path=str(path),journal_sha256=checked.digest(path));rows.append(row)
                    write(output/(p['root_sha256']+'_'+mode+'.json'),row)
        after=checked.tensor_state_hash(model)
    except (Exception,KeyboardInterrupt) as exc:failure=dict(type=type(exc).__name__,message=str(exc))
    seen={(r['root_sha256'],r['mode']) for r in rows}
    for p in body['packets']:
        for mode in METHODS:
            if (p['root_sha256'],mode) not in seen:
                path=output/(p['root_sha256']+'_'+mode+'.events.jsonl')
                rows.append(dict(root_sha256=p['root_sha256'],mode=mode,status='blocked',input_sha256=p['input_sha256'],
                    binding=None,search=None,prediction_receipt=None,failure=failure or {'message':'interrupted before persistence'},
                    actual_predict_samples=None,unknown_attempted_work=True,journal_path=str(path) if path.exists() else None,
                    journal_sha256=checked.digest(path) if path.exists() else None))
    source_failure=None
    try:validate_prepared(prepared,prepare_sha256)
    except Exception as exc:source_failure=str(exc)
    pairs=compare(rows,body['roots'])
    integrity=(failure is None and source_failure is None and before is not None and before==after and
        all(p['classification']=='matched_full_H3' for p in pairs))
    report=dict(schema=NAME+'.report',status='calibration_recorded' if integrity else 'incomplete',rows=rows,pairs=pairs,
        protocol_id=identity,host=host,failure=failure,source_failure=source_failure,model_unchanged=before is not None and before==after,
        elapsed_seconds_including_setup=perf_counter()-started,calibration_roots=4,new_independent_roots=0,
        P1_admission=False,P3_admission=False,physical_quality_measured=False,total_flops_equality=False,
        total_pipeline_cost_equivalent=False,inherited_cost=inherited_cost(),inherited_arrays_charged_once=4,
        known_predict_samples=sum(r['prediction_receipt']['attempted_predict_samples'] for r in rows if r['prediction_receipt']),
        unknown_attempted_work=any(r.get('unknown_attempted_work',False) for r in rows),
        new_jacobian_materializations=0,new_physical_steps=0)
    write(output/'report.json',report)
    if not integrity:raise RuntimeError('incomplete calibration; retain receipts, no retry/padding')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true');action.add_argument('--execute-cuda',action='store_true')
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha256')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.prepare:prepare(args.project_root,args.output)
    else:
        require(args.prepared is not None and args.prepare_sha256 is not None,'reviewed prepare path/SHA required')
        execute_cuda(prepared=args.prepared,prepare_sha256=args.prepare_sha256,output=args.output)
