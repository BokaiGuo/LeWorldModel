"""Fresh finite-H2/H3 Jacobian scheduling, predeclared under authorized implementation.

Cached-input calibration only. No inherited derivative used as a fresh derivative.
Only --execute-cuda may load a model; prepare/verify are source-binding operations.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
import math
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from experiments.spectral_reachability import p3_action_jacobian_budget_schedule_20261027_v3 as prior
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

NAME = 'p3_horizon_residual_budget_20261027_v4'
PREVIOUS = 'results/p3_action_jacobian_budget_schedule_20261027_v3_prepared'
PREVIOUS_SHA = '27988bdb563563bd2cb0ea5ef3f69851627a2cae4d648e1e545f2e68c7d5c134'
METHODS = ('uniform', 'dynamic')
io, require, write = prior.io, prior.require, prior.write


def digest(value):
    return io.sha(io.canonical(value).encode())


def contract():
    return dict(parameter_authority='implementation-thread predeclared under authorized implementation',
        horizons=[2, 3], coordinates='joint time-major 10H logits; raw=tanh(logits); no intermediate detach',
        jacobians=['192x20 at zero', '192x30 at zero'], metric='mean((Q@(terminal-goal0))**2)',
        signal='chiH=||r-Proj_col(QJH)r||^2/||r||^2; r=Q@(goal0-zero_nominalH)',
        cutoff=1e-6, degeneracy='zero residual: chi=0; zero rank: chi=1 unless zero residual',
        dynamic='gain=max(chi2-chi3,0)/max(chi2,1e-12); blocks3=floor(10*(.25+.5*gain)+.5)',
        uniform='5 H2 blocks +5 H3 blocks', block_predicts=6, search_predicts_per_arm=60,
        candidates_per_block={'2':3, '3':2}, candidate_pool='zero then PCG64 uniform joint-logit ball',
        seed='20261027+100*root_index+horizon', radius='sqrt(5H)*.75, joint not per-packet',
        selection='minimum original Q terminal cost, then H, then candidate index; variable H2/H3',
        shared_predict_cap=10, shared_detail='H2 nominal2+ADforward2; H3 nominal3+ADforward3',
        arm_exclusive_predict_cap=66, verification='whole winner plus independently segmented replay, 2H<=6',
        pair_predict_cap=142, pair_action_encoder_cap=142, pair_neural_entry_cap=284,
        four_root_predict_cap=568, four_root_neural_entry_cap=1136,
        jacobians_per_root=2, svds_per_root=2, mathematical_vjp_directions_per_root=384,
        autograd_jacobian_api_calls_per_root=2, backward_engine_api_count='not measured; not VJP directions',
        action_context_tokens_per_predict=3, visual_encoder_samples=0,
        cached_input_history_cost='inherited unknown; excluded, not zero; no total-pipeline equivalence',
        no_padding_no_refunds=True, total_flops_equivalent=False, P1_admission=False, P3_admission=False)


def residual_signal(jacobian, nominal, goal, projection, horizon):
    require(horizon in (2, 3), 'H2/H3 only')
    j=prior.array(jacobian,(192,10*horizon)); n=prior.array(nominal,(192,))
    g=prior.array(goal,(192,)); q=prior.array(projection,(192,192))
    require(np.allclose(q,q.T,atol=1e-8,rtol=0) and np.allclose(q@q,q,atol=1e-8,rtol=0),'original projection')
    r=q@(g-n); u,s,_=np.linalg.svd(q@j,full_matrices=False)
    rank=int(np.sum(s>s[0]*1e-6)) if s[0]>0 else 0
    norm=float(r@r)
    chi=0. if norm==0 else float(np.clip(np.sum((r-u[:,:rank]@(u[:,:rank].T@r))**2)/norm,0,1))
    return dict(horizon=horizon,chi=chi,rank=rank,residual_squared_norm=norm,singular_values=s.tolist())


def allocate(mode, signals):
    require(mode in METHODS and set(signals)=={'2','3'},'explicit modes and both finite horizons')
    c2,c3=(signals[str(h)]['chi'] for h in (2,3))
    require(all(math.isfinite(c) and 0<=c<=1 for c in (c2,c3)),'bounded finite chi')
    gain=max(c2-c3,0)/max(c2,1e-12)
    b3=5 if mode=='uniform' else math.floor(10*(.25+.5*gain)+.5)
    require(1<=b3<=9,'both horizon groups retained')
    return dict(blocks={'2':10-b3,'3':b3}, candidates={'2':3*(10-b3),'3':2*b3},
                search_predicts=60,gain=gain)


def pool(root_index,horizon,count):
    require(type(root_index) is int and 0<=root_index<4 and horizon in (2,3) and type(count) is int and count>=1,'fixed pool domain')
    rng=np.random.Generator(np.random.PCG64(20261027+100*root_index+horizon))
    values=[np.zeros(10*horizon)]
    for _ in range(count-1):
        direction=rng.normal(size=10*horizon)
        values.append(direction/np.linalg.norm(direction)*rng.random()**(1/(10*horizon))*np.sqrt(5*horizon)*.75)
    return [np.tanh(u).reshape(5*horizon,2) for u in values]


class NeuralCap:
    """Pre-dispatch, nonrefundable, root-wide guard on BOTH neural entry types.

    Durable journal includes attempted/failed calls, even incomplete AD forward.
    Backward operations are not disguised as predict or encoder entries.
    """
    def __init__(self,model,journal):
        self.model,self.journal=model,journal
        self.role=None;self.events=[];self.blocked=0

    def dispatch(self,kind,callback,args,kwargs):
        require(self.role in ('shared',*METHODS),'unscoped neural entry forbidden')
        x=args[0]
        require(getattr(getattr(x,'device',None),'type',None)=='cuda' and tuple(x.shape)==((1,3,192) if kind=='predict' else (1,3,10)), 'single full-context CUDA input required')
        counts=lambda role:sum(e['kind']==kind and (role is None or e['role']==role) for e in self.events)
        limit=10 if self.role=='shared' else 66
        if counts(None)>=142 or counts(self.role)>=limit:
            self.blocked+=1
            self.journal(dict(kind='blocked',entry=kind,role=self.role))
            raise ValueError('nonrefundable neural cap exceeded before dispatch')
        event=dict(kind=kind,role=self.role,context_tokens=3,completed=False,index=len(self.events))
        self.events.append(event);self.journal(deepcopy(event))
        try:
            result=callback(*args,**kwargs);event['completed']=True
            return result
        except BaseException as exc:
            event['error']=type(exc).__name__+': '+str(exc);raise
        finally:self.journal(deepcopy(event))

    @contextmanager
    def scope(self,role):
        require(self.role is None,'nested neural scopes forbidden')
        self.role=role
        try:yield
        finally:self.role=None

    def __enter__(self):
        self.original=(self.model.predict,self.model.action_encoder.forward,self.model.encoder.forward)
        self.model.predict=lambda *a,**k:self.dispatch('predict',self.original[0],a,k)
        self.model.action_encoder.forward=lambda *a,**k:self.dispatch('action_encoder',self.original[1],a,k)
        def forbidden(*a,**k):
            self.journal(dict(kind='forbidden_visual_encoder'));raise ValueError('cached-input protocol forbids visual encoder')
        self.model.encoder.forward=forbidden
        return self

    def __exit__(self,*exc):
        self.model.predict,self.model.action_encoder.forward,self.model.encoder.forward=self.original

    def receipt(self):
        return dict(events=deepcopy(self.events),blocked=self.blocked,
            attempted={k:sum(e['kind']==k for e in self.events) for k in ('predict','action_encoder')},
            completed={k:sum(e['kind']==k and e['completed'] for e in self.events) for k in ('predict','action_encoder')})


def compile_cuda(model,device,p,journal):
    from experiments.spectral_reachability import pusht_full_packet_compiler_20261027 as compiler
    import torch
    transition=compiler.make_transition(model,device)
    emb=torch.tensor(p['root_context']['latent_history'],device=device,dtype=torch.float32)[None]
    history=torch.tensor(p['root_context']['action_history'],device=device,dtype=torch.float32)[None]
    records={}
    for h in (2,3):
        def terminal(u):
            state,between=emb,history
            for raw in torch.tanh(u).reshape(h,5,2):state,between=transition(state,between,raw)
            return state[0,-1]
        zero=emb.new_zeros(10*h)
        with torch.no_grad():nominal=terminal(zero)
        request=dict(kind='jacobian',horizon=h,shape=[192,10*h],vjp_directions=192,completed=False)
        journal(deepcopy(request))
        jac=torch.autograd.functional.jacobian(terminal,zero.requires_grad_(True),vectorize=True)
        request['completed']=True;journal(deepcopy(request))
        n=nominal.detach().cpu().numpy().astype(float);j=jac.detach().cpu().numpy().astype(float)
        event=dict(kind='svd',horizon=h,completed=False);journal(deepcopy(event))
        signal=residual_signal(j,n,p['goal'],p['projection'],h)
        event['completed']=True;journal(deepcopy(event))
        records[str(h)]=dict(signal=signal,jacobian=j.tolist(),nominal=n.tolist())
    return records


def search(p,root_index,mode,signals,replay,journal):
    allocation=allocate(mode,signals);receipts=[]
    q=np.asarray(p['projection']);goal=np.asarray(p['goal'])
    for h in (2,3):
        for index,actions in enumerate(pool(root_index,h,allocation['candidates'][str(h)])):
            attempt=dict(kind='candidate',horizon=h,index=index,actions=actions.tolist(),completed=False)
            journal(deepcopy(attempt))
            context=replay(deepcopy(p['root_context']),actions)
            terminal=prior.array(context['latent_history'],(3,192))[-1]
            cost=float(np.mean((q@(terminal-goal))**2));require(math.isfinite(cost),'finite nonlinear Q')
            attempt.update(completed=True,context=context,cost=cost)
            journal(deepcopy(attempt));receipts.append(attempt)
    winner=deepcopy(min(receipts,key=lambda x:(x['cost'],x['horizon'],x['index'])))
    actions=np.asarray(winner['actions']);whole=replay(deepcopy(p['root_context']),actions)
    segmented=deepcopy(p['root_context'])
    for packet in actions.reshape(-1,5,2):segmented=replay(segmented,packet)
    require(io.canonical(whole)==io.canonical(segmented)==io.canonical(winner['context']),'exact full-context double replay witness')
    return dict(allocation=allocation,candidates=receipts,winner=winner,whole_replay_context=whole,
                segmented_replay_context=segmented,whole_and_segmented_exact=True,
                verification_predicts=2*winner['horizon'],status='completed',
                cpu_operations=dict(allocation_calls=1,candidate_Q_evaluations=len(receipts),
                    winner_key_comparisons=len(receipts)-1,proposal_pool_calls=2,
                    random_proposals=sum(allocation['candidates'].values())-2))


def accounting_ok(record,rows):
    receipt=record['neural_receipt'];events=receipt['events']
    if receipt['blocked'] or any(not e['completed'] for e in events):return False
    if set(record['compilation'])!={'2','3'}:return False
    arms={r['mode']:r for r in rows if r['root_sha256']==record['root_sha256']}
    if set(arms)!=set(METHODS) or any(r['status']!='completed' for r in arms.values()):return False
    expected={'shared':10,**{mode:60+arms[mode]['search']['verification_predicts'] for mode in METHODS}}
    return all(sum(e['kind']==kind and e['role']==role for e in events)==n
        for kind in ('predict','action_encoder') for role,n in expected.items()) and all(e['context_tokens']==3 for e in events)


def prepare(root,output):
    root=Path(root).resolve();output=Path(output).absolute()
    require(not output.exists(),'exclusive NEW prepare path required')
    previous,_=prior.validate_prepared(root/PREVIOUS,PREVIOUS_SHA)
    # Project only source context/goal/Q. Never copy inherited derivatives or outcomes.
    packets=[]
    for p in previous['packets']:
        new={k:deepcopy(p[k]) for k in ('root_sha256','root_context','goal','projection')}
        new['input_sha256']=digest(new);packets.append(new)
    sources=prior.physical.source_closure(root,[Path(__file__).resolve(),root/'tests'/('test_'+NAME+'.py'),root/'innerdocs'/(NAME+'.md')])
    hashes={str(p):io.sha(p.read_bytes()) for p in sources}
    body=dict(schema=NAME+'.protocol',contract=contract(),roots=previous['roots'],packets=packets,
        model_bindings=previous['model_bindings'],sources_sha256=hashes,
        prior_prepared=str(root/PREVIOUS),prior_prepare_sha256=PREVIOUS_SHA,
        arm_order=[list(METHODS if i%2==0 else METHODS[::-1]) for i in range(4)],
        P1_admission=False,P3_admission=False,new_independent_roots=0)
    identity=digest(body);output.mkdir(exist_ok=False);write(output/'protocol.json',dict(protocol=body,protocol_id=identity))
    files=sorted(sources);seals=[create_seal(files=files[i:i+40],args_json=io.canonical({'protocol_id':identity}),output_dir=output/f'source_seal_{i//40}') for i in range(0,len(files),40)]
    receipt=dict(status='prepared_not_executed',protocol_id=identity,protocol_sha256=io.sha((output/'protocol.json').read_bytes()),source_seals=seals,
                 model_loaded=False,new_predict_samples=0,new_physical_steps=0,P1_admission=False,P3_admission=False)
    write(output/'prepare.json',receipt)
    validate_prepared(output,io.sha((output/'prepare.json').read_bytes()))
    return receipt


def validate_prepared(prepared,expected_sha):
    prepared=Path(prepared).absolute();raw=(prepared/'prepare.json').read_bytes()
    require(io.sha(raw)==expected_sha,'prepare SHA mismatch');receipt=io.decode(raw)
    raw=(prepared/'protocol.json').read_bytes();require(io.sha(raw)==receipt['protocol_sha256'],'protocol byte mismatch')
    envelope=io.decode(raw);body=envelope['protocol'];identity=envelope['protocol_id']
    require(digest(body)==identity==receipt['protocol_id'] and body['schema']==NAME+'.protocol','protocol identity')
    require(body['contract']==contract(),'frozen contract drift')
    previous,_=prior.validate_prepared(body['prior_prepared'],body['prior_prepare_sha256'])
    require(body['roots']==previous['roots'] and len(set(body['roots']))==4,'fixed roots')
    require(body['model_bindings']==previous['model_bindings'],'original frozen model')
    require(body['arm_order']==[list(METHODS if i%2==0 else METHODS[::-1]) for i in range(4)],'fixed arm ordering')
    for p,old in zip(body['packets'],previous['packets'],strict=True):
        expected={k:old[k] for k in ('root_sha256','root_context','goal','projection')}
        require(p==dict(expected,input_sha256=digest(expected)),'source input binding')
    for p,h in body['sources_sha256'].items():require(io.sha(Path(p).read_bytes())==h,'source drift: '+p)
    for seal in receipt['source_seals']:verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    return body,identity


def execute_cuda(prepared,expected_sha,output):
    body,identity=validate_prepared(prepared,expected_sha);output=Path(output).absolute()
    require(not output.exists(),'exclusive NEW execution path required')
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.pusht_composition_cuda_replay_v2 import CudaReplay
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation
    import torch
    for p,h in body['model_bindings'].items():require(checked.digest(p)==h,'model binding drift')
    policy=Path(next(p for p in body['model_bindings'] if Path(p).name=='weights.pt')).parent
    output.mkdir(exist_ok=False);write(output/'execution_protocol.json',dict(protocol_id=identity,prepared=str(Path(prepared).absolute()),prepare_sha256=expected_sha))
    rows=[];roots=[];failure=None;before=after=None;host=None;started=perf_counter()
    try:
        host=require_cuda_allocation('cuda');device=torch.device('cuda:0')
        model=load_compatible_pretrained(str(policy),device=device).eval().requires_grad_(False)
        before=checked.tensor_state_hash(model);replay=CudaReplay(model,device)
        for i,p in enumerate(body['packets']):
            record=dict(root_sha256=p['root_sha256'],compilation=None,status='incomplete');roots.append(record)
            with (output/f'root_{i}.events.jsonl').open('x') as stream:
                def journal(event):
                    stream.write(io.canonical(event)+'\n');stream.flush();os.fsync(stream.fileno())
                cap=NeuralCap(model,journal)
                try:
                    with cap:
                        with cap.scope('shared'):record['compilation']=compile_cuda(model,device,p,journal)
                        signals={h:c['signal'] for h,c in record['compilation'].items()}
                        for mode in body['arm_order'][i]:
                            row=dict(root_sha256=p['root_sha256'],input_sha256=p['input_sha256'],mode=mode,status='incomplete');rows.append(row)
                            with cap.scope(mode):row['search']=search(p,i,mode,signals,replay,journal)
                            row['status']='completed';write(output/f'root_{i}_{mode}.json',row)
                    torch.cuda.synchronize();record['status']='completed'
                finally:
                    record['neural_receipt']=cap.receipt()
                    record['journal_path']=str(output/f'root_{i}.events.jsonl')
            record['journal_sha256']=checked.digest(record['journal_path'])
            require(accounting_ok(record,rows),'actual scoped predict/encoder/ADforward count mismatch')
            record['accounting_exact']=True
            write(output/f'root_{i}.json',record)
        after=checked.tensor_state_hash(model)
    except (Exception,KeyboardInterrupt) as exc:failure=dict(type=type(exc).__name__,message=str(exc))
    # Even an interrupted compile retains fsync journal and attempted neural entries.
    for i,record in enumerate(roots):
        path=output/f'root_{i}.events.jsonl'
        if path.exists():record.update(journal_path=str(path),journal_sha256=checked.digest(path))
        if not (output/f'root_{i}.json').exists():write(output/f'root_{i}.json',record)
    for row in rows:
        if row['status']!='completed':row['failure']=failure
    for p in body['packets']:
        for mode in METHODS:
            if not any(r['root_sha256']==p['root_sha256'] and r['mode']==mode for r in rows):
                rows.append(dict(root_sha256=p['root_sha256'],mode=mode,status='blocked',failure=failure))
    try:validate_prepared(prepared,expected_sha)
    except Exception as exc:failure=dict(type='SourceDrift',message=str(exc))
    complete=failure is None and before is not None and before==after and len(rows)==8 and all(r['status']=='completed' for r in rows)
    report=dict(schema=NAME+'.report',status='cuda_completed' if complete else 'incomplete',protocol_id=identity,rows=rows,roots=roots,
        failure=failure,host=host,model_unchanged=before is not None and before==after,elapsed_seconds=perf_counter()-started,
        contract=contract(),P1_admission=False,P3_admission=False,new_physical_steps=0,physical_quality_measured=False)
    report['resources']=dict(attempted_predict_samples=sum(r['neural_receipt']['attempted']['predict'] for r in roots if 'neural_receipt' in r),
        attempted_action_encoder_samples=sum(r['neural_receipt']['attempted']['action_encoder'] for r in roots if 'neural_receipt' in r),
        completed_jacobians=sum(len(r['compilation'] or {}) for r in roots),
        completed_compilation_cost_lower_bound_only=not complete,
        completed_svds=sum(len(r['compilation'] or {}) for r in roots),
        completed_mathematical_vjp_directions=192*sum(len(r['compilation'] or {}) for r in roots),
        incomplete_operation_authority='durable per-root journal; no missing operation charged as zero',
        backward_engine_api_calls=None,
        actual_complete_cpu_search_operations=[dict(root_sha256=r['root_sha256'],mode=r['mode'],**r['search']['cpu_operations'])
            for r in rows if r['status']=='completed'],
        unknown_attempted_work=any('neural_receipt' not in r for r in roots))
    write(output/'report.json',report)
    if not complete:raise RuntimeError('incomplete; preserve partials, no automatic retry')
    return report


def selected_actions(output):
    """Model-free source/input/report/witness verification for future consumers.

    Recomputes eight saved-array SVD signals, not neural calls. Full mixed-horizon
    candidates are reconstructed; partial rows cannot be exported as winners.
    """
    output=Path(output).absolute();binding=io.decode((output/'execution_protocol.json').read_bytes())
    body,identity=validate_prepared(binding['prepared'],binding['prepare_sha256'])
    report_raw=(output/'report.json').read_bytes();report=io.decode(report_raw)
    require(binding['protocol_id']==identity==report['protocol_id'] and report['schema']==NAME+'.report','execution source identity')
    require(report['status']=='cuda_completed' and report['model_unchanged'] is True and report['failure'] is None,'complete unchanged-model execution required')
    require(report['contract']==contract() and report['P1_admission'] is False and report['P3_admission'] is False,'scope and contract')
    roots=io.index(report['roots'],('root_sha256',));rows=io.index(report['rows'],('root_sha256','mode'))
    require(set(roots)=={(r,) for r in body['roots']} and set(rows)=={(r,m) for r in body['roots'] for m in METHODS},'strict four-root eight-arm identity census')
    plans=[]
    for i,p in enumerate(body['packets']):
        record=roots[p['root_sha256'],]
        require(record['accounting_exact'] is True and accounting_ok(record,report['rows']),'actual accounting')
        require(io.sha(Path(record['journal_path']).read_bytes())==record['journal_sha256'],'durable journal binding')
        journal=[io.decode(line.encode()) for line in Path(record['journal_path']).read_text().splitlines()]
        actual=[e for e in journal if e['kind'] in ('predict','action_encoder')]
        indexed={e['index']:e for e in actual}
        require(len(actual)==2*len(record['neural_receipt']['events']) and
            [indexed[k] for k in sorted(indexed)]==record['neural_receipt']['events'],'actual entry journal/receipt binding')
        operations=[e for e in journal if e['kind'] in ('jacobian','svd') and e['completed']]
        require([(e['kind'],e['horizon']) for e in operations]==[('jacobian',2),('svd',2),('jacobian',3),('svd',3)],'completed AD/SVD operations')
        signals={}
        for h in (2,3):
            c=record['compilation'][str(h)]
            signal=residual_signal(c['jacobian'],c['nominal'],p['goal'],p['projection'],h)
            require(signal==c['signal'],'finite-horizon saved-array residual witness')
            signals[str(h)]=signal
        for mode in METHODS:
            row=rows[p['root_sha256'],mode];s=row['search']
            require(row['input_sha256']==p['input_sha256'] and s['status']=='completed' and s['whole_and_segmented_exact'] is True,'complete bound search')
            allocation=allocate(mode,signals);require(s['allocation']==allocation,'residual-driven allocation')
            expected=[]
            for h in (2,3):
                expected.extend((h,k,a) for k,a in enumerate(pool(i,h,allocation['candidates'][str(h)])))
            require(len(s['candidates'])==len(expected),'full candidate census')
            for c,(h,k,a) in zip(s['candidates'],expected,strict=True):
                require(c['horizon']==h and c['index']==k and c['completed'] is True and np.array_equal(c['actions'],a),'candidate action identity')
                terminal=prior.array(c['context']['latent_history'],(3,192))[-1]
                cost=float(np.mean((np.asarray(p['projection'])@(terminal-np.asarray(p['goal'])))**2))
                require(c['cost']==cost,'original Q witness')
            winner=min(s['candidates'],key=lambda c:(c['cost'],c['horizon'],c['index']))
            require(winner==s['winner'] and s['verification_predicts']==2*winner['horizon'] and
                s['whole_replay_context']==s['segmented_replay_context']==winner['context'],'winner and exact full-context witness identity')
            plan=dict(root_sha256=p['root_sha256'],root_index=i,mode=mode,horizon=winner['horizon'],
                actions=winner['actions'],selected_predicted_context=winner['context'],predicted_Q_cost=winner['cost'],
                root_context=deepcopy(p['root_context']),goal=p['goal'],projection=p['projection'],
                input_sha256=p['input_sha256'],protocol_id=identity,report_sha256=io.sha(report_raw),
                P1_admission=False,P3_admission=False,export_numpy_svd_calls_shared_total=8,new_predict_samples=0)
            plan['selection_sha256']=digest(plan);plans.append(plan)
    return plans


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','verify','execute-cuda'):action.add_argument('--'+name,action='store_true')
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha256');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.prepare:
        require(args.output is not None,'new output required');result=prepare(args.project_root,args.output)
    else:
        require(args.prepared is not None and args.prepare_sha256 is not None,'reviewed receipt path/hash required')
        if args.verify:
            _,identity=validate_prepared(args.prepared,args.prepare_sha256);result=dict(status='verified',protocol_id=identity)
        else:
            require(args.output is not None,'new execution output required');result=execute_cuda(args.prepared,args.prepare_sha256,args.output)
    print(io.canonical(result))


if __name__=='__main__':main()
