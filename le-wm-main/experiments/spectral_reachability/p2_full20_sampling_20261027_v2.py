"""Full20 raw-action CEM/random, known-root engineering calibration only.

Two independent time-major 5x2 packets. No training, physical selection, CPU
neural fallback, or P1/P2 admission. Prepare binds inputs before host CUDA.
"""
from contextlib import nullcontext, contextmanager
from functools import wraps
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from experiments.spectral_reachability.d2p_composition_witness_v2 import compose, digest
from experiments.spectral_reachability.d2p_kinodynamic_program_search import _cost, _validated_context
from experiments.spectral_reachability.p2_matched_composition_20261027 import _Ledger, RootAnchoredCudaReplay
from experiments.spectral_reachability.run_pusht_program_search_cuda import metric_for

TAG = 'p2_full20_sampling_20261027_v2'
DEFAULT_PAIRED_REPORT = (Path(__file__).resolve().parents[2] / 'results' /
    'pusht_p2_matched_composition4_v1_20261027_cuda/p2_matched_report.json')
ARMS = ('cem', 'random')
SEEDS = (20261027, 20261028, 20261029, 20261030)
SCHEDULE = dict(horizon_tokens=2, primitive_steps=10, candidates=40,
    planning_predict_calls=80, planning_predict_samples=80,
    replay_predict_calls=4, replay_predict_samples=4, batch_size=1,
    cem=dict(iterations=4, population=10, elite=2, initial_mean=0., initial_std=.5,
             std_floor=.05, ddof=0, mean_first=True, smoothing=0, clipped_to_raw_box=True),
    random=dict(explicit_zero_first=True, additional_uniform_raw_candidates=39),
    domain='raw float32 [-1,1]^20; two independent time-major 5x2 packets',
    proposal_dimensions=20, packet_shape=[2,5,2])


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, allow_nan=False, indent=2)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def raw_actions(commands):
    a = np.asarray(commands)
    if (a.shape != (2,5,2) or a.dtype.kind not in 'fi'
            or not np.isfinite(a).all() or np.any(np.abs(a)>1)):
        raise ValueError('finite two independent 5x2 raw packets inside [-1,1] required')
    return a.astype(np.float32).reshape(10,2).tolist()


@contextmanager
def actual_entries(model, replay):
    """Record actual predict/encoder entries, failures and candidate/packet scope.

    Wrap outside PredictionBudget: predict entries here are the delegated actual
    model calls, not blocked budget attempts. No tensor computation in wrapper.
    """
    entries, restored = [], []
    try:
        for owner, name, kind in ((model, 'predict', 'predict'),
                                  (model.action_encoder, 'forward', 'action_encoder')):
            original = getattr(owner, name)
            restored.append((owner,name,name in vars(owner),vars(owner).get(name)))
            def wrap(original, kind):
                @wraps(original)
                def counted(*args, **kw):
                    event = dict(kind=kind, samples=int(args[0].shape[0]),
                        context_tokens=int(args[0].shape[0]*args[0].shape[1]),
                        scope=deepcopy(getattr(replay, '_p2_scope', None)), completed=False)
                    entries.append(event)
                    try:
                        result = original(*args, **kw)
                        event['completed'] = True
                        return result
                    except BaseException as exc:
                        event.update(error_type=type(exc).__name__, error=str(exc))
                        raise
                return counted
            setattr(owner,name,wrap(original,kind))
        yield entries
    finally:
        for owner,name,existed,value in reversed(restored):
            if existed:
                setattr(owner,name,value)
            else:
                delattr(owner,name)


def entries_match(entries, ledger, arm=None):
    """Require both actual entry streams to match every packet dispatch."""
    for kind in ('predict', 'action_encoder'):
        events = [e for e in entries if e['kind']==kind]
        if len(events)!=len(ledger) or any(not e['completed'] for e in events):
            return False
        for event, dispatch in zip(events,ledger,strict=True):
            scope=event['scope']
            if (event['samples']!=1 or event['context_tokens']!=3 or not scope
                    or scope['phase']!=dispatch['phase']
                    or scope['path']!=dispatch['path']
                    or (arm is not None and scope['arm']!=arm)
                    or not dispatch['completed']
                    or dispatch['predict_calls']!=1 or dispatch['predict_samples']!=1):
                return False
    return True


def _arm(*, arm, seed, root, provenance, action_domain, target_metric, replay,
         fixture, phase=lambda name: nullcontext(), journal=lambda value: None):
    if arm not in ARMS or type(seed) is not int or seed < 0:
        raise ValueError('explicit supported arm and nonnegative integer seed required')
    root, provenance, action_domain = deepcopy((root, provenance, action_domain))
    initial = _validated_context(root['context'])
    def edge(inp, out, actions, identity):
        return dict(program_id=identity, kind='progress', root_id=root['root_id'],
            provenance=deepcopy(provenance), action_domain=deepcopy(action_domain),
            input_context=deepcopy(inp), output_context=deepcopy(out), actions=deepcopy(actions))
    # Validate complete metadata before callback dispatch, without model work.
    compose(root=root, provenance=provenance, action_domain=action_domain,
        edges=[edge(initial, initial, raw_actions(np.zeros((2,5,2))), 'check')],
        replay=lambda c,a:c, max_edges=1, max_actions=10)
    if (action_domain['name']!='raw' or action_domain['action_block']!=5 or
            action_domain['low']!=[-1.,-1.] or action_domain['high']!=[1.,1.]):
        raise ValueError('official raw PushT box and block required')
    rng = np.random.default_rng(seed)
    ledger = _Ledger(replay, arm)
    attempts, best = [], None
    mean, std = np.zeros((2,5,2)), np.full((2,5,2), .5)
    started = time.perf_counter()
    with phase('planning'):
        for generation in range(4):
            if arm == 'cem':
                population = np.clip(rng.normal(mean, std, size=(10,2,5,2)), -1, 1).astype(np.float32)
                population[0] = mean.astype(np.float32)
            else:
                population = rng.uniform(-1,1,size=(10,2,5,2)).astype(np.float32)
                if generation==0:
                    population[0] = 0
            finite = []
            for commands in population:
                event = dict(candidate_id=f'{arm}_{len(attempts):02d}', generation=generation,
                    commands=commands.tolist(), actions=raw_actions(commands), completed=False,
                    proposal_dimensions=20, packet_shape=[2,5,2])
                attempts.append(event)  # never replace failed candidate or refund work
                try:
                    terminal = ledger.call(initial, event['actions'], 'planning', [len(attempts)-1])
                    cost = _cost(target_metric(deepcopy(terminal)))
                    event.update(completed=True, cost=cost)
                    finite.append((cost, len(attempts)-1, commands.copy()))
                    if best is None or cost < best['cost']:
                        best = dict(candidate_id=event['candidate_id'], cost=cost,
                                    actions=deepcopy(event['actions']), terminal=deepcopy(terminal))
                except Exception as exc:
                    event.update(error_type=type(exc).__name__, error=str(exc))
                journal(deepcopy(event))
            if arm=='cem' and finite:
                finite.sort(key=lambda x:(x[0],x[1]))
                elite = np.stack([x[2] for x in finite[:2]]).astype(float)
                mean, std = elite.mean(axis=0), np.maximum(elite.std(axis=0,ddof=0), .05)
    witness, replay_error = None, None
    if best is not None:
        with phase('evaluation'):
            try:
                actions = best['actions']
                mid = ledger.call(initial, actions[:5], 'replay', [40])
                end = ledger.call(mid, actions[5:], 'replay', [41])
                full = ledger.call(initial, actions, 'replay', [42])
                if full!=end or full!=best['terminal']:
                    raise ValueError('independent segmented/full/search contexts differ')
                edges = [edge(initial, end, actions, best['candidate_id'])]
                receipt = compose(root=root, provenance=provenance, action_domain=action_domain,
                    edges=edges, replay=lambda c,a:deepcopy(full), max_edges=1,max_actions=10)
                witness = dict(receipt=receipt, edges=edges, independent_full_replay_exact=True)
            except Exception as exc:
                replay_error = dict(error_type=type(exc).__name__,error=str(exc))
    spent = {p:sum(e['predict_samples'] for e in ledger.events if e['phase']==p)
             for p in ('planning','replay')}
    okay = (len(attempts)==40 and all(e['completed'] for e in attempts) and
            all(e['completed'] for e in ledger.events) and spent=={'planning':80,'replay':4} and witness is not None)
    return dict(arm=arm, seed=seed, fixture_only=fixture, attempts=attempts,
        selected_candidate_id=best['candidate_id'] if best else None,
        selected_model_cost=best['cost'] if best else None, witness=witness,replay_error=replay_error,
        ledger=ledger.events, planning_predict_samples=spent['planning'], replay_predict_samples=spent['replay'],
        failed_candidates=sum(not e['completed'] for e in attempts), prediction_budget_matched=okay,
        status='prediction_matched_only' if okay else 'incomplete_or_failed',
        actual_wall_time_s=time.perf_counter()-started, fair_total_budget_comparison=False,
        P1_admission=False,P2_admission=False, physical_efficacy_verified=False)


def run_fixture_arm(**kwargs):
    """Non-neural callback fixtures only; never CUDA evidence."""
    return _arm(fixture=True, **kwargs)


def reference_inputs(paired_report, compilation, policy):
    """Read-only reference seal/bindings/root/action-domain verification."""
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.run_p2_matched_physical_20261027 import selected_actions as old_validate
    source, bindings, config = checked.validate_inputs(compilation,policy)
    parent = Path(paired_report).resolve().parent
    pp, ss = parent/'p2_matched_protocol.json', parent/'p2_matched_sources.json'
    report, protocol, snapshots = (json.loads(Path(p).read_text()) for p in (paired_report,pp,ss))
    if protocol['bindings']!=bindings or digest(snapshots)!=protocol['source_manifest_sha256']:
        raise ValueError('reference P2 input/source manifest differs')
    for item in snapshots:
        if (sha(item['path'])!=item['sha256'] or
                hashlib.sha256(item['source'].encode()).hexdigest()!=item['sha256']):
            raise ValueError('reference P2 current or archived source changed')
    old_validate(report, protocol, source)
    for p in (paired_report,pp,ss):
        bindings[str(Path(p).resolve())] = sha(p)
    return source,bindings,config,report


def prepare(paired_report, compilation, policy, output):
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal
    source,bindings,config,reference = reference_inputs(paired_report,compilation,policy)
    from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
    normalizer=dict(mean=p0.PUBLIC_ACTION_MEAN.tolist(),std=p0.PUBLIC_ACTION_STD.tolist())
    if len(source['rows'])!=4 or {r['root_index'] for r in source['rows']}!={0,1,2,3}:
        raise ValueError('original four indexed roots required')
    roots = {r['root_sha256']:r for r in reference['rows']}
    prepared = []
    for row in source['rows']:
        declaration = roots[row['root_sha256']]['declaration']
        metric_for(row['compiled'])
        initial=row['compiled']['input_context']
        expected_context=dict(layout='states_with_between_state_actions_v1',context_length=3,
            latent_dim=192,action_dim=10,latent_history=initial['emb'][0],
            action_history=initial['actions'][0][1:])
        if (declaration['root']['context']!=expected_context or
                declaration['action_domain']['normalization_sha256']!=digest(normalizer)):
            raise ValueError('source recurrence or raw-action normalization differs')
        prepared.append(dict(root_index=row['root_index'],root_sha256=row['root_sha256'],
            root=declaration['root'],action_domain=declaration['action_domain'],
            metric_sha256=row['compiled']['scoring']['metric']['sha256'], seed=SEEDS[row['root_index']]))
    protocol = dict(schema=TAG+'.protocol',bindings=bindings,prepared=prepared,schedule=SCHEDULE,
        paired_report=str(Path(paired_report).resolve()),compilation=str(Path(compilation).resolve()),
        policy=str(Path(policy).resolve()),arms=list(ARMS),
        selection='best finite ever, stable first tie; zero included INSIDE40, no extra mean evaluation',
        comparison='full20 continuous raw box vs v1 constant2 repeated5 (4 dims) and reference16-command bank; unequal spaces/priors',
        normalization=normalizer, objective='mean square Q@(terminal-goal_vectors[0])',
        physical_outcomes_used=False, training=False, readout=False,
        equal_prediction_budget_only=True, fair_total_budget_comparison=False,
        unmeasured=['FLOPs','upstream_encoding_and_bank_construction','equal_wall_time'],
        P1_admission=False,P2_admission=False, task_metric_repaired=False)
    files = checked.sources_before_allocate(source,config)
    import sys
    files += [Path(m.__file__) for n,m in tuple(sys.modules.items()) if n.startswith('experiments.')
              and getattr(m,'__file__','').endswith('.py')]
    files += [Path(__file__).with_name('pusht_full_packet_compiler_20261027.py'),
              Path(__file__).resolve(),Path(__file__).resolve().parents[2]/f'tests/test_{TAG}.py',
              Path(__file__).resolve().parents[3]/f'docs/{TAG}.md']
    files=sorted(set(p.resolve() for p in files))
    output=Path(output).resolve()
    if TAG not in output.name:
        raise ValueError('exclusive new tagged output directory required')
    output.mkdir(exist_ok=False)
    write(output/'protocol.json',protocol)
    seals=[create_seal(files=files[j:j+40],args_json=json.dumps(dict(protocol_sha256=digest(protocol))),
                      output_dir=output/f'source_seal_{j//40}') for j in range(0,len(files),40)]
    write(output/'prepared.json',dict(protocol_sha256=digest(protocol),source_seals=seals))
    print(f'prepared without CUDA: {output}',flush=True)


def load_prepared(output):
    from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal
    output=Path(output).resolve()
    protocol=json.loads((output/'protocol.json').read_text())
    prepared=json.loads((output/'prepared.json').read_text())
    if protocol['schema']!=TAG+'.protocol' or digest(protocol)!=prepared['protocol_sha256'] or protocol['schedule']!=SCHEDULE:
        raise ValueError('prepared protocol changed')
    for s in prepared['source_seals']:
        verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    source,bindings,_,reference=reference_inputs(protocol['paired_report'],protocol['compilation'],protocol['policy'])
    if bindings!=protocol['bindings']:
        raise ValueError('prepared inputs changed')
    return output,protocol,prepared,source


def cuda(output):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.d2p_adaptive_budget_v2 import PredictionBudget,Binding
    from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal
    output,protocol,prepared,source=load_prepared(output)
    write(output/'started.json',dict(status='cuda_requested',protocol_sha256=digest(protocol)))
    host=base.require_cuda_allocation('cuda')
    import torch
    device=torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32=False
    model=base.load_compatible_pretrained(Path(protocol['policy']),device=device).eval().requires_grad_(False)
    before=checked.tensor_state_hash(model)
    torch.cuda.reset_peak_memory_stats(device)
    rows=[]
    for saved,ready in zip(source['rows'],protocol['prepared'],strict=True):
        arms=[]
        for arm in ARMS:
            provenance=dict(roots_provenance(source_sha=prepared['source_seals'][0]['manifest_sha256'],
                protocol=protocol,ready=ready))
            binding=Binding(provenance['source_sha256'],TAG,ready['root_sha256'],checked.OFFICIAL_SHA)
            work=dict(calls=0,samples=0,processed_context_tokens=0)
            def hook(module,args):
                work['calls']+=1; work['samples']+=int(args[0].shape[0])
                work['processed_context_tokens']+=int(args[0].shape[0]*args[0].shape[1])
            handle=model.action_encoder.register_forward_pre_hook(hook)
            try:
                with (output/f'root_{ready["root_index"]}_{arm}_candidates.jsonl').open('x') as journal:
                    def save(event):
                        journal.write(json.dumps(event,allow_nan=False)+'\n');journal.flush()
                    replay=RootAnchoredCudaReplay(model,device)
                    with actual_entries(model,replay) as entries, PredictionBudget(model,binding=binding,method=arm,sample_limit=84) as budget:
                        result=_arm(arm=arm,seed=ready['seed'],root=ready['root'],provenance=provenance,
                            action_domain=ready['action_domain'],target_metric=metric_for(saved['compiled']),
                            replay=replay,fixture=False,
                            phase=lambda name:budget.phase(name,method=arm,horizon_tokens=2),journal=save)
                        actual=budget.receipt()
            finally:
                handle.remove()
            phases=actual['phases']
            actual_ok=(len(phases)==2 and all(p['completed'] for p in phases) and
                [(p['phase'],p['predict_calls'],p['predict_samples']) for p in phases]==
                [('planning',80,80),('evaluation',4,4)] and work['calls']==84 and work['samples']==84
                and entries_match(entries,result['ledger'],arm))
            result.update(actual_prediction_budget=actual,actual_model_entries=entries,action_encoder_work=work,
                          prediction_budget_matched=result['prediction_budget_matched'] and actual_ok)
            if not result['prediction_budget_matched']:
                result['status']='incomplete_or_failed'
            write(output/f'root_{ready["root_index"]}_{arm}.json',result)
            arms.append(result)
            print(f'root {ready["root_index"]} {arm}: {result["status"]}',flush=True)
        rows.append(dict(root_sha256=ready['root_sha256'],root_index=ready['root_index'],
                         root=ready['root'],action_domain=ready['action_domain'],rows=arms))
    gates=dict(all_eight_retained=len(rows)==4 and sum(len(r['rows']) for r in rows)==8,
        actual80_plus4_all=all(a['prediction_budget_matched'] for r in rows for a in r['rows']),
        model_unchanged=before==checked.tensor_state_hash(model),
        frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        inputs_unchanged=all(sha(p)==d for p,d in protocol['bindings'].items()))
    verified=[verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256']) for s in prepared['source_seals']]
    report=dict(schema=TAG+'.report',status='calibration_completed' if all(gates.values()) else 'integrity_fail',
        protocol_sha256=digest(protocol),rows=rows,gates=gates,host=host,
        source_seals=prepared['source_seals'],source_verification=verified,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        P1_admission=False,P2_admission=False,fair_total_budget_comparison=False,physical_efficacy_verified=False,
        actual_predict_calls=sum(e['kind']=='predict' for r in rows for a in r['rows'] for e in a['actual_model_entries']),
        actual_predict_samples=sum(e['samples'] for r in rows for a in r['rows'] for e in a['actual_model_entries'] if e['kind']=='predict'))
    write(output/'report.json',report)
    if not all(gates.values()):
        raise RuntimeError('sampling calibration failed; retain all attempts, do not execute physics')


def roots_provenance(*, source_sha, protocol, ready):
    from experiments.spectral_reachability.run_pusht_d2p_budget_guard_cuda import OFFICIAL_SHA
    return dict(checkpoint_sha256=OFFICIAL_SHA,encoder_sha256=OFFICIAL_SHA,source_sha256=source_sha,
        protocol_sha256=digest(protocol),metric_sha256=ready['metric_sha256'],
        candidate_bank_sha256=digest(dict(schedule=SCHEDULE,seed=ready['seed'])))


def selected_actions(report, protocol, compilation):
    """Read-only consumer for NEW physical driver: {(root_sha, arm): raw[10,2]}.

    Caller MUST independently verify report/protocol files and all source seals
    using load_prepared, then bind report hash before physical execution.
    Old physical driver has a fixed arm whitelist and cannot consume this schema.
    """
    if (report.get('schema')!=TAG+'.report' or report.get('status')!='calibration_completed' or
            report.get('protocol_sha256')!=digest(protocol) or not report.get('gates') or
            not all(v is True for v in report['gates'].values())):
        raise ValueError('complete protocol-bound sampling report required')
    source={r['root_sha256']:r['compiled'] for r in compilation['rows']}
    rows=report['rows']; ready={r['root_sha256']:r for r in protocol['prepared']}
    if len(rows)!=4 or len(source)!=4 or len({r['root_sha256'] for r in rows})!=4 or set(source)!=set(ready) or {r['root_sha256'] for r in rows}!=set(source):
        raise ValueError('all four unique roots required')
    actions={}
    for row in rows:
        key=row['root_sha256']; root=ready[key]['root']; compiled=source[key]
        initial=compiled['input_context']
        context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
                     latent_history=initial['emb'][0],action_history=initial['actions'][0][1:])
        if root['root_id']!=key or root['context']!=context or row['root']!=root or row['action_domain']!=ready[key]['action_domain']:
            raise ValueError('root context/domain mismatch')
        if len(row['rows'])!=2 or {a['arm'] for a in row['rows']}!=set(ARMS):
            raise ValueError('both unique sampling arms required')
        for arm in row['rows']:
            w=arm['witness']
            if (arm['fixture_only'] is not False or arm['prediction_budget_matched'] is not True or
                    arm['planning_predict_samples']!=80 or arm['replay_predict_samples']!=4 or
                    arm['replay_error'] is not None or w is None or not w['independent_full_replay_exact']):
                raise ValueError('actual complete80+4 independent witness required')
            receipt=w['receipt']; body={k:v for k,v in receipt.items() if k!='receipt_sha256'}
            if digest(body)!=receipt['receipt_sha256'] or receipt['root']!=root or receipt['action_domain']!=row['action_domain']:
                raise ValueError('receipt digest/root/domain mismatch')
            expected=roots_provenance(source_sha=report['source_seals'][0]['manifest_sha256'],protocol=protocol,ready=ready[key])
            if receipt['provenance']!=expected:
                raise ValueError('selected provenance differs')
            rebuilt=compose(root=root,provenance=expected,action_domain=row['action_domain'],edges=w['edges'],
                replay=lambda c,a:deepcopy(receipt['terminal_context']),max_edges=1,max_actions=10)
            if rebuilt!=receipt:
                raise ValueError('witness edge differs')
            if (not entries_match(arm.get('actual_model_entries',[]),arm['ledger'],arm['arm'])
                    or len(arm['ledger'])!=84
                    or [e['phase'] for e in arm['ledger']]!=['planning']*80+['replay']*4):
                raise ValueError('actual model entries differ from packet ledger')
            candidates=arm['attempts']
            if len(candidates)!=40 or not all(c['completed'] for c in candidates):
                raise ValueError('all40 candidates required')
            if [c['candidate_id'] for c in candidates]!=[f'{arm["arm"]}_{i:02d}' for i in range(40)]:
                raise ValueError('ordered unique forty candidates required')
            for candidate in candidates:
                if raw_actions(candidate['commands'])!=candidate['actions']:
                    raise ValueError('candidate full20 raw packet differs')
            best=min(candidates,key=lambda c:_cost(c['cost']))
            raw=raw_actions(best['commands'])
            if (raw!=best['actions'] or raw!=receipt['actions'] or best['candidate_id']!=arm['selected_candidate_id'] or
                    best['cost']!=arm['selected_model_cost'] or len(raw)!=10 or
                    abs(_cost(arm['selected_model_cost'])-metric_for(compiled)(receipt['terminal_context']))>1e-12):
                raise ValueError('selected actions/cost differs from40-candidate incumbent')
            actions[key,arm['arm']]=np.asarray(raw)
    return actions


def plans_from_saved(report, protocol, compilation):
    """Export all eight plans for durable.physical_branch, not a physics launch.

    Caller must first load_prepared, compare report source_seals with prepared,
    bind the report hash, and verify actual root/goal RGB before execution.
    The old sampling physical CLI imports v1 and is NOT compatible unchanged.
    """
    selected = selected_actions(report, protocol, compilation)
    ready = {r['root_sha256']:r for r in protocol['prepared']}
    plans = []
    for row in compilation['rows']:
        key, index = row['root_sha256'], row['root_index']
        peer = next(r for r in report['rows'] if r['root_sha256']==key)
        for arm in (ARMS if index%2==0 else ARMS[::-1]):
            result = next(a for a in peer['rows'] if a['arm']==arm)
            plan = dict(root_index=index, root_sha256=key, method=arm,
                commands=selected[key,arm].tolist(), planned_horizon_tokens=2,
                planned_commands=10, corrected_search_horizon=2,
                root=deepcopy(ready[key]['root']),
                action_domain=deepcopy(ready[key]['action_domain']),
                selected_candidate_id=result['selected_candidate_id'],
                receipt_sha256=result['witness']['receipt']['receipt_sha256'])
            plan['selection_sha256']=digest(plan)
            plans.append(plan)
    return plans


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='phase',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--paired-report',type=Path,default=DEFAULT_PAIRED_REPORT)
    for name in ('compilation','policy','output'):
        prep.add_argument('--'+name,type=Path,required=True)
    sub.add_parser('cuda').add_argument('--output',type=Path,required=True)
    args=vars(p.parse_args()); command=args.pop('phase')
    (prepare if command=='prepare' else cuda)(**args)

