"""Full20 MPPI-inspired shooting, exact reuse of the existing NumPy kernel.

Not standard/official MPPI: clipped Gaussian proposals, no importance-density
correction or automatic control cross-term. 40 H2 candidates = 80 planning
packet predictions, plus four independent replay predictions. No physics.
Prepare is model-free; CUDA is only an explicit, main-thread reviewed action.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np

from experiments.spectral_reachability import d2p_sampling_baselines as kernels
from experiments.spectral_reachability import p2_full20_sampling_20261027_v2 as full20
from experiments.spectral_reachability.d2p_composition_witness_v2 import compose, digest
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG = 'p2_full20_mppi_20261027_v1'
ARM = 'mppi_inspired_shooting'
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = PROJECT/'results/p2_full20_sampling_20261027_v2_cuda_v1'
SEEDS = (20261027, 20261028, 20261029, 20261030)
SCHEDULE = dict(horizon_tokens=2, primitive_steps=10, candidates=40,
    planning_predict_calls=80, planning_predict_samples=80,
    replay_predict_calls=4, replay_predict_samples=4, batch_size=1,
    proposal_dimensions=20, packet_shape=[2,5,2], iterations=3, samples=12,
    temperature=1., noise_scale=.5, initial_nominal=0.,
    candidate_accounting='1 initial + 3*(12 clipped Gaussian + 1 updated nominal) = 40 H2',
    domain='raw float32 [-1,1]^20; two independent time-major 5x2 packets',
    kernel_parameters='float64; cost evaluates the exact float32 raw actions',
    selection='best ever including initial/proposals/nominals; first stable tie',
    importance_correction='none', automatic_control_cross_term=False,
    official_reproduction=False)
CLAIM = ('MPPI-inspired shooting: exact existing clipped_gaussian_mppi_heuristic kernel; '
         'not standard/official MPPI, no importance-density correction, no automatic '
         'control-energy/cross term; negative results cannot exclude faithful MPPI')
BUDGET_NOTE = ('80 planning packet predictions equals 40 complete H2 candidates, not 80 '
               'complete candidates. Literal 80 H2 candidates requires 160+4 predictions '
               'and is not matched. Main-thread acknowledgment of units is mandatory.')


def write(path, data):
    full20.write(path, data)


def _arm(*, seed, root, provenance, action_domain, target_metric, replay, fixture,
         phase=lambda name:nullcontext(), journal=lambda event:None):
    """Reuse the unchanged kernel; callbacks dispatch two batch1 packets per cost."""
    if type(seed) is not int or seed < 0:
        raise ValueError('nonnegative integer seed required')
    if not fixture and not isinstance(replay, full20.RootAnchoredCudaReplay):
        raise ValueError('production requires the sealed host CUDA replay adapter')
    root, provenance, action_domain = deepcopy((root, provenance, action_domain))
    initial = full20._validated_context(root['context'])
    def edge(terminal, actions, name):
        return dict(program_id=name, kind='progress', root_id=root['root_id'],
            provenance=deepcopy(provenance), action_domain=deepcopy(action_domain),
            input_context=deepcopy(initial), output_context=deepcopy(terminal), actions=deepcopy(actions))
    compose(root=root, provenance=provenance, action_domain=action_domain,
        edges=[edge(initial, full20.raw_actions(np.zeros((2,5,2))), 'metadata_check')],
        replay=lambda c,a:c, max_edges=1, max_actions=10)
    if (action_domain['name'] != 'raw' or action_domain['action_block'] != 5 or
            action_domain['low'] != [-1.,-1.] or action_domain['high'] != [1.,1.]):
        raise ValueError('same official raw box and five-step action block required')
    ledger = full20._Ledger(replay, ARM)
    attempts, best, batch_count = [], None, 0
    started = time.perf_counter()

    def costs(batch):
        nonlocal best, batch_count
        batch_count += 1
        stage = 'initial' if batch_count == 1 else 'perturbation' if batch_count % 2 == 0 else 'updated_nominal'
        expected = 12 if stage == 'perturbation' else 1
        if batch.shape != (expected,2,5,2):
            raise ValueError('kernel schedule differs from frozen candidate accounting')
        values = []
        for parameters in batch:
            commands = np.asarray(parameters, dtype=np.float32)
            event = dict(candidate_id=f'{ARM}_{len(attempts):02d}', stage=stage,
                generation=-1 if stage == 'initial' else (batch_count-2)//2,
                commands=commands.tolist(), actions=full20.raw_actions(commands),
                proposal_dimensions=20, packet_shape=[2,5,2], completed=False)
            attempts.append(event)
            try:
                terminal = ledger.call(initial, event['actions'], 'planning', [len(attempts)-1])
                cost = full20._cost(target_metric(deepcopy(terminal)))
                event.update(completed=True, cost=cost)
                if best is None or cost < best['cost']:
                    best = dict(candidate_id=event['candidate_id'], cost=cost,
                        actions=deepcopy(event['actions']), terminal=deepcopy(terminal))
                values.append(cost)
            except Exception as exc:
                event.update(error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                journal(deepcopy(event))
        return np.asarray(values, dtype=np.float64)

    kernel_receipt, kernel_error = None, None
    with phase('planning'):
        try:
            selected, kernel_receipt = kernels.mppi(costs, initial=np.zeros((2,5,2)),
                lower=-1., upper=1., samples=SCHEDULE['samples'], iterations=SCHEDULE['iterations'],
                temperature=SCHEDULE['temperature'], noise_scale=SCHEDULE['noise_scale'], seed=seed)
            if (kernel_receipt['cost_evaluations'] != 40 or batch_count != 7 or best is None
                    or kernel_receipt['best_cost'] != best['cost']
                    or full20.raw_actions(selected) != best['actions']):
                raise ValueError('kernel best/action or candidate accounting mismatch')
        except Exception as exc:
            kernel_receipt = getattr(exc, 'receipt', kernel_receipt)
            kernel_error = dict(error_type=type(exc).__name__, error=str(exc))
    saved_witness, replay_error = None, None
    if kernel_error is None and best is not None:
        with phase('evaluation'):
            try:
                actions = best['actions']
                mid = ledger.call(initial, actions[:5], 'replay', [40])
                end = ledger.call(mid, actions[5:], 'replay', [41])
                full = ledger.call(initial, actions, 'replay', [42])
                if full != end or full != best['terminal']:
                    raise ValueError('independent segmented/full/search contexts differ')
                edges = [edge(end, actions, best['candidate_id'])]
                receipt = compose(root=root, provenance=provenance, action_domain=action_domain,
                    edges=edges, replay=lambda c,a:deepcopy(full), max_edges=1, max_actions=10)
                saved_witness = dict(receipt=receipt, edges=edges, independent_full_replay_exact=True)
            except Exception as exc:
                replay_error = dict(error_type=type(exc).__name__, error=str(exc))
    spent = {name:sum(e['predict_samples'] for e in ledger.events if e['phase'] == name)
             for name in ('planning','replay')}
    complete = (kernel_error is None and saved_witness is not None and len(attempts) == 40
        and all(e['completed'] for e in attempts) and all(e['completed'] for e in ledger.events)
        and spent == dict(planning=80,replay=4))
    return dict(arm=ARM, seed=seed, fixture_only=fixture, attempts=attempts,
        kernel_receipt=kernel_receipt, kernel_error=kernel_error,
        kernel_count_scope='kernel attempted batch size may exceed dispatched candidates after failure; packet ledger is actual dispatch',
        selected_candidate_id=best['candidate_id'] if complete else None,
        selected_model_cost=best['cost'] if complete else None,
        witness=saved_witness if complete else None, replay_error=replay_error,
        ledger=ledger.events, planning_predict_samples=spent['planning'], replay_predict_samples=spent['replay'],
        prediction_budget_matched=complete, status='prediction_matched_only' if complete else 'incomplete_or_failed',
        actual_wall_time_s=time.perf_counter()-started, claim_boundary=CLAIM,
        P1_admission=False, P2_admission=False, fair_total_budget_comparison=False,
        physical_efficacy_verified=False, official_MPPI=False)


def run_fixture_arm(**kwargs):
    """Explicit non-neural mathematical fixture, never real-model evidence."""
    return _arm(fixture=True, **kwargs)


def prepare(reference, output):
    """Bind the sealed full20 context, schedule, instrumentation, and new sources."""
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if TAG not in output.name:
        raise ValueError('new dedicated tagged output required')
    path, original, previous, source = full20.load_prepared(reference)
    reference_report = json.loads((path/'report.json').read_text())
    if reference_report['source_seals'] != previous['source_seals']:
        raise ValueError('reference prepared/report source seal mismatch')
    full20.selected_actions(reference_report, original, source)
    if (original['schedule']['candidates'] != 40 or original['schedule']['planning_predict_samples'] != 80
            or original['schedule']['replay_predict_samples'] != 4 or original['schedule']['proposal_dimensions'] != 20):
        raise ValueError('strict existing full20 40-H2/80+4 budget required')
    bindings = deepcopy(original['bindings'])
    for filename in ('protocol.json','prepared.json','report.json'):
        target = path/filename
        bindings[str(target)] = full20.sha(target)
    source_rows = {r['root_sha256']:r for r in source['rows']}
    ready = deepcopy(original['prepared'])
    if len(source_rows) != 4 or len(ready) != 4 or len({r['root_sha256'] for r in ready}) != 4:
        raise ValueError('four unique original calibration roots required')
    for row in ready:
        if (row['root_sha256'] not in source_rows or type(row['root_index']) is not int
                or not 0 <= row['root_index'] < 4 or row['seed'] != SEEDS[row['root_index']]):
            raise ValueError('unchanged root identities/index/seed required')
    protocol = dict(schema=TAG+'.protocol', reference=str(path), bindings=bindings,
        original_source_seals=previous['source_seals'], prepared=ready, schedule=deepcopy(SCHEDULE),
        policy=original['policy'], compilation=original['compilation'], normalization=original['normalization'],
        objective=original['objective'], arms=[ARM], claim_boundary=CLAIM, budget_unit_resolution=BUDGET_NOTE,
        literal_80_complete_candidates_supported=False, main_thread_budget_review_required=True,
        calibration_only=True, new_results_accessed_for_method_selection=False,
        physical_outcomes_used=False, training=False, P1_admission=False, P2_admission=False,
        fair_total_budget_comparison=False, unmeasured=deepcopy(original['unmeasured']))
    _,_,config = checked.validate_inputs(original['compilation'], original['policy'])
    files = checked.sources_before_allocate(source,config)
    files += [Path(module.__file__) for name,module in tuple(sys.modules.items())
              if name.startswith('experiments.') and getattr(module,'__file__','').endswith('.py')]
    files += [Path(__file__), PROJECT/'tests'/f'test_{TAG}.py', PROJECT/'docs'/f'{TAG}.md']
    files = sorted({p.resolve() for p in files})
    output.mkdir(exist_ok=False)
    write(output/'protocol.json',protocol)
    seals = [create_seal(files=files[i:i+40], args_json=json.dumps(dict(protocol_sha256=digest(protocol))),
                        output_dir=output/f'source_seal_{i//40}') for i in range(0,len(files),40)]
    write(output/'prepared.json',dict(schema=TAG+'.prepared', status='prepared_pending_main_thread_review',
        protocol_sha256=digest(protocol), source_seals=seals, new_neural_calls=0,
        new_physical_steps=0, temperature_frozen=1., P2_admission=False))
    load_prepared(output)
    print(json.dumps(dict(status='prepared_pending_main_thread_review',output=str(output),
        candidates_per_root=40,planning_predict_samples_per_root=80,replay_predict_samples_per_root=4,
        new_neural_calls=0,new_physical_steps=0)))


def load_prepared(output):
    output = Path(output).resolve()
    protocol = json.loads((output/'protocol.json').read_text())
    prepared = json.loads((output/'prepared.json').read_text())
    if (protocol['schema'] != TAG+'.protocol' or digest(protocol) != prepared['protocol_sha256']
            or protocol['schedule'] != SCHEDULE or protocol['claim_boundary'] != CLAIM
            or protocol['budget_unit_resolution'] != BUDGET_NOTE or protocol['arms'] != [ARM]):
        raise ValueError('frozen MPPI protocol differs')
    for seal in prepared['source_seals']:
        verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    for path, expected in protocol['bindings'].items():
        if full20.sha(path) != expected:
            raise ValueError('bound input changed: '+path)
    path, original, previous, source = full20.load_prepared(protocol['reference'])
    if (protocol['prepared'] != original['prepared'] or protocol['normalization'] != original['normalization']
            or protocol['original_source_seals'] != previous['source_seals']
            or protocol['policy'] != original['policy'] or protocol['compilation'] != original['compilation']):
        raise ValueError('MPPI reference context/domain/seed changed')
    report = json.loads((path/'report.json').read_text())
    if report['source_seals'] != previous['source_seals']:
        raise ValueError('reference report seal mismatch')
    full20.selected_actions(report,original,source)
    return output,protocol,prepared,source


def roots_provenance(protocol,prepared,ready):
    from experiments.spectral_reachability.run_pusht_d2p_budget_guard_cuda import OFFICIAL_SHA
    return dict(checkpoint_sha256=OFFICIAL_SHA, encoder_sha256=OFFICIAL_SHA,
        source_sha256=prepared['source_seals'][0]['manifest_sha256'], protocol_sha256=digest(protocol),
        metric_sha256=ready['metric_sha256'], candidate_bank_sha256=digest(dict(schedule=SCHEDULE,seed=ready['seed'])))


def cuda(output, *, acknowledge_40_h2=False):
    """Main thread only after source review; never called by prepare or tests."""
    if acknowledge_40_h2 is not True:
        raise ValueError(BUDGET_NOTE)
    output,protocol,prepared,source = load_prepared(output)
    # Exclusive started receipt prevents reuse, including after fatal allocation.
    write(output/'started.json',dict(protocol_sha256=digest(protocol),budget_acknowledged=True))
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    from experiments.spectral_reachability.d2p_adaptive_budget_v2 import PredictionBudget,Binding
    host = base.require_cuda_allocation('cuda')
    import torch
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    model = base.load_compatible_pretrained(Path(protocol['policy']),device=device).eval().requires_grad_(False)
    before = checked.tensor_state_hash(model)
    torch.cuda.reset_peak_memory_stats(device)
    source_rows = {r['root_sha256']:r for r in source['rows']}
    rows = []
    for ready in protocol['prepared']:
        provenance = roots_provenance(protocol,prepared,ready)
        binding = Binding(provenance['source_sha256'],TAG,ready['root_sha256'],checked.OFFICIAL_SHA)
        work = dict(calls=0,samples=0,processed_context_tokens=0)
        def hook(module,args):
            work['calls'] += 1
            work['samples'] += int(args[0].shape[0])
            work['processed_context_tokens'] += int(args[0].shape[0]*args[0].shape[1])
        handle = model.action_encoder.register_forward_pre_hook(hook)
        try:
            with (output/f'root_{ready["root_index"]}_candidates.jsonl').open('x') as journal:
                def save(event):
                    journal.write(json.dumps(event,allow_nan=False)+'\n')
                    journal.flush()
                replay = full20.RootAnchoredCudaReplay(model,device)
                with full20.actual_entries(model,replay) as entries, PredictionBudget(
                        model,binding=binding,method=ARM,sample_limit=84) as budget:
                    result = _arm(seed=ready['seed'],root=ready['root'],provenance=provenance,
                        action_domain=ready['action_domain'], target_metric=full20.metric_for(source_rows[ready['root_sha256']]['compiled']),
                        replay=replay,fixture=False, phase=lambda name:budget.phase(name,method=ARM,horizon_tokens=2),journal=save)
                    actual = budget.receipt()
        finally:
            handle.remove()
        actual_ok = ([(p['phase'],p['predict_calls'],p['predict_samples']) for p in actual['phases']] ==
            [('planning',80,80),('evaluation',4,4)] and all(p['completed'] for p in actual['phases'])
            and work == dict(calls=84,samples=84,processed_context_tokens=252)
            and full20.entries_match(entries,result['ledger'],ARM))
        result.update(actual_prediction_budget=actual,actual_model_entries=entries,action_encoder_work=work,
            prediction_budget_matched=result['prediction_budget_matched'] and actual_ok)
        if not result['prediction_budget_matched']:
            result['status'] = 'incomplete_or_failed'
        write(output/f'root_{ready["root_index"]}_{ARM}.json',result)
        rows.append(dict(root_sha256=ready['root_sha256'],root_index=ready['root_index'],
                         root=ready['root'],action_domain=ready['action_domain'],rows=[result]))
    gates = dict(all_four_retained=len(rows) == 4,
        actual80_plus4_all=all(r['rows'][0]['prediction_budget_matched'] for r in rows),
        model_unchanged=before == checked.tensor_state_hash(model),
        frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()),
        inputs_unchanged=all(full20.sha(p) == h for p,h in protocol['bindings'].items()))
    verified = [verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256']) for s in prepared['source_seals']]
    load_prepared(output)
    report = dict(schema=TAG+'.report',status='calibration_completed' if all(gates.values()) else 'integrity_fail',
        protocol_sha256=digest(protocol),rows=rows,gates=gates,host=host,source_seals=prepared['source_seals'],
        source_verification=verified,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        actual_predict_samples=sum(e['samples'] for r in rows for e in r['rows'][0]['actual_model_entries'] if e['kind'] == 'predict'),
        P1_admission=False,P2_admission=False,fair_total_budget_comparison=False,
        physical_efficacy_verified=False,official_MPPI=False,claim_boundary=CLAIM)
    write(output/'report.json',report)
    if not all(gates.values()):
        raise RuntimeError('MPPI-inspired calibration incomplete; all attempts retained, no physics')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE)
    prep.add_argument('--output',type=Path,required=True)
    run = commands.add_parser('cuda')
    run.add_argument('--output',type=Path,required=True)
    run.add_argument('--acknowledge-40-h2',action='store_true')
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    prepare(**args) if command == 'prepare' else cuda(**args)


if __name__ == '__main__':
    main()
