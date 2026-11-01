"""Independent saved-receipt arithmetic and conservative matched P3 frontier.

No neural/simulator imports, no new execution evidence. The old audit is used
only for byte-integrity helpers and a cross-check, never for reconstruction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from experiments.spectral_reachability import p3_quality_cost_audit_20261027_v2 as saved

NAME = 'p3_budget_frontier_20261027_v1'
ORIGINAL = 'results/P3_ADAPTIVE_RUNTIME_20261027_cuda_v1/P3_ADAPTIVE_RUNTIME_report_20261027.json'
RUNTIME = 'results/P3_ADAPTIVE_RUNTIME_20261027_cuda_v1/P3_ADAPTIVE_RUNTIME_protocol_20261027.json'
PHYSICAL_PROTOCOL = 'results/P3_MATCHED_PHYSICAL_20261027_v1_cuda/protocol.json'
OLD = 'results/p3_quality_cost_audit_20261027_v2/report.json'
METHODS = saved.METHODS
BACKWARD = saved.BACKWARD
require, count, canonical, sha = saved.require, saved.count, saved.canonical, saved.sha


def equal(a, b):
    """Typed JSON equality; bool is not an integer counter."""
    return canonical(a) == canonical(b)


def census(rows, fields, expected):
    result = saved.index(rows, fields)
    require(set(result) == expected, 'missing/extra census: '+str(fields))
    return result


def reconstruct(row, corrected, bound, plan, binding, limits):
    """Count one-step kernel events; verification is a separate predict phase.

    Counters are coordinates, NOT summed units: backward calls contain VJP
    rows, action encoder calls accompany predictions, and context tokens are
    processed histories, not additional predictions.
    """
    receipt, search = row['prediction_receipt'], row['search']
    require(equal(receipt['binding'], binding), 'task/model/root/protocol receipt binding')
    require(receipt['method'] == row['method'], 'receipt method mismatch')
    phases = saved.index(receipt['phases'], ('phase',))
    require(set(phases) == {('planning',), ('evaluation',)}, 'exact planning and verification phases')
    planning, verify = phases['planning',], phases['evaluation',]
    for phase in (planning, verify):
        require(phase['method'] == row['method'] and phase['completed'] is True and
                count(phase['failed_predict_calls']) == 0, 'incomplete predict phase')
        for field in ('predict_samples', 'predict_calls'):
            count(phase[field])
    events, horizon, kernels = 0, 0, 0
    for depth, layer in enumerate(search['layers'], 1):
        require(type(layer['depth']) is int and layer['depth'] == depth, 'layer depth')
        used, parents = 0, set()
        for kernel in layer['kernels']:
            parent = tuple(kernel['parent_indices'])
            require(len(parent) == depth-1 and parent not in parents, 'duplicate/wrong parent kernel')
            parents.add(parent)
            kr = kernel['receipt']
            require(count(kr['max_depth']) == 1, 'one-step kernel required')
            require(all(count(kr[k]) == 0 for k in
                    ('failed_attempts', 'metric_failed_attempts', 'replay_failed_attempts')), 'failed kernel receipt')
            require(count(kr['attempted_expansions']) == len(kr['attempts']), 'kernel event count')
            for primitive, event in enumerate(kr['attempts']):
                require(equal(event['primitive_indices'], [primitive]) and event['depth'] == 1 and
                        event['succeeded'] is True and event['replay_completed'] is True and
                        event['stage'] == 'metric', 'candidate receipt identity/completion')
            used += len(kr['attempts']); kernels += 1
        require(count(layer['attempted_expansions']) == used, 'layer event count')
        if used:
            horizon = depth  # An empty depth-3 layer must not increase H.
        events += used
    best = search['best_plan']
    depth = count(best['depth'])
    require(depth == len(best['primitive_indices']) and depth > 0, 'selected path length')
    require(events == count(search['attempted_expansions']) == count(search['metric_calls']) ==
            planning['predict_samples'] == planning['predict_calls'], 'planning receipt total')
    require(verify['predict_samples'] == verify['predict_calls'] == depth, 'verification receipt total')
    total = events+depth
    require(count(receipt['attempted_predict_samples']) == total and
            count(receipt['blocked_calls']) == 0, 'predict total/blocked calls')
    resources = deepcopy(row['resources'])
    for value in resources.values():
        count(value)
    resources['max_planning_horizon_tokens'] = max(horizon, depth)
    require(resources['predict_samples'] == resources['predict_calls'] ==
            resources['imagined_action_tokens'] == total, 'predict coordinates')
    require(resources['rollout_samples'] == events+1, 'verification trajectory double count')
    require(all(resources[k] == 0 for k in BACKWARD), 'no-grad runtime derivative policy')
    require(all(resources[k] == 0 for k in ('encoder_forwards', 'encoder_samples',
            'physical_action_tokens', 'physical_env_steps', 'physical_branches')), 'runtime scope leakage')
    require(equal(resources, bound['corrected_resources']), 'independent resource vector differs')
    require(all(resources[k] <= count(limits[k]) for k in resources), 'budget cap exceeded')
    require(corrected['predict_samples'] == total and bound['planning_events'] == events and bound['kernels'] == kernels and
            corrected['actual_horizon'] == bound['actual_horizon'] == resources['max_planning_horizon_tokens'],
            'corrected event/horizon join')
    require(equal(best['primitive_indices'], corrected['best_path']) and
            equal(best['primitive_indices'], bound['selected_path']) and
            equal(best['primitive_indices'], plan['primitive_indices']), 'path binding')
    require(equal(plan['original_prediction_receipt'], receipt), 'physical prediction receipt')
    require(equal(plan['commands'], best['actions']) and
            plan['planned_commands'] == 5*depth == len(plan['commands']) and
            plan['planned_horizon_tokens'] == depth and
            plan['corrected_search_horizon'] == resources['max_planning_horizon_tokens'], 'plan horizon/actions')
    require(sha(canonical({k:v for k,v in plan.items() if k != 'selection_sha256'}).encode()) ==
            plan['selection_sha256'], 'selection SHA')
    context_tokens = sum(int(t)*count(n) for phase in phases.values()
                         for t,n in phase['context_lengths'].items())
    require(row['action_encoder_samples'] == row['action_encoder_forwards'] == total and
            context_tokens == row['processed_action_context_tokens'] == 3*total, 'action/context accounting')
    return dict(root_sha256=row['root_sha256'], method=row['method'], binding=binding,
                planning_predict_samples=events, verification_predict_samples=depth,
                total_predict_samples=total, saved_resource_vector=resources,
                action_encoder_samples=row['action_encoder_samples'], processed_context_tokens=context_tokens,
                runtime_backward_and_vjp={k:resources[k] for k in BACKWARD},
                dense_jacobian_materializations=None, upstream_compilation_cost=None,
                cost_scope='runtime search+independent verification only; shared planning charged once',
                selected_path=best['primitive_indices'], search_status=search['status'],
                selected_horizon=depth, actual_search_horizon=resources['max_planning_horizon_tokens'])


def pareto(left_cost, right_cost, left_quality, right_quality):
    """All costs and both task distances minimized; unknown axes forbid claims."""
    require(set(left_cost) == set(right_cost), 'cost axes differ')
    if any(v is None for v in (*left_cost.values(), *right_cost.values())):
        return 'unknown_cost_not_ordered'
    a = list(left_cost.values())+list(left_quality)
    b = [right_cost[k] for k in left_cost]+list(right_quality)
    if a == b:
        return 'tie'
    if all(x <= y for x,y in zip(a,b)):
        return 'fixed_dominates'
    if all(y <= x for x,y in zip(a,b)):
        return 'adaptive_dominates'
    return 'tradeoff'


def audit(main, binding_report, physical, original, runtime, protocol, old):
    """Independent arithmetic; no call to the earlier audit's audit() function."""
    require(main['status'] == 'saved_accounting_corrected' and
            binding_report['status'] == 'saved_binding_audit_pass' and
            binding_report['all_checks_pass'] is True and physical['status'] == 'calibration_recorded',
            'upstream completion required')
    for report in (main, binding_report, physical, old):
        require(report['P1_admission'] is False and report['P3_admission'] is False, 'scope escalation')
    identity = sha(canonical(runtime['protocol']).encode())
    require(identity == runtime['protocol_id'] == original['protocol_id'] == binding_report['protocol_id'],
            'protocol SHA identity')
    roots = runtime['protocol']['roots']
    require(len(roots) == len(set(roots)) == 4 and all(len(r) == 64 for r in roots), 'frozen four roots')
    expected = {(r,m) for r in roots for m in METHODS}
    fields = ('root_sha256', 'method')
    orig, corr, bound, plans = [census(rows, fields, expected) for rows in
        (original['rows'], main['corrections'], binding_report['rows'], protocol['plans'])]
    contracts = {
        c['contract']['binding']['root_id']:c['contract'] for c in runtime['paired_plan']['roots']}
    require(len(runtime['paired_plan']['roots']) == len(contracts) == 4 and set(contracts) == set(roots),
            'contract root census')
    require(protocol['physical_repetitions'] == 2 and protocol['root_selection_changed'] is False,
            'frozen dependent replay contract')
    expanded = {k+(rep,) for k in expected for rep in (0,1)}
    branches = census(physical['rows'], fields+('repetition',), expanded)
    encodings = census(physical['encodings'], fields+('repetition',), expanded)
    arm_rows, rows, comparisons = [], [], []
    for root in roots:
        contract = contracts[root]
        require(contract['binding']['root_id'] == root and contract['binding']['protocol_id'] == identity and
                contract['binding']['task_id'] == 'pusht_cached_latent_runtime', 'task/root contract')
        for method in METHODS:
            key = root, method
            arm_rows.append(reconstruct(orig[key], corr[key], bound[key], plans[key],
                                        contract['binding'], contract['limits']))
            for rep in (0,1):
                branch, enc = branches[key+(rep,)], encodings[key+(rep,)]
                require(type(branch['repetition']) is int, 'integer repetition')
                require(branch['selection_sha256'] == plans[key]['selection_sha256'], 'physical selection')
                attempts, returns = count(branch['physical_env_step_attempts']), count(branch['physical_env_step_returns'])
                planned = plans[key]['planned_commands']
                require(branch['planned_commands'] == planned and returns <= attempts <= planned and
                        returns == branch['completed_commands'] == branch['actual_endpoint_primitive_steps'] == len(branch['trace']) and
                        attempts == branch['attempted_commands'] and
                        branch['remaining_commands_not_executed'] == planned-returns, 'physical attempts/returns')
                require(branch['guard']['root_id'] == root and type(branch['guard']['abstained']) is bool,
                        'guard root/abstention')
                full = branch['status'] == 'completed' and returns == planned
                require(type(branch['full_horizon_completed']) is bool and branch['full_horizon_completed'] == full and
                        not (full and branch['guard']['abstained']), 'full completion/guard contradiction')
                require(enc['status'] == 'completed' and count(enc['attempted_encode_samples']) == 1,
                        'terminal encode receipt')
                # Keep every replay, guard failure and endpoint. No successful-only projection.
                rows.append(deepcopy(branch))
        for rep in (0,1):
            l,r = (branches[root,m,rep] for m in METHODS)
            lp,rp = (plans[root,m] for m in METHODS)
            la,ra = arm_rows[-2:]
            matches = dict(search_horizon=la['actual_search_horizon']==ra['actual_search_horizon'],
                selected_horizon=la['selected_horizon']==ra['selected_horizon'],
                predict_samples=la['total_predict_samples']==ra['total_predict_samples'],
                planning_samples=la['planning_predict_samples']==ra['planning_predict_samples'],
                verification_samples=la['verification_predict_samples']==ra['verification_predict_samples'],
                saved_resource_vector=equal(la['saved_resource_vector'],ra['saved_resource_vector']),
                physical_attempts=l['physical_env_step_attempts']==r['physical_env_step_attempts'],
                actual_endpoint_steps=l['physical_env_step_returns']==r['physical_env_step_returns'])
            # Source-derived feedback contract: same frozen executor/guard, no replanning.
            # Receipt outcomes (guard observations/abort time) are NOT contract identities.
            contract_matches = dict(task_model=equal(la['binding'],ra['binding']),
                root_context=equal(l['context_identity'],r['context_identity']),
                task_metric=all(equal(lp[k],rp[k]) for k in ('goal_vectors_sha256','projection_sha256')),
                action_bank=equal(lp['bank'],rp['bank']),
                predictor_context=equal(lp['predictor_context'],rp['predictor_context']),
                feedback_guard=l['guard']['scope']==r['guard']['scope'])
            full = l['full_horizon_completed'] and r['full_horizon_completed']
            guard_partial = not full and l['guard']['abstained'] and r['guard']['abstained']
            classification = ('full-path' if full else 'guard-partial') if (
                all(matches.values()) and all(contract_matches.values()) and (full or guard_partial)) else 'unmatched'
            costs = []
            for arm,branch in ((la,l),(ra,r)):
                costs.append(dict(predict_calls=arm['saved_resource_vector']['predict_calls'],
                    predict_samples=arm['total_predict_samples'], action_encoder_samples=arm['action_encoder_samples'],
                    **arm['runtime_backward_and_vjp'], physical_step_attempts=branch['physical_env_step_attempts'],
                    terminal_encode_samples=encodings[root,arm['method'],rep]['attempted_encode_samples']))
            quality = [[b['pose']['pose_cost'], b['official']['official_state_distance']] for b in (l,r)]
            canonical(quality)  # Reject nonfinite saved endpoint arithmetic.
            for q in quality:
                require(all(type(v) in (int,float) and v >= 0 for v in q), 'invalid task quality')
            relation = pareto(*costs,*quality) if classification == 'full-path' else 'ineligible'
            comparisons.append(dict(root_sha256=root,repetition=rep,classification=classification,
                resource_matches=matches,contract_matches=contract_matches,
                unmatched_dimensions=[k for k,v in {**matches,**contract_matches}.items() if not v],
                fixed_cost_vector=costs[0],adaptive_cost_vector=costs[1],
                adaptive_minus_fixed_quality=[b-a for a,b in zip(*quality)],
                identical_selected_path=equal(lp['primitive_indices'],rp['primitive_indices']),
                saved_counter_frontier_relation=relation,total_pipeline_frontier_relation='unknown_cost_not_ordered',
                quality_scope='paired endpoint arithmetic only; no efficacy; partial/unmatched excluded from frontier'))
    totals = dict(planning_predict_samples=sum(a['planning_predict_samples'] for a in arm_rows),
        verification_predict_samples=sum(a['verification_predict_samples'] for a in arm_rows),
        predict_samples=sum(a['total_predict_samples'] for a in arm_rows),
        action_encoder_samples=sum(a['action_encoder_samples'] for a in arm_rows),
        processed_action_context_tokens=sum(a['processed_context_tokens'] for a in arm_rows),
        physical_step_attempts=sum(b['physical_env_step_attempts'] for b in rows),
        physical_step_returns=sum(b['physical_env_step_returns'] for b in rows),
        terminal_encode_samples=sum(e['attempted_encode_samples'] for e in encodings.values()))
    require(totals['predict_samples'] == main['actual_predict_samples'] == binding_report['actual_predict_samples'] and
            totals['planning_predict_samples'] == binding_report['planning_events'], 'cohort predict census')
    require(totals['physical_step_attempts'] == physical['selected_path_env_step_attempts'] and
            totals['physical_step_returns'] == physical['selected_path_env_step_returns'] and
            totals['terminal_encode_samples'] == physical['attempted_terminal_encode_samples'], 'physical/encoding census')
    # Independent cross-check of old output, NOT its classifications as input.
    old_pairs = census(old['comparisons'], ('root_sha256','repetition'), {(r,n) for r in roots for n in (0,1)})
    names = {'full-path':'matched_full_path_saved_resources','guard-partial':'matched_partial_guard_endpoint','unmatched':'unmatched'}
    for c in comparisons:
        previous = old_pairs[c['root_sha256'],c['repetition']]
        require(equal(c['resource_matches'],previous['matches']), 'old resource match cross-check')
        # Stronger contracts may downgrade a formerly matched pair, never upgrade it.
        if all(c['contract_matches'].values()):
            require(names[c['classification']] == previous['classification'], 'old classification cross-check')
        require(equal(c['adaptive_minus_fixed_quality'],[previous['adaptive_minus_fixed_pose_cost'],
                previous['adaptive_minus_fixed_official_state_distance']]), 'old quality arithmetic cross-check')
    previous_rows = census(old['rows'],fields+('repetition',),expanded)
    for b in rows:
        previous = previous_rows[b['root_sha256'],b['method'],b['repetition']]
        for field in ('guard','failure','failures','status','full_horizon_completed','pose',
                      'physical_env_step_attempts','physical_env_step_returns','selection_sha256'):
            require(equal(b[field],previous[field]), 'old branch retention cross-check: '+field)
        require(equal(b['official'],previous['officialstate']), 'old official metric cross-check')
    gates = dict(P1_passed=runtime['protocol']['p1_dependency']=='passed',
        all_pairs_contract_matched=all(all(c['contract_matches'].values()) for c in comparisons),
        all_pairs_full_path_matched=all(c['classification']=='full-path' for c in comparisons),
        total_pipeline_cost_complete=False, independent_heldout_roots=False,
        prospective_budget_frontier_sweep=False, paired_uncertainty_and_multiplicity=False,
        demonstrated_adaptive_quality_cost_gain=False)
    return dict(schema=NAME+'.v1',status='saved_receipt_frontier_recomputed',
        classifications=dict(Counter(c['classification'] for c in comparisons)),
        totals=totals,arm_costs=arm_rows,rows=rows,comparisons=comparisons,
        root_count=len(roots),independent_new_roots=0,physical_execution_count=len(rows),
        guard_aborted_executions=sum(b['guard']['abstained'] for b in rows),
        repeats_are_dependent=True,shared_planning_charged_once=True,
        derivative_totals={k:sum(a['runtime_backward_and_vjp'][k] for a in arm_rows) for k in BACKWARD},
        unknown_or_excluded_costs=dict(dense_jacobian_materializations=None,upstream_compilation=None,
            checkpoint_loading=None,reset_warmup=None,source_hashing=None,svd_flops=None,total_flops=None),
        feedback_contract_scope='source-derived sealed open-loop selected commands + per-step RGB presence guard; no online replanning; evaluator state is not feedback',
        frontier_scope='two saved arms at one frozen cap, within root/replay only; no pooled or swept frontier',
        gates=gates,remaining_gaps=[k for k,v in gates.items() if not v],
        P1_admission=False,P3_admission=all(gates.values()),adaptive_superiority=False,
        total_flops_equality=False,new_predict_samples=0,new_physical_steps=0,
        previous_audit_arithmetic_reproduced=True)


def run(root):
    root = Path(root).resolve()
    output = root/'results'/NAME
    require(not output.exists(), 'exclusive NEW output required')
    evidence = saved.Evidence(root)
    old = evidence.json(root/OLD)
    old_dir = (root/OLD).parent
    manifest = evidence.json(old_dir/'manifest.json')
    for relative,digest in manifest['artifacts_sha256'].items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'unsafe artifact path')
        evidence.raw(old_dir/relative,digest)
    require(equal(manifest['inputs_sha256'],old['inputs_sha256']), 'old manifest input census')
    for path,digest in old['inputs_sha256'].items():
        evidence.raw(Path(path),digest)
    paths = [saved.MAIN,saved.BINDING,saved.PHYSICAL,ORIGINAL,RUNTIME,PHYSICAL_PROTOCOL]
    reports = [evidence.json(root/p) for p in paths]
    # Verify source seals now as well as the immutable upstream input closure.
    seals = [evidence.seal(s) for s in (reports[0]['source_seal'],reports[1]['source_seal'],
                                      *reports[2]['source_seals'])]
    result = audit(*reports,old)
    for branch in reports[2]['rows']:
        raw = evidence.raw(Path(branch['journal_path']),branch['journal_sha256'])
        journal = [saved.decode(line) for line in raw.splitlines()]
        attempts = [e for e in journal if e['kind']=='attempt']
        returned = [e for e in journal if e['kind']=='step_returned']
        observations = [e for e in journal if e['kind']=='observation' and 'command' in e]
        require(len(attempts)==branch['physical_env_step_attempts'] and
                len(returned)==branch['physical_env_step_returns'], 'journal attempts/returns')
        require([e['primitive_step'] for e in attempts]==list(range(1,len(attempts)+1)) and
                [e['primitive_step'] for e in returned]==list(range(1,len(returned)+1)), 'journal step census')
        require(len(observations)==len(branch['trace']) and all(
                all(equal(e.get(k),v) for k,v in t.items())
                for t,e in zip(branch['trace'],observations)), 'journal trace differs')
        for event in journal:
            if 'rgb_path' in event:
                evidence.raw(Path(event['rgb_path']),event['rgb_file_sha256'])
        evidence.raw(Path(branch['terminal_rgb_path']),branch['terminal_rgb_sha256'])
    sources = [Path(__file__).resolve(),root/'tests'/('test_'+NAME+'.py'),root/'docs'/(NAME+'.md')]
    snapshots = {p.name:evidence.raw(p) for p in sources}
    evidence.recheck()
    result.update(inputs_sha256=dict(sorted(evidence.hashes.items())),source_seals_verified=seals,
        provenance_scope='upstream manifest closure and live/archive sources byte-verified, rechecked; weights/datasets not loaded; hashes do not prove execution',
        external_model_bindings_not_rehashed=old['external_model_bindings_not_rehashed'])
    output.mkdir(exist_ok=False)
    archive = output/'sources'; archive.mkdir()
    for name,raw in snapshots.items():
        with (archive/name).open('xb') as stream:
            stream.write(raw)
    with (output/'report.json').open('x') as stream:
        stream.write(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    with (output/'report.md').open('x') as stream:
        stream.write('# P3 matched-root saved-counter frontier\n\n'+
            canonical(result['classifications'])+'\n\n'+canonical(result['totals'])+'\n\n'+
            'P3 admission: false. No new experiment evidence. Repeats remain dependent.\n\n'+
            'Remaining gates: '+', '.join(result['remaining_gaps'])+'\n')
    evidence.recheck()
    artifacts = {str(p.relative_to(output)):sha(p.read_bytes()) for p in
                 [output/'report.json',output/'report.md',*(archive/n for n in snapshots)]}
    with (output/'manifest.json').open('x') as stream:
        stream.write(json.dumps(dict(artifacts_sha256=artifacts,inputs_sha256=evidence.hashes),
                                sort_keys=True,indent=2)+'\n')
    print(canonical({k:result[k] for k in ('status','classifications','totals','P3_admission','remaining_gaps')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2])
    run(parser.parse_args().project_root)
