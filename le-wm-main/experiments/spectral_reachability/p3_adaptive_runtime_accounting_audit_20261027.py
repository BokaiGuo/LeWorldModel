"""Correct saved actual-depth accounting; no neural or physical execution."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path

from experiments.spectral_reachability.d2p_planning_compute_ledger import Binding, BudgetContract, Resources
from experiments.spectral_reachability.d2p_p3_paired_accounting import (
    BranchReceipt, PairedPlan, RootPlan, RunReceipt, compare_paired,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corrected_resources(row):
    """Reconcile complete saved batch-one events, not declared empty layers.

    This audit rejects incomplete/failed receipts rather than inventing missing
    counts. It changes only actual max horizon; original failures stay upstream.
    Phase horizon_tokens is a declaration, not evidence of executed depth.
    """
    if row['failure'] is not None or row['full_replay_exact'] is not True:
        raise ValueError('complete original search and verification required')
    attempted_depths, attempts = [], 0
    for layer in row['search']['layers']:
        depth, count = layer['depth'], layer['attempted_expansions']
        if type(depth) is not int or depth < 1 or type(count) is not int or count < 0:
            raise ValueError('nonnegative actual layer counts required')
        events = 0
        for kernel in layer['kernels']:
            receipt = kernel['receipt']
            if len(kernel['parent_indices']) != depth-1 or receipt['failed_attempts'] != 0:
                raise ValueError('incomplete or misbound kernel')
            if len(receipt['attempts']) != receipt['attempted_expansions']:
                raise ValueError('kernel event count mismatch')
            if any(a['replay_completed'] is not True or a['succeeded'] is not True or
                   len(a['primitive_indices']) != 1 for a in receipt['attempts']):
                raise ValueError('complete one-token kernel events required')
            events += len(receipt['attempts'])
        if events != count:
            raise ValueError('layer event count mismatch')
        if count:
            attempted_depths.append(depth)
        attempts += count
    best = row['search']['best_plan']
    verified_depth = best['depth']
    if (type(verified_depth) is not int or verified_depth < 1 or
            len(best['primitive_indices']) != verified_depth or len(best['actions']) != 5*verified_depth):
        raise ValueError('actual verified witness length required')
    phases = row['prediction_receipt']['phases']
    if len(phases) != 2 or [p['phase'] for p in phases] != ['planning', 'evaluation']:
        raise ValueError('exact planning/evaluation phase pair required')
    if any(p['completed'] is not True or p['failed_predict_calls'] != 0 for p in phases):
        raise ValueError('complete original predict phases required')
    expected = [attempts, verified_depth]
    if any(p['predict_calls'] != n or p['predict_samples'] != n for p,n in zip(phases,expected)):
        raise ValueError('actual batch-one predictor counts disagree with events')
    if phases[1]['horizon_tokens'] != verified_depth:
        raise ValueError('verification horizon differs from witness')
    original = Resources(**row['resources'])
    if (original.predict_samples != sum(expected) or original.predict_calls != sum(expected)
            or row['prediction_receipt']['attempted_predict_samples'] != sum(expected)
            or original.rollout_samples != attempts+1
            or original.imagined_action_tokens != sum(expected)
            or row['action_encoder_samples'] != sum(expected)
            or row['processed_action_context_tokens'] != 3*sum(expected)):
        raise ValueError('saved resources disagree with actual receipts')
    actual_depth = max([verified_depth]+attempted_depths)
    return Resources(**(row['resources'] | {'max_planning_horizon_tokens': actual_depth}))


def audit(report, protocol):
    if report['status'] != 'runtime_integration_completed' or not report['gates'] or not all(v is True for v in report['gates'].values()):
        raise ValueError('original complete integration report required')
    saved = protocol['paired_plan']
    roots = tuple(RootPlan(BudgetContract(Binding(**r['contract']['binding']),
        Resources(**r['contract']['limits'])), tuple(r['branch_ids'])) for r in saved['roots'])
    plan = PairedPlan(saved['source_seal'], roots, saved['fixed_method'], saved['adaptive_method'])
    if protocol['protocol_id'] != report['protocol_id'] or protocol['protocol_id'] != plan.source_seal:
        raise ValueError('original protocol binding differs')
    expected = {(r.contract.binding.root_id,m) for r in roots for m in (plan.fixed_method,plan.adaptive_method)}
    rows = report['rows']
    if len(rows) != len(expected) or {(r['root_sha256'],r['method']) for r in rows} != expected:
        raise ValueError('complete unique paired root/method census required')
    contracts = {r.contract.binding.root_id:r for r in roots}
    zero = Resources(**{f.name:0 for f in fields(Resources)})
    arms = {plan.fixed_method:[], plan.adaptive_method:[]}
    corrections = []
    for row in rows:
        resources = corrected_resources(row)
        root = contracts[row['root_sha256']]
        branch = BranchReceipt('search_and_verification','completed',False,resources)
        receipt = RunReceipt(root.contract.binding,row['method'],(branch,),zero,
            row['prediction_receipt'],row['walltime_seconds'])
        arms[row['method']].append(receipt)
        corrections.append(dict(root_sha256=row['root_sha256'], method=row['method'],
            old_horizon=row['resources']['max_planning_horizon_tokens'],
            actual_horizon=resources.max_planning_horizon_tokens,
            predict_samples=resources.predict_samples,
            search_status=row['search']['status'], best_path=row['search']['best_plan']['primitive_indices']))
    paired = compare_paired(plan,arms[plan.fixed_method],arms[plan.adaptive_method])
    return dict(schema='p3_saved_accounting_correction_20261027.v1',status='saved_accounting_corrected',
        corrections=corrections,paired=paired,
        corrected_rows=sum(c['old_horizon'] != c['actual_horizon'] for c in corrections),
        actual_predict_samples=sum(c['predict_samples'] for c in corrections),
        scope='saved successful event accounting only; no neural reexecution or changed plans/caps',
        P1_admission=False,P3_admission=False,efficacy_claim=False,compute_advantage=False)


def run(report, protocol, output):
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
    from experiments.spectral_reachability import d2p_p3_paired_accounting as accounting
    report,protocol,output=map(Path,(report,protocol,output))
    bindings={str(p.resolve()):digest(p) for p in (report,protocol)}
    p=json.loads(protocol.read_text())
    result=audit(json.loads(report.read_text()),p)
    if any(digest(path) != sha for path,sha in p['protocol']['sources'].items()):
        raise ValueError('original runner sources changed')
    output.mkdir(exist_ok=False)
    seal=create_seal(files=[Path(__file__),Path(accounting.__file__)],args_json=json.dumps(bindings),output_dir=output/'source_seal')
    result['inputs']=bindings
    result['original_current_sources_verified']=len(p['protocol']['sources'])
    if any(digest(path) != sha for path,sha in bindings.items()):
        raise ValueError('input files changed during audit')
    result['source_seal']=seal
    result['source_verification']=verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    with (output/'report.json').open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n')
    print(json.dumps({k:result[k] for k in ('status','corrected_rows','actual_predict_samples')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('report','protocol','output'):p.add_argument('--'+name,type=Path,required=True)
    run(**vars(p.parse_args()))
