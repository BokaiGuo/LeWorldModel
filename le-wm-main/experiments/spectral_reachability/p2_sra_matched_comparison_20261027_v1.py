"""Arithmetic-only, source-bound four-root SRA/CEM/random/MPPI comparison.

Saved calibration evidence only. Candidate budget is shared, total resource
budget is NOT matched. No model execution or composition/P2 admission.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p2_mppi_matched_comparison_20261027_v1 as previous
from experiments.spectral_reachability import p2_no_composition_sra_physical_20261027_v2 as sra
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG = 'p2_sra_matched_comparison_20261027_v1'
PROJECT = Path(__file__).resolve().parents[2]
SRA_RUN = 'p2_no_composition_sra_physical_20261027_v2_cuda_main_v1'
SRA_ARM = sra.sampling.ARM
BASE_ARMS = ('cem','random',previous.mppi.sampling.ARM)
ARMS = (SRA_ARM,*BASE_ARMS)
METRICS = previous.METRICS
STATE_COORDINATES = previous.STATE_COORDINATES
evidence = previous.evidence
require = evidence.require
index = previous.index


def load_inputs(project, reader):
    """Explicit completed calibration references; no result-directory discovery."""
    bundle = previous.load_inputs(project,reader)
    directory = Path(project)/'results'/SRA_RUN
    report,run_protocol = (reader.load(directory/name) for name in ('report.json','protocol.json'))
    require(report['schema'] == sra.TAG+'.report', 'SRA physical schema mismatch')
    prepared_dir = Path(run_protocol['prepared'])
    protocol,prepared = (reader.load(prepared_dir/name) for name in ('protocol.json','prepared.json'))
    reader.bindings(protocol['bindings'])
    reader.seals(prepared['source_seals'])
    _,verified,sealed,context = sra.load_prepared(prepared_dir)
    require(verified == protocol and sealed == prepared, 'SRA physical prepare changed')
    require(run_protocol['prepared_protocol_sha256'] == report['prepared_protocol_sha256'] ==
        prepared['protocol_sha256'] == sra.digest(protocol), 'SRA physical run/prepare binding mismatch')
    require(report['source_seals'] == prepared['source_seals']
        and run_protocol['plans'] == protocol['plans'] == context[5]
        and run_protocol['encode_terminals'] is True, 'SRA source/plan/encoding request mismatch')
    planning_dir = Path(protocol['planning'])
    planning,planning_protocol,planning_prepared = (reader.load(planning_dir/name)
        for name in ('report.json','protocol.json','prepared.json'))
    reader.bindings(planning_protocol['bindings'])
    reader.seals(planning_prepared['source_seals'])
    require(context[2] == bundle['compilation'], 'same source compilation required')
    require(planning_protocol['context']['normalization'] == bundle['groups'][0]['planning_protocol']['normalization']
        and planning_protocol['context']['objective'] == bundle['groups'][0]['planning_protocol']['objective'],
        'shared action normalization / original objective required')
    banks = {}
    saved_rows = index(planning['rows'],('root_sha256',))
    for ready in planning_protocol['context']['cases']:
        banks[ready['root_sha256']] = reader.load(planning_dir/f'root_{ready["root_index"]}_bank.json',
            protocol['bindings'][str(planning_dir/f'root_{ready["root_index"]}_bank.json')])
        row = saved_rows[ready['root_sha256'],]
        require(reader.load(planning_dir/f'root_{ready["root_index"]}.json') == row,
                'SRA persisted planning row differs')
        journal = planning_dir/f'root_{ready["root_index"]}_journal.jsonl'
        reader.bind(journal)
        events = [json.loads(line,object_pairs_hook=evidence.strict_object) for line in journal.read_text().splitlines()]
        require([e for e in events if e.get('kind') in ('jacobian','bounded_ridge_svd','mode_svd')] ==
                row['compilation_operations'], 'SRA planning compilation journal differs')
    bundle['sra'] = dict(directory=directory,report=report,run_protocol=run_protocol,
        protocol=protocol,prepared=prepared,plans=context[5],planning=planning,
        planning_protocol=planning_protocol,planning_prepared=planning_prepared,banks=banks)
    return bundle


def audit_artifacts(bundle,reader):
    """Bind all durable branch/encoding/journal/RGB artifacts and source archives."""
    for group in (*bundle['groups'],bundle['sra']):
        previous.audit_artifacts(group,reader)
    group = bundle['sra']
    report = group['report']
    required = {'all_fixed_H2','all_four_retained','encodings_complete','frozen_eval','model_unchanged',
        'no_branch_errors','physical_persisted','sources_and_inputs_unchanged','trace_matches_selected'}
    require(required <= report['gates'].keys(), 'SRA physical gates missing')
    physical = reader.load(group['directory']/'physical_report.json')
    require(physical['schema'] == sra.TAG+'.physical_before_encoding'
        and physical['planning_accounting'] == report['planning_accounting'] == sra.ACCOUNTING,
        'durable SRA overhead accounting changed')
    for name in ('P1_admission','P2_admission','physical_efficacy_verified',
                 'prediction_budget_matched','fair_total_budget_comparison'):
        require(report[name] is False, 'SRA claim boundary changed: '+name)


def validate_sra_planning(group,compilation):
    """Planner exporter is read-only and was already in its pre-execution seal."""
    plans = index(group['plans'],('root_sha256','method'))
    rows = index(group['planning']['rows'],('root_sha256',))
    ready = index(group['planning_protocol']['context']['cases'],('root_sha256',))
    require(set(group['banks']) == {key[0] for key in rows}, 'SRA bank root census mismatch')
    for (root,),row in rows.items():
        result,bank = row['result'],group['banks'][root]
        attempts = result['attempts']
        require(len(attempts) == 40 and all(e['completed'] is True for e in attempts),
                'all forty complete SRA candidates required')
        for i,event in enumerate(attempts):
            require(event['candidate_id'] == f'{SRA_ARM}_{i:02d}' and event['label'] == bank['labels'][i]
                and event['logits'] == bank['logits'][i] and event['commands'] == bank['commands'][i]
                and event['actions'] == sra.sampling.full20.raw_actions(bank['commands'][i]),
                'SRA bank/candidate action binding mismatch')
            evidence.finite(event['cost'])
        best = min(attempts,key=lambda e:e['cost'])
        require(best['candidate_id'] == result['selected_candidate_id']
            and best['cost'] == result['selected_model_cost']
            and best['actions'] == result['witness']['receipt']['actions'], 'SRA saved winner mismatch')
        provenance = sra.sampling.provenance_for(group['planning_protocol'],group['planning_prepared'],ready[root,])
        provenance['candidate_bank_sha256'] = sra.digest(bank)
        require(result['witness']['receipt']['provenance'] == provenance,
                'SRA bank/source/metric provenance mismatch')
    # Reconstruct only the supplied action mapping; the source-bound exporter
    # was checked by load_inputs -> physical.load_prepared -> inputs. This
    # additional check detects mutation between load and arithmetic comparison.
    actions = {(root,SRA_ARM):np.asarray(row['result']['witness']['receipt']['actions'])
               for (root,),row in rows.items()}
    rebound = sra.plans_from_saved(group['planning'],group['planning_protocol'],
        group['planning_prepared'],compilation,actions)
    require(index(rebound,('root_sha256','method')) == plans, 'SRA physical/planning witness mismatch')
    require(group['protocol']['planning_accounting'] == group['report']['planning_accounting'] == sra.ACCOUNTING,
            'SRA compile/Jacobian/SVD overhead accounting mismatch')
    return rows


def build_rows(bundle):
    # Prior validator independently recomputes all saved official/pose/Q scores
    # and validates full20 candidate/predict/encoder budgets for the three arms.
    rows = previous.build_rows(bundle)
    group = bundle['sra']
    planned = validate_sra_planning(group,bundle['compilation'])
    compiled = index(bundle['compilation']['rows'],('root_sha256',))
    plans = index(group['plans'],('root_sha256','method'))
    branches = index(group['report']['rows'],('root_sha256','method'))
    encodings = index(group['report']['encodings'],('root_sha256','method'))
    require(plans.keys() == branches.keys() == encodings.keys() and len(plans) == 4,
            'all four unique SRA branches/encodings required')
    for key,plan in plans.items():
        root,method = key
        require(method == SRA_ARM and branches[key]['root_index'] == plan['root_index'],
                'SRA stable root index/arm mismatch')
        result = planned[root,]['result']
        row = previous.validate_saved_branch(branches[key],plan,encodings[key],
            compiled[root,]['compiled'],result['selected_model_cost'])
        # The generic state validator's historical 84 encoder default applies
        # only to sampling arms. Replace it with verified actual SRA accounting.
        row['budget'].update(compilation_predict_samples=4,total_predict_samples=88,
            action_encoder_calls=88,processed_context_tokens=264,jacobian_calls=1,
            svd_calls=2,scalar_output_vjp_directions=192)
        row['saved_planning_wall_time_s'] = evidence.finite(planned[root,]['actual_root_wall_time_s'])
        rows.append(row)
    report = group['report']
    require(report['selected_path_env_step_attempts'] == report['selected_path_env_step_returns'] == 40
        and report['attempted_terminal_encode_samples'] == 4, 'SRA physical/encode totals mismatch')
    for row in rows:
        if row['arm'] != SRA_ARM:
            # No extra compile predicts in these recorded dispatch ledgers;
            # unmeasured upstream work is not asserted to be zero.
            row['budget'].update(compilation_predict_samples=0,total_predict_samples=84,
                jacobian_calls=None,svd_calls=None,scalar_output_vjp_directions=None)
    return rows


def compare_rows(rows):
    lookup = index(rows,('root_sha256','arm'))
    roots = sorted({r['root_sha256'] for r in rows})
    require(len(roots) == 4 and set(lookup) == {(r,a) for r in roots for a in ARMS},
            'four roots / four arms required; no filtering')
    paired = []
    common_units = dict(full_H2_candidates=40,planning_predict_samples=80,replay_predict_samples=4,
        physical_env_steps=10,terminal_encode_samples=1)
    for root in roots:
        arms = {arm:lookup[root,arm] for arm in ARMS}
        for field in ('root_binding_sha256','action_domain','metric_sha256','goal_vectors_sha256','physical_identity'):
            require(all(arms[a][field] == arms[SRA_ARM][field] for a in ARMS),
                    'cross-arm '+field+' mismatch')
        for arm,row in arms.items():
            require(all(row['budget'][k] == v for k,v in common_units.items()),
                    'shared candidate/replay/physical budget mismatch')
            expected = dict(compilation_predict_samples=4,total_predict_samples=88,action_encoder_calls=88,
                processed_context_tokens=264,jacobian_calls=1,svd_calls=2,scalar_output_vjp_directions=192)
            if arm != SRA_ARM:
                expected = dict(compilation_predict_samples=0,total_predict_samples=84,action_encoder_calls=84,
                    processed_context_tokens=252,jacobian_calls=None,svd_calls=None,scalar_output_vjp_directions=None)
            require(all(row['budget'][k] == v for k,v in expected.items()), 'cost overhead accounting mismatch')
        paired.append(dict(root_sha256=root,arms=deepcopy(arms),
            sra_minus={a:{m:arms[SRA_ARM][m]-arms[a][m] for m in METRICS} for a in BASE_ARMS},
            sra_minus_state_absolute_error={a:(np.asarray(arms[SRA_ARM]['state_absolute_error'])-
                np.asarray(arms[a]['state_absolute_error'])).tolist() for a in BASE_ARMS}))
    return dict(schema=TAG+'.report',status='saved_calibration_comparison_completed',evidence_integrity_valid=True,
        paired_roots=4,branches=16,rows=paired,
        paired_summaries={a:{m:previous.describe([r['sra_minus'][a][m] for r in paired],
            higher_better=m in ('official_success','object_only_success')) for m in METRICS} for a in BASE_ARMS},
        paired_state_error_summaries={a:{c:previous.describe([r['sra_minus_state_absolute_error'][a][i] for r in paired])
            for i,c in enumerate(STATE_COORDINATES)} for a in BASE_ARMS},
        state_absolute_error_coordinates=list(STATE_COORDINATES),
        state_error_scope='absolute raw seven-state coordinate errors; raw angle not circular; mixed units, not pooled',
        arm_means={a:{m:float(np.mean([lookup[root,a][m] for root in roots])) for m in METRICS} for a in ARMS},
        official_successes={a:sum(lookup[root,a]['official_success'] for root in roots) for a in ARMS},
        cost_accounting=dict(per_root={a:deepcopy(lookup[roots[0],a]['budget']) for a in ARMS},
            four_root_total_predict_samples={a:sum(lookup[root,a]['budget']['total_predict_samples'] for root in roots) for a in ARMS},
            sra_additional_predict_samples_per_root=4,sra_additional_predict_samples_four_roots=16,
            sra_total_predict_over_reference_ratio=88/84,sra_jacobian_calls=4,sra_svd_calls=8,
            sra_scalar_output_vjp_directions=768,sra_processed_context_token_samples=1056,
            reference_processed_context_token_samples_per_arm=1008,
            array_verification=deepcopy(sra.ACCOUNTING['verification_work']),
            verification_scope='8 SVD/4 bounded solves per source-bound SRA physical.inputs invocation, outside recorded planning; repeated verification is not free',
            reference_zero_compile_scope='zero extra compilation predicts in saved sampling dispatch ledger only; not zero upstream bank/encoding cost',
            unknown_is_not_zero=True),
        delta_convention='SRA minus named reference; lower better except success; exact ties, no new thresholds',
        uncertainty_scope='sample SD/range over four already-used calibration roots; no CI, p-value or independent-population inference',
        budget_scope='80 candidate predicts = 40 complete H2 programs +4 replay shared; SRA adds4 compile predicts/root and Jacobian/SVD work; NOT equal total budget',
        unmeasured_or_unmatched=deepcopy(sra.ACCOUNTING['unmeasured'])+['proposal distributions','shared end-to-end wall-time cap'],
        official_MPPI=False,mppi_label='MPPI-inspired shooting heuristic, not official MPPI',
        no_composition_arm=True,composition_admission=False,P1_admission=False,P2_admission=False,
        fair_total_budget_comparison=False,prediction_budget_matched=False,physical_efficacy_verified=False,
        composition_scope='No composition arm or matched composition ablation here; cannot identify composition benefit or open P2',
        new_neural_calls=0,new_physical_steps=0,future_heldout_results_opened=False)


def markdown(result):
    lines = ['# SRA / CEM / random / MPPI-inspired saved calibration comparison','',
        f'Status: {result["status"]}; P2_admission=false; composition_admission=false.','']
    if not result['evidence_integrity_valid']:
        return '\n'.join(lines+result['integrity_errors'])+'\n'
    lines += ['Four known roots; stable identity join; arithmetic only, no new neural/physics work.',
        'MPPI-inspired shooting, not official MPPI. No equal-total-budget claim.',
        '80 candidate predicts = 40 complete H2 programs; replay4 for all; SRA adds4 compile predicts/root + Jacobian/SVD.', '',
        '| Arm | Mean pose | Mean official state distance | Mean real encoder Q MSE | Official success | Predict total (four roots) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        values = result['arm_means'][arm]
        lines.append(f'| {arm} | {values["object_pose_cost"]:.6f} | {values["official_state_distance"]:.6f} | {values["real_encoder_q_mse"]:.8f} | {result["official_successes"][arm]}/4 | {result["cost_accounting"]["four_root_total_predict_samples"][arm]} |')
    lines += ['', 'SRA minus reference: negative favors SRA on costs; not a method-selection rule.','',
        '| Reference | Metric | Mean delta | Sample SD | SRA W/T/L |', '| --- | --- | ---: | ---: | --- |']
    for arm in BASE_ARMS:
        for metric in ('object_pose_cost','official_state_distance','real_encoder_q_mse'):
            s = result['paired_summaries'][arm][metric]
            lines.append(f'| {arm} | {metric} | {s["mean"]:.8f} | {s["sample_sd"]:.8f} | {s["left_wins"]}/{s["ties"]}/{s["left_losses"]} |')
    lines += ['',result['uncertainty_scope'],result['composition_scope'],
        'SRA planning:352 predict,4 Jacobian,8 SVD,768 mathematical scalar-output VJP directions (not measured backward API calls).',
        'Read-only bank verification adds8 SVD/4 bounded solves per inputs call, separately accounted, not neural work.',
        'FLOPs, upstream RGB/bank work, backward-engine details and common wall-time budget remain unmeasured.']
    return '\n'.join(lines)+'\n'


def run(project,output):
    project,output = Path(project).resolve(),Path(output).resolve()
    require(TAG in output.name, 'new dedicated tagged comparison output required')
    output.mkdir(exist_ok=False)
    reader = evidence.EvidenceReader()
    own = [Path(__file__),project/'tests'/f'test_{TAG}.py',project/'innerdocs'/f'{TAG}.md',
        Path(previous.__file__),Path(sra.__file__),Path(sra.sampling.__file__),Path(evidence.__file__),
        Path(previous.labels.__code__.co_filename),Path(previous.metric.__code__.co_filename)]
    try:
        for path in own: reader.bind(path)
        seal = create_seal(files=own,args_json=json.dumps(dict(scope='read-only known-root SRA arithmetic audit',
            P2_admission=False,composition_admission=False)),output_dir=output/'source_seal')
        bundle = load_inputs(project,reader)
        audit_artifacts(bundle,reader)
        result = compare_rows(build_rows(bundle))
        reader.unchanged()
        result['source_seal'] = seal
        result['source_verification'] = verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    except (ValueError,OSError,KeyError,TypeError,IndexError) as exc:
        result = dict(schema=TAG+'.report',status='invalid_evidence',evidence_integrity_valid=False,
            integrity_errors=[f'{type(exc).__name__}: {exc}'],P2_admission=False,composition_admission=False,
            fair_total_budget_comparison=False,new_neural_calls=0,new_physical_steps=0)
    result['input_and_source_sha256'] = reader.hashes
    result['hash_scope'] = 'audit-time observed bytes and historical archives; not independent proof of actual execution, timing or prospective independence'
    evidence.write_new(output/'report.json',result)
    with (output/'report.md').open('x') as stream: stream.write(markdown(result))
    print(json.dumps(dict(status=result['status'],P2_admission=False,composition_admission=False,
        bound_files=len(reader.hashes),output=str(output))))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=PROJECT)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    return 0 if run(args.project,args.output)['evidence_integrity_valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
