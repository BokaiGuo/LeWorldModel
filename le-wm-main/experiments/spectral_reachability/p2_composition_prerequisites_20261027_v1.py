"""Read-only prerequisite inventory/arithmetic; no comparisons or stage admission.

Explicit historical assets only. Does not import a neural runtime, discover
outcomes, certify execution, rewrite receipts, or run physical/P3 consumers.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from experiments.spectral_reachability import p2_composition_admission_20261027_v1 as audit

PROJECT = Path(__file__).resolve().parents[2]
ASSETS = dict(
    bank='pusht_p2_matched_composition4_v1_20261027_cuda/p2_matched_report.json',
    bank_physical='pusht_p2_matched_physical4_v1_20261027_cuda/report.json',
    native='pusht_full_packet_compiler_20261027_cuda_v1/report.json',
    native_physical='pusht_full_packet_physical_20261027_cuda_v1/report.json',
    sra='p2_no_composition_sra_20261027_v2_prepared01/report.json',
    sra_physical='p2_no_composition_sra_physical_20261027_v2_cuda_main_v1/report.json',
    p1_model='pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda/report.json',
    p1_outcome='pusht_d2p_p1_outcome_physical_20261024_cuda/report.json')
SOURCES = ('p2_matched_composition_20261027.py','d2p_kinodynamic_program_search.py',
    'pusht_full_packet_compiler_20261027.py','pusht_d2p_metric_program_compiler.py',
    'd2p_p2_comparison_contract.py','d2p_stage_admission.py',
    'p2_no_composition_sra_20261027_v2.py')


def arithmetic():
    """Static schedule arithmetic, not measurement of FLOPs/backward kernels."""
    return dict(bank_per_root=dict(prefix_expansions=16,beam=4,terminal_candidates=4*16,
        planning_predict=16+4*16,flat_terminal_candidates=40,flat_planning_predict=40*2,
        replay_predict_per_arm=4,terminal_candidate_counts_equal=False),
        native_four_roots=dict(predict=4*(23+2*10*(4+2)),
            jacobian_api_calls=4*(2+2*10),scalar_output_vjp_directions=4*(2+2*10)*192,
            explicit_svd_calls_from_source=4*(2*2+2*10),
            matrix_rank_diagnostic_svd_calls_from_source=4*2,
            bounded_ridge_calls_from_source=4*(2+2*10),
            svd_scope='source-derived NumPy SVD invocations incl rank diagnostics, NOT instrumented runtime evidence'),
        sra_four_roots=dict(predict=4*(2+2+40*2+4),jacobian_api_calls=4,
            scalar_output_vjp_directions=4*192,svd=4*2),
        all_slot_example_only=dict(slots_per_root_arm=40,steps_per_slot=10,
            roots=4,arms=2,physical_steps=40*10*4*2,
            scope='conditional arithmetic, not an authorized experiment or new acceptance threshold'))


def inventory(project=PROJECT):
    reader = audit.EvidenceReader()
    payloads, observed = {}, {}
    project = Path(project).resolve()
    for name,relative in ASSETS.items():
        path = project/'results'/relative
        if not path.is_file():
            observed[name] = dict(path=str(path),availability='missing')
            continue
        data = reader.load(path)
        payloads[name] = data
        observed[name] = dict(path=str(path),availability='present',schema=data['schema'],
            status=data['status'],gates=data.get('gates'),sha256=reader.hashes[str(path)])
    details = {}
    if 'bank' in payloads:
        rows = payloads['bank']['rows']
        audit.require(len(rows)==4 and len({r['root_sha256'] for r in rows})==4,'four unique bank roots required')
        details['bank'] = [dict(root_index=r['root_index'],root_sha256=r['root_sha256'],
            arms=[dict(arm=a['arm'],attempts=len(a['attempts']),
                H2_candidates=sum(len(e['path'])==2 for e in a['attempts']),
                saved_witnesses=int(a['witness'] is not None),
                attempt_fields=sorted(a['attempts'][0]),planning_predict=a['planning_predict_samples'],
                replay_predict=a['replay_predict_samples']) for a in r['rows']]) for r in rows]
        details['root_metric_identity_count'] = len({r['declaration']['provenance']['metric_sha256'] for r in rows})
    if 'native' in payloads:
        data = payloads['native']
        details['native'] = dict(predict_from_phase_records=sum(r['predict_samples'] for r in data['predict']),
            report_predict=data['actual_predict_samples'],
            reported_jacobian_backwards=data['jacobian_backwards'],
            reported_scalar_output_vjp_directions=data['scalar_output_vjp_directions'],
            candidates=sum(len(a['candidates']) for p in data['plans'] for a in p['arms']),
            selection=data['protocol']['selection'],
            scope='two-stage spectral compiler exists; constants/full chart ablation, not integrated strict P2 comparison')
    for name in ('native_physical','sra_physical'):
        if name in payloads:
            data = payloads[name]
            details[name] = dict(rows=len(data['rows']),statuses=dict(Counter(r['status'] for r in data['rows'])),
                scope='coverage/integrity only; no effect sizes, rankings or comparison computation')
    if 'sra' in payloads:
        data = payloads['sra']
        details['sra'] = {k:data[k] for k in ('actual_predict_calls','completed_jacobian_calls',
            'completed_svd_calls','completed_scalar_output_vjp_directions','fair_total_budget_comparison')}
        details['sra']['candidate_receipts_scope'] = 'forty scored candidates per root, winner-only deployable witness'
    if 'bank' in payloads and 'sra' in payloads:
        bank = {r['root_sha256']:r for r in payloads['bank']['rows']}
        sra = {r['root_sha256']:r for r in payloads['sra']['rows']}
        audit.require(bank.keys()==sra.keys(),'bank/SRA stable root census differs')
        details['provenance'] = dict(same_root_set=True,
            equal_six_field_provenance_each_root=all(bank[k]['declaration']['provenance']==
                sra[k]['result']['witness']['receipt']['provenance'] for k in bank),
            scope='old strict validator requires one shared provenance; do not overwrite real per-arm source/protocol/bank hashes')
    for filename in SOURCES:
        reader.bind(project/'experiments/spectral_reachability'/filename)
    reader.unchanged()
    return dict(schema='p2_composition_prerequisites_20261027_v1.inventory',
        status='inventory_only_not_admission',observed_assets=observed,details=details,
        schedule_arithmetic=arithmetic(),input_and_source_sha256=reader.hashes,
        scope='explicit allowlist snapshot, not repository-wide absence proof or empirical comparison',
        new_neural_calls=0,new_physical_steps=0,P2_admission=False,
        fair_total_budget_comparison=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=PROJECT)
    result = inventory(parser.parse_args(argv).project)
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
    return 0  # Inventory success only, not a stage pass.


if __name__ == '__main__':
    raise SystemExit(main())
