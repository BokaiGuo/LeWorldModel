"""Join saved H1/H2 identities to the existing fixed164 physical join, CPU only.

This checks report metadata and inherited receipts, not tensors or fullstrict.
Prints a diagnostic to stdout; does not authorize execution or admission.
"""
import json

from experiments.spectral_reachability import agent_p5_fixed164_physical_model_join_20261029_v1 as base

SOURCES = {
    'h1': ('agent_p5_raw164_jepa_cuda_20261029_v1',
           '4f6bbed041f077818627e68157b0cfab99ddba2e38100827db10f46c4aa1b759',
           'e7ac4ef9eb7491f619de25b04c55f5e56560e2833a840dd5305f3db6207b3a5d',
           'all164_factual_H1_and80_pairs_verified', 'fixed164_actual15_conditioned_factual_H1_recorded'),
    'h2': ('agent_p5_raw164_jepa_h2_cuda_20261029_v2',
           '783b65f67899791b5697bfe0819ccfb3231d57212b9e3e85aefc02419459c0ea',
           'e5a561215bfe7b6f1c1c224e7661211e3f237fb16458a6a409f5a1da0fc08579',
           'all164_open_loop_H2_and80_pairs_verified', 'fixed164_root10_open_loop_H2_recorded'),
}


def join(physical, reports):
    rows, pairs = physical['rows'], physical['pairs']
    base.need(len(rows) == 164 and len(pairs) == 80, 'physical denominator')
    row_keys = ('index', 'root_slot', 'condition', 'task_id', 'selection_digest')
    pair_keys = ('slot', 'coord', 'h', 'minus_index', 'plus_index')
    for name, (receipt, report) in reports.items():
        stem, _, report_sha, receipt_status, report_status = SOURCES[name]
        base.need(receipt['schema'] == stem + '.consume' and receipt['status'] == receipt_status
                  and receipt['requests'] == 164 and receipt['pairs'] == 80
                  and receipt['report_sha256'] == report_sha, 'receipt binding')
        base.need(report['schema'] == stem + '.report' and not report['errors']
                  and report['status'] == report_status, 'report ABI')
        base.need(len(report['rows']) == 164 and len(report['pairs']) == 80, 'report denominator')
        for i, (reference, row) in enumerate(zip(rows, report['rows'], strict=True)):
            base.need(reference['index'] == i and all(reference[k] == row[k] for k in row_keys), 'row identity')
        for reference, pair in zip(pairs, report['pairs'], strict=True):
            base.need(all(reference[k] == pair[k] for k in pair_keys), 'pair identity')
            for idx in (pair['minus_index'], pair['plus_index']):
                base.need(rows[idx]['root_slot'] == pair['root_slot'] and
                          rows[idx]['condition'] == pair['condition'], 'pair root/condition')
    base.need(set(reports) == set(SOURCES), 'both horizons required')
    return {'status': 'saved_h1_h2_identity_join_only', 'rows': len(rows), 'pairs': len(pairs),
            'known_roots': 2, 'fresh_roots': 0, 'P1_admission': False, 'P5_admission': False,
            'new_neural_calls': 0, 'new_physics_calls': 0, 'full_strict_rerun': False,
            'scope': 'report identities and inherited physical join only; different history and horizon; no cross-model performance claim',
            'inputs': {k: {'receipt_sha256': v[1], 'report_sha256': v[2]} for k, v in SOURCES.items()}}


def main():
    physical = base.analyze(base.load())
    reports = {}
    for key, (stem, receipt_sha, report_sha, _, _) in SOURCES.items():
        receipt = base.read(base.ROOT / 'results' / (stem + '_consume_MAIN01.json'), receipt_sha)
        report = base.read(base.ROOT / 'results' / (stem + '_cuda_MAIN01') / 'report.json', report_sha)
        reports[key] = (receipt, report)
    print(json.dumps(join(physical, reports), sort_keys=True))


if __name__ == '__main__':
    main()
