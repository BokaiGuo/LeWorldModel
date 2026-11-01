"""Read-only saved-record audit for fixed16 same-action prediction diagnostics.

Checks internally saved identities and arithmetic, not independent model
execution or authenticity of a report's claimed hardware and call counts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from experiments.spectral_reachability import codex_p1_null_same_action_error_cuda_20261030_v1 as exp

SCHEMA = 'codex_p1_null_same_action_saved_audit_20261030_v1'
FIELDS = ('index', 'task_id', 'role', 'slot', 'root_slot', 'condition', 'arm',
          'physical_root_sha256', 'raw15_sha256', 'images', 'native_status',
          'duplicate_action_task_group')


def check(report, prepared, prepared_path, prepare_sha):
    need = exp.require
    need(type(report) is dict and type(prepared) is dict and
         prepared.get('schema') == exp.NAME+'.prepare' and
         report.get('schema') == exp.NAME+'.report' and report.get('fixed_rows') == 16 and
         prepared.get('fixed_rows') == 16 and
         report.get('prepared') == str(prepared_path) and report.get('prepare_sha256') == prepare_sha,
         'report or preparation identity')
    for key in ('training', 'physical_execution', 'P1_admission', 'independent_confirmation', 'freshness'):
        need(report.get(key) is False and prepared.get(key) is False, 'scientific boundary: '+key)
    model_hash = prepared.get('model_hash')
    need(report.get('model_before') == report.get('model_after') == model_hash and
         type(model_hash) is dict and set(model_hash) == {'original_state_dict', 'all_buffers'} and
         all(type(v) is str and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
             for v in model_hash.values()),
         'model identity')
    need(report.get('calls') == {'encoder': 64, 'projector': 64, 'predictor': 80, 'action_encoder': 80},
         'recorded call counts')
    host = report.get('host')
    need(type(host) is dict and host.get('device') == 'cuda:0', 'recorded CUDA device')
    rows, source = report['rows'], prepared['rows']
    need(type(rows) is list and type(source) is list and len(rows) == len(source) == 16,
         'fixed sixteen rows')
    for i, (row, original) in enumerate(zip(rows, source, strict=True)):
        need(type(row) is dict and type(original) is dict and type(row.get('index')) is int and
             row['index'] == original['index'] == i and
             all(row[k] == original[k] for k in FIELDS), 'row/source identity')
        raw, mean, std, history, _ = exp.action_inputs(original)
        tokens = np.asarray(row['normalized_tokens'], dtype=np.float32)
        expected = (raw - np.tile(mean, 5)) / np.tile(std, 5)
        need(tokens.shape == (3, 10) and np.isfinite(tokens).all() and
             np.allclose(tokens, expected, rtol=1e-6, atol=1e-6), 'normalized tokens')
        saved_h = np.asarray(row['open_action_contexts'], dtype=np.float32)
        need(saved_h.shape == (3, 3, 10) and
             np.array_equal(saved_h, exp.action_contexts(history, tokens)), 'action contexts')
        comparisons = row['comparisons']
        need(type(comparisons) is list and len(comparisons) == 3 and
             [c['step'] for c in comparisons] == [5, 10, 15], 'comparison steps')
        need(all(type(c['step']) is int for c in comparisons), 'integer step identities')
        need(all(type(c[k]) is float for c in comparisons for k in ('open_mse', 'open_l2')) and
             all(type(c[k]) is float for c in comparisons[1:] for k in ('refreshed_mse', 'refreshed_l2')),
             'numeric error types')
        actual = [row['actual_root_latent']] + [c['actual_latent'] for c in comparisons]
        opens = [c['open_prediction'] for c in comparisons]
        refreshed = [c['refreshed_prediction'] for c in comparisons[1:]]
        expected_comparisons = exp.compare_latents(actual, opens, refreshed)
        need(comparisons == expected_comparisons, 'comparison values or error arithmetic')
    return {'schema': SCHEMA+'.result', 'status': 'saved_consistency_passed',
            'rows': 16, 'time_points_per_row': [5, 10, 15],
            'report_claimed_cuda_not_independently_attested': True,
            'neural_reexecution': False, 'physical_reexecution': False,
            'P1_admission': False, 'independent_confirmation': False, 'freshness': False}


def read_bounded(path, digest, kind):
    exp.native.actual_hash(digest)
    p = Path(path)
    exp.require(p.is_absolute() and p.parent == exp.ROOT/'results' and p.parent == p.parent.resolve()
                and p.name.startswith(exp.NAME+'_'+kind+'_') and p.is_file() and not p.is_symlink(),
                'canonical saved '+kind+' path')
    with p.open('rb') as stream:
        data = stream.read(64*1024**2 + 1)
    exp.require(len(data) <= 64*1024**2 and hashlib.sha256(data).hexdigest() == digest,
                'saved '+kind+' digest')
    value = json.loads(data)
    exp.require(data == exp.encode_json(value), 'saved '+kind+' not canonical')
    return value


def audit(report_path, report_sha, prepared_path, prepare_sha):
    own = exp.sha(Path(__file__).resolve())
    report = read_bounded(report_path, report_sha, 'run')
    prepared = read_bounded(prepared_path, prepare_sha, 'prepare')
    exp.require(prepared == exp.material(), 'current prepared sources differ')
    result = check(report, prepared, prepared_path, prepare_sha)
    exp.require(read_bounded(report_path, report_sha, 'run') == report and
                read_bounded(prepared_path, prepare_sha, 'prepare') == prepared and
                exp.sha(Path(__file__).resolve()) == own, 'saved files or auditor drift')
    return {**result, 'report_sha256': report_sha, 'prepare_sha256': prepare_sha,
            'auditor_sha256': own}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--report-sha', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--prepare-sha', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.report, args.report_sha, args.prepared, args.prepare_sha),
                         sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError) as exc:
        print(json.dumps({'status': 'blocked_invalid', 'reason': str(exc),
                          'P1_admission': False}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
