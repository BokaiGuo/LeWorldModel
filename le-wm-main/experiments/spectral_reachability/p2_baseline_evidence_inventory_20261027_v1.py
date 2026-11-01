"""Read-only, stdlib-only inventory of saved evidence, NOT a reproduction runner.

No model/environment imports, recursive result discovery, checkpoint reads or writes.
Exit 0 = inventory structurally valid (scientific gaps may remain); 2 = invalid input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from .d2p_reported_baselines import validate_record

NAME = 'p2_baseline_evidence_inventory_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
CAP = 16 * 1024 * 1024
TOTAL_CAP = 96 * 1024 * 1024
EXPORT = 'results/p2_reported_baseline_table_20261027_v3/export/'
COMPARISON = 'results/p2_gradient_mpc_matched_comparison_20261027_v1_audit01/report.json'
GRASP = 'results/p2_grasp_h1_saved_outcome_audit_20261027_v1_main_verify01/report.json'
PROTOCOL = 'results/p2_full20_sampling_20261027_v2_cuda_v1/protocol.json'
METHODS = {
    'CEM': ('cem', 'p2_full20_sampling_20261027_v2', 'p2_full20_physical_20261027_v2_cuda_v1'),
    'MPPI': ('mppi_inspired_shooting', 'p2_full20_mppi_20261027_v1', 'p2_full20_mppi_physical_20261027_v1_cuda_main_v1'),
    'gradient MPC': ('internal_projected_gradient_mpc', 'p2_full20_gradient_mpc_20261027_v1', 'p2_gradient_mpc_physical_20261027_v1_cuda_main_v1'),
    'random shooting': ('random', 'p2_full20_sampling_20261027_v2', 'p2_full20_physical_20261027_v2_cuda_v1'),
    'no-composition SRA': ('joint_h2_no_composition_sra', 'p2_no_composition_sra_20261027_v2', 'p2_no_composition_sra_physical_20261027_v2_cuda_main_v1'),
}
LABELS = {
    'CEM': 'local full20 open-loop CEM; not paper protocol reproduction',
    'MPPI': 'clipped-Gaussian MPPI-inspired, NOT official MPPI',
    'gradient MPC': 'projected full-H2 open-loop action gradient, NOT receding MPC',
    'GN': 'Reacher P0 shared D2P/DLS solver equivalence, NOT independent GN-MPC',
    'GRASP': 'local official-mechanism subset H1 vs Action-Adam; NOT full paper reproduction',
    'Hi-LeWM': 'paper_reported only; no verified local hierarchical reproduction',
    'LeWM': 'local frozen backbone != Table10 protocol reproduction',
    'PlanWAM': 'paper_reported driving context only, NOT PushT baseline',
}
GAPS = {
    'CEM': ['Bind same feedback schedule and full resource ledger before D2P superiority claims.'],
    'MPPI': ['Audit importance weighting/noise/control-cost contract before calling this official MPPI.'],
    'gradient MPC': ['Add real observe/replan schedule for a receding-MPC claim; retain 156 AD/update costs and backend differences.'],
    'GN': ['Create independent multistart GN-MPC arm with shared goal/action/feedback/cost contract; do not relabel D2P equivalence as a win.'],
    'GRASP': ['Retain all eight slots including four out-of-box rejections; only one full-path pair, not four-root efficacy.',
              'H1 relaxed optimization and real-H3 diagnostic/execution are different horizons; no H2 equal-budget comparison.'],
    'Hi-LeWM': ['No local high-level checkpoint, macro bank or hierarchical evaluation receipts verified; training is outside this task.'],
    'LeWM': ['Pin Table10 evaluated checkpoint/trial IDs, gradient budgets and uncertainty definition; local LeWM use is not reproduction.'],
    'PlanWAM': ['Driving assets, checkpoint and scorer receipts unverified; keep NAVSIM/HUGSIM separate from PushT.'],
}


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite numeric metric required')
    return value


def sha_id(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('invalid stable root/hash')
    return value


def keyed(rows, key='root_sha256'):
    result = {}
    for row in rows:
        ident = sha_id(row[key])
        if ident in result:
            raise ValueError('duplicate stable root')
        result[ident] = row
    if not result:
        raise ValueError('empty root cohort')
    return result


def budget_check(budget):
    fields = ('planning_predict_samples', 'replay_predict_samples',
              'compilation_predict_samples', 'total_predict_samples')
    vals = [budget[k] for k in fields]
    if any(type(v) is not int or v < 0 for v in vals) or sum(vals[:3]) != vals[3]:
        raise ValueError('prediction budget arithmetic mismatch')
    return dict(budget)  # null costs stay unknown, never coerced to zero


class Reader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.bytes = 0
        self.receipts = {}

    def raw(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('input escapes project root')
        if path.stat().st_size > CAP:
            raise ValueError('input exceeds byte cap')
        with path.open('rb') as stream:
            data = stream.read(CAP + 1)
        self.bytes += len(data)
        if len(data) > CAP or self.bytes > TOTAL_CAP:
            raise ValueError('inventory byte cap exceeded')
        self.receipts[relative] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        return data

    def read(self, relative):
        def reject(value):
            raise ValueError('nonfinite JSON: ' + value)
        def unique(pairs):
            obj = {}
            for key, value in pairs:
                if key in obj:
                    raise ValueError('duplicate JSON key')
                obj[key] = value
            return obj
        return json.loads(self.raw(relative), parse_constant=reject, object_pairs_hook=unique)


def reported(reader):
    manifest = reader.raw(EXPORT + 'SHA256SUMS').decode()
    pins = {line.split('  ', 1)[1]: line.split('  ', 1)[0]
            for line in manifest.splitlines() if '  ' in line}
    report = reader.read(EXPORT + 'report.json')
    readiness = reader.read(EXPORT + 'comparison_readiness.json')
    for file in ('report.json', 'comparison_readiness.json'):
        relative = EXPORT + file
        if reader.receipts[relative]['sha256'] != pins.get(relative):
            raise ValueError('reported export SHA mismatch: ' + file)
    if report['local_reproduction'] is not False or report['cross_paper_ranking_allowed'] is not False:
        raise ValueError('reported/reproduced boundary violated')
    seen, groups = set(), {}
    for row in report['rows']:
        if row['id'] in seen:
            raise ValueError('duplicate paper cell')
        seen.add(row['id'])
        record = validate_record(row['record'])
        sha_id(row['source']['html_sha256'])
        if row['classification'] != 'context_only':
            raise ValueError('paper cell promoted to local comparison')
        groups.setdefault(record['paper'], []).append(row)
    counts = {k: len(v) for k, v in groups.items()}
    if counts != report['counts']:
        raise ValueError('reported cell counts mismatch')
    return groups, readiness['gaps']


def local_rows(comparison, physical, arm):
    saved = keyed([r for r in physical['rows'] if r['method'] == arm])
    out = []
    for root, row in sorted(keyed(comparison['rows']).items()):
        cell = row['arms'][arm]
        if cell['root_sha256'] != root or root not in saved:
            raise ValueError('root/arm join mismatch')
        branch = saved[root]
        if branch['context_identity'] != cell['physical_identity']:
            raise ValueError('physical context identity mismatch')
        if (branch['full_horizon_completed'] is not True or
                branch['completed_commands'] != 10 or branch['planned_commands'] != 10):
            raise ValueError('not a complete H2 endpoint')
        for key, value in [('object_pose_cost', branch['pose']['pose_cost']),
                           ('official_state_distance', branch['official']['official_state_distance'])]:
            if finite(cell[key]) != finite(value):
                raise ValueError('saved metric mismatch')
        # The comparison table explicitly encodes Bernoulli labels as integer 0/1.
        label = cell['official_success']
        if type(label) not in (bool, int) or label not in (0, 1):
            raise ValueError('success must be boolean or integer 0/1')
        if type(branch['official']['official_success']) is not bool:
            raise ValueError('physical success must be boolean')
        if bool(label) is not branch['official']['official_success']:
            raise ValueError('success label mismatch')
        out.append(dict(root_sha256=root, environment='PushT (saved local protocol)',
            horizon={'model_tokens': 2, 'primitive_steps': 10},
            metric_sha256=sha_id(cell['metric_sha256']),
            goal_vectors_sha256=sha_id(cell['goal_vectors_sha256']),
            action_domain=cell['action_domain'], backend=cell['backend_policy'],
            budget=budget_check(cell['budget']),
            metrics={k: finite(cell[k]) for k in ('predicted_q_mse', 'real_encoder_q_mse',
                     'object_pose_cost', 'official_state_distance')},
            official_success=cell['official_success']))
    if set(saved) != {r['root_sha256'] for r in out}:
        raise ValueError('extra physical roots')
    return out


def build_inventory(root=ROOT):
    reader = Reader(root)
    groups, paper_gaps = reported(reader)
    comparison = reader.read(COMPARISON)
    protocol = reader.read(PROTOCOL)
    grasp = reader.read(GRASP)
    if comparison['evidence_integrity_valid'] is not True:
        raise ValueError('upstream comparison integrity not valid')
    inventory, missing = [], []
    for name in ('CEM', 'MPPI', 'gradient MPC', 'GN', 'GRASP', 'Hi-LeWM', 'LeWM', 'PlanWAM',
                 'random shooting', 'no-composition SRA'):
        rows, assets, gaps = [], [], list(GAPS.get(name, []))
        if name in METHODS:
            arm, code, result = METHODS[name]
            assets = ['experiments/spectral_reachability/' + code + '.py', 'results/' + result + '/report.json']
            rows = local_rows(comparison, reader.read(assets[1]), arm)
            expected = comparison['input_and_source_sha256'].get(str(reader.root / assets[1]))
            if reader.receipts[assets[1]]['sha256'] != expected:
                raise ValueError('physical report SHA differs from comparison binding')
            gaps += comparison['unmeasured_or_unmatched']
        elif name == 'GRASP':
            assets = ['experiments/spectral_reachability/p2_grasp_h1_cuda_runner_20261027_v1.py', GRASP]
            counts = {}
            for pair in keyed(grasp['pairs']).values():
                kind = pair['classification']
                counts[kind] = counts.get(kind, 0) + 1
                delta = pair['grasp_minus_action_adam_pose_cost']
                if kind == 'full-path':
                    finite(delta)
                elif delta is not None:
                    raise ValueError('unmatched GRASP pair has a delta')
            if counts != grasp['pair_counts']:
                raise ValueError('GRASP denominator mismatch')
            rows = grasp['pairs']
            gaps += grasp['gaps']
        elif name == 'GN':
            assets = ['experiments/spectral_reachability/run_reacher_d2p_p0_goal_progress.py']
        for path in assets:
            if path.endswith('.py'):
                try:
                    reader.raw(path)
                except FileNotFoundError:
                    missing.append(path)
        paper_rows = groups.get(name, [])
        # CEM cells also occur in other papers; preserve their source identities.
        if name == 'CEM':
            paper_rows = [r for values in groups.values() for r in values if r['method'] == 'CEM']
        gaps += paper_gaps.get(name, [])
        inventory.append(dict(method=name, identity=LABELS.get(name, name), assets=assets,
            paper_reported_cells=len(paper_rows), paper_records=paper_rows,
            local_evidence=rows, paper_protocol_reproduced=False,
            local_environment='PushT' if name in METHODS or name == 'GRASP' else 'Reacher code only' if name == 'GN' else 'unknown',
            local_checkpoint_certification='unknown: inventory does not rehash weights or verify transitive seals',
            local_budget=grasp['budget'] if name == 'GRASP' else 'see per-root budget' if name in METHODS else 'unknown',
            local_roots_scope='known calibration roots only; no held-out inference' if rows else 'no verified local cohort',
            direct_paper_comparison=False, total_compute_matched=False,
            missing_or_unmatched=gaps))
    checkpoint = {p: h for p, h in protocol['bindings'].items() if p.endswith(('weights.pt', 'config.json'))}
    if not checkpoint:
        raise ValueError('missing checkpoint binding')
    for digest in checkpoint.values():
        sha_id(digest)
    return dict(schema=NAME, status='inventory_valid_scientific_gaps_open' if not missing else 'missing_code_assets',
        scientific_admission=False, cross_paper_ranking_allowed=False, new_neural_calls=0,
        new_physical_steps=0, checkpoint_bindings_recorded_not_rehashed=checkpoint,
        checkpoint_scope='shared sampling protocol only; not independent per-method checkpoint certification',
        source_scope='saved report arithmetic and identities, NOT full transitive source/array seal verification',
        environment_scope='local PushT H2; GN Reacher; paper environments remain in each record',
        missing_assets=missing, methods=inventory, inputs=reader.receipts,
        input_bytes=reader.bytes, output_policy='stdout only; no files created or overwritten')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        result = build_inventory(args.root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'schema': NAME, 'status': 'invalid_or_missing_evidence', 'error': str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if not result['missing_assets'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
