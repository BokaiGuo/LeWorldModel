"""Predeclare six future candidates, not a comparison of saved outcomes.

Standard library, metadata only. stdout is the machine-readable planning artifact.
No runner, retuning, cost-unit conversion, neural or physics execution exists here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

STEM = 'moving_equal_budget_prospective_queue_20261027_v1'
MATRIX_SHA = '721d978d29516f8ecf1d1056e159736ef231723e6849490a83678bd76baa1ebc'
SOURCE_SHA = '229f8f6116bc5771d3738b5ecafe52026a00d439e3d63b0e5e9efe2f08fe94fb'
COMPARISON_SHA = '96cf27b143bb1a839edf7bc1e4de02f4a97e433c6475c4c731c78af76300d32e'
ARMS = ('moving_gradient_open_loop', 'CEM', 'kernel_faithful_mppi_nominal',
        'fixed_H2_kinodynamic_beam16x4', 'GRASP_H3_bounded', 'joint_H2_spectral_packet_mapped')
TASKS = ('a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b',
         '101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74')
COUNTERS = ('predict', 'forward_samples', 'action_encoder', 'pred_projector', 'backward',
            'grad_API', 'jacobian', 'jacobian_API', 'svd', 'bounded_solve', 'predict_API',
            'scalar_output_VJP_directions')
SCALARS = ('AD_forward_in_predict', 'mathematical_reverse_directions', 'scalar_loss_reverse_sweeps')
EXTRA = ('image_encoder', 'image_projector', 'physics_step', 'public_step',
         'pipeline_walltime_seconds', 'CPU_work')
CONSTRAINTS = ('population', 'history', 'horizon', 'action_support', 'proposal_support',
               'replay_feedback', 'selection', 'batching_device_precision',
               'encoding_sharing', 'pipeline_scope', 'budget_scope', 'derivative_semantics')
UNITS = {k: ('attempted_completed_' + k + '_operations') for k in COUNTERS}
UNITS.update({k: k for k in SCALARS})
UNITS.update(image_encoder='image_encoder_operations', image_projector='image_projector_operations',
             physics_step='internal_physics_steps', public_step='public_environment_steps',
             pipeline_walltime_seconds='seconds_complete_pipeline_per_method', CPU_work='typed_CPU_operation_vector')
RULES = [
    'No scalar total or cross-unit addition/conversion; equality is a complete componentwise contract.',
    'AD_forward_in_predict is included in predict; action_encoder/pred_projector are overlapping module meters.',
    'backward/grad_API/J/J_API/VJP are separate overlapping descriptions, not additive work.',
    'scalar-loss reverse sweeps are NOT scalar-output VJP directions; mathematical_reverse_directions alone is ambiguous.',
    'Predict API calls, predictor forwards and forward samples retain distinct units and batching scopes.',
    'CPU SVD/solve and reconstruction verification are distinct from HOST CUDA neural operations.',
    'Physics/public steps, image encoder/projector, and full-pipeline walltime retain separate units.',
    'Cohort latency/physics/RGB may not be divided among methods; inherited encoding is shared once, not six times.',
    'Equal caps or equal predict alone do not establish equal work; no padding, truncation, repeats or retuning to force equality.',
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encode(x):
    return (json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    require(isinstance(value, str) and len(value) == 64 and
            all(c in '0123456789abcdef' for c in value), 'sha256 required')
    return value


def unknown(reason):
    return dict(status='unestablished', value=None, reason=reason)


def known(value, basis):
    return dict(status='actual', value=deepcopy(value), basis=basis)


def counter(value):
    require(isinstance(value, dict) and set(value) == {'attempted', 'completed'}, 'counter phases')
    require(all(type(v) is int and v >= 0 for v in value.values()), 'nonnegative integer counter')
    require(value['completed'] <= value['attempted'], 'completed exceeds attempted')


def cell(c, kind):
    require(isinstance(c, dict) and c.get('status') in
            ('actual', 'unestablished', 'missing', 'partial', 'unmeasured') and 'value' in c, 'typed cell')
    if c['status'] != 'actual':
        require(c['value'] is None and bool(c.get('reason')), 'missing is null with reason, never zero')
    else:
        require(c['value'] is not None and bool(c.get('basis')), 'actual needs value and basis')
        if kind == 'counter':
            counter(c['value'])
        elif kind == 'scalar':
            v = c['value']
            require(type(v) in (int, float) and math.isfinite(v) and v >= 0, 'finite nonnegative scalar')
        elif kind == 'constraint':
            require(isinstance(c['value'], (dict, str, list)) and bool(c['value']), 'nonempty constraint')
            encode(c['value'])
        else:
            require(kind == 'CPU', 'unknown cell kind')
            require(isinstance(c['value'], dict) and bool(c['value']), 'typed CPU operation vector')
            for v in c['value'].values():
                counter(v)
    return deepcopy(c)


def kind(axis):
    return 'CPU' if axis == 'CPU_work' else 'scalar' if axis in SCALARS + ('pipeline_walltime_seconds',) else 'counter'


def assess(contracts):
    """Strict complete-vector equality; fixtures are not executable budget authority.

    Production assemble only supplies pinned sealed references, never user overrides.
    """
    require(isinstance(contracts, list) and len(contracts) == 6, 'six contracts required')
    gaps = []
    for section, axes in (('costs', tuple(UNITS)), ('constraints', CONSTRAINTS)):
        for axis in axes:
            values, incomplete = [], False
            for i, c in enumerate(contracts):
                raw = c.get(section, {}).get(axis, unknown('required field absent'))
                v = cell(raw, kind(axis) if section == 'costs' else 'constraint')
                if v['status'] != 'actual':
                    gaps.append(dict(arm=ARMS[i], field=section + '.' + axis, reason=v['reason']))
                    incomplete = True
                else:
                    values.append(v['value'])
                    if section == 'costs' and kind(axis) == 'counter' and v['value']['attempted'] != v['value']['completed']:
                        gaps.append(dict(arm=ARMS[i], field=section + '.' + axis, reason='incomplete attempts'))
                    if section == 'costs' and kind(axis) == 'CPU' and any(
                            item['attempted'] != item['completed'] for item in v['value'].values()):
                        gaps.append(dict(arm=ARMS[i], field=section + '.' + axis, reason='incomplete CPU attempts'))
            if not incomplete and any(v != values[0] for v in values[1:]):
                gaps.append(dict(arm='all', field=section + '.' + axis, reason='componentwise mismatch'))
    return dict(status='unestablished' if gaps else 'equal', gaps=gaps,
                meaning='declared complete vector only; not measured equal total compute or scientific evidence')


def prospective_contract(m):
    """Read accounting units, not outcomes or calibration performance."""
    costs = {k: cell(m['two_root_totals'].get(k, unknown('not metered')), kind(k))
             for k in COUNTERS + SCALARS}
    for k in EXTRA:
        costs[k] = unknown('fresh complete per-method pipeline receipt/budget unavailable')
    # Only exact single-arm receipts are attributable; never allocate a cohort.
    physics = m['per_method_physics']
    if physics['status'] == 'actual':
        for k in ('physics_step', 'public_step'):
            if k in physics['value']:
                counter(physics['value'][k])
                costs[k] = known(physics['value'][k], 'single-arm saved cohort; not a fresh budget')
    else:
        cell(physics, 'constraint')
    constraints = {k: unknown('no complete matched prospective ' + k + ' contract') for k in CONSTRAINTS}
    h = cell(m['future_packets'], 'scalar')
    if h['status'] == 'actual':
        constraints['horizon'] = known(dict(future_packets=h['value']), 'explicit future packets, NOT observed H3')
    constraints['budget_scope'] = known('two_roots_sealed_reference_only_not_fresh_prediction', 'matrix scope')
    return dict(costs=costs, constraints=constraints)


def freshness(cert, trusted_sha, trusted_authority, plan_sha, now_ns):
    if cert is None:
        return dict(status='blocked', gaps=['independent freshness authority absent'])
    try:
        require(trusted_sha is not None and trusted_authority is not None, 'external trust anchors required')
        require(sha(encode(cert)) == digest(trusted_sha), 'freshness canonical SHA mismatch')
        require(cert['schema'] == STEM + '.freshness' and cert['authority'] == trusted_authority == 'MAIN',
                'independent MAIN authority required')
        require(cert['plan_sha256'] == plan_sha and cert['calibration_matrix_sha256'] == MATRIX_SHA,
                'freshness plan/calibration binding')
        require(cert['producer'] != cert['authority'] and bool(cert['producer']), 'self authority forbidden')
        for flag in ('independent_roots', 'not_used_for_design', 'no_candidate_cost_or_outcome_seen',
                     'predeclared_order', 'complete_prior_root_exclusion_checked'):
            require(cert[flag] is True, 'freshness assertion required: ' + flag)
        require(cert['excluded_calibration_task_sha256'] == list(TASKS), 'calibration tasks explicitly excluded')
        roots = cert['ordered_fresh_root_sha256']
        require(isinstance(roots, list) and len(roots) == 2 and len(set(roots)) == 2, 'two distinct ordered fresh roots')
        for r in roots:
            digest(r)
            require(r not in TASKS, 'calibration reuse forbidden')
        require(type(cert['issued_at_ns']) is int and type(cert['expires_at_ns']) is int and
                type(now_ns) is int and 0 < cert['issued_at_ns'] <= now_ns < cert['expires_at_ns'], 'freshness chronology/expiry')
        return dict(status='verified_external_attestation', sha256=trusted_sha,
                    authority=trusted_authority, ordered_fresh_root_sha256=roots,
                    limitation='caller-supplied trust anchors, not cryptographic proof of independence')
    except (KeyError, TypeError, ValueError) as exc:
        return dict(status='blocked', gaps=[str(exc)])


def assemble(matrix, cert=None, trusted_sha=None, trusted_authority=None, now_ns=None):
    require(matrix['schema'] == 'p1_moving_baseline_budget_matrix_20261027_v1.matrix' and
            matrix['status'] == 'saved_resource_projection_not_admitted', 'pinned matrix schema/status')
    require(matrix['input']['sha256'] == COMPARISON_SHA and matrix['equal_total_compute'] is False and
            matrix['P1_admission'] is False, 'calibration is not equal-compute/admission evidence')
    require([m['algorithm'] for m in matrix['methods']] == list(ARMS), 'fixed ordered six baselines')
    candidates = []
    for i, m in enumerate(matrix['methods']):
        require([r['source_slot'] for r in m['root_receipts']] == [0, 1] and
                [r['task_sha256'] for r in m['root_receipts']] == list(TASKS), 'calibration root binding')
        candidates.append(dict(queue_index=i, algorithm=ARMS[i],
            purpose='future reexecution candidate only; saved outcomes never compared',
            configuration_policy='sealed_reference_unchanged_no_retuning',
            sealed_config_sha256=sha(encode(m['contract_fields'])),
            sealed_config=deepcopy(m['contract_fields']),
            matrix_pointer=f'/methods/{i}',
            reference_contract=prospective_contract(m),
            prospective_budget=unknown('sealed reference cost is not a certified fresh equal-budget allocation')))
    plan = dict(schema=STEM + '.plan', calibration_matrix_sha256=MATRIX_SHA,
                accounting_source_sha256=SOURCE_SHA, queue_order='fixed_six_arms_no_replacement_no_retries',
                population=unknown('requires independent fresh-root authority, never calibration roots'),
                units=UNITS, rules=RULES, candidates=candidates,
                shared_accounting=dict(inherited_encoding_shared_once=deepcopy(matrix['inherited_encoding_shared_once']),
                    physical_cohorts=deepcopy(matrix['physical_cohorts']), RGB_cohorts=deepcopy(matrix['RGB_cohorts'])))
    binding = sha(encode(plan))
    eq = assess([c['reference_contract'] for c in candidates])
    f = freshness(cert, trusted_sha, trusted_authority, binding, time.time_ns() if now_ns is None else now_ns)
    blockers = ['no independently approved prospective per-method budgets; sealed budgets cannot be changed',
                'calibration receipts cannot certify future complete-pipeline vector or support equality']
    if eq['status'] != 'equal':
        blockers.append('componentwise budget/support equality unestablished; see equality.gaps')
    if f['status'] == 'blocked':
        blockers.extend(f['gaps'])
    # v1 is a planning producer only. Even a valid attestation cannot turn old
    # heterogeneous receipts into a new experiment or authorize HOST execution.
    return dict(schema=STEM + '.queue', status='blocked_planning_artifact', plan=plan,
        plan_sha256=binding, equality=eq, freshness=f, blockers=blockers,
        ready=False, launch_authorized=False, executable_queue=[],
        review_required='MAIN review then HOST CUDA only; no runner provided',
        equal_total_compute=False, scientific_evidence=False, P1_admission=False,
        failure_policy='missing/null blocks; preserve every candidate; no selection from old results')


def unique(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, 'duplicate JSON key')
        result[k] = v
    return result


def read_json(path):
    p = Path(path)
    require(p.stat().st_size <= 1024**2, '1MiB metadata cap')
    data = p.read_bytes()
    require(len(data) <= 1024**2, '1MiB metadata cap')
    return json.loads(data, object_pairs_hook=unique,
                      parse_constant=lambda _: require(False, 'nonfinite JSON')), data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, default=Path(__file__).resolve().parents[2] /
                   'results/p1_moving_baseline_budget_matrix_20261027_v1_main01/budget_matrix.json')
    p.add_argument('--freshness', type=Path)
    p.add_argument('--freshness-sha', help='external canonical certificate SHA, not self-calculated authority')
    p.add_argument('--trusted-freshness-authority', choices=['MAIN'])
    a = p.parse_args()
    matrix, raw = read_json(a.matrix)
    require(sha(raw) == MATRIX_SHA, 'pinned matrix SHA mismatch')
    source = Path(__file__).with_name('p1_moving_baseline_budget_matrix_20261027_v1.py')
    require(sha(source.read_bytes()) == SOURCE_SHA, 'accounting source changed')
    require(a.freshness is not None or (a.freshness_sha is None and a.trusted_freshness_authority is None),
            'freshness file required with trust anchors')
    cert = read_json(a.freshness)[0] if a.freshness else None
    artifact = assemble(matrix, cert, a.freshness_sha, a.trusted_freshness_authority)
    require(len(encode(artifact)) <= 256 * 1024, '256KiB output cap')
    print(encode(artifact).decode(), end='')


if __name__ == '__main__':
    main()
