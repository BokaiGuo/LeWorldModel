"""Two explicit prediction-budget contracts and a stdlib callback meter.

No model runner, outcome reader, freshness signer or total-compute certificate.
CLI only prints an engineering report from the pinned calibration resource matrix.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time

STEM = 'moving_prediction_budget_contract_20261027_v1'
MATRIX_SHA = '721d978d29516f8ecf1d1056e159736ef231723e6849490a83678bd76baa1ebc'
ARMS = ('moving_gradient_open_loop', 'CEM', 'kernel_faithful_mppi_nominal',
        'fixed_H2_kinodynamic_beam16x4', 'GRASP_H3_bounded', 'joint_H2_spectral_packet_mapped')
TASKS = ('a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b',
         '101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74')
MODES = {'predict_API_calls_matched': 'predict_API_calls',
         'predictor_forward_samples_matched': 'predictor_forward_samples'}
KINDS = ('predict_API', 'predictor_forward', 'action_encoder', 'pred_projector')
STAGES = ('prepare', 'search', 'replay')
AXES = ('predict_API_calls', 'predictor_forward_calls', 'predictor_forward_samples',
        'action_encoder_calls', 'action_encoder_samples', 'action_encoder_rows',
        'pred_projector_calls', 'pred_projector_samples', 'pred_projector_rows')
OTHER_RESOURCES = ('backward', 'grad_API', 'jacobian', 'jacobian_API', 'scalar_output_VJP_directions',
                   'scalar_loss_reverse_sweeps', 'SVD', 'bounded_solve', 'image_encoder',
                   'image_projector', 'physics_step', 'public_step', 'pipeline_walltime_seconds')
CAP = 1024**2
LIMITATION = ('The frozen all-axes equality contract cannot establish the proposal-specific '
              'matched world-model prediction-call budget claim. Walltime/derivative equality '
              'is not a universal comparison gate; API calls and forward samples are distinct claims.')


class ContractError(ValueError):
    pass


class BudgetExceeded(ContractError):
    pass


def require(ok, message):
    if not ok:
        raise ContractError(message)


def encode(x):
    return (json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(x):
    return sha(encode(x))


def integer(x, label, minimum=1):
    require(type(x) is int and x >= minimum, label + ': integer required, bool forbidden')


def keys(x, expected):
    require(isinstance(x, dict) and set(x) == set(expected.split()), 'exact fields: ' + expected)


def hash_value(x):
    require(type(x) is str and len(x) == 64 and all(c in '0123456789abcdef' for c in x), 'literal SHA256 required')
    return x


def validate_cell(c):
    require(isinstance(c, dict) and c.get('status') in ('actual', 'missing', 'partial', 'unmeasured') and 'value' in c,
            'typed saved resource cell')
    if c['status'] != 'actual':
        require(c['value'] is None and bool(c.get('reason')), 'unknown is null, not zero')
    else:
        require(bool(c.get('basis')), 'saved actual basis required')
        value = c['value']
        if isinstance(value, dict):
            if set(value) == {'attempted', 'completed'}:
                for v in value.values(): integer(v, 'counter', 0)
                require(value['completed'] <= value['attempted'], 'counter phases')
            else:
                for v in value.values(): validate_cell(dict(status='actual', value=v, basis=c['basis']))
        else:
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'finite saved cost')
    return deepcopy(c)


def calibration_groups(matrix):
    require(matrix['schema'] == 'p1_moving_baseline_budget_matrix_20261027_v1.matrix' and
            matrix['status'] == 'saved_resource_projection_not_admitted' and
            matrix['equal_total_compute'] is False and matrix['P1_admission'] is False, 'saved calibration matrix boundary')
    require([m['algorithm'] for m in matrix['methods']] == list(ARMS), 'ordered six local baseline arms')
    groups = []
    for mi, method in enumerate(matrix['methods']):
        require([r['source_slot'] for r in method['root_receipts']] == [0, 1] and
                all(type(r['source_slot']) is int for r in method['root_receipts']), 'ordered integer roots')
        for slot, row in enumerate(method['root_receipts']):
            require(row['task_sha256'] == TASKS[slot], 'original calibration task SHA')
            groups.append(dict(group_id=f'arm{mi}:root{slot}', arm=ARMS[mi], source_slot=slot,
                task_sha256=TASKS[slot], sealed_configuration_sha256=digest(method['contract_fields'])))
    return groups


def declaration(value, expected_sha, trusted_authority, groups, now_ns=None):
    """Validate an external MAIN predeclaration; never issue or repair one here.

    Caller supplies independent trust anchors. A matching SHA is integrity, not
    cryptographic proof that MAIN or a fresh-root authority signed anything.
    """
    keys(value, 'schema authority declaration_id issued_at_ns matrix_sha256 population mode cap allocation groups scope batch no_outcome_based_configuration_change')
    require(sha(encode(value)) == hash_value(expected_sha), 'external canonical declaration SHA mismatch')
    require(trusted_authority == value['authority'] == 'MAIN', 'external MAIN trust anchor required')
    require(value['schema'] == STEM + '.MAIN_declaration' and value['matrix_sha256'] == MATRIX_SHA, 'declaration schema/matrix binding')
    require(type(value['declaration_id']) is str and bool(value['declaration_id'].strip()), 'declaration identity')
    integer(value['issued_at_ns'], 'predeclaration time')
    now = time.time_ns() if now_ns is None else now_ns
    integer(now, 'verification time')
    require(value['issued_at_ns'] <= now, 'future declaration')
    require(value['population'] == 'existing_calibration_two_roots' and value['allocation'] == 'per_root_per_arm', 'explicit calibration allocation')
    require(value['groups'] == groups and len(groups) == 12, 'exact immutable configuration/root/arm binding')
    for i, g in enumerate(groups):
        keys(g, 'group_id arm source_slot task_sha256 sealed_configuration_sha256')
        require(g['group_id'] == f'arm{i // 2}:root{i % 2}' and g['arm'] == ARMS[i // 2] and
                type(g['source_slot']) is int and g['source_slot'] == i % 2 and
                g['task_sha256'] == TASKS[i % 2], 'fixed distinct calibration group identities')
        hash_value(g['sealed_configuration_sha256'])
    require(type(value['mode']) is str and value['mode'] in MODES, 'explicit prediction budget mode required')
    integer(value['cap'], 'common per-group attempted prediction cap')
    require(value['no_outcome_based_configuration_change'] is True, 'no outcome-based retuning')
    keys(value['scope'], 'prepare search ADforward replay failed_attempts count_on')
    require(all(type(value['scope'][k]) is bool for k in ('prepare', 'search', 'ADforward', 'replay')),
            'explicit prepare/search/ADforward/replay booleans')
    require(value['scope']['search'] is True and value['scope']['failed_attempts'] == 'charged_no_refund' and
            value['scope']['count_on'] == 'before_dispatch', 'attempted budgets charge before dispatch; failure cannot refund')
    keys(value['batch'], 'allowed_sizes max_predictor_forwards_per_API API_predictor_batch_relation')
    sizes = value['batch']['allowed_sizes']
    require(isinstance(sizes, list) and bool(sizes), 'explicit allowed batch sizes')
    for n in sizes: integer(n, 'batch size')
    require(sizes == sorted(set(sizes)), 'unique ordered batch sizes')
    integer(value['batch']['max_predictor_forwards_per_API'], 'explicit API fanout bound')
    require(value['batch']['API_predictor_batch_relation'] in ('same_logical_batch', 'bounded_child_batches'),
            'explicit API/predictor batching relation')
    return deepcopy(value)


@dataclass(frozen=True)
class Ticket:
    owner: object
    serial: int


class PredictionBudgetMeter:
    """Hook/callback ledger, NOT proof that callbacks instrumented a real model.

    Wrap model.predict with predict_API, and predictor.forward with a child
    predictor_forward dispatch. AE/pred_proj child hooks are diagnostic, never
    additional predictor budget. Callback return means host return, not a GPU
    synchronization claim. No model outputs are read or interpreted.
    """
    def __init__(self, main_declaration, expected_sha, trusted_authority, groups, *, sink=None, now_ns=None):
        self._declaration = declaration(main_declaration, expected_sha, trusted_authority, groups, now_ns)
        self._owner = object()
        self._lock = threading.RLock()
        self._sink = sink
        self._emitting = False
        self._fault = None
        self._records = {}
        self._tickets = {}
        self._events = []
        self._groups = {g['group_id'] for g in groups}
        self._counts = {g: {scope: {axis: dict(attempted=0, completed=0) for axis in AXES}
                            for scope in ('observed', 'included')} for g in self._groups}
        self._denied = 0

    def _usable(self):
        require(self._fault is None and not self._emitting, 'meter fault/reentrant sink; dispatch closed')

    def _record(self, ticket):
        require(type(ticket) is Ticket and ticket.owner is self._owner and
                self._tickets.get(ticket.serial) is ticket,
                'foreign/forged ticket')
        return self._records[ticket.serial]

    def _emit(self, event):
        self._events.append(deepcopy(event))
        if self._sink is None:
            return
        self._emitting = True
        try:
            self._sink(deepcopy(event))
        except BaseException as exc:
            self._fault = 'sink failure: ' + type(exc).__name__
            raise
        finally:
            self._emitting = False

    @staticmethod
    def _axes(kind, batch_size, rows_per_sample):
        if kind == 'predict_API': return dict(predict_API_calls=1)
        if kind == 'predictor_forward': return dict(predictor_forward_calls=1, predictor_forward_samples=batch_size)
        return {kind + '_calls': 1, kind + '_samples': batch_size, kind + '_rows': batch_size * rows_per_sample}

    def before_dispatch(self, kind, *, group_id=None, stage=None, ad_forward=None,
                        batch_size=1, rows_per_sample=1, parent=None):
        """Reserve attempted units atomically before callback/actual hook dispatch.

        API owns group/stage/AD tags. Every module inherits them from the API;
        children cannot relabel AD/replay as excluded work to evade a cap.
        Excluded stages remain in observed ledgers, not in the selected cap.
        """
        with self._lock:
            self._usable()
            require(kind in KINDS, 'dispatch kind')
            integer(batch_size, 'batch size'); integer(rows_per_sample, 'rows per sample')
            require(batch_size in self._declaration['batch']['allowed_sizes'], 'batch forbidden by MAIN')
            if kind in ('predict_API', 'predictor_forward'):
                require(rows_per_sample == 1, 'prediction samples are batch items, not token/projector rows')
            if kind == 'predict_API':
                require(parent is None and type(group_id) is str and group_id in self._groups and stage in STAGES and type(ad_forward) is bool,
                        'API needs registered group/stage/AD tag, no parent')
                included = self._declaration['scope'][stage] and (not ad_forward or self._declaration['scope']['ADforward'])
                parent_record = None
            else:
                parent_record = self._record(parent)
                require(parent_record['kind'] == 'predict_API' and parent_record['state'] == 'pending', 'live API parent required')
                require(group_id is stage is ad_forward is None, 'children inherit scope; no override')
                group_id, stage, ad_forward, included = (parent_record[k] for k in ('group_id', 'stage', 'ad_forward', 'included'))
                if self._declaration['batch']['API_predictor_batch_relation'] == 'same_logical_batch':
                    require(batch_size == parent_record['batch_size'], 'API/module logical batch mismatch')
                if kind == 'predictor_forward':
                    require(parent_record['forward_attempts'] < self._declaration['batch']['max_predictor_forwards_per_API'],
                            'MAIN API predictor fanout bound; failed forward counts too')
            increments = self._axes(kind, batch_size, rows_per_sample)
            primary = MODES[self._declaration['mode']]
            charge = increments.get(primary, 0) if included else 0
            if self._counts[group_id]['included'][primary]['attempted'] + charge > self._declaration['cap']:
                self._denied += 1
                self._emit(dict(event='dispatch_denied', kind=kind, group_id=group_id, charge=charge, reason='cap'))
                raise BudgetExceeded('prediction attempt cap exceeded before dispatch')
            serial = len(self._records)
            rec = dict(kind=kind, group_id=group_id, stage=stage, ad_forward=ad_forward, included=included,
                batch_size=batch_size, rows_per_sample=rows_per_sample, increments=increments, state='pending',
                parent_serial=None if parent is None else parent.serial, forward_attempts=0)
            self._records[serial] = rec
            ticket = Ticket(self._owner, serial)
            self._tickets[serial] = ticket
            for axis, n in increments.items():
                self._counts[group_id]['observed'][axis]['attempted'] += n
                if included: self._counts[group_id]['included'][axis]['attempted'] += n
            if kind == 'predictor_forward': parent_record['forward_attempts'] += 1
            self._emit(dict(event='before_dispatch', serial=serial, **deepcopy(rec)))
            return ticket

    def _finish(self, ticket, success, error_type=None):
        with self._lock:
            self._usable()
            rec = self._record(ticket)
            require(rec['state'] == 'pending', 'ticket already terminal; no duplicate completion/refund')
            require(not any(r['state'] == 'pending' and r['parent_serial'] == ticket.serial for r in self._records.values()),
                    'API cannot close before its children')
            if not success: require(type(error_type) is str and bool(error_type), 'failure type required')
            rec['state'] = 'returned' if success else 'failed'
            if success:
                for axis, n in rec['increments'].items():
                    self._counts[rec['group_id']]['observed'][axis]['completed'] += n
                    if rec['included']: self._counts[rec['group_id']]['included'][axis]['completed'] += n
            self._emit(dict(event='after_return' if success else 'after_failure', serial=ticket.serial,
                            error_type=error_type, no_refund=True))

    def after_return(self, ticket):
        self._finish(ticket, True)

    def after_failure(self, ticket, error_type):
        self._finish(ticket, False, error_type)

    def invoke(self, kind, callback, **dispatch):
        """Generic callback(token)->value seam. No implicit module/model dispatch."""
        require(callable(callback), 'callable required')
        token = self.before_dispatch(kind, **dispatch)
        try:
            value = callback(token)
        except BaseException as exc:
            try: self.after_failure(token, type(exc).__name__)
            except BaseException as ledger_error: exc.add_note('meter close failed: ' + type(ledger_error).__name__)
            raise
        self.after_return(token)
        return value

    def snapshot(self):
        with self._lock:
            primary = MODES[self._declaration['mode']]
            pending = sum(r['state'] == 'pending' for r in self._records.values())
            ready = pending == 0 and self._fault is None
            return dict(schema=STEM + '.callback_ledger', mode=self._declaration['mode'], primary_unit=primary,
                declaration_sha256=digest(self._declaration), cap_per_group=self._declaration['cap'],
                counts=deepcopy(self._counts), events=deepcopy(self._events), pending=pending, denied=self._denied,
                fault=self._fault, engineering_report_ready=ready,
                callback_attempt_allocation_matched=ready and all(
                    c['included'][primary]['attempted'] == self._declaration['cap'] for c in self._counts.values()),
                evidence_kind='unverified_callback_observations_not_real_model_receipts',
                actual_model_execution_verified=False, scientific_admission=False,
                other_resources={axis: dict(status='unmeasured', value=None,
                    reason='not observed by this callback meter; saved reference receipts are separate, not current-run costs')
                    for axis in OTHER_RESOURCES},
                equal_total_compute=False, equal_FLOPs=False,
                completion_semantics='successful synchronous callback return; not CUDA synchronization or predictive accuracy')


def calibration_report(matrix, main_declaration=None, expected_sha=None, trusted_authority=None, now_ns=None):
    groups = calibration_groups(matrix)
    policy = None
    if main_declaration is not None:
        policy = declaration(main_declaration, expected_sha, trusted_authority, groups, now_ns)
    else:
        require(expected_sha is trusted_authority is None, 'declaration file missing for trust anchors')
    methods = []
    for method in matrix['methods']:
        for cell in method['two_root_totals'].values(): validate_cell(cell)
        for root in method['root_receipts']:
            for cell in root['counters'].values(): validate_cell(cell)
        methods.append(deepcopy(method))
    selected_axis = None if policy is None else ('predict_API' if policy['mode'] == 'predict_API_calls_matched' else 'forward_samples')
    return dict(schema=STEM + '.calibration_report', engineering_status='calibration_report_ready', report_ready=True,
        saved_input_verified=False, input_provenance='RAM_or_unverified_until_external_SHA_gate',
        matrix_sha256=MATRIX_SHA, groups=groups, methods=methods,
        primary_mode=None if policy is None else policy['mode'], main_declaration=policy,
        meter_configuration_ready=policy is not None, common_cap_allocation_predeclared=policy is not None,
        selected_saved_counter_axis=selected_axis, prediction_budget_match='unestablished',
        match_limitations=['Existing six configurations have unequal saved counts; never retune from outcomes.',
            'Matrix API cells absent for several methods; cannot infer API counts from forwards/samples/AE.',
            'Matrix alone does not resolve MAIN-selected prepare/ADforward/replay/batch/attempt scope; no retrospective match certification.'],
        legacy_all_axes_contract_supports_prediction_budget_claim=False, legacy_contract_limitation=LIMITATION,
        other_resource_equality_required=False, walltime_equality_required=False,
        shared_accounting={k: deepcopy(matrix[k]) for k in ('inherited_encoding_shared_once', 'model_invocations', 'physical_cohorts', 'RGB_cohorts')},
        overlap=['AE and pred_proj are module meters, never additional predictor samples or API calls.',
            'AD forwards are tagged subsets of predictor forwards, never counted twice.',
            'Batch B at one predictor dispatch means one forward call and B forward samples, not B API calls.',
            'Rows_per_sample tracks AE/projector context rows, not predictor budget units.',
            'Shared/cohort and standalone views retain their original scopes; never summed twice or refunded.'],
        scientific_admission=False, missing_scientific_authority=['independent freshness', 'P1 admission'],
        new_root_certificate_issued=False, model_runner_provided=False, launch_authorized=False,
        equal_total_compute=False, equal_FLOPs=False,
        status_meaning='ready to read saved calibration accounting only, NOT scientific evidence or prediction-budget equality')


def unique(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, 'duplicate JSON key')
        result[k] = v
    return result


def read_json(path, expected_sha=None):
    p = Path(path).absolute()
    require(p.is_file() and not p.is_symlink() and p == p.resolve() and p.stat().st_size <= CAP, 'bounded nonsymlink JSON')
    raw = p.read_bytes()
    require(len(raw) <= CAP, 'metadata cap')
    if expected_sha is not None: require(sha(raw) == hash_value(expected_sha), 'pinned input SHA mismatch')
    result = json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: require(False, 'nonfinite JSON'))
    encode(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, default=Path(__file__).resolve().parents[2] /
                   'results/p1_moving_baseline_budget_matrix_20261027_v1_main01/budget_matrix.json')
    p.add_argument('--declaration', type=Path)
    p.add_argument('--declaration-sha', help='external canonical MAIN declaration SHA, not raw-file SHA')
    p.add_argument('--trusted-authority', choices=['MAIN'])
    a = p.parse_args()
    matrix = read_json(a.matrix, MATRIX_SHA)
    d = read_json(a.declaration) if a.declaration else None
    report = calibration_report(matrix, d, a.declaration_sha, a.trusted_authority)
    report['saved_input_verified'] = True
    report['input_provenance'] = 'pinned_saved_calibration_resources'
    require(len(encode(report)) <= CAP, 'report cap')
    print(encode(report).decode(), end='')


if __name__ == '__main__': main()
