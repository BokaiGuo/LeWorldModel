"""Source-bound store seam for the existing K17 outer/meter; not a runner.

Use from_saved() before passing its data, rows, and store to the existing outer.
No CLI, environment construction, authority grant, or GPU operations exist here.
"""
from copy import deepcopy
import hashlib
import time
from pathlib import Path

from experiments.spectral_reachability import agent_p3_c3_k17_candidate_join_20261029_v1 as join
from experiments.spectral_reachability import agent_p3_c3_k17_native_meter_20261029_v1 as meter
from experiments.spectral_reachability import agent_p3_c3_k17_native_outer_20261029_v1 as outer
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as transport


def digest(value):
    return hashlib.sha256(transport.canonical(value)).hexdigest()


def source_digest(data, rows):
    return digest(join.snapshot(data, rows))


def row_digest(row):
    return digest(join.normalize(row))


SOURCE_FILES = ('p3_k17_live_source_budget_seam_20261029.py',
                'agent_p3_c3_k17_native_outer_20261029_v1.py',
                'agent_p3_c3_k17_native_episode_20261029_v1.py',
                'agent_p3_c3_k17_native_meter_20261029_v1.py',
                'agent_p3_c3_k17_native_store_20261029_v1.py',
                'agent_p3_c3_k17_physical_gate_20261029_v1.py',
                'agent_p3_c3_k17_physical_transport_20261029_v1.py')


def source_files():
    directory = Path(__file__).resolve().parent
    result = {}
    for name in SOURCE_FILES:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('missing or symlink source')
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class SourceBoundStore:
    """NativeStore-compatible observer; only use with its own loaded material.

    Event authenticity remains dependent on the caller, environment and adapter.
    This is not a saved consumer and grants no execution authority.
    """
    @classmethod
    def from_saved(cls, store):
        data, rows = outer.material()  # pinned handoff and replay source closure
        return cls(store, data, rows)

    def __init__(self, store, data, rows):
        self.started_at = time.monotonic()
        self.store, self.path, self.data, self.rows = store, store.path, data, rows
        self.source_sha256 = source_digest(data, rows)
        self.row_hashes = [row_digest(r) for r in rows]
        self.actions = []
        for (item, _, _), identity in zip(rows, outer.ORDER, strict=True):
            if (item['slot'], item['root'], item['mode'], item['K']) != (*identity, 17):
                raise ValueError('source slot identity')
            try:
                raw = bytes.fromhex(item['first5_raw_f32_hex'])
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError('source action bytes') from exc
            if len(raw) != 40:
                raise ValueError('source first5 bytes')
            self.actions.append([raw[i:i+8].hex() for i in range(0, 40, 8)])
        self.counts = {k: {'attempted': 0, 'completed': 0, 'failed': 0} for k in meter.CAPS}
        self.pending = {}
        self.branch = -1
        self.branch_open = False
        self.guard = False
        self.plan_step = 0
        self.transition_step = 0
        self.native_stopped = False
        self.fault = False
        self.started = False
        self.store.emit('source_budget_binding', source_sha256=self.source_sha256,
                        row_sha256=self.row_hashes, order=[list(x) for x in outer.ORDER],
                        source_files_sha256=source_files(),
                        resource_caps=dict(meter.CAPS), wall_seconds=meter.WALL,
                        production_authorized=False, native_provenance_verified=False)

    def check(self, *args, **kwargs):
        if self.fault:
            raise ValueError('source budget seam fault')
        return self.store.check(*args, **kwargs)

    def frame(self, *args, **kwargs):
        self.check()
        return self.store.frame(*args, **kwargs)

    def arrays(self, *args, **kwargs):
        return self.store.arrays(*args, **kwargs)

    def _row_unchanged(self):
        if self.branch < 0 or row_digest(self.rows[self.branch]) != self.row_hashes[self.branch]:
            raise ValueError('bound source row changed')

    def _identity(self, rec):
        if self.branch < 0:
            raise ValueError('no branch')
        slot, root, mode = outer.ORDER[self.branch]
        if (rec.get('branch'), rec.get('slot'), rec.get('root_slot'), rec.get('mode')) != (self.branch, slot, root, mode):
            raise ValueError('event source identity')

    def _intent(self, rec):
        kind, index = rec.get('resource'), rec.get('index')
        if kind not in self.counts or type(index) is not int or index < 0 or index in self.pending:
            raise ValueError('invalid operation intent')
        if index != sum(v['attempted'] for v in self.counts.values()):
            raise ValueError('operation index')
        if kind == 'constructor':
            if self.pending or self.branch_open or self.branch >= 5 or rec.get('branch') != self.branch + 1:
                raise ValueError('constructor branch order')
            if source_digest(self.data, self.rows) != self.source_sha256:
                raise ValueError('bound source changed')
            self.branch += 1
            self.branch_open = True
            self.guard, self.plan_step, self.transition_step, self.native_stopped = False, 0, 0, False
        self._identity(rec)
        if not self.branch_open:
            raise ValueError('no open branch')
        if self.counts[kind]['attempted'] >= meter.CAPS[kind]:
            raise ValueError('resource cap')
        if kind == 'public_step' and rec.get('phase') == 'plan':
            self._row_unchanged()
            step = rec.get('step')
            if (not self.guard or self.native_stopped or self.transition_step != self.plan_step or
                type(step) is not int or step != self.plan_step + 1 or step > 5):
                raise ValueError('guard or plan order')
            if rec.get('raw_bits') != self.actions[self.branch][step-1]:
                raise ValueError('paid replay action mismatch')
            self.plan_step = step
        self.counts[kind]['attempted'] += 1
        self.pending[index] = deepcopy(rec)

    def _finish(self, rec, failed):
        idx = rec.get('index')
        original = self.pending.get(idx)
        expected = dict(original or {})
        expected['status'] = 'failed' if failed else 'returned'
        if failed:
            expected['error'] = rec.get('error')
        if original is None or rec != expected or (failed and not isinstance(rec.get('error'), dict)):
            raise ValueError('operation receipt mismatch')
        self._identity(rec)
        self.counts[rec['resource']]['failed' if failed else 'completed'] += 1
        del self.pending[idx]
        if failed:
            self.fault = True

    def emit(self, kind, **fields):
        """Intercept actual NativeStore emit calls; intent is persisted pre-callback."""
        try:
            rec = fields.get('receipt')
            cleanup = kind.startswith('operation_') and isinstance(rec, dict) and rec.get('resource') == 'close'
            if self.fault and not (cleanup or kind in ('operation_failed', 'branch_closed')):
                raise ValueError('source budget seam fault')
            if kind == 'run_started':
                if self.started or source_digest(self.data, self.rows) != self.source_sha256:
                    raise ValueError('source changed or repeated start')
                if fields != {'denominator': 6, 'order': [list(x) for x in outer.ORDER]}:
                    raise ValueError('run contract')
                self.started = True
            elif kind == 'operation_dispatch_intent':
                if not self.started or not isinstance(rec, dict) or rec.get('status') != 'pending':
                    raise ValueError('intent contract')
                self._intent(rec)
                if not cleanup and time.monotonic() - self.started_at >= meter.WALL:
                    raise ValueError('seam wall limit before intent')
            elif kind in ('operation_returned', 'operation_failed'):
                if not isinstance(rec, dict):
                    raise ValueError('receipt contract')
                self._finish(rec, kind == 'operation_failed')
            elif kind == 'k17_guard':
                self._row_unchanged()
                proof = fields.get('proof', {})
                if (self.guard or fields.get('accepted') is not True or
                    proof.get('physical_dispatch_ready') is not False or
                    proof.get('history', {}).get('accepted') is not True or
                    proof.get('action', {}).get('actual_replay_tokens_exact') is not True):
                    raise ValueError('guard rejected')
                self.guard = True
            elif kind == 'branch_closed':
                row = fields.get('row')
                if not self.branch_open or not isinstance(row, dict):
                    raise ValueError('branch close contract')
                slot, root, mode = outer.ORDER[self.branch]
                if (row.get('index'), row.get('slot'), row.get('root_slot'), row.get('mode')) != (self.branch, slot, root, mode):
                    raise ValueError('branch close identity')
                if self.pending and row.get('status') != 'engineering_failed':
                    raise ValueError('pending operation at branch close')
                self.branch_open = False
            elif kind == 'returned_transition' and fields.get('phase') == 'plan':
                self._identity(fields)
                step = fields.get('step')
                if (type(step) is not int or step != self.plan_step or step < 1 or step > 5 or
                    self.transition_step != step-1 or any(r.get('resource') == 'public_step' for r in self.pending.values()) or
                    fields.get('requested_bits') != self.actions[self.branch][step-1] or
                    fields.get('actual_bits') != self.actions[self.branch][step-1] or
                    fields.get('actual_dtype') != '<f4' or
                    type(fields.get('terminated')) is not bool or type(fields.get('truncated')) is not bool):
                    raise ValueError('actual action mismatch')
                self.transition_step = step
                self.native_stopped = fields['terminated'] or fields['truncated']
            result = self.store.emit(kind, **fields)
            if kind == 'operation_dispatch_intent' and not cleanup and time.monotonic() - self.started_at >= meter.WALL:
                raise ValueError('seam wall limit after intent')
            return result
        except BaseException:
            self.fault = True
            raise

    def summary(self):
        return {'source_sha256': self.source_sha256, 'counts': deepcopy(self.counts),
                'pending': deepcopy(self.pending), 'fault': self.fault,
                'native_provenance_verified': False, 'production_authorized': False,
                'scientific_pass': False}
