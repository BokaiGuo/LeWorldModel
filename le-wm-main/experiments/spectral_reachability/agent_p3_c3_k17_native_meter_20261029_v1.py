"""Six-slot K17 prospective native resource meter; not a physical runner."""
from copy import deepcopy
import time

CAPS = {'constructor': 6, 'close': 6, 'reset': 12, 'set_state': 24,
        'goal_render': 12, 'goal_setter': 18, 'goal_bind': 6,
        'public_step': 90, 'physics_step': 924, 'explicit_render': 306}
WALL = 900


class Meter:
    """An actual callback is never invoked until its intent has been fsynced.

    The caller supplies the bounded K17 native store. It must retain any failure
    to publish the final record; this class is not itself a complete runner.
    """
    def __init__(self, store, *, clock=time.monotonic, started=None):
        self.store, self.clock = store, clock
        self.started = clock() if started is None else started
        self.where = {}
        self.counts = {k: {'attempted': 0, 'completed': 0} for k in CAPS}
        self.active = {k: 0 for k in CAPS}
        self.receipts = []
        self.secondary = []
        self.fault = False

    def call(self, kind, fn, *, cleanup=False, **fields):
        if kind not in CAPS or (cleanup and kind != 'close'):
            raise ValueError('unknown resource/cleanup')
        if not cleanup:
            if self.fault or self.clock() - self.started >= WALL:
                raise ValueError('wall or meter fault before dispatch')
            self.store.check()
        count = self.counts[kind]
        if count['attempted'] >= CAPS[kind]:
            self.fault = True
            raise ValueError('resource cap before dispatch')
        rec = {'index': len(self.receipts), 'resource': kind, **deepcopy(self.where),
               **deepcopy(fields), 'status': 'pending'}
        try:
            self.store.emit('operation_dispatch_intent', receipt=deepcopy(rec))
        except BaseException as exc:
            self.fault = True
            if not cleanup:
                raise
            self.secondary.append({'stage': 'cleanup_intent', 'type': type(exc).__name__})
        self.receipts.append(rec)
        count['attempted'] += 1
        self.active[kind] += 1
        try:
            result = fn()
        except BaseException as exc:
            rec['status'] = 'failed'
            rec['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            self.fault = True
            try:
                self.store.emit('operation_failed', receipt=deepcopy(rec))
            except BaseException as secondary:
                self.secondary.append({'stage': 'failure_journal', 'type': type(secondary).__name__})
            raise
        finally:
            self.active[kind] -= 1
        count['completed'] += 1
        rec['status'] = 'returned'
        try:
            self.store.emit('operation_returned', receipt=deepcopy(rec))
        except BaseException as exc:
            self.fault = True
            if not cleanup:
                raise
            self.secondary.append({'stage': 'cleanup_return', 'type': type(exc).__name__})
        return result
