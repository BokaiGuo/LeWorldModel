"""K17 outer adapter separating scheduler slot from native reset slot.

Reuses frozen episode and meter; no factory, entry point, or execution authority.
"""
from copy import deepcopy
import time
from experiments.spectral_reachability import agent_p3_c3_k17_native_outer_20261029_v1 as old
from experiments.spectral_reachability import agent_p3_c3_k17_native_meter_20261029_v1 as meter_module


class ScopedStore:
    def __init__(self, store):
        self.store, self.path, self.where = store, store.path, {}

    def __getattr__(self, name):
        return getattr(self.store, name)

    def emit(self, kind, **fields):
        values = deepcopy(fields)
        if self.where:
            slot = old.ORDER[self.where['branch']][0]
            if kind == 'placement_recipe':
                values['reset_identity'] = values.pop('slot')
            if 'receipt' in values and kind.startswith('operation_'):
                values['receipt']['slot'] = slot
            elif kind not in ('run_started',):
                for key, value in self.where.items():
                    values.setdefault(key, value)
                values.setdefault('slot', slot)
        return self.store.emit(kind, **values)


def execute_material(data, source_rows, store, factory, *, started=None, validate_instance=None):
    if len(source_rows) != 6:
        raise ValueError('fixed six source records')
    scoped = ScopedStore(store)
    meter = meter_module.Meter(scoped, started=started)
    rows, seen, errors = old.make_rows(), [], []
    scoped.emit('run_started', denominator=6, order=[list(x) for x in old.ORDER])
    for row, (item, task, ref) in zip(rows, source_rows, strict=True):
        if errors:
            break
        if (item['slot'], item['root'], item['mode'], item['K']) != (*old.ORDER[row['index']], 17):
            raise ValueError('source row order')
        # Native place() writes slot=<reset identity>; do not put slot in **where.
        meter.where = {'branch': row['index'], 'root_slot': row['root_slot'], 'mode': row['mode']}
        scoped.where = meter.where
        env = None
        row['status'] = 'running'
        try:
            env = meter.call('constructor', factory)
            if validate_instance is not None:
                validate_instance(env)
            if any(env is x or env.unwrapped is x.unwrapped for x in seen):
                raise ValueError('fresh native environment per slot')
            seen.append(env)
            old.h3.validate_env(env.unwrapped)
            if env.unwrapped.render_action is not False or env.unwrapped.render_size != 224 or env.unwrapped.render_mode != 'rgb_array':
                raise ValueError('native renderer contract')
            with old.h3.native_hooks(env, meter, row['secondary_errors']):
                result = old.episode.episode(env, data['roots'][row['root_slot']], data['source_plan'],
                                             item, task, ref, row, meter, scoped)
            row['status'], row['result'] = result['status'], result
        except old.h3.native.allmode.NativeStop as exc:
            row['status'] = 'native_stopped'
            row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        except BaseException as exc:
            row['status'] = 'engineering_failed'
            row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
            errors.append({'branch': row['index'], **row['error']})
        finally:
            if env is not None:
                try:
                    meter.call('close', env.close, cleanup=True)
                except BaseException as exc:
                    errors.append({'branch': row['index'], 'type': type(exc).__name__, 'message': str(exc)})
                    row['status'] = 'engineering_failed'
            if row['secondary_errors']:
                errors.append({'branch': row['index'], 'type': 'hook_cleanup_failure'})
                row['status'] = 'engineering_failed'
            for phase, label in (('warmup', 'warmup'), ('plan', 'own')):
                calls = [r for r in meter.receipts if r.get('branch') == row['index'] and
                         r['resource'] == 'public_step' and r.get('phase') == phase]
                row[label + '_attempted'] = len(calls)
                row[label + '_returned'] = sum(r['status'] == 'returned' for r in calls)
        try:
            scoped.emit('branch_closed', row=deepcopy(row))
        except BaseException as exc:
            errors.append({'branch': row['index'], 'type': type(exc).__name__, 'message': str(exc)})
            break
    return {'rows': rows, 'fixed_denominator': 6, 'errors': errors,
            'counts': deepcopy(meter.counts), 'receipts': deepcopy(meter.receipts),
            'secondary': deepcopy(meter.secondary), 'wall_seconds': time.monotonic() - meter.started,
            'production_authorized': False, 'physical_release': False}
