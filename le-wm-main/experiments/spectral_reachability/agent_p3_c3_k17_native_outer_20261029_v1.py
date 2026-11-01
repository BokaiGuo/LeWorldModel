"""K17 six-slot execution core; not yet a production CLI or strict consumer."""
from copy import deepcopy
import time
from experiments.spectral_reachability import agent_p3_c3_k17_native_episode_20261029_v1 as episode
from experiments.spectral_reachability import agent_p3_c3_k17_native_meter_20261029_v1 as meter_module
from experiments.spectral_reachability import agent_p3_c3_k17_physical_gate_20261029_v1 as gate

h3 = episode.h3
ORDER = ((3, 0, 'fixedH1'), (4, 0, 'staticH3_progress'), (5, 0, 'adaptive'),
         (9, 1, 'fixedH1'), (10, 1, 'staticH3_progress'), (11, 1, 'adaptive'))


def material():
    """Load the pinned paid replays, source references and original native input."""
    data = h3.native.allmode.source_material()
    replay_rows = gate.bridge.build()['rows']
    material_rows = []
    for slot, root, mode in ORDER:
        item, task, ref = gate._authorities_from_snapshot(slot, replay_rows)
        if (item['slot'], item['root'], item['mode'], item['K']) != (slot, root, mode, 17):
            raise ValueError('fixed slot identity')
        native_root = data['roots'][root]
        if (native_root['slot'] != ref['reset_identity'] or native_root['warm_trace'] != ref['warm_trace']
                or native_root['root_sha256'] != ref['root_sha256']):
            raise ValueError('native reset/history/root source mismatch')
        material_rows.append((item, task, ref))
    return data, material_rows


def make_rows():
    rows = []
    for index, (slot, root, mode) in enumerate(ORDER):
        rows.append(dict(index=index, slot=slot, root_slot=root, mode=mode, condition='off',
            status='not_started', error=None, secondary_errors=[], warmup_attempted=0, warmup_returned=0,
            own_attempted=0, own_returned=0, PNG=0, trajectory=[], checkpoints=[], missing_checkpoints=[0, 5],
            last_observation=None, prefix51=dict(complete=False, frames=[]), source_unpaired_on_steps=[],
            result=None))
    return rows


def execute_material(data, source_rows, store, factory, *, started=None):
    """Six-slot core. Caller must authenticate inputs and publish/consume output.

    No runner entry point uses this function yet: full source sealing and saved
    verification are mandatory before physical execution can be authorized.
    """
    if len(source_rows) != 6:
        raise ValueError('fixed six source records')
    meter = meter_module.Meter(store, started=started)
    rows, seen, errors = make_rows(), [], []
    store.emit('run_started', denominator=6, order=[list(x) for x in ORDER])
    for row, (item, task, ref) in zip(rows, source_rows, strict=True):
        if errors:
            break
        if (item['slot'], item['root'], item['mode'], item['K']) != (*ORDER[row['index']], 17):
            raise ValueError('source row order')
        meter.where = {'branch': row['index'], 'slot': row['slot'], 'root_slot': row['root_slot'], 'mode': row['mode']}
        env = None
        row['status'] = 'running'
        try:
            env = meter.call('constructor', factory)
            if any(env is x or env.unwrapped is x.unwrapped for x in seen):
                raise ValueError('fresh native environment per slot')
            seen.append(env)
            h3.validate_env(env.unwrapped)
            if env.unwrapped.render_action is not False or env.unwrapped.render_size != 224 or env.unwrapped.render_mode != 'rgb_array':
                raise ValueError('native renderer contract')
            with h3.native_hooks(env, meter, row['secondary_errors']):
                result = episode.episode(env, data['roots'][row['root_slot']], data['source_plan'],
                                         item, task, ref, row, meter, store)
            row['status'] = result['status']
            row['result'] = result
        except h3.native.allmode.NativeStop as exc:
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
            store.emit('branch_closed', row=deepcopy(row))
        except BaseException as exc:
            errors.append({'branch': row['index'], 'type': type(exc).__name__, 'message': str(exc)})
            break
    return {'rows': rows, 'fixed_denominator': 6, 'errors': errors,
            'counts': deepcopy(meter.counts), 'receipts': deepcopy(meter.receipts),
            'secondary': deepcopy(meter.secondary), 'wall_seconds': time.monotonic() - meter.started,
            'production_authorized': False, 'physical_release': False}
