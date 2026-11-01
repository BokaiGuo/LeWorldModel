"""Fixed same-prefix interventions; no neural model, outcome selection or fitting.

Same two calibration roots, three model-only roles, all ten suffix coordinates,
two fixed logit radii and both signs. Not an independent P1 confirmation.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import inspect
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_witness_physical_20261027_v1 as base

NAME = 'p1_same_prefix_physical_main_20261027_v1'
ROOT, source, core, require = base.ROOT, base.source, base.core, base.require
PARENT = ROOT/'results/p1_witness_physical_20261027_v1_prepared01'
PARENT_SHA = 'e4fc7b7adeac3a23036196adaadfbbe69d8873f1e8649ebbf5fe9e8fb1eca0ff'
ROLES = ('zero', 'bank_greedy', 'two_stage')
RADII = (.25, .5)
CAP, FLOOR = 64*1024**2, 200*1024**2


def suffixes():
    items = [dict(index=0, coordinate=None, radius=0., sign=0, logits=[0.]*10)]
    for radius in RADII:
        for coordinate in range(10):
            for sign in (-1, 1):
                x = np.zeros(10, np.float32); x[coordinate] = sign*radius
                items.append(dict(index=len(items), coordinate=coordinate, radius=radius,
                                  sign=sign, logits=x.astype(float).tolist()))
    return items


def choose_prefixes(task, report):
    """Only saved predicted step5 costs. Never read physical successor outcomes."""
    base.witness_selection(task, report)
    representatives = [report['candidates'][4*i] for i in range(16)]
    for i, candidate in enumerate(representatives):
        require(all(c['commands'][:5] == candidate['commands'][:5] and
                    c['contexts'][0] == candidate['contexts'][0]
                    for c in report['candidates'][4*i:4*i+4]), 'same bank prefix across suffixes')
    costs = [core.model_cost(task, c['contexts'][0]['latent_history'][-1]) for c in representatives]
    greedy = min(range(16), key=lambda i: (costs[i], i))
    winner = report['winner']['prefix_index']
    return [dict(role='zero', bank_index=None, prefix=np.zeros((5, 2)).tolist(), model_Q5=None),
            *[dict(role=role, bank_index=i, prefix=representatives[i]['commands'][:5], model_Q5=costs[i])
              for role, i in (('bank_greedy', greedy), ('two_stage', winner))]]


def contract():
    return dict(roots=2, roles=list(ROLES), suffixes=41, branches=246,
        radii=list(RADII), coordinates=10, signs=[-1, 1],
        order='original source slot, zero/bank_greedy/two_stage, zero then radius/coordinate/sign',
        reset_cap=492, step_cap=2460, render_cap=2706, png_cap=738,
        partial_policy='honor termination/guard; retain every planned denominator, no replacement',
        prefix_equality='state7/RGB and both-body kinematics/forces; contact solver caches not serialized',
        neural_calls=0, training=False, fitted_controller=False, new_Q=False,
        output_cap_bytes=CAP, floor_bytes=FLOOR, independent_samples=0, P1_admission=False,
        interpretation='Finite intervention measurement, not global controllability, stationarity or P1 success')


def collect():
    p = base.verify(PARENT, PARENT_SHA)
    bindings = {str(PARENT/'prepare.json'): PARENT_SHA, **p['sources'], **p['inputs']['bindings']}
    plans, roles = [], []
    for old in [a for a in p['inputs']['plans'] if a['arm'] == 'two_stage']:
        ref = next(t for t in base.witness.verify(Path(p['inputs']['arguments']['witness_prepared']),
                   p['inputs']['arguments']['witness_prepare_sha'])['inputs']['tasks']
                   if t['root_sha256'] == old['root_sha256'])
        task = ref['task']; report = base.read(old['origin'])
        chosen = choose_prefixes(task, report)
        for role in chosen:
            prefix = np.asarray(role['prefix'], np.float32)
            require(prefix.shape == (5, 2), 'five original raw commands')
            roles.append(dict(root_sha256=old['root_sha256'], task_id=core.task_id(task), **role))
            for probe in suffixes():
                logits = np.asarray(probe['logits'], np.float32)
                require(float(np.linalg.norm(logits.astype(float))) <= task['packet_logit_cap'], 'legal original suffix packet ball')
                commands = np.concatenate((prefix, np.tanh(logits).reshape(5, 2)))
                base.raw_commands(commands)
                plan = deepcopy(old)
                plan.update(arm=role['role']+'_probe_'+str(probe['index']), role=role['role'],
                    bank_index=role['bank_index'], probe=probe, commands=commands.astype(float).tolist(),
                    prediction_sha256=None, origin='fixed_intervention_not_a_predicted_H2_winner',
                    parent_selection_sha256=old['selection_sha256'],
                    search_cost=dict(new_neural_calls=0, inherited_search_separate=True))
                plan.pop('selection_sha256'); plan['selection_sha256'] = source.digest(plan)
                plans.append(plan)
    require(len(plans) == 246 and len(roles) == 6, 'full two-root three-role 41-probe census')
    base.bindings_check(bindings)
    return dict(plans=plans, roles=roles, source_plan=p['inputs']['source_plan'], bindings=bindings)


def own_sources():
    paths = set(base.own_sources()) | {str(Path(__file__).resolve()),
        str(ROOT/'tests'/('test_'+NAME+'.py')), str(ROOT/'innerdocs'/(NAME+'.md'))}
    return {path: base.sha(path) for path in sorted(paths)}


def material():
    return dict(schema=NAME+'.prepare', contract=contract(), inputs=collect(), sources=own_sources())


def prepare(output):
    p = material(); raw = (source.canonical(p)+'\n').encode(); output = Path(output).absolute()
    require(len(raw) < 2*1024**2 and not output.exists(), 'exclusive bounded prepare')
    require(shutil.disk_usage(output.parent).free >= FLOOR+len(raw), 'prepare plus floor')
    with output.open('xb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    return dict(prepared=str(output), prepare_sha256=base.sha(output), bytes=len(raw))


def verify(path, expected):
    require(base.sha(path) == expected, 'external prepare SHA')
    p = base.read(path); require(p == material(), 'full source/role/probe reconstruction')
    return p


class Store(base.Store):
    def __init__(self, path):
        self.path = Path(path).absolute(); self.used = 0; self.previous = None; self.events = 0
        self.active_body = None; self.dynamics = {}
        require(shutil.disk_usage(self.path.parent).free >= FLOOR+CAP, 'full64MiB output and200MiB floor')
        self.path.mkdir(exist_ok=False)
    def check(self, size=65536):
        actual = sum(f.stat().st_size for f in self.path.iterdir() if f.is_file()) if self.path.exists() else 0
        require(max(actual, self.used)+size <= CAP, '64MiB full output cap')
        require(shutil.disk_usage(self.path).free >= FLOOR+size, '200MiB write floor')

    def emit(self, kind, **values):
        if kind == 'fresh_rgb_observed' and values['receipt']['step'] in (0, 5, 10):
            require(self.active_body is not None, 'live native body at checkpoint')
            snapshot = body_dynamics(self.active_body)
            values['body_dynamics'] = snapshot
            self.dynamics.setdefault(values['selection_sha256'], {})[values['receipt']['step']] = snapshot
        return super().emit(kind, **values)


def body_dynamics(body):
    """Read dynamic body values without changing simulator or wrapping angles.

    This does not claim serialization of contact solver warm-start caches.
    """
    result = {}
    for name in ('agent', 'block'):
        b = getattr(body, name)
        a = np.asarray([*b.position, *b.velocity, b.angle, b.angular_velocity, *b.force, b.torque], dtype=float)
        require(a.shape == (9,) and np.isfinite(a).all(), 'finite native body dynamics')
        result[name] = dict(position=a[:2].tolist(), velocity=a[2:4].tolist(),
            unwrapped_angle=float(a[4]), angular_velocity=float(a[5]), force=a[6:8].tolist(), torque=float(a[8]),
            is_sleeping=bool(b.is_sleeping))
    return result


def geometric_delta(a, b):
    """State7 subtraction, angle wrapped separately from official evaluator."""
    delta = source.vector(a)-source.vector(b)
    delta[4] = np.arctan2(np.sin(delta[4]), np.cos(delta[4]))
    return delta


def analyze(plans, rows, goals):
    require(len(plans) == len(rows) == 246, 'keep all planned branches')
    groups = {}
    for plan, row in zip(plans, rows, strict=True):
        require(all(row[k] == plan[k] for k in ('root_sha256', 'arm', 'selection_sha256')), 'stable plan/row join')
        groups.setdefault((plan['root_sha256'], plan['role']), []).append((plan, row))
    result = []
    for (root, role), pairs in groups.items():
        require([p['probe']['index'] for p, _ in pairs] == list(range(41)), 'all probes including failures')
        center = pairs[0][1]
        completed = [r for _, r in pairs if r.get('status') == 'completed' and r.get('full_horizon')]
        common = groups[root, 'zero'][0][1]
        summary = dict(root_sha256=root, role=role, planned=41, completed=len(completed),
                       native_successes=sum(bool(r['official_evaluation']['success']) for r in completed),
                       centered_responses=[], unavailable_probe_indices=[])
        for plan, row in pairs:
            i = plan['probe']['index']
            if not (row.get('status') == center.get('status') == common.get('status') == 'completed'
                    and row.get('full_horizon') and center.get('full_horizon') and common.get('full_horizon')):
                summary['unavailable_probe_indices'].append(i); continue
            require(row['checkpoints'][1]['step'] == center['checkpoints'][1]['step'] == 5 and
                    row['checkpoints'][1]['state'] == center['checkpoints'][1]['state'] and
                    row['checkpoints'][1]['rgb_sha256'] == center['checkpoints'][1]['rgb_sha256'] and
                    row['body_dynamics']['5'] == center['body_dynamics']['5'],
                    'exact same actual prefix before each probe')
            displacement = geometric_delta(row['final_state'], center['final_state'])
            own = geometric_delta(goals[root], center['checkpoints'][1]['state'])
            shared = geometric_delta(goals[root], common['checkpoints'][1]['state'])
            summary['centered_responses'].append(dict(probe_index=i, probe=plan['probe'],
                endpoint_delta_state7=displacement.tolist(),
                own_position_direction_response=float(own[:4]@displacement[:4]/max(np.linalg.norm(own[:4]), 1e-12)),
                common_position_direction_response=float(shared[:4]@displacement[:4]/max(np.linalg.norm(shared[:4]), 1e-12)),
                native_distance_change=float(row['official_evaluation']['state_distance']-center['official_evaluation']['state_distance']),
                wrapped_object_angle_change=float(row['pose_diagnostics']['independent']['wrapped_angle_error']-
                                                 center['pose_diagnostics']['independent']['wrapped_angle_error'])))
        result.append(summary)
    return dict(groups=result, P1_admission=False, fitted_controller=False,
        scope='same roots fixed interventions; directional responses are diagnostics, no posthoc action selection')


def run(prepared, expected, output, factory, *, authorize_physics=False):
    require(authorize_physics is True, 'MAIN explicit physics authorization')
    p = verify(prepared, expected); store = Store(output)
    plans = p['inputs']['plans']; rows = []
    counts = {k:dict(attempted=0, completed=0) for k in ('reset', 'step', 'render')}
    errors = []; goals = {}
    try:
        store.emit('run_started', prepare_sha256=expected, contract=contract())
        for plan in plans:
            row = dict(root_sha256=plan['root_sha256'], arm=plan['arm'],
                       selection_sha256=plan['selection_sha256'], status='not_started')
            env = None
            try:
                store.check(262144); env = factory(); body = env.unwrapped
                path = inspect.getsourcefile(type(body))
                require(path is not None and p['inputs']['source_plan']['source_closure'].get(str(Path(path).resolve())) ==
                        p['inputs']['source_plan']['environment_sha256'], 'original native constructor source')
                require(body.relative is True and body.action_scale == 100 and body.render_mode == 'rgb_array' and body.render_size == 224,
                        'unchanged raw mapping/render contract')
                packet = source.load_raw_source(plan['raw_source_file'], p['inputs']['source_plan'])
                goals[plan['root_sha256']] = packet['goal'].tolist()
                store.active_body = body
                row = base.branch(plan, p['inputs']['source_plan'], env, store, counts)
                row['body_dynamics'] = {str(k):v for k, v in store.dynamics.get(plan['selection_sha256'], {}).items()}
            except (Exception, KeyboardInterrupt) as exc:
                row.update(status='failed', first_error=dict(type=type(exc).__name__, message=str(exc)))
            finally:
                if env is not None:
                    try: env.close()
                    except Exception as exc: row.update(status='failed', close_error=str(exc))
                store.active_body = None
                rows.append(row); store.emit('branch_closed', receipt=row)
        analysis = analyze(plans, rows, goals); store.json('analysis.json', analysis)
    except (Exception, KeyboardInterrupt) as exc:
        errors.append(dict(type=type(exc).__name__, message=str(exc)))
    try: verify(prepared, expected)
    except Exception as exc: errors.append(dict(stage='final_binding', type=type(exc).__name__, message=str(exc)))
    complete = len(rows) == 246 and not errors and all(r.get('status') == 'completed' for r in rows)
    if complete:
        try:
            require(counts == {k:dict(attempted=n, completed=n) for k, n in dict(reset=492, step=2460, render=2706).items()},
                    'complete public operation census')
        except Exception as exc:
            errors.append(dict(stage='final_counts', type=type(exc).__name__, message=str(exc))); complete = False
    status = 'same_prefix_interventions_completed' if complete else 'partial_or_failed'
    store.emit('run_finished', status=status, counts=counts, errors=errors)
    report = dict(schema=NAME+'.report', status=status, prepare_sha256=expected, counts=counts,
        errors=errors, rows=rows, journal_sha256=base.sha(store.path/'journal.jsonl'),
        contract=contract(), new_neural_calls=0, P1_admission=False, new_independent_samples=0,
        internal_physics='reset/substep costs implied by source, not separately instrumented')
    store.json('report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run'))
    parser.add_argument('--prepared', type=Path); parser.add_argument('--prepare-sha')
    parser.add_argument('--output', type=Path); parser.add_argument('--authorize-physics', action='store_true')
    a = parser.parse_args()
    if a.command == 'prepare': result = prepare(a.output)
    elif a.command == 'verify': result = dict(status=verify(a.prepared, a.prepare_sha)['schema'])
    else:
        def factory():
            import gymnasium as gym
            import stable_worldmodel
            return gym.make('swm/PushT-v1', render_mode='rgb_array', resolution=224)
        r = run(a.prepared, a.prepare_sha, a.output, factory, authorize_physics=a.authorize_physics)
        result = {k:v for k, v in r.items() if k != 'rows'}
    print(source.canonical(result))


if __name__ == '__main__': main()
