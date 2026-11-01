"""MAIN-only, predeclared moving baseline physical replay; no neural execution."""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import importlib
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import p1_constructive_moving_history_physical_20261027_v1 as kernel

NAME = 'p1_moving_baseline_physical_20261027_v1'
AUTH = 'MAIN_MOVING_BASELINE_SELECTED_PHYSICS_ONLY'
ROOT = kernel.ROOT
CAP, FLOOR, FINAL, PREPARE_CAP = 16*1024**2, 200*1024**2, 512*1024, 256*1024
require, sha, read, raw, source = kernel.require, kernel.sha, kernel.read, kernel.raw, kernel.source
AUTHORITIES = {
    'sampling': ('p1_moving_sampling_baselines_cuda_20261027_v1',
                 'f30a1c5ac5ef618cfe3add37326aae29513278fe7c4319b5a743d9f6422300e6',
                 ('CEM', 'kernel_faithful_mppi_nominal'), 'rows'),
    'gradient': ('p1_moving_gradient_baseline_cuda_20261027_v1',
                 '3224d6a094e74b3845c2c695c6c85fe29c6522cd4a88c91c57f2cc03d127a70e',
                 ('moving_gradient_open_loop',), 'rows'),
    'kinodynamic': ('p1_moving_kinodynamic_baseline_cuda_20261027_v1',
                    '66c090cb98ad8a954bae5c94e39a8cbb707b5a2e8dc82027e63fa00f2583ae6d',
                    ('fixed_H2_kinodynamic_beam16x4',), 'plans'),
}
SOURCE_SHA = 'c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6'


def fields(value, keys):
    require(isinstance(value, dict) and set(value) == set(keys.split()), 'exact fields: '+keys)


def absolute(value):
    require(type(value) is str and Path(value).is_absolute(), 'explicit absolute path')
    return str(Path(value).resolve())


def checked_arms(arms):
    allowed = {arm for _, _, names, _ in AUTHORITIES.values() for arm in names}
    require(isinstance(arms, list) and 1 <= len(arms) <= 4 and
            all(type(a) is str and a in allowed for a in arms) and len(set(arms)) == len(arms),
            'predeclared unique known arms, no outcome selection')
    return list(arms)


def ownconsume(entry):
    """Whitelist selected module BEFORE import. Never accept callable/module paths."""
    fields(entry, 'authority output report_sha256')
    require(type(entry['authority']) is str and entry['authority'] in AUTHORITIES, 'unknown authority')
    name, expected, _, key = AUTHORITIES[entry['authority']]
    path = ROOT/'experiments/spectral_reachability'/(name+'.py')
    require(sha(path) == expected, 'MAIN frozen authority SHA')
    directory = Path(absolute(entry['output']))
    require(sha(directory/'report.json') == entry['report_sha256'], 'external actual compiler report SHA')
    module = importlib.import_module('experiments.spectral_reachability.'+name)
    require(Path(module.__file__).resolve() == path.resolve() and sha(path) == expected, 'own module identity')
    result = module.consume(directory, entry['report_sha256'])
    require(result['schema'] == name+'.consume' and result['P1_admission'] is False and
            isinstance(result[key], list), 'own saved consume only')
    report = read(directory/'report.json')
    prepared = Path(absolute(report['prepared']))
    require(sha(prepared) == report['prepare_sha256'], 'actual compiler prepare seal')
    p = read(prepared)
    bindings = dict(p['bindings_sha256'])
    bindings.update({str(prepared): report['prepare_sha256'], str(directory/'report.json'): entry['report_sha256'],
                     str(path.resolve()): expected})
    bindings.update({str(directory/n): h for n, h in report['artifact_sha256'].items()})
    for f, h in bindings.items():
        require(sha(f) == h, 'ownconsume closure: '+f)
    return deepcopy(result[key]), bindings


def task_bridge(plan, ref, bindings):
    """Evaluator bridge: no witness commands; source goal is never a control."""
    from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as encoder
    path = absolute(plan['task_export'])
    interface = encoder.load_task(path, plan['task_sha256'])
    envelope = encoder.read(path)
    tc = envelope['task_contract']
    require(type(plan['source_slot']) is int and envelope['source_slot'] == plan['source_slot'] and
            plan['blind_sha256'] == envelope['blind_sha256'] == ref['blind_sha256'] and
            plan['task_id'] == interface['task_id'], 'same source slot/blind/task identity')
    require(tc['pixel_sha256'] == [kernel.moving.image_sha(a) for a in [*ref['history'], ref['goal_rgb']]] and
            tc['raw_action_sha256'] == kernel.moving.image_sha(ref['raw_history']) and
            kernel.moving.exact(np.asarray(interface['gn_task']['root_raw_action_history'], np.float32),
                                kernel.raw_history(ref['raw_history'])), 'same actual history/goal pixels/raw')
    bindings[path] = plan['task_sha256']


def selection(manifest, data):
    fields(manifest, 'schema sourcepack sourcepack_sha256 source_prepared source_prepare_sha256 roots arms authorities selection_fixed_before_physics physical_outcome_selection wall_seconds')
    require(manifest['schema'] == NAME+'.selection' and manifest['selection_fixed_before_physics'] is True and
            manifest['physical_outcome_selection'] is False, 'new predeclared no-outcome selection')
    require(type(manifest['wall_seconds']) is int and 0 < manifest['wall_seconds'] <= 600, 'wall<=600')
    arms = checked_arms(manifest['arms'])
    require(manifest['roots'] == [dict(root_slot=i, root_sha256=r['root_sha256'])
                                 for i, r in enumerate(data['references'])] and
            all(type(r['root_slot']) is int for r in manifest['roots']), 'two original roots in order')
    expected = [k for k, (_, _, names, _) in AUTHORITIES.items() if any(a in arms for a in names)]
    require(isinstance(manifest['authorities'], list) and
            [e['authority'] for e in manifest['authorities']] == expected, 'exact used authority order/denominator')
    plans_by_key, bindings = {}, {}
    for entry in manifest['authorities']:
        rows, bound = ownconsume(entry)
        allowed = AUTHORITIES[entry['authority']][2]
        seen = set()
        for row in rows:
            s, a = row['source_slot'], row['arm']
            require(type(s) is int and s in (0, 1) and a in allowed and (s, a) not in seen,
                    'ownconsume no unknown arm/slot/duplicates')
            seen.add((s, a))
            task_bridge(row, data['references'][s], bindings)
            if a not in arms:
                continue  # only a PREDECLARED subset, never based on cost/outcome
            literal = kernel.commands(row['commands']).tolist()
            plan = dict(root_slot=s, arm=a, commands=literal, task_id=row['task_id'],
                        blind_sha256=row['blind_sha256'], task_export=row['task_export'], task_sha256=row['task_sha256'],
                        compiler_authority=entry['authority'], compiler_report=str(Path(entry['output'])/'report.json'),
                        compiler_report_sha256=entry['report_sha256'], own_plan_sha256=source.digest(row))
            plan['selection_digest'] = source.digest(plan)
            plans_by_key[s, a] = plan
        require(seen == {(s, a) for s in (0, 1) for a in {a for _, a in seen}}, 'complete ownconsume roots')
        for f, h in bound.items():
            require(f not in bindings or bindings[f] == h, 'nonconflicting source seals')
            bindings[f] = h
    require(set(plans_by_key) == {(s, a) for s in (0, 1) for a in arms}, 'all fixed slots x requested arms')
    return [plans_by_key[s, a] for s in (0, 1) for a in arms], bindings


def contract(arms, wall):
    arms = checked_arms(arms)
    c = kernel.low.contract(2*len(arms), wall)
    c.update(arms=arms, output_cap_bytes=CAP, final_reserve_bytes=FINAL,
             warmup_raw=kernel.moving.warmup_commands().tolist(),
             scope='old two moving calibration tasks; fixed ownconsume arms; no P1/equal total compute claim')
    return c


def physical_files():
    """Read/hash local static import closure; never execute imported run paths."""
    seen, todo = set(), [Path(__file__).resolve()]
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('experiments'):
                names = [node.module]+[node.module+'.'+a.name for a in node.names]
            for name in names:
                if not name.startswith('experiments.'):
                    continue
                candidate = ROOT/Path(*name.split('.'))
                for child in (candidate.with_suffix('.py'), candidate/'__init__.py'):
                    if child.is_file() and child.resolve() not in seen:
                        todo.append(child.resolve())
    for package in (ROOT/'experiments/__init__.py', ROOT/'experiments/spectral_reachability/__init__.py'):
        if package.is_file():
            seen.add(package.resolve())
    return sorted(seen)


def material(manifest, expected):
    path = Path(absolute(str(manifest)))
    require(path.stat().st_size < PREPARE_CAP and sha(path) == expected, 'external bounded manifest SHA')
    m = read(path)
    require(m['sourcepack_sha256'] == SOURCE_SHA, 'only approved old moving source')
    for k in ('sourcepack', 'source_prepared'):
        absolute(m[k])
    data = kernel.load_source(m['sourcepack'], m['sourcepack_sha256'], m['source_prepared'], m['source_prepare_sha256'])
    plans, bindings = selection(m, data)
    bindings.update(data['bindings'])
    bindings[str(path)] = expected
    # Static physical import closure plus dependency files bound by source and ownconsume.
    for f in physical_files():
        bindings[str(f)] = sha(f)
    for f in (Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')):
        bindings[str(f)] = sha(f)
    require(sha(path) == expected, 'manifest read TOCTOU')
    for f, h in bindings.items():
        require(sha(f) == h, 'final prepare input seal')
    return dict(schema=NAME+'.prepare', manifest=str(path), manifest_sha256=expected, plans=plans,
                contract=contract(m['arms'], m['wall_seconds']), bindings=bindings,
                sourcepack_sha256=m['sourcepack_sha256'], source_prepare_id=source.digest(data['plan']))


def prepare(manifest, expected, output):
    p = material(manifest, expected); payload = raw(p); path = Path(output)
    require(len(payload) < PREPARE_CAP and shutil.disk_usage(path.parent).free >= FLOOR+len(payload),
            'prepare<256KiB/200MiB floor')
    with path.open('xb') as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())
    return dict(prepared=str(path.resolve()), prepare_sha256=sha(path), bytes=len(payload))


def verify(path, expected):
    require(Path(path).stat().st_size < PREPARE_CAP and sha(path) == expected, 'external small prepare SHA')
    p = read(path)
    require(p == material(p['manifest'], p['manifest_sha256']), 'entire selection/source/ownconsume reconstruction')
    require(sha(path) == expected, 'prepare TOCTOU')
    return p


class Store(kernel.Store):
    def check(self, size=4096):
        used = sum(f.stat().st_size for f in self.path.iterdir() if f.is_file())
        require(used+size <= CAP-FINAL and shutil.disk_usage(self.path).free >= FLOOR+size, '16MiB/200MiB gate')

    def write(self, name, value, append=False, final=False):
        require(Path(name).name == name and name not in ('', '.', '..') and not (self.path/name).is_symlink(), 'flat output')
        used = sum(f.stat().st_size for f in self.path.iterdir() if f.is_file())
        require(used+len(value) <= CAP-(0 if final else FINAL) and
                shutil.disk_usage(self.path).free >= FLOOR+len(value), '16MiB includes all partial/final; reserve200MiB')
        with (self.path/name).open('ab' if append else 'xb') as f:
            f.write(value); f.flush(); os.fsync(f.fileno())
        fd = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return sha(self.path/name)


def execute(p, data, store, factory, *, authorization):
    require(authorization == AUTH, 'MAIN explicit physics authorization')
    arms = checked_arms(p['contract']['arms'])
    require([(x['root_slot'], x['arm']) for x in p['plans']] == [(s, a) for s in (0, 1) for a in arms] and
            len(data['references']) == 2, 'complete frozen branch denominator')
    rows = [dict(root_slot=x['root_slot'], arm=x['arm'], selection_digest=x['selection_digest'], status='not_started',
                 first_error=None, secondary_errors=[], warmup_returned=0, own_attempted=0, own_returned=0,
                 initial_exact=False, full_horizon=False, native_success=False, terminated=False, truncated=False,
                 PNG=0, checkpoints=[]) for x in p['plans']]
    result = dict(schema=NAME+'.report', status='running', rows=rows, errors=[], contract=p['contract'],
                  sourcepack_sha256=p['sourcepack_sha256'], fixed_branch_denominator=len(rows),
                  P1_admission=False, training=False, new_neural_calls=0, new_independent_samples=0)
    ops = kernel.Ops(p, store)
    def error(stage, exc):
        result['errors'].append(dict(stage=stage, type=type(exc).__name__, message=str(exc)))
    try:
        store.emit('run_started', prepare=p)
        for i, (plan, row) in enumerate(zip(p['plans'], rows, strict=True)):
            env = None
            try:
                store.check(FINAL); env = factory(); kernel.moving.validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size == 224 and
                        env.unwrapped.render_mode == 'rgb_array', 'original native target-free RGB224')
                row['status'] = 'running'
                with kernel.native_hooks(env, ops, row):
                    kernel.branch(env, data['references'][plan['root_slot']], data['plan'], plan, row, ops, store, i)
            except BaseException as exc:
                kernel.retain(row, exc, 'setup_or_branch')
            finally:
                if env is not None:
                    try:
                        env.close()
                    except BaseException as exc:
                        kernel.retain(row, exc, 'close')
                store.emit('branch_closed', receipt=row)
    except BaseException as exc:
        error('execution', exc)
    try:
        for f, h in p['bindings'].items():
            require(sha(f) == h, 'final source seal: '+f)
    except BaseException as exc:
        error('final_bindings', exc)
    clean = not result['errors'] and all(r['status'] in ('horizon_completed', 'native_success', 'native_stopped') for r in rows)
    if clean:
        try:
            c = ops.counts; n = len(rows)
            require(all(v['attempted'] == v['completed'] for v in c.values()) and
                    c['reset']['completed'] == 2*n and c['set_state']['completed'] == 4*n and
                    c['goal_render']['completed'] == 2*n and c['goal_bind']['completed'] == n and
                    c['public_step']['completed'] == sum(r['warmup_returned']+r['own_returned'] for r in rows) and
                    c['physics_step']['completed'] == 4*n+10*c['public_step']['completed'] and
                    c['explicit_render']['completed'] == sum(r['PNG'] for r in rows), 'actual returned resource census')
        except BaseException as exc:
            error('final_census', exc); clean = False
    result.update(status='moving_baseline_physical_recorded' if clean else 'partial_or_failed', counts=ops.counts,
                  full_horizons=sum(r['full_horizon'] for r in rows),
                  native_successes=sum(r['status'] == 'native_success' for r in rows), wall_seconds=time.monotonic()-ops.started)
    try:
        store.emit('run_finished', status=result['status'], counts=ops.counts, errors=result['errors'])
        result.update(journal_sha256=sha(store.path/'journal.jsonl'), journal_chain_tail=store.previous)
    except BaseException as exc:
        error('final_journal', exc); result['status'] = 'partial_or_failed'
    try:
        store.put('report.json', result, final=True)
    except BaseException as exc:
        error('final_report', exc); result['status'] = 'partial_or_failed'
    return result


def run(prepared, expected, output, factory, *, authorization):
    require(authorization == AUTH, 'MAIN authorization before input/output work')
    p = verify(prepared, expected); m = read(p['manifest'])
    data = kernel.load_source(m['sourcepack'], m['sourcepack_sha256'], m['source_prepared'], m['source_prepare_sha256'])
    require(shutil.disk_usage(Path(output).parent).free >= FLOOR+CAP, '16MiB run headroom plus200MiB reserve')
    store = Store(output)
    store.put('execution_binding.json', dict(prepared=str(Path(prepared).resolve()), prepare_sha256=expected))
    return execute(p, data, store, factory, authorization=authorization)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run'))
    for key in ('manifest', 'manifest-sha', 'prepared', 'prepare-sha', 'output'):
        parser.add_argument('--'+key)
    parser.add_argument('--authorize-physics', action='store_true'); a = parser.parse_args()
    if a.command == 'prepare':
        r = prepare(a.manifest, a.manifest_sha, a.output)
    elif a.command == 'verify':
        r = dict(schema=verify(a.prepared, a.prepare_sha)['schema'])
    else:
        require(a.authorize_physics, 'MAIN CLI authorization')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1', render_mode='rgb_array', resolution=224, with_target=False, render_action=False)
        r = run(a.prepared, a.prepare_sha, a.output, factory, authorization=AUTH)
    print(source.canonical({k: v for k, v in r.items() if k != 'rows'}))
    if a.command == 'run' and r['status'] != 'moving_baseline_physical_recorded':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
