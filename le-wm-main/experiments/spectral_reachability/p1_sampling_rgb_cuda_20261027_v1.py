"""MAIN-only encode twelve fixed sampling physical PNGs; no predictor/physics.

prepare/verify/analyze use saved arrays and PNG decoding only. No model imports
outside run. Hash-only manifests reference the original goal/Q/tasks rather than
copying them twelve times. Duplicate PNGs still pay separate encoder calls.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import p1_sampling_physical_20261027_v1 as physical
from experiments.spectral_reachability import p1_witness_rgb_cuda_main_20261027_v1 as rgb24

NAME = 'p1_sampling_rgb_cuda_20261027_v1'
ROOT = rgb24.ROOT
RUN = ROOT / 'results/p1_sampling_physical_20261027_v1_main01'
PREPARED = ROOT / 'results/p1_sampling_physical_20261027_v1_prepared01'
RUN_SHA = 'c3eef328bee57ce5d42b25fcf5a9935af2d2282c9de3f8caf46f2b35b53a2ef6'
PREPARE_SHA = '450e620dbe3bf1ec1957c8d2d06fda9dfc05c7cc6bd50453939dc07a2133033c'
CAP = 2 * 1024**2
FLOOR = 200 * 1024**2
FINAL_RESERVE = 128 * 1024
ARMS = ('cem', 'iid_random')
STEPS = (0, 5, 10)
io, core, source = rgb24.io, physical.core, physical.source
require = physical.require


def canonical(value):
    return (source.canonical(value) + '\n').encode()


def merge_bindings(destination, incoming):
    for path, hashed in incoming.items():
        require(path not in destination or destination[path] == hashed, 'conflicting binding: ' + path)
        destination[path] = hashed


def validate_physical(report, prepared, events):
    require(report['schema'] == physical.NAME + '.report' and
            report['status'] == 'physical_calibration_recorded' and not report['errors'] and
            report['prepare_sha256'] == PREPARE_SHA and report['neural_calls'] == 0 and
            report['P1_admission'] is False, 'complete fixed physical report')
    expected = {k: dict(attempted=n, completed=n) for k, n in dict(reset=8, render=44, step=40).items()}
    require(report['counts'] == expected and report['contract'] == physical.contract(), 'physical contract/census')
    require(events and events[0]['kind'] == 'run_started' and events[0]['prepare_sha256'] == PREPARE_SHA and
            events[-1]['kind'] == 'run_finished' and events[-1]['status'] == report['status'] and
            events[-1]['counts'] == expected and not events[-1]['errors'], 'physical journal terminal')
    require([e['receipt'] for e in events if e['kind'] == 'branch_closed'] == report['rows'],
            'physical rows must equal durable branch receipts')
    require(len(prepared['inputs']['plans']) == len(report['rows']) == 4, 'all four branches')
    for kind, counts in expected.items():
        attempted = [e for e in events if e['kind'] == kind + '_attempted']
        returned = [e for e in events if e['kind'] == kind + '_returned']
        require(len(attempted) == counts['attempted'] and len(returned) == counts['completed'],
                'actual physical journal operation count: ' + kind)
    roots = [p['root_sha256'] for p in prepared['inputs']['plans'][::2]]
    require(len(set(roots)) == 2 and
            [(p['root_sha256'], p['arm']) for p in prepared['inputs']['plans']] ==
            [(root, arm) for root in roots for arm in ARMS], 'original two-root four-arm order')
    for row, plan in zip(report['rows'], prepared['inputs']['plans']):
        require(all(row[k] == plan[k] for k in ('root_sha256', 'arm', 'selection_sha256')) and
                row['status'] == 'completed' and row['full_horizon'] is True and row['initial_exact'] is True and
                row['first_error'] is None and not row['secondary_errors'] and
                row['attempted_steps'] == row['returned_steps'] == 10 and
                [c['step'] for c in row['checkpoints']] == list(STEPS), 'original complete physical branch')
        require(len(row['trace']) == 10, 'all ten commands/observations')
        require([x['step'] for x in row['trace']] == list(range(1, 11)) and
                [x['command'] for x in row['trace']] == plan['commands'] and
                all(x['attempted'] is True and x['returned'] is True and
                    x['terminated'] is False and x['truncated'] is False for x in row['trace']),
                'actual physical chronology and exact selected commands')
        own = [e for e in events if e.get('selection_sha256') == plan['selection_sha256']]
        require([e['plan'] for e in own if e['kind'] == 'branch_started'] == [plan] and
                [e['receipt'] for e in own if e['kind'] == 'checkpoint_saved'] == row['checkpoints'],
                'actual plan and PNGs in durable journal')


def task_table(prepared):
    args = prepared['inputs']['arguments']
    p = physical.sampling.verify(args['sampling_prepared'], args['sampling_prepare_sha'])
    items = p['inputs']['tasks']
    require(len(items) == 2 and len({x['root_sha256'] for x in items}) == 2, 'two original sampling tasks')
    return {x['root_sha256']: x['task'] for x in items}


def task_reference(root, task):
    core.validate_task(task)
    return dict(root_sha256=root, task_id=core.task_id(task),
                goal_Q_context_sha256=source.digest({k: task[k] for k in
                    ('goal', 'projection', 'root_context', 'action_mean', 'action_std', 'model_sha256')}))


def collect():
    require(io.sha(RUN / 'report.json') == RUN_SHA, 'fixed physical report SHA')
    prepared = physical.verify(PREPARED, PREPARE_SHA)
    report = physical.read(RUN / 'report.json')
    require(io.sha(RUN / 'journal.jsonl') == report['journal_sha256'], 'whole physical journal SHA')
    events = source.read_journal(RUN / 'journal.jsonl')  # full hash chain + no torn suffix
    validate_physical(report, prepared, events)
    tasks = task_table(prepared)
    bindings = {str(RUN / 'report.json'): RUN_SHA, str(PREPARED / 'prepare.json'): PREPARE_SHA,
                str(RUN / 'journal.jsonl'): report['journal_sha256']}
    merge_bindings(bindings, prepared['sources'])
    merge_bindings(bindings, prepared['inputs']['bindings'])
    refs, frames = [], []
    for row, plan in zip(report['rows'], prepared['inputs']['plans']):
        task = tasks[plan['root_sha256']]
        reference = task_reference(plan['root_sha256'], task)
        require(reference['task_id'] == plan['task_id'] and
                reference['goal_Q_context_sha256'] == plan['goal_Q_context_sha256'], 'original sampling goal/Q/context')
        if row['arm'] == ARMS[0]:
            refs.append(reference)
        prediction = plan['prediction']
        core.check_prediction(task, prediction)
        require(source.digest(prediction) == plan['prediction_sha256'] and
                prediction['commands'] == plan['commands'], 'original fixed plan prediction')
        original = physical.read(plan['origin'])
        require(original['witness'] == prediction, 'prediction equals original sampling independent replay')
        latent = [task['root_context']['latent_history'][-1],
                  *[c['latent_history'][-1] for c in prediction['contexts']]]
        for index, check in enumerate(row['checkpoints']):
            path = Path(check['path'])
            path = path if path.is_absolute() else ROOT / path
            path = path.absolute()
            require(path.parent == RUN and not path.is_symlink(), 'fixed own-run PNG')
            io.pixels(path, check['file_sha256'], check['rgb_sha256'])
            merge_bindings(bindings, {str(path): check['file_sha256']})
            step = STEPS[index]
            require(check['state'] == (row['checkpoints'][0]['state'] if step == 0 else
                                      row['trace'][step-1]['actual_state']), 'PNG state chronology')
            frames.append(dict(order=len(frames), root_sha256=row['root_sha256'], arm=row['arm'], step=step,
                task_id=reference['task_id'], path=str(path), file_sha256=check['file_sha256'],
                pixel_sha256=check['rgb_sha256'], prediction_sha256=plan['prediction_sha256'],
                selection_sha256=plan['selection_sha256'], predicted_native192=latent[index],
                physical_diagnostic=(row['initial_pose_diagnostics'] if step == 0 else
                                     row['trace'][step-1]['pose_diagnostics'])))
    original, _ = rgb24.storage.native.native_metadata()
    require(all(task['model_sha256'] in original['model_bindings'].values() and
                task['model_config_sha256'] in original['model_bindings'].values() for task in tasks.values()),
            'same original sampling model/config')
    merge_bindings(bindings, original['model_bindings'])
    paths = set(io.source_files()) | {
        Path(__file__).resolve(), Path(physical.__file__).resolve(), Path(rgb24.__file__).resolve(),
        Path(rgb24.storage.__file__).resolve(), Path(rgb24.storage.native.__file__).resolve(),
        Path(rgb24.storage.screen.__file__).resolve(), ROOT / 'tests' / ('test_' + NAME + '.py'),
        ROOT / 'innerdocs' / (NAME + '.md')}
    merge_bindings(bindings, {str(p.resolve()): io.sha(p) for p in paths})
    physical.physical.bindings_check(bindings)
    require(len(frames) == 12 and len({f['path'] for f in frames}) == 12, 'twelve PNG paths, no dedup')
    return dict(schema=NAME + '.protocol', frames=frames, tasks=refs, bindings=bindings, **original,
        encoder_cap=12, projector_cap=12, predict_cap=0, action_encoder_cap=0, AD_cap=0, physical_steps_cap=0,
        output_cap_bytes=CAP, disk_floor_bytes=FLOOR, finalization_reserve_bytes=FINAL_RESERVE,
        normalization='same rgb24 MAIN p0.image_transform; no augmentation/new Q',
        training=False, new_Q=False, P1_admission=False, independent_confirmation=False,
        history='original sampling causal padding; not observed transitions or new independent roots',
        scope='all4 selected sampling arms at0/5/10; descriptive model/metric/physics diagnostics only')


def validate_frames(frames, tasks):
    require(len(frames) == 12 and len(tasks) == 2, 'complete all12/two roots')
    roots = [frames[0]['root_sha256'], frames[6]['root_sha256']]
    require(len(set(roots)) == 2 and set(roots) == set(tasks), 'same two original task roots')
    require([(f['root_sha256'], f['arm'], f['step']) for f in frames] ==
            [(root, arm, step) for root in roots for arm in ARMS for step in STEPS], 'fixed all12 order')
    for i, f in enumerate(frames):
        require(f['order'] == i and f['task_id'] == core.task_id(tasks[f['root_sha256']]), 'frame/task identity')
        core.array(f['predicted_native192'], (192,))


def analyze(frames, values, tasks):
    validate_frames(frames, tasks)
    values = core.array(values, (12, 192))
    rows = []
    for f, actual in zip(frames, values):
        task = tasks[f['root_sha256']]
        pred = core.array(f['predicted_native192'], (192,))
        rows.append(dict(order=f['order'], root_sha256=f['root_sha256'], task_id=f['task_id'],
            arm=f['arm'], step=f['step'], pixel_sha256=f['pixel_sha256'],
            native_prediction_mse=float(np.mean((pred-actual)**2)),
            predicted_Q_mse=core.model_cost(task, pred), actual_Q_mse=core.model_cost(task, actual),
            physical_diagnostic=f['physical_diagnostic']))
    return dict(schema=NAME + '.analysis', rows=rows, frames=12, training=False, new_Q=False, P1_admission=False,
                independent_confirmation=False,
                scope='same original goal/Q; no selection, fitted metric, causal attribution or theory failure verdict')


class Store:
    """One quota across every file; exclusive writes; every event fsynced."""
    def __init__(self, output):
        self.output = Path(output).absolute()
        require(not self.output.exists(), 'exclusive NEW output')
        require(shutil.disk_usage(self.output.parent).free >= FLOOR+CAP, '2MiB output +200MiB reserve required')
        self.output.mkdir()

    def write(self, name, raw, *, append=False, final=False):
        require(Path(name).name == name and name not in ('', '.', '..'), 'flat output filename')
        used = sum(p.stat().st_size for p in self.output.iterdir() if p.is_file())
        require(used + len(raw) <= CAP - (0 if final else FINAL_RESERVE), 'shared complete output cap')
        require(shutil.disk_usage(self.output).free >= FLOOR + len(raw), '200MiB write floor')
        with (self.output / name).open('ab' if append else 'xb') as stream:
            require(stream.write(raw) == len(raw), 'short write')
            stream.flush()
            os.fsync(stream.fileno())

    def put(self, name, value, *, final=False):
        self.write(name, canonical(value), final=final)


def prepare(output):
    output = Path(output).absolute()
    require(not output.exists(), 'exclusive NEW small prepare')
    p = collect()
    raw = canonical(p)
    require(len(raw) < CAP - FINAL_RESERVE, 'hash manifest below2MiB')
    require(shutil.disk_usage(output.parent).free >= FLOOR + len(raw), 'prepare200MiB floor')
    with output.open('xb') as stream:
        require(stream.write(raw) == len(raw), 'prepare short write')
        stream.flush()
        os.fsync(stream.fileno())
    return io.sha(output)


def verify(prepared, expected):
    source.hash_value(expected)
    io.regular(prepared, CAP)
    require(io.sha(prepared) == expected, 'external small prepare SHA')
    p = physical.read(prepared)
    require(p == collect(), 'full current physical/sampling/PNG/model/source reconstruction')
    return p


def counts(events):
    return {k: dict(attempted=sum(e.get('samples', 0) for e in events if
                    e['kind'] == 'native_image_attempted' and e['resource'] == k),
                    completed=sum(e['counts'][k] for e in events if e['kind'] == 'native_image_completed'))
            for k in ('encoder', 'projector')}


def run(prepared, expected, output):
    p = verify(prepared, expected)
    tasks = task_table(physical.read(PREPARED / 'prepare.json'))
    validate_frames(p['frames'], tasks)
    require([task_reference(x['root_sha256'], tasks[x['root_sha256']]) for x in p['tasks']] == p['tasks'],
            'runtime task hash reconstruction')
    store = Store(output)
    events, values, errors, forbidden, hooks = [], [], [], [], []
    model = None
    before = after = host = None
    active = None
    def emit(event):
        require(active is not None, 'frame context required for every native event')
        e = dict(event, frame_order=active['order'], root_sha256=active['root_sha256'],
                 arm=active['arm'], step=active['step'], pixel_sha256=active['pixel_sha256'])
        if e['kind'] == 'native_image_completed':
            require(e['rgb_sha256'] == active['pixel_sha256'], 'actual native image receipt binding')
            core.array(e['latent'], (192,))
        store.write('image.events.jsonl', canonical(e), append=True)
        events.append(deepcopy(e))
    try:
        import torch
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host = require_cuda_allocation('cuda:0')
        device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False
        model = load_compatible_pretrained(p['policy'], device=device).eval().requires_grad_(False)
        io.frozen(model)
        before = io.tensor_hash(model)
        def deny(module, args):
            forbidden.append(type(module).__name__)
            raise ValueError('predictor/action_encoder forbidden; image-only contract')
        for module in (model.predictor, model.action_encoder):
            hooks.append(module.register_forward_pre_hook(deny))
        encode = rgb24.storage.screen.native_image_encoder(model, device, p0.image_transform(), emit)
        for active in p['frames']:  # no unique(), cache, thinning, batch or retries
            image, _ = io.pixels(active['path'], active['file_sha256'], active['pixel_sha256'])
            emit(dict(kind='frame_started', order=active['order'], task_id=active['task_id']))
            value = encode(image).tolist()
            torch.cuda.synchronize(device)
            emit(dict(kind='frame_completed', order=active['order'], raw_native192=value))
            values.append(value)
        store.put('analysis.json', analyze(p['frames'], values, tasks))
    except BaseException as exc:
        errors.append(io.error('execution', exc))
    finally:
        for hook in reversed(hooks):
            try:
                hook.remove()
            except BaseException as exc:
                errors.append(io.error('hook_cleanup', exc))
        if model is not None:
            try:
                io.frozen(model)
                after = io.tensor_hash(model)
            except BaseException as exc:
                errors.append(io.error('final_model', exc))
        try:
            verify(prepared, expected)
        except BaseException as exc:
            errors.append(io.error('final_source', exc))
        resources = counts(events)
        complete = (not errors and not forbidden and len(values) == 12 and before is not None and before == after and
                    resources == {k: dict(attempted=12, completed=12) for k in resources})
        result = dict(schema=NAME + '.report', status='all12_sampling_checkpoints_encoded' if complete else 'partial_or_failed',
            errors=errors, host=host, resources=resources, frames_completed=len(values),
            prepare_sha256=expected, physical_report_sha256=RUN_SHA, physical_prepare_sha256=PREPARE_SHA,
            model_hash_before=before, model_hash_after=after, model_unchanged=before is not None and before == after,
            forbidden_predictor_action_encoder_attempts=forbidden, new_predict_calls=0, new_AD_calls=0,
            new_physical_steps=0, training=False, new_Q=False, P1_admission=False,
            independent_confirmation=False,
            image_completion_semantics='shared rgb24 helper: both completed only after whole encode and native receipt',
            output_cap_bytes=CAP, disk_floor_bytes=FLOOR)
        for name in ('image.events.jsonl', 'analysis.json'):
            if (store.output / name).exists():
                result[name.replace('.', '_') + '_sha256'] = io.sha(store.output / name)
        try:
            store.put('report.json', result, final=True)
        except BaseException as exc:
            errors.append(io.error('final_report', exc))
            result['status'] = 'partial_or_failed'
    require(result['status'] == 'all12_sampling_checkpoints_encoded', 'partial run; preserve artifacts, no automatic retry')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run'))
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--prepare-sha')
    parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    result = prepare(a.output) if a.command == 'prepare' else (
        verify(a.prepared, a.prepare_sha)['schema'] if a.command == 'verify' else
        run(a.prepared, a.prepare_sha, a.output))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
