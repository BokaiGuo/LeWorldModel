"""MAIN-only fixed-segment B8 GN diagnostic; no new AD/Q/image/physics.

prepare/verify read saved gzip arrays and hashes only. All four frozen scales
are evaluated even if an earlier scale improves. No continuation or retries.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import p1_conditional_direct_screen_20261027_v1 as screen
from experiments.spectral_reachability import p3_joint_prefix_cuda_encoding_20261027_v1 as io

NAME = 'p1_gn_step_diagnostic_cuda_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
B8 = ROOT/'results/p1_observed_direct_cuda_20261027_v2_cuda_main01'
B8_SHA = '954af696d88eecebdcdc7345de702809517491967263f395a8f2819753d889d5'
B8_PREPARED = ROOT/'results/p1_observed_direct_cuda_20261027_v2_prepare01.json'
SCALES = (.5, .25, .125, .0625)
FLOOR = 200*1024**2
CAP = 2*1024**2
PREPARE_CAP = 512*1024
FINAL_RESERVE = 192*1024
require = screen.require
core = screen.core


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'regular bound file')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    path = Path(path)
    require(path.stat().st_size <= 32*1024**2, 'bounded saved input')
    raw = path.read_bytes()
    if path.name.endswith('.gz'):
        raw = gzip.decompress(raw)
    require(len(raw) <= 32*1024**2, 'bounded expanded saved input')
    return json.loads(raw)


def array(v, shape):
    a = np.asarray(v)
    require(a.shape == shape and a.dtype.kind in 'fi' and np.isfinite(a).all(), 'finite real aligned array')
    return a.astype(float)


def segment(x, rejected, scale, cap):
    """Float64 convex interpolation, ONE float32 rounding, NumPy float32 tanh.

    Rounding/bounds failures reject before dispatch; never clip/project/retune.
    """
    require(type(scale) is float and scale in SCALES, 'only four fixed interior scales')
    x, y = array(x, (2, 10)), array(rejected, (2, 10))
    require(all(np.array_equal(a, a.astype(np.float32)) and (np.linalg.norm(a, axis=1) <= cap).all()
                for a in (x, y)), 'original float32 feasible endpoints')
    z = (x+scale*(y-x)).astype(np.float32)
    require((np.linalg.norm(z.astype(float), axis=1) <= cap).all(), 'rounded segment outside packet ball; no repair')
    return dict(scale=scale, logits=z.astype(float).tolist(), commands=np.tanh(z).reshape(10, 2).astype(float).tolist())


def extract(task, component, journal):
    """Original trajectory and raw journal, not a stationarity/selfchecker flag."""
    core.validate_task(task)
    require(component['status'] == 'direct_component_completed_not_admitted' and component['error'] is None and
            component['secondary_errors'] == [] and component['task_id'] == core.task_id(task), 'complete original same-task GN')
    c = component['configuration']
    require(c == dict(method='full_H2_GN', B1=4, B2=8, initial_logits=[np.zeros((2, 10)).tolist()],
                      step_scale=1., damping=.002, extra_line_search_trials=0), 'original frozen B8 GN configuration')
    require(len(component['direct_runs']) == 1, 'one original GN restart')
    run = component['direct_runs'][0]
    require(len(run) == 9 and [s['iteration'] for s in run] == list(range(9)), 'complete ordered B8 trajectory')
    require(journal[0]['kind'] == 'direct_started' and journal[0]['task'] == task and journal[0]['configuration'] == c,
            'original journal task/normalizer/Q binding')
    require([e['step'] for e in journal if e['kind'] == 'trajectory_step'] == run, 'all original saved trajectory steps')
    x, step = run[7]['incumbent'], run[8]
    rejected = step['proposal']; d = step['derivative']
    core.check_prediction(task, x); core.check_prediction(task, rejected); core.check_derivative(task, d, x)
    require(step['accepted'] is False and step['incumbent'] == x and rejected['cost'] >= x['cost'], 'original iter8 rejected proposal')
    require([e['derivative'] for e in journal if e['kind'] == 'derivative_complete'][-1] == d and
            [e['logits'] for e in journal if e['kind'] == 'proposal_logits'][-1] == rejected['logits'], 'raw derivative/proposal path')
    joint = [e for e in journal if e['kind'] == 'joint_logit_jacobian'][-1]
    require(joint['jacobian'] == d['raw_terminal_jacobian'] and joint['terminal'] == d['at_terminal'], 'original derivative arrays')
    gradient = array(d['objective_gradient'], (20,))
    descent = float(gradient @ (np.asarray(rejected['logits'])-x['logits']).reshape(20))
    return dict(incumbent_logits=x['logits'], rejected_logits=rejected['logits'], saved_incumbent_cost=x['cost'],
        saved_rejected_cost=rejected['cost'], saved_gradient_dot_direction=descent,
        derivative_point='direct_runs[0][8].derivative at direct_runs[0][7].incumbent',
        incumbent_location='direct_runs[0][7].incumbent', proposal_location='direct_runs[0][8].proposal',
        candidates=[segment(x['logits'], rejected['logits'], s, task['packet_logit_cap']) for s in SCALES])


def material():
    bindings = {}
    def bound(path, expected=None):
        path = Path(path).resolve(); h = sha(path)
        require(expected is None or h == expected, 'bound input/source drift: '+str(path))
        bindings[str(path)] = h
        return path
    top = read(bound(B8/'report.json', B8_SHA))
    require(top['status'] == 'observed_direct_calibration_completed' and top['errors'] == [] and top['model_unchanged'] is True and
            top['model_hash_before'] is not None and top['model_hash_before'] == top['model_hash_after'], 'actual B8 completed frozen model')
    p = read(bound(B8_PREPARED, top['prepare_sha256']))
    for path, h in {**p['source_sha256'], **p['source_bindings'], **p['model']['bindings']}.items():
        bound(path, h)
    require(str(Path(screen.__file__).resolve()) in p['source_sha256'] and
            str(Path(core.__file__).resolve()) in p['source_sha256'], 'original screen/core source-bound')
    config = read(next(path for path in p['model']['bindings'] if Path(path).name == 'config.json'))
    require(config['encoder'].get('pretrained') is False, 'no checkpoint downloads')
    require(len(top['direct']) == len({(a['root_sha256'], a['method']) for a in top['direct']}) == 4, 'complete original two-root/two-method census')
    roots = []
    for index in range(2):
        task_path = bound(B8/f'task_{index}.json.gz'); encoded = read(task_path); task = encoded['task']
        require(encoded['task_id'] == core.task_id(task) and encoded['actual_observed_public_steps'] == [0, 5, 10] and
                encoded['history_semantics'] == 'observed_not_padding_even_when_numerically_identical', 'original observed task, not padding bridge')
        for filename, key in (('weights.pt', 'model_sha256'), ('config.json', 'model_config_sha256')):
            require(task[key] == next(h for path, h in p['model']['bindings'].items() if Path(path).name == filename), 'same frozen task model')
        require(task['action_mean'] == p['normalization']['mean'] and task['action_std'] == p['normalization']['std'], 'same source normalizer')
        ref = next(a for a in top['direct'] if a['root_sha256'] == encoded['root_sha256'] and a['method'] == 'full_H2_GN')
        component_path = bound(ref['report_path'], ref['report_sha256'])
        journal_path = bound(ref['journal_path'], ref['journal_sha256'])
        require(component_path == B8/f'root_{index}_full_H2_GN/report.json.gz' and
                journal_path == B8/f'root_{index}_full_H2_GN/direct_journal.jsonl.gz', 'own original GN directory')
        raw = gzip.decompress(journal_path.read_bytes()); require(len(raw) < 16*1024**2, 'bounded original journal')
        events = [json.loads(line) for line in raw.splitlines()]
        item = extract(task, read(component_path), events)
        roots.append(dict(root_index=index, root_sha256=encoded['root_sha256'], task_id=encoded['task_id'],
            task_path=str(task_path), component_path=str(component_path), journal_path=str(journal_path), **item))
    require(len({r['root_sha256'] for r in roots}) == 2, 'both original roots, no independent sample increment')
    files = set(io.source_files()+[Path(__file__), Path(screen.__file__), Path(core.__file__),
        ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')])
    for path in files:
        bound(path)
    for path, expected in bindings.items():
        require(sha(path) == expected, 'input/source changed during saved reconstruction: '+path)
    # Hash-only task/Q/source provenance: no big source archive or Q duplication.
    return dict(schema=NAME+'.prepare', roots=roots, scales=list(SCALES), bindings_sha256=bindings,
        policy=p['model']['policy'], model_bindings=p['model']['bindings'], expected_model_hash=top['model_hash_before'],
        execution_order='root0 then root1; incumbent, all four scales, selected nominal/whole/segmented',
        selection='stable minimum(fresh_incumbent, all four fresh scale costs); ties retain earlier order',
        interpolation='float64 x+s*(rejected-x); float32 once; NumPy float32 tanh; no clipping/projection',
        predict_cap=32, action_encoder_cap=32, per_root_cap=16, full_H2_evaluations=16,
        new_AD_cap=0, new_image_cap=0, new_physics_cap=0, training=False, new_Q=False,
        output_cap_bytes=CAP, prepare_cap_bytes=PREPARE_CAP, floor_bytes=FLOOR,
        inherited_direct_resources_once=top['actual_direct_resources'], inherited_image_resources_once=top['actual_image_resources'],
        stationarity_threshold=None, scientific_pass_threshold=None, P1_admission=False, independent_sample_increment=0)


def prepare(output):
    p = material(); raw = canonical(p)
    require(len(raw) < PREPARE_CAP, 'small prepare <512KiB')
    path = Path(output).absolute()
    require(not path.exists() and shutil.disk_usage(path.parent).free >= FLOOR+len(raw), 'exclusive small prepare /200MiB floor')
    with path.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return dict(prepare_sha256=sha(path), prepare_bytes=len(raw), prepared=str(path), predict_cap=32, action_encoder_cap=32)


def verify(path, expected):
    require(sha(path) == expected, 'external prepare SHA')
    p = read(path); require(p == material(), 'full B8/source/scale reconstruction')
    return p


class Storage:
    def __init__(self, folder):
        self.folder = Path(folder).absolute(); self.written = 0
        require(not self.folder.exists() and shutil.disk_usage(self.folder.parent).free >= FLOOR+CAP, 'exclusive 2MiB output plus200MiB reserve')
        self.folder.mkdir()

    def write(self, name, value, final=False, append=False):
        require(Path(name).name == name and name not in ('', '.', '..'), 'flat exclusive artifact')
        raw = canonical(value)
        # A failed write/fsync may already have consumed bytes: no refund.
        used = sum(f.stat().st_size for f in self.folder.iterdir() if f.is_file())
        require(used+len(raw) <= CAP-(0 if final else FINAL_RESERVE), '2MiB total cap incl final reserve')
        require(shutil.disk_usage(self.folder).free >= FLOOR+len(raw), '200MiB disk floor')
        with (self.folder/name).open('ab' if append else 'xb') as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        self.written += len(raw)

    def event(self, e):
        self.write('events.jsonl', e, append=True)


def schedule():
    return [(root, label, mode) for root in range(2) for label, mode in
        [('incumbent', 'whole_replay'), *[('scale_'+str(i), 'whole_replay') for i in range(4)],
         ('selected', 'nominal'), ('selected', 'whole_replay'), ('selected', 'segmented_replay')]]


class Meter:
    """NativeCuda-compatible: one nonrefundable budget across all16 fullH2s."""
    def __init__(self, storage, sync=lambda: None, no_grad=lambda: True):
        self.storage, self.sync, self.no_grad = storage, sync, no_grad
        self.events = []; self.ordinal = -1; self.packet = 0; self.paths = []; self.current = None

    def start(self, root, label, mode):
        require(self.ordinal < 0 or self.packet == 2, 'previous fullH2 incomplete; no retry')
        require(all(e['completed'] for e in self.events) and len(self.events) == 4*(self.ordinal+1), 'no refund/retry after partial entry')
        next_order = self.ordinal+1
        require(next_order < len(schedule()) and schedule()[next_order] == (root, label, mode), 'all scales frozen order; no early winner shortcut')
        self.ordinal = next_order; self.packet = 0; self.paths = []; self.current = (root, label, mode)
        self.storage.event(dict(kind='evaluation_started', order=self.ordinal, root_index=root, label=label, mode=mode))

    def begin(self, resource):
        require(self.current is not None and self.packet < 2 and self.no_grad(), 'active no-grad fullH2 only')
        offset = len(self.events)-4*self.ordinal
        require(resource == ('action_encoder', 'predict')[offset % 2] and offset//2 == self.packet and
                all(e['completed'] for e in self.events), 'exact packet/module order')
        require(sum(e['resource'] == resource for e in self.events) < 32 and
                sum(e['resource'] == resource and e['root_index'] == self.current[0] for e in self.events) < 16,
                'global32/perroot16 predispatch cap')
        e = dict(index=len(self.events), resource=resource, samples=1, completed=False, root_index=self.current[0],
                 label=self.current[1], mode=self.current[2], packet=self.packet, evaluation_order=self.ordinal)
        self.events.append(e); self.storage.event(dict(kind='module_attempt', entry=deepcopy(e))); return e

    def end(self, token):
        require(token is self.events[-1] and token['completed'] is False, 'unique matching return')
        self.sync(); token['completed'] = True
        self.storage.event(dict(kind='module_return', entry=deepcopy(token)))

    def emit(self, kind, **values):
        require(kind == 'native_packet' and values['packet'] == self.packet and values['mode'] == self.current[2] and
                len(self.events) == 4*self.ordinal+2*(self.packet+1) and all(e['completed'] for e in self.events), 'durable complete packet chronology')
        self.storage.event(dict(kind=kind, order=self.ordinal, root_index=self.current[0], label=self.current[1], **deepcopy(values)))
        self.paths.append(deepcopy(values['context'])); self.packet += 1

    def counts(self):
        return {k: dict(attempted=sum(e['resource'] == k for e in self.events),
                        completed=sum(e['resource'] == k and e['completed'] for e in self.events)) for k in ('predict', 'action_encoder')}


def check_path(task, commands, path):
    require(len(path) == 2, 'fullH2 two returned contexts')
    previous = task['root_context']; tokens = core.normalized_tokens(task, commands)
    for k, c in enumerate(path):
        core.context(c)
        require(c['latent_history'][:2] == previous['latent_history'][1:] and np.array_equal(c['action_history'],
            np.concatenate((np.asarray(previous['action_history']), tokens[k:k+1]))[-2:]), 'literal normalized history/order')
        previous = c


def evaluate_once(task, logits, backend, meter, store, root, label, mode):
    logits = array(logits, (2, 10)); raw = np.tanh(logits.astype(np.float32)).reshape(10, 2)
    meter.start(root, label, mode)
    request = dict(root_index=root, task_id=core.task_id(task), label=label, mode=mode, logits=logits.tolist(), commands=raw.astype(float).tolist())
    store.event(dict(kind='evaluation_request', order=meter.ordinal, **request))
    contexts = backend.rollout(task, raw, meter, mode)
    require(contexts == meter.paths, 'actual contexts durably journaled')
    check_path(task, raw, contexts)
    value = dict(**request, contexts=contexts, cost=core.model_cost(task, contexts[-1]['latent_history'][-1]))
    store.write(f'eval_{meter.ordinal:02d}.json', value)
    return value


def select(evaluations):
    require(len(evaluations) == 5 and [e['label'] for e in evaluations] == ['incumbent', *['scale_'+str(i) for i in range(4)]],
            'incumbent AND all four line trials before selecting')
    require(all(type(e['cost']) in (int, float) and np.isfinite(e['cost']) and e['cost'] >= 0 for e in evaluations), 'finite nonnegative original-Q costs')
    return min(range(5), key=lambda i: (evaluations[i]['cost'], i))


def diagnostic_root(r, task, incumbent, backend, meter, store):
    root = r['root_index']; evaluations = []
    fresh = evaluate_once(task, r['incumbent_logits'], backend, meter, store, root, 'incumbent', 'whole_replay')
    evaluations.append(fresh)
    require(fresh['contexts'] == incumbent['contexts'] and fresh['cost'] == incumbent['cost'], 'fresh original incumbent exact reproduction')
    for i, c in enumerate(r['candidates']):
        evaluations.append(evaluate_once(task, c['logits'], backend, meter, store, root, 'scale_'+str(i), 'whole_replay'))
    index = select(evaluations); winner = evaluations[index]
    selection = dict(root_sha256=r['root_sha256'], root_index=root, evaluations=evaluations, winner_index=index,
                     selected_scale=None if index == 0 else SCALES[index-1], strict_improvement=winner['cost'] < fresh['cost'])
    store.write(f'root_{root}.selection.json', selection)
    paths = []
    for mode in ('nominal', 'whole_replay', 'segmented_replay'):
        paths.append(evaluate_once(task, winner['logits'], backend, meter, store, root, 'selected', mode)['contexts'])
    prediction = dict(task_id=r['task_id'], logits=winner['logits'], commands=winner['commands'], contexts=paths[0],
                      whole_replay=paths[1][-1], segmented_replay=paths[2], cost=core.model_cost(task, paths[0][-1]['latent_history'][-1]))
    require(paths[0] == paths[1] == paths[2] == winner['contexts'], 'selected independent nominal/whole/segmented exact')
    core.check_prediction(task, prediction)
    result = dict(root_sha256=r['root_sha256'], root_index=root, selected_scale=selection['selected_scale'],
        incumbent_cost=fresh['cost'], scale_costs=[dict(scale=s, cost=e['cost'], improvement=fresh['cost']-e['cost'])
            for s, e in zip(SCALES, evaluations[1:], strict=True)], selected_cost=prediction['cost'],
        strict_improvement=selection['strict_improvement'], verified_prediction=prediction, full32_required=True)
    store.write(f'root_{root}.result.json', result); return result


def error(stage, exc):
    return dict(stage=stage, type=type(exc).__name__, message=str(exc)[:1500], notes=list(getattr(exc, '__notes__', [])))


def run(prepared, expected, output):
    p = verify(prepared, expected); store = Storage(output); meter = Meter(store)
    store.write('execution_binding.json', dict(prepared=str(Path(prepared).resolve()), prepare_sha256=expected))
    model = None; before = after = host = None; errors = []; rows = []; source_ok = False
    try:
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        host = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False
        model = load_compatible_pretrained(p['policy'], device=device).eval().requires_grad_(False)
        io.frozen(model); before = io.tensor_hash(model)
        require(before == p['expected_model_hash'], 'original B8 frozen tensor state')
        meter.sync = lambda: torch.cuda.synchronize(device); meter.no_grad = lambda: not torch.is_grad_enabled()
        with sdpa_kernel(SDPBackend.MATH):
            for r in p['roots']:
                task = read(r['task_path'])['task']; incumbent = read(r['component_path'])['direct_runs'][0][7]['incumbent']
                backend = screen.NativeCuda(model, device, task)
                with backend.guard(meter):
                    rows.append(diagnostic_root(r, task, incumbent, backend, meter, store))
    except BaseException as exc:
        errors.append(error('execution', exc))
        try: store.event(dict(kind='execution_error', error=errors[0]))
        except BaseException as secondary: errors.append(error('error_journal', secondary))
    finally:
        if model is not None:
            try: io.frozen(model); after = io.tensor_hash(model)
            except BaseException as exc: errors.append(error('final_model', exc))
        try: verify(prepared, expected); source_ok = True
        except BaseException as exc: errors.append(error('final_source', exc))
        counts = meter.counts()
        complete = (not errors and len(rows) == 2 and meter.ordinal == 15 and meter.packet == 2 and
                    counts == {k: dict(attempted=32, completed=32) for k in counts} and before is not None and before == after and source_ok)
        files = {}
        for f in store.folder.iterdir():
            if f.is_file():
                try: files[f.name] = sha(f)
                except BaseException as exc: errors.append(error('final_artifact_hash', exc)); complete = False
        report = dict(schema=NAME+'.report', status='cuda_fixed_segment_diagnostic_completed' if complete else 'partial_or_failed',
            prepare_sha256=expected, rows=rows, errors=errors, first_error=errors[0] if errors else None, resources=counts,
            module_entries=meter.events, model_hash_before=before, model_hash_after=after,
            model_unchanged=before == after if before is not None else None, source_verified=source_ok, host=host,
            artifacts_sha256=files, scales=list(SCALES), new_AD_calls=0, new_image_calls=0, new_physical_steps=0,
            training=False, new_Q=False, scientific_pass_threshold=None, P1_admission=False,
            scope='fixed four-point step diagnostic on same two B8 calibration roots; not stationarity/theory/physical efficacy')
        try: store.write('report.json', report, final=True)
        except BaseException as exc:
            errors.append(error('final_report', exc)); complete = False
            print(canonical(dict(first_error=errors[0], errors=errors)).decode(), flush=True)
    require(complete, 'partial diagnostic; preserve artifacts; no automatic retry: '+str(errors))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run'))
    parser.add_argument('--prepared', type=Path); parser.add_argument('--prepare-sha'); parser.add_argument('--output', type=Path)
    a = parser.parse_args()
    result = prepare(a.output) if a.command == 'prepare' else (
        dict(status='verified_no_execution', roots=len(verify(a.prepared, a.prepare_sha)['roots'])) if a.command == 'verify'
        else run(a.prepared, a.prepare_sha, a.output))
    print(canonical({k:v for k,v in result.items() if k not in ('module_entries', 'rows')}).decode().strip())
