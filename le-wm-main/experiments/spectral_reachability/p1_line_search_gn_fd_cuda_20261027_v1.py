"""MAIN-only full-H2 central FD at all eight saved fresh line-search GN J points.

Prepare/verify/tests are saved arrays only. Explicit run is CUDA-only, no new
AD, images, physics, optimization, projection, selection or threshold tuning.
"""
from copy import deepcopy
from contextlib import ExitStack
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import zlib

import numpy as np

from experiments.spectral_reachability import p1_line_search_gn_saved_audit_20261027_v1 as audit
from experiments.spectral_reachability import p1_conditional_direct_screen_20261027_v1 as screen
from experiments.spectral_reachability import p3_joint_prefix_cuda_encoding_20261027_v1 as io

NAME = 'p1_line_search_gn_fd_cuda_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
EPS = (1e-3, 3e-3, 1e-2)
SIGNS = (-1, 1)
CAP, FLOOR, FINAL = 2*1024**2, 200*1024**2, 128*1024
PREPARE_CAP = 256*1024
require, sha, read, canonical, array = audit.require, audit.sha, audit.read, audit.canonical, audit.array


def directions():
    a, b = np.zeros(20), np.zeros(20)
    a[0], b[10] = 1., 1.
    # Explicit signs, not RNG, fitted singular vectors, gradients, or outcomes.
    c = np.asarray([1, -1, 1, 1, -1, -1, 1, -1, -1, 1,
                    -1, 1, 1, -1, 1, -1, -1, 1, 1, -1], dtype=float)/np.sqrt(20.)
    return [dict(id=k, vector=v.tolist()) for k, v in
        (('packet0_e0', a), ('packet1_e0', b), ('fixed20_sign_unit', c))]


def budget():
    return dict(points=8, directions=3, epsilons=3, central_comparisons=72,
        signed_H2_evaluations=144, nominal_H2_evaluations=8, total_H2_evaluations=152,
        predict=304, action_encoder=304, AD_calls=0, new_jacobians=0,
        new_vjp_directions=0, encoder=0, projector=0, physics_steps=0,
        FD_predict=288, nominal_predict=16,
        inherited_line_search_predict=156, inherited_line_search_action_encoder=156,
        inherited_AD_forward_predicts_already_inside156=16,
        inherited_jacobians=8, inherited_mathematical_vjp_directions=1536,
        line_search_plus_FD_predict=460, line_search_plus_FD_action_encoder=460,
        upstream_B8_and_image_acquisition='separate inherited source costs, not zero; never multiplied by FD points')


def collect():
    b = audit.load_bundle()
    verified = audit.audit(b)  # Raw eight-J chronology/rows; not FD correctness.
    config = read(next(path for path in b['bindings'] if Path(path).name == 'config.json'))
    require(config['encoder'].get('pretrained') is False, 'local model config must forbid pretrained download')
    points, matrices, tasks = [], {}, {}
    ds = directions()
    for root, task in zip(b['prepare']['roots'], b['tasks'], strict=True):
        index = root['root_index']; tasks[index] = task
        events = [e for e in b['events'] if e['root_index'] == index]
        derivatives = [e for e in events if e['kind'] == 'derivative_complete']
        require(len(derivatives) == 4, 'all four fresh pre-update J points')
        for iteration, event in enumerate(derivatives, 1):
            d = event['derivative']; j = array(d['raw_terminal_jacobian'], (192, 20))
            require(event['evaluation_id'] == f'iter_{iteration}_AD' and d['task_id'] == root['task_id'], 'fresh J identity')
            path = [e['context'] for e in events if e['kind'] == 'jacobian_forward_packet' and e['evaluation_id'] == event['evaluation_id']]
            require(len(path) == 2 and path[-1]['latent_history'][-1] == d['at_terminal'], 'actual fullH2 AD nominal path')
            point = dict(point_order=len(points), root_index=index, root_sha256=root['root_sha256'], task_id=root['task_id'],
                task_path=root['task_path'], iteration=iteration, evaluation_id=event['evaluation_id'],
                at_logits=d['at_logits'], at_terminal=d['at_terminal'],
                saved_context_sha256=[screen.core.digest(c) for c in path],
                logit_J_sha256=screen.core.digest(j.tolist()),
                saved_Jv={v['id']:(j@np.asarray(v['vector'])).tolist() for v in ds})
            matrices[len(points)] = j; points.append(point)
    bindings = deepcopy(b['bindings'])
    for f in (Path(__file__).resolve(), Path(audit.__file__).resolve(), Path(screen.__file__).resolve(),
              Path(screen.core.__file__).resolve(), Path(io.__file__).resolve(),
              ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')):
        bindings[str(f)] = sha(f)
    require(len(points) == 8 and [p['point_order'] for p in points] == list(range(8)), 'all eight points')
    p = dict(schema=NAME+'.prepare', input_report_sha256=audit.INPUT_SHA, input_prepare_sha256=audit.PREPARE_SHA,
        points=points, directions=ds, epsilons=list(EPS), signs=list(SIGNS), budget=budget(),
        order='root,pre-update-iteration; nominal then direction,epsilon,minus/plus; both packets from ORIGINAL task root',
        perturbation='float64 x +/- requested_eps*unit_v, cast logits float32 once, NumPy float32 tanh once; NO projection',
        feasibility='FD probes unconstrained network extension near packet-ball boundary; not feasible optimization proposals',
        reference='saved logit-J, NOT raw-control rows; Jv and J@effective_float32_secant_direction both reported',
        runtime_numeric_policy='same explicit original SDP MATH and matmul TF32=false; cuDNN TF32 inherited/recorded (original flag unknown), exact nominal anchors mandatory',
        comparison_metrics=['absolute_l2', 'absolute_linf', 'absolute_rmse', 'relative_l2', 'symmetric_relative_l2',
                            'cosine', 'midpoint_drift_l2', 'effective_direction_error_l2', 'adjacent_epsilon_fd_l2'],
        zero_denominators='null, never fitted floor', scientific_pass_threshold=None, derivative_verdict='not_assigned',
        policy=b['prepare']['policy'], expected_model_hash=b['prepare']['expected_model_hash'], bindings_sha256=bindings,
        inherited_line_search_resources=verified['resources'],
        inherited_B8_resources_once=b['prepare']['inherited_direct_resources_once'],
        inherited_image_resources_once=b['prepare']['inherited_image_resources_once'],
        saved_validation_work='each collect includes 8 NumPy20x20 ridge solves + raw chronology checks; no AD or model calls',
        output_cap_bytes=CAP, floor_bytes=FLOOR, final_reserve_bytes=FINAL,
        new_Q=False, training=False, new_physics=0, P1_admission=False, independent_sample_increment=0)
    for path, expected in bindings.items():
        require(sha(path) == expected, 'source changed during FD preparation: '+path)
    return p, tasks, matrices


def material():
    return collect()[0]


def prepare(output):
    p = material(); raw = canonical(p); output = Path(output).absolute()
    require(not output.exists() and len(raw) < PREPARE_CAP, 'exclusive small prepare <256KiB')
    require(shutil.disk_usage(output.parent).free >= FLOOR+len(raw)+4096, '200MiB prepare floor')
    with output.open('xb') as f:
        require(f.write(raw) == len(raw), 'short prepare write'); f.flush(); os.fsync(f.fileno())
    io.fsync_directory(output.parent)
    return dict(prepared=str(output), prepare_sha256=sha(output), bytes=len(raw), budget=budget(), new_neural_calls=0)


def verify(path, expected):
    screen.core.hash_value(expected)
    require(Path(path).stat().st_size < PREPARE_CAP and sha(path) == expected, 'external prepare SHA')
    p = read(path); current, tasks, matrices = collect()
    require(p == current, 'complete saved tasks/eight J/source/fixed protocol reconstruction')
    return p, tasks, matrices


def perturb(point, vector, epsilon, sign):
    require(epsilon in EPS and sign in SIGNS, 'frozen epsilon and side')
    x, v = array(point['at_logits'], (2, 10)), array(vector, (20,))
    require(np.array_equal(x, x.astype(np.float32)) and abs(np.linalg.norm(v)-1) < 1e-14, 'float32 base / fixed unit direction')
    value = (x+sign*epsilon*v.reshape(2, 10)).astype(np.float32)
    require(np.isfinite(value).all(), 'finite logits')
    return dict(logits=value.astype(float).tolist(), commands=np.tanh(value).reshape(10, 2).astype(float).tolist())


def requests(p):
    rows = []
    for point in p['points']:
        common = {k:point[k] for k in ('point_order', 'root_index', 'root_sha256', 'task_id', 'iteration')}
        x = np.asarray(point['at_logits'], np.float32)
        rows.append(dict(common, order=len(rows), mode='nominal', direction=None, epsilon=None, sign=0,
            logits=x.astype(float).tolist(), commands=np.tanh(x).reshape(10, 2).astype(float).tolist()))
        for d in p['directions']:
            for eps in EPS:
                for sign in SIGNS:
                    rows.append(dict(common, order=len(rows), mode='central_side', direction=d['id'], epsilon=eps, sign=sign,
                        **perturb(point, d['vector'], eps, sign)))
    require(len(rows) == 152, '152 precommitted evaluations')
    return rows


def path_check(task, commands, contexts):
    require(len(contexts) == 2, 'fullH2 path')
    previous = task['root_context']
    raw = array(commands, (10, 2))
    require((np.abs(raw) <= 1).all(), 'bounded tanh raw controls')
    for k, c in enumerate(contexts):
        audit.history(task, previous, c, raw[5*k:5*k+5]); previous = c


def errors(fd, ref):
    fd, ref = array(fd, (192,)), array(ref, (192,))
    delta = fd-ref; n, f = float(np.linalg.norm(ref)), float(np.linalg.norm(fd))
    return dict(absolute_l2=float(np.linalg.norm(delta)), absolute_linf=float(np.max(np.abs(delta))),
        absolute_rmse=float(np.sqrt(np.mean(delta*delta))), reference_l2=n, fd_l2=f,
        relative_l2=float(np.linalg.norm(delta)/n) if n else None,
        symmetric_relative_l2=float(np.linalg.norm(delta)/max(n, f)) if max(n, f) else None,
        cosine=float(fd@ref/(n*f)) if n and f else None)


def compare(point, j, d, eps, minus, plus):
    m, p = array(minus['terminal'], (192,)), array(plus['terminal'], (192,))
    v = array(d['vector'], (20,)); j = array(j, (192, 20))
    fd = (p-m)/(2*eps)
    effective = (np.asarray(plus['logits']).reshape(20)-np.asarray(minus['logits']).reshape(20))/(2*eps)
    ref = j@v; effective_ref = j@effective
    require(np.array_equal(ref, point['saved_Jv'][d['id']]), 'saved Jv binding')
    return dict(point_order=point['point_order'], root_index=point['root_index'], iteration=point['iteration'],
        direction=d['id'], epsilon=eps, fd=fd.tolist(), saved_Jv=ref.tolist(),
        effective_float32_Jv=effective_ref.tolist(), effective_direction=effective.tolist(),
        effective_direction_error_l2=float(np.linalg.norm(effective-v)),
        midpoint_drift_l2=float(np.linalg.norm((p+m)/2-np.asarray(point['at_terminal']))),
        requested_direction_metrics=errors(fd, ref), effective_direction_metrics=errors(fd, effective_ref))


def analyze(p, rows, matrices):
    wanted = requests(p)
    require(len(rows) == 152 and all({k:e[k] for k in w} == w for e,w in zip(rows, wanted, strict=True)), 'all152 ordered requests, no filtering')
    result, stability = [], []
    offset = 0
    for point in p['points']:
        nominal = rows[offset]
        require(nominal['terminal'] == point['at_terminal'] and nominal['context_sha256'] == point['saved_context_sha256'], 'fresh nominal EXACT saved AD path')
        offset += 1
        for d in p['directions']:
            group = []
            for eps in EPS:
                group.append(compare(point, matrices[point['point_order']], d, eps, rows[offset], rows[offset+1])); offset += 2
            result.extend(group)
            stability.append(dict(point_order=point['point_order'], direction=d['id'],
                adjacent_epsilon_fd_l2=[float(np.linalg.norm(np.asarray(a['fd'])-b['fd'])) for a,b in zip(group, group[1:])]))
    require(len(result) == 72 and offset == 152, 'all72 comparisons')
    return dict(schema=NAME+'.analysis', comparisons=result, epsilon_stability=stability, nominal_anchors=8,
        scientific_pass_threshold=None, derivative_verdict='not_assigned', P1_admission=False)


def check_space(used, size, free, final=False):
    require(used+size <= CAP-(0 if final else FINAL), 'whole-output 2MiB cap including partial bytes')
    require(free >= FLOOR+size, '200MiB floor')


class Store:
    """One compressed fsynced event stream; partial gzip never called complete."""
    def __init__(self, output):
        self.output = Path(output).absolute(); self.broken = self.closed = False
        require(not self.output.exists() and shutil.disk_usage(self.output.parent).free >= FLOOR+CAP, 'NEW output +2MiB +200MiB reserve')
        self.output.mkdir(); io.fsync_directory(self.output.parent)
        self.comp = zlib.compressobj(6, zlib.DEFLATED, 31)
        self.write('events.jsonl.gz', b'')

    def write(self, name, raw, append=False, final=False):
        require(Path(name).name == name and name not in ('', '.', '..'), 'flat output')
        require(not (self.output/name).is_symlink(), 'no output symlink')
        used = sum(f.stat().st_size for f in self.output.iterdir() if f.is_file())
        check_space(used, len(raw), shutil.disk_usage(self.output).free, final)
        with (self.output/name).open('ab' if append else 'xb') as f:
            require(f.write(raw) == len(raw), 'short output write'); f.flush(); os.fsync(f.fileno())
        io.fsync_directory(self.output)

    def put(self, name, value, final=False):
        raw = canonical(value)
        self.write(name, gzip.compress(raw, compresslevel=6, mtime=0) if name.endswith('.gz') else raw, final=final)

    def event(self, value):
        require(not self.broken and not self.closed, 'no retry after journal failure')
        try:
            self.write('events.jsonl.gz', self.comp.compress(canonical(value))+self.comp.flush(zlib.Z_SYNC_FLUSH), append=True)
        except BaseException:
            self.broken = True
            raise

    def close(self):
        require(not self.broken and not self.closed, 'cannot finalize broken journal')
        self.closed = True
        self.write('events.jsonl.gz', self.comp.flush(zlib.Z_FINISH), append=True, final=True)


class Meter:
    def __init__(self, emit, sync=lambda:None, no_grad=lambda:True):
        self.emit_event, self.sync, self.no_grad = emit, sync, no_grad
        self.events, self.paths = [], []
        self.request = None; self.packet = 0

    def start(self, request):
        require((self.request is None or self.packet == 2) and request['order'] < 152 and
            len(self.events) == request['order']*4 and all(e['completed'] for e in self.events), 'sequential once-only complete evaluations')
        self.request = request; self.packet = 0; self.paths = []
        self.emit_event(dict(kind='evaluation_attempt', request=deepcopy(request)))

    def begin(self, resource):
        require(self.request is not None and self.packet < 2 and self.no_grad(), 'no-grad fullH2 only')
        offset = len(self.events)-self.request['order']*4
        require(resource == ('action_encoder', 'predict')[offset%2] and offset//2 == self.packet and
            all(e['completed'] for e in self.events), 'module/packet order, no failed retry')
        require(sum(e['resource'] == resource for e in self.events) < 304, 'hard predispatch 304 cap')
        e = dict(index=len(self.events), resource=resource, samples=1, completed=False,
            evaluation_order=self.request['order'], packet=self.packet)
        self.events.append(e); self.emit_event(dict(kind='module_attempt', entry=deepcopy(e)))
        return e

    def end(self, e):
        require(e is self.events[-1] and not e['completed'], 'unique matching module return')
        self.sync(); e['completed'] = True
        self.emit_event(dict(kind='module_completed', entry=deepcopy(e)))

    def emit(self, kind, **value):
        require(kind == 'native_packet' and value['mode'] == 'whole_replay' and value['packet'] == self.packet and
            len(self.events) == self.request['order']*4+2*(self.packet+1) and all(e['completed'] for e in self.events), 'exact native packet chronology')
        c = deepcopy(value['context']); self.paths.append(c)
        self.emit_event(dict(kind='native_packet', evaluation_order=self.request['order'], packet=self.packet,
            context_sha256=screen.core.digest(c), terminal=c['latent_history'][-1]))
        self.packet += 1

    def counts(self):
        return {k:dict(attempted=sum(e['resource'] == k for e in self.events),
            completed=sum(e['resource'] == k and e['completed'] for e in self.events)) for k in ('predict', 'action_encoder')}


def evaluate(p, tasks, backend_for_root, meter, sink, rows=None):
    rows = [] if rows is None else rows
    current = None
    with ExitStack() as active:
        for request in requests(p):
            index = request['root_index']
            backend = backend_for_root(index)
            if index != current:
                active.close()
                active.enter_context(backend.guard(meter))
                current = index
            task = tasks[index]
            meter.start(request)
            contexts = backend.rollout(task, np.asarray(request['commands'], np.float32), meter, 'whole_replay')
            require(contexts == meter.paths, 'actual complete packet evidence')
            path_check(task, request['commands'], contexts)
            row = dict(request, terminal=contexts[-1]['latent_history'][-1], context_sha256=[screen.core.digest(c) for c in contexts])
            if request['mode'] == 'nominal':
                point = p['points'][request['point_order']]
                require(row['terminal'] == point['at_terminal'] and row['context_sha256'] == point['saved_context_sha256'], 'fresh nominal AD-point mismatch; stop without retry')
            sink(dict(kind='evaluation_complete', evaluation=row)); rows.append(row)
    return rows


def failure(stage, exc):
    return dict(stage=stage, type=type(exc).__name__, message=str(exc)[:1200], notes=[str(x)[:1200] for x in getattr(exc, '__notes__', [])])


def run(prepared, expected, output):
    p, tasks, matrices = verify(prepared, expected)
    store = Store(output); store.put('binding.json', dict(prepared=str(Path(prepared).resolve()), prepare_sha256=expected))
    errors_list, forbidden = [], []; hooks = io.Hooks()
    model = before = after = host = analysis_sha = numeric_settings = None
    source_ok = False; rows = []; validation = dict(attempted=1, completed=1)
    meter = Meter(store.event)
    try:
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False
        numeric_settings = dict(sdpa='MATH', matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, original_cudnn_flag=None)
        model = load_compatible_pretrained(p['policy'], device=device).eval().requires_grad_(False)
        io.frozen(model); before = io.tensor_hash(model)
        require(before == p['expected_model_hash'], 'same frozen line-search model tensor state')
        meter.sync = lambda:torch.cuda.synchronize(device)
        meter.no_grad = lambda:not torch.is_grad_enabled()
        targets = [(torch.autograd, k) for k in ('grad', 'backward', '_engine_run_backward')]
        targets += [(torch.autograd.functional, k) for k in ('jacobian', 'hessian', 'jvp', 'vjp')]
        targets += [(torch.Tensor, 'backward')]
        if hasattr(torch, 'func'):
            targets += [(torch.func, k) for k in ('jacrev', 'jacfwd', 'hessian', 'jvp', 'vjp')]
        for obj, key in targets:
            if not hasattr(obj, key): continue
            def denied(*args, _key=key, **kwargs):
                forbidden.append(_key); store.event(dict(kind='forbidden_AD', entry=_key))
                raise ValueError('FD prohibits new AD: '+_key)
            hooks.install(obj, key, denied)
        backends = {i:screen.NativeCuda(model, device, task) for i,task in tasks.items()}
        with sdpa_kernel(SDPBackend.MATH):
            evaluate(p, tasks, lambda i:backends[i], meter, store.event, rows)
        result = analyze(p, rows, matrices); store.put('analysis.json.gz', result)
        analysis_sha = sha(store.output/'analysis.json.gz')
    except BaseException as exc:
        errors_list.append(failure('execution', exc))
        try: store.event(dict(kind='execution_error', error=errors_list[0]))
        except BaseException as secondary: errors_list.append(failure('error_journal', secondary))
    finally:
        hooks.restore(errors_list)
        if model is not None:
            try: io.frozen(model); after = io.tensor_hash(model)
            except BaseException as exc: errors_list.append(failure('final_model', exc))
        try:
            validation['attempted'] += 1; verify(prepared, expected)
            validation['completed'] += 1; source_ok = True
        except BaseException as exc: errors_list.append(failure('final_source', exc))
        try: store.close()
        except BaseException as exc: errors_list.append(failure('journal_finalize', exc))
        counts = meter.counts()
        complete = (not errors_list and not forbidden and len(rows) == 152 and source_ok and before is not None and before == after and
            analysis_sha is not None and counts == {k:dict(attempted=304, completed=304) for k in ('predict', 'action_encoder')})
        hashes = {}
        for path in store.output.iterdir():
            if path.is_file():
                try: hashes[path.name] = sha(path)
                except BaseException as exc: errors_list.append(failure('artifact_hash', exc)); complete = False
        report = dict(schema=NAME+'.report', status='cuda_fd_completed' if complete else 'partial_or_failed',
            prepare_sha256=expected, input_line_search_report_sha256=audit.INPUT_SHA,
            errors=errors_list, first_error=errors_list[0] if errors_list else None, forbidden_AD=forbidden,
            host=host, numeric_settings=numeric_settings, resources=counts, evaluations_completed=len(rows),
            completed_count_on_exception='module and evaluation journal is authority; returned rows may be a lower bound',
            budget=budget(), inherited_line_search_resources=p['inherited_line_search_resources'],
            inherited_B8_resources_once=p['inherited_B8_resources_once'], inherited_image_resources_once=p['inherited_image_resources_once'],
            saved_validation_calls=validation, completed_saved_validation_numpy20x20_solves=8*validation['completed'],
            model_hash_before=before, model_hash_after=after, model_unchanged=before is not None and before == after,
            source_verified=source_ok, artifact_sha256=hashes, analysis_sha256=analysis_sha,
            output_cap_bytes=CAP, floor_bytes=FLOOR, new_AD_calls=0 if complete else None,
            new_images=0, new_physics=0, new_Q=False, training=False, independent_sample_increment=0,
            scientific_pass_threshold=None, derivative_verdict='not_assigned', P1_admission=False)
        try: store.put('report.json', report, final=True)
        except BaseException as exc:
            errors_list.append(failure('final_report', exc)); complete = False
            print(canonical(dict(first_error=errors_list[0], errors=errors_list)).decode(), flush=True)
    require(complete and not errors_list, 'partial FD preserved; no automatic retry: '+str(errors_list))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run'))
    parser.add_argument('--prepared', type=Path); parser.add_argument('--prepare-sha'); parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = prepare(args.output) if args.command == 'prepare' else (
        dict(status='verified_saved_arrays_only', points=len(verify(args.prepared, args.prepare_sha)[0]['points']), budget=budget())
        if args.command == 'verify' else run(args.prepared, args.prepare_sha, args.output))
    print(canonical(result).decode().strip())
