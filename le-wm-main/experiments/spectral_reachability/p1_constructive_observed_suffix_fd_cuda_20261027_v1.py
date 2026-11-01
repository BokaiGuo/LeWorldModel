"""MAIN-only observed-prefix H1 central FD. No optimization or new AD.

Only D2P branches1/3 of the externally pinned four-branch receipt are points.
Prepare/verify use saved arrays; run alone loads the frozen CUDA model.
"""
from copy import deepcopy
from contextlib import ExitStack
import argparse
import ast
import io as bytesio
import os
from pathlib import Path
import shutil
import numpy as np

from experiments.spectral_reachability import p1_line_search_gn_fd_cuda_20261027_v1 as fd
from experiments.spectral_reachability import p1_constructive_observed_suffix_cuda_20261027_v1 as source

NAME = 'p1_constructive_observed_suffix_fd_cuda_20261027_v1'
ROOT = source.ROOT
INPUT = ROOT/'results/p1_constructive_observed_suffix_cuda_20261027_v1_cuda_main01/report.json'
INPUT_SHA = 'ac8a65900b94639f94b3797e90f80e73fa0500e13a055629750dc4dc418aaf79'
PARENT_PREPARE_SHA = '5b9f970a72e3cc3ef45b31d5d35416af6cdc779189f55fd3fd8ad5670a9a3e88'
BRANCHES, EPS, SIGNS = (1, 3), (1e-3, 3e-3, 1e-2), (-1, 1)
CAP, FLOOR, FINAL, PREPARE_CAP = fd.CAP, fd.FLOOR, fd.FINAL, 256*1024
require, sha, canonical = source.require, source.sha, source.canonical
base, screen, io = source.base, fd.screen, fd.io
Store, failure, errors = fd.Store, fd.failure, fd.errors


def directions():
    a, b = np.zeros(10), np.zeros(10)
    a[0], b[1] = 1., 1.
    c = np.asarray([1, -1, 1, 1, -1, -1, 1, -1, -1, 1], float)/np.sqrt(10.)
    return [dict(id=k, vector=v.tolist()) for k, v in
            (('packetflat0', a), ('packetflat1', b), ('fixed10_sign_unit', c))]


def budget():
    return dict(points=2, directions=3, epsilons=3, central_comparisons=18,
        nominal_H1_evaluations=2, signed_H1_evaluations=36, total_H1_evaluations=38,
        predict=38, action_encoder=38, nominal_predict=2, FD_predict=36,
        AD_calls=0, new_jacobians=0, new_vjp_directions=0, encoder=0, projector=0,
        physics_steps=0, SVD=0, solve=0, selection=0,
        inherited_feedback_predict=20, inherited_feedback_action_encoder=20,
        inherited_feedback_jacobians=4, inherited_feedback_vjp_directions=768,
        inherited_feedback_AD_forward_inside20=4,
        upstream_cost='separate inherited planner/encoding/physical receipts; not zero or recomputed',
        equal_total_compute_claim=False)


def array(x, shape):
    return fd.array(x, shape)


def advance(task, before, commands, terminal):
    screen.core.context(before)
    raw = array(commands, (5, 2))
    require(np.array_equal(raw, raw.astype(np.float32)), 'literal float32 commands')
    token = screen.core.normalized_tokens(task, raw)[0].astype(float).tolist()
    c = dict(before, latent_history=deepcopy(before['latent_history'][1:])+[array(terminal, (192,)).tolist()],
             action_history=[deepcopy(before['action_history'][-1]), token])
    fd.audit.history(task, before, c, raw)
    return c


def extract_point(task, record, result, plan, events, point_order, prefix):
    """Bind raw original implementation path, not downstream checker verdicts."""
    def one(kind):
        values = [e for e in events if e['kind'] == kind]
        require(len(values) == 1, 'unique saved '+kind)
        return values[0]
    frozen, zero, fresh = (one(k) for k in ('suffix_problem_frozen', 'zero_nominal_saved', 'fresh_J_saved'))
    kinds = [e['kind'] for e in events]
    require(kinds.index('suffix_problem_frozen') < kinds.index('zero_nominal_saved') <
            kinds.index('fresh_J_saved') < kinds.index('ridge_frozen_before_prediction'), 'causal zero/J before solve')
    screen.core.validate_task(task)
    require(record['branch'] in BRANCHES and record['arm'] == base.ARMS[1] and
        record['source_slot'] == record['root_slot'] == point_order and
        record['parent_task_id'] == screen.core.task_id(task) == frozen['parent_task_id'], 'own D2P task/branch')
    context = result['context']; prefix_commands = source.raw5(plan['original_prefix_commands'])
    wanted = dict(task['root_context'],
        latent_history=deepcopy(task['root_context']['latent_history'][1:])+[array(prefix, (192,)).tolist()],
        action_history=[deepcopy(task['root_context']['action_history'][-1]),
                        screen.core.normalized_tokens(task, prefix_commands)[0].astype(float).tolist()])
    screen.core.context(context)
    require(context == frozen['context'] == plan['observed_context'] == wanted and
        screen.core.digest(context) == record['observed_context_sha256'] and
        plan['record'] == record, 'actual prefix full3-state/2-action content and layout')
    j = array(fresh['jacobian'], (192, 10))
    require(np.array_equal(j, array(result['fresh_jacobian'], (192, 10))) and
        zero['terminal'] == fresh['nominal'] == result['zero_nominal'], 'raw fresh J and zero anchor durable copies')
    nominal = advance(task, context, np.zeros((5, 2), np.float32), zero['terminal'])
    c0 = result['candidates'][0]
    require(c0['index'] == 0 and c0['kind'] == 'zero_nominal' and
        c0['commands'] == np.zeros((5, 2)).tolist() and c0['context'] == nominal, 'original zero_nominal full path')
    je = [e for e in events if e['kind'] == 'jacobian']
    require(len(je) == 2 and je[0]['completed'] is False and je[1]['completed'] is True and
        all(e['shape'] == [192, 10] and e['mathematical_vjp_directions'] == 192 for e in je), 'original H1 AD shape and completion')
    p = dict(point_order=point_order, branch=record['branch'], source_slot=record['source_slot'],
        arm=record['arm'], parent_task_id=record['parent_task_id'],
        observed_problem_id=record['observed_problem_id'], blind_sha256=record['blind_sha256'],
        task_export=record['task_export'], task_sha256=record['task_sha256'],
        context_sha256=screen.core.digest(context), at_logits=np.zeros((5, 2)).tolist(),
        at_terminal=zero['terminal'], nominal_context_sha256=screen.core.digest(nominal),
        logit_J_sha256=screen.core.digest(j.tolist()),
        saved_Jv={d['id']: (j@np.asarray(d['vector'])).tolist() for d in directions()})
    return p, j, dict(task=task, context=context)


def source_closure(bindings):
    pending = [Path(__file__), Path(fd.__file__), Path(source.__file__),
        ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')]
    seen = set()
    while pending:
        f = pending.pop().resolve()
        if f in seen: continue
        seen.add(f); bindings[str(f)] = sha(f)
        if f.suffix != '.py' or not f.is_relative_to(ROOT/'experiments'): continue
        for node in ast.walk(ast.parse(f.read_text())):
            names = []
            if isinstance(node, ast.Import): names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('experiments'):
                names = [node.module]+[node.module+'.'+a.name for a in node.names]
            for name in names:
                child = ROOT/Path(*name.split('.')).with_suffix('.py')
                if name.startswith('experiments.') and child.is_file(): pending.append(child)


def collect():
    bindings = {}
    def load(path, expected):
        path = Path(path).absolute()
        require(sha(path) == expected, 'external saved SHA: '+str(path))
        bindings[str(path)] = expected
        return base.read(path)
    r = load(INPUT, INPUT_SHA)
    require(r['schema'] == source.NAME+'.report' and r['status'] == 'observed_suffix_plans_sealed_not_admitted' and
        r['errors'] == [] and r['first_error'] is None and r['source_verified'] is True and
        r['model_unchanged'] is True and r['suffix_commands_released'] is True and
        r['prepare_sha256'] == PARENT_PREPARE_SHA and len(r['rows']) == 4, 'actual four-branch complete feedback source')
    base.exact_counts(r['resources'], dict(predict=20, action_encoder=20, jacobian=4, svd=4))
    binding = load(INPUT.parent/'binding.json', r['artifact_sha256']['binding.json'])
    require(binding['prepare_sha256'] == PARENT_PREPARE_SHA, 'actual parent prepare chain')
    p = load(binding['prepared'], PARENT_PREPARE_SHA)
    require(p['schema'] == source.NAME+'.prepare' and p['contract'] == r['contract'] == source.contract() and
        r['model_hash_before'] == r['model_hash_after'] == p['expected_model_hash'] == base.encode.MODEL_TENSOR_SHA,
        'original source/model/numerical contract')
    require(p['bindings'][str(Path(source.__file__).resolve())] == sha(source.__file__), 'frozen original derivative implementation')
    # Pin upstream seals but never call source.material/verify/consume: those
    # routines also traverse selected terminal outcomes, irrelevant to this FD.
    for path in base.encode.MODELS:
        require(path in p['bindings'], 'original policy/config binding')
        bindings[path] = p['bindings'][path]
    config = base.read(Path(p['policy'])/'config.json')
    require(config['encoder'].get('pretrained') is False and str(Path(p['policy'])/'config.json') in bindings,
        'local frozen config, no downloads')
    rgb_path = Path(p['arguments']['rgb_report'])
    rgb = load(rgb_path, p['arguments']['rgb_sha'])
    require(rgb['status'] == 'selected_rgb_encoded_not_admitted' and rgb['errors'] == [] and
        rgb['source_verified'] is True and rgb['model_unchanged'] is True and
        rgb['model_hash_after'] == p['expected_model_hash'], 'actual observed prefix encoder receipt')
    journal_path = INPUT.parent/'events.jsonl.gz'
    require(sha(journal_path) == r['artifact_sha256'][journal_path.name], 'original suffix journal SHA')
    bindings[str(journal_path)] = r['artifact_sha256'][journal_path.name]
    raw, eof = base.inflate(journal_path.read_bytes())
    require(eof and raw.endswith(b'\n'), 'complete original saved journal')
    events = [base.decode(line) for line in raw.splitlines()]
    require([x['branch'] for x in p['branches']] == list(range(4)) and
        [(x['source_slot'], x['arm']) for x in r['rows']] == [(i,a) for i in range(2) for a in base.ARMS], 'original full4 census')
    points, problems, matrices = [], {}, {}
    for k, branch in enumerate(BRANCHES):
        rec = p['branches'][branch]
        require(r['rows'][branch]['record'] == rec and rec['RGB_report_sha256'] == p['arguments']['rgb_sha'], 'own parent branch receipt')
        export = load(rec['task_export'], rec['task_sha256'])
        approved = base.load_export(rec['task_export'], rec['task_sha256'])
        require(export['source_slot'] == k and approved['task_id'] == rec['parent_task_id'] and
            export['blind_sha256'] == rec['blind_sha256'], 'approved opaque task, no invented root')
        result = load(INPUT.parent/f'branch_{branch}.result.json.gz', r['artifact_sha256'][f'branch_{branch}.result.json.gz'])
        plan = load(INPUT.parent/f'branch_{branch}.plan.json', r['artifact_sha256'][f'branch_{branch}.plan.json'])
        row = rgb['rows'][2*branch]; entry = row['latent']
        require(row['branch'] == branch and row['role'] == 'prefix5' and row['root_slot'] == k and
            row['arm'] == rec['arm'] and row['task_id'] == rec['parent_task_id'] and
            row['task_sha256'] == rec['task_sha256'] and entry['file'] == rec['prefix_latent_file'] and
            entry['sha256'] == rec['prefix_latent_sha256'] == rgb['artifact_sha256'][entry['file']] and
            entry['shape'] == [192] and entry['dtype'] == 'float32' and Path(entry['file']).name == entry['file'], 'own prefix5 only')
        f = rgb_path.parent/entry['file']; require(sha(f) == entry['sha256'] and f.stat().st_size < 4096, 'saved prefix NPY seal')
        bindings[str(f)] = entry['sha256']
        stream = bytesio.BytesIO(f.read_bytes()); prefix = np.load(stream, allow_pickle=False)
        require(stream.tell() == len(stream.getvalue()) and prefix.shape == (192,) and
            prefix.dtype == np.dtype('float32') and np.isfinite(prefix).all(), 'literal complete native192 prefix')
        own = [e for e in events if e['branch'] == branch]
        require(own and all(e['observed_problem_id'] == rec['observed_problem_id'] for e in own), 'journal observed-problem join')
        point, j, problem = extract_point(approved['gn_task'], rec, result, plan, [e['event'] for e in own], k, prefix)
        points.append(point); matrices[k] = j; problems[k] = problem
    source_closure(bindings)
    for path, expected in bindings.items(): require(sha(path) == expected, 'final source/input seal: '+path)
    prepared = dict(schema=NAME+'.prepare', input_report=str(INPUT), input_report_sha256=INPUT_SHA,
        input_prepare_sha256=PARENT_PREPARE_SHA, points=points, directions=directions(), epsilons=list(EPS), signs=list(SIGNS),
        budget=budget(), policy=p['policy'], expected_model_hash=p['expected_model_hash'],
        numerical=p['contract']['numerical'], bindings_sha256=bindings,
        order='D2P branch1 then3; zero anchor then direction/epsilon/minus-plus; all restart same observed context',
        perturbation='float64 +/- eps unit v -> float32 logits5x2 once -> NumPy float32 tanh once',
        derivative='stored192x10 logit-J at zero includes tanh chain=1; do not normalize or apply chain again',
        history='original observed states3/actions2, appended one normalized raw packet, never fullH2 rollout',
        outcome_access='no terminal NPY/physical outcomes or costs used; only prefix NPY and saved zero/J fields',
        inherited_feedback_resources=r['resources'], new_Q=False, training=False, new_physics=0,
        scientific_pass_threshold=None, derivative_verdict='not_assigned', P1_admission=False,
        output_cap_bytes=CAP, floor_bytes=FLOOR, final_reserve_bytes=FINAL)
    return prepared, problems, matrices


def prepare(output):
    p, _, _ = collect(); raw = canonical(p); output = Path(output).absolute()
    require(not output.exists() and len(raw) < PREPARE_CAP, 'exclusive small prepare')
    require(shutil.disk_usage(output.parent).free >= FLOOR+len(raw)+4096, '200MiB prepare reserve')
    with output.open('xb') as f:
        require(f.write(raw) == len(raw), 'short prepare'); f.flush(); os.fsync(f.fileno())
    io.fsync_directory(output.parent)
    return dict(prepared=str(output), prepare_sha256=sha(output), bytes=len(raw), budget=budget())


def verify(path, expected):
    screen.core.hash_value(expected)
    require(Path(path).stat().st_size < PREPARE_CAP and sha(path) == expected, 'external prepare SHA')
    p = base.read(path); current, problems, matrices = collect()
    require(p == current, 'complete saved reconstruction before/after execution')
    return p, problems, matrices


def perturb(vector, epsilon, sign):
    v = array(vector, (10,))
    require(epsilon in EPS and sign in SIGNS and abs(np.linalg.norm(v)-1) < 1e-14, 'frozen unit direction/epsilon/sign')
    x = (sign*epsilon*v.reshape(5, 2)).astype(np.float32)
    return dict(logits=x.astype(float).tolist(), commands=np.tanh(x).astype(float).tolist())


def requests(p):
    require(len(p['points']) == 2 and [x['point_order'] for x in p['points']] == [0,1] and
        [x['branch'] for x in p['points']] == list(BRANCHES) and p['directions'] == directions() and
        p['epsilons'] == list(EPS) and p['signs'] == list(SIGNS), 'fixed two H1 points and protocol')
    rows = []
    for point in p['points']:
        require(not array(point['at_logits'], (5,2)).any(), 'zero nominal only')
        common = {k:point[k] for k in ('point_order','branch','source_slot','arm','parent_task_id','observed_problem_id')}
        rows.append(dict(common, order=len(rows), mode='nominal', direction=None, epsilon=None, sign=0,
                         logits=point['at_logits'], commands=np.zeros((5,2)).tolist()))
        for d in directions():
            for eps in EPS:
                for sign in SIGNS:
                    rows.append(dict(common, order=len(rows), mode='central_side', direction=d['id'], epsilon=eps,
                                     sign=sign, **perturb(d['vector'], eps, sign)))
    require(len(rows) == 38, 'exact38 H1 evaluations')
    return rows


class Meter(fd.Meter):
    """Own H1 dispatch/packet caps; never inherit H2 start/begin/emit logic."""
    def start(self, request):
        require((self.request is None or self.packet == 1) and request['order'] == len(self.events)//2 and
            request['order'] < 38 and len(self.events) == 2*request['order'] and
            all(e['completed'] for e in self.events), 'once-only ordered H1 requests')
        self.request = request; self.packet = 0; self.paths = []
        self.emit_event(dict(kind='evaluation_attempt', request=deepcopy(request)))
    def begin(self, resource):
        require(self.request is not None and self.packet == 0 and self.no_grad(), 'no-grad one-packet only')
        offset = len(self.events)-2*self.request['order']
        require(offset in (0,1) and resource == ('action_encoder','predict')[offset] and
            all(e['completed'] for e in self.events), 'H1 entry order, no failed retry')
        require(sum(e['resource'] == resource for e in self.events) < 38, 'hard predispatch38 cap')
        e = dict(index=len(self.events), resource=resource, samples=1, completed=False,
                 evaluation_order=self.request['order'], packet=0)
        self.events.append(e); self.emit_event(dict(kind='module_attempt', entry=deepcopy(e)))
        return e
    def emit(self, kind, **value):
        require(kind == 'native_packet' and value['packet'] == self.packet == 0 and
            value['mode'] == 'observed_H1' and len(self.events) == 2*(self.request['order']+1) and
            all(e['completed'] for e in self.events), 'exact H1 packet chronology')
        c = deepcopy(value['context']); self.paths.append(c)
        self.emit_event(dict(kind='native_packet', evaluation_order=self.request['order'], packet=0,
                             context=c, context_sha256=screen.core.digest(c)))
        self.packet = 1


class NativeH1(screen.NativeCuda):
    def __init__(self, model, device, task):
        super().__init__(model, device, task); self.task = deepcopy(task)
    def step(self, context, raw, meter):
        screen.core.context(context)
        require(meter is self._active_meter, 'own live meter')
        with self.torch.no_grad():
            z = self._tensor(context['latent_history'])[None]
            h = self._tensor(context['action_history'])[None]
            z, h = self._step(self.task, z, h, self._tensor(raw))
            c = self._saved(z, h)
            meter.emit('native_packet', mode='observed_H1', packet=0, context=c)
        return c


def evaluate(p, problems, backend_for_point, meter, sink, rows=None):
    rows = [] if rows is None else rows; current = None
    with ExitStack() as active:
        for request in requests(p):
            k = request['point_order']; problem = problems[k]; backend = backend_for_point(k)
            if k != current:
                active.close(); active.enter_context(backend.guard(meter)); current = k
            require(screen.core.digest(problem['context']) == p['points'][k]['context_sha256'], 'bound observed context')
            meter.start(request)
            c = backend.step(deepcopy(problem['context']), np.asarray(request['commands'], np.float32), meter)
            require(meter.paths == [c], 'one complete native context evidence')
            fd.audit.history(problem['task'], problem['context'], c, request['commands'])
            row = dict(request, context=c, context_sha256=screen.core.digest(c), terminal=c['latent_history'][-1])
            sink(dict(kind='evaluation_complete', evaluation=row)); rows.append(row)
            if request['mode'] == 'nominal': check_anchor(p['points'][k], row)
    return rows


def check_anchor(point, row):
    require(row['terminal'] == point['at_terminal'] and row['context_sha256'] == point['nominal_context_sha256'],
            'nominal EXACT original zero_nominal context; no retry')


def analyze(p, rows, matrices):
    wanted = requests(p)
    require(len(rows) == 38 and all({k:r[k] for k in w} == w for r,w in zip(rows,wanted,strict=True)), 'all38 requests no filtering')
    comparisons, stability = [], []; offset = 0
    for point in p['points']:
        check_anchor(point, rows[offset]); offset += 1
        j = array(matrices[point['point_order']], (192,10))
        require(screen.core.digest(j.tolist()) == point['logit_J_sha256'], 'full saved logit J seal')
        for d in directions():
            group = []
            for eps in EPS:
                minus, plus = rows[offset:offset+2]; offset += 2
                for row in (minus,plus):
                    require(screen.core.digest(row['context']) == row['context_sha256'] and
                        row['terminal'] == row['context']['latent_history'][-1], 'returned context/terminal binding')
                a,b = array(minus['terminal'], (192,)),array(plus['terminal'], (192,))
                measured = (b-a)/(2*eps); v = np.asarray(d['vector'])
                effective = (np.asarray(plus['logits']).reshape(10)-np.asarray(minus['logits']).reshape(10))/(2*eps)
                ref = j@v; eref = j@effective
                require(np.array_equal(ref,point['saved_Jv'][d['id']]), 'saved Jv')
                item = dict(point_order=point['point_order'], branch=point['branch'], source_slot=point['source_slot'],
                    direction=d['id'], epsilon=eps, fd=measured.tolist(), saved_Jv=ref.tolist(),
                    effective_float32_Jv=eref.tolist(), effective_direction=effective.tolist(),
                    effective_direction_error_l2=float(np.linalg.norm(effective-v)),
                    midpoint_drift_l2=float(np.linalg.norm((a+b)/2-np.asarray(point['at_terminal']))),
                    requested_direction_metrics=errors(measured,ref), effective_direction_metrics=errors(measured,eref))
                comparisons.append(item); group.append(item)
            stability.append(dict(point_order=point['point_order'], branch=point['branch'], direction=d['id'],
                adjacent_epsilon_fd_l2=[float(np.linalg.norm(np.asarray(a['fd'])-b['fd'])) for a,b in zip(group,group[1:])]))
    require(offset == 38 and len(comparisons) == 18, 'complete18 central comparisons')
    return dict(schema=NAME+'.analysis', comparisons=comparisons, epsilon_stability=stability,
        nominal_anchors=2, scientific_pass_threshold=None, derivative_verdict='not_assigned', P1_admission=False)


def run(prepared, expected, output):
    p, problems, matrices = verify(prepared, expected)
    store = Store(output); issues, forbidden = [], []; hooks = io.Hooks()
    model = before = after = host = analysis_sha = None
    source_ok = False; rows = []; meter = Meter(store.event)
    validations = dict(attempted=1, completed=1)
    try:
        store.put('binding.json', dict(prepared=str(Path(prepared).absolute()), prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        host = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        model = load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        io.frozen(model); before = io.tensor_hash(model)
        require(before == p['expected_model_hash'], 'same original frozen model')
        meter.sync=lambda:torch.cuda.synchronize(device); meter.no_grad=lambda:not torch.is_grad_enabled()
        targets = [(torch.autograd,k) for k in ('grad','backward','_engine_run_backward')]
        targets += [(torch.autograd.functional,k) for k in ('jacobian','hessian','jvp','vjp','hvp','vhp')]
        targets += [(torch.Tensor,'backward')]
        targets += [(torch.func,k) for k in ('jacrev','jacfwd','hessian','jvp','vjp','grad','grad_and_value')]
        for obj,key in targets:
            if not hasattr(obj,key): continue
            def denied(*args,_key=key,**kwargs):
                forbidden.append(_key); store.event(dict(kind='forbidden_AD',entry=_key))
                raise ValueError('FD prohibits new AD: '+_key)
            hooks.install(obj,key,denied)
        backends = {k:NativeH1(model,device,v['task']) for k,v in problems.items()}
        with sdpa_kernel(SDPBackend.MATH): evaluate(p,problems,lambda k:backends[k],meter,store.event,rows)
        store.put('analysis.json.gz',analyze(p,rows,matrices)); analysis_sha=sha(store.output/'analysis.json.gz')
    except BaseException as exc:
        issues.append(failure('execution',exc))
        try: store.event(dict(kind='execution_error',error=issues[0]))
        except BaseException as e: issues.append(failure('error_journal',e))
    finally:
        hooks.restore(issues)
        if model is not None:
            try: io.frozen(model); after=io.tensor_hash(model)
            except BaseException as e: issues.append(failure('final_model',e))
        try:
            validations['attempted']+=1; verify(prepared,expected); validations['completed']+=1; source_ok=True
        except BaseException as e: issues.append(failure('final_source',e))
        try: store.close()
        except BaseException as e: issues.append(failure('journal_finalize',e))
        counts=meter.counts()
        complete=not issues and not forbidden and source_ok and before is not None and before==after and len(rows)==38 and analysis_sha is not None and counts=={k:dict(attempted=38,completed=38) for k in ('predict','action_encoder')}
        hashes={}
        for f in store.output.iterdir():
            if f.is_file():
                try: hashes[f.name]=sha(f)
                except BaseException as e: issues.append(failure('artifact_hash',e)); complete=False
        report=dict(schema=NAME+'.report',status='cuda_observed_H1_fd_completed' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).absolute()),prepare_sha256=expected,input_report_sha256=INPUT_SHA,
            errors=issues,first_error=issues[0] if issues else None,forbidden_AD=forbidden,host=host,
            numerical=p['numerical'],resources=counts,budget=budget(),evaluations_completed=len(rows),
            inherited_feedback_resources=p['inherited_feedback_resources'],saved_validation_calls=validations,
            model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,artifact_sha256=hashes,analysis_sha256=analysis_sha,
            new_AD_calls=0 if complete else None,new_images=0,new_physics=0,new_Q=False,training=False,
            scientific_pass_threshold=None,derivative_verdict='not_assigned',P1_admission=False,
            output_cap_bytes=CAP,floor_bytes=FLOOR)
        try: store.put('report.json',report,final=True)
        except BaseException as e:
            issues.append(failure('final_report',e)); complete=False
            print(canonical(dict(first_error=issues[0],errors=issues)).decode(),flush=True)
    require(complete and not issues,'partial H1 FD preserved; no retry: '+str(issues))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('prepare','verify','run'))
    for key in ('prepared','prepare-sha','output'): p.add_argument('--'+key)
    a=p.parse_args()
    if a.command=='prepare': result=prepare(a.output)
    elif a.command=='verify': result=dict(status='verified_saved_only',points=len(verify(a.prepared,a.prepare_sha)[0]['points']),budget=budget())
    else: result=run(a.prepared,a.prepare_sha,a.output)
    print(canonical(result).decode().strip())


if __name__=='__main__': main()
