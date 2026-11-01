"""Fixed all8 last-latent-only teacher-forced suffix5. MAIN HOST CUDA only.

Saved prepare/verify/consume import no neural/physics runtime. No encoder/refit.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p2_midpoint_teacher_20261028_v1 as mid

NAME = 'agent_p2_midpoint_teacher_suffix_20261028_v1'
ROOT, ARMS = mid.ROOT, mid.ARMS
require, sha, read, write, digest = mid.require, mid.sha, mid.read, mid.write, mid.digest
INPUTS = {
    'run': ('results/agent_p2_midpoint_teacher_20261028_v1_cuda_MAIN01/report.json',
            '7b0b5c00735a7326326a4619c86539fdb51b1bf99f384fe8794a71a6b4e6321c'),
    'strict': ('results/agent_p2_midpoint_teacher_20261028_v1_consume_MAIN01.json',
               '4e61d9251f86550d0830c33d5607cfe58d4b308b97865f7fc64de0013d25d135'),
    'audit': ('results/audit_p2_midpoint_MAIN_20261028_v1_MAIN01.json',
              'cff183a8726af0e9f19a16f6b68534f60651edf289bdbf7331b95f1d83ca66ae'),
}
MID_SOURCES = {
    ROOT / f'experiments/spectral_reachability/{mid.NAME}.py': 'e1f505809313550a110db1029e6c38d1addafdf7846dbd4f26c4ddea92246cfc',
    ROOT / f'tests/test_{mid.NAME}.py': '11d1b4e83114304e13858ee0815f7eb07465c2c684926546645524778314c4f7',
    ROOT / f'innerdocs/{mid.NAME}.md': '95d5b95938e904e3e496041110b829b743f98f34ffc0579c908aa6e71bcac9f5',
}
KINDS = ('action_encoder', 'predict', 'predictor', 'pred_proj')


def contract():
    return dict(roots=4, arms=list(ARMS), branches=8, suffix_raw_steps=5, full_execution_raw_steps=10,
                intervention='replace only paid midcontext latent_history[-1] with matching actual saved z5',
                keep='older2 latents, both incoming action tokens, raw suffix5, original goal0/Q/model/normalizer',
                order='root-major joint then composition; no outcome filtering/reselection/deduplication',
                predict_API_cap=8, action_encoder_cap=8, predictor_cap=8, pred_proj_cap=8,
                native_sample_axes='AE/API/predictor:8 batches x3 temporal rows; pred_proj:24 flat rows;8 suffix endpoints only',
                per_branch_each_cap=1, encoder_cap=0, projector_cap=0, AD_cap=0, physics_cap=0,
                batch=1, history_tokens=3, raw_packet_length=5, device='cuda:0', CPU_neural_fallback=False,
                precision=mid.precision_contract(), retries=False, training=False,
                output_cap_bytes=mid.CAP, floor_bytes=mid.FLOOR, wall_seconds=mid.WALL,
                wall_scope='operation boundaries, not preemptive watchdog',
                wall_accounting='predict API inclusive of predictor/pred_proj; axes must not be added',
                P1_admission=False, P2_admission=False, scientific_admission=False)


def native_abi(config):
    a, p, proj = (config[k] for k in ('action_encoder', 'predictor', 'pred_proj'))
    require(a['_target_'] == 'stable_worldmodel.wm.lewm.module.Embedder' and a['input_dim'] == 10 and a['emb_dim'] == 192 and
            p['_target_'] == 'stable_worldmodel.wm.lewm.module.Predictor' and p['num_frames'] == 3 and
            p['input_dim'] == p['output_dim'] == 192 and proj['_target_'] == 'stable_worldmodel.wm.lewm.module.MLP' and
            proj['input_dim'] == proj['output_dim'] == 192, 'bound original 3latent/3outgoing-token predictor ABI')
    return dict(latents=[1, 3, 192], outgoing_tokens=[1, 3, 10], action_embeddings=[1, 3, 192],
                predictor_output=[1, 3, 192], pred_proj_flat=[3, 192], predict_output=[1, 3, 192])


def f32(value, shape):
    a = mid.gap.array(value, shape)
    require(np.array_equal(a, a.astype(np.float32)), 'literal saved float32 precision')
    return a.astype(np.float32)


def intervention(paid, actual_z5, raw10, normalization):
    """Pure arrays: no root-history substitution, no teacher-forcing older states."""
    mid.context(paid)
    lat, history = f32(paid['latent_history'], (3, 192)), f32(paid['action_history'], (2, 10))
    actual, raw = f32(actual_z5, (192,)), f32(raw10, (10, 2))
    require(np.max(abs(raw)) <= 1, 'sealed bounded raw10')
    mean, std = (f32(normalization[k], (2,)) for k in ('mean', 'std'))
    require(np.all(std > 0), 'original positive std')
    token = ((raw[5:] - mean) / std).reshape(10)
    teacher = lat.copy()
    teacher[-1] = actual
    outgoing = np.concatenate((history, token[None]), axis=0)
    c = deepcopy(paid)
    c['latent_history'] = teacher.tolist()
    require(c['latent_history'][:2] == paid['latent_history'][:2] and c['action_history'] == paid['action_history'],
            'only last latent changed')
    return c, dict(paid_latents=lat, history_actions=history, actual_z5=actual, teacher_latents=teacher,
                   suffix_raw5=raw[5:], action_mean=mean, action_std=std, suffix_token=token, outgoing_tokens=outgoing)


def collect():
    bindings = {}
    def seal(path, expected):
        path = mid.regular(path)
        require(sha(path) == expected, 'saved source/input SHA: ' + str(path))
        bindings[str(path)] = expected
        return read(path)
    for path, h in MID_SOURCES.items():
        require(sha(path) == h, 'frozen midpoint source/test/doc')
        bindings[str(path)] = h
    docs = {k: seal(ROOT / path, h) for k, (path, h) in INPUTS.items()}
    run, strict, audit = (docs[k] for k in ('run', 'strict', 'audit'))
    folder = (ROOT / INPUTS['run'][0]).parent
    launch = read(folder / 'execution_binding.json')
    p = mid.verify(launch['prepared'], launch['prepare_sha256'])
    arrays = mid.validate_run(folder, run, p)
    require(strict['status'] == 'all8_saved_midpoint_verified_not_admitted' and strict['report_sha256'] == INPUTS['run'][1] and
            strict['prepare_sha256'] == run['prepare_sha256'] == launch['prepare_sha256'] and
            strict['root_denominator'] == 4 and strict['arm_denominator'] == 8 and strict['scientific_admission'] is False,
            'actual MAIN strict midpoint authority')
    require(audit['known_root_denominator'] == 4 and audit['independent_roots'] == 0 and
            audit['scientific_admission'] is False, 'MAIN all4 audit, no scientific authority')
    sidx, aidx = mid.gap.census(strict['rows']), mid.gap.census(audit['rows'])
    g = read(mid.GAP)
    gidx = mid.gap.census(g['rows'])
    original_teacher = read(ROOT / mid.gap.INPUTS['teacher'][0])
    teachers = {(e['root_sha256'], e['method']): e for e in original_teacher['encodings']}
    compiled = {x['root_sha256']: x['compiled'] for x in read(p['compilation_path'])['rows']}
    original_plans = mid.gap.census(read(ROOT / mid.gap.INPUTS['prepare'][0])['plans'])
    jobs = []
    for f, buffers, row in zip(p['frames'], arrays, run['rows'], strict=True):
        key = (f['root_slot'], f['arm'])
        c = compiled[f['root_sha256']]
        z5 = buffers['native192']
        scores = mid.numeric(z5, f['predicted_vector'], f['goal_vector'], c['scoring']['metric']['projection'])
        require(sidx[key]['root_sha256'] == f['root_sha256'] and sidx[key]['frame_binding_sha256'] == f['frame_binding_sha256'],
                'strict frame/actual z5 identity')
        for name, value in scores.items():
            mid.gap.close(sidx[key]['midpoint'][name], value, 'strict original midpoint metric')
            mid.gap.close(aidx[key][name], value, 'independent MAIN midpoint arithmetic')
        teacher, buffers0 = intervention(f['predicted_context'], z5.tolist(), f['raw_actions'], p['normalization'])
        terminal = gidx[key]
        encoded = teachers[f['root_sha256'], f['arm']]
        require(encoded['terminal_rgb_sha256'] == terminal['terminal_rgb_sha256'], 'same actual terminal image')
        z10 = f32(encoded['terminal_vector'], (192,))
        actual_scores = mid.numeric(z10, z10.tolist(), f['goal_vector'], c['scoring']['metric']['projection'])
        mid.gap.close(actual_scores['teacher_Q'], terminal['prediction']['physical_teacher_Q'], 'same terminal Q')
        job = dict(job_id=f['frame_id'] + '_suffix5', root_slot=f['root_slot'], root_sha256=f['root_sha256'], arm=f['arm'],
                   midpoint_frame_binding_sha256=f['frame_binding_sha256'],
                   midpoint_png_sha256=f['rgb_file_sha256'], midpoint_pixel_sha256=f['pixel_sha256'],
                   actual_z5_receipt=row['arrays']['native192'], actual_z5_folder=str(folder),
                   paid_mid_context=f['predicted_context'], paid_mid_context_sha256=f['predicted_context_sha256'],
                   teacher_mid_context=teacher, raw10=f['raw_actions'], raw_suffix5=f['raw_actions'][5:],
                   selection_digest=f['selection_digest'], handoff_sha256=f['handoff_sha256'],
                   goal=f['goal_vector'], metric_id=f['active_metric_id'], projection=c['scoring']['metric']['projection'],
                   action_normalization=p['normalization'], actual_terminal_z=z10.tolist(),
                   actual_terminal_png_sha256=terminal['terminal_rgb_sha256'],
                   original_terminal_prediction=terminal['prediction'])
        # The original predicted terminal comes from the same sealed selected plan,
        # not an encoded physical outcome or a newly replayed model.
        plan = original_plans[key]['execution_plan']
        require(plan['root_sha256'] == f['root_sha256'] and plan['commands'] == f['raw_actions'] and
                plan['goal'] == f['goal_vector'] and plan['active_metric_id'] == f['active_metric_id'] and
                plan['selected_predicted_context']['latent_history'][1] == f['predicted_vector'],
                'same full raw10/goal/Q/paid step5/terminal chain')
        job['original_terminal_latent'] = plan['selected_predicted_context']['latent_history'][-1]
        job['buffer_digests'] = {k: digest(v.tolist()) for k, v in buffers0.items()}
        job['job_binding_sha256'] = digest(job)
        jobs.append(job)
    bindings.update(p['bindings'])
    bindings.update(p['historical_sources'])
    bindings.update(p['sources_sha256'])
    for path in [folder / 'execution_binding.json', folder / 'events.jsonl', mid.GAP,
                 Path(launch['prepared']) / 'prepare.json', Path(launch['prepared']) / 'protocol.json']:
        bindings[str(path)] = sha(path)
    for row in run['rows']:
        path = folder / (row['frame_id'] + '.row.json')
        bindings[str(path)] = sha(path)
        for receipt in row['arrays'].values():
            path = folder / receipt['file']
            bindings[str(path)] = receipt['sha256']
    for entry in run['entries']:
        for receipt in entry['arrays'].values():
            path = folder / receipt['file']
            bindings[str(path)] = receipt['sha256']
    config = read(Path(p['policy']) / 'config.json')
    for path, h in {**bindings, **p['model_bindings']}.items():
        require(sha(mid.regular(path)) == h, 'consumed closure drift')
    return dict(schema=NAME + '.protocol', contract=contract(), jobs=jobs, abi=native_abi(config), policy=p['policy'],
                expected_model_state_sha256=run['model_hash_before'],
                model_bindings=p['model_bindings'], bindings=bindings, source_midpoint_report_sha256=INPUTS['run'][1],
                source_midpoint_strict_sha256=INPUTS['strict'][1], inherited_resources=p['inherited_resources'],
                inherited_midpoint_encoding_counts=run['counts'], inherited_terminal_encoder_samples=8,
                unique_midpoint_images=p['unique_step5_image_count'], new_independent_roots=0)


def source_files():
    return sorted(set(mid.source_files() + [Path(__file__).resolve(), ROOT / f'tests/test_{NAME}.py', ROOT / f'innerdocs/{NAME}.md']))


def prepare(output):
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink() and shutil.disk_usage(output.parent).free > mid.CAP + mid.FLOOR,
            'NEW prepare/headroom')
    p = collect()
    p['sources_sha256'] = {str(path): sha(mid.regular(path)) for path in source_files()}
    output.mkdir(exist_ok=False)
    write(output / 'protocol.json', p)
    s = mid.create_seal(files=source_files(), args_json=mid.gap.encode(dict(protocol_sha256=sha(output/'protocol.json'))).decode(),
                        output_dir=output/'source_seal')
    write(output/'prepare.json', dict(schema=NAME+'.prepare', status='prepared_not_executed', source_seal=s,
                                      protocol_sha256=sha(output/'protocol.json'), new_neural_calls=0, new_physics_steps=0))
    return dict(prepared=str(output), prepare_sha256=sha(output/'prepare.json'))


def verify(prepared, expected):
    prepared = Path(prepared).absolute()
    require(sha(mid.regular(prepared/'prepare.json')) == expected, 'external prepare SHA')
    r, p = read(prepared/'prepare.json'), read(prepared/'protocol.json')
    require(r['schema'] == NAME+'.prepare' and r['status'] == 'prepared_not_executed' and
            sha(prepared/'protocol.json') == r['protocol_sha256'] and p['contract'] == contract(), 'own frozen protocol')
    s = r['source_seal']
    mid.verify_seal(output_dir=s['output_dir'], expected_manifest_sha256=s['manifest_sha256'])
    for path, h in {**p['bindings'], **p['sources_sha256'], **p['model_bindings']}.items():
        require(sha(mid.regular(path)) == h, 'bound input/source/model drift')
    rebuilt = collect()
    rebuilt['sources_sha256'] = p['sources_sha256']
    require(rebuilt == p, 'exact all8 teacher input reconstruction')
    return p


class Ledger:
    """8 each of nested AE/API/predictor/proj; nonrefundable, exact call tree."""
    def __init__(self, jobs, journal, persist, check=lambda: None, sync=lambda: None, precision=lambda: None):
        require(len(jobs) == 8 and len({j['job_id'] for j in jobs}) == 8, 'fixed8 jobs')
        self.order = [(j['job_id'], k) for j in jobs for k in KINDS]
        self.journal, self.persist, self.check, self.sync, self.precision = journal, persist, check, sync, precision
        self.active, self.entries, self.stack, self.blocked, self.journal_errors = None, [], [], [], []

    def emit(self, value):
        try: self.journal(value)
        except BaseException as exc:
            self.journal_errors.append(mid.failure('journal', exc))
            raise

    def dispatch(self, kind, callback, args, kwargs):
        try:
            self.check()
            require(len(self.entries) < 32 and (self.active, kind) == self.order[len(self.entries)] and
                    not self.blocked and not self.journal_errors, 'fixed order/cap/no retry')
            require([self.entries[n]['kind'] for n in self.stack] == (['predict'] if kind in ('predictor', 'pred_proj') else []),
                    'native nested API topology')
            require(all(e['saved'] or e['index'] in self.stack for e in self.entries), 'no retry after failed entry')
            shapes = ((1, 3, 10),) if kind == 'action_encoder' else ((3, 192),) if kind == 'pred_proj' else ((1, 3, 192),)*2
            require(not kwargs and len(args) == len(shapes) and all(
                getattr(getattr(x, 'device', None), 'type', None) == 'cuda' and tuple(x.shape) == shape
                for x, shape in zip(args, shapes)), 'native CUDA argument ABI; no CPU fallback')
        except BaseException as exc:
            self.blocked.append(mid.failure('predispatch', exc))
            raise
        e = dict(index=len(self.entries), job_id=self.active, kind=kind, arrays={}, native_entered=False,
                 native_returned=False, saved=False, precision=None, wall_seconds=None)
        self.entries.append(e)
        primary, started = None, None
        try:
            self.emit(dict(event='attempt', index=e['index'], job_id=self.active, kind=kind))
            for n, value in enumerate(args):
                e['arrays']['input_'+str(n)] = self.persist(self.active+'.'+kind+'.input_'+str(n), value)
            e['precision'] = self.precision()
            self.check()
            self.emit(dict(event='enter', index=e['index'], job_id=self.active, kind=kind))
            self.stack.append(e['index'])
            e['native_entered'] = True
            started = time.monotonic()
            result = callback(*args)
            e['native_returned'] = True
            self.sync()
            e['wall_seconds'] = time.monotonic() - started
            shape = (3, 192) if kind == 'pred_proj' else (1, 3, 192)
            require(tuple(result.shape) == shape, 'native module output ABI')
            e['arrays']['output'] = self.persist(self.active+'.'+kind+'.output', result)
            e['saved'] = True
            return result
        except BaseException as exc:
            primary = exc
            e['error'] = mid.failure('native', exc)
            if started is not None and e['wall_seconds'] is None: e['wall_seconds'] = time.monotonic()-started
            raise
        finally:
            if self.stack and self.stack[-1] == e['index']: self.stack.pop()
            try: self.emit(dict(event='return_or_error', entry=deepcopy(e)))
            except BaseException:
                if primary is None: raise

    def forbid(self, kind):
        def denied(*args, **kwargs):
            self.blocked.append(mid.failure('forbidden', RuntimeError(kind)))
            self.emit(dict(event='forbidden', job_id=self.active, kind=kind))
            raise RuntimeError('forbidden '+kind)
        return denied

    def counts(self):
        result = {}
        for kind in KINDS:
            es = [e for e in self.entries if e['kind'] == kind]
            times = [e['wall_seconds'] for e in es if e['native_entered']]
            result[kind] = dict(attempted=len(es), entered=sum(e['native_entered'] for e in es),
                                returned=sum(e['native_returned'] for e in es), saved=sum(e['saved'] for e in es),
                                input_axis0_samples=sum(e['arrays']['input_0']['shape'][0] for e in es if e['native_entered']),
                                input_temporal_rows=sum(e['arrays']['input_0']['shape'][0] *
                                    (1 if kind == 'pred_proj' else e['arrays']['input_0']['shape'][1])
                                    for e in es if e['native_entered']),
                                inclusive_wall_seconds=None if any(t is None for t in times) else sum(times))
        return result


def predict_jobs(p, output, adapter, ledger, journal, rows, errors, check=lambda: None):
    for job in p['jobs']:
        row = dict(job_id=job['job_id'], job_binding_sha256=job['job_binding_sha256'], status='blocked_not_attempted', arrays={})
        if not errors:
            ledger.active = job['job_id']
            row['status'] = 'attempted'
            try:
                check()
                journal(dict(event='job_attempt', job_id=job['job_id'], job_binding_sha256=job['job_binding_sha256']))
                teacher, buffers = intervention(job['paid_mid_context'],
                    job['teacher_mid_context']['latent_history'][-1], job['raw10'], job['action_normalization'])
                require(teacher == job['teacher_mid_context'] and
                        {k: digest(v.tolist()) for k, v in buffers.items()} == job['buffer_digests'], 'exact source intervention buffers')
                for key, value in buffers.items(): row['arrays'][key] = mid.save_array(output, job['job_id']+'.'+key, value)
                whole = np.asarray(adapter(job, buffers))
                require(whole.dtype == np.float32 and whole.shape == (1, 3, 192) and np.isfinite(whole).all(), 'native full3 predictions')
                latents = np.concatenate((buffers['teacher_latents'][1:], whole[0, -1:]), axis=0)
                arrays = dict(api_predictions=whole, terminal_z=whole[0, -1], next_latents=latents,
                              next_history_actions=buffers['outgoing_tokens'][1:])
                for key, value in arrays.items(): row['arrays'][key] = mid.save_array(output, job['job_id']+'.'+key, value)
                require([e['kind'] for e in ledger.entries if e['job_id'] == job['job_id'] and e['saved']] == list(KINDS),
                        'exact four paid entries per job')
                row['status'] = 'predicted_saved'
                journal(dict(event='job_saved', job_id=job['job_id'], arrays=deepcopy(row['arrays'])))
            except BaseException as exc:
                errors.append(mid.failure('job.'+job['job_id'], exc))
                row.update(status='failed', error=errors[-1])
            finally: ledger.active = None
        rows.append(row)
        try: write(Path(output)/(job['job_id']+'.row.json'), row)
        except BaseException as exc: errors.append(mid.failure('row_persistence', exc))


def execute_cuda(prepared, expected, output):
    """Explicit MAIN HOST CUDA entry; workers never invoke this."""
    p, output = verify(prepared, expected), Path(output).absolute()
    require(not output.exists() and not output.is_symlink() and shutil.disk_usage(output.parent).free > mid.FLOOR+mid.CAP,
            'NEW run / reserve')
    write(Path(prepared).absolute().with_name(Path(prepared).name+'.launch.json'), dict(output=str(output), prepare_sha256=expected))
    output.mkdir(exist_ok=False)
    write(output/'execution_binding.json', dict(prepared=str(Path(prepared).absolute()), prepare_sha256=expected,
                                               source_midpoint_report_sha256=INPUTS['run'][1]))
    rows, errors, hooks, contexts = [], [], mid.Hooks(), ExitStack()
    model = host = before = after = precision_before = precision_after = original_tf32 = None
    source_ok, started = False, time.monotonic()
    with (output/'events.jsonl').open('x') as stream:
        journal = mid.Journal(stream)
        def persist(name, tensor):
            require(tensor.device.type == 'cuda' and str(tensor.dtype) == 'torch.float32', 'native float32 CUDA buffer')
            return mid.save_array(output, name, tensor.detach().cpu().numpy().copy())
        ledger = Ledger(p['jobs'], journal, persist, check=lambda: mid.boundary(output, started))
        try:
            import torch
            from torch.nn.attention import sdpa_kernel, SDPBackend
            from experiments.checkpoint_compat import load_compatible_pretrained
            from experiments.gpu_preflight import require_cuda_allocation
            host = require_cuda_allocation('cuda:0')
            device = torch.device('cuda:0')
            original_tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            contexts.enter_context(sdpa_kernel(SDPBackend.MATH))
            contexts.enter_context(torch.autocast(device_type='cuda', enabled=False))
            precision_before = mid.checked_precision(torch)
            ledger.precision = lambda: mid.checked_precision(torch)
            ledger.sync = lambda: torch.cuda.synchronize(device)
            model = load_compatible_pretrained(p['policy'], device=device).eval().requires_grad_(False)
            mid.frozen(model)
            before = mid.model_hash(model)
            require(before == p['expected_model_state_sha256'], 'same original midpoint frozen model state')
            for kind, obj, attr in (('action_encoder', model.action_encoder, 'forward'), ('predict', model, 'predict'),
                                     ('predictor', model.predictor, 'forward'), ('pred_proj', model.pred_proj, 'forward')):
                original = getattr(obj, attr)
                def counted(*args, _kind=kind, _original=original, **kwargs):
                    return ledger.dispatch(_kind, _original, args, kwargs)
                hooks.install(obj, attr, counted)
            for obj, key, kind in ((model, 'encode', 'encode'), (model, 'rollout', 'rollout'),
                                   (model.encoder, 'forward', 'encoder'), (model.projector, 'forward', 'projector'),
                                   (torch.autograd, 'grad', 'AD.grad'), (torch.autograd, 'backward', 'AD.backward'),
                                   (torch.Tensor, 'backward', 'AD.Tensor.backward')):
                hooks.install(obj, key, ledger.forbid(kind))
            for key in ('jacobian', 'hessian', 'jvp', 'vjp', 'hvp', 'vhp'):
                hooks.install(torch.autograd.functional, key, ledger.forbid('AD.'+key))
            for key in ('jvp', 'vjp', 'jacrev', 'jacfwd', 'hessian', 'grad', 'grad_and_value'):
                hooks.install(torch.func, key, ledger.forbid('AD.func.'+key))
            def adapter(job, buffers):
                with torch.inference_mode():
                    lat = torch.tensor(buffers['teacher_latents'], dtype=torch.float32, device=device)[None]
                    outgoing = torch.tensor(buffers['outgoing_tokens'], dtype=torch.float32, device=device)[None]
                    predictions = model.predict(lat, model.action_encoder(outgoing))
                    require(predictions.device.type == 'cuda' and predictions.dtype == torch.float32 and
                            tuple(predictions.shape) == (1, 3, 192) and bool(torch.isfinite(predictions).all()), 'native suffix full3 output')
                    return predictions.detach().cpu().numpy().copy()
            predict_jobs(p, output, adapter, ledger, journal, rows, errors, lambda: mid.boundary(output, started))
        except BaseException as exc: errors.append(mid.failure('setup_or_execution', exc))
        finally:
            hooks.restore(errors)
            if model is not None:
                try: mid.frozen(model); after = mid.model_hash(model)
                except BaseException as exc: errors.append(mid.failure('final_model_hash', exc))
            if precision_before is not None:
                try: precision_after = mid.checked_precision(torch)
                except BaseException as exc: errors.append(mid.failure('final_precision', exc))
            try:
                contexts.close()
                if original_tf32 is not None:
                    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original_tf32
            except BaseException as exc: errors.append(mid.failure('precision_restore', exc))
            try: verify(prepared, expected); mid.boundary(output, started); source_ok = True
            except BaseException as exc: errors.append(mid.failure('final_input_wall', exc))
            if len(rows) < 8:
                predict_jobs(dict(jobs=p['jobs'][len(rows):]), output, None, ledger, journal, rows,
                             errors or [dict(stage='setup', message='adapter unavailable')])
            model_ok = before is not None and before == after
            if not model_ok: errors.append(dict(stage='model_integrity', message='unknown or changed model'))
            complete = (not errors and not ledger.blocked and not ledger.journal_errors and source_ok and model_ok and
                        precision_before == precision_after == mid.precision_contract() and len(rows) == 8 and
                        all(row['status'] == 'predicted_saved' for row in rows) and len(ledger.entries) == 32)
            if not complete and not errors: errors.append(dict(stage='census', message='incomplete job/native entries'))
            r = dict(schema=NAME+'.report', status='all8_suffix_predictions_saved' if complete else 'partial_or_failed',
                     prepare_sha256=expected, source_midpoint_report_sha256=INPUTS['run'][1], contract=contract(), rows=rows,
                     entries=ledger.entries, counts=ledger.counts(), blocked=ledger.blocked, journal_errors=ledger.journal_errors,
                     errors=errors, first_error=errors[0] if errors else None, journal_sha256=sha(output/'events.jsonl'),
                     journal_tail_sha256=journal.previous, host=host, model_hash_before=before, model_hash_after=after,
                     model_unchanged=model_ok, source_verified=source_ok, model_bindings=p['model_bindings'],
                     precision_before=precision_before, precision_after=precision_after, total_wall_seconds=time.monotonic()-started,
                     new_encoder_calls=0, new_projector_calls=0, new_AD_calls=0, new_physics_steps=0,
                     CUDA_ns=None, FLOPs=None, P1_admission=False, P2_admission=False, scientific_admission=False)
            write(output/'report.json', r)
    if not complete: raise RuntimeError('partial/failed suffix run; full8 retained, no retry')
    return dict(status=r['status'], report=str(output/'report.json'), report_sha256=sha(output/'report.json'))


def validate_run(folder, r, p):
    folder = Path(folder).absolute()
    require(all(type(j['root_slot']) is int for j in p['jobs']) and
            [(j['root_slot'], j['arm']) for j in p['jobs']] == [(i, a) for i in range(4) for a in ARMS] and
            len({j['job_id'] for j in p['jobs']}) == 8, 'fixed canonical four-root paired denominator')
    require(r['schema'] == NAME+'.report' and r['status'] == 'all8_suffix_predictions_saved' and r['contract'] == contract() and
            r['errors'] == r['blocked'] == r['journal_errors'] == [] and r['first_error'] is None and
            r['host']['device'] == 'cuda:0' and r['model_hash_before'] is not None and r['model_hash_before'] == r['model_hash_after'] and
            r['model_hash_before'] == p['expected_model_state_sha256'] and
            r['model_unchanged'] is r['source_verified'] is True and r['model_bindings'] == p['model_bindings'] and
            r['precision_before'] == r['precision_after'] == mid.precision_contract(), 'complete actual source/model/precision receipt')
    require(r['new_encoder_calls'] == r['new_projector_calls'] == r['new_AD_calls'] == r['new_physics_steps'] == 0 and
            r['P1_admission'] is r['P2_admission'] is r['scientific_admission'] is False and
            len(p['jobs']) == len(r['rows']) == 8 and len(r['entries']) == 32 and 0 <= r['total_wall_seconds'] < mid.WALL,
            'fixed8 closed scope/budget')
    require(sha(folder/'events.jsonl') == r['journal_sha256'], 'journal SHA')
    previous, payloads = None, []
    for i, line in enumerate((folder/'events.jsonl').read_bytes().splitlines()):
        e = mid.json.loads(line)
        h = e.pop('event_sha256')
        require(e['index'] == i and e['previous_sha256'] == previous and digest(e) == h, 'journal chain')
        previous = h
        payloads.append(e['payload'])
    require(previous == r['journal_tail_sha256'], 'journal tail')
    expected, results = [], []
    for i, (job, row) in enumerate(zip(p['jobs'], r['rows'], strict=True)):
        jid = job['job_id']
        require(digest({k: v for k, v in job.items() if k != 'job_binding_sha256'}) == job['job_binding_sha256'] and
                row['job_id'] == jid and row['job_binding_sha256'] == job['job_binding_sha256'] and row['status'] == 'predicted_saved' and
                read(folder/(jid+'.row.json')) == row, 'ordered original job/intervention/row identity')
        teacher, source = intervention(job['paid_mid_context'], job['teacher_mid_context']['latent_history'][-1], job['raw10'], job['action_normalization'])
        require(digest(job['paid_mid_context']) == job['paid_mid_context_sha256'] and
                teacher == job['teacher_mid_context'] and {k: digest(v.tolist()) for k, v in source.items()} == job['buffer_digests'],
                'last-latent-only intervention closure')
        expected.append(dict(event='job_attempt', job_id=jid, job_binding_sha256=job['job_binding_sha256']))
        native = {}
        es = r['entries'][4*i:4*i+4]
        for n, (kind, entry) in enumerate(zip(KINDS, es, strict=True)):
            inputs = ('input_0', 'input_1') if kind in ('predict', 'predictor') else ('input_0',)
            require(entry['index'] == 4*i+n and entry['job_id'] == jid and entry['kind'] == kind and
                    entry['native_entered'] is entry['native_returned'] is entry['saved'] is True and
                    entry['precision'] == mid.precision_contract() and 0 <= entry['wall_seconds'] < mid.WALL and
                    set(entry['arrays']) == set(inputs) | {'output'}, 'exact nested paid native entry')
            for key, receipt in entry['arrays'].items():
                require(receipt['file'] == jid+'.'+kind+'.'+key+'.npy', 'native file identity')
                a = mid.load_array(folder, receipt)
                require(a.dtype == np.float32, 'original float32 native buffers')
                native[kind+'.'+key] = a
        # AE completes before API; predictor/proj are nested inside API return.
        def begin(entry):
            return [dict(event=ev, index=entry['index'], job_id=jid, kind=entry['kind']) for ev in ('attempt', 'enter')]
        def end(entry): return dict(event='return_or_error', entry=entry)
        expected += begin(es[0])+[end(es[0])]+begin(es[1])+begin(es[2])+[end(es[2])]+begin(es[3])+[end(es[3]), end(es[1])]
        require(set(row['arrays']) == set(source) | {'api_predictions', 'terminal_z', 'next_latents', 'next_history_actions'}, 'all raw/normalization/context/output arrays')
        saved = {}
        for key, receipt in row['arrays'].items():
            require(receipt['file'] == jid+'.'+key+'.npy', 'own frame buffer name')
            saved[key] = mid.load_array(folder, receipt)
            require(saved[key].dtype == np.float32, 'full float32 frame buffers')
        for key, a in source.items(): require(np.array_equal(saved[key], a), 'frozen source buffer '+key)
        shape = lambda key, s: native[key].shape == s
        require(shape('action_encoder.input_0', (1,3,10)) and shape('action_encoder.output', (1,3,192)) and
                all(shape(k, (1,3,192)) for k in ('predict.input_0','predict.input_1','predict.output','predictor.input_0','predictor.input_1','predictor.output')) and
                shape('pred_proj.input_0', (3,192)) and shape('pred_proj.output', (3,192)), 'native full3 ABI shapes')
        require(np.array_equal(native['action_encoder.input_0'][0], source['outgoing_tokens']) and
                np.array_equal(native['predict.input_0'][0], source['teacher_latents']) and
                np.array_equal(native['action_encoder.output'], native['predict.input_1']) and
                np.array_equal(native['predictor.input_0'], native['predict.input_0']) and
                np.array_equal(native['predictor.input_1'], native['predict.input_1']) and
                np.array_equal(native['predictor.output'].reshape(3,192), native['pred_proj.input_0']) and
                np.array_equal(native['pred_proj.output'].reshape(1,3,192), native['predict.output']) and
                np.array_equal(saved['api_predictions'], native['predict.output']) and
                np.array_equal(saved['terminal_z'], native['predict.output'][0,-1]) and
                np.array_equal(saved['next_latents'], np.concatenate((source['teacher_latents'][1:], saved['terminal_z'][None]))) and
                np.array_equal(saved['next_history_actions'], source['outgoing_tokens'][1:]), 'native buffer/context/normalization closure')
        expected.append(dict(event='job_saved', job_id=jid, arrays=row['arrays']))
        results.append(saved['terminal_z'])
    require(payloads == expected, 'exact journal call-tree/frame sequence')
    for kind in KINDS:
        es = [e for e in r['entries'] if e['kind'] == kind]
        require(r['counts'][kind] == dict(attempted=8, entered=8, returned=8, saved=8,
                                         input_axis0_samples=24 if kind == 'pred_proj' else 8, input_temporal_rows=24,
                                         inclusive_wall_seconds=sum(e['wall_seconds'] for e in es)), 'per-axis8 counts, no wall double count')
    return results


def comparisons(p, predictions):
    rows = []
    require(len(predictions) == len(p['jobs']) == 8, 'full8 saved predictions')
    for job, z in zip(p['jobs'], predictions, strict=True):
        actual, old = job['actual_terminal_z'], job['original_terminal_latent']
        new_scores = mid.numeric(actual, z.tolist(), job['goal'], job['projection'])
        old_scores = mid.numeric(actual, old, job['goal'], job['projection'])
        mid.gap.close(old_scores['predicted_Q'], job['original_terminal_prediction']['Q_cost'], 'same original terminal objective')
        mid.gap.close(new_scores['teacher_Q'], job['original_terminal_prediction']['physical_teacher_Q'], 'same actual terminal Q')
        rows.append(dict(root_slot=job['root_slot'], root_sha256=job['root_sha256'], arm=job['arm'],
                         job_binding_sha256=job['job_binding_sha256'], actual_terminal_Q=new_scores['teacher_Q'],
                         original_open_loop=old_scores, teacher_forced_suffix=new_scores,
                         teacher_forced_minus_open_loop_Q=new_scores['predicted_Q']-old_scores['predicted_Q'],
                         latent_MSE_change=new_scores['latent_prediction_mse']-old_scores['latent_prediction_mse']))
    index = mid.gap.census(rows)
    def winner(d): return 'tie' if d == 0 else ARMS[1] if d < 0 else ARMS[0]
    pairs = []
    for i in range(4):
        j, c = index[i, ARMS[0]], index[i, ARMS[1]]
        old = c['original_open_loop']['predicted_Q']-j['original_open_loop']['predicted_Q']
        new = c['teacher_forced_suffix']['predicted_Q']-j['teacher_forced_suffix']['predicted_Q']
        actual = c['actual_terminal_Q']-j['actual_terminal_Q']
        pairs.append(dict(root_slot=i, composition_minus_joint=dict(original_predicted_Q=old, teacher_forced_Q=new, actual_terminal_Q=actual),
                          original_pair_Q_gap=old-actual, teacher_forced_pair_Q_gap=new-actual,
                          original_order_matches_teacher=winner(old)==winner(actual),
                          teacher_forced_order_matches_teacher=winner(new)==winner(actual),
                          order_changed_by_last_latent_replacement=winner(old)!=winner(new)))
    return rows, pairs


def consume(folder, expected, output):
    folder = Path(folder).absolute()
    require(sha(mid.regular(folder/'report.json')) == expected, 'external run report SHA')
    r, b = read(folder/'report.json'), read(folder/'execution_binding.json')
    p = verify(b['prepared'], b['prepare_sha256'])
    require(r['prepare_sha256'] == b['prepare_sha256'] and r['source_midpoint_report_sha256'] == b['source_midpoint_report_sha256'] == INPUTS['run'][1],
            'original midpoint/prepare/run binding')
    predicted = validate_run(folder, r, p)
    rows, pairs = comparisons(p, predicted)
    result = dict(schema=NAME+'.consume', status='all8_teacher_suffix_saved_verified_not_admitted', report_sha256=expected,
                  prepare_sha256=b['prepare_sha256'], source_midpoint_report_sha256=INPUTS['run'][1], root_denominator=4, arm_denominator=8,
                  rows=rows, pairs=pairs, executed_run_counts=r['counts'], inherited_resources=p['inherited_resources'],
                  inherited_midpoint_encoding_counts=p['inherited_midpoint_encoding_counts'], inherited_terminal_encoder_samples=8,
                  new_neural_calls=0, new_physics_steps=0, CUDA_ns=None, FLOPs=None, equal_full_resource_cost=False,
                  P1_admission=False, P2_admission=False, scientific_admission=False,
                  limitations=['Only last mid latent replaced; older history remains the original paid history',
                               'Teacher Q rank agreement tests sufficiency for this model intervention, not physical causal identification',
                               'No new actions, outcomes, roots, physics or native success improvement; full4 known calibration only'])
    write(output, result)
    return dict(output=str(Path(output).absolute()), sha256=sha(output), status=result['status'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('prepare','verify','run','consume'))
    p.add_argument('--prepared', type=Path); p.add_argument('--prepare-sha'); p.add_argument('--output', type=Path)
    p.add_argument('--run-dir', type=Path); p.add_argument('--report-sha'); p.add_argument('--authorize-host-cuda', action='store_true')
    a = p.parse_args()
    if a.command == 'prepare': result = prepare(a.output)
    elif a.command == 'verify': result = dict(status='verified_not_executed', jobs=len(verify(a.prepared,a.prepare_sha)['jobs']))
    elif a.command == 'run':
        require(a.authorize_host_cuda, 'explicit MAIN HOST CUDA authorization required')
        result = execute_cuda(a.prepared,a.prepare_sha,a.output)
    else: result = consume(a.run_dir,a.report_sha,a.output)
    print(mid.json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__': main()
