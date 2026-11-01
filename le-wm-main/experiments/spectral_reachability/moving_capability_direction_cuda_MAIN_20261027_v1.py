"""Fixed40 RGB endpoints +80 teacher/open predictions; MAIN host CUDA only."""
from copy import deepcopy
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import os
import time
import numpy as np
from experiments.spectral_reachability import p1_moving_overlay_recompile_rgb_cuda_20261027_v1 as rgb
from experiments.spectral_reachability import p1_moving_overlay_recompile_cuda_20261027_v1 as planner

NAME = 'moving_capability_direction_cuda_MAIN_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
AUTH = 'MAIN_FIXED40_CAPABILITY_DIRECTION_CUDA_ONLY'
PROBE = ROOT/'results/moving_overlay_capability_probe_physical_20261027_v1_MAIN01'
PROBE_SHA = 'e075d3b75c7b5a16c47d8c76c846eaa2ac4d8ebd202abd18037a7242792b076a'
RGB = ROOT/'results/p1_moving_overlay_recompile_rgb_cuda_20261027_v1_MAIN01'
RGB_SHA = '18d2f192a75e397c01e6c7eb7f6422e9bb073af865107ef469debe5b8ae32d1e'
PLANNER = ROOT/'results/p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01'
PLANNER_SHA = '2b2e5064f74f8aca712ae0d0415fcbfb64a2c8e6325bdffe24ae4fdc3b373381'
require, sha = rgb.require, rgb.sha


def contract():
    return dict(roots=2, fresh_roots=0, anchors=8, branches=40, paths=['teacher', 'open'],
        order='anchor/probe(zero,+x,-x,+y,-y)/path(teacher,open)',
        caps=dict(encoder=40, projector=40, action_encoder=80, predictor=80, pred_proj=80),
        predict_API=80, batch_samples=80, gradient_calls=0, physics_calls=0,
        CUDA_only=True, TF32=False, training=False, Qfit=False, P1_admission=False,
        wall_seconds=900, output_cap_bytes=64*1024**2, disk_floor_bytes=200*1024**2,
        final_reserve_bytes=4*1024**2, keep_all_zero_negative=True,
        inherited_encoding_cost='28 saved image encodings; compiler16 saved encodings, paid historically, not zero',
        interpretation='fixed-scale representation/ranking diagnostic, not fresh P1 or native state decoding')


def key(r):
    return r['root_slot'], r['condition'], r['arm']


def tokens(raw, task):
    a = np.asarray(raw, dtype=np.float32)
    require(a.ndim == 2 and a.shape[1] == 2 and len(a) % 5 == 0 and np.isfinite(a).all() and np.max(np.abs(a)) <= 1, 'bounded raw packet')
    mean = np.tile(np.asarray(task['action_mean'], dtype=np.float32), 5)
    std = np.tile(np.asarray(task['action_std'], dtype=np.float32), 5)
    require(mean.shape == std.shape == (10,) and np.all(std > 0), 'original action normalizer')
    return (a.reshape(-1, 10)-mean)/std


def contexts(task, commands, open_context, actual_prefix):
    root = task['root_context']
    history = np.asarray(root['latent_history'], dtype=np.float32)
    require(history.shape == (3, 192) and np.asarray(actual_prefix).shape == (192,), 'native history shapes')
    outgoing = tokens(np.asarray(commands, dtype=np.float32)[:5], task)[0]
    past = np.asarray([root['action_history'][-1], outgoing], dtype=np.float32)
    require(np.array_equal(np.asarray(open_context['action_history'], dtype=np.float32), past), 'prefix action history chronology')
    open_history = np.asarray(open_context['latent_history'], dtype=np.float32)
    require(open_history.shape == (3, 192) and np.array_equal(open_history[:2], history[1:]), 'same true past frames')
    teacher = np.concatenate([history[1:], np.asarray(actual_prefix, dtype=np.float32)[None]], axis=0)
    return dict(teacher=teacher.tolist(), open=open_history.tolist(), past=past.tolist())


def build():
    bindings = {}
    def bind(path, expected=None):
        path = str(Path(path).resolve()); h = sha(path)
        require(expected is None or h == expected, 'input SHA mismatch: '+path)
        require(path not in bindings or bindings[path] == h, 'binding conflict')
        bindings[path] = h
        return path
    def report(directory, expected):
        return rgb.read(bind(directory/'report.json', expected), 8*1024**2)
    physical, visual, compiled = report(PROBE, PROBE_SHA), report(RGB, RGB_SHA), report(PLANNER, PLANNER_SHA)
    require(physical['status'] == 'capability_probes_recorded' and not physical['errors'] and physical['inputs_final_verified'] is True and len(physical['rows']) == 40, 'all40 real probes')
    # Genuine saved consumers validate original visual and plan module bridges.
    real_rows = rgb.consume(str(RGB), RGB_SHA)
    plans = planner.consume(str(PLANNER), PLANNER_SHA)
    binding = rgb.read(PROBE/'execution_binding.json')
    # Probe producer binding contains prepare SHA; locate by the known fixed path.
    prep_path = ROOT/'results/moving_overlay_capability_probe_physical_20261027_v1_prepare_MAIN01.json'
    prep = rgb.read(bind(prep_path, binding['prepare_sha256']), 8*1024**2)
    require(prep['physical_report_sha256'] == physical['physical_report_sha256'], 'probe source join')
    for path, h in prep['bindings'].items(): bind(path, h)
    bind(PROBE/'execution_binding.json'); bind(PROBE/'journal.jsonl', physical['journal_sha256'])
    for directory, r in ((RGB, visual), (PLANNER, compiled)):
        for name, h in r['artifact_sha256'].items(): bind(rgb.flat(directory, name), h)
    # The probe journal has forty branches; old eight-branch RGB helper caps at16MiB.
    from experiments.spectral_reachability import moving_overlay_capability_probe_analysis_20261027_v1 as analysis
    events = analysis.read_journal(PROBE/'journal.jsonl', physical['journal_chain_tail'])
    rows = []
    for i, (plan, real, pr) in enumerate(zip(plans, real_rows, compiled['rows'], strict=True)):
        require(key(plan) == key(real) == key(pr) and plan['selection_digest'] == real['selection_digest'], 'anchor identity')
        task = json.loads(gzip.decompress((PLANNER/pr['task_file']).read_bytes()))['task']
        result = json.loads(gzip.decompress((PLANNER/pr['result_file']).read_bytes()))
        if plan['arm'] == 'line_search_GN':
            open_context = result['final_prediction']['contexts'][0]
        else:
            w = result['winner']; root = task['root_context']
            open_context = deepcopy(root)
            open_context['latent_history'] = root['latent_history'][1:]+[w['predicted_prefix']]
            open_context['action_history'] = [root['action_history'][-1], tokens(np.asarray(plan['commands'])[:5], task)[0].tolist()]
        cp = real['checkpoints'][1]
        actual = rgb.load_array(cp['latent'], lambda name: (RGB/name).read_bytes())
        c = contexts(task, plan['commands'], open_context, actual)
        vr = visual['rows'][i]
        goal = rgb.load_array(visual['images'][vr['goal_image']]['latent'], lambda name: (RGB/name).read_bytes())
        for j, probe in enumerate(prep['contract']['probes']):
            b = physical['rows'][5*i+j]
            require(key(b) == key(plan) and b['probe'] == probe['name'] and b['start_verified'] is True and b['status'] == 'probe_completed', 'fixed branch identity')
            require(b['start_state'] == prep['anchors'][i]['checkpoint']['state'] and b['selection_digest'] == plan['selection_digest'], 'exact anchor')
            endpoint = next(x for x in b['trajectory'] if x['phase'] == 'probe' and x['step'] == 5)
            selected = [e for e in events if e['kind'] == 'fresh_observation' and e.get('branch') == 5*i+j and e['receipt']['phase'] == 'probe' and e['receipt']['step'] == 5 and e['receipt']['overlay_index'] == (1 if plan['condition'] == 'on' else 0)]
            require(len(selected) == 1 and selected[0]['receipt'] == endpoint and selected[0]['pre'] == selected[0]['post'], 'real condition endpoint receipt')
            item = dict(path=bind(PROBE/endpoint['png'], endpoint['file_sha256']), file_sha256=endpoint['file_sha256'], pixel_sha256=endpoint['pixel_sha256'], kind='checkpoint')
            rgb.load_image(item)
            rows.append(dict(index=5*i+j, anchor=i, root_slot=plan['root_slot'], condition=plan['condition'], arm=plan['arm'],
                probe=probe['name'], selection_digest=plan['selection_digest'], task_id=plan['task_id'], image=item,
                contexts=c, tokens=np.concatenate([np.asarray(c['past'], np.float32), tokens(probe['commands'], task)]).tolist(),
                raw_probe=probe['commands'], goal=goal.tolist(), native_endpoint=endpoint['state']))
    from experiments.spectral_reachability import p1_constructive_blind_encode_cuda_20261027_v1 as native
    for path in native.source_files(): bind(path)
    for path in (Path(__file__), Path(analysis.__file__), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md'),
                 ROOT/'experiments/checkpoint_compat.py', ROOT/'experiments/gpu_preflight.py',
                 ROOT.parent/'docs/D2P_FIXED_CAPABILITY_DIRECTION_PROTOCOL_MAIN_20261027.md', Path(rgb.__file__), Path(planner.__file__)):
        bind(path)
    for path, h in rgb.MODELS.items(): bind(path, h)
    return dict(schema=NAME+'.prepare', contract=contract(), rows=rows, bindings=bindings,
                policy=str(rgb.POLICY), model_hash=rgb.MODEL_TENSOR_SHA)


def prepare(output):
    p = build()
    data = rgb.canonical(p)
    require(len(p['rows']) == 40 and len(data) < 8*1024**2, 'fixed bounded preparation')
    with Path(output).open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    return dict(prepared=str(Path(output).absolute()), sha256=sha(output))


def verify(path, expected):
    require(sha(path) == expected, 'external prepare seal')
    p = rgb.read(path, 8*1024**2)
    require(p['schema'] == NAME+'.prepare' and p['contract'] == contract() and len(p['rows']) == 40, 'fixed preparation')
    for path, h in p['bindings'].items(): require(sha(path) == h, 'saved dependency drift')
    return p


class Trace:
    def __init__(self, model, torch, store, check):
        self.model, self.torch, self.store, self.check = model, torch, store, check
        self.entries = []; self.pending = {}; self.handles = []; self.scope = None
    def save(self, tensor, name):
        require(tensor.device.type == 'cuda' and tensor.dtype == self.torch.float32, 'native CUDA float32')
        return self.store.array(name, tensor.detach().cpu().contiguous().numpy().copy())
    def __enter__(self):
        def pre(kind):
            def before(module, args):
                self.check(); require(self.scope is not None and not self.torch.is_grad_enabled(), 'scoped inference')
                require((kind in ('encoder', 'projector')) == (self.scope['phase'] == 'encode'), 'phase separation')
                require(sum(e['resource'] == kind for e in self.entries) < contract()['caps'][kind], 'predispatch native module cap')
                require(module not in self.pending, 'nonrecursive module')
                e = dict(index=len(self.entries), resource=kind, scope=deepcopy(self.scope), completed=False)
                self.entries.append(e); self.pending[module] = e
                self.store.event(dict(kind='module_attempt', entry=deepcopy(e)))
                e['inputs'] = [self.save(x, f'm{e["index"]:04}_in{j}') for j, x in enumerate(args)]
            return before
        def after(module, args, result):
            e = self.pending.pop(module)
            if hasattr(result, 'last_hidden_state'): result = result.last_hidden_state
            e['output'] = self.save(result, f'm{e["index"]:04}_out')
            self.torch.cuda.synchronize(); self.check(); e['completed'] = True
            self.store.event(dict(kind='module_return', entry=deepcopy(e)))
        try:
            for kind in contract()['caps']:
                module = getattr(self.model, kind)
                self.handles.extend([module.register_forward_pre_hook(pre(kind)), module.register_forward_hook(after)])
        except BaseException:
            self.__exit__(None, None, None); raise
        return self
    def __exit__(self, typ, exc, tb):
        for handle in reversed(self.handles): handle.remove()


def run(prepared, expected, output, authorization):
    require(authorization == AUTH, 'MAIN explicit authorization')
    started = time.monotonic()
    def check(): require(time.monotonic()-started < 900, 'whole job wall cap900s')
    p = verify(prepared, expected); check()
    store = planner.Store(output, check)
    errors = []; rows = [dict(index=i, status='not_started', predictions={}) for i in range(40)]
    model = None; trace = None; before = after = None; host = None
    try:
        store.put('binding.json', dict(prepared=str(Path(prepared).absolute()), prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from torch.nn.attention import sdpa_kernel, SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import p1_constructive_blind_encode_cuda_20261027_v1 as native
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host = require_cuda_allocation('cuda:0'); device = torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        model = load_compatible_pretrained(p['policy'], device=device).eval().requires_grad_(False)
        native.frozen(model); before = native.tensor_hash(model); require(before == p['model_hash'], 'original frozen model')
        transform = p0.image_transform(); trace = Trace(model, torch, store, check)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH), trace:
            for source in p['rows']:
                i = source['index']; row = rows[i]; row['status'] = 'attempted'
                pixels, original = rgb.load_image(source['image'])
                store._write(f'i{i:02}_original.png', original)
                row['pixels'] = store.array(f'i{i:02}_pixels', pixels)
                x = p0._image_tensor(pixels, transform=transform, device=device)
                trace.scope = dict(phase='encode', index=i)
                row['preprocessed'] = trace.save(x, f'i{i:02}_preprocessed')
                z = model.encode({'pixels': x})['emb']
                require(tuple(z.shape) == (1, 1, 192), 'original projected native192')
                row['actual'] = trace.save(z[0, 0], f'i{i:02}_actual')
                for path in ('teacher', 'open'):
                    trace.scope = dict(phase='predict', index=i, path=path)
                    history = torch.tensor(source['contexts'][path], dtype=torch.float32, device=device)[None]
                    action = torch.tensor(source['tokens'], dtype=torch.float32, device=device)[None]
                    require(tuple(history.shape) == (1, 3, 192) and tuple(action.shape) == (1, 3, 10), 'between-state plus outgoing')
                    output_tensor = model.predict(history, model.action_encoder(action))
                    require(tuple(output_tensor.shape) == (1, 3, 192), 'full native prediction')
                    row['predictions'][path] = trace.save(output_tensor, f'i{i:02}_{path}_prediction')
                row['status'] = 'sealed'; store.event(dict(kind='row_sealed', row=deepcopy(row)))
        native.frozen(model); after = native.tensor_hash(model)
        require(before == after and not trace.pending, 'frozen completed model')
        verify(prepared, expected); check()
        require(all(sum(e['resource'] == k and e['completed'] for e in trace.entries) == v for k, v in contract()['caps'].items()), 'all native counts exact')
    except BaseException as e:
        errors.append(dict(type=type(e).__name__, message=str(e)[:1800]))
        if model is not None:
            try: after = native.tensor_hash(model)
            except BaseException as final: errors.append(dict(type=type(final).__name__, message=str(final)[:1000]))
    finally:
        try: store.close()
        except BaseException as e: errors.append(dict(type=type(e).__name__, message=str(e)[:1000]))
    good = not errors and len(rows) == 40 and all(r['status'] == 'sealed' for r in rows) and before == after == p['model_hash']
    result = dict(schema=NAME+'.report', status='predictions_and_encodings_sealed' if good else 'partial_or_failed',
        rows=rows, errors=errors, contract=contract(), prepared=str(Path(prepared).absolute()), prepare_sha256=expected,
        model_hash_before=before, model_hash_after=after, host=host, ledger=[] if trace is None else trace.entries,
        wall_seconds=time.monotonic()-started, P1_admission=False,
        artifact_sha256={f.name: sha(f) for f in store.output.iterdir() if f.is_file()})
    # Honest partial seal is permitted after exhaustion, no further neural calls.
    store.check = lambda: None
    store.put('report.json', result, final=True)
    require(good, 'partial preserved, no retry '+str(errors))
    return dict(status=result['status'], sha256=sha(store.output/'report.json'), rows=40)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify', 'run'))
    for name in ('prepared', 'prepare-sha', 'output', 'authorization'): parser.add_argument('--'+name)
    a = parser.parse_args()
    if a.command == 'prepare': result = prepare(a.output)
    elif a.command == 'verify': result = dict(status='verified_no_execution', rows=len(verify(a.prepared, a.prepare_sha)['rows']))
    else: result = run(a.prepared, a.prepare_sha, a.output, a.authorization)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
