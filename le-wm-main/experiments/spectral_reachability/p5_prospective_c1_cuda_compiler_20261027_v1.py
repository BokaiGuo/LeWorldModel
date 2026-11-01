"""Prospective C1 compiler: prepare only by default; CUDA is an explicit main action.

No physics entry point. Numeric inputs are exclusively bridge.compiler_bundle.
Never invoke archived-data v8.prepare/verify_manifest or factual fidelity loaders.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p5_prospective_control_bridge_20261027_v1 as bridge
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG = 'p5_prospective_c1_cuda_compiler_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
V8 = ROOT / 'results/p5_jepa_native_action_response_20261027_v8/manifest_frozen.json'
STATS = ROOT / 'external/jepa-wms/app/plan_common/datasets/pusht_dset.py'
DIM = 256 * 384
ROLES = ('response_columns', 'goal_features', 'nominal_successor', 'ridge_successor',
         'selected_replay', 'gram', 'rhs', 'ridge_delta')
require = bridge.require
digest = bridge.digest


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return bridge.saved.strict_json(Path(path).read_bytes())


def source_stats(path):
    """Extract source literals without importing torch/native dataset code."""
    names = {'ACTION_MEAN', 'ACTION_STD', 'PROPRIO_MEAN', 'PROPRIO_STD'}
    result = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                require(isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                        and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'torch'
                        and node.value.func.attr == 'tensor' and len(node.value.args) == 1,
                        'source stats must be literal torch.tensor')
                require(name not in result, 'duplicate source stats')
                result[name] = ast.literal_eval(node.value.args[0])
    require(set(result) == names, 'all four source stats required')
    for name, values in result.items():
        n = 2 if name.startswith('ACTION') else 4
        a = np.asarray(values, np.float32)
        require(a.shape == (n,) and np.isfinite(a).all(), 'source stats shape/finite')
        if name.endswith('STD'):
            require(np.all(a > 0), 'positive source std required')
    return result


def normalize_actions(u, stats):
    u = np.asarray(u, np.float64)
    require(u.shape[-1:] == (2,) and np.isfinite(u).all(), 'finite native xy actions required')
    # Exactly the native dataset projection, NOT u/100.
    projected = (100. * u).astype(np.float32) / np.float32(100.)
    return ((projected - np.asarray(stats['ACTION_MEAN'], np.float32)) /
            np.asarray(stats['ACTION_STD'], np.float32))


def native_ridge_packet(delta, stats):
    d = np.asarray(delta, np.float64)
    require(d.shape == (10,) and np.isfinite(d).all(), 'finite ridge delta[10] required')
    # Centered inverse: native nominal is EXACT zero, not roundoff in z0*std+mean.
    packet = d.reshape(5, 2) * np.asarray(stats['ACTION_STD'], np.float32).astype(np.float64)
    bridge.packet(packet)  # Entire compilation aborts; never clip/project/rescale.
    return packet


def choose(candidates):
    require([c['id'] for c in candidates] == ['nominal', 'ridge'], 'fixed candidates required')
    for c in candidates:
        bridge.packet(c['env_packet'])
    scores = np.asarray([c['independent_goal_mse'] for c in candidates], np.float64)
    require(np.isfinite(scores).all() and np.all(scores >= 0), 'finite nonnegative costs required')
    return candidates[int(np.argmin(scores))]


def policy():
    return dict(bridge_contract=bridge.contract()['compiler'], seed=0,
        nominal='five exact native [0,0], normalized with source float32 dataset projection',
        inverse='u64=delta64*float64(source_std32); centered at exact native-zero; no clipping',
        nonlinear='re-normalize actual u64 through dataset float32 projection before ridge prediction/replay',
        derivative='sequential functional.jvp strict=True create_graph=False; double backward, not forward AD',
        residual='independent goal - nominal; no factual frame15', damping=.01,
        reductions='CUDA float64 J.T@J/D and J.T@residual/D, solve10x10; D=98304',
        backend='SDPA MATH for every encoding/prediction/JVP/replay; TF32 disabled',
        gates=dict(jvp_primal_max_abs=1e-6, selected_replay='torch.equal', free_scratch_bytes=4*1024**3),
        arrays='raw returned patches/proprio/all predictor tuple tensors; each JVP primal/column; ridge arithmetic',
        training=False, physical_calls=0, cpu_neural_fallback=False, P5_admission=False, C2_established=False)


def validate_bridge(prepared, expected_digest, ready=False):
    require(digest(prepared) == expected_digest, 'independently retained bridge digest differs')
    require(prepared['schema'] == bridge.TAG and prepared['contract'] == bridge.contract(), 'bridge contract differs')
    require(prepared['compiler_inputs_sha256'] == digest(prepared['compiler_inputs']), 'compiler input digest differs')
    seal = prepared['audit_source_seal']
    verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    runner = prepared['runner_sources'].get('compiler')
    own = dict(path=str(Path(__file__).resolve()), sha256=file_sha(__file__))
    if runner is not None:
        require(runner == own, 'bridge compiler source is not this exact runner')
    if ready:
        require(prepared['status'] == 'ready_for_main_CUDA_compile' and not prepared['blockers'], 'bridge is blocked')
        require(runner == own, 'reviewed compiler binding missing')
        inputs = prepared['compiler_inputs']
        require([f['frame'] for f in inputs['frames']] == [0, 5, 10], 'fixed context order required')
        paths = {f[k]['path']: f[k]['sha256'] for f in inputs['frames'] for k in ('rgb', 'proprio')}
        require(len(paths) == 6, 'six distinct context assets required')
        paths[inputs['goal_rgb']['path']] = inputs['goal_rgb']['sha256']
        require(len(paths) == 7 and paths == inputs['allowed_data_files'], 'exact seven-file data allowlist required')


def check_bindings(bindings):
    errors = []
    for path, expected in bindings.items():
        try:
            require(file_sha(path) == expected, 'SHA differs')
        except Exception as exc:
            errors.append(dict(path=path, error=type(exc).__name__ + ': ' + str(exc)))
    return errors


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def new_output(path):
    path = Path(path).resolve()
    require(path.parent == ROOT / 'results' and path.name.startswith(TAG + '_') and not path.exists(),
            'new exclusive direct results child required')
    path.mkdir(exist_ok=False)
    return path


def prepare(protocol_path, expected_digest, output):
    """No torch import/model construction; blocked goal is not invented."""
    protocol_path = Path(protocol_path).resolve()
    p = read_json(protocol_path)
    validate_bridge(p, expected_digest)
    frozen = read_json(V8)
    require(frozen['schema'] == 'p5_jepa_native_action_response_20261027_v8' and frozen['seals'], 'frozen v8 manifest required')
    # Carry ONLY loader metadata. Archived episode rows/targets/costs excluded.
    loader = {k: frozen[k] for k in ('seals', 'official_weight_mapping', 'expected_dynamics_schema', 'expected_model_receipts')}
    require(loader['expected_model_receipts'] == p['model_identity'], 'same frozen model identity required')
    stats = source_stats(STATS)
    own = [Path(__file__).resolve(), ROOT / 'tests' / ('test_' + TAG + '.py'),
           ROOT / 'innerdocs' / (TAG + '.md'), Path(bridge.__file__).resolve(),
           ROOT / 'experiments/spectral_reachability/d2p_prerun_source_seal.py', STATS]
    bindings = {**frozen['seals'], **{str(path): file_sha(path) for path in [*own, V8, protocol_path]}}
    bindings.update(p['compiler_inputs']['allowed_data_files'])
    bindings.update({v['path']: v['sha256'] for v in p['runner_sources'].values()})
    require(not check_bindings(bindings), 'frozen loader/source bytes drift')
    ready = p['status'] == 'ready_for_main_CUDA_compile' and not p['blockers']
    if ready:
        validate_bridge(p, expected_digest, ready=True)
        bridge.compiler_bundle(p, expected_digest)  # Validate only the seven allowed numerical assets.
    out = new_output(output)
    seal = create_seal(files=own, args_json=json.dumps(dict(protocol=str(protocol_path), protocol_sha256=expected_digest)),
                       output_dir=out / 'source_seal')
    manifest = dict(schema=TAG, status='prepared_for_main_CUDA' if ready else 'blocked_pre_execution_inputs',
        blockers=p['blockers'], bridge_path=str(protocol_path), bridge_sha256=expected_digest,
        loader_manifest=loader, stats=stats, bindings=bindings, source_seal=seal, policy=policy(),
        new_neural_calls=0, new_physics_steps=0, P5_admission=False)
    require(not check_bindings(bindings), 'prepare source drift')
    bridge.write(out / 'manifest_frozen.json', manifest)
    return manifest


class Ledger:
    """Fsync intent before dispatch; immutable arrays and hash-chained returns.

    Constant-array tests use this same interface, without torch/neural work.
    A kill/OOM leaves a journal prefix, never a falsely successful report.
    """
    def __init__(self, out, receipt):
        self.out = Path(out)
        self.result = receipt
        self.sequence = 0
        self.previous = None
        self.phase = 'preflight'
        keys = bridge.contract()['compiler']['expected_completed']
        receipt.update(attempted=dict.fromkeys(keys, 0), completed=dict.fromkeys(keys, 0),
                       artifact_bindings={}, artifact_roles={}, arrays={}, candidates=[], finalization_errors=[])

    def event(self, kind, **values):
        payload = dict(sequence=self.sequence, previous_sha256=self.previous,
            event=dict(kind=kind, phase=self.phase, attempted=dict(self.result['attempted']),
                       completed=dict(self.result['completed']), **values))
        h = digest(payload)
        with (self.out / 'journal.jsonl').open('ab') as stream:
            stream.write((json.dumps({**payload, 'entry_sha256': h}, sort_keys=True, allow_nan=False) + '\n').encode())
            stream.flush(); os.fsync(stream.fileno())
        fsync_dir(self.out)
        self.sequence += 1
        self.previous = h

    def count(self, state, key, n=1):
        self.result[state][key] += n
        self.event(state, operation=key, units=n)

    def array(self, name, value, role=None):
        a = np.asarray(value)
        require(a.dtype.kind in 'fiub' and a.ndim <= 8, 'plain numeric arrays only')
        path = self.out / (name + '.npy')
        with path.open('xb') as stream:
            np.save(stream, a, allow_pickle=False)
            stream.flush(); os.fsync(stream.fileno())
        fsync_dir(self.out)
        h = file_sha(path)
        self.result['artifact_bindings'][str(path)] = h
        self.result['arrays'][name] = dict(path=str(path), sha256=h, shape=list(a.shape), dtype=str(a.dtype))
        if role:
            require(role in ROLES and role not in self.result['artifact_roles'], 'unique fixed artifact role required')
            self.result['artifact_roles'][role] = str(path)
        self.event('array_durable', name=name, record=self.result['arrays'][name], role=role)

    def candidate(self, name, packet, cost):
        record = dict(id=name, env_packet=bridge.packet(packet), independent_goal_mse=float(cost))
        require(np.isfinite(cost) and cost >= 0, 'finite candidate cost required')
        self.result['candidates'].append(record)
        self.event('candidate_cost', candidate=record)


@contextmanager
def grad_counts(torch, ledger):
    original_grad, original_backward = torch.autograd.grad, torch.autograd.backward
    def grad(*args, **kwargs):
        ledger.count('attempted', 'autograd_grad')
        value = original_grad(*args, **kwargs)
        ledger.count('completed', 'autograd_grad')
        return value
    def backward(*args, **kwargs):
        ledger.event('unexpected_backward_api')
        raise RuntimeError('backward API is outside the frozen JVP contract')
    torch.autograd.grad, torch.autograd.backward = grad, backward
    try:
        yield
    finally:
        torch.autograd.grad, torch.autograd.backward = original_grad, original_backward


def guarded_final(result, label, function):
    try:
        function()
    except Exception as exc:
        result['finalization_errors'].append(dict(check=label, error=type(exc).__name__ + ': ' + str(exc)))


def cuda_core(manifest, prepared, ledger):
    """Main-only backend. No CPU model construction, physics or target fitting."""
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as v8
    r = ledger.result
    models, handles = None, []
    v4 = None
    try:
        require(not os.environ.get('PYTHONPATH'), 'fresh neural process without physics PYTHONPATH required')
        v8.require_cuda()
        bundle = bridge.compiler_bundle(prepared, manifest['bridge_sha256'])
        models, strict, sources, before = v8.load_cuda(ROOT, manifest['loader_manifest'])
        r.update(model_before=before, strict_loads=strict, imported_sources=sources)
        require(before == prepared['model_identity'], 'frozen identity differs')
        v7 = v8.helper(ROOT)
        _, v5, _ = v7.dependencies(ROOT)
        v4 = v5.import_v4(ROOT)
        free, total = torch.cuda.mem_get_info()
        r['memory_after_load'] = dict(free=free, total=total)
        require(free >= 4*1024**3, 'frozen 4GiB free scratch policy failed')

        def save_tensors(prefix, value):
            if isinstance(value, torch.Tensor):
                ledger.array(prefix, value.detach().cpu().numpy())
            elif isinstance(value, dict):
                for k, item in value.items():
                    save_tensors(prefix + '_' + k, item)
            elif isinstance(value, (tuple, list)):
                for i, item in enumerate(value):
                    save_tensors(prefix + '_' + str(i), item)

        def pre(key):
            def hook(module, args):
                ledger.count('attempted', key)
                if key == 'official_encoder':
                    ledger.count('attempted', 'official_rgb_frames', int(args[0].shape[0]))
            return hook
        def post(key):
            def hook(module, args, output):
                ledger.count('completed', key)
                if key == 'official_encoder':
                    ledger.count('completed', 'official_rgb_frames', int(args[0].shape[0]))
                save_tensors(key + '_' + str(r['completed'][key]), output)
            return hook
        for key, model in models.items():
            handles.append(model.register_forward_pre_hook(pre(key)))
            handles.append(model.register_forward_hook(post(key)))
        from app.plan_common.datasets.transforms import make_transforms
        transform = make_transforms(img_size=224, random_horizontal_flip=False,
            random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
            reprob=0., auto_augment=False, motion_shift=False)
        stats = manifest['stats']
        prop_np = ((bundle['raw_proprio_context'].astype(np.float32) - np.asarray(stats['PROPRIO_MEAN'], np.float32)) /
                   np.asarray(stats['PROPRIO_STD'], np.float32))
        history_np = normalize_actions(bundle['history_env_packets'], stats).reshape(1, 2, 10)
        zero_u = np.zeros((5, 2), np.float64)
        nominal_np = normalize_actions(zero_u, stats).reshape(10)
        r['action_coordinates'] = dict(
            action_mean_float32=np.asarray(stats['ACTION_MEAN'], np.float32).tolist(),
            action_std_float32=np.asarray(stats['ACTION_STD'], np.float32).tolist(),
            nominal_native_u64=zero_u.tolist(), nominal_normalized_f32=nominal_np.reshape(5, 2).tolist(),
            history_native_u64=bundle['history_env_packets'].tolist(),
            history_normalized_f32=history_np.reshape(2, 5, 2).tolist(),
            dataset_projection='pixel32=(100*u64).astype(float32); u32=pixel32/float32(100); z32=(u32-mean32)/std32',
            inverse='native_u64=delta64.reshape(5,2)*float64(source_std32), centered at exact native-zero')
        ledger.event('action_coordinates', values=r['action_coordinates'])
        for name, a in dict(normalized_history=history_np, normalized_proprio=prop_np, normalized_nominal=nominal_np).items():
            ledger.array(name, a)
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            rgb = np.concatenate([bundle['rgb_context'], bundle['goal_rgb'][None]], axis=0)
            pixels = torch.from_numpy(rgb).permute(0, 3, 1, 2).to('cuda', torch.float32) / 255.
            x = transform(pixels.clone())
            ref = (pixels - torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]) / torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
            require((x-ref).abs().max().item() <= 1e-6, 'native preprocess gate failed')
            ledger.phase = 'encode_four_frames'
            patches = models['official_encoder'](x, is_training=True)['x_norm_patchtokens']
            require(patches.shape == (4, 256, 384) and torch.isfinite(patches).all().item(), 'patch shape/finite gate failed')
            ledger.array('goal_features', patches[3:4].cpu().numpy(), 'goal_features')
            visual = patches[:3].reshape(1, 3, 1, 16, 16, 384).detach()
            goal = patches[3:4].reshape(DIM).detach()
            ledger.phase = 'encode_three_proprio'
            features = models['proprio_encoder'](torch.from_numpy(prop_np).unsqueeze(0).to('cuda')).expand(1, 3, 256, 16).detach()
            history = torch.from_numpy(history_np).to('cuda')
            z0 = torch.from_numpy(nominal_np).to('cuda')

        def predict_packet(z):
            acts = torch.cat((history, z.reshape(1, 1, 10)), dim=1)
            result = models['predictor'](visual, acts, features)
            require(result[0].shape == (1, 3, 256, 384) and result[2].shape == (1, 3, 256, 16), 'native output shape rejected')
            y = result[0][:, -1].reshape(DIM)
            require(torch.isfinite(y).all().item(), 'nonfinite successor')
            return y

        def successor(delta):
            return predict_packet(z0 + delta)

        with sdpa_kernel(SDPBackend.MATH):
            zero = torch.zeros(10, device='cuda', dtype=torch.float32)
            ledger.phase = 'nominal'
            with torch.no_grad():
                nominal = successor(zero)
            ledger.array('nominal_successor', nominal.reshape(1, 256, 384).cpu().numpy(), 'nominal_successor')
            ledger.candidate('nominal', zero_u, (goal.double()-nominal.double()).square().mean().item())
            columns = []
            with grad_counts(torch, ledger):
                for i in range(10):
                    ledger.phase = 'jvp_' + str(i)
                    basis = torch.zeros_like(zero); basis[i] = 1
                    ledger.count('attempted', 'jvp_columns')
                    with torch.enable_grad():
                        primal, column = torch.autograd.functional.jvp(successor, zero, basis, strict=True, create_graph=False)
                    ledger.count('completed', 'jvp_columns')
                    ledger.array('jvp_primal_' + str(i), primal.cpu().numpy())
                    ledger.array('jvp_column_' + str(i), column.cpu().numpy())
                    require(column.shape == (DIM,) and torch.isfinite(column).all().item(), 'JVP shape/finite rejected')
                    require((primal-nominal).abs().max().item() <= 1e-6, 'JVP primal vs nominal gate failed')
                    columns.append(column.detach())
            response = torch.stack(columns, dim=1)
            ledger.array('response_columns', response.cpu().numpy(), 'response_columns')
            ledger.phase = 'ridge_solve'
            with torch.no_grad():
                j = response.double()
                residual = goal.double() - nominal.double()
                ledger.array('goal_minus_nominal', residual.cpu().numpy())
                gram = j.T @ j / DIM
                rhs = j.T @ residual / DIM
                ledger.array('gram', gram.cpu().numpy(), 'gram')
                ledger.array('rhs', rhs.cpu().numpy(), 'rhs')
                delta = torch.linalg.solve(gram + .01*torch.eye(10, device='cuda', dtype=torch.float64), rhs)
                ledger.array('ridge_delta', delta.cpu().numpy(), 'ridge_delta')
                # Preserve even an inadmissible proposal before support verification.
                proposal = delta.reshape(5, 2).cpu().numpy() * np.asarray(stats['ACTION_STD'], np.float32).astype(np.float64)
                ledger.array('ridge_native_proposal', proposal)
                r['action_coordinates']['ridge_native_u64'] = proposal.tolist() if np.isfinite(proposal).all() else None
                ledger.event('ridge_proposal', values=r['action_coordinates'])
                u = native_ridge_packet(delta.cpu().numpy(), stats)
                z_np = normalize_actions(u, stats).reshape(10)
                ledger.array('normalized_ridge', z_np)
                r['ridge_roundtrip_max_abs'] = float(np.max(np.abs(z_np.astype(np.float64) - (nominal_np.astype(np.float64) + delta.cpu().numpy()))))
                r['action_coordinates'].update(ridge_normalized_f32=z_np.reshape(5, 2).tolist(),
                    ridge_ideal_normalized_f64=(nominal_np.astype(np.float64)+delta.cpu().numpy()).reshape(5, 2).tolist(),
                    ridge_roundtrip_max_abs=r['ridge_roundtrip_max_abs'])
                ledger.event('ridge_roundtrip', values=r['action_coordinates'])
                z = torch.from_numpy(z_np).to('cuda')
                ledger.phase = 'ridge_nonlinear'
                ridge = predict_packet(z)
                ledger.array('ridge_successor', ridge.reshape(1, 256, 384).cpu().numpy(), 'ridge_successor')
                ledger.candidate('ridge', u, (goal.double()-ridge.double()).square().mean().item())
                selected = choose(r['candidates'])
                r.update(selected_id=selected['id'], selected_env_packet=selected['env_packet'])
                ledger.event('selected', candidate=selected)
                selected_z = torch.from_numpy(normalize_actions(selected['env_packet'], stats).reshape(10)).to('cuda')
                ledger.phase = 'selected_replay'
                replay = predict_packet(selected_z)
                ledger.array('selected_replay', replay.reshape(1, 256, 384).cpu().numpy(), 'selected_replay')
                r['selected_replay_exact'] = bool(torch.equal(replay, nominal if selected['id'] == 'nominal' else ridge))
                require(r['selected_replay_exact'], 'MATH search/selected replay not bit-exact')
        r['status'] = 'core_completed_pending_final_checks'
    except Exception as exc:
        r.update(status='compiler_failed', error=type(exc).__name__ + ': ' + str(exc), failed_phase=ledger.phase)
    finally:
        for i, handle in enumerate(handles):
            guarded_final(r, 'hook_remove_' + str(i), handle.remove)
        if models is not None and v4 is not None:
            def identity():
                r['model_after'] = {key: v4.model_receipt(model) for key, model in models.items()}
                require(r['model_after'] == r['model_before'] == prepared['model_identity'], 'model changed')
            guarded_final(r, 'frozen_model_after', identity)
        guarded_final(r, 'cuda_synchronize', torch.cuda.synchronize)


def run(manifest_path, expected_manifest_sha256, output):
    """Explicit main action only. Failures have nonzero exit and durable receipts."""
    require(file_sha(manifest_path) == expected_manifest_sha256, 'retained manifest FILE SHA differs')
    m = read_json(manifest_path)
    require(m['schema'] == TAG and m['status'] == 'prepared_for_main_CUDA' and not m['blockers'], 'compiler prepare blocked')
    require(m['policy'] == policy(), 'frozen compiler policy drift')
    verify_seal(output_dir=m['source_seal']['output_dir'], expected_manifest_sha256=m['source_seal']['manifest_sha256'])
    require(not check_bindings(m['bindings']), 'frozen source/loader/input drift')
    p = read_json(m['bridge_path'])
    validate_bridge(p, m['bridge_sha256'], ready=True)
    out = new_output(output)
    r = dict(schema=bridge.TAG+'.compiler', runner_schema=TAG, status='compiler_failed', error=None,
        manifest_file_sha256=expected_manifest_sha256, prepared_sha256=m['bridge_sha256'],
        compiler_inputs_sha256=p['compiler_inputs_sha256'], compiler_data_bindings=p['compiler_inputs']['allowed_data_files'],
        goal_rgb_sha256=p['goal']['rgb_sha256'], compiler_source=p['runner_sources']['compiler'],
        backend='host_cuda_only', used_observed_frame15=False, physical_calls=0, training=False,
        P5_admission=False, C2_established=False, partial_work_unknown=True, selected_replay_exact=False)
    ledger = Ledger(out, r)
    ledger.event('begin', policy=m['policy'])
    try:
        cuda_core(m, p, ledger)
    except Exception as exc:
        r.update(status='compiler_failed', error=type(exc).__name__ + ': ' + str(exc), failed_phase=ledger.phase)
    # Each check is independent; never replace an original runtime error.
    def sources():
        r['source_check_errors'] = check_bindings(m['bindings'])
        require(not r['source_check_errors'], 'final source integrity failed')
        verify_seal(output_dir=p['audit_source_seal']['output_dir'],
                    expected_manifest_sha256=p['audit_source_seal']['manifest_sha256'])
    guarded_final(r, 'source_after', sources)
    guarded_final(r, 'arrays_after', lambda: require(not check_bindings(r['artifact_bindings']), 'artifact integrity failed'))
    expected = bridge.contract()['compiler']['expected_completed']
    r['counts_match'] = r['attempted'] == r['completed'] == expected
    if r['status'] == 'core_completed_pending_final_checks' and not r['finalization_errors'] and r['counts_match']:
        r.update(status='selected_packet_replayed', partial_work_unknown=False)
        try:
            bridge.frozen_branches(p, r)
        except Exception as exc:
            r.update(status='compiler_failed', partial_work_unknown=True, error=type(exc).__name__ + ': ' + str(exc))
    else:
        r['status'] = 'compiler_failed'
        if r['error'] is None:
            r['error'] = 'final checks or exact attempted/completed budget failed'
    ledger.phase = 'terminal'
    ledger.event('terminal', status=r['status'], error=r['error'], finalization_errors=r['finalization_errors'])
    r['journal_sha256'] = file_sha(out / 'journal.jsonl')
    bridge.write(out / 'compiler_receipt.json', r)
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--bridge-protocol', type=Path, required=True)
    a.add_argument('--bridge-sha256', required=True, help='canonical bridge.digest retained independently')
    a.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('run-cuda', help='MAIN ONLY; never runs physics')
    b.add_argument('--manifest', type=Path, required=True)
    b.add_argument('--manifest-sha256', required=True, help='FILE SHA retained independently')
    b.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        r = prepare(args.bridge_protocol, args.bridge_sha256, args.output)
    else:
        r = run(args.manifest, args.manifest_sha256, args.output)
    print(json.dumps(dict(status=r['status'], output=str(args.output), P5_admission=False)))
    if args.command == 'run-cuda' and r['status'] != 'selected_packet_replayed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
