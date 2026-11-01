"""Main-only native normalized-coordinate FD consumer; prepare/tests are non-neural.

Reuse prospective compiler contexts, never archived target readers or frame15.
Two fixed epsilons, all ten columns, forty new predictions, zero new encodings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p5_prospective_c1_cuda_compiler_20261027_v1 as compiler
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG = 'p5_prospective_c1_native_fd_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
COMPILER_OUTPUT = ROOT/'results/p5_prospective_c1_cuda_compiler_20261027_v1_cuda_main_v1'
EPSILONS = (.01, .005)
DIM = 256*384
require, file_sha, read_json = compiler.require, compiler.file_sha, compiler.read_json
digest = compiler.digest


def policy():
    return dict(coordinates='native normalized outgoing packet, ten time-major XY entries',
        center='saved compiler normalized_nominal, native u exactly zero; NOT normalized zero',
        epsilons=list(EPSILONS), columns=list(range(10)), signs=[1, -1],
        order='epsilon then column then plus/minus; no replacement, tuning, clipping or renormalization',
        formula='(F(z0+eps*ei).float64-F(z0-eps*ei).float64)/(2*float64(eps))',
        perturbation='z0 float32 + signed eps float32 basis; retain effective rounded input and displacement',
        backend='same frozen native loader; eval/frozen CUDA float32 SDPA MATH, TF32 off, seed0',
        cached_context='compiler official_encoder_1_x_norm_patchtokens[:3] and proprio_encoder_1, same normalized_history',
        completed=dict(official_encoder=0, official_rgb_frames=0, proprio_encoder=0,
                       predictor=40, jvp_columns=0, autograd_grad=0),
        compare='saved JVP vs central FD per-column and full matrix; descriptive only, no acceptance threshold',
        new_nominal_predictions=0, selected_packet_changed=False, training=False,
        physical_calls=0, cpu_neural_fallback=False, P5_admission=False, C2_established=False)


def required_arrays():
    result = dict(official_encoder_1_x_norm_patchtokens=((4, 256, 384), 'float32'),
                  proprio_encoder_1=((1, 3, 1, 16), 'float32'),
                  normalized_history=((1, 2, 10), 'float32'),
                  normalized_proprio=((3, 4), 'float32'), normalized_nominal=((10,), 'float32'),
                  nominal_successor=((1, 256, 384), 'float32'), response_columns=((DIM, 10), 'float32'))
    for i in range(10):
        result['jvp_column_'+str(i)] = ((DIM,), 'float32')
        result['jvp_primal_'+str(i)] = ((DIM,), 'float32')
    return result


def load_array(record, shape, dtype):
    path = Path(record['path'])
    require(path.is_file() and path.stat().st_size <= 8*1024**2 and file_sha(path) == record['sha256'], 'bounded array SHA differs')
    a = np.load(path, allow_pickle=False)
    require(a.shape == tuple(shape) and a.dtype == np.dtype(dtype) and np.isfinite(a).all(), 'array shape/dtype/finite differs')
    require(record['shape'] == list(shape) and record['dtype'] == dtype, 'array metadata differs')
    return a


def new_output(path):
    path = Path(path).resolve()
    require(path.parent == ROOT/'results' and path.name.startswith(TAG+'_') and not path.exists(), 'exclusive new results child required')
    path.mkdir(exist_ok=False)
    return path


def verify_manifest(path, expected):
    require(file_sha(path) == expected, 'retained manifest FILE SHA differs')
    m = read_json(path)
    require(m['schema'] == TAG and m['policy'] == policy(), 'frozen FD policy differs')
    require(not compiler.check_bindings(m['bindings']), 'frozen source/loader/input/reference drift')
    seal = m['source_seal']
    verify_seal(output_dir=seal['output_dir'], expected_manifest_sha256=seal['manifest_sha256'])
    c = read_json(m['compiler_manifest_path'])
    require(c['policy'] == compiler.policy() and c['status'] == 'prepared_for_main_CUDA' and not c['blockers'], 'same ready compiler manifest required')
    compiler.validate_bridge(read_json(c['bridge_path']), c['bridge_sha256'], ready=True)
    verify_seal(output_dir=c['source_seal']['output_dir'], expected_manifest_sha256=c['source_seal']['manifest_sha256'])
    return m, c


def prepare(compiler_manifest, expected, output):
    require(file_sha(compiler_manifest) == expected, 'compiler manifest FILE SHA differs')
    c = read_json(compiler_manifest)
    require(c['schema'] == compiler.TAG and c['status'] == 'prepared_for_main_CUDA' and not c['blockers'], 'ready compiler source prepare required')
    require(c['policy'] == compiler.policy(), 'compiler policy drift')
    p = read_json(c['bridge_path'])
    compiler.validate_bridge(p, c['bridge_sha256'], ready=True)
    compiler.bridge.compiler_bundle(p, c['bridge_sha256'])  # Exactly seven allowed numerical assets.
    own = [Path(__file__).resolve(), ROOT/'tests'/('test_'+TAG+'.py'), ROOT/'innerdocs'/(TAG+'.md')]
    bindings = dict(c['bindings'])
    bindings.update({str(path.resolve()): file_sha(path) for path in [*own, Path(compiler_manifest)]})
    require(not compiler.check_bindings(bindings), 'frozen loader/source/input bytes drift')
    out = new_output(output)
    seal = create_seal(files=own, args_json=json.dumps(dict(compiler_manifest=str(Path(compiler_manifest).resolve()), policy=policy())), output_dir=out/'source_seal')
    m = dict(schema=TAG, status='source_prepared_awaiting_actual_JVP_reference',
        compiler_manifest_path=str(Path(compiler_manifest).resolve()), compiler_manifest_sha256=expected,
        expected_compiler_output=str(COMPILER_OUTPUT), bridge_sha256=c['bridge_sha256'],
        compiler_inputs_sha256=p['compiler_inputs_sha256'], source_seal=seal, bindings=bindings,
        policy=policy(), reference=None, new_neural_calls=0, new_physics_steps=0, P5_admission=False)
    compiler.bridge.write(out/'manifest_frozen.json', m)
    verify_manifest(out/'manifest_frozen.json', file_sha(out/'manifest_frozen.json'))
    return m


def reference_arrays(receipt, c, p, diagnostic=False):
    """No selection/outcome use; complete ten columns required even in diagnostic mode."""
    expected_status = 'compiler_failed' if diagnostic else 'selected_packet_replayed'
    require(receipt['status'] == expected_status and receipt['runner_schema'] == compiler.TAG, 'actual compiler status not authorized for this mode')
    require(receipt['manifest_file_sha256'] == file_sha(c['path'])
            and receipt['prepared_sha256'] == c['manifest']['bridge_sha256']
            and receipt['compiler_inputs_sha256'] == p['compiler_inputs_sha256']
            and receipt['compiler_data_bindings'] == p['compiler_inputs']['allowed_data_files'], 'same compiler/bridge input identity required')
    require(receipt['compiler_source'] == p['runner_sources']['compiler']
            and receipt['backend'] == 'host_cuda_only' and receipt['used_observed_frame15'] is False
            and receipt['physical_calls'] == 0 and receipt['training'] is False, 'prospective source/backend/input scope')
    require(receipt['model_before'] == receipt['model_after'] == p['model_identity'], 'same unchanged native model identity required')
    if diagnostic:
        require(receipt['error'] is not None, 'failed compiler must retain original failure')
    else:
        require(receipt['partial_work_unknown'] is False and receipt['counts_match'] is True
                and not receipt['finalization_errors'] and receipt['error'] is None
                and receipt['attempted'] == receipt['completed'] == compiler.bridge.contract()['compiler']['expected_completed'], 'complete actual compiler counters/integrity required')
    arrays = {}
    for name, (shape, dtype) in required_arrays().items():
        require(name in receipt['arrays'], 'actual compiler array missing '+name)
        rec = receipt['arrays'][name]
        require(receipt['artifact_bindings'].get(rec['path']) == rec['sha256'], 'array artifact binding differs')
        arrays[name] = load_array(rec, shape, dtype)
    require(np.array_equal(arrays['response_columns'], np.stack([arrays['jvp_column_'+str(i)] for i in range(10)], axis=1)), 'ten saved JVP columns vs matrix differ')
    bundle = compiler.bridge.compiler_bundle(p, c['manifest']['bridge_sha256'])
    stats = c['manifest']['stats']
    require(np.array_equal(arrays['normalized_nominal'], compiler.normalize_actions(np.zeros((5, 2)), stats).reshape(10)), 'exact native-zero nominal required')
    require(np.array_equal(arrays['normalized_history'], compiler.normalize_actions(bundle['history_env_packets'], stats).reshape(1, 2, 10)), 'same two history packets required')
    prop = (bundle['raw_proprio_context'].astype(np.float32)-np.asarray(stats['PROPRIO_MEAN'], np.float32))/np.asarray(stats['PROPRIO_STD'], np.float32)
    require(np.array_equal(arrays['normalized_proprio'], prop), 'same normalized proprio inputs required')
    return arrays


def bind_reference(manifest_path, expected, receipt_sha, output, diagnostic=False):
    m, c = verify_manifest(manifest_path, expected)
    require(m['reference'] is None, 'bind only original source prepare')
    receipt_path = Path(m['expected_compiler_output'])/'compiler_receipt.json'
    require(file_sha(receipt_path) == receipt_sha, 'explicit actual compiler receipt FILE SHA differs')
    r = read_json(receipt_path)
    p = read_json(c['bridge_path'])
    reference_arrays(r, dict(path=m['compiler_manifest_path'], manifest=c), p, diagnostic)
    require(file_sha(Path(m['expected_compiler_output'])/'journal.jsonl') == r['journal_sha256'], 'actual compiler journal SHA differs')
    bound = dict(m)
    bound['bindings'] = dict(m['bindings'])
    bound['bindings'].update({str(Path(manifest_path).resolve()): expected, str(receipt_path): receipt_sha,
        str(Path(m['expected_compiler_output'])/'journal.jsonl'): r['journal_sha256']})
    selected = {name: r['arrays'][name] for name in required_arrays()}
    bound['bindings'].update({v['path']: v['sha256'] for v in selected.values()})
    bound.update(status='prepared_diagnostic_failed_compiler' if diagnostic else 'prepared_for_main_CUDA_FD',
        reference=dict(receipt_path=str(receipt_path), receipt_sha256=receipt_sha, arrays=selected,
                       compiler_status=r['status'], compiler_error=r['error'], diagnostic_failed_compiler=diagnostic))
    require(not compiler.check_bindings(bound['bindings']), 'reference bind drift')
    out = new_output(output)
    compiler.bridge.write(out/'manifest_frozen.json', bound)
    verify_manifest(out/'manifest_frozen.json', file_sha(out/'manifest_frozen.json'))
    return bound


def perturbations(nominal):
    z = np.asarray(nominal)
    require(z.shape == (10,) and z.dtype == np.float32 and np.isfinite(z).all(), 'float32 normalized nominal[10] required')
    for epsilon in EPSILONS:
        for i in range(10):
            for sign in (1, -1):
                delta = np.zeros(10, np.float32)
                delta[i] = np.float32(sign*epsilon)
                yield epsilon, i, sign, z+delta


def central(plus, minus, epsilon):
    p, m = np.asarray(plus), np.asarray(minus)
    require(epsilon in EPSILONS and p.shape == m.shape and p.ndim == 1
            and p.dtype == m.dtype == np.float32 and np.isfinite(p).all() and np.isfinite(m).all(), 'finite float32 paired FD vectors required')
    return (p.astype(np.float64)-m.astype(np.float64))/(2*epsilon)


def compare(jvp, fd):
    j, f = np.asarray(jvp, np.float64), np.asarray(fd, np.float64)
    require(j.shape == f.shape and j.ndim == 2 and j.shape[1] == 10
            and np.isfinite(j).all() and np.isfinite(f).all(), 'finite aligned D-by-10 matrices required')
    def measures(x, y):
        nx, ny, error = float(np.linalg.norm(x)), float(np.linalg.norm(y)), float(np.linalg.norm(x-y))
        return dict(jvp_norm=nx, fd_norm=ny, error_norm=error,
                    relative_error_to_jvp=error/nx if nx else None,
                    cosine=float(np.sum(x*y)/(nx*ny)) if nx and ny else None,
                    max_abs_error=float(np.max(np.abs(x-y))))
    return dict(matrix=measures(j, f), columns=[dict(column=i, **measures(j[:, i], f[:, i])) for i in range(10)],
                acceptance_threshold=None, accepted=None)


def fd_loop(predict, nominal, ledger):
    """Same loop for native CUDA callback and explicitly non-neural arithmetic fixtures."""
    groups = {str(e): [] for e in EPSILONS}
    pair = {}
    for eps, column, sign, z in perturbations(nominal):
        name = f'eps_{eps}_column_{column}_'+('plus' if sign == 1 else 'minus')
        ledger.phase = name
        ledger.array(name+'_normalized_input', z)
        ledger.event('FD_dispatch', epsilon=eps, column=column, sign=sign,
                     effective_displacement=(z.astype(float)-nominal.astype(float)).tolist())
        returned = predict(z)
        # Save immediately, before finite checking, so nonfinite returns are auditable.
        ledger.array(name+'_successor', returned)
        require(returned.shape == (DIM,) and returned.dtype == np.float32 and np.isfinite(returned).all(), 'finite full native successor required')
        pair[sign] = returned
        if sign == -1:
            groups[str(eps)].append(central(pair[1], pair[-1], eps))
            pair = {}
    matrices = {e: np.stack(columns, axis=1) for e, columns in groups.items()}
    for eps, matrix in matrices.items():
        ledger.array('FD_matrix_eps_'+eps, matrix)
    return matrices


def cuda_core(m, c, p, reference, ledger):
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as frozen
    r = ledger.result
    models, handles, v4 = None, [], None
    try:
        require(not os.environ.get('PYTHONPATH'), 'fresh native process without physics PYTHONPATH required')
        frozen.require_cuda()
        # Loader metadata only; do NOT call frozen.prepare/verify_manifest/validate_cuda.
        models, strict, sources, before = frozen.load_cuda(ROOT, c['loader_manifest'])
        r.update(model_before=before, strict_loads=strict, imported_sources=sources)
        require(before == p['model_identity'], 'same compiler model identity required')
        helper = frozen.helper(ROOT)
        _, v5, _ = helper.dependencies(ROOT)
        v4 = v5.import_v4(ROOT)
        require(not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32, 'TF32 must remain disabled')
        def pre(key):
            def hook(module, args):
                require(key == 'predictor', 'FD consumer forbids visual/proprio encoding')
                require(r['attempted']['predictor'] < 40, '40 predict cap before dispatch')
                ledger.count('attempted', key)
            return hook
        def post(key):
            def hook(module, args, output):
                ledger.count('completed', key)
            return hook
        for key, model in models.items():
            handles.extend([model.register_forward_pre_hook(pre(key)), model.register_forward_hook(post(key))])
        visual = torch.from_numpy(reference['official_encoder_1_x_norm_patchtokens'][:3].copy()).to('cuda').reshape(1, 3, 1, 16, 16, 384)
        proprio = torch.from_numpy(reference['proprio_encoder_1']).to('cuda').expand(1, 3, 256, 16)
        history = torch.from_numpy(reference['normalized_history']).to('cuda')
        r['cached_context_scope'] = dict(new_encoder_calls=0, new_rgb_frames=0, context_frames=[0, 5, 10],
            goal_patch_not_used_in_FD=True, history_shape=list(history.shape))
        def predict(z):
            outgoing = torch.cat((history, torch.from_numpy(z).to('cuda').reshape(1, 1, 10)), dim=1)
            result = models['predictor'](visual, outgoing, proprio)
            require(result[0].shape == (1, 3, 256, 384) and result[2].shape == (1, 3, 256, 16), 'native output shape differs')
            value = result[0][:, -1].reshape(DIM).detach().cpu().numpy()
            return value
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            matrices = fd_loop(predict, reference['normalized_nominal'], ledger)
        r['comparisons'] = {e: compare(reference['response_columns'], fd) for e, fd in matrices.items()}
        r['eps_sensitivity'] = compare(matrices[str(EPSILONS[0])], matrices[str(EPSILONS[1])])
        r['eps_sensitivity']['labels'] = dict(jvp_norm='FD eps=.01 reference', fd_norm='FD eps=.005 comparator')
        nominal = reference['nominal_successor'].reshape(DIM).astype(float)
        r['saved_JVP_primal_vs_saved_nominal_max_abs'] = [float(np.max(np.abs(reference['jvp_primal_'+str(i)].astype(float)-nominal))) for i in range(10)]
        r['status'] = 'core_completed_pending_final_checks'
    except Exception as exc:
        r.update(status='FD_failed', error=type(exc).__name__+': '+str(exc), failed_phase=ledger.phase)
    finally:
        for i, handle in enumerate(handles):
            compiler.guarded_final(r, 'remove_hook_'+str(i), handle.remove)
        if models is not None and v4 is not None:
            def identity():
                r['model_after'] = {key: v4.model_receipt(model) for key, model in models.items()}
                require(r['model_after'] == r['model_before'] == p['model_identity'], 'native model changed')
            compiler.guarded_final(r, 'model_after', identity)
        compiler.guarded_final(r, 'synchronize', torch.cuda.synchronize)


def run(manifest_path, expected, output):
    m, c = verify_manifest(manifest_path, expected)
    require(m['status'] in ('prepared_for_main_CUDA_FD', 'prepared_diagnostic_failed_compiler') and m['reference'] is not None, 'actual JVP reference must be explicitly bound first')
    p = read_json(c['bridge_path'])
    receipt = read_json(m['reference']['receipt_path'])
    reference = reference_arrays(receipt, dict(path=m['compiler_manifest_path'], manifest=c), p, m['reference']['diagnostic_failed_compiler'])
    out = new_output(output)
    r = dict(schema=TAG+'.receipt', status='FD_failed', error=None, manifest_file_sha256=expected,
        compiler_receipt_sha256=m['reference']['receipt_sha256'], reference_compiler_status=receipt['status'],
        reference_compiler_error=receipt['error'], diagnostic_failed_compiler=m['reference']['diagnostic_failed_compiler'],
        compiler_inputs_sha256=m['compiler_inputs_sha256'], backend='host_cuda_only_SDPA_MATH',
        training=False, physical_calls=0, used_observed_frame15=False, selected_packet_changed=False,
        P5_admission=False, C2_established=False, partial_work_unknown=True)
    ledger = compiler.Ledger(out, r)
    ledger.event('begin', policy=m['policy'])
    try:
        cuda_core(m, c, p, reference, ledger)
    except Exception as exc:
        r.update(status='FD_failed', error=type(exc).__name__+': '+str(exc), failed_phase=ledger.phase)
    compiler.guarded_final(r, 'source_reference_after', lambda: verify_manifest(manifest_path, expected))
    compiler.guarded_final(r, 'arrays_after', lambda: require(not compiler.check_bindings(r['artifact_bindings']), 'FD arrays drift'))
    r['counts_match'] = r['attempted'] == r['completed'] == policy()['completed']
    if r['status'] == 'core_completed_pending_final_checks' and not r['finalization_errors'] and r['counts_match']:
        r.update(status='diagnostic_FD_recorded_from_failed_compiler' if m['reference']['diagnostic_failed_compiler'] else 'native_JVP_FD_comparison_recorded', partial_work_unknown=False)
    else:
        r['status'] = 'FD_failed'
        if r['error'] is None:
            r['error'] = 'final source/model/artifact checks or exact 40-predict count failed'
    ledger.phase = 'terminal'
    ledger.event('terminal', status=r['status'], error=r['error'])
    r['journal_sha256'] = file_sha(out/'journal.jsonl')
    compiler.bridge.write(out/'fd_receipt.json', r)
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--compiler-manifest', type=Path, required=True)
    a.add_argument('--compiler-manifest-sha256', required=True)
    a.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('bind-reference')
    b.add_argument('--compiler-receipt-sha256', required=True)
    b.add_argument('--diagnostic-failed-compiler', action='store_true')
    for q in (b, sub.add_parser('run-cuda')):
        q.add_argument('--manifest', type=Path, required=True)
        q.add_argument('--manifest-sha256', required=True)
        q.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.compiler_manifest, args.compiler_manifest_sha256, args.output)
    elif args.command == 'bind-reference':
        result = bind_reference(args.manifest, args.manifest_sha256, args.compiler_receipt_sha256, args.output, args.diagnostic_failed_compiler)
    else:
        result = run(args.manifest, args.manifest_sha256, args.output)
    print(json.dumps(dict(status=result['status'], output=str(args.output), P5_admission=False)))
    if args.command == 'run-cuda' and result['status'] == 'FD_failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
