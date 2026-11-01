"""Fresh, lossless native factual action-fidelity diagnostic; HOST CUDA only.

No observed-target optimization, physics calls, training, or P5 admission.
Prepare binds both repeat trajectories and all source bytes before neural work.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as parent

NAME = 'p5_jepa_fresh_fidelity_main_20261027_v10'
FACTUAL = 'p5_jepa_factual_nonzero_20261027_v2'
ROOT = Path(__file__).resolve().parents[2]
FRAMES = (0, 5, 10, 15)
ARMS = ('actual', 'normalized_zero', 'negative_normalized', 'reverse_steps')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def protocol():
    return dict(schema=NAME, factual_seed=2026102701, role='a', frames=list(FRAMES), arms=list(ARMS),
        counts=dict(official_encoder=1, proprio_encoder=1, predictor=4),
        action='native env u64; dataset pixel32=(100*u64).astype(float32); z=(pixel32/100-mean32)/std32; no u/100 unit error',
        intervention='last packet only; preserve first two packets and context proprio',
        metric='float64 global MSE predicted last successor vs factual frame15 native patch tokens',
        backend='official native SDPA MATH; TF32 disabled; all neural modules CUDA frozen eval',
        scope='one fresh fixed-seed nonzero factual trajectory, repeated physical identity; no counterfactual ground truth',
        prospective_control=False, target_optimized=False, training=False, P5_admission=False,
        cpu_neural_fallback=False)


def numeric(value, shape):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in 'fi' and np.isfinite(a).all(), 'finite numeric shape required')
    return a


def normalized_inputs(rgb, proprio, commands, stats):
    require(rgb.shape == (4, 224, 224, 3) and rgb.dtype == np.uint8, 'lossless native RGB required')
    p = numeric(proprio, (4, 4)).astype(np.float32)
    u = numeric(commands, (15, 2)).astype(np.float64)
    dataset_u = (100. * u).astype(np.float32) / np.float32(100.)
    mean = numeric(stats['ACTION_MEAN'], (2,)).astype(np.float32)
    std = numeric(stats['ACTION_STD'], (2,)).astype(np.float32)
    pm = numeric(stats['PROPRIO_MEAN'], (4,)).astype(np.float32)
    ps = numeric(stats['PROPRIO_STD'], (4,)).astype(np.float32)
    require(np.all(std > 0) and np.all(ps > 0), 'positive source std required')
    return (p - pm) / ps, ((dataset_u - mean) / std).reshape(3, 10)


def controls(packets):
    a = numeric(packets, (3, 10)).astype(np.float32)
    out = {arm: a.copy() for arm in ARMS}
    out['normalized_zero'][-1] = 0
    out['negative_normalized'][-1] *= -1
    out['reverse_steps'][-1] = a[-1].reshape(5, 2)[::-1].reshape(10)
    require(all(not np.array_equal(out['actual'], out[arm]) for arm in ARMS[1:]),
        'nontrivial last-packet controls required; zero-only data prohibited')
    return out


def factual_inputs(directory):
    directory = Path(directory).resolve()
    require(directory.parent.name == FACTUAL, 'fixed source-native nonzero collector required')
    bindings = {}
    def read(path):
        path = Path(path)
        bindings[str(path)] = parent.sha(path)
        return json.loads(path.read_text())
    manifest_path = directory.parent / 'manifest_frozen.json'
    manifest, report = read(manifest_path), read(directory / 'report.json')
    require(manifest['schema'] == FACTUAL and bool(manifest['seals']), 'nonempty source-native seal required')
    require(report['schema'] == FACTUAL and report['status'] == 'fresh_native_nonzero_factual_recorded'
        and report['error'] is None and report['source_hashes_unchanged'] is True
        and report['manifest_file_sha256'] == bindings[str(manifest_path)]
        and report['protocol'] == manifest['protocol'], 'complete source-bound fresh factual report required')
    require(report['trajectory_frames'] == {'a': 16, 'b': 16} and len(report['rows']) == 16,
        'both complete independent repeats required')
    require(report['protocol']['seed'] == 2026102701
        and report['protocol']['frames'] == list(FRAMES), 'fixed seed and native time contract required')
    expected_calls = dict(env_construct=2, seed=4, reset=2, control_step=30, space_step=302,
        native_render=32, native_get_obs=32, close=2)
    require(all(report['native_call_receipts']['python_call_entries'].get(k) == v
        for k, v in expected_calls.items()), 'complete independent native physical census required')
    for i, row in enumerate(report['rows']):
        require(row['frame'] == i and row['all_exact'] is True and all(row['exact'].values()),
            'all corresponding factual frames must be exact')
    for path, value in manifest['seals'].items():
        require(parent.sha(path) == value, 'collector source seal drift: ' + path)
        bindings[path] = value
    journal = directory / 'journal.jsonl'
    require(parent.sha(journal) == report['journal_sha256'], 'factual journal digest mismatch')
    bindings[str(journal)] = parent.sha(journal)
    previous = None
    asset_records = {}
    for sequence, line in enumerate(journal.read_text().splitlines()):
        entry = json.loads(line)
        require(entry['sequence'] == sequence and entry['previous_sha256'] == previous,
            'factual journal sequence/hash chain mismatch')
        payload = {k: v for k, v in entry.items() if k != 'entry_sha256'}
        require(digest(payload) == entry['entry_sha256'], 'factual journal entry hash mismatch')
        previous = entry['entry_sha256']
        event = entry['event']
        if event.get('phase') == 'assets_durable':
            record = event['record']
            key = event['role'], record['frame']
            require(key not in asset_records, 'duplicate durable factual frame')
            asset_records[key] = record
    require(set(asset_records) == {(role, i) for role in ('a', 'b') for i in range(16)},
        'all32 durable factual frame receipts required')
    require(entry['event']['phase'] == 'terminal' and entry['event']['summary_sha256'] == digest(
        {k: v for k, v in report.items() if k != 'journal_sha256'}), 'terminal factual summary hash mismatch')
    # Bind every file in BOTH factual trajectories, not merely selected RGB.
    for role in ('a', 'b'):
        for path in sorted((directory / role).rglob('*')):
            if path.is_file():
                require(not path.is_symlink(), 'no symlink factual assets')
                bindings[str(path)] = parent.sha(path)
    for i in range(16):
        for name in ('rgb.npy', 'state.npy', 'proprio.npy', 'full_public_state.json', 'transition.json'):
            paths = [directory / role / f'frame_{i:02d}' / name for role in ('a', 'b')]
            require(bindings[str(paths[0])] == bindings[str(paths[1])], 'full repeat asset mismatch')
            for role, path in zip(('a', 'b'), paths):
                require(asset_records[role, i]['files'][name]['sha256'] == bindings[str(path)],
                    'durable frame receipt/file mismatch')
    rgb, prop = [], []
    for i in FRAMES:
        left, right = [directory / role / f'frame_{i:02d}' for role in ('a', 'b')]
        for name in ('rgb.npy', 'state.npy', 'proprio.npy', 'full_public_state.json', 'transition.json'):
            require(bindings[str(left / name)] == bindings[str(right / name)], 'repeat asset identity differs')
        image = np.load(left / 'rgb.npy', allow_pickle=False)
        state = numeric(np.load(left / 'state.npy', allow_pickle=False), (7,))
        proprio = numeric(np.load(left / 'proprio.npy', allow_pickle=False), (4,))
        require(np.array_equal(proprio, state[[0, 1, 5, 6]]), 'native proprio-state mismatch')
        rgb.append(image)
        prop.append(proprio)
    commands = []
    for i in range(1, 16):
        command = read(directory / 'a' / f'command_{i:02d}.json')
        require(command['frame'] == i, 'command time binding mismatch')
        commands.append(command['raw_env_argument'])
    require(np.array_equal(numeric(commands, (15, 2)),
        numeric(report['protocol']['schedule'], (3, 5, 2)).reshape(15, 2)), 'commands differ from predeclared source schedule')
    require(any(np.any(numeric(u, (2,)) != 0) for u in commands), 'nonzero factual controls required')
    return np.stack(rgb), np.stack(prop), np.asarray(commands), bindings


def prepare(factual):
    frozen_path = ROOT / 'results/p5_jepa_native_action_response_20261027_v8/manifest_frozen.json'
    frozen = json.loads(frozen_path.read_text())
    parent.verify_manifest(ROOT, frozen)
    rgb, prop, commands, bindings = factual_inputs(factual)
    v7 = parent.helper(ROOT)
    _, v5, _ = v7.dependencies(ROOT)
    v4 = v5.import_v4(ROOT)
    stats_path = v4.native_root(ROOT) / 'app/plan_common/datasets/pusht_dset.py'
    stats = v4.source_stats(stats_path)
    normalized_prop, packets = normalized_inputs(rgb, prop, commands, stats)
    banks = controls(packets)
    exported = Path(factual).resolve() / 'a/consumer/normalized_packets.npy'
    require(str(exported) in bindings and np.array_equal(packets, np.load(exported, allow_pickle=False)),
        'consumer action normalization differs from factual dataset projection')
    bindings.update(frozen['seals'])
    own = [Path(__file__).resolve(), ROOT / 'tests' / f'test_{NAME}.py',
        ROOT.parent / 'docs' / f'{NAME}.md', frozen_path, Path(parent.__file__), stats_path]
    bindings.update({str(p): parent.sha(p) for p in own})
    return dict(schema=NAME, status='prepared_non_neural', protocol=protocol(),
        factual=str(Path(factual).resolve()), parent_manifest=frozen, bindings=bindings,
        input_sha256=dict(rgb=v4.array_sha(rgb), normalized_proprio=v4.array_sha(normalized_prop),
            packets=v4.array_sha(packets)), stats=stats, controls={k: v.tolist() for k, v in banks.items()})


def run_cuda(manifest):
    import torch
    require(manifest == prepare(manifest['factual']), 'fresh consumer manifest drift')
    require(not __import__('os').environ.get('PYTHONPATH'), 'neural process must not use physics overlay')
    parent.require_cuda()
    counts = dict(official_encoder=0, proprio_encoder=0, predictor=0)
    result = dict(schema=NAME, status='fresh_fidelity_failed', protocol=protocol(), counts=counts,
        manifest_sha256=digest(manifest), cuda_executed=True, new_physical_steps=0, P5_admission=False)
    models, handles = None, []
    try:
        models, strict, sources, before = parent.load_cuda(ROOT, manifest['parent_manifest'])
        v7 = parent.helper(ROOT)
        _, v5, _ = v7.dependencies(ROOT)
        v4 = v5.import_v4(ROOT)
        result.update(model_before=before, strict_loads=strict, imported_sources=sources)
        def counted(key):
            def hook(*unused):
                counts[key] += 1
            return hook
        for key, model in models.items():
            handles.append(model.register_forward_hook(counted(key)))
        rgb, prop, commands, _ = factual_inputs(manifest['factual'])
        norm_prop, packets = normalized_inputs(rgb, prop, commands, manifest['stats'])
        from app.plan_common.datasets.transforms import make_transforms
        from torch.nn.attention import sdpa_kernel, SDPBackend
        transform = make_transforms(img_size=224, random_horizontal_flip=False,
            random_resize_aspect_ratio=(1., 1.), random_resize_scale=(1., 1.),
            reprob=0., auto_augment=False, motion_shift=False)
        artifacts = {}
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            pixels = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).to('cuda', torch.float32) / 255.
            patch = models['official_encoder'](transform(pixels), is_training=True)['x_norm_patchtokens']
            require(patch.shape == (4, 256, 384) and patch.is_cuda and torch.isfinite(patch).all(), 'CUDA patch contract')
            visual, target = patch[:3].reshape(1, 3, 1, 16, 16, 384), patch[3:4]
            features = models['proprio_encoder'](torch.from_numpy(norm_prop[:3]).unsqueeze(0).to('cuda')).expand(1, 3, 256, 16)
            costs = {}
            artifacts['official_patch_features'] = patch.cpu().numpy()
            for arm, acts in controls(packets).items():
                require(acts.tolist() == manifest['controls'][arm], 'control packets drift')
                output = models['predictor'](visual, torch.from_numpy(acts).unsqueeze(0).to('cuda'), features)
                require(output[0].shape == (1, 3, 256, 384) and output[0].is_cuda
                    and torch.isfinite(output[0]).all(), 'CUDA successor contract')
                successor = output[0][:, -1]
                costs[arm] = (successor.double() - target.double()).square().mean().item()
                artifacts[arm + '_successor'] = successor.cpu().numpy()
            result.update(status='fresh_native_fidelity_recorded', costs=costs,
                actual_minus_control={a: costs['actual'] - costs[a] for a in ARMS[1:]},
                artifact_arrays=artifacts, prediction_target='observed factual successor; no counterfactual ground truth')
        result['model_after'] = {k: v4.model_receipt(m) for k, m in models.items()}
        result['model_unchanged'] = result['model_after'] == before
    except Exception as exc:
        result.update(status='fresh_fidelity_failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        for handle in handles:
            handle.remove()
        result['counts_match'] = counts == protocol()['counts']
        result['sources_unchanged'] = all(parent.sha(p) == h for p, h in manifest['bindings'].items())
        if not all(result.get(k, False) for k in ('counts_match', 'sources_unchanged', 'model_unchanged')):
            result['status'] = 'fresh_fidelity_failed'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'cuda'))
    parser.add_argument('--factual', type=Path, default=ROOT / f'results/{FACTUAL}/factual_main_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.factual)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    else:
        frozen = json.loads(args.output.read_text())
        destination = args.output.parent / 'cuda_main_v1'
        destination.mkdir(exist_ok=False)
        result = run_cuda(frozen)
        arrays = result.pop('artifact_arrays', {})
        result['artifacts'] = {}
        for key, value in arrays.items():
            path = destination / f'{key}.npy'
            with path.open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
            result['artifacts'][key] = dict(path=str(path), sha256=parent.sha(path), shape=list(value.shape))
        with (destination / 'report.json').open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result['status'], P5_admission=False)))
    if result['status'] == 'fresh_fidelity_failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
