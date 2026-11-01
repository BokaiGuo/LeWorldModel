"""Local-only P5 asset binding; optional scheduled TwoRoom calibration replay.

Default audit imports no neural runtime. No stage admission is granted here.
Run with -B; CUDA replay is opt-in and never falls back to CPU inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
CACHE = Path('/home/gbk/.stable_worldmodel')
PREFIX = 'p5_local_asset_binding_20261027_v1'
PREFLIGHT = ROOT / 'results/tworoom_d2p_p1_preflight_support_coverage64_samebank_20261025_cuda/report.json'
CALIBRATION = ROOT / 'results/tworoom_d2p_p1_calibration_support_coverage64_samebank_20261025_cuda/report.json'
STATS = ROOT / 'results/tworoom_d2p_training_action_stats_20261024/tworoom_training_action_stats.json'
POLICY = CACHE / 'checkpoints/tworoom/lewm'
DATASET = CACHE / 'datasets/tworoom.h5'
SEEDS = tuple(range(202610360, 202610368))
MODULES = ('checkpoint_compat.py', 'spectral_reachability/preflight_tworoom_d2p_p0.py',
           'spectral_reachability/preflight_tworoom_d2p_p1_reconfiguration.py',
           'spectral_reachability/tworoom_d2p_packet_support.py',
           'spectral_reachability/build_tworoom_d2p_training_action_stats.py',
           'run_live_tworoom_planning.py', 'gpu_preflight.py',
           'spectral_reachability/d2p_model_call_instrumentation.py')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def strict_json(path):
    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('JSON exceeds 8 MiB')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)
    result = json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=invalid)
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('nonfinite JSON number')
        if isinstance(value, dict):
            for child in value.values(): finite(child)
        elif isinstance(value, list):
            for child in value: finite(child)
    finite(result)
    return result


def asset(path, *, hash_bytes=True):
    path = Path(path)
    if path.resolve() != path.absolute():
        raise ValueError('symlink/relative traversal not allowed: ' + str(path))
    result = dict(path=str(path), exists=path.is_file())
    if result['exists']:
        before = path.stat()
        result.update(bytes=before.st_size, sha256=digest(path) if hash_bytes else None,
                      hash_scope='full_file' if hash_bytes else 'not_read_dataset_metadata_only')
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('input changed during hash')
    return result


def config_contract(config):
    expected = (config.get('_target_') == 'stable_worldmodel.wm.lewm.LeWM'
                and config.get('predictor', {}).get('num_frames') == 3
                and config.get('predictor', {}).get('output_dim') == 192
                and config.get('action_encoder', {}).get('input_dim') == 10)
    return dict(static_contract_compatible=expected, observation='RGB history and RGB goal',
                context='emb3 plus between-state outgoing actions2 plus new packet1',
                action='raw2 z-score before time-major 5x2 packing',
                runtime_load_verified=False)


def validate_calibration(pre, cal, stats):
    if pre.get('schema') != 'tworoom_d2p_p1_reconfiguration_preflight.v2' or pre.get('status') != 'pass':
        raise ValueError('wrong preflight schema/status')
    if not pre.get('gates') or any(v is not True for v in pre['gates'].values()):
        raise ValueError('preflight integrity gates')
    if (cal.get('schema') != 'tworoom_d2p_p1_physical_calibration.v2'
            or cal['protocol'].get('cohort') != 'calibration'
            or cal['protocol'].get('outcome_roots_opened') is not False
            or cal.get('upstream_p1') != str(PREFLIGHT.parent)):
        raise ValueError('calibration provenance mismatch')
    for roots in (cal['protocol']['root_seeds'], [r['seed'] for r in cal['rows']],
                  [r['seed'] for r in pre['rows']]):
        if tuple(roots) != SEEDS or any(type(s) is not int for s in roots):
            raise ValueError('fixed known calibration roots only')
    if (pre['protocol']['history_len'], pre['protocol']['action_block'],
            pre['protocol']['prefix_horizon']) != (3, 5, 1):
        raise ValueError('context/action dimensions changed')
    if stats.get('schema') != 'tworoom_d2p_training_action_stats.v1' or stats.get('status') != 'training_semantics_validated':
        raise ValueError('normalizer identity')
    if stats['source']['dataset'] != str(DATASET):
        raise ValueError('unexpected training dataset')
    row, = stats['reports']
    mean, std = row['calibration_action_mean'], row['calibration_action_std']
    if (row.get('all_packet_slots_share_physical_statistics') is not True
            or len(mean) != 10 or len(std) != 10 or mean != mean[:2] * 5 or std != std[:2] * 5
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in mean + std)
            or any(x <= 0 for x in std)):
        raise ValueError('invalid shared physical normalizer')
    if stats['source']['dataset_sha256'] != pre['training_packet_support']['dataset_sha256']:
        raise ValueError('saved training provenance differs')
    return row


def input_paths():
    paths = [PREFLIGHT, CALIBRATION, STATS, Path(__file__).resolve(),
             ROOT / 'tests/test_p5_local_asset_binding_20261027_v1.py']
    paths += [ROOT / 'experiments' / p for p in MODULES]
    paths += [REPO / 'docs' / p for p in ('P45_PREREQUISITE_SNAPSHOT_20261027.md',
              'D2P_CROSS_MODEL_BASELINE_ASSET_AUDIT_20261026.md')]
    for task in ('tworoom', 'reacher', 'pusht', 'cube'):
        paths += [CACHE / 'checkpoints' / task / 'lewm' / name for name in ('config.json', 'weights.pt', 'README.md')]
        paths += [ROOT / 'config/eval' / (task + '.yaml')]
    for task in ('pointmaze', 'pusht', 'wall'):
        paths += [ROOT / 'data/jepa_wm_checkpoints' / ('jepa_wm_' + task + '.pth.tar')]
    for task in ('point_maze', 'pusht', 'wall_single'):
        paths += [ROOT / 'data/dino_wm_osf/extracted/outputs' / task / 'checkpoints/model_latest.pth']
    paths += [ROOT / 'experiments/dino_checkpoint_compat/runtime.py',
              ROOT / 'external/local_dinov2/hubconf.py',
              ROOT / '.cache/dinov2/dinov2-small/config.json',
              ROOT / '.cache/dinov2/dinov2-small/model.safetensors']
    return paths


def audit():
    files = [asset(p) for p in input_paths()]
    pre, cal, stats = map(strict_json, (PREFLIGHT, CALIBRATION, STATS))
    validate_calibration(pre, cal, stats)
    configs = {task: config_contract(strict_json(CACHE / 'checkpoints' / task / 'lewm/config.json'))
               for task in ('tworoom', 'reacher', 'pusht', 'cube')}
    stable = all(not f['exists'] or digest(f['path']) == f['sha256'] for f in files)
    required = {str(p) for p in [PREFLIGHT, CALIBRATION, STATS, POLICY/'weights.pt', POLICY/'config.json']}
    ready = stable and DATASET.is_file() and configs['tworoom']['static_contract_compatible'] and all(
        f['exists'] for f in files if f['path'] in required)
    return dict(schema=PREFIX + '.v1', status='local_asset_audit_completed', files=files,
                dataset=asset(DATASET, hash_bytes=False), all_hashed_bytes_stable=stable,
                interfaces=configs, calibration_roots=list(SEEDS),
                calibration_evidence_status=cal['status'],
                tworoom_replay_assets_ready=ready, cuda_replay_executed=False,
                distinct_architecture_replay_ready=False,
                distinct_architecture_boundary='Known DINO-WM dynamics paths absent; cached DINOv2 encoder is insufficient. JEPA archives exist but their model identity/runtime are not certified by file presence.',
                excluded='All repair/adapters/readouts/teachers; no outcome reports or archives opened',
                scope='Fixed local paths only, not an exhaustive filesystem or public-asset absence claim',
                P1_admission=False, P5_admission=False)


def new_output(path):
    path = Path(path).absolute()
    if path.resolve() != path or path.parent != ROOT / 'results' or not path.name.startswith(PREFIX):
        raise ValueError('output must be a new P5-owned direct results child')
    path.mkdir(exist_ok=False)
    return path


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def verify_manifest(manifest):
    """No arbitrary paths from a manifest can become neural inputs."""
    if (manifest.get('schema') != PREFIX + '.v1' or manifest.get('P1_admission') is not False
            or manifest.get('P5_admission') is not False or manifest.get('calibration_roots') != list(SEEDS)
            or manifest.get('tworoom_replay_assets_ready') is not True):
        raise ValueError('invalid/non-ready binding manifest')
    entries = manifest['files']
    if [f['path'] for f in entries] != [str(p) for p in input_paths()]:
        raise ValueError('fixed asset allowlist mismatch')
    for entry in entries:
        if asset(entry['path']) != entry:
            raise ValueError('asset binding changed: ' + entry['path'])


def packet_pool_from_columns(expected):
    """Read training action/episode columns only, never HDF5 RGB or labels.

    Mirrors the sealed support construction, verifying its *full pool* digest.
    Does not claim to reverify the 12GB HDF5 full-file SHA.
    """
    import h5py
    import numpy as np
    chunks = []
    with h5py.File(DATASET, 'r', swmr=True) as handle:
        actions = handle['action']
        if actions.ndim != 2 or actions.shape[1] != 2:
            raise ValueError('training action shape')
        lengths, offsets = handle['ep_len'][:], handle['ep_offset'][:]
        if lengths.shape != offsets.shape:
            raise ValueError('episode column shape')
        for length, offset in zip(lengths, offsets):
            if int(length) < 5: continue
            episode = np.asarray(actions[int(offset):int(offset)+int(length)], dtype=np.float32)
            windows = np.moveaxis(np.lib.stride_tricks.sliding_window_view(episode, 5, axis=0), -1, 1)
            chunks.append(np.ascontiguousarray(windows[np.isfinite(windows).all(axis=(1, 2))]))
    pool = np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)
    if len(pool) != expected['packet_count'] or hashlib.sha256(pool.tobytes()).hexdigest() != expected['pool_sha256']:
        raise ValueError('full training packet pool changed')
    return pool


def check_context(actual, expected):
    for key in ('history_rgb_sha256', 'root_rgb_sha256', 'goal_rgb_sha256', 'warmup_actions_sha256'):
        if actual[key] != expected[key]:
            raise ValueError('same calibration context not reproduced: ' + key)


def replay(binding_path, output):
    manifest = strict_json(binding_path)
    verify_manifest(manifest)  # all fixed inputs verified before imports/allocation
    out = new_output(output)
    write_new(out / 'binding.json', manifest)
    import numpy as np
    import torch
    from experiments.spectral_reachability import preflight_tworoom_d2p_p1_reconfiguration as p1
    from experiments.spectral_reachability import tworoom_d2p_packet_support as support
    from experiments.gpu_preflight import require_cuda_allocation
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.spectral_reachability.d2p_model_call_instrumentation import PredictInstrumentation
    started = time.monotonic()
    host = require_cuda_allocation('cuda')
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    pre, cal, stats = map(strict_json, (PREFLIGHT, CALIBRATION, STATS))
    norm = validate_calibration(pre, cal, stats)
    mean, std = (np.asarray(norm[k], np.float32) for k in ('calibration_action_mean', 'calibration_action_std'))
    pool = packet_pool_from_columns(pre['training_packet_support'])
    model = load_compatible_pretrained(str(POLICY), device=device).eval().requires_grad_(False)
    before = p1.p0._parameter_hash(model)
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    env = p1.TwoRoomGoalAliasWrapper(p1.gym.make('swm/TwoRoom-v1', render_mode='rgb_array'))
    rows, records = [], []
    try:
        for saved in pre['rows'][:1]:  # fixed first already-used calibration root; no selection
            seed = saved['seed']
            warmup, indices = support.warmup_packets(pool, history_len=3, seed=seed)
            if indices.tolist() != saved['context']['training_packet_indices']:
                raise ValueError('warmup indices changed')
            context = p1.p0._collect_executed_context(env, seed=seed, history_len=3, action_block=5, warmup_packets=warmup)
            check_context(context, saved['context'])
            low = torch.as_tensor(env.action_space.low, device=device)
            high = torch.as_tensor(env.action_space.high, device=device)
            p1.p0._validate_tworoom_action_protocol(model, mean, std, action_dim=2, action_block=5)
            with torch.no_grad():
                emb, history, goal = p1._root_context(model, context, mean=mean, std=std,
                    action_block=5, action_dim=2, transform=p1._image_transform(), device=device)
                # Replay zero and one saved source-supported packet; no optimization/search.
                bank, meta = p1._proposal_bank(pool=pool, proposals=64, horizon=1, seed=seed, low=low, high=high)
                if meta['proposal_source_indices'] != saved['compilation']['proposal_source_indices']:
                    raise ValueError('proposal support indices changed')
                results = []
                for index in (0, 1):
                    params = bank[index:index+1]
                    args = dict(mean=mean, std=std, action_block=5, action_dim=2, low=low, high=high)
                    with PredictInstrumentation(model) as counter:
                        with counter.phase('evaluation', method='calibration_prediction_replay', horizon_tokens=1):
                            pred, _ = p1._advance(model, emb, history, params, **args)
                            repeated, _ = p1._advance(model, emb, history, params, **args)
                        records.extend(counter.records)
                    cost = float((pred[:, -1] - goal[None]).square().mean())
                    expected = saved['compilation']['candidate_rows'][index]['immediate_goal_cost']
                    matched = math.isclose(cost, expected, rel_tol=1e-4, abs_tol=1e-4)
                    deterministic = bool(torch.allclose(pred, repeated, atol=1e-6, rtol=1e-6))
                    if not torch.isfinite(pred).all() or not matched or not deterministic:
                        raise ValueError('saved prediction/determinism mismatch')
                    results.append(dict(candidate_index=index, cost=cost, saved_cost=expected,
                        saved_cost_matches=matched, deterministic=deterministic,
                        predicted_terminal=pred[0, -1].cpu().tolist()))
            rows.append(dict(seed=seed, context=saved['context'], predictions=results))
    finally:
        env.close()
    verify_manifest(manifest)
    gates = dict(parameters_unchanged=before == p1.p0._parameter_hash(model),
                 buffers_unchanged=all(torch.equal(buffers[k], v) for k, v in model.named_buffers()),
                 frozen_eval=all(not p.requires_grad for p in model.parameters()) and all(not m.training for m in model.modules()),
                 known_calibration_only=True, no_outcome_opened=True, input_bindings_unchanged=True,
                 actual_predict_accounting=sum(r['predict_calls'] for r in records) == 4
                 and sum(r['predict_samples'] for r in records) == 4 and all(r['completed'] for r in records))
    report = dict(schema=PREFIX + '.cuda.v1', status='replay_completed' if all(gates.values()) else 'integrity_fail',
                  host=host, gates=gates, rows=rows, predict_records=records,
                  predict_calls=sum(r['predict_calls'] for r in records),
                  predict_samples=sum(r['predict_samples'] for r in records),
                  rgb_history_frames_encoded=3, goal_images_encoded=1,
                  elapsed_seconds=time.monotonic()-started, backward_calls=0,
                  physical_actions='15 warmup actions only; no candidate physical execution',
                  dataset_verification='full training action packet pool SHA verified; HDF5 full SHA NOT reverified',
                  P1_admission=False, P5_admission=False)
    write_new(out / 'report.json', report)
    if not all(gates.values()): raise RuntimeError('frozen model integrity failed')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replay-cuda', action='store_true')
    parser.add_argument('--binding-manifest', type=Path)
    args = parser.parse_args()
    if args.replay_cuda:
        if args.output is None or args.binding_manifest is None:
            parser.error('CUDA replay needs --binding-manifest and new --output')
        result = replay(args.binding_manifest, args.output)
    else:
        if args.binding_manifest is not None: parser.error('--binding-manifest only for replay')
        result = audit()
        if args.output is not None: write_new(new_output(args.output) / 'manifest.json', result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
