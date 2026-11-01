"""Recompute known2 saved RGB/history/action joins; no model or admission."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    'e1': ('results/agent_experiment1_common_controller_20261028_v1_prepared_MAIN01.json', '8c472485d44c40ce3d95f8043d2c58f9ec2694fef74786a840d1699d5a1a86bd'),
    'hi': ('results/agent_hilewm_rgb_high_low_20261028_v1_prepare_MAIN01/manifest.json', 'a3255d38e9cf718f9560c1b9e381abe38750edd9e91d3f04e5ad332cf414d7e9'),
    'parity': ('results/agent_p2_hilewm_computation_parity_20261029_v1_consume_MAIN01.json', '3555696c02329062084d791c0c40958084d7352e6f7680938bade0068afe9480'),
    'first': ('results/agent_e1_first_history_bridge_20261029_v1_MAIN01.json', '3e4eb4839c8b2cb9e6b4fb272a59d3a295f1fbc225d0ec6ed401afa40fbf489d'),
    'producer': ('experiments/spectral_reachability/p1_constructive_moving_history_encode_cuda_20261027_v1.py', '246b9194fd31c874d7f0da67c4dec17365f93294e6ed460fe46e2509673e1b63'),
    'normalize_source': ('experiments/spectral_reachability/p1_conditional_task_admission_20261027_v1.py', 'd28d48c183791bfe1cae03e8a78ba27e28039cfd355993a19b32992cd7f6e211'),
}
TASK_SHA = ('a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b', '101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74')
BLIND_SHA = ('3c749e224e075061353dee590aff5883b202854451ef95f9bbf951510e0f6f0c', '4e9aeb8504db653552c4a9924cbc0c3ddfc53f1e6f3a56e5b73fc6c8f1b3be39')


def sha(b): return hashlib.sha256(b).hexdigest()


def read(path, expected, limit=2_000_000):
    path = Path(path)
    if path.stat().st_size > limit: raise ValueError('input too large')
    data = path.read_bytes()
    if sha(data) != expected: raise ValueError('SHA mismatch: '+str(path))
    return data


def require(ok, reason):
    if not ok: raise ValueError(reason)


def action_join(raw, recorded, mean, std):
    raw, recorded, mean, std = [np.asarray(x, dtype=np.float32) for x in (raw, recorded, mean, std)]
    require(raw.shape == (10, 2) and recorded.shape == (2, 10) and mean.shape == std.shape == (2,), 'action shape')
    require(all(np.isfinite(v).all() for v in (raw, recorded, mean, std)) and (std > 0).all() and (abs(raw) <= 1).all(), 'action values')
    calculated = (raw.reshape(2, 10) - np.tile(mean, 5)) / np.tile(std, 5)
    require(calculated.tobytes() == recorded.tobytes(), 'action order or normalized token mismatch')
    return calculated


def build(root=ROOT):
    sources = {k: read(root / p, h) for k, (p, h) in FILES.items()}
    e1, hi, parity, first = [json.loads(sources[x]) for x in ('e1','hi','parity','first')]
    require(parity['all_values_exact'] is True and parity['raw_deployment'] is False, 'parity receipt scope')
    rows = []
    for i in range(2):
        es, hs, fs, pr = e1['slots'][i], hi['slots'][2*i], first['rows'][2*i], parity['diagnostics']['roots'][i]
        require(es['source_slot'] == hs['root_slot'] == fs['root_slot'] == pr['root'] == i and hs['condition'] == fs['condition'] == 'off', 'slot identity')
        require(es['task_sha256'] == TASK_SHA[i] and es['blind_sha256'] == hs['blind_sha256'] == BLIND_SHA[i], 'manifest identity')
        path = root / f'results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_0{i}.json.gz'
        raw_task = read(path, TASK_SHA[i])
        with gzip.GzipFile(fileobj=io.BytesIO(raw_task)) as gz: data = gz.read(2_000_001)
        require(len(data) <= 2_000_000, 'task expansion')
        task = json.loads(data); t, c = task['task'], task['task_contract']
        require(task['task_id'] == es['task_id'] and task['source_slot'] == i and c['source_slot'] == i and c['observed_steps'] == [0,5,10] and c['blind_sha256'] == BLIND_SHA[i] and task['history_semantics'] == 'actual_moving_raw_between_observed0_5_10', 'task identity/time')
        blind_path = root / f'results/p1_moving_target_overlay_source_20261027_v1_main01/s{i}_off_blind_inputs.npz'
        blind = read(blind_path, BLIND_SHA[i])
        with np.load(io.BytesIO(blind), allow_pickle=False) as archive:
            require(set(archive.files) == {'rgb_history','goal_rgb','raw_action_history'}, 'blind inventory')
            rgb, goal, raw = (archive[k] for k in ('rgb_history','goal_rgb','raw_action_history'))
        require(rgb.shape[0] == 3 and raw.dtype.str == '<f4' and raw.shape == (10,2), 'blind ABI')
        pixel = [sha(x.tobytes()) for x in (*rgb, goal)]
        require(pixel == c['pixel_sha256'] == hs['pixel_sha256'] and sha(raw.tobytes()) == c['raw_action_sha256'] == hs['raw_sha256'], 'pixel/raw identities')
        require(raw.tobytes() == np.asarray(t['root_raw_action_history'], np.float32).tobytes(), 'raw action task join')
        require(t['action_mean'] == c['normalization']['mean'] and t['action_std'] == c['normalization']['std'], 'recorded P2 stats join')
        z = np.asarray(t['root_context']['latent_history'], dtype=np.float32)
        a = action_join(raw, t['root_context']['action_history'], t['action_mean'], t['action_std'])
        require(z.shape == (3,192) and np.isfinite(z).all() and sha(z[-1:].tobytes()) == fs['current_raw_sha256'], 'FIRST input history endpoint')
        checks = pr['paid_history_checks']
        names = {f'pixel{k}.{route}.vs_paid_P2_history' for k in range(3) for route in ('p2','hi')}
        require(len(checks) == 6 and {x['name'] for x in checks} == names and all(x['exact'] is True and x['same_ABI'] is True for x in checks), 'paid encoding receipt')
        rows.append(dict(root=i, task_sha256=TASK_SHA[i], blind_sha256=BLIND_SHA[i], pixel_sha256=pixel,
                         latent_history_sha256=sha(z.tobytes()), action_history_sha256=sha(a.tobytes()),
                         raw_packet_sha256=[sha(raw[j:j+5].tobytes()) for j in (0,5)],
                         latent_frame_sha256=[sha(x.tobytes()) for x in z],
                         first_target_sha256=fs['target_raw_sha256'],
                         frames=[0,5,10], action_intervals=[[0,5],[5,10]],
                         paid_encoding_equality='prior saved parity receipt; not independently rerun'))
    return dict(schema='e1_history3_provenance_bridge.v1', sources={k:h for k,(_,h) in FILES.items()}, rows=rows,
                stats_scope='recorded P2 engineering constants, not certified Hi historical scaler',
                history3_tracker_dispatched=False, raw_deployment=False, scientific_admission=False,
                target_physical_time=None, source_admission='original manifest only, not independent verification',
                inherited_compute_cost=None)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); result = build()
    data = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    with args.output.open('xb') as f: f.write(data)
    print(sha(data))


if __name__ == '__main__': main()
