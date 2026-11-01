"""Join the completed E3 chain to independent physical and CUDA encoder replays."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability import e3_cell03_encoder_parity_inputs_codex_20261030_v1 as source

ROOT = source.ROOT
PHYS = ROOT/'e3_cell03_physical_replay_codex_20261030_v1_RUN01'
CUDA = ROOT/'e3_cell03_cuda_parity_codex_20261030_v2_RUN01'
OUTPUT = ROOT/'e3_cell03_replay_join_codex_20261030_v1.json'
HASHES = {
    'physical_report': 'b7909f23615bd2412ff2a283daa8c9c005a5703b4d3509d5981757a0025800de',
    'physical_events': '7ee894019c38ce0c67818c4acff2fb6e512b0112bf35b355311dbe5af49cb792',
    'physical_manifest': 'f09d27611338c24caaa8df59a8744c20e66408b5927216feb13fdf184805fab3',
    'cuda_report': '64e616ee26cf150362a6ba548fa2aef3b6e06eef2b61fb8d0b39604ee9a4827d',
    'cuda_events': 'ef235ccb47fb4258d69ef7835547a230a721ea4092992b35a226c9e60df59327',
    'cuda_manifest': '29de759260c25e4bae99112971a17910bdf918d9b6911aa20e7c1815fe49eae9',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path, expected):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), 'regular file required')
    data = p.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected, 'SHA mismatch: '+str(p))
    return data


def join_boundary(boundary, original_events, replay_events, input_row, cuda_row, physical_dir, cuda_dir):
    def one(events, key, pred):
        found = [e for e in events if pred(e) and key(e)]
        require(len(found) == 1, 'unique boundary event')
        return found[0]
    def original(resource):
        return one(original_events, lambda e: e['resource'] == resource and
                   e['detail'].get('boundary') == boundary, lambda e: True)
    obs = one(replay_events, lambda e: e['kind'] == 'fresh_observation' and
              e['detail']['receipt']['phase'] == 'plan' and
              e['detail']['receipt']['step'] == boundary, lambda e: True)['detail']['receipt']
    enc_intent = original('native_e3_encode_intent')
    enc_complete = original('encoding_complete')
    enc_return = original('native_e3_encode_return')
    node_intent = original('native_e3_node_intent')
    node_return = original('native_e3_node_return')
    packet = original('native_e3_packet_intent')
    require(enc_intent['index'] < enc_complete['index'] < enc_return['index'] <
            node_intent['index'] < node_return['index'] < packet['index'], 'feedback chronology')
    original_obs = enc_intent['detail']['observation']
    require(original_obs['step'] == boundary and
            original_obs['pixel_sha256'] == obs['pixel_sha256'] == input_row['frame']['pixel_sha256'],
            'actual replay and encoder reference pixel binding')
    require(enc_complete['detail']['frame'] == input_row['frame'] and
            enc_return['detail']['observation_sha'] == original_obs['file_sha256'] ==
            input_row['frame']['sha256'], 'original frame encoding identity')
    require(node_return['detail']['body_sha256'] == node_intent['detail']['body_sha256'] and
            node_return['detail']['raw_bits'] == packet['detail']['raw_bits'], 'node action next packet binding')
    expected_controls = [e for e in replay_events if e['kind'] == 'control_return' and
                         boundary < e['detail']['step'] <= boundary+5]
    expected_controls.sort(key=lambda e: e['detail']['step'])
    require([e['detail']['step'] for e in expected_controls] == list(range(boundary+1, boundary+6)),
            'subsequent replay action denominator')
    action_bytes = b''.join(bytes.fromhex(e['detail']['raw_bits']) for e in expected_controls)
    require(action_bytes.hex() == node_return['detail']['raw_bits'], 'physical replay next actions')
    frame = physical_dir/Path(obs['path']).name
    data = read(frame, obs['file_sha256'])
    import io
    with Image.open(io.BytesIO(data)) as image:
        pixels = np.asarray(image)
    require(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8 and
            hashlib.sha256(pixels.tobytes()).hexdigest() == obs['pixel_sha256'], 'actual replay PNG')
    require(cuda_row['boundary'] == boundary, 'CUDA boundary')
    for role in source.ROLES:
        ref = source.checked_array(source.OUTPUT, input_row['arrays'][role])
        name = f'boundary{boundary}_{role}.npy'
        from io import BytesIO
        data = read(cuda_dir/name, cuda_row['actual_sha256'][role])
        stream = BytesIO(data)
        actual = np.load(stream, allow_pickle=False)
        require(stream.tell() == len(data) and ref.dtype == actual.dtype == np.float32 and
                ref.shape == actual.shape == source.SHAPES[role] and
                np.isfinite(actual).all() and ref.tobytes() == actual.tobytes(), 'CUDA array exact')
        require(cuda_row['comparison'][role] == dict(bitwise_equal=True, max_abs_error=0.0),
                'CUDA comparison result')
    return dict(boundary=boundary, pixel_sha256=obs['pixel_sha256'],
                next_action_sha256=hashlib.sha256(action_bytes).hexdigest(),
                original_node_event=node_return['index'], physical_frames_match=True,
                encoder_recompute_bitwise=True)


def audit():
    prepared = source.inputs()  # authenticates original publication, sources and numeric saved chain
    physical_report = json.loads(read(PHYS/'report.json', HASHES['physical_report']))
    pe = [json.loads(x) for x in read(PHYS/'events.jsonl', HASHES['physical_events']).splitlines()]
    pm = json.loads(read(ROOT/'e3_cell03_physical_replay_codex_20261030_v1_manifest.json', HASHES['physical_manifest']))
    cuda_report = json.loads(read(CUDA/'report.json', HASHES['cuda_report']))
    ce = [json.loads(x) for x in read(CUDA/'events.jsonl', HASHES['cuda_events']).splitlines()]
    cm = json.loads(read(ROOT/'e3_cell03_cuda_parity_codex_20261030_v2_manifest.json', HASHES['cuda_manifest']))
    require(physical_report['status'] == cuda_report['status'] == 'completed' and
            physical_report['manifest_sha256'] == HASHES['physical_manifest'] and
            cuda_report['manifest_sha256'] == HASHES['cuda_manifest'] and
            pm['source_report'] == cm['original_report'] == source.REPORT_SHA and
            pm['source_completion'] == cm['original_completion'] == source.COMPLETION_SHA,
            'completed replays of same pinned source')
    require([e['index'] for e in pe] == list(range(len(pe))) and
            [e['stage'] for e in ce if e.get('stage') in ('encode_intent', 'encode_return')] ==
            ['encode_intent', 'encode_return']*2 and
            [e['boundary'] for e in ce if e.get('stage') == 'encode_return'] == [10, 15],
            'replay event chronology')
    original = [json.loads(x) for x in (source.OUTPUT/'events.jsonl').read_bytes().splitlines()]
    require([x['boundary'] for x in prepared['rows']] == [10, 15] and
            [x['boundary'] for x in cuda_report['rows']] == [10, 15], 'fixed boundaries')
    rows = [join_boundary(b, original, pe, ir, cr, PHYS, CUDA)
            for b, ir, cr in zip((10, 15), prepared['rows'], cuda_report['rows'], strict=True)]
    return dict(schema='e3_cell03_replay_join_v1', input_sha256=HASHES, rows=rows,
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='cross-run saved link; no new execution, not live closed-loop parity',
                planning_cuda_recomputed=False, full_saved_strict=False, scientific_admission=False)


def main():
    result = audit()
    data = (json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
    fd = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    print(hashlib.sha256(data).hexdigest())


if __name__ == '__main__':
    main()
