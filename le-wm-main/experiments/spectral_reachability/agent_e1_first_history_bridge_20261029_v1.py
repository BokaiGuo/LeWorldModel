"""Read-only FIRST delivery inspection and non-authorizing history3 ABI check."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / 'results/agent_hilewm_rgb_high_low_20261028_v1_cuda_MAIN01'
CONSUME = ROOT / 'results/agent_hilewm_rgb_high_low_20261028_v1_consume_MAIN01.json'
REPORT_SHA = '083d0d9aa190b561e6ccf865b09d7224b97d404b16ba38c792bd8af386f5622a'
CONSUME_SHA = '40f28414682c9540e9e195b361a36dfc514f816358adbeacb379fedc460e4ef0'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked(path, expected, limit):
    data = path.read_bytes()
    if len(data) > limit or digest(data) != expected:
        raise ValueError('size or SHA mismatch: ' + str(path))
    return data


def array(path, expected, shape):
    data = checked(path, expected, 100_000)
    stream = io.BytesIO(data)
    value = np.load(stream, allow_pickle=False)
    if (not isinstance(value, np.ndarray) or value.dtype.str != '<f4' or
            value.shape != shape or stream.tell() != len(data) or
            not np.isfinite(value).all()):
        raise ValueError('invalid array ABI: ' + str(path))
    return value


def inspect(capture=CAPTURE, consume=CONSUME):
    report = json.loads(checked(capture / 'report.json', REPORT_SHA, 2_000_000))
    receipt = json.loads(checked(consume, CONSUME_SHA, 100_000))
    if not report['completed'] or receipt['report_sha256'] != REPORT_SHA or len(report['rows']) != 4 or len(receipt['rows']) != 4:
        raise ValueError('capture/receipt mismatch')
    def get(name, shape):
        if not isinstance(name, str) or Path(name).name != name or not name.startswith('array_') or not name.endswith('.npy'):
            raise ValueError('invalid array reference')
        meta = report['arrays'][name]
        val = array(capture / name, meta['file_sha256'], shape)
        if digest(val.tobytes()) != meta['data_sha256']:
            raise ValueError('array raw digest mismatch')
        return val
    rows = []
    for i, (row, prior) in enumerate(zip(report['rows'], receipt['rows'], strict=True)):
        root, condition = i // 2, ('off', 'on')[i % 2]
        if (row['index'], row['root_slot'], row['condition']) != (i, root, condition) or (prior['index'], prior['root_slot'], prior['condition']) != (i, root, condition):
            raise ValueError('row identity mismatch')
        rollout = get(row['selected_rollout'], (1, 1, 2, 192))
        target = get(row['high_target'], (1, 192))
        goal = get(row['low']['z_subgoal'], (1, 192))
        current = get(row['z_init'], (1, 192))
        z = get(row['low']['z_hist'], (1, 1, 192))
        a = get(row['low']['a_hist'], (1, 1, 10))
        raw = digest(target.tobytes())
        if (target.tobytes() != rollout[:, 0, 0, :].tobytes() or target.tobytes() != goal.tobytes() or
                current.tobytes() != z[:, 0, :].tobytes() or np.any(a != 0) or
                prior['high_target_sha256'] != raw or prior['low_goal_sha256'] != raw or prior['target_delivered_exact'] is not True):
            raise ValueError('FIRST delivery or native history mismatch')
        rows.append(dict(index=i, root_slot=root, condition=condition, target_file=row['high_target'],
                         target_raw_sha256=raw, current_raw_sha256=digest(current.tobytes()),
                         source_native_history_shapes=[[1, 1, 192], [1, 1, 10]], target_physical_time=None))
    return dict(schema='e1_first_history_bridge.v1', source_report_sha256=REPORT_SHA,
                source_consume_sha256=CONSUME_SHA, rows=rows,
                scientific_admission=False, physical_release=False, official_parity=False,
                history3_delivery_executed=False, scaler_authority=None,
                historical_process=None, production_cost=None,
                limitation='Saved FIRST-to-native-low subset only; not matched E1 or full saved audit.')


def check_history(zpath, zsha, apath, asha, current_sha):
    z = array(Path(zpath), zsha, (1, 3, 192))
    a = array(Path(apath), asha, (1, 2, 10))
    if digest(z[:, -1, :].tobytes()) != current_sha:
        raise ValueError('last history frame does not match saved high input')
    return {'shape_and_last_frame_match': True, 'history_raw_sha256': digest(z.tobytes()),
            'action_history_raw_sha256': digest(a.tobytes()), 'source_authentication': False,
            'coordinate_and_action_provenance_verified': False, 'admission': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='new exclusive JSON file')
    args = p.parse_args()
    result = inspect()
    payload = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with args.output.open('xb') as f:
        f.write(payload)
    print(digest(payload))


if __name__ == '__main__':
    main()
