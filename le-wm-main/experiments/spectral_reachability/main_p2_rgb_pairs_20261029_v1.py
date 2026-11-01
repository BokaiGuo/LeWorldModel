"""Independent fixed6 same-action/time pairs from sealed native RGB arrays."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'main_p2_rgb_pairs_20261029_v1'
FILES = {
    'rgb': ('agent_p2_three_program_rgb_20261029_v1_cuda_MAIN01/report.json', '3714a65889384be10df00552a3b023b5d55959825635d3b4e975203d12a47eee'),
    'receipt': ('agent_p2_three_program_rgb_20261029_v1_consume_MAIN01.json', '26b65ef96371d8e09cd5f5601225aff901e87234c6f240ae6f817bc5534210ab'),
    'physical': ('agent_p2_three_program_physical_20261029_v1_MAIN01/report.json', '49b682d0ca326590c88ef69d74eecfdd9d5c19f21667e790a7c1458aac98e064'),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def mse(a, b):
    return float(np.mean((np.asarray(a, np.float64)-np.asarray(b, np.float64))**2))


def run(output):
    docs = {}
    for key, (name, expected) in FILES.items():
        body = (ROOT/'results'/name).read_bytes()
        require(hashlib.sha256(body).hexdigest() == expected, 'sealed input '+key)
        docs[key] = json.loads(body)
    r, c, p = (docs[k] for k in ('rgb', 'receipt', 'physical'))
    require(len(r['rows']) == len(c['rows']) == 12 and len(p['rows']) == 4, 'all12/4')
    values = []
    for i, (row, accepted) in enumerate(zip(r['rows'], c['rows'], strict=True)):
        physical = p['rows'][i//3]
        require(row['index'] == i and row['plan_index'] == i//3 and row['root_slot'] == i//6
                and row['checkpoint'] == (5, 10, 15)[i%3] and row['arm'] == physical['arm'], 'fixed complete order')
        require(row['selection_digest'] == physical['selection_digest'], 'literal physical plan')
        ref = row['latent']; require(Path(ref['file']).name == ref['file'], 'flat latent')
        body = (ROOT/'results'/FILES['rgb'][0]).parent.joinpath(ref['file']).read_bytes()
        require(hashlib.sha256(body).hexdigest() == ref['sha256'], 'actual latent file')
        z = np.load(io.BytesIO(gzip.decompress(body)), allow_pickle=False)
        require(z.shape == (192,) and z.dtype == np.float32 and np.isfinite(z).all(), 'actual192 f32')
        pred, goal = row['predicted192'], row['goal192']
        require(mse(pred, goal) == row['model_Q'] == accepted['model_Q'], 'independent modelQ')
        require(mse(z, goal) == accepted['actual_encoded_Q'] and mse(z, pred) == accepted['prediction_vs_encoded_MSE'], 'independent encodedQ/MSE')
        response = physical['responses'][row['checkpoint']-1]
        require(response == row['response'] == accepted['response'], 'actual same-time native response')
        d = np.asarray(response['state'])-np.asarray(physical['goal'])
        joint, obj = float(np.linalg.norm(d[:4])), float(np.linalg.norm(d[2:4]))
        angle = float(abs(np.arctan2(np.sin(d[4]), np.cos(d[4]))))
        require(joint == accepted['native_metrics']['joint_xy_error'] and obj == accepted['native_metrics']['native']['object_position_error'], 'independent physical metrics')
        values.append(dict(model_Q=row['model_Q'], encoded_Q=mse(z, goal), joint_xy=joint, object_xy=obj, wrapped_angle=angle))
    pairs = []
    for root in (0, 1):
        for j, step in enumerate((5, 10, 15)):
            a, b = values[root*6+j], values[root*6+3+j]
            delta = {k:b[k]-a[k] for k in a}
            pairs.append(dict(root_slot=root, checkpoint=step, composition_minus_joint=delta,
                              model_to_encoded_reversal=bool(delta['model_Q']*delta['encoded_Q'] < 0),
                              encoded_to_joint_reversal=bool(delta['encoded_Q']*delta['joint_xy'] < 0)))
    result = dict(schema=PREFIX, fixed_pairs=6, fixed_rgb_slots=12, pairs=pairs,
                  sources={k:v[1] for k,v in FILES.items()}, new_neural_calls=0, new_physics_calls=0,
                  known_roots=2, fresh_roots=0, scientific_admission=False)
    output = Path(output).resolve()
    require(output.parent == ROOT/'results' and output.name.startswith(PREFIX+'_'), 'exclusive own result')
    with output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2); stream.write('\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
