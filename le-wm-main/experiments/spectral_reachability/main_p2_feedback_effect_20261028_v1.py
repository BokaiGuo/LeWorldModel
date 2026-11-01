"""Independent raw-array recomputation of the fixed eight P2 feedback plans."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import agent_p2_contextual_physical_rgb_encode_20261028_v1 as bridge

ROOT = Path(__file__).resolve().parents[2]
NAME = 'main_p2_feedback_effect_20261028_v1'
REPORT = ROOT/'results/agent_p2_contextual_physical_rgb_encode_20261028_v1_cuda_MAIN01/report.json'
REPORT_SHA = 'b57fa03560696c3c0b75fe842ec8263d6ec3961341d6736a4f6e782d1f0e92ea'
RECEIPT = ROOT/'results/agent_p2_contextual_physical_rgb_encode_20261028_v1_consume_MAIN01.json'
RECEIPT_SHA = '3d3d020c113959aec25f4e310b8512b814f7a78976f2bd7d5e58e0e435027f4b'
ARMS = ('joint_h2_no_composition', 'conditional_suffix_composition')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mse(a, b):
    return float(np.mean((np.asarray(a, np.float64)-np.asarray(b, np.float64))**2))


def agree(a, b):
    return None if a is None or b is None or a == 0 or b == 0 else (a > 0) == (b > 0)


def assemble(rows):
    require(len(rows) == 8, 'all eight plans')
    pairs = []
    for slot in range(4):
        left, right = rows[2*slot:2*slot+2]
        require((left['index'], right['index']) == (2*slot, 2*slot+1) and
                (left['arm'], right['arm']) == ARMS, 'fixed original plan order')
        require(all((r['root_slot'], r['condition']) == (slot//2, ('off', 'on')[slot % 2])
                    for r in (left, right)), 'same original root/condition pair')
        delta = {key: None if left[key] is None or right[key] is None else right[key]-left[key]
                 for key in ('original_Q', 'feedback_Q', 'actual_Q', 'joint_xy_error',
                             'object_xy_error', 'wrapped_angle_error')}
        pairs.append(dict(slot=slot, root_slot=slot//2, condition=('off', 'on')[slot % 2],
            composition_minus_joint=delta,
            original_to_actual_Q_order=agree(delta['original_Q'], delta['actual_Q']),
            feedback_to_actual_Q_order=agree(delta['feedback_Q'], delta['actual_Q']),
            actual_Q_to_jointXY_order=agree(delta['actual_Q'], delta['joint_xy_error']),
            actual_Q_to_objectXY_order=agree(delta['actual_Q'], delta['object_xy_error']),
            actual_Q_to_angle_order=agree(delta['actual_Q'], delta['wrapped_angle_error'])))
    return dict(rows=rows, pairs=pairs, fixed_plans=8, known_roots=2, fresh_roots=0,
        endpoint_MSE_improved=sum(r['feedback_MSE'] < r['original_MSE'] for r in rows
                                  if r['feedback_MSE'] is not None and r['original_MSE'] is not None),
        absolute_Q_error_improved=sum(abs(r['feedback_Q_error']) < abs(r['original_Q_error']) for r in rows
            if r['feedback_Q_error'] is not None and r['original_Q_error'] is not None),
        pair_order_counts={key: {label: sum(p[key] is value for p in pairs)
                                for label, value in (('agree', True), ('reverse', False), ('unknown_or_tie', None))}
            for key in ('original_to_actual_Q_order', 'feedback_to_actual_Q_order',
                        'actual_Q_to_jointXY_order', 'actual_Q_to_objectXY_order', 'actual_Q_to_angle_order')},
        native_success_plans=sum(r['native_success'] is True for r in rows),
        no_new_actions=True, no_physical_intervention=True, P1_admission=False, P2_admission=False,
        independent_confirmation=False, new_neural_calls=0, new_physics_calls=0)


def build():
    require(sha(REPORT) == REPORT_SHA and sha(RECEIPT) == RECEIPT_SHA, 'pinned actual CUDA and strict')
    report = json.loads(REPORT.read_bytes()); accepted = json.loads(RECEIPT.read_bytes())
    require(accepted['status'] == 'actual_RGB_and_same_suffix_saved_verified_NOT_science' and
            len(report['rows']) == len(accepted['rows']) == 8, 'completed actual full-eight strict snapshot')
    def load(ref):
        return bridge.rgb.load_array(ref, lambda name: bridge.rgb.flat(REPORT.parent, name).read_bytes())
    rows = []
    for raw, verified in zip(report['rows'], accepted['rows'], strict=True):
        goal = np.asarray(raw['goal192'], np.float32)
        frames = {f['step']: load(f['latent']) if f['latent'] is not None else None for f in raw['frames']}
        actual = frames[10]
        original = np.asarray(raw['imagined_terminal192'], np.float32)
        feedback = load(raw['feedback']['prediction'])[0, -1] if raw['feedback']['prediction'] is not None else None
        actual_q = None if actual is None else mse(actual, goal)
        q0 = mse(original, goal); q1 = None if feedback is None else mse(feedback, goal)
        metrics = dict(original_Q=q0, feedback_Q=q1, actual_Q=actual_q,
            original_MSE=None if actual is None else mse(original, actual),
            feedback_MSE=None if actual is None or feedback is None else mse(feedback, actual),
            original_Q_error=None if actual_q is None else q0-actual_q,
            feedback_Q_error=None if actual_q is None or q1 is None else q1-actual_q)
        for key, target in (('original_Q', 'original_predicted_Q'), ('feedback_Q', 'real_mid_same_suffix_predicted_Q'),
                            ('original_MSE', 'imagined_terminal_MSE'), ('feedback_MSE', 'real_mid_same_suffix_MSE'),
                            ('original_Q_error', 'imagined_Q_error'), ('feedback_Q_error', 'real_mid_same_suffix_Q_error')):
            require(metrics[key] == verified[target], 'independent original NPY scalar equality '+key)
        require(actual_q == verified['checkpoints'][2]['latent_Q'], 'same-time actual Q')
        root_error = None if frames[0] is None else mse(frames[0], raw['root_context']['latent_history'][-1])
        require(root_error == verified['original_root_reencode_MSE'], 'original root encoding arithmetic')
        native = verified['final_native_metrics']
        rows.append(dict(index=raw['index'], root_slot=raw['root_slot'], condition=raw['condition'], arm=raw['arm'],
            root_reencode_MSE=root_error, **metrics, native_success=native['native_success'],
            **{k: native[k] for k in ('joint_xy_error', 'object_xy_error', 'wrapped_angle_error')}))
    result = assemble(rows)
    result.update(schema=NAME, authority=dict(report=str(REPORT), report_sha256=REPORT_SHA,
        receipt=str(RECEIPT), receipt_sha256=RECEIPT_SHA, raw_NPY_recomputed=True,
        full_upstream_rebuilt_by_this_derivative=False),
        interpretation='Same fixed suffix predicted on actual RGB; not a new controller, non-greedy mechanism or independent efficacy result.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    args = parser.parse_args(); path = Path(args.output).absolute()
    require(path.resolve() == path and path.parent == ROOT/'results' and path.name.startswith(NAME+'_'), 'own exclusive output')
    value = build()
    with path.open('xb') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(path), sha256=sha(path), endpoint_MSE_improved=value['endpoint_MSE_improved'])))


if __name__ == '__main__':
    main()
