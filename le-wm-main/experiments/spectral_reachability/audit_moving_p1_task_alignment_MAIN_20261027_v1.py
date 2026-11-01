"""MAIN saved-array prediction/RGB/native alignment; no inference or selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

NAME = 'audit_moving_p1_task_alignment_MAIN_20261027_v1'
ROOT = Path(__file__).resolve().parents[2]
INPUTS = {
    'sampling_physical': ('moving_prediction_matched_sampling_physical_20261027_v1_MAIN01/report.json', '8ce8634eddcd3e2e5b25d5b55572d80d06da1fd5d3513ee06aecec89001efa41'),
    'sampling_rgb': ('moving_prediction_matched_sampling_rgb_cuda_20261027_v2_MAIN01/report.json', '63c48c16b91d16b8f20c5f6ba08010d366544558f6e7eb9dd7a9542a5e50fa2f'),
    'sampling_planner': ('moving_prediction_matched_sampling_cuda_20261027_v1_cuda_MAIN01/report.json', '60872ad114700731d82d293a616a4dede54a53b7be6514ad0cdf231d838d3ed8'),
    'overlay_bridge': ('moving_overlay_task_metric_bridge_20261027_v1_MAIN01.json', '2fa115f012dadebcca8bedefb6b6e65396f17042945d7f89c84a470a1840e045'),
    'overlay_rgb': ('p1_moving_overlay_recompile_rgb_cuda_20261027_v1_MAIN01/report.json', '18d2f192a75e397c01e6c7eb7f6422e9bb073af865107ef469debe5b8ae32d1e'),
    'overlay_planner': ('p1_moving_overlay_recompile_cuda_20261027_v1_cuda_MAIN01/report.json', '2b2e5064f74f8aca712ae0d0415fcbfb64a2c8e6325bdffe24ae4fdc3b373381'),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def alignment(prediction, actual, goal, predicted_cost, actual_cost):
    """Independent NumPy mean192 and exact residual/cross-term identity."""
    arrays = [np.asarray(x, dtype=np.float64) for x in (prediction, actual, goal)]
    require(all(x.shape == (192,) and np.isfinite(x).all() for x in arrays), 'finite native192 vectors')
    pred, real, target = arrays
    model_q = float(np.mean((pred-target)**2))
    real_q = float(np.mean((real-target)**2))
    require(np.isclose(model_q, predicted_cost, rtol=1e-12, atol=1e-12), 'saved predicted mean192 Q')
    require(real_q == actual_cost, 'saved actual mean192 Q')
    gap = real-pred
    mse = float(np.mean(gap**2))
    cross = float(2*np.mean((pred-target)*gap))
    require(np.isclose(real_q-model_q, mse+cross, rtol=1e-12, atol=1e-12), 'residual algebra')
    return dict(predicted_Q=model_q, actual_Q=real_q, prediction_MSE=mse,
                actual_minus_predicted_Q=real_q-model_q, error_square_term=mse,
                error_goal_cross_term=cross)


def run(output):
    # These imports expose saved consumers only. No run/model/physics is called.
    from experiments.spectral_reachability import moving_prediction_matched_sampling_physical_20261027_v1 as physical
    from experiments.spectral_reachability import moving_prediction_matched_sampling_rgb_cuda_20261027_v2 as rgb
    from experiments.spectral_reachability import moving_prediction_matched_sampling_cuda_20261027_v1 as sampling
    from experiments.spectral_reachability import p1_moving_overlay_recompile_rgb_cuda_20261027_v1 as overlay_rgb
    from experiments.spectral_reachability import p1_moving_overlay_recompile_cuda_20261027_v1 as overlay
    from experiments.spectral_reachability import p1_moving_overlay_recompile_comparison_20261027_v1 as metrics
    paths = {k: ROOT/'results'/v[0] for k, v in INPUTS.items()}
    def read(path):
        return json.loads(path.read_bytes())
    for k, p in paths.items():
        require(sha(p.read_bytes()) == INPUTS[k][1], 'external frozen input SHA: '+k)
    ph = physical.consume(str(paths['sampling_physical'].parent), INPUTS['sampling_physical'][1])
    rr = rgb.consume(str(paths['sampling_rgb'].parent), INPUTS['sampling_rgb'][1])
    sp = sampling.consume(str(paths['sampling_planner'].parent), INPUTS['sampling_planner'][1])
    op = overlay.consume(str(paths['overlay_planner'].parent), INPUTS['overlay_planner'][1])
    oc = overlay_rgb.consume(str(paths['overlay_rgb'].parent), INPUTS['overlay_rgb'][1])
    sr, oreport, bridge = read(paths['sampling_rgb']), read(paths['overlay_rgb']), read(paths['overlay_bridge'])
    physical_rows = {(r['root_slot'], r['arm']): r for r in ph['rows']}
    sampling_plans = {(r['source_slot'], r['arm']): r for r in sp['rows']}
    sampling_rows = {(r['root_slot'], r['arm']): r for r in rr['rows']}
    overlay_plans = {(r['root_slot'], r['condition'], r['arm']): r for r in op}
    overlay_rows = {(r['root_slot'], r['condition'], r['arm']): r for r in oc}
    allrows = []
    def load(receipt, parent):
        return overlay_rgb.load_array(receipt, lambda name: (parent/name).read_bytes())
    for b in bridge['rows']:
        k = b['root_slot'], b['condition'], b['arm']
        plan, scored = overlay_plans[k], overlay_rows[k]
        require(b['selection_digest'] == scored['selection_digest'] == plan['selection_digest'], 'overlay same selected plan')
        item = next(x for x in oreport['rows'] if (x['root_slot'], x['condition'], x['arm']) == k)
        cp = scored['checkpoints'][-1]
        target = load(oreport['images'][item['goal_image']]['latent'], paths['overlay_rgb'].parent)
        real = load(cp['latent'], paths['overlay_rgb'].parent)
        a = alignment(plan['prediction']['latent_history'][-1], real, target, plan['model_terminal_cost'], cp['Q'])
        state = b['checkpoints'][-1]['source_checkpoint']['state']
        allrows.append(dict(root_slot=k[0], condition=k[1], arm=k[2], root_sha256=b['root_sha256'],
                            goal=b['goal'], state=state, native=metrics.metrics(b['goal'], state),
                            selection_digest=plan['selection_digest'], initial_Q=scored['checkpoints'][0]['Q'],
                            **a, prediction_budget_matched_to_sampling=False))
    for k, b in physical_rows.items():
        plan, scored = sampling_plans[k], sampling_rows[k]
        require(b['selected_plan']['compiler_selection_sha256'] == plan['selection_sha256'] and
                b['selection_digest'] == scored['selection_digest'], 'sampling same selected raw plan')
        cp = scored['checkpoints'][-1]
        target = load(sr['images'][scored['goal_image']]['latent'], paths['sampling_rgb'].parent)
        real = load(cp['latent'], paths['sampling_rgb'].parent)
        a = alignment(plan['prediction']['latent_history'][-1], real, target, plan['model_terminal_cost'], cp['Q'])
        references = [r for r in allrows if r['root_slot'] == k[0] and r['condition'] == 'off'
                      and r['arm'] in ('line_search_GN', 'full10_conditional_D2P')]
        require(len(references) == 2 and all(r['root_sha256'] == b['selected_plan']['root_sha256']
                and r['goal'] == b['goal'] and r['initial_Q'] == scored['checkpoints'][0]['Q'] for r in references),
                'same physical root/goal/actual initial RGB Q across methods')
        allrows.append(dict(root_slot=k[0], condition='off', arm=k[1], root_sha256=b['selected_plan']['root_sha256'],
                            goal=b['goal'], state=b['final_state'], native=metrics.metrics(b['goal'], b['final_state']),
                            selection_digest=b['selection_digest'], initial_Q=scored['checkpoints'][0]['Q'],
                            **a, predict_API_calls=144, predictor_samples=144))
    require(len(allrows) == 12, 'all eight overlay plus four sampling rows')
    off = [r for r in allrows if r['condition'] == 'off']
    require(len(off) == 8 and sum(r['native']['native_success'] for r in allrows) == 0, 'complete observed native outcome denominator')
    result = dict(schema=NAME+'.report', status='saved_prediction_task_alignment_recorded', rows=allrows,
                  known_root_denominator=2, independent_root_count=0,
                  same_prediction_budget='CEM versus local nominal-MPPI only; GN/D2P unmatched',
                  new_neural_calls=0, new_physics_calls=0, new_selection=False, P1_admission=False,
                  limits=['Native state distance contains velocity; native success does not.',
                          'No relative performance or significance claim from two reused roots.',
                          'Not a non-greedy or capability-changing mechanism test.'],
                  inputs={k: dict(file=str(p), sha256=INPUTS[k][1]) for k, p in paths.items()},
                  source_sha256=sha(Path(__file__).read_bytes()))
    for k, p in paths.items():
        require(sha(p.read_bytes()) == INPUTS[k][1], 'final input drift: '+k)
    metrics.write_new(str(Path(output).absolute()), result)
    print(json.dumps(dict(status=result['status'], rows=len(allrows),
                          sha256=sha(Path(output).read_bytes()), P1_admission=False)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
