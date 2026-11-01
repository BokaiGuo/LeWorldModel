"""Compare saved local reach scores with the same candidates' model rollouts.

Exploratory, known-root, model-only analysis; this does not change the P1 gate.
Inputs must be externally hash anchored and the saved differential audit passes first.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p1_joint_vreach_saved_diff_20261029_v1 as audit

NAME = 'p1_saved_nonlinear_reach_diagnostic_20261029'


def score_candidate(goal, candidate):
    goal = np.asarray(goal, np.float64)
    signal = candidate['signal']
    nominal = np.asarray(signal['nominal_context']['latent_history'][-1], np.float64)
    endpoint = np.asarray(candidate['end']['latent_history'][-1], np.float64)
    jacobian = np.asarray(signal['jacobian'], np.float64)
    x = np.asarray(candidate['suffix_solve']['params'], np.float64)
    action = np.asarray(candidate['suffix'], np.float32).astype(np.float64).reshape(-1)
    if goal.shape != (192,) or nominal.shape != goal.shape or endpoint.shape != goal.shape or jacobian.shape != (192, 10) or x.shape != (10,) or action.shape != (10,):
        raise ValueError('invalid candidate shapes')
    if not all(np.isfinite(v).all() for v in (goal, nominal, endpoint, jacobian, x, action)):
        raise ValueError('nonfinite candidate')
    penalty = audit.DAMPING * float(x @ x) / goal.size
    raw_prediction = nominal + jacobian @ x
    sent_prediction = nominal + jacobian @ action
    def objective(z):
        return float(np.mean((goal-z)**2) + penalty)
    return dict(linear_logits=objective(raw_prediction), linear_sent=objective(sent_prediction),
                nonlinear_sent=objective(endpoint), penalty=penalty,
                linearization_error_l2=float(np.linalg.norm(endpoint-sent_prediction)),
                model_displacement_l2=float(np.linalg.norm(endpoint-nominal)),
                terminal_q=float(np.mean((goal-endpoint)**2)))


def analyze(report, receipt, prepared):
    audit.analyze(report, receipt, prepared)
    rows = []
    for arm_index, arm in enumerate(report['rows']):
        result = arm['result']
        goal = prepared['requests'][arm_index//2]['task']['goal']
        scores = [score_candidate(goal, c) for c in result['candidates']]
        base = scores[0]
        for candidate, score in zip(result['candidates'], scores, strict=True):
            if not np.isclose(score['terminal_q'], candidate['terminal_Q'], rtol=1e-10, atol=1e-12):
                raise ValueError('terminal score disagrees with saved rollout')
            if not np.isclose(score['linear_logits'], candidate['own_reach_value'], rtol=1e-10, atol=1e-12):
                raise ValueError('linear score disagrees with saved reach')
            delta = {key: score[key]-base[key] for key in ('linear_logits', 'linear_sent', 'nonlinear_sent')}
            rows.append(dict(slot=arm['slot'], arm=arm['arm'], start=candidate['start'], iteration=candidate['iteration'],
                             scores=score, delta_from_same_prefix_baseline=delta))
    nonbaseline = [r for r in rows if (r['start'], r['iteration']) != (0, 0)]
    return dict(schema=NAME, scope='known-root saved model rollouts; no new inference or physics; exploratory',
                summary=dict(candidates=len(rows), nonbaseline=len(nonbaseline),
                    **{key+'_better': sum(r['delta_from_same_prefix_baseline'][key] < 0 for r in nonbaseline)
                       for key in ('linear_logits', 'linear_sent', 'nonlinear_sent')},
                    linear_sent_vs_nonlinear_sign_disagree=int(sum(
                        np.sign(r['delta_from_same_prefix_baseline']['linear_sent']) !=
                        np.sign(r['delta_from_same_prefix_baseline']['nonlinear_sent']) for r in nonbaseline)),
                    max_linearization_error_l2=max(r['scores']['linearization_error_l2'] for r in rows)),
                rows=rows, P1_admission=False, freshness=False, physical_release=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'report-sha', 'receipt', 'receipt-sha', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    report = audit.read_verified(args.report, args.report_sha)
    receipt = audit.read_verified(args.receipt, args.receipt_sha)
    if receipt.get('report_sha256') != args.report_sha:
        raise ValueError('receipt report mismatch')
    prepared = audit.read_verified(report['prepared'], report['prepare_sha256'])
    result = analyze(report, receipt, prepared)
    result['source'] = dict(report_sha256=args.report_sha, receipt_sha256=args.receipt_sha,
                            prepare_sha256=report['prepare_sha256'],
                            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    root = Path(__file__).resolve().parents[2] / 'results'
    output = Path(args.output)
    if not output.is_absolute() or output.parent != root or output.resolve() != output or not output.name.startswith(NAME+'_'):
        raise ValueError('exclusive output path required')
    print(json.dumps(dict(path=str(output), sha256=audit.write(output, result), summary=result['summary'])))


if __name__ == '__main__':
    main()
