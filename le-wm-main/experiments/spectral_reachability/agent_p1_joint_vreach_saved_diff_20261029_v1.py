"""Read-only numerical differential on externally anchored P1 saved outputs.

No model import/execution, GPU, environment, physical action or threshold change.
Counterfactual scores are linear approximations, not new model predictions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np

NAME = 'agent_p1_joint_vreach_saved_diff_20261029_v1'
DAMPING = .002
CAP = float(np.sqrt(5)*.75)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def read_verified(path, expected):
    p = Path(path)
    if not p.is_file() or p.is_symlink(): raise ValueError('missing or symlink source')
    data = p.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected: raise ValueError('external SHA mismatch')
    return json.loads(data)


def independent_solve(j, r, damping, cap):
    """Gram eigensystem + monotone multiplier bisection; no producer SVD."""
    j, r = np.asarray(j, np.float64), np.asarray(r, np.float64)
    if j.ndim != 2 or r.shape != (j.shape[0],) or not np.isfinite(j).all() or not np.isfinite(r).all() or damping <= 0 or cap <= 0:
        raise ValueError('invalid ridge inputs')
    eigen, basis = np.linalg.eigh(j.T@j)
    if eigen.min() < -1e-9: raise ValueError('indefinite Gram matrix')
    eigen = np.maximum(eigen, 0.); coeff = basis.T@(j.T@r)
    def at(mu): return basis@(coeff/(eigen+damping+mu))
    x = at(0.)
    if np.linalg.norm(x) <= cap: return x, 0.
    lo, hi = 0., 1.
    while np.linalg.norm(at(hi)) > cap: hi *= 2
    for _ in range(100):
        mid = (lo+hi)/2
        if np.linalg.norm(at(mid)) > cap: lo = mid
        else: hi = mid
    return at(hi), hi


def compare(j, r, x, damping):
    j, r, x = np.asarray(j, np.float64), np.asarray(r, np.float64), np.asarray(x, np.float64)
    # Existing chart casts logits to float32 before tanh.
    sent = np.tanh(x.astype(np.float32)).astype(np.float32).astype(np.float64)
    def score(action, penalty):
        residual = r-j@action
        return float((residual@residual+damping*(penalty@penalty))/len(r))
    return dict(score_raw=score(x,x), score_tanh_raw=score(sent,x),
        score_tanh_raw_penalty=score(sent,sent), negative_sign_score=score(-x,x),
        raw_vs_tanh_l2=float(np.linalg.norm(x-sent)), raw_norm=float(np.linalg.norm(x)),
        sent_raw_norm=float(np.linalg.norm(sent)), sent_max_abs=float(np.max(np.abs(sent))))


def crossed_scores(base_j, base_r, candidate_j, candidate_r, damping, cap):
    def score(j, r):
        x, _ = independent_solve(j,r,damping,cap)
        return compare(j,r,x,damping)['score_raw']
    return dict(baseline_j_base_residual=score(base_j,base_r),
        baseline_j_candidate_residual=score(base_j,candidate_r),
        candidate_j_base_residual=score(candidate_j,base_r),
        candidate_j_candidate_residual=score(candidate_j,candidate_r))


def check_transition(task, previous, raw, after):
    mean = np.asarray(task['action_mean'],np.float32)
    std = np.asarray(task['action_std'],np.float32)
    actions = np.asarray(raw,np.float32)
    if mean.shape != (2,) or std.shape != (2,) or actions.shape != (5,2) or not np.isfinite(actions).all() or not np.isfinite(std).all() or (std<=0).any() or (np.abs(actions)>1).any():
        raise ValueError('action units or normalizer')
    token = ((actions-mean)/std).reshape(10)
    expected = np.stack((np.asarray(previous['action_history'][-1],np.float32),token))
    if not np.array_equal(expected,np.asarray(after['action_history'],np.float32)):
        raise ValueError('context action normalization mismatch')


def analyze(report, receipt, prepared):
    if report['status'] != 'complete_model_only' or receipt.get('complete') is not True or receipt.get('status') != 'verified_model_only':
        raise ValueError('completed saved authority required')
    if receipt['rows'] != report['rows'] or len(report['rows']) != 8 or len(prepared['requests']) != 4:
        raise ValueError('receipt/report roster mismatch')
    rows = []
    for i, arm in enumerate(report['rows']):
        q = prepared['requests'][i//2]; result = arm['result']
        if arm['status'] != 'sealed' or arm['slot'] != i//2 or q['slot'] != i//2 or result['arm'] != arm['arm']:
            raise ValueError('ordered slot/arm')
        candidates = result['candidates']
        if len(candidates) != 20: raise ValueError('candidate denominator')
        base = result['baseline_reach_value']
        if candidates[0]['start'] != 0 or candidates[0]['iteration'] != 0 or candidates[0]['own_reach_value'] != base:
            raise ValueError('same winner-prefix baseline')
        if not np.array_equal(np.asarray(candidates[0]['prefix'],np.float32),np.asarray(q['winner_commands'][:5],np.float32)):
            raise ValueError('baseline winner prefix identity')
        goal = np.asarray(q['task']['goal'], np.float64)
        base_j = np.asarray(candidates[0]['signal']['jacobian'],np.float64)
        base_r = goal - np.asarray(candidates[0]['signal']['nominal_context']['latent_history'][-1],np.float64)
        for index, c in enumerate(candidates):
            if (c['start'],c['iteration']) != (index//5,index%5): raise ValueError('candidate order')
            j = np.asarray(c['signal']['jacobian'], np.float64)
            nominal = np.asarray(c['signal']['nominal_context']['latent_history'][-1], np.float64)
            if j.shape != (192,10) or nominal.shape != (192,) or goal.shape != (192,): raise ValueError('signal dimensions')
            r = goal-nominal
            check_transition(q['task'],q['task']['root_context'],c['prefix'],c['mid'])
            check_transition(q['task'],c['mid'],[[0.,0.]]*5,c['signal']['nominal_context'])
            check_transition(q['task'],c['mid'],c['suffix'],c['end'])
            solve = c['suffix_solve']; x = np.asarray(solve['params'],np.float64)
            independent, mu = independent_solve(j,r,DAMPING,CAP)
            comp = compare(j,r,x,DAMPING)
            stationarity = np.linalg.norm(j.T@(j@x-r)+(DAMPING+solve['ball_multiplier'])*x)
            raw_actual = np.asarray(c['suffix'], np.float32).reshape(10)
            if not np.array_equal(raw_actual,np.tanh(x.astype(np.float32)).astype(np.float32)):
                raise ValueError('saved executed suffix chart mismatch')
            if not np.isclose(comp['score_raw'],solve['own_reach_value'],rtol=1e-11,atol=1e-12) or c['own_reach_value'] != solve['own_reach_value']:
                raise ValueError('saved score recomputation')
            if c['reach_better'] != (c['own_reach_value'] < base): raise ValueError('strict comparison mismatch')
            if c['stage_uphill'] != (c['stage_Q'] > result['initial_Q']) or c['terminal_better'] != (c['terminal_Q'] < result['baseline_terminal_Q']):
                raise ValueError('saved comparator mismatch')
            rows.append(dict(slot=i//2, arm=arm['arm'], start=c['start'], iteration=c['iteration'],
                baseline=base, reach=c['own_reach_value'], delta=c['own_reach_value']-base,
                reach_better=c['reach_better'], stage_uphill=c['stage_uphill'], terminal_better=c['terminal_better'],
                terminal_Q=c['terminal_Q'], baseline_terminal_Q=result['baseline_terminal_Q'],
                initial_Q=result['initial_Q'], stage_Q=c['stage_Q'],
                independent_params_error=float(np.linalg.norm(x-independent)), independent_multiplier_error=float(abs(mu-solve['ball_multiplier'])),
                stationarity_l2=float(stationarity), ball_slack=float(CAP-np.linalg.norm(x)),
                complementary_slack=float(abs(solve['ball_multiplier']*(np.linalg.norm(x)-CAP))),
                multiplier=solve['ball_multiplier'], residual_norm=float(np.linalg.norm(r)),
                jacobian_frobenius=float(np.linalg.norm(j)), jacobian_spectral=float(np.linalg.norm(j,2)),
                jacobian_delta_frobenius=float(np.linalg.norm(j-base_j)),
                crossed=crossed_scores(base_j,base_r,j,r,DAMPING,CAP),
                chart=comp))
    summary = dict(candidates=len(rows), reach_better=sum(r['reach_better'] for r in rows),
        stage_uphill=sum(r['stage_uphill'] for r in rows), terminal_better=sum(r['terminal_better'] for r in rows),
        active_ball=sum(r['multiplier']>0 for r in rows),
        max_independent_params_error=max(r['independent_params_error'] for r in rows),
        max_stationarity_l2=max(r['stationarity_l2'] for r in rows),
        max_raw_tanh_l2=max(r['chart']['raw_vs_tanh_l2'] for r in rows),
        min_delta=min(r['delta'] for r in rows), max_delta=max(r['delta'] for r in rows),
        best_nonbaseline_delta=min(r['delta'] for r in rows if (r['start'],r['iteration']) != (0,0)))
    return dict(schema=NAME+'.analysis', summary=summary, rows=rows,
        interpretation='saved-only linear differential; tanh counterfactual is not a model prediction; no physical or theoretical conclusion',
        inherited='externally anchored saved consumer receipt, not independent reexecution of native audit',
        P1_admission=False, physical_release=False, freshness=False)


def write(path, value):
    p = Path(path); data = canonical(value)
    with p.open('xb') as f:
        if f.write(data) != len(data): raise OSError('short write')
        f.flush(); os.fsync(f.fileno())
    fd = os.open(p.parent, os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('report','report-sha','receipt','receipt-sha','output'): parser.add_argument('--'+name,required=True)
    a=parser.parse_args()
    report=read_verified(a.report,a.report_sha); receipt=read_verified(a.receipt,a.receipt_sha)
    if receipt.get('report_sha256') != a.report_sha: raise ValueError('receipt report binding')
    prepared=read_verified(report['prepared'],report['prepare_sha256'])
    result=analyze(report,receipt,prepared)
    result['source'] = dict(report_sha256=a.report_sha,receipt_sha256=a.receipt_sha,prepare_sha256=report['prepare_sha256'],
                            diagnostic_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    root=Path(__file__).resolve().parents[2]/'results'; output=Path(a.output)
    if not output.is_absolute() or output.parent != root or output.resolve()!=output or not output.name.startswith(NAME+'_'):
        raise ValueError('exclusive own results path')
    print(json.dumps(dict(path=str(output),sha256=write(output,result),summary=result['summary'])))


if __name__ == '__main__': main()
