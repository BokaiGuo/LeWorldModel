"""Independent saved arithmetic: explain surrogate-vs-witness ranking, not P1."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NAME = 'main_selector_gap_decomposition_20261028_v1'
REPORT = ROOT/'results/agent_mixedmode_prefix_20261028_v1_cuda_MAIN01/report.json'
REPORT_SHA = 'd6166b98958af70cecb8529135396c77467b7c803bfe12a1a5de29c422c22ad2'
TERMS = ('stage', 'ridge', 'rounding_cost_change', 'nonlinear_cost_change')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(value, shape):
    a = np.asarray(value, dtype=np.float64)
    require(a.shape == shape and np.isfinite(a).all(), 'finite aligned array')
    return a


def close(x, y, message):
    require(np.isfinite([x, y]).all() and abs(x-y) <= 2e-12*max(1., abs(x), abs(y)), message)


def decompose(candidate, goal, q, root):
    """Exact identity includes signed cross terms, not magnitude-only attribution."""
    goal, root = array(goal, (192,)), array(root, (192,))
    q = array(q, (192, 192))
    j = array(candidate['jacobian'], (192, 10))
    nominal = array(candidate['nominal_terminal'], (192,))
    terminal = array(candidate['predicted_terminal'], (192,))
    prefix = array(candidate['predicted_prefix'], (192,))
    u = array(candidate['suffix_params'], (10,))
    rounded = u.astype(np.float32).astype(np.float64)
    raw = np.tanh(u.astype(np.float32)).reshape(5, 2)
    require(np.array_equal(raw, np.asarray(candidate['suffix_commands'])), 'literal float32 tanh commands')
    linear = q@(nominal+j@u-goal)
    rounding = q@(j@(rounded-u))
    nonlinear = q@(terminal-(nominal+j@rounded))
    d = 192
    linear_cost = float(linear@linear/d)
    rounding_cross = float(2*linear@rounding/d)
    rounding_square = float(rounding@rounding/d)
    nonlinear_cross = float(2*(linear+rounding)@nonlinear/d)
    nonlinear_square = float(nonlinear@nonlinear/d)
    rounding_change = rounding_cross+rounding_square
    nonlinear_change = nonlinear_cross+nonlinear_square
    stage = float(np.sum((q@(prefix-goal))**2)/d)
    terminal_cost = float(np.sum((q@(terminal-goal))**2)/d)
    ridge = float(.002*(u@u)/d)
    own_reach = linear_cost+ridge
    objective = stage+own_reach
    close(linear_cost, candidate['suffix_solve']['linear_terminal_cost'], 'linear residual score')
    close(ridge, candidate['suffix_solve']['ridge_penalty'], 'fixed damping penalty')
    close(own_reach, candidate['own_reach_value'], 'own ridge value')
    close(stage, candidate['stage_cost'], 'stage score')
    close(objective, candidate['objective'], 'original objective')
    close(terminal_cost, candidate['nonlinear_terminal_cost'], 'whole witness terminal score')
    close(terminal_cost, linear_cost+rounding_change+nonlinear_change, 'terminal telescoping identity')
    gap = stage+ridge-rounding_change-nonlinear_change
    close(objective-terminal_cost, gap, 'selector gap identity')
    root_cost = float(np.sum((q@(root-goal))**2)/d)
    return dict(index=candidate['index'], stage=stage, ridge=ridge, linear_cost=linear_cost,
        rounding_cross=rounding_cross, rounding_square=rounding_square,
        rounding_cost_change=rounding_change, nonlinear_cross=nonlinear_cross,
        nonlinear_square=nonlinear_square, nonlinear_cost_change=nonlinear_change,
        objective=objective, terminal_cost=terminal_cost, selector_gap=gap,
        root_cost=root_cost, predicted_prefix_worsens_root=stage>root_cost,
        param_float32_max_error=float(np.max(abs(rounded-u))),
        finite_step_latent_defect_norm=float(np.linalg.norm(nonlinear)),
        suffix_logit_norm=float(np.linalg.norm(u)))


def contrast(old, new):
    delta = {k:new[k]-old[k] for k in TERMS}
    old_change = new['objective']-old['objective']
    terminal_change = new['terminal_cost']-old['terminal_cost']
    contribution = dict(stage=delta['stage'], ridge=delta['ridge'],
        rounding=-delta['rounding_cost_change'], nonlinear=-delta['nonlinear_cost_change'])
    close(old_change-terminal_change, sum(contribution.values()), 'pairwise selector gap identity')
    return dict(old_index=old['index'], new_index=new['index'],
        objective_delta=old_change, terminal_delta=terminal_change,
        signed_gap_contributions=contribution,
        same_candidate=old['index']==new['index'],
        predicted_prefix_worsens_root=new['predicted_prefix_worsens_root'])


def build():
    require(REPORT.is_file() and not REPORT.is_symlink() and sha(REPORT)==REPORT_SHA, 'pinned actual CUDA report')
    report = json.loads(REPORT.read_bytes())
    prepare_path = Path(report['prepared'])
    require(prepare_path.is_file() and not prepare_path.is_symlink()
        and sha(prepare_path)==report['prepare_sha256'], 'producer-bound prepare')
    prepared = json.loads(prepare_path.read_bytes())
    require(report['status']=='all4_mixedmode_plans_sealed_not_admitted'
        and report['errors']==[] and report['model_unchanged'] is True
        and len(report['rows'])==len(prepared['slots'])==4, 'complete fixed denominator')
    rows = []
    for i, (row, slot) in enumerate(zip(report['rows'], prepared['slots'], strict=True)):
        require(row['index']==slot['index']==i and row['root_slot']==slot['root_slot']
            and row['condition']==slot['condition'], 'task identity')
        require(np.array_equal(slot['q'], np.eye(192)), 'original identity metric, not fitted')
        packet = slot['packet']
        candidates = row['result']['candidates']
        require(len(candidates)==222 and [c['index'] for c in candidates]==list(range(222)), 'all222 stable order')
        cells = [decompose(c, packet['goal'], slot['q'], packet['root_context']['latent_history'][-1]) for c in candidates]
        # Rank by literal producer scores, rather than silently changing ties via recomputation.
        original = min(candidates, key=lambda c:(c['objective'], c['index']))['index']
        terminal = min(candidates, key=lambda c:(c['nonlinear_terminal_cost'], c['index']))['index']
        greedy = min(candidates, key=lambda c:(c['stage_cost'], c['index']))['index']
        require(original==row['result']['winner']['index'], 'original exact winner')
        non_greedy = [c for c in cells if c['predicted_prefix_worsens_root']]
        best_ng = min(non_greedy, key=lambda c:(candidates[c['index']]['nonlinear_terminal_cost'], c['index'])) if non_greedy else None
        rows.append(dict(index=i, root_slot=row['root_slot'], condition=row['condition'],
            original_index=original, terminal_index=terminal, stage_greedy_index=greedy,
            terminal_vs_original=contrast(cells[original], cells[terminal]),
            best_predicted_non_greedy_vs_original=contrast(cells[original], best_ng) if best_ng else None,
            non_greedy_candidate_count=len(non_greedy), candidates=cells,
            max_abs_rounding_cost_change=max(abs(c['rounding_cost_change']) for c in cells),
            max_abs_nonlinear_cost_change=max(abs(c['nonlinear_cost_change']) for c in cells)))
    return dict(schema=NAME, rows=rows, inputs=dict(report=str(REPORT), report_sha256=REPORT_SHA,
        prepare=str(prepare_path), prepare_sha256=report['prepare_sha256']),
        candidate_denominator=888, known_roots=2, root_conditions=4, fresh_roots=0,
        formula='O-T=stage+ridge-rounding_cost_change-nonlinear_cost_change',
        scope='saved model-only exact arithmetic; signed components may cancel; no unique-cause assertion',
        new_neural_calls=0, new_physics=0, physical_outcomes_read=False,
        full_upstream_rebuilt_here=False, selector_changed=False, P1_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = build()
    output = Path(args.output).absolute()
    require(output.parent==ROOT/'results' and output.name.startswith(NAME+'_'), 'exclusive result path')
    with output.open('xb') as f:
        f.write(json.dumps(result, sort_keys=True, allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(output), sha256=sha(output),
        rows=[{k:v for k,v in row.items() if k!='candidates'} for row in result['rows']])))


if __name__=='__main__':
    main()
