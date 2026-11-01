"""Independent native-state arithmetic for the frozen safeguarded all-eight run."""
import argparse
import json
from pathlib import Path

from experiments.spectral_reachability import main_midfeedback_physical_effect_20261028_v1 as base

ROOT = base.ROOT
NAME = 'main_safeguarded_physical_effect_20261028_v1'
PINS = ((base.OLD, base.PINS[0][1]),
    (ROOT/'results/agent_p2_contextual_midfeedback_recompile_20261028_v2_physical_MAIN01/report.json',
     '039e23ce87980e5a66f38cb47bc50c106690d17d2adb7169a4d4bed0adb70956'),
    (ROOT/'results/agent_p2_contextual_midfeedback_recompile_20261028_v2_cuda_MAIN01/report.json',
     '363925ff3714c6c9a5fbd2ec6c59e86dbcb63fe685cde1945d292ec1a9196117'))


def compare(old, new, pair, compiled):
    require = base.require
    require(old['index'] == new['index'] == pair['index'] == compiled['index'], 'same fixed index')
    require(all(old[k] == new[k] for k in ('root_slot', 'condition', 'arm', 'task_id', 'goal', 'root_state')),
            'same physical task and root')
    require(old['status'] == new['status'] == 'horizon_completed' and
            old['own_returned'] == new['own_returned'] == 10, 'complete actual10, no substitution')
    checkpoints = [[x for x in r['checkpoints'] if x['step'] == 5] for r in (old, new)]
    require(len(checkpoints[0]) == 1 and checkpoints[0] == checkpoints[1], 'exact common prefix5')
    metrics = [base.metrics(r['final_state'], r['goal']) for r in (old, new)]
    for m, strict in zip(metrics, (pair['control'], pair['intervention']), strict=True):
        require(all(m[k] == strict['final_metrics'][k] for k in (*base.AXES, 'native_success')),
                'independent native arithmetic equals full strict')
    delta = {k: metrics[1][k]-metrics[0][k] for k in base.AXES}
    require(all(delta[k] == pair['new_minus_control'][k] for k in base.AXES), 'paired differences')
    selected = compiled['selected_index']
    require(type(selected) is int and 0 <= selected < len(compiled['evaluations']) == 5, 'fixed five model candidates')
    old_q = compiled['evaluations'][0]['cost']
    new_q = compiled['evaluations'][selected]['cost']
    guard = pair['safeguard']
    require(guard['selected_index'] == selected and guard['incumbent_Q'] == old_q and
            guard['selected_Q'] == new_q and new_q <= old_q, 'strict model-only safeguard')
    return dict(index=new['index'], root_slot=new['root_slot'], condition=new['condition'], arm=new['arm'],
        same_prefix_exact=True, control=metrics[0], intervention=metrics[1], new_minus_control=delta,
        selected_index=selected, incumbent_Q=old_q, selected_Q=new_q, predicted_Q_delta=new_q-old_q)


def build(receipt, expected):
    for path, h in PINS:
        base.require(base.sha(path) == h, 'sealed actual report')
    base.require(base.sha(receipt) == expected, 'pinned full strict receipt')
    strict = json.loads(Path(receipt).read_bytes())
    base.require(strict['schema'] == 'agent_p2_contextual_midfeedback_recompile_20261028_v2.physical_consume' and
        strict['status'] == 'paired_known_intervention_recorded_NOT_science' and
        strict['physical_report_sha256'] == PINS[1][1] and strict['feedback_compile_sha256'] == PINS[2][1],
        'actual safeguarded physical authority')
    reports = [json.loads(p.read_bytes()) for p, _ in PINS]
    base.require(all(len(r['rows']) == 8 for r in (*reports, strict)), 'all eight, no filtering')
    rows = [compare(a, b, s, c) for a, b, s, c in zip(
        reports[0]['rows'], reports[1]['rows'], strict['rows'], reports[2]['rows'], strict=True)]
    return dict(schema=NAME, rows=rows, fixed_plans=8, known_roots=2, fresh_roots=0,
        axis_counts={k: dict(better=sum(r['new_minus_control'][k] < 0 for r in rows),
            worse=sum(r['new_minus_control'][k] > 0 for r in rows),
            tie=sum(r['new_minus_control'][k] == 0 for r in rows)) for k in base.AXES if k != 'native_reward'},
        control_success=sum(r['control']['native_success'] for r in rows),
        intervention_success=sum(r['intervention']['native_success'] for r in rows),
        modeled_strictly_better=sum(r['predicted_Q_delta'] < 0 for r in rows),
        authority=dict(reports={str(p): h for p, h in PINS}, receipt=str(Path(receipt).resolve()),
            receipt_sha256=expected, independent_state_arithmetic=True, full_upstream_rebuilt_here=False),
        P1_admission=False, P2_admission=False, independent_confirmation=False, equal_FLOPs=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('receipt', 'receipt-sha', 'output'):
        p.add_argument('--'+key, required=True)
    a = p.parse_args(); output = Path(a.output).absolute()
    base.require(output.resolve() == output and output.parent == ROOT/'results' and
                 output.name.startswith(NAME+'_') and output.suffix == '.json', 'own exclusive output')
    result = build(a.receipt, a.receipt_sha)
    with output.open('xb') as f:
        f.write(json.dumps(result, sort_keys=True, allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(output), sha256=base.sha(output), axis_counts=result['axis_counts'],
        control_success=result['control_success'], intervention_success=result['intervention_success'])))


if __name__ == '__main__':
    main()
