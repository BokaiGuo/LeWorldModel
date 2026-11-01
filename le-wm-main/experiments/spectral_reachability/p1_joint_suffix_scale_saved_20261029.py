"""Post-hoc fixed-scale comparison of saved model H1 task costs, not a selector."""
import argparse
import json
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p1_joint_suffix_fd_consume_20261029 as consumer
from experiments.spectral_reachability import p1_joint_suffix_fd_host_20261029 as producer
from experiments.spectral_reachability import agent_p1_joint_vreach_saved_diff_20261029_v1 as saved

NAME = 'p1_joint_suffix_scale_saved_20261029'


def costs(goal, row):
    goal = np.asarray(goal, np.float64)
    if goal.shape != (192,) or not np.isfinite(goal).all():
        raise ValueError('goal')
    def mse(z):
        z = np.asarray(z, np.float64)
        if z.shape != goal.shape or not np.isfinite(z).all():
            raise ValueError('endpoint')
        return float(np.mean((goal-z)**2))
    result = {'zero': mse(row['zero'])}
    for item in row['scales']:
        result[str(item['scale'])] = mse(item['plus'])
    return result


def analyze(data, report, prep):
    consumer.verify(data, report, prep)
    rows = []
    for row, (_, _, task, _) in zip(data['rows'], producer.candidates(report, prep), strict=True):
        scores = costs(task['goal'], row)
        small = min(scores[key] for key in ('0.001', '0.01', '0.1'))
        rows.append(dict(slot=row['slot'], iteration=row['iteration'], costs=scores,
                         best_small_minus_full=small-scores['1.0'],
                         full_minus_zero=scores['1.0']-scores['zero']))
    return dict(schema=NAME, rows=rows, summary=dict(rows=len(rows),
        small_better_than_full=sum(x['best_small_minus_full'] < 0 for x in rows),
        full_better_than_zero=sum(x['full_minus_zero'] < 0 for x in rows)),
        limitation='post-hoc model-only task costs; known roots; no penalty, selector, physics or P1 verdict',
        P1_admission=False, physical_release=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('result', 'sha256', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    data = saved.read_verified(args.result, args.sha256)
    report = saved.read_verified(producer.REPORT, producer.REPORT_SHA)
    receipt = saved.read_verified(producer.RECEIPT, producer.RECEIPT_SHA)
    if receipt['report_sha256'] != producer.REPORT_SHA:
        raise ValueError('receipt binding')
    prep = saved.read_verified(report['prepared'], report['prepare_sha256'])
    saved.analyze(report, receipt, prep)
    result = analyze(data, report, prep)
    result['input_sha256'] = args.sha256
    output = Path(args.output)
    if not output.is_absolute() or output.parent != producer.ROOT/'results' or output.resolve() != output or not output.name.startswith(NAME+'_'):
        raise ValueError('exclusive results path')
    print(json.dumps(dict(path=str(output), sha256=saved.write(output, result), summary=result['summary'])))


if __name__ == '__main__':
    main()
