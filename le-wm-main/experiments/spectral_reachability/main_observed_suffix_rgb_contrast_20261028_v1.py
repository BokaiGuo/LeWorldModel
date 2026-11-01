"""Same observed cp5, old/new suffix: predicted -> actual RGB -> native task.

Full new encoder consumer reconstructs all inherited authorities. No new models,
fitting, threshold search, root selection, or physics in this saved-array analysis.
"""
import argparse
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_observed_suffix_terminal_rgb_20261028_v1 as source
from experiments.spectral_reachability import main_terminal_witness_rgb_contrast_20261028_v1 as algebra
from experiments.spectral_reachability import main_observed_suffix_physical_contrast_20261028_v1 as physical

NAME = 'main_observed_suffix_rgb_contrast_20261028_v1'


def contrast(p0, p1, a0, a1, goal, old_physical, new_physical):
    vectors = list(map(algebra.vector, (p0, p1, a0, a1, goal)))
    p0, p1, a0, a1, goal = vectors
    pair = algebra.decompose(*vectors)
    task = physical.compare(old_physical, new_physical)
    source.require(task['comparable_step10'], 'both actual step10 observations required')
    return dict(old_predicted_Q=algebra.mse(p0, goal), new_predicted_Q=algebra.mse(p1, goal),
                old_actual_Q=algebra.mse(a0, goal), new_actual_Q=algebra.mse(a1, goal),
                old_endpoint_MSE=algebra.mse(p0, a0), new_endpoint_MSE=algebra.mse(p1, a1),
                native=task, **pair)


def build(directory, report_sha, receipt, receipt_sha):
    directory = Path(directory).absolute()
    source.require(source.sha(receipt) == receipt_sha, 'external saved new RGB receipt SHA')
    strict = source.consume(directory, report_sha)
    source.require(source.raw(strict) == Path(receipt).read_bytes(), 'whole new RGB strict rebuilt and persisted bytes equal')
    report = source.read(directory/'report.json', source.META_CAP)
    prep = source.read(report['prepared'], source.META_CAP)
    pp = source.read(prep['arguments']['physical_prepared'], source.META_CAP)
    native = source.read(Path(prep['arguments']['physical_output'])/'report.json', source.META_CAP)
    compiler = source.producer.compiler
    cr = source.read(Path(pp['arguments']['compile_output'])/'report.json', source.META_CAP)
    cp = source.read(cr['prepared'], source.META_CAP)
    _, old_native = source.producer.reference(cp)
    mr = source.read(Path(cp['arguments']['midpoint_output'])/'report.json', source.META_CAP)
    mp = source.read(mr['prepared'], source.META_CAP)
    old_dir = Path(mp['arguments']['terminal_output'])
    old_report = source.read(old_dir/'report.json', source.META_CAP)
    source.require(source.sha(old_dir/'report.json') == mp['arguments']['terminal_report_sha']
                   == prep['bindings'][str(old_dir/'report.json')], 'original literal terminal report in entire inherited closure')
    source.require(all(len(v) == 4 for v in (strict['rows'], report['rows'], prep['rows'], cp['rows'],
                   old_report['rows'], old_native['rows'], native['rows'])), 'entire four related conditions')
    rows = []
    for i, (s, r, accepted, compiled, old, a, b) in enumerate(zip(prep['rows'], report['rows'], strict['rows'],
            cp['rows'], old_report['rows'], old_native['rows'], native['rows'], strict=True)):
        source.require(s['index'] == r['index'] == accepted['index'] == compiled['index'] == old['index'] == i
                       and s['original_task'] == compiled['request']['task']
                       and all(s[k] == old[k] for k in ('root_slot', 'condition', 'goal192', 'model_task_binding')),
                       'same task/model/goal192/condition, distinct bound suffix actions')
        source.require(np.asarray(old['full10'], np.float32).tobytes() == np.asarray(a['plan']['commands'], np.float32).tobytes()
                       and np.asarray(s['full10'], np.float32).tobytes() == np.asarray(b['plan']['commands'], np.float32).tobytes(),
                       'old/new terminal encodings belong to each actual raw10')
        if r['latent'] is None:
            rows.append(dict(index=i, root_slot=s['root_slot'], condition=s['condition'], missing=True,
                             reason='genuine native stop before new cp10; no padding', P1_admission=False))
            continue
        z0 = algebra.latent(old_dir, old['latent']); z1 = algebra.latent(directory, r['latent'])
        source.require(np.array_equal(z0, mp['rows'][i]['actual_terminal192'])
                       and np.array_equal(z1, accepted['actual192'])
                       and s['source']['state'] == b['final_state'] and old['source']['state'] == a['final_state'],
                       'original NPY bytes/actual RGB/native time join')
        result = contrast(compiled['observed_incumbent_prediction']['latent_history'][-1], s['predicted192'],
                          z0, z1, s['goal192'], a, b)
        source.require(result['old_predicted_Q'] == compiled['observed_incumbent_Q']
                       and result['new_predicted_Q'] == accepted['predicted_Q']
                       and result['old_actual_Q'] == mp['rows'][i]['actual_terminal_Q']
                       and result['new_actual_Q'] == accepted['actual_Q'], 'independent costs equal original sealed authorities')
        rows.append(dict(index=i, root_slot=s['root_slot'], condition=s['condition'], missing=False, **result))
    source.require(source.sha(directory/'report.json') == report_sha and source.sha(receipt) == receipt_sha, 'final input seals unchanged')
    return dict(schema=NAME, rows=rows, full_upstream_rebuilt_here=True, report_sha256=report_sha,
                receipt_sha256=receipt_sha, fixed_denominator=4, known_roots=2, fresh_roots=0,
                new_neural_calls=0, new_physics_calls=0, P1_admission=False, scientific_admission=False,
                caveat='Same-cp5 suffix intervention; signed error identity is descriptive, not fitted correction or unique-cause proof.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('producer-output', 'report-sha', 'receipt', 'receipt-sha', 'output'):
        p.add_argument('--'+k, required=True)
    a = p.parse_args(); out = Path(a.output).absolute()
    source.require(out.parent == source.ROOT/'results' and out.name.startswith(NAME+'_')
                   and not out.exists() and not out.is_symlink(), 'new exclusive analysis')
    value = build(a.producer_output, a.report_sha, a.receipt, a.receipt_sha)
    with out.open('xb') as f:
        f.write(source.raw(value))
    print(source.raw(dict(output=str(out), sha256=source.sha(out), rows=value['rows'])).decode())


if __name__ == '__main__':
    main()
