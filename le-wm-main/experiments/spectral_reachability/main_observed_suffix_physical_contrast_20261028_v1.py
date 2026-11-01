"""Native7 arithmetic for observed suffix intervention, without selecting outcomes."""
import argparse
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_observed_suffix_physical_20261028_v1 as source
from experiments.spectral_reachability import main_terminal_witness_physical_contrast_20261028_v1 as arithmetic

NAME = 'main_observed_suffix_physical_contrast_20261028_v1'


def compare(a, b):
    source.require(all(a[k] == b[k] for k in ('index', 'root_slot', 'condition', 'task_id',
                   'root_state', 'goal', 'goal_pose', 'root_sha256', 'goal_pixel_sha256'))
                   and a['plan']['model_task_binding'] == b['plan']['model_task_binding'], 'same full task/root/native goal')
    ac = np.asarray(a['plan']['commands'], np.float32); bc = np.asarray(b['plan']['commands'], np.float32)
    source.require(ac.shape == bc.shape == (10, 2) and ac[:5].tobytes() == bc[:5].tobytes()
                   and b['prefix_reference_exact'] is True, 'same executed prefix5 and authenticated midpoint')
    ap = [c for c in a['checkpoints'] if c['step'] == 5]
    bp = [c for c in b['checkpoints'] if c['step'] == 5]
    source.require(len(ap) == len(bp) == 1 and all(ap[0][k] == bp[0][k]
                   for k in ('state', 'snapshot', 'file_sha256', 'pixel_sha256')), 'same actual cp5 state and RGB')
    # Native stops retain actual terminal and missing time, never pad to step10.
    summary = dict(index=b['index'], root_slot=b['root_slot'], condition=b['condition'],
                   old_returned_steps=a['own_returned'], new_returned_steps=b['own_returned'],
                   old_success=a['native_success'], new_success=b['native_success'],
                   observed_model_Q_delta=b['plan']['new_suffix_Q']-b['plan']['observed_incumbent_Q'],
                   comparable_step10=False, same_prefix5=True, metrics=None)
    if a['own_returned'] != 10 or b['own_returned'] != 10:
        summary['missing_reason'] = 'native stop: no common step10 outcome; terminal success retained'
        return summary
    metrics = arithmetic.pair(a['final_state'], b['final_state'], b['goal'])
    source.require(metrics['old']['native_success'] is a['native_success']
                   and metrics['new']['native_success'] is b['native_success']
                   and metrics['old']['native_distance'] == a['final_metrics']['native_distance']
                   and metrics['new']['native_distance'] == b['final_metrics']['native_distance'], 'independently reconstructed native labels')
    if ac.tobytes() == bc.tobytes():
        source.require(a['final_state'] == b['final_state'], 'same raw10 exact physical replay')
    summary.update(comparable_step10=True, metrics=metrics, missing_reason=None)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('producer-output', 'report-sha', 'receipt', 'receipt-sha', 'output'):
        p.add_argument('--'+k, required=True)
    a = p.parse_args(); out = Path(a.output).absolute()
    source.require(out.parent == source.ROOT/'results' and out.name.startswith(NAME+'_')
                   and not out.exists() and not out.is_symlink(), 'new exclusive analysis')
    source.require(source.sha(a.receipt) == a.receipt_sha, 'actual persisted physical receipt SHA')
    strict = source.consume(a.producer_output, a.report_sha)
    source.require(strict['status'] == source.RECORDED and source.raw(strict) == Path(a.receipt).read_bytes(),
                   'whole physical strict independently rebuilt, receipt exactly equal')
    report = source.read(Path(a.producer_output)/'report.json', source.META_CAP)
    prepared = source.read(report['execution_binding']['prepared'], source.META_CAP)
    compiled = source.read(Path(prepared['arguments']['compile_output'])/'report.json', source.META_CAP)
    compile_prepared = source.read(compiled['prepared'], source.META_CAP)
    _, old = source.reference(compile_prepared)
    rows = [compare(x, y) for x, y in zip(old['rows'], report['rows'], strict=True)]
    result = dict(schema=NAME, rows=rows, report_sha256=a.report_sha, receipt_sha256=a.receipt_sha,
                  fixed_conditions=4, known_roots=2, fresh_roots=0, full_upstream_rebuilt_here=True,
                  new_neural_calls=0, new_physics_calls=0, P1_admission=False, scientific_admission=False,
                  caveat='Fixed known-root suffix intervention, not independent confirmation or non-greedy C2.')
    with out.open('xb') as f:
        f.write(source.raw(result))
    print(source.raw(dict(output=str(out), sha256=source.sha(out), rows=rows)).decode())


if __name__ == '__main__':
    main()
