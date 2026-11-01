"""Independent saved-array arithmetic after full real midpoint consumption."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import main_nonlinear_suffix_midpoint_effect_20261028_v1 as primary
from experiments.spectral_reachability import main_nonlinear_suffix_history_control_20261028_v1 as history

NAME = 'main_nonlinear_suffix_midpoint_actual_analysis_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_row(spec, row, original_mid, accepted, load):
    primary.require(spec['eligible'] is True and row['status'] == 'predicted',
                    'actual known four-condition cohort fully available, no dropping')
    primary.require(all(spec[k] == row[k] for k in ('index', 'root_slot', 'condition',
                    'task_sha256', 'selection_digest', 'raw_float32_sha256', 'goal_sha256')),
                    'same literal task/action/goal/time panel')
    observed = load(row['actual_full3'])
    prediction = load(row['prediction'])
    primary.require(observed.shape == prediction.shape == (1, 3, 192), 'actual full3 buffers')
    p, q = spec['predicted192'], prediction[0, -1]
    a, g = spec['actual_terminal192'], spec['goal192']
    result = primary.compare(p, q, a, g)
    primary.require(result['imagined_predicted_Q'] == accepted['imagined_final_Q']
                    and result['observed_mid_predicted_Q'] == accepted['conditioned_final_Q']
                    and result['actual_Q'] == accepted['actual_final_Q']
                    and result['imagined_endpoint_MSE'] == accepted['imagined_endpoint_MSE']
                    and result['observed_mid_endpoint_MSE'] == accepted['conditioned_endpoint_MSE']
                    and np.array_equal(q, accepted['conditioned_final192']),
                    'independent native NPY costs equal whole strict receipt')
    control = history.history_diagnostics(spec['original_root_context']['latent_history'],
                                        original_mid['latent_history'], observed[0])
    return dict(index=spec['index'], root_slot=spec['root_slot'], condition=spec['condition'],
                history_control=control, **result)


def build(output, report_sha, receipt, receipt_sha, source_sha):
    # Import only a complete published worker module, authenticate its literal source.
    from experiments.spectral_reachability import agent_nonlinear_suffix_midpoint_rgb_20261028_v1 as source
    primary.require(sha(source.__file__) == source_sha, 'externally supplied frozen midpoint source')
    output, receipt = Path(output).resolve(), Path(receipt).resolve()
    primary.require(sha(output/'report.json') == report_sha and sha(receipt) == receipt_sha,
                    'external actual report and strict receipt SHA')
    strict = source.consume(output, report_sha)
    persisted = source.read(receipt, source.producer.PREP_CAP)
    primary.require(source.raw(strict) == source.raw(persisted), 'entire actual strict rebuilt here')
    report = source.read(output/'report.json', source.producer.PREP_CAP)
    prep = source.read(report['prepared'], source.producer.PREP_CAP)
    auth = prep['physical_authority']
    original_path = Path(auth['producer_output'])/'report.json'
    primary.require(sha(original_path) == auth['producer_report_sha256'], 'actual full168 model report')
    original = source.read(original_path, source.suffix.contract()['final_reserve_bytes'])
    primary.require(len(original['rows']) == 168 and len(prep['rows']) == len(report['rows']) == len(strict['rows']) == 4,
                    'all original168 and actual four conditions')
    def load(ref):
        return source.bridge.rgb.load_array(ref, lambda n: source.bridge.rgb.flat(output, n).read_bytes())
    rows = []
    for i, (spec, row, accepted) in enumerate(zip(prep['rows'], report['rows'], strict['rows'], strict=True)):
        index = spec['selected_index']
        original_row = original['rows'][42*i+index]
        primary.require(original_row['slot'] == i and original_row['prefix_index'] == index
                        and original_row['result']['index'] == index
                        and original_row['result']['commands'] == spec['full10'],
                        'exact original chosen prefix/raw10, not best hindsight replacement')
        rows.append(compare_row(spec, row, original_row['result']['mid'], accepted, load))
    value = primary.summarize(rows)
    primary.require(sha(source.__file__) == source_sha and sha(output/'report.json') == report_sha
                    and sha(receipt) == receipt_sha, 'source/results unchanged during independent analysis')
    return dict(schema=NAME, **value,
                inputs=dict(report_sha256=report_sha, receipt_sha256=receipt_sha,
                            source_sha256=source_sha, full168_report_sha256=auth['producer_report_sha256'],
                            analysis_protocol_sha256=prep['contract']['analysis_protocol_sha256']),
                full_upstream_rebuilt_here=True, new_neural_calls=0, new_physics_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('midpoint-output', 'report-sha', 'receipt', 'receipt-sha', 'source-sha', 'output'):
        parser.add_argument('--'+key, required=True)
    a = parser.parse_args()
    path = Path(a.output).absolute()
    primary.require(path.resolve() == path and path.parent == ROOT/'results'
                    and path.name.startswith(NAME+'_') and path.suffix == '.json', 'exclusive own output')
    value = build(a.midpoint_output, a.report_sha, a.receipt, a.receipt_sha, a.source_sha)
    payload = json.dumps(value, sort_keys=True, allow_nan=False).encode()+b'\n'
    with path.open('xb') as f:
        primary.require(f.write(payload) == len(payload), 'whole independent report')
        f.flush(); os.fsync(f.fileno())
    print(json.dumps(dict(output=str(path), sha256=sha(path), rows=value['rows'])))


if __name__ == '__main__':
    main()
