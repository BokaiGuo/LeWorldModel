"""Persist full strict baseline consumption and descriptive model-cost table."""
import argparse
import json
from pathlib import Path

from experiments.spectral_reachability import agent_p2_equal144_baseline_cuda_20261028_v1 as native

NAME = 'main_p2_equal144_saved_20261028_v1'
OUTPUT = native.ROOT/'results/agent_p2_equal144_baseline_cuda_20261028_v1_cuda_MAIN01'
REPORT_SHA = '7eeaa059d58895545340458455ddcaabceda4a813a41051e1a2c60f22bc10228'


def analyze():
    receipt = native.consume(OUTPUT, REPORT_SHA)
    rows = []
    for slot in (0,1):
        plans = [p for p in receipt['plans'] if p['source_slot'] == slot]
        native.require([p['arm'] for p in plans] == list(native.ARMS), 'complete paired slots')
        rows.append(dict(source_slot=slot, task_id=plans[0]['task_id'],
            beam_model_cost=plans[0]['model_terminal_cost'], spectral_model_cost=plans[1]['model_terminal_cost'],
            spectral_minus_beam_model_cost=plans[1]['model_terminal_cost']-plans[0]['model_terminal_cost'],
            commands_equal=plans[0]['commands'] == plans[1]['commands'],
            physical_comparison=None))
    return dict(schema=NAME, native_strict_receipt=receipt, rows=rows,
        report_sha256=REPORT_SHA, independent_roots=False, known_roots=2,
        physical_executed=False, task_benefit=None, FullD2P_comparison=False,
        equal_total_compute=False, scientific_admission=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['analyze','verify'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    raw = native.canonical(result)
    if args.command == 'analyze':
        native.require(args.output.absolute().parent == native.ROOT/'results' and args.output.name.startswith(NAME), 'own result only')
        with args.output.open('xb') as stream:
            stream.write(raw)
    else:
        native.require(args.output.read_bytes() == raw, 'full saved recomputation')
    print(json.dumps(dict(status='strict_verified',sha256=native.sha(args.output),rows=result['rows'])))


if __name__ == '__main__':
    main()
