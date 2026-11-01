"""Persist full strict saved consumers, without model or physical execution."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('P3', 'P4', 'beam'))
    parser.add_argument('--report-sha', required=True)
    args = parser.parse_args()
    if args.stage == 'P3':
        stem = 'agent_p3_fixed_execution_physical_20261028_v2'
        folder = ROOT/'results'/(stem+'_MAIN01')
        module = importlib.import_module('experiments.spectral_reachability.'+stem)
        result = module.consume(folder, args.report_sha)
    elif args.stage == 'beam':
        stem = 'agent_kinodynamic_matched_physical_consume_20261028_v1'
        folder = ROOT/'results/agent_kinodynamic_matched_physical_20261028_v1_MAIN01'
        module = importlib.import_module('experiments.spectral_reachability.'+stem)
        result = module.consume(folder, args.report_sha)
    else:
        stem = 'agent_p4_moving_multiboundary_20261028_v1'
        folder = ROOT/'results'/(stem+'_MAIN02')
        module = importlib.import_module('experiments.spectral_reachability.'+stem)
        result = module.consume(folder, args.report_sha,
            ROOT/'results'/(stem+'_prepare_MAIN02.json'),
            '8528e8d7f9b688b1c93be154305ccc35b8ed6218de63e736b1ae567691c432a2')
    destination = ROOT/'results'/('consume_downstream_MAIN_20261028_v1_'+args.stage+'_MAIN01.json')
    body = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    if len(body) > 64*1024**2:
        raise ValueError('bounded full receipt, no silent truncation')
    with destination.open('xb') as stream:
        if stream.write(body) != len(body):
            raise OSError('short strict consumer receipt')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(stage=args.stage, output=str(destination), bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(), status=result.get('status'),
        scientific_admission=False)))


if __name__ == '__main__':
    main()
