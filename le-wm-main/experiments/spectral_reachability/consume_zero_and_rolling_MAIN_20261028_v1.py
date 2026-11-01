"""Persist strict saved zero-time / independent rolling-history receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('zero', 'rolling'))
    args = parser.parse_args()
    if args.stage == 'zero':
        from experiments.spectral_reachability import agent_mixedmode_zero_time_control_20261028_v1 as module
        result = module.consume(ROOT/'results/agent_mixedmode_zero_time_control_20261028_v1_MAIN01',
            'dfba2b92f1e4c7a4f56a8d8168fca427c0595236a21588aea1a97cae0f897e8e')
    else:
        from experiments.spectral_reachability import agent_p4_rolling_saved_consume_20261028_v1 as module
        result = module.consume(ROOT/'results/agent_p4_moving_multiboundary_20261028_v1_MAIN02',
            '60489b7db3d4eff70d1be46d47d92209e6adcf333ae1d0a0e389b259ec1f6084',
            ROOT/'results/agent_p4_moving_multiboundary_20261028_v1_prepare_MAIN02.json',
            '8528e8d7f9b688b1c93be154305ccc35b8ed6218de63e736b1ae567691c432a2')
    destination = ROOT/'results'/('consume_zero_and_rolling_MAIN_20261028_v1_'+args.stage+'_MAIN01.json')
    body = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    if len(body) > 64*1024**2:
        raise ValueError('bounded complete saved receipt')
    with destination.open('xb') as stream:
        if stream.write(body) != len(body):
            raise OSError('short full receipt')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(destination), bytes=len(body),
        sha256=hashlib.sha256(body).hexdigest(), status=result['status'])))


if __name__ == '__main__':
    main()
