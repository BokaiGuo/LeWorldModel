"""Persist complete new strict saved consumer of unchanged fixed5 execution."""
import hashlib
import json
import os

from experiments.spectral_reachability import agent_p3_fixed_execution_saved_20261028_v1 as consumer


def main():
    result = consumer.consume(consumer.ACTUAL_REPORT.parent, consumer.ACTUAL_REPORT_SHA)
    destination = consumer.ROOT/'results/consume_p3_native_saved_MAIN_20261028_v1_MAIN01.json'
    payload = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    if len(payload) > 64*1024**2:
        raise ValueError('complete receipt bounded, no truncation')
    with destination.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise OSError('short full receipt')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(output=str(destination), bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(), status=result['status'],
        outcomes=result['outcomes'], scientific_admission=False)))


if __name__ == '__main__':
    main()
