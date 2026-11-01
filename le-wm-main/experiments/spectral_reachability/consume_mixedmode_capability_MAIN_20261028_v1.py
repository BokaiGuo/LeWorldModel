"""Persist the actual full physical consumer receipt, not a summary substitute."""
import os
from pathlib import Path

from experiments.spectral_reachability import agent_mixedmode_capability_physical_20261028_v1 as physical


def main():
    folder = physical.ROOT/'results/agent_mixedmode_capability_physical_20261028_v1_MAIN01'
    report_sha = '57ac436ff6cd7fcffcda62e0a165782b719c0383404647f2f87c1fa97ab009c8'
    destination = physical.ROOT/'results/consume_mixedmode_capability_MAIN_20261028_v1_MAIN01.json'
    if destination.exists():
        raise FileExistsError(destination)
    receipt = physical.consume(folder, report_sha)
    payload = physical.canonical(dict(schema='MAIN.mixedmode_capability_acceptance.v1',
        receipt=receipt, source_sha256=physical.sha(Path(__file__)),
        new_neural_calls=0, new_physics_calls=0, P1_admission=False,
        independent_confirmation=False, time_confound=True))
    with destination.open('xb') as stream:
        if stream.write(payload) != len(payload):
            raise OSError('short receipt write')
        stream.flush()
        os.fsync(stream.fileno())
    physical.adapter.paired.fsync_dir(destination.parent)
    print(physical.canonical(dict(output=str(destination), sha256=physical.sha(destination),
        status=receipt['status'], counts=receipt['counts'], denominator=receipt['fixed_denominator'],
        errors=receipt['errors'])).decode())


if __name__ == '__main__':
    main()
