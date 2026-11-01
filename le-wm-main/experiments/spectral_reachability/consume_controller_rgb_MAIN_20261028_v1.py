"""Persist the complete strict RGB8 receipt; saved arithmetic only."""
import os
import shutil

from experiments.spectral_reachability import agent_experiment1_common_controller_rgb_20261028_v1 as rgb


def main():
    output = rgb.ROOT / 'results/agent_experiment1_common_controller_rgb_20261028_v1_cuda_MAIN01'
    expected = '3f64f2b3f74be98fdd0f0eda36fb0c309990ae42d046a71ce4ca8d7289d654c9'
    destination = rgb.ROOT / 'results/agent_experiment1_common_controller_rgb_20261028_v1_consume_MAIN01.json'
    rgb.require(not destination.exists() and not destination.is_symlink(), 'NEW receipt only')
    result = rgb.consume(output, expected)
    payload = rgb.canonical(result)
    rgb.require(len(payload) <= 8 * 1024**2 and shutil.disk_usage(destination.parent).free >= rgb.FLOOR + len(payload), 'receipt cap/floor')
    with destination.open('xb') as stream:
        rgb.require(stream.write(payload) == len(payload), 'full receipt persisted')
        stream.flush()
        os.fsync(stream.fileno())
    rgb.arrays.syncdir(destination.parent)
    rgb.require(rgb.read(destination) == result, 'complete receipt roundtrip')
    print(rgb.canonical(dict(receipt=str(destination), sha256=rgb.sha(destination), rows=len(result['rows']),
                             scientific_admission=result['scientific_admission'],
                             metrics=[dict(root_slot=r['root_slot'], arm=r['arm'],
                                           actual_tracking_cost=r['actual_tracking_cost'],
                                           terminal_Q=r['checkpoints'][-1]['Q'],
                                           terminal_model_Q=r['checkpoints'][-1]['model_Q'],
                                           final_metrics=r['final_metrics']) for r in result['rows']])).decode())


if __name__ == '__main__':
    main()
