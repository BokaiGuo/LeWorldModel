"""Offline source mapping and two-root state machine; never dispatches a model."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import p2_h15_sra_prepare_20261029 as prep
from experiments.spectral_reachability import agent_p2_common_resource_grasp_cuda_20261029_v1 as common

NAME = 'p2_h15_source_lifecycle_20261029_v1'
HOST = prep.ROOT / 'results/agent_p2_intersection_host_cuda_20261029_v2_prepare_MAIN01.json'
HOST_SHA = 'd0c8856463b6846438655c52c59d504aa4fbfd7b8439decb7e1599610ec08fd8'
H15 = prep.ROOT / 'results/p2_h15_sra_prepare_20261029_MAIN02.json'
H15_SHA = 'f583f6ff1dc4db5614f20570a7f6498a77db6cc81a2123d6b29216b62914e556'
CHECKPOINT = Path('/home/gbk/.stable_worldmodel/checkpoints/pusht/lewm/weights.pt')
CONFIG = CHECKPOINT.with_name('config.json')
PRECISION = dict(matmul_TF32=False, cudnn_TF32=False, autocast=False, MATH=True,
                 FLASH=False, MEM_EFFICIENT=False, CUDNN=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build():
    h15 = prep.read(H15, H15_SHA)
    prep.need(h15 == prep.build(), 'H15 whole source rebuild')
    host = prep.read(HOST, HOST_SHA)
    tasks = common.join_tasks(host['slots'])
    roots = []
    for case, slot, task in zip(h15['cases'], host['slots'], tasks):
        prep.need(all(case[k] == slot[k] for k in ('root_slot', 'task_id', 'task_file', 'task_sha256')), 'root/task join')
        prep.need(case['root_context'] == task['root_context'] and case['goal'] == task['goal'] and
                  case['q'] == task['projection'] and case['warm_raw10'] == task['root_raw_action_history'], 'complete causal task join')
        prep.need(case['action_normalizer'] == {'mean': task['action_mean'], 'std': task['action_std']}, 'normalizer join')
        prep.need(case['model_sha256'] == task['model_sha256'] and case['model_config_sha256'] == task['model_config_sha256'], 'model task join')
        roots.append(dict(root_slot=case['root_slot'], task_id=case['task_id'], task_file=case['task_file'],
                          task_sha256=case['task_sha256'], physical_native_root=case['native_root']))
    bindings = host['bindings_sha256']
    for path, expected in ((CHECKPOINT, tasks[0]['model_sha256']), (CONFIG, tasks[0]['model_config_sha256'])):
        prep.need(path.is_file() and not path.is_symlink() and bindings[str(path)] == expected and sha(path) == expected, 'checkpoint/config file binding')
    prep.need(all(t['model_sha256'] == tasks[0]['model_sha256'] and t['model_config_sha256'] == tasks[0]['model_config_sha256'] for t in tasks), 'shared checkpoint')
    return dict(schema=NAME+'.manifest', h15_sha256=H15_SHA, historical_host_prepare_sha256=HOST_SHA,
                roots=roots, checkpoint_file_sha256=tasks[0]['model_sha256'], config_file_sha256=tasks[0]['model_config_sha256'],
                expected_state_dict_sha256=host['expected_model_hash'], precision_required=PRECISION,
                precision_source='historical host policy; runtime snapshot not checked',
                historical_source_inventory_sha256=hashlib.sha256(prep.encoded(bindings)).hexdigest(),
                historical_source_inventory_rehashed=False,
                mapping_code_sha256=sha(__file__),
                candidate_code_sha256=sha(prep.ROOT/'experiments/spectral_reachability/p2_h15_sra_host_candidate_20261029_v1.py'),
                runtime_model_verified=False,
                runtime_precision_verified=False, execution_authorized=False, physical_release=False,
                lifecycle_initial=['not_started', 'not_started'], fixed_denominator=2,
                note='offline mapping only; historical host prepare is a pinned input, not independently rebuilt here')


def lifecycle(events):
    state = ['not_started', 'not_started']
    for action, root in events:
        prep.need(type(root) is int and root in (0, 1), 'root')
        if action == 'start':
            prep.need(state[root] == 'not_started' and (root == 0 or state[0] == 'sealed'), 'ordered start')
            state[root] = 'running'
        elif action in ('seal', 'fail'):
            prep.need(state[root] == 'running', 'terminal only after start')
            state[root] = 'sealed' if action == 'seal' else 'failed'
        else:
            raise ValueError('unknown transition')
    return state


def write_new(path, value):
    path = Path(path)
    prep.need(path.name.startswith(NAME+'_') and value == build(), 'exclusive rebuilt manifest')
    data = prep.encoded(value)
    with path.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return hashlib.sha256(data).hexdigest()


def consume(path, expected):
    value = prep.read(Path(path), expected)
    prep.need(value == build(), 'manifest rebuild mismatch')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    if args.command == 'prepare':
        prep.need(args.sha256 is None, 'unexpected SHA')
        digest = write_new(args.output, build())
    else:
        prep.need(args.sha256 is not None, 'external SHA required')
        consume(args.output, args.sha256)
        digest = args.sha256
    print(json.dumps(dict(path=str(args.output), sha256=digest, execution_authorized=False)))


if __name__ == '__main__':
    main()
