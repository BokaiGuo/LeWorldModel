"""Static, non-authorizing future P3 source/executor ABI handoff."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.spectral_reachability import agent_p3_capability_prospective_binding_20261029_v1 as binding

NAME = 'agent_p3_future_capability_abi_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
OLD_EXECUTOR_SHA = 'dac75c998e1a5ba5b1e8ab562cbaf02c04662844d884c3ae637ccbed3dbf25b4'
OLD_BRIDGE_SHA = '40d51e4d5ad5fce5db9417fecc639e0288f286b678a5e7b0eb4e22adea392b77'
ORDER = [[0, 'C'], [0, 'Z'], [0, 'P'], [1, 'Z'], [1, 'P']]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path, sha):
    binding.byte_hash(path, sha)
    def pairs(items):
        result = {}
        for k, v in items:
            need(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def build(proposal_path, proposal_sha, declaration_path, declaration_sha):
    p = read_json(proposal_path, proposal_sha)
    need(p.get('schema') == binding.NAME + '.proposal', 'proposal schema')
    for key in ('future_cohort', 'future_prepare', 'future_executor'):
        need(set(p[key]) == {'path', 'sha256'}, 'proposal descriptor')
    rebuilt = binding.proposal(p['run_id'], p['future_cohort']['path'], p['future_cohort']['sha256'],
        p['future_prepare']['path'], p['future_prepare']['sha256'],
        p['future_executor']['path'], p['future_executor']['sha256'])
    need(p == rebuilt, 'proposal reconstruction')
    d = read_json(declaration_path, declaration_sha)
    need(set(d) == {'schema', 'run_id', 'cohort', 'static_prepare', 'executor', 'bridge_source',
                    'endpoint_order', 'signal_count', 'endpoint_count', 'executor_input_schema'}, 'declaration fields')
    need(d['schema'] == NAME + '.declaration' and d['run_id'] == p['run_id'], 'declaration identity')
    need(d['cohort'] == p['future_cohort'] and d['static_prepare'] == p['future_prepare']
         and d['executor'] == p['future_executor'], 'exact precommit descriptor join')
    need(d['endpoint_order'] == ORDER and type(d['signal_count']) is int and d['signal_count'] == 7
         and type(d['endpoint_count']) is int and d['endpoint_count'] == 5, 'fixed ABI denominator')
    need(isinstance(d['executor_input_schema'], str) and 1 <= len(d['executor_input_schema']) <= 200,
         'executor input schema declaration')
    bridge = d['bridge_source']
    need(isinstance(bridge, dict) and set(bridge) == {'path', 'sha256'}, 'bridge descriptor')
    binding.byte_hash(bridge['path'], bridge['sha256'])
    need(d['executor']['sha256'] != OLD_EXECUTOR_SHA and bridge['sha256'] != OLD_BRIDGE_SHA,
         'historical pinned executors are not future adapters')
    closure = {}
    for f in (Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md')):
        closure[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    return {'schema': NAME+'.handoff', 'run_id': p['run_id'],
        'inputs': {'proposal': {'path': proposal_path, 'sha256': proposal_sha},
                   'declaration': {'path': declaration_path, 'sha256': declaration_sha}},
        'declared_abi': d, 'contract': binding.contract(), 'source_closure': closure,
        'status': 'blocked_unverified_future_authority_and_executor_abi',
        'missing': ['authenticated_future_cohort_and_source_selection', 'reviewed_future_bridge_and_executor_abi',
                    'future_bridge_prepare_report_completion_and_full_receipt', 'actual_task_and_history_binding',
                    'external_commitment_dispatch_order_and_launcher_enforcement'],
        'descriptor_bytes_verified': True, 'source_authority_verified': False,
        'executor_abi_verified': False, 'external_order_verified': False,
        'trusted_timestamp_proven': False, 'fresh_roots_certified': False,
        'execution_authorized': False, 'scientific_admission': False}


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=('build', 'verify'))
    for key in ('proposal', 'proposal-sha', 'declaration', 'declaration-sha', 'output', 'output-sha'):
        ap.add_argument('--'+key, required=key != 'output-sha')
    a = ap.parse_args()
    data = encode(build(a.proposal, a.proposal_sha, a.declaration, a.declaration_sha))
    out = Path(a.output)
    need(out.is_absolute() and out.resolve() == out and not out.is_symlink()
         and out.name.startswith(NAME+'_'), 'canonical own output')
    if a.command == 'verify':
        need(binding.byte_hash(out, a.output_sha) and out.read_bytes() == data, 'whole output mismatch')
    else:
        need(a.output_sha is None, 'output sha only for verify')
        with out.open('xb') as f:
            need(f.write(data) == len(data), 'short write')
            f.flush(); os.fsync(f.fileno())
        fd = os.open(out.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try: os.fsync(fd)
        finally: os.close(fd)
    print(hashlib.sha256(data).hexdigest())


if __name__ == '__main__': main()
