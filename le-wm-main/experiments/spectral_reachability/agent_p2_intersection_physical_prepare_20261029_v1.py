"""P2 intersection v2 six-plan saved-only physical preparation; no native runner."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np

from experiments.spectral_reachability import agent_p2_intersection_host_cuda_20261029_v2 as producer
from experiments.spectral_reachability import agent_p2_common_resource_physical_20261029_v1 as physical

NAME = 'agent_p2_intersection_physical_prepare_20261029_v1'
ROOT = physical.ROOT
ORDER = physical.ORDER
read, sha, raw, require = physical.read, physical.sha, physical.raw, physical.require


def assemble(compiled, receipt, roots, report_sha, receipt_sha):
    """Decode exactly six fixed exports without model or physical callbacks."""
    require(receipt['schema'] == producer.NAME+'.consume' and receipt['status'] == 'all6_joint_box_saved_verified_not_scientific_admission'
        and receipt['report_sha256'] == report_sha and receipt['fixed_denominator'] == 6
        and len(receipt['plans']) == len(receipt['accounts']) == 6, 'actual v2 complete receipt')
    require(compiled['schema'] == producer.NAME+'.prepare' and compiled['root_arm_order'] == [[r,a] for r,_,a in ORDER]
        and len(compiled['slots']) == len(roots) == 2, 'exact frozen original order and roots')
    require(receipt['new_neural_calls'] == receipt['new_physics_calls'] == receipt['training'] == 0
        and all(receipt[k] is False for k in ('physical_release','P1_admission','P2_mechanism','equal_FLOPs','actual_cost_equality')),
        'original admission and cost boundaries')
    plans = []
    for i,(root,condition,arm) in enumerate(ORDER):
        witness = receipt['plans'][i]; account = receipt['accounts'][i]; slot = compiled['slots'][root]
        require(account['root'] == root and account['arm'] == arm, 'fixed account identity')
        if witness is None:
            plans.append(None); continue
        require(set(witness) == {'logits','contexts'} and len(witness['contexts']) == 3, 'literal export witness')
        request = dict(root_context=slot['packet']['root_context'], goal=slot['packet']['goal'],
                       q=slot['q'], normalization=physical.domain.normalization())
        decoded = physical.domain.raw15_inputs(request,witness['logits'])
        previous = request['root_context']
        for j,context in enumerate(witness['contexts']):
            physical.domain.core.context(context)
            expected = np.concatenate((np.asarray(previous['action_history'],np.float32),
                np.asarray(decoded['normalized_packets'][j:j+1],np.float32)))[-2:]
            require(np.asarray(context['latent_history'],np.float32)[:-1].tobytes() == np.asarray(previous['latent_history'],np.float32)[1:].tobytes()
                and np.asarray(context['action_history'],np.float32).tobytes() == expected.tobytes(), 'exact full3/action2 replay chain')
            previous = context
        a = np.asarray(decoded['raw15'],np.float32)
        plan = dict(index=i,root_slot=root,condition=condition,arm=arm,task_id=slot['task_id'],
            task_file=slot['task_file'],task_sha256=slot['task_sha256'],root_context=request['root_context'],
            goal=request['goal'],q=request['q'],warm_raw10=slot['warm_raw10'], raw15=decoded['raw15'],
            raw_float32_sha256=hashlib.sha256(a.tobytes()).hexdigest(), raw_float32_bits=a.tobytes().hex(),
            logits=deepcopy(witness['logits']),contexts=deepcopy(witness['contexts']),native_root=deepcopy(roots[root]),
            selection_uses_physics=False)
        plans.append(plan)
    missing = [i for i,x in enumerate(plans) if x is None]
    require(receipt['missing_plan_indices'] == missing, 'all six missing slots preserved')
    return dict(schema=NAME+'.prepare', status='saved_plan_preparation_only_not_physical_release',
        report_sha256=report_sha, receipt_sha256=receipt_sha, fixed_denominator=6,
        plans=plans, missing_plan_indices=missing, partial_arm_indices=deepcopy(receipt['partial_arm_indices']),
        accounts=deepcopy(receipt['accounts']), native_physical_run_implemented=False,
        P1_admission=False,P2_mechanism=False,physical_release=False,equal_FLOPs=False,
        new_neural_calls=0,new_physics_calls=0,selection_uses_physics=False)


def build(receipt_path, receipt_sha, output, report_sha):
    physical.native.actual_hash(receipt_sha); physical.native.actual_hash(report_sha)
    require(sha(receipt_path) == receipt_sha, 'actual external strict receipt SHA')
    receipt = read(receipt_path,128*1024**2)
    output = Path(physical.native.absolute(str(output)))
    require(sha(output/'report.json') == report_sha, 'actual source report SHA')
    report = read(output/'report.json',8*1024**2)
    require(receipt['report_sha256'] == report_sha and receipt['bindings'][str(output/'report.json')] == report_sha, 'report authority')
    compiled = producer.verify(report['prepared'],report['prepare_sha256'])
    data = physical.source_material()
    result = assemble(compiled,receipt,physical.native.allmode.root_metadata(data),report_sha,receipt_sha)
    result.update(receipt=str(Path(receipt_path).absolute()),producer_output=str(output),
        producer_prepared=report['prepared'],producer_prepare_sha256=report['prepare_sha256'],
        source_report_sha256=physical.native.SOURCE_SHA)
    require(sha(receipt_path) == receipt_sha and sha(output/'report.json') == report_sha, 'authority stable after read')
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('command',choices=('prepare','verify'))
    for n in ('receipt','receipt-sha','producer-output','report-sha','prepared'): ap.add_argument('--'+n,required=True)
    ap.add_argument('--prepare-sha'); a=ap.parse_args(argv)
    result=build(a.receipt,a.receipt_sha,a.producer_output,a.report_sha); path=Path(a.prepared).absolute()
    if a.command == 'prepare':
        require(a.prepare_sha is None and path.parent == ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists(), 'exclusive new prepare')
        b=raw(result); require(len(b) <= 128*1024**2,'bounded preparation')
        with path.open('xb') as f: f.write(b)
        print(json.dumps(dict(path=str(path),sha256=sha(path))))
    else:
        require(a.prepare_sha is not None and sha(path)==a.prepare_sha and raw(read(path,128*1024**2))==raw(result), 'whole saved preparation equality')
        print(json.dumps(dict(verified=True,sha256=a.prepare_sha)))


if __name__ == '__main__': main()
