"""Saved-array P2 contract checks. Never load a model or create an environment."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import p2_composition_paired_code_audit_20261027_v1 as math
from experiments.spectral_reachability import p2_composition_predict_frontier_physical_20261027_v1 as physical
from experiments.spectral_reachability import p2_full_resource_matching_20261027_v1 as resources

TAG = 'p2_composition_contract_audit_20261027_v1'
PROJECT = Path(__file__).resolve().parents[2]
CAP = 2 * 1024**2
FLOOR = 200 * 1024**2
require = math.require
digest = math.digest
index = math.keyed


def candidate_contract(result, compiled, initial, normalization, bank):
    """Check all raw packets, own-prefix contexts, Q costs and stable argmin."""
    arm = result['arm']
    joint = arm == resources.ARMS[0]
    count = 80 if joint else 40
    attempts = result['attempts']
    require(len(attempts) == count and result['selection_evaluations'] == count,
            'complete candidate denominator')
    q = compiled['scoring']['metric']['projection']; goal = compiled['goal_vectors'][0]
    diagnostics = []
    for i, saved in enumerate(attempts):
        require(saved['index'] == i and saved['candidate_id'] == f'{arm}_{i:02d}'
                and saved['completed'] and saved['metric_attempted'], 'ordered candidate identity')
        logits = math.array(saved['logits'], (20,))
        require(np.linalg.norm(logits) <= math.planner.sra.CAP*(1+1e-12), 'full20 logit ball')
        math.near(saved['actions'], np.tanh(logits).astype(np.float32).reshape(10, 2),
                  'raw mapping without clipping', 0)
        math.near(logits[:10], np.asarray(bank['logits'])[i,:10], 'bank prefix', 0)
        edges = saved['witness']['edges']
        end = saved['witness']['receipt']['terminal_context']
        require(edges[0]['input_context'] == initial and edges[-1]['output_context'] == end,
                'witness initial/terminal binding')
        if joint:
            math.near(logits, bank['logits'][i], 'joint full bank coordinates', 0)
            require(len(edges) == 1 and edges[0]['actions'] == saved['actions'], 'joint full H2 edge')
            require(end['latent_history'][0] == initial['latent_history'][-1]
                    and end['action_history'] == [math.normalized(saved['actions'][:5], normalization),
                    math.normalized(saved['actions'][5:], normalization)], 'joint packet history')
        else:
            require(len(edges) == 2 and edges[0]['actions']+edges[1]['actions'] == saved['actions']
                    and edges[0]['output_context'] == edges[1]['input_context']
                    and digest(edges[1]['input_context']) == saved['suffix_compilation']['input_context_sha256'],
                    'conditional own-prefix splice')
            for edge in edges:
                math.rolling(edge['input_context'], edge['output_context'], edge['actions'], normalization)
            local = dict(saved, prefix_logits=logits[:10].tolist())
            diagnostics.append(math.local_diagnostic(local, compiled))
        math.near(saved['cost'], math.metric(end['latent_history'][-1], goal, q), 'same Q cost')
    winner = min(attempts, key=lambda a:a['cost'])
    receipt = result['winner']
    require(receipt['candidate_id'] == winner['candidate_id'] and receipt['cost'] == winner['cost']
            and receipt['witness'] == winner['witness'] and receipt['independent_full_replay_exact'] is True,
            'stable argmin/winner replay receipt')
    end = winner['witness']['receipt']['terminal_context']
    require(receipt['whole_context'] == receipt['segmented_contexts'][-1] == end,
            'whole/segmented saved replay equality')
    return dict(candidates=count, winner_id=winner['candidate_id'], winner_cost=winner['cost'],
                max_scaled_kkt_residual=max((d['scaled_kkt_residual'] for d in diagnostics), default=0.))


def frozen_join(plans, committed, rows, encodings):
    """No array-position pairing and no replacement after observing physics."""
    fields = ('root_sha256', 'method')
    a,b,c,d = (index(x, fields) for x in (plans, committed, rows, encodings))
    require(len(a) == 8 and a.keys() == b.keys() == c.keys() == d.keys(), 'complete four-root two-arm join')
    for key, plan in a.items():
        require(plan == b[key], 'pre-physical frozen plan changed')
        for row in (c[key],d[key]):
            require(row['selection_sha256'] == plan['selection_sha256'], 'selected branch binding')
        require(d[key]['terminal_rgb_sha256'] == c[key]['terminal_rgb_sha256'], 'actual endpoint mapping')
    return a,c,d


def audit(project=PROJECT):
    project = Path(project).resolve(); reader = math.evidence.EvidenceReader()
    run = project/resources.PLANNER; actual = project/resources.PHYSICAL
    reader.bind(run/'report.json', resources.PLANNER_SHA)
    reader.bind(actual/'report.json', resources.PHYSICAL_SHA)
    report = reader.load(run/'report.json'); observed = reader.load(actual/'report.json')
    plans, _, _, _, bindings, _ = physical.planning_material(project, run, resources.PLANNER_SHA)
    reader.bindings(bindings)  # all-slot exporter rebuilds source-bound program IDs, not legacy40 export
    pp = reader.load(Path(report['prepared'])/'protocol.json')
    compiled = index(reader.load(pp['context']['compilation'])['rows'], ('root_sha256',))
    cases = index(pp['context']['cases'], ('root_sha256',))
    prepared_run = reader.load(actual/'execution_protocol.json')
    commitment = physical.validate_prepared(prepared_run['prepared'], prepared_run['prepare_sha256'])
    require(observed['consumer_prepare_sha256'] == prepared_run['prepare_sha256'], 'actual consumer commitment')
    durable = reader.load(actual/'physical_report.json')
    require(index(durable['rows'], ('root_sha256','method')) == index(observed['rows'], ('root_sha256','method')),
            'durable/final row agreement')
    selected, rows, encodings = frozen_join(plans, commitment['plans'], observed['rows'], observed['encodings'])
    root_receipts = index(report['rows'], ('root_sha256',)); roots = []
    require(root_receipts.keys() == cases.keys() == compiled.keys(), 'stable root corpus')
    for key, receipt in sorted(root_receipts.items()):
        i = receipt['root_index']; saved = reader.load(run/f'root_{i}.json')
        require(digest(saved) == receipt['row_sha256'] and saved['root_sha256'] == key[0], 'root content binding')
        bank = reader.load(run/f'root_{i}_bank.json')
        arms = index(saved['results'], ('arm',)); checks = {}; gains = {}
        c = compiled[key]['compiled']; initial = cases[key]['root']['context']
        for arm in resources.ARMS:
            checks[arm] = candidate_contract(arms[(arm,)], c, initial, pp['context']['normalization'], bank['expanded'])
            k = (key[0],arm); p,r,e = selected[k],rows[k],encodings[k]
            physical.verify_branch(p,r)
            require(r['full_horizon_completed'] and r['failure'] is None and e['status'] == 'completed', 'actual complete branch')
            reader.bind(r['journal_path'],r['journal_sha256'])
            for line in Path(r['journal_path']).read_text().splitlines():
                event = json.loads(line)
                if 'rgb_path' in event: reader.bind(event['rgb_path'],event['rgb_file_sha256'])
            predicted = math.metric(p['selected_predicted_context']['latent_history'][-1], c['goal_vectors'][0], c['scoring']['metric']['projection'])
            actual_cost = math.metric(e['terminal_vector'], c['goal_vectors'][0], c['scoring']['metric']['projection'])
            math.near(e['predicted_costs']['source_Q'], predicted, 'encoding predicted Q')
            math.near(e['actual_costs']['source_Q'], actual_cost, 'encoding actual Q')
            gains[arm] = dict(predicted_q_mse=predicted,encoded_q_mse=actual_cost,pose_cost=r['pose']['pose_cost'])
        require(rows[(key[0],resources.ARMS[0])]['context_identity'] == rows[(key[0],resources.ARMS[1])]['context_identity'], 'paired actual reset identity')
        roots.append(dict(root_sha256=key[0],root_index=i,contracts=checks,
                          composition_improvement=math.gains(*(gains[a] for a in resources.ARMS))))
    cost = resources.audit_current(project)
    reader.bindings(cost['bindings'])
    # Recheck the earlier paired40 cohort; do not pool outcomes with frontier80/40.
    old = math.compute(project/'results/p2_composition_paired_physical_20261027_v2_main01', reader)
    reader.unchanged()
    return dict(schema=TAG+'.report', status='saved_contract_checks_completed', roots=roots,
        earlier_paired40=dict(roots=len(old['roots']),suffix_KKT_checks=sum(len(r['candidate_diagnostics']) for r in old['roots']),
            demonstrated_fixable_implementation_error=old['demonstrated_fixable_implementation_error']),
        full_resources={k:v for k,v in cost.items() if k != 'bindings'},
        demonstrated_action_residual_history_Q_or_selection_error=False,
        findings=['168/168 matches attributed predict calls only, not total compute: conditional adds40 suffix Jacobians192x10,7680 mathematical VJP directions,40SVD and40solves per root.',
            'v1 failure-path finalization may mask a first exception; new CUDA v2 hardens it. This is not evidence of corruption in the completed v1 run.',
            'All-slot rebuild, KKT and history arithmetic validate saved implementation contracts, not derivative truth or model fidelity.'],
        freeze_scope='Prepare/source/action commitments agree with actual branch journals. Hashes alone do not prove wall-clock chronology, absence of hidden trials or independent prospective confirmation.',
        remaining_prerequisites=['Independent derivative/fidelity evidence under a preregistered protocol.',
            'Predeclare full resource caps and measure missing backward/time/FLOP/internal-physics axes; no empty padding.',
            'Independent roots and composition ablation/admission evidence; calibration0/8 does not establish a theoretical negative.'],
        P1_admission=False,P2_admission=False,new_neural_calls=0,new_physical_steps=0,
        input_bindings=deepcopy(reader.hashes))


def write_new(path, value):
    path = Path(path).resolve()
    require(TAG in path.name and not path.exists(), 'new exclusive audit filename')
    raw = (json.dumps(value,sort_keys=True,allow_nan=False,indent=2)+'\n').encode()
    require(len(raw) <= CAP, '2MiB output cap')
    require(shutil.disk_usage(path.parent).free >= FLOOR+len(raw), '200MiB reserve')
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    result = audit()
    result['audit_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(dict(output=str(args.output.resolve()),sha256=write_new(args.output,result),status=result['status'])))


if __name__ == '__main__': main()
