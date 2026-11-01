"""Saved-data arithmetic for paired composition: full H2 separate from partials.

No neural calls, simulator, new encodings, resampling, population inference,
equal-resource claim or scientific admission. Original official/Q/pose only.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics

import numpy as np

from experiments.spectral_reachability import p2_composition_paired_physical_20261027_v2 as physical
from experiments.spectral_reachability import p2_composition_admission_20261027_v1 as evidence
from experiments.spectral_reachability.audit_pusht_sealed_task_contract import labels
from experiments.spectral_reachability.d2p_composition_witness_v2 import digest
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG = 'p2_composition_paired_outcomes_20261027_v2'
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_PHYSICAL = PROJECT/'results/p2_composition_paired_physical_20261027_v2_main01'
ARMS = physical.sampling.ARMS
METRICS = ('official_state_distance','official_position_error','agent_position_error',
    'object_position_error','official_angle_error','official_success','object_only_success',
    'object_xy_distance','object_angle_distance','object_pose_cost','predicted_q_mse','real_encoder_q_mse')
HIGHER = {'official_success','object_only_success'}
BOUNDARY = dict(P1_admission=False,P2_admission=False,physical_efficacy_verified=False,
    equal_total_compute=False,fair_total_budget_comparison=False,independent_cohort=False,
    descriptive_only=True,inferential_tests_performed=False,new_neural_calls=0,new_physical_steps=0,
    partial_policy='retain every rootarm; observed partial metrics never enter full-H2 differences or Q',
    scope='four already-used calibration roots, conditional suffix ablation; not general composition efficacy')
require = evidence.require
write = physical.write


def index(rows, keys, count=None):
    require(isinstance(rows,list),'explicit row census required')
    result = {tuple(r[k] for k in keys):r for r in rows}
    require(len(result)==len(rows) and (count is None or len(rows)==count),'missing/duplicate stable identity')
    return result


def array(value, shape):
    raw = np.asarray(value)
    require(raw.dtype.kind in 'fi' and raw.shape==shape and np.isfinite(raw).all(),'finite exact-shape numeric array required')
    return raw.astype(np.float64)


def close(actual, expected, name):
    require(type(actual) in (int,float) and np.isfinite(actual)
        and np.isclose(actual,expected,rtol=0,atol=1e-10),'saved '+name+' mismatch')


def pose(state, goal):
    xy = float(np.linalg.norm(state[2:4]-goal[2:4]))
    angle = float(abs(np.arctan2(np.sin(state[4]-goal[4]),np.cos(state[4]-goal[4]))))
    return dict(xy_distance=xy,angle_distance=angle,pose_cost=xy+20*angle)


def q_cost(vector, compiled):
    v = array(vector,(192,))
    g = array(compiled['goal_vectors'][0],(192,))
    q = array(compiled['scoring']['metric']['projection'],(192,192))
    return float(np.mean((q@(v-g))**2))


def branch_outcome(plan, row, encoding, compiled):
    key = (plan['root_sha256'],plan['method'])
    require(key==(row['root_sha256'],row['method']) and row['root_index']==plan['root_index']
        and row['selection_sha256']==plan['selection_sha256'] and row['repetition']==0,
        'branch/rootarm/selection identity mismatch')
    require(row['planned_horizon_tokens']==2 and row['planned_commands']==10,'original fixed H2 required')
    attempts,returns = row['physical_env_step_attempts'],row['physical_env_step_returns']
    require(type(attempts) is type(returns) is int and 0<=returns<=attempts<=10
        and row['attempted_commands']==attempts and row['completed_commands']==returns,
        'nonrefunded physical counters mismatch')
    trace = row['trace']
    require(len(trace)<=returns and [e['primitive_step'] for e in trace]==list(range(1,len(trace)+1))
        and [e['command'] for e in trace]==plan['commands'][:len(trace)],'observed selected-action prefix mismatch')
    fixed_endpoint = row['full_horizon_completed'] is True and returns==10 and len(trace)==10
    complete = fixed_endpoint and row['status']=='completed' and not row.get('failures')
    if row['status']=='completed':
        require(complete,'completed status cannot promote incomplete/failed H2')
    if 'guard' in row:
        require(row['guard']['root_id']==key[0] and row['guard']['primitive_commands_authorized']==attempts,
            'guard root/authorized attempt mismatch')
        if complete:
            require(row['guard']['abstained'] is False,'full H2 cannot be an abstention')
    metrics,goal,initial = None,None,None
    identity = row.get('context_identity')
    if identity is not None:
        require(identity['history_rgb_sha256']==plan['root']['root_rgb_sha256']
            and identity['goal_rgb_sha256']==plan['root']['goal_rgb_sha256']
            and identity['normalized_between_history']==plan['root']['context']['action_history'],
            'physical root/goal/history binding mismatch')
        goal = array(identity['goal'],(7,))
        initial = array(identity['state'],(7,))
        physical.validate_reset_history(identity['raw_history'],compiled['_normalization'],
            plan['root']['context']['action_history'])
        state = row.get('final_state')
        if state is not None:
            state = array(state,(7,))
            require(state.tolist()==(trace[-1]['state'] if trace else initial.tolist()),
                'final state is not last observed endpoint')
            official, p = labels(state,goal),pose(state,goal)
            for name,value in row.get('official',{}).items():
                require(name in official,'unknown official metric')
                if type(value) is bool:
                    require(value is official[name],'saved official success mismatch')
                else:
                    close(value,official[name],'official '+name)
            for name,value in row.get('pose',{}).items():
                require(name in p,'unknown pose metric')
                close(value,p[name],'pose '+name)
            if complete:
                require(set(row.get('official',{}))==set(official) and set(row.get('pose',{}))==set(p),
                    'complete original official/pose metrics required')
            metrics = dict(official_state_distance=official['official_state_distance'],
                official_position_error=official['official_position_error'],agent_position_error=official['agent_position_error'],
                object_position_error=official['object_position_error'],official_angle_error=official['angle_error'],
                official_success=int(official['official_success']),object_only_success=int(official['object_only_success']),
                object_xy_distance=p['xy_distance'],object_angle_distance=p['angle_distance'],object_pose_cost=p['pose_cost'])
    if complete:
        require(metrics is not None,'full H2 requires observed terminal state and strict goal binding')
    predicted = q_cost(plan['witness']['receipt']['terminal_context']['latent_history'][-1],compiled)
    actual_q = None
    if encoding is not None:
        require(key==(encoding['root_sha256'],encoding['method']) and encoding['repetition']==0,
            'encoding rootarm mismatch')
        require(type(encoding['attempted_encode_samples']) is int and encoding['attempted_encode_samples'] in (0,1),
            'encoding attempt count mismatch')
        if encoding['status']=='completed':
            require(fixed_endpoint and encoding['attempted_encode_samples']==1 and encoding['failure'] is None,
                'partial/failed endpoint cannot have full-H2 encoding')
            array(encoding['terminal_vector'],(192,))
            if complete:
                actual_q = q_cost(encoding['terminal_vector'],compiled)
    full_metrics = None if not complete else dict(metrics,predicted_q_mse=predicted,real_encoder_q_mse=actual_q)
    return dict(root_index=plan['root_index'],root_sha256=key[0],arm=key[1],
        status=row['status'],classification='full_H2' if complete else 'partial_or_failed',full_H2=complete,
        attempted_commands=attempts,returned_commands=returns,observed_commands=len(trace),
        returned_without_observation=returns-len(trace),observed_endpoint_step=len(trace) if metrics is not None else None,
        observed_endpoint_metrics=metrics,full_H2_metrics=full_metrics,goal=None if goal is None else goal.tolist(),
        initial_state=None if initial is None else initial.tolist(),predicted_q_mse=predicted,
        encoding_status='not_requested' if encoding is None else encoding['status'],
        excluded_complete_endpoint_encoding=bool(not complete and encoding is not None and encoding['status']=='completed'),
        failures=deepcopy(row.get('failures',[])),selected_candidate_id=plan['selected_candidate_id'],
        receipt_sha256=plan['receipt_sha256'],planning_accounting=deepcopy(plan['planning_accounting']))


def describe(values, higher=False):
    values = list(values)
    require(all(type(v) in (int,float) and np.isfinite(v) for v in values),'finite descriptive differences required')
    if not values:
        return dict(n=0,mean=None,median=None,sample_sd=None,min=None,max=None,wins=0,ties=0,losses=0)
    return dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),
        sample_sd=statistics.stdev(values) if len(values)>1 else None,min=min(values),max=max(values),
        wins=sum(v>0 if higher else v<0 for v in values),ties=sum(v==0 for v in values),
        losses=sum(v<0 if higher else v>0 for v in values))


def summarize(rows):
    keyed = index(rows,('root_sha256','arm'),8)
    roots = sorted({r['root_sha256'] for r in rows},key=lambda root:keyed[root,ARMS[0]]['root_index'])
    require(len(roots)==4 and set(keyed)=={(root,arm) for root in roots for arm in ARMS},'strict four-root eight-arm census required')
    pairs = []
    for root in roots:
        left,right = keyed[root,ARMS[1]],keyed[root,ARMS[0]]
        require(left['root_index']==right['root_index'],'root index pairing mismatch')
        if left['goal'] is not None and right['goal'] is not None:
            require(left['goal']==right['goal'] and left['initial_state']==right['initial_state'],
                'paired goal/reset state differs')
        full = left['full_H2'] and right['full_H2']
        delta = None if not full else {name:(None if left['full_H2_metrics'][name] is None or right['full_H2_metrics'][name] is None
            else left['full_H2_metrics'][name]-right['full_H2_metrics'][name]) for name in METRICS}
        pairs.append(dict(root_index=left['root_index'],root_sha256=root,both_full_H2=full,
            composition_status=left['classification'],no_composition_status=right['classification'],
            composition_minus_no_composition=delta,
            exclusion_reason=None if full else 'retained partial/failed branch; no fixed-H2 difference'))
    stats = {metric:dict(describe([p['composition_minus_no_composition'][metric] for p in pairs
        if p['both_full_H2'] and p['composition_minus_no_composition'][metric] is not None],metric in HIGHER),
        root_ids=[p['root_sha256'] for p in pairs if p['both_full_H2'] and p['composition_minus_no_composition'][metric] is not None])
        for metric in METRICS}
    return dict(paired=pairs,full_H2_pair_count=sum(p['both_full_H2'] for p in pairs),
        partial_or_failed_root_ids=[p['root_sha256'] for p in pairs if not p['both_full_H2']],
        deltas_descriptive=stats,delta_direction='composition minus no-composition; lower better except success indicators',
        missing_policy='no imputation, dropping hidden behind n=4, partial pooling or shortened-H2 scores')


def audit_artifacts(directory, report, reader, plans):
    physical_report = reader.load(directory/'physical_report.json')
    require(index(physical_report['rows'],('root_sha256','method'),8)==index(report['rows'],('root_sha256','method'),8),
        'durable physical report/final census differs')
    enc = index(report['encodings'],('root_sha256','method'))
    for row in report['rows']:
        stem = f'root_{row["root_index"]}_{row["method"]}_replay0'
        require(reader.load(directory/(stem+'.physical.json'))==row,'persisted physical row differs')
        if row.get('journal_path'):
            path = Path(row['journal_path'])
            require(path.resolve()==directory/(stem+'.events.jsonl'),'journal belongs to wrong branch/output')
            reader.bind(path,row['journal_sha256'])
            events = [json.loads(line,object_pairs_hook=evidence.strict_object) for line in path.read_text().splitlines()]
            attempts = [e for e in events if e['kind']=='attempt']
            returns = [e for e in events if e['kind']=='step_returned']
            observed = [e for e in events if e['kind']=='observation']
            require(len(attempts)==row['physical_env_step_attempts'] and len(returns)==row['physical_env_step_returns']
                and [{k:v for k,v in e.items() if k not in ('rgb_path','rgb_file_sha256')} for e in observed]==row['trace'],
                'durable attempts/returns/observations differ')
            plan = plans[row['root_sha256'],row['method']]
            require(attempts==[dict(kind='attempt',primitive_step=i+1,command=plan['commands'][i]) for i in range(len(attempts))]
                and returns==[dict(kind='step_returned',primitive_step=i+1) for i in range(len(returns))],
                'journal attempted commands/returned steps differ from selected program')
            expected_order = []
            if events:
                require(events[0]['kind']=='initial' and events[0]['state']==row['context_identity']['state'],
                    'durable initial context mismatch')
                expected_order.append('initial')
            for i in range(len(attempts)):
                expected_order.append('attempt')
                if i<len(returns): expected_order.append('step_returned')
                if i<len(observed): expected_order.append('observation')
            require([e['kind'] for e in events]==expected_order,'durable event chronology differs')
            for e in events:
                if 'rgb_path' in e:
                    image = Path(e['rgb_path']).resolve()
                    require(image.parent==directory and image.name.startswith(stem+'_rgb_'),'RGB branch path mismatch')
                    reader.bind(image,e['rgb_file_sha256'])
            if row.get('terminal_rgb_path'):
                images = [e for e in events if 'rgb_path' in e]
                require(images and images[-1]['rgb_path']==row['terminal_rgb_path']
                    and images[-1]['rgb_file_sha256']==row['terminal_rgb_sha256'],'last observed RGB differs')
        key = row['root_sha256'],row['method']
        if key in enc:
            require(reader.load(directory/(stem+'.encoding.json'))==enc[key],'persisted encoding differs')


def load_and_compute(directory, reader):
    directory = Path(directory).resolve()
    report,execution = (reader.load(directory/n) for n in ('report.json','protocol.json'))
    require(report['schema']==physical.TAG+'.report' and report['status'] in ('calibration_recorded','integrity_fail'),
        'terminal physical consumer report required')
    _,protocol,prepared,context = physical.load_prepared(execution['prepared'])
    reader.bindings(protocol['bindings'])
    reader.seals(prepared['source_seals'])
    for name in ('protocol.json','prepared.json'):
        reader.load(Path(execution['prepared'])/name)
    require(execution['prepared_protocol_sha256']==report['prepared_protocol_sha256']==prepared['protocol_sha256']
        and report['source_seals']==prepared['source_seals'] and execution['plans']==protocol['plans']==context['plans'],
        'physical prepare/run/source plan binding mismatch')
    require(all(report[k]==v for k,v in physical.BOUNDARY.items()) and report['new_predict_calls']==0,
        'physical claim boundary changed')
    plans = index(protocol['plans'],('root_sha256','method'),8)
    branches = index(report['rows'],('root_sha256','method'),8)
    encodings = index(report['encodings'],('root_sha256','method'))
    require(plans.keys()==branches.keys(),'missing physical rootarm')
    require((execution['encode_terminals'] is True and plans.keys()==encodings.keys())
        or (execution['encode_terminals'] is False and not encodings),'encoding request/census mismatch')
    audit_artifacts(directory,report,reader,plans)
    recomputed_gates = physical.physical_gates(list(plans.values()),report['rows'],directory)
    require(all(report['gates'].get(k) is v for k,v in recomputed_gates.items()),'saved physical integrity gates differ')
    compiled = {r['root_sha256']:dict(r['compiled'],_normalization=context['planning_protocol']['context']['normalization'])
        for r in context['source']['rows']}
    rows = [branch_outcome(plan,branches[key],encodings.get(key),compiled[key[0]]) for key,plan in plans.items()]
    require(report['selected_path_env_step_attempts']==sum(r['attempted_commands'] for r in rows)
        and report['selected_path_env_step_returns']==sum(r['returned_commands'] for r in rows)
        and report['attempted_terminal_encode_samples']==sum(e['attempted_encode_samples'] for e in encodings.values()),
        'actual physical/encoding total mismatch')
    result = dict(schema=TAG+'.report',status='descriptive_arithmetic_completed',rows=rows,**summarize(rows),
        source_physical_status=report['status'],source_physical_gates=report['gates'],
        planning_totals=protocol['planning_totals'],
        additional_verification_work=context['verification_work'],
        additional_verification_scope='one inherited source-bound exporter array reconstruction; not planning or new neural work',
        selected_path_env_step_attempts=report['selected_path_env_step_attempts'],
        selected_path_env_step_returns=report['selected_path_env_step_returns'],
        attempted_terminal_encode_samples=report['attempted_terminal_encode_samples'],**deepcopy(BOUNDARY))
    reader.unchanged()
    return result


def source_files():
    return sorted({Path(__file__).resolve(),PROJECT/'tests'/f'test_{TAG}.py',PROJECT/'innerdocs'/f'{TAG}.md',
        Path(physical.__file__).resolve(),Path(physical.sampling.__file__).resolve(),Path(evidence.__file__).resolve(),
        Path(labels.__code__.co_filename).resolve(),Path(create_seal.__code__.co_filename).resolve()})


def run(physical_run, output):
    output = Path(output).resolve()
    require(TAG in output.name,'new tagged arithmetic results directory required')
    if output.exists():
        raise FileExistsError(output)
    reader = evidence.EvidenceReader()
    result = load_and_compute(physical_run,reader)
    protocol = dict(schema=TAG+'.protocol',physical_run=str(Path(physical_run).resolve()),
        bindings=deepcopy(reader.hashes),boundary=deepcopy(BOUNDARY),source_paths=list(map(str,source_files())),
        metrics=list(METRICS),selection='none; descriptive post-execution arithmetic, not a prospective efficacy test')
    output.mkdir(exist_ok=False)
    write(output/'protocol.json',protocol)
    seal = create_seal(files=source_files(),args_json=json.dumps(dict(protocol_sha256=digest(protocol))),
        output_dir=output/'source_seal_0')
    result.update(protocol_sha256=digest(protocol),source_seals=[seal],input_bindings=deepcopy(reader.hashes))
    write(output/'report.json',result)
    reader.unchanged()
    verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    print(json.dumps(dict(status=result['status'],output=str(output),full_H2_pairs=result['full_H2_pair_count'],P2_admission=False)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical-run',type=Path,default=DEFAULT_PHYSICAL)
    parser.add_argument('--output',type=Path,required=True)
    run(**vars(parser.parse_args(argv)))


if __name__=='__main__':
    main()
