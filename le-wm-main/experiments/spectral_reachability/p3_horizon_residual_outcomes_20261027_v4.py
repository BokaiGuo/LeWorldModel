"""Independent saved-evidence P3 v4 outcomes; arithmetic/RGB reads only.

No neural/simulator imports, re-encoding, root selection, inference or admission.
Mixed/partial endpoints are retained outside full same-horizon summaries.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import statistics
import numpy as np
from experiments.spectral_reachability import p3_horizon_residual_physical_20261027_v4 as physical
from experiments.spectral_reachability.audit_pusht_sealed_task_contract import labels
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal

NAME='p3_horizon_residual_outcomes_20261027_v4'
PROJECT=Path(__file__).resolve().parents[2]
PHYSICAL_PREPARED='results/p3_horizon_residual_physical_20261027_v4_prepared01'
PHYSICAL_SHA='a6661c8fe3ff665af88a493b56b439fa6e98908bf588cf895678c83239eea8c7'
DEFAULT_RUN='results/p3_horizon_residual_physical_20261027_v4_main01'
io,require,write=physical.io,physical.require,physical.write
METRICS=('official_state_distance','official_position_error','agent_position_error',
    'object_position_error','angle_error','official_success','object_only_success',
    'pose_cost','xy_distance','angle_distance','predicted_Q_cost','actual_Q_cost')
HIGHER={'official_success','object_only_success'}


def contract():
    return dict(methods=list(physical.METHODS),delta='dynamic minus uniform; lower better except success',
        strata=['full_H2_pair','full_H3_pair','full_mixed_H2_H3','partial_same_steps','partial_or_unmatched'],
        primary_deltas='full same-horizon pairs only; no mixed/partial pooling',
        reversals='strict opposite nonzero signs of predicted Q delta and actual lower-is-better cost delta; exact ties separate',
        official='original installed-compatible labels including agent and state distance; no metric substitution',
        pose='object_xy +20*abs(atan2(sin(angle_diff),cos(angle_diff)))',
        Q='mean((original_Q@(vector-goal0))**2); cached original goal0, not physical-state pose',
        missing='retain all eight branches; missing encoding gives null actual Q, never impute',
        descriptive_only=True,inferential_tests_performed=False,equal_total_compute=False,
        new_neural_calls=0,new_physical_steps=0,P1_admission=False,P3_admission=False,efficacy_claim=False)


def array(value,shape):return physical.planning.prior.array(value,shape)


def close(actual,expected,name):
    require(type(actual) in (int,float) and np.isfinite(actual) and np.isclose(actual,expected,rtol=0,atol=1e-10),'saved '+name+' mismatch')


def pose(state,goal):
    xy=float(np.linalg.norm(state[2:4]-goal[2:4]))
    angle=float(abs(np.arctan2(np.sin(state[4]-goal[4]),np.cos(state[4]-goal[4]))))
    return dict(xy_distance=xy,angle_distance=angle,pose_cost=xy+20*angle)


def q_cost(vector,plan):
    return float(np.mean((array(plan['projection'],(192,192))@(array(vector,(192,))-array(plan['goal'],(192,))))**2))


def branch_outcome(plan,row,encoding,reference):
    physical.validate_plan(plan)
    require(all(row[k]==plan[k] for k in ('root_sha256','root_index','method','selection_sha256','planned_horizon_tokens','planned_commands')) and row['repetition']==0,'strict physical rootarm/selection identity')
    attempted,returned=(io.count(row[k]) for k in ('physical_env_step_attempts','physical_env_step_returns'))
    require(returned<=attempted<=plan['planned_commands'] and row['attempted_commands']==attempted and row['completed_commands']==returned,'physical attempt/return accounting')
    trace=row['trace'];require(len(trace)<=returned and [e['primitive_step'] for e in trace]==list(range(1,len(trace)+1)) and
        [e['command'] for e in trace]==plan['commands'][:len(trace)],'observed action prefix')
    full=(row['status']=='completed' and row['full_horizon_completed'] is True and returned==plan['planned_commands'] and
        len(trace)==returned and row['endpoint_fresh'] is True and not row.get('failures'))
    if row['status']=='completed':require(full,'partial/failed row cannot claim full selected horizon')
    if row['full_horizon_completed']:require(returned==plan['planned_commands'] and len(trace)==returned,'full command census cannot hide truncated trace')
    predicted=q_cost(plan['selected_predicted_context']['latent_history'][-1],plan)
    close(plan['predicted_Q_cost'],predicted,'predicted original Q')
    identity=row.get('context_identity');observed=None;initial=goal=None
    if identity is not None:
        require(all(identity[k]==reference[k] for k in ('history_rgb_sha256','goal_rgb_sha256')),'original root/goal RGB reference')
        warmup=physical.validate_reset_history(identity['raw_history'],plan['predictor_context']['action_history'])
        require(warmup['normalized_between_history']==identity['normalized_between_history'],'saved normalized warmup history')
        goal=array(identity['goal'],(7,));initial=array(identity['state'],(7,))
        if row.get('final_state') is not None:
            state=array(row['final_state'],(7,))
            require(state.tolist()==(trace[-1]['state'] if trace else initial.tolist()),'actual final state is last observed endpoint')
            official=labels(state,goal);p=pose(state,goal)
            for kind,expected in (('official',official),('pose',p)):
                saved=row.get(kind,{})
                require(set(saved)<=set(expected),'unknown saved metric')
                if full:require(set(saved)==set(expected),'full path requires all original official/pose metrics')
                for k,v in saved.items():
                    if type(expected[k]) is bool:require(type(v) is bool and v==expected[k],'saved success label')
                    else:close(v,expected[k],kind+'.'+k)
            observed=dict(official,**p)
    if full:require(observed is not None and identity is not None,'full path requires observed bound endpoint')
    actual_q=None
    if encoding is not None:
        require(all(encoding[k]==plan[k] for k in ('root_sha256','method','selection_sha256','planned_horizon_tokens')),'encoding rootarm/horizon/selection')
        n=io.count(encoding['attempted_encode_samples']);require(n<=1 and encoding['endpoint_scope']==row['endpoint_scope'],'endpoint encoding scope/cap')
        if encoding['status']=='completed':
            require(row['endpoint_fresh'] is True and n==1 and encoding['failure'] is None and encoding['terminal_rgb_sha256']==row['terminal_rgb_sha256'],'no stale or substituted encoded endpoint')
            require(encoding['goal_vector_sha256']==plan['goal_vector_sha256'] and encoding['projection_sha256']==plan['projection_sha256'],'original goal0/Q encoding identity')
            actual_q=q_cost(encoding['terminal_vector'],plan);close(encoding['original_Q_cost'],actual_q,'actual original Q')
    metrics=None if observed is None else dict(observed,predicted_Q_cost=predicted,actual_Q_cost=actual_q)
    return dict(root_index=plan['root_index'],root_sha256=plan['root_sha256'],method=plan['method'],
        selection_sha256=plan['selection_sha256'],selected_horizon=plan['planned_horizon_tokens'],status=row['status'],
        full_selected_horizon=full,endpoint_fresh=row['endpoint_fresh'],endpoint_scope=row['endpoint_scope'],
        attempted_commands=attempted,returned_commands=returned,observed_commands=len(trace),returned_without_observation=returned-len(trace),
        initial_context=deepcopy(identity),observed_endpoint_metrics=metrics,full_metrics=metrics if full else None,
        predicted_Q_cost=predicted,actual_Q_cost=actual_q,encoding_status=encoding['status'] if encoding else 'not_requested',
        failures=deepcopy(row.get('failures',[])))


def reversal(predicted_delta,actual_delta):
    if actual_delta is None:return 'unavailable'
    if predicted_delta==0 or actual_delta==0:return 'tie_in_either'
    return 'strict_reversal' if (predicted_delta<0)!=(actual_delta<0) else 'same_direction'


def describe(values,higher=False):
    values=list(values)
    return dict(n=len(values),mean=statistics.mean(values) if values else None,median=statistics.median(values) if values else None,
        wins=sum(v>0 if higher else v<0 for v in values),ties=sum(v==0 for v in values),losses=sum(v<0 if higher else v>0 for v in values))


def summarize(rows,roots):
    index=io.index(rows,('root_sha256','method'))
    require(len(roots)==len(set(roots))==4 and set(index)=={(r,m) for r in roots for m in physical.METHODS},'four roots eight-arm pairing')
    pairs=[]
    for i,root in enumerate(roots):
        u,d=(index[root,m] for m in physical.METHODS)
        require(u['root_index']==d['root_index']==i,'stable root index pairing')
        bound=u['initial_context'] is not None and d['initial_context'] is not None
        if bound:require(physical.same(u['initial_context'],d['initial_context']),'paired root/goal/reset/history mismatch')
        full=bound and u['full_selected_horizon'] and d['full_selected_horizon']
        same_h=u['selected_horizon']==d['selected_horizon']
        partial=bound and u['endpoint_fresh'] and d['endpoint_fresh'] and u['returned_commands']==d['returned_commands']
        label=(f'full_H{u["selected_horizon"]}_pair' if same_h else 'full_mixed_H2_H3') if full else ('partial_same_steps' if partial else 'partial_or_unmatched')
        eligible=full and same_h
        delta=None
        if eligible:
            delta={k:None if u['full_metrics'][k] is None or d['full_metrics'][k] is None else
                float(d['full_metrics'][k])-float(u['full_metrics'][k]) for k in METRICS}
        rev=None if delta is None else {k:reversal(delta['predicted_Q_cost'],delta[k]) for k in ('actual_Q_cost','pose_cost','official_state_distance')}
        pairs.append(dict(root_index=i,root_sha256=root,classification=label,selected_horizons=[u['selected_horizon'],d['selected_horizon']],
            full_same_horizon=eligible,dynamic_minus_uniform=delta,predicted_Q_vs_actual_direction=rev,
            arm_statuses=[u['status'],d['status']],actual_steps=[u['returned_commands'],d['returned_commands']],
            exclusion_reason=None if eligible else 'mixed duration or partial/unmatched; retained, never pooled'))
    strata={}
    for h in (2,3):
        subset=[p for p in pairs if p['classification']==f'full_H{h}_pair']
        strata[f'full_H{h}_pair']=dict(root_ids=[p['root_sha256'] for p in subset],
            delta_descriptive={k:describe([p['dynamic_minus_uniform'][k] for p in subset if p['dynamic_minus_uniform'][k] is not None],k in HIGHER) for k in METRICS},
            strict_reversals={k:sum(p['predicted_Q_vs_actual_direction'][k]=='strict_reversal' for p in subset) for k in ('actual_Q_cost','pose_cost','official_state_distance')})
    return dict(pairs=pairs,strata=strata,full_same_horizon_pair_count=sum(p['full_same_horizon'] for p in pairs),
        retained_noncomparable_root_ids=[p['root_sha256'] for p in pairs if not p['full_same_horizon']])


def audit_branch_artifacts(directory,plan,row,reader):
    stem=f'root_{plan["root_index"]}_{plan["method"]}_replay0'
    saved=reader.json(directory/(stem+'.physical.json'))
    base={k:v for k,v in row.items() if k not in ('endpoint_fresh','endpoint_scope')}
    require(physical.same(saved,row if row['status']=='blocked' else base),'durable branch receipt/final report differs')
    if row['status']=='blocked':
        require(not row['endpoint_fresh'] and not row['full_horizon_completed'] and row['physical_env_step_attempts']==row['physical_env_step_returns']==0,'blocked branch cannot have work/endpoint')
        return
    endpoint=reader.json(directory/(stem+'.endpoint.json'))
    require(endpoint==dict(root_sha256=plan['root_sha256'],method=plan['method'],selection_sha256=plan['selection_sha256'],endpoint_fresh=row['endpoint_fresh'],endpoint_scope=row['endpoint_scope']),'durable endpoint annotation')
    path=Path(row['journal_path']);require(path==directory/(stem+'.events.jsonl'),'branch journal path identity')
    events=[io.decode(line) for line in reader.raw(path,row['journal_sha256']).splitlines()]
    attempts=[e for e in events if e['kind']=='attempt'];returns=[e for e in events if e['kind']=='step_returned'];observed=[e for e in events if e['kind']=='observation']
    require(len(attempts)==row['physical_env_step_attempts'] and len(returns)==row['physical_env_step_returns'],'journal actual attempts/returns')
    require(attempts==[dict(kind='attempt',primitive_step=i+1,command=plan['commands'][i]) for i in range(len(attempts))] and
        returns==[dict(kind='step_returned',primitive_step=i+1) for i in range(len(returns))],'all journaled selected commands')
    require([{k:v for k,v in e.items() if k not in ('rgb_path','rgb_file_sha256')} for e in observed]==row['trace'],'journal observed trace')
    chronology=[]
    if events:
        require(events[0]['kind']=='initial' and events[0]['state']==row['context_identity']['state'],'journal initial state binding')
        chronology.append('initial')
    for i in range(len(attempts)):
        chronology.append('attempt')
        if i<len(returns):chronology.append('step_returned')
        if i<len(observed):chronology.append('observation')
    require([e['kind'] for e in events]==chronology,'durable event chronology')
    images=[e for e in events if 'rgb_path' in e];pixels=[]
    from PIL import Image
    for e in images:
        image=Path(e['rgb_path']);step=e.get('primitive_step',0)
        require(image==directory/f'{stem}_rgb_{step}.png','RGB belongs to exact branch and step')
        reader.raw(image,e['rgb_file_sha256'])
        with Image.open(image) as im:rgb=np.asarray(im)
        require(rgb.shape==(224,224,3) and rgb.dtype==np.uint8,'original physical RGB layout')
        pixels.append(dict(primitive_step=step,canonical_agent_pixels=int(np.count_nonzero(np.all(rgb==[65,105,225],axis=-1)))))
    if 'guard' in row:
        guard=row['guard'];require(guard['root_id']==plan['root_sha256'] and guard['primitive_commands_authorized']==len(attempts) and guard['observations']==pixels,'independent saved-RGB guard accounting')
        require(all(pixels[i]['primitive_step']==i and pixels[i]['canonical_agent_pixels']>0 for i in range(len(attempts))),'fresh RoyalBlue before each authorized command')
        require(guard['abstained']==bool(pixels and pixels[-1]['canonical_agent_pixels']==0),'guard abstention receipt')
    fresh=bool(images and images[-1].get('primitive_step',0)==len(returns) and len(attempts)==len(returns) and row['status'] in ('completed','observation_unavailable','environment_terminated'))
    scope=f'full_H{plan["planned_horizon_tokens"]}' if row['full_horizon_completed'] else (f'partial_H{plan["planned_horizon_tokens"]}' if fresh else 'unknown_or_stale_endpoint')
    require(row['endpoint_fresh']==fresh and row['endpoint_scope']==scope,'independent endpoint freshness/scope')
    if images:require(row['terminal_rgb_path']==images[-1]['rgb_path'] and row['terminal_rgb_sha256']==images[-1]['rgb_file_sha256'],'last durable endpoint RGB')


def load_and_compute(directory,protocol,reader,expected_report_sha):
    directory=Path(directory).absolute();report=reader.json(directory/'report.json',expected_report_sha)
    execution=reader.json(directory/'execution_protocol.json')
    require(report['schema']==physical.NAME+'.report' and report['status'] in ('guarded_calibration_recorded','incomplete'),'terminal physical consumer report')
    require(execution['prepared']==protocol['physical_prepared'] and execution['prepare_sha256']==report['consumer_prepare_sha256']==PHYSICAL_SHA,'exact physical preparation binding')
    physical_protocol=physical.validate_prepared(protocol['physical_prepared'],PHYSICAL_SHA)
    reader.raw(Path(protocol['physical_prepared'])/'prepare.json',PHYSICAL_SHA)
    reader.raw(Path(protocol['physical_prepared'])/'protocol.json')
    for path,h in physical_protocol['inputs_sha256'].items():reader.raw(Path(path),h)
    for path,h in physical_protocol['sources_sha256'].items():reader.raw(Path(path),h)
    require(report['source_verified'] is True and report['model_unchanged'] is True,'source/model integrity required for outcomes')
    for k in ('P1_admission','P3_admission','efficacy_claim','total_flops_equivalent'):require(report[k] is False,'source scope changed')
    require(report['new_predict_calls']==report['new_jacobian_materializations']==report['new_independent_roots']==0,'no extra search/AD/roots')
    require(report['planning_report_sha256']==execution['planning_report_sha256']==physical_protocol['planning_report_sha256'] and
        report['planner_protocol_id']==physical_protocol['planner_protocol_id'],'actual planner report/protocol binding')
    physical_report=reader.json(directory/'physical_report.json',report['physical_report_sha256'])
    require(physical_report['rows']==report['rows'] and physical_report['consumer_prepare_sha256']==PHYSICAL_SHA and physical_report['contract']==physical.contract(),'physical durable-before-neural report')
    plans=io.index(physical_protocol['plans'],('root_sha256','method'));rows=io.index(report['rows'],('root_sha256','method'));enc=io.index(report['encodings'],('root_sha256','method'))
    require(len(plans)==8 and set(plans)==set(rows),'strict eight physical rootarms')
    require((execution['encode_terminals'] is True and set(enc)==set(plans)) or (execution['encode_terminals'] is False and not enc),'encoding request and census')
    results=[]
    for key,plan in plans.items():
        row=rows[key];audit_branch_artifacts(directory,plan,row,reader)
        if key in enc:
            require(reader.json(directory/f'root_{plan["root_index"]}_{plan["method"]}_replay0.encoding.json')==enc[key],'durable encoding receipt')
        results.append(branch_outcome(plan,row,enc.get(key),physical_protocol['references'][key[0]]))
    require(report['pairs']==physical.comparisons(physical_protocol['plans'],report['rows']),'saved physical pairing classifier')
    if enc:
        journal=[io.decode(line) for line in reader.raw(directory/'encoding.events.jsonl').splitlines()]
        outputs=[{k:v for k,v in e.items() if k!='kind'} for e in journal if e.get('kind')=='endpoint_result']
        require(outputs==report['encodings'],'durable encoding result chronology')
        attempts=[e for e in journal if e.get('kind')=='endpoint_encode']
        require([(e['root_sha256'],e['method']) for e in attempts]==[(e['root_sha256'],e['method']) for e in report['encodings'] if e['attempted_encode_samples']==1],'durable encoding attempts')
    require(report['physical_env_step_attempts']==sum(r['attempted_commands'] for r in results) and
        report['physical_env_step_returns']==sum(r['returned_commands'] for r in results) and
        report['planned_selected_path_steps']==sum(p['planned_commands'] for p in plans.values()) and
        report['attempted_terminal_encode_samples']==sum(e['attempted_encode_samples'] for e in enc.values()),'nonrefunded actual total accounting')
    stats=report['encoding_stats'];require(stats['forbidden_predict_attempts']==0 and 0<=stats['failed_encode_calls']<=stats['encode_calls']==stats['encode_samples']<=8,'endpoint neural receipt bounds')
    require(stats['encode_calls']<=report['attempted_terminal_encode_samples'],'no unjournaled endpoint forward')
    if report['status']=='guarded_calibration_recorded':
        require(not report['failures'] and all(r['status'] in ('completed','observation_unavailable','environment_terminated') for r in results),'successful status cannot hide branch errors')
        if enc:require(all(e['status']=='completed' for e in enc.values()) and stats['encode_calls']==report['attempted_terminal_encode_samples']==8,'successful encoding status/count')
    planner=reader.json(Path(physical_protocol['planning_output'])/'report.json',physical_protocol['planning_report_sha256'])
    entries=[e for r in planner['roots'] for e in r['neural_receipt']['events']]
    inherited=dict(predict_attempts=sum(e['kind']=='predict' for e in entries),
        action_encoder_attempts=sum(e['kind']=='action_encoder' for e in entries),
        predict_context_tokens=sum(e['context_tokens'] for e in entries if e['kind']=='predict'),
        action_encoder_context_tokens=sum(e['context_tokens'] for e in entries if e['kind']=='action_encoder'),
        jacobians=sum(len(r['compilation']) for r in planner['roots']),
        svds=sum(len(r['compilation']) for r in planner['roots']),
        mathematical_vjp_directions=192*sum(len(r['compilation']) for r in planner['roots']),
        physical_visual_encoder_calls=stats['encode_calls'],physical_visual_encoder_samples=stats['encode_samples'],
        shared_charged_once=True,backward_engine_api_calls=None,historical_cache_acquisition_cost='inherited_unknown_not_zero')
    require(planner['resources']['attempted_predict_samples']==inherited['predict_attempts'] and
        planner['resources']['attempted_action_encoder_samples']==inherited['action_encoder_attempts'],'inherited actual planner resource ledger')
    result=dict(schema=NAME+'.report',status='descriptive_arithmetic_completed',rows=results,**summarize(results,physical_protocol['roots']),
        source_physical_status=report['status'],source_physical_report_sha256=expected_report_sha,
        physical_env_step_attempts=report['physical_env_step_attempts'],physical_env_step_returns=report['physical_env_step_returns'],
        attempted_terminal_encode_samples=report['attempted_terminal_encode_samples'],
        upstream_planner_report_sha256=physical_protocol['planning_report_sha256'],
        inherited_known_cached_runtime_cost=inherited,
        arithmetic_verification_overhead=dict(exporter_saved_array_svds=8,neural_calls=0,physical_steps=0),contract=contract(),
        P1_admission=False,P3_admission=False,efficacy_claim=False)
    reader.recheck();return result


def prepare(root,output,physical_run=None):
    root=Path(root).resolve();output=Path(output).absolute();require(not output.exists(),'exclusive NEW auditor preparation')
    bound=physical.validate_prepared(root/PHYSICAL_PREPARED,PHYSICAL_SHA)
    sources=physical.planning.prior.physical.source_closure(root,[Path(__file__).resolve(),root/'tests'/('test_'+NAME+'.py'),root/'innerdocs'/(NAME+'.md')])
    hashes={str(p):physical.durable.digest(p) for p in sources}
    protocol=dict(schema=NAME+'.protocol',contract=contract(),physical_prepared=str(root/PHYSICAL_PREPARED),physical_prepare_sha256=PHYSICAL_SHA,
        physical_run=str(Path(physical_run).absolute() if physical_run else root/DEFAULT_RUN),roots=bound['roots'],sources_sha256=hashes,
        preparation_scope='post-execution descriptive arithmetic implementation; not prospective efficacy preregistration')
    output.mkdir(exist_ok=False);write(output/'protocol.json',protocol)
    seals=[create_seal(files=sources[i:i+40],args_json=io.canonical({'auditor':NAME}),output_dir=output/f'source_seal_{i//40}') for i in range(0,len(sources),40)]
    receipt=dict(status='prepared_arithmetic_only',protocol_sha256=physical.durable.digest(output/'protocol.json'),source_seals=seals,new_neural_calls=0,new_physical_steps=0)
    write(output/'prepare.json',receipt);return receipt


def validate_prepared(prepared,expected_sha):
    prepared=Path(prepared).absolute();require(physical.durable.digest(prepared/'prepare.json')==expected_sha,'auditor prepare SHA')
    receipt=io.decode((prepared/'prepare.json').read_bytes());require(physical.durable.digest(prepared/'protocol.json')==receipt['protocol_sha256'],'auditor protocol SHA')
    p=io.decode((prepared/'protocol.json').read_bytes());require(p['schema']==NAME+'.protocol' and p['contract']==contract() and p['physical_prepare_sha256']==PHYSICAL_SHA,'frozen arithmetic contract')
    for path,h in p['sources_sha256'].items():require(physical.durable.digest(path)==h,'auditor source drift')
    for seal in receipt['source_seals']:verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    return p


def run(prepared,expected_sha,report_sha,output):
    protocol=validate_prepared(prepared,expected_sha);output=Path(output).absolute();require(not output.exists(),'exclusive NEW arithmetic report output')
    reader=io.Evidence(PROJECT)
    reader.raw(Path(prepared).absolute()/'prepare.json',expected_sha);reader.raw(Path(prepared).absolute()/'protocol.json')
    result=load_and_compute(protocol['physical_run'],protocol,reader,report_sha)
    validate_prepared(prepared,expected_sha);reader.recheck();output.mkdir(exist_ok=False)
    result.update(auditor_prepare_sha256=expected_sha,input_bindings=dict(sorted(reader.hashes.items())))
    write(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);action=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare','verify','run'):action.add_argument('--'+name,action='store_true')
    parser.add_argument('--project-root',type=Path,default=PROJECT);parser.add_argument('--physical-run',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha256');parser.add_argument('--report-sha256')
    args=parser.parse_args()
    if args.prepare:
        require(args.output is not None,'new preparation output');result=prepare(args.project_root,args.output,args.physical_run)
    else:
        require(args.prepared is not None and args.prepare_sha256 is not None,'reviewed auditor preparation receipt')
        if args.verify:validate_prepared(args.prepared,args.prepare_sha256);result=dict(status='verified')
        else:
            require(args.report_sha256 is not None and args.output is not None,'actual report hash and new arithmetic output')
            result=run(args.prepared,args.prepare_sha256,args.report_sha256,args.output)
    if args.run:
        print(io.canonical(dict(status=result['status'],output=str(args.output),full_same_horizon_pairs=result['full_same_horizon_pair_count'],P3_admission=False)))
    else:print(io.canonical(result))


if __name__=='__main__':main()
