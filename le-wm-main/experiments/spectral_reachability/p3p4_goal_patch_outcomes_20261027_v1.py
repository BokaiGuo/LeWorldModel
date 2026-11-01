"""Independent eight-arm saved-data audit; no neural or simulator execution."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p3p4_goal_patch_physical_20261027_v1 as physical
from experiments.spectral_reachability import p3_horizon_residual_outcomes_20261027_v4 as arithmetic
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal

NAME='p3p4_goal_patch_outcomes_20261027_v1'
PROJECT=Path(__file__).resolve().parents[2]
PREPARED='results/p3p4_goal_patch_physical_20261027_v1_prepared01'
PREPARE_SHA='a389f79ac08a497fb95815c9a47682574880f0256161bbb60f629f7ba3e0cacf'
RUN='results/p3p4_goal_patch_physical_20261027_v1_main01'
io,require,write=physical.io,physical.require,physical.write
METRICS=arithmetic.METRICS[:-2]+('predicted_old_Q_cost','predicted_goal_patch_Q_cost','actual_old_Q_cost','actual_goal_patch_Q_cost')
array=arithmetic.array


def contract():
    return dict(arms=list(physical.replay.ARMS),rootarms=8,delta='goal_patch_Q minus old_Q; lower better except success',
        strata=['full_H2_pair','full_H3_pair','full_mixed_H2_H3','partial_same_steps','partial_or_unmatched'],
        pairing='stable root+arm+source selection+horizon; exact entire initial context identity',
        mixed='descriptive per-root deltas retained separately, NOT same-H or efficacy aggregation',
        partial='observed endpoint separate; no full-path promotion, stale Q forbidden',
        Q='same original goal0; independently recompute old and new Q for saved predicted and fresh actual vectors',
        official='original labels; object pose xy+20*wrapped angle; no changed evaluator',
        reversals='strict opposite nonzero signs; exact ties and missing values explicit',
        inference=False,root_cause_established=False,total_resource_matched=False,
        new_neural_calls=0,new_physical_steps=0,P1_admission=False,P3_admission=False,P4_admission=False,efficacy_claim=False)


def q(vector,goal,projection):
    return float(np.mean((array(projection,(192,192))@(array(vector,(192,))-array(goal,(192,))))**2))


def branch(plan,row,encoding,reference):
    physical.validate_plan(plan)
    require(all(row[k]==plan[k] for k in ('root_index','root_sha256','method','selection_sha256','planned_horizon_tokens','planned_commands')) and row['repetition']==0,'root-arm/selection identity')
    a,b=(io.count(row[k]) for k in ('physical_env_step_attempts','physical_env_step_returns'))
    require(b<=a<=plan['planned_commands'] and row['attempted_commands']==a and row['completed_commands']==b,'attempt/return accounting')
    trace=row['trace'];require(len(trace)<=b and [t['primitive_step'] for t in trace]==list(range(1,len(trace)+1)) and [t['command'] for t in trace]==plan['commands'][:len(trace)],'actual observed action prefix')
    full=row['status']=='completed' and row['full_horizon_completed'] is True and b==len(trace)==plan['planned_commands'] and row['endpoint_fresh'] is True and not row.get('failures')
    require(not row['full_horizon_completed'] or full,'no partial promotion to full')
    if row['status']=='completed':require(full,'completed status must be full')
    predicted={}
    for name,proj in (('old',plan['old_projection']),('goal_patch',plan['projection'])):
        cost=q(plan['selected_predicted_context']['latent_history'][-1],plan['goal'],proj)
        arithmetic.close(plan['predicted_'+name+'_Q_cost'],cost,'predicted '+name+' Q');predicted['predicted_'+name+'_Q_cost']=cost
    identity=row.get('context_identity');observed=None
    if identity is not None:
        require(all(identity[k]==reference[k] for k in ('history_rgb_sha256','goal_rgb_sha256')),'root/goal RGB reference')
        warmup=physical.helpers.validate_reset_history(identity['raw_history'],plan['predictor_context']['action_history'])
        require(warmup['normalized_between_history']==identity['normalized_between_history'],'exact incoming tokens')
        goal=array(identity['goal'],(7,));initial=array(identity['state'],(7,))
        if row.get('final_state') is not None:
            state=array(row['final_state'],(7,));require(state.tolist()==(trace[-1]['state'] if trace else initial.tolist()),'last observed endpoint state')
            observed=dict(arithmetic.labels(state,goal),**arithmetic.pose(state,goal))
            for kind in ('official','pose'):
                expected=arithmetic.labels(state,goal) if kind=='official' else arithmetic.pose(state,goal)
                saved=row.get(kind,{})
                require(set(saved)<=set(expected) and (not full or set(saved)==set(expected)),'official/pose metric census')
                for key,v in saved.items():
                    if type(expected[key]) is bool:require(type(v) is bool and v==expected[key],'success label')
                    else:arithmetic.close(v,expected[key],kind+'.'+key)
    if full:require(observed is not None and identity is not None,'bound full observed endpoint')
    actual=dict(actual_old_Q_cost=None,actual_goal_patch_Q_cost=None)
    if encoding is not None:
        require(all(encoding[k]==plan[k] for k in ('root_sha256','method','selection_sha256','planned_horizon_tokens')) and encoding['endpoint_scope']==row['endpoint_scope'],'encoding root/selection/horizon/scope')
        require(io.count(encoding['attempted_encode_samples'])<=1,'one endpoint attempt per arm')
        if encoding['status']=='completed':
            require(row['endpoint_fresh'] is True and encoding['attempted_encode_samples']==1 and encoding['failure'] is None and encoding['terminal_rgb_sha256']==row['terminal_rgb_sha256'],'fresh actual endpoint only')
            require(encoding['whole_bank_sha256']==plan['whole_bank_sha256'] and encoding['metric_sha256']==plan['metric_sha256'],'whole bank and metric identity')
            for name,proj in (('old',plan['old_projection']),('goal_patch',plan['projection'])):
                key='actual_'+name+'_Q_cost';actual[key]=q(encoding['terminal_vector'],plan['goal'],proj)
                arithmetic.close(encoding[key],actual[key],key)
                arithmetic.close(encoding['predicted_'+name+'_Q_cost'],predicted['predicted_'+name+'_Q_cost'],'encoding predicted '+name)
    metrics=None if observed is None else dict(observed,**predicted,**actual)
    return dict(root_index=plan['root_index'],root_sha256=plan['root_sha256'],method=plan['method'],selection_sha256=plan['selection_sha256'],
        selected_horizon=plan['planned_horizon_tokens'],status=row['status'],full_selected_horizon=full,endpoint_fresh=row['endpoint_fresh'],endpoint_scope=row['endpoint_scope'],
        attempted_commands=a,returned_commands=b,observed_commands=len(trace),returned_without_observation=b-len(trace),initial_context=deepcopy(identity),
        observed_endpoint_metrics=metrics,full_metrics=metrics if full else None,predicted_metrics=predicted,actual_Q=actual,
        encoding_status=encoding['status'] if encoding else 'not_requested',failures=deepcopy(row.get('failures',[])))


def summarize(rows,roots):
    index=io.index(rows,('root_sha256','method'))
    require(len(roots)==len(set(roots))==4 and set(index)=={(r,m) for r in roots for m in physical.replay.ARMS},'strict eight-arm four-root pairing')
    pairs=[]
    for i,root in enumerate(roots):
        a,b=(index[root,m] for m in physical.replay.ARMS);require(a['root_index']==b['root_index']==i,'stable root indices')
        bound=a['initial_context'] is not None and b['initial_context'] is not None
        if bound:require(physical.same(a['initial_context'],b['initial_context']),'full paired initial context identity')
        full=bound and a['full_selected_horizon'] and b['full_selected_horizon'];same=a['selected_horizon']==b['selected_horizon']
        partial=bound and a['endpoint_fresh'] and b['endpoint_fresh'] and a['returned_commands']==b['returned_commands']
        label=(f'full_H{a["selected_horizon"]}_pair' if same else 'full_mixed_H2_H3') if full else ('partial_same_steps' if partial else 'partial_or_unmatched')
        delta=None
        if full:
            delta={k:None if a['full_metrics'][k] is None or b['full_metrics'][k] is None else float(b['full_metrics'][k])-float(a['full_metrics'][k]) for k in METRICS}
        observed_delta=None
        if partial and all(r['observed_endpoint_metrics'] is not None for r in (a,b)):
            observed_delta={k:None if a['observed_endpoint_metrics'][k] is None or b['observed_endpoint_metrics'][k] is None else float(b['observed_endpoint_metrics'][k])-float(a['observed_endpoint_metrics'][k]) for k in METRICS if not k.startswith('predicted_')}
        reversals=None if delta is None else {name:{target:arithmetic.reversal(delta['predicted_'+name+'_Q_cost'],delta[target]) for target in ('actual_'+name+'_Q_cost','pose_cost','official_state_distance')} for name in ('old','goal_patch')}
        pairs.append(dict(root_index=i,root_sha256=root,classification=label,selected_horizons=[a['selected_horizon'],b['selected_horizon']],
            full_same_horizon=bool(full and same),new_minus_old_full_endpoint_descriptive=delta,partial_equal_step_observed_delta=observed_delta if not full else None,
            predicted_vs_actual_directions=reversals,actual_steps=[a['returned_commands'],b['returned_commands']],arm_statuses=[a['status'],b['status']],
            scope='same-H full only in aggregates; mixed-H per-root duration-confounded description; partial separate'))
    strata={}
    for label in contract()['strata']:
        subset=[p for p in pairs if p['classification']==label]
        strata[label]=dict(root_ids=[p['root_sha256'] for p in subset],n=len(subset),same_h_full_summary=None)
        if label in ('full_H2_pair','full_H3_pair'):
            strata[label]['same_h_full_summary']={k:arithmetic.describe([p['new_minus_old_full_endpoint_descriptive'][k] for p in subset if p['new_minus_old_full_endpoint_descriptive'][k] is not None],k in arithmetic.HIGHER) for k in METRICS}
    return dict(pairs=pairs,strata=strata,full_same_horizon_pair_count=sum(p['full_same_horizon'] for p in pairs))


def audit_artifacts(directory,p,r,reader):
    stem=f'root_{p["root_index"]}_{p["method"]}_replay0'
    saved=reader.json(directory/(stem+'.physical.json'))
    require(saved==(r if r['status']=='blocked' else {k:v for k,v in r.items() if k not in ('endpoint_fresh','endpoint_scope')}),'durable physical receipt')
    if r['status']=='blocked':
        require(r['physical_env_step_attempts']==r['physical_env_step_returns']==0 and not r['endpoint_fresh'] and not r['full_horizon_completed'],'blocked has no endpoint/work');return
    require(reader.json(directory/(stem+'.endpoint.json'))==r,'durable entire endpoint annotation')
    require(Path(r['journal_path'])==directory/(stem+'.events.jsonl'),'branch journal path')
    physical.verify_branch(p,r)
    events=[io.decode(s) for s in reader.raw(Path(r['journal_path']),r['journal_sha256']).splitlines()]
    attempts=[e for e in events if e['kind']=='attempt'];returned=[e for e in events if e['kind']=='step_returned'];observed=[e for e in events if e['kind']=='observation']
    kinds=[]
    if events:require(events[0]['kind']=='initial' and events[0]['state']==r['context_identity']['state'],'initial state');kinds.append('initial')
    for i in range(len(attempts)):
        kinds.append('attempt')
        if i<len(returned):kinds.append('step_returned')
        if i<len(observed):kinds.append('observation')
    require([e['kind'] for e in events]==kinds,'strict physical chronology')
    pixels=[];images=[]
    from PIL import Image
    for e in events:
        if 'rgb_path' not in e:continue
        step=e.get('primitive_step',0);path=Path(e['rgb_path']);require(path==directory/f'{stem}_rgb_{step}.png','exact root-arm-step image')
        reader.raw(path,e['rgb_file_sha256'])
        with Image.open(path) as im:rgb=np.asarray(im)
        require(rgb.shape==(224,224,3) and rgb.dtype==np.uint8,'physical RGB layout')
        pixels.append(dict(primitive_step=step,canonical_agent_pixels=int(np.count_nonzero(np.all(rgb==[65,105,225],axis=-1)))));images.append(e)
    if 'guard' in r:
        g=r['guard'];require(g['root_id']==p['root_sha256'] and g['primitive_commands_authorized']==len(attempts) and g['observations']==pixels,'saved-RGB guard accounting')
        require(all(pixels[i]['primitive_step']==i and pixels[i]['canonical_agent_pixels']>0 for i in range(len(attempts))),'guard before EACH dispatch')
        require(g['abstained']==bool(pixels and pixels[-1]['canonical_agent_pixels']==0),'guard abstention')
    fresh=bool(images and images[-1].get('primitive_step',0)==len(returned) and len(attempts)==len(returned) and r['status'] in ('completed','observation_unavailable','environment_terminated'))
    scope=(('full_H' if r['full_horizon_completed'] else 'partial_H')+str(p['planned_horizon_tokens'])) if fresh else 'unknown_or_stale_endpoint'
    require(r['endpoint_fresh']==fresh and r['endpoint_scope']==scope,'independent fresh/partial scope')
    if images:require(r['terminal_rgb_path']==images[-1]['rgb_path'] and r['terminal_rgb_sha256']==images[-1]['rgb_file_sha256'],'last durable endpoint')


def compute(directory,report_sha,reader):
    directory=Path(directory).absolute();r=reader.json(directory/'report.json',report_sha);e=reader.json(directory/'execution_protocol.json')
    require(e['prepared']==str(PROJECT/PREPARED) and e['prepare_sha256']==r['consumer_prepare_sha256']==PREPARE_SHA,'original physical preparation')
    p=physical.validate_prepared(e['prepared'],PREPARE_SHA)
    reader.raw(Path(e['prepared'])/'prepare.json',PREPARE_SHA);reader.raw(Path(e['prepared'])/'protocol.json')
    for path,h in p['sources_sha256'].items():reader.raw(Path(path),h)
    reader.raw(Path(p['replay_output'])/'report.json',p['replay_report_sha256'])
    require(r['schema']==physical.NAME+'.report' and r['status'] in ('guarded_calibration_recorded','incomplete') and r['contract']==physical.contract(),'terminal physical report')
    require(r['source_verified'] is True and r['model_unchanged'] is True and all(r[k] is False for k in ('P1_admission','P3_admission','P4_admission','efficacy_claim')) and r['new_predict_calls']==0,'scope/model/source integrity')
    require(r['replay_report_sha256']==e['replay_report_sha256']==p['replay_report_sha256'] and r['inherited_resources']==p['inherited_resources'] and r['replay_shared_resources']==p['plans'][0]['replay_shared_resources'],'fresh replay and inherited resource contract')
    durable=reader.json(directory/'physical_report.json',r['physical_report_sha256'])
    require(durable['rows']==r['rows'] and durable['consumer_prepare_sha256']==PREPARE_SHA and durable['contract']==physical.contract() and durable['replay_report_sha256']==p['replay_report_sha256'],'durable before-neural physical report')
    plans=io.index(p['plans'],('root_sha256','method'));rows=io.index(r['rows'],('root_sha256','method'));enc=io.index(r['encodings'],('root_sha256','method'))
    require(len(plans)==8 and set(plans)==set(rows),'all eight physical arms')
    require((e['encode_terminals'] is True and set(enc)==set(plans)) or (e['encode_terminals'] is False and not enc),'encoding census')
    results=[]
    for key,plan in plans.items():
        row=rows[key];audit_artifacts(directory,plan,row,reader)
        if key in enc:require(reader.json(directory/f'root_{plan["root_index"]}_{plan["method"]}_replay0.encoding.json')==enc[key],'durable encoding receipt')
        results.append(branch(plan,row,enc.get(key),p['references'][key[0]]))
    require(r['pairs']==physical.comparisons(p['plans'],r['rows']),'saved pairing classifier')
    stats=r['encoding_stats'];require(stats['forbidden_predict_attempts']==0,'no forbidden predicts')
    if enc:
        journal=[io.decode(s) for s in reader.raw(directory/'encoding.events.jsonl').splitlines()]
        require([{k:v for k,v in x.items() if k!='kind'} for x in journal if x.get('kind')=='endpoint_result']==r['encodings'],'durable encoding outputs')
        require([(x['root_sha256'],x['method']) for x in journal if x.get('kind')=='endpoint_encode']==[(x['root_sha256'],x['method']) for x in r['encodings'] if x['attempted_encode_samples']==1],'encoding attempts')
        for kind in ('encoder','projector'):
            entries=[x for x in journal if x.get('kind')==kind];starts=[x for x in entries if not x['completed'] and 'error' not in x];ends=[x for x in entries if x['completed']]
            require([x['index'] for x in starts]==list(range(len(starts))) and len(starts)==stats[kind+'_attempts']<=8 and len(ends)==stats[kind+'_completed']<=len(starts),'encoder/projector actual accounting')
            require(len(entries)==2*len(starts),'neural attempts must retain completion/error')
            for start in starts:
                pair=[x for x in entries if x['index']==start['index']];require(pair[0]==start and len(pair)==2 and (pair[1]['completed'] or 'error' in pair[1]),'durable module pair')
        require(stats['encoder_attempts']<=sum(x['attempted_encode_samples'] for x in enc.values()) and
            stats['projector_attempts']<=stats['encoder_completed'],'no unjournaled/early endpoint neural dispatch')
    else:require(all(stats[k]==0 for k in stats),'no encoding means no neural work')
    require(r['physical_env_step_attempts']==sum(x['attempted_commands'] for x in results) and r['physical_env_step_returns']==sum(x['returned_commands'] for x in results) and r['planned_selected_path_steps']==sum(x['planned_commands'] for x in p['plans'])==90,'actual physical resource accounting')
    if r['status']=='guarded_calibration_recorded':
        require(not r['failures'] and all(x['status'] in ('completed','observation_unavailable','environment_terminated') for x in results),'no hidden branch failure')
        if enc:require(all(x['status']=='completed' for x in enc.values()) and all(stats[k]==8 for k in ('encoder_attempts','encoder_completed','projector_attempts','projector_completed')),'complete encoding status/cost')
    roots=list(dict.fromkeys(x['root_sha256'] for x in p['plans']))
    result=dict(schema=NAME+'.report',status='saved_data_arithmetic_completed',rows=results,**summarize(results,roots),
        source_physical_report_sha256=report_sha,source_physical_status=r['status'],replay_report_sha256=p['replay_report_sha256'],
        inherited_resources=r['inherited_resources'],replay_shared_resources=r['replay_shared_resources'],physical_encoding_stats=stats,
        physical_env_step_attempts=r['physical_env_step_attempts'],physical_env_step_returns=r['physical_env_step_returns'],
        verification_overhead=dict(saved_array_svds=48,new_neural_calls=0,new_physical_steps=0),contract=contract(),P1_admission=False,P3_admission=False,P4_admission=False,efficacy_claim=False)
    reader.recheck();return result


def prepare(root,output):
    root=Path(root).resolve();output=Path(output).absolute();require(not output.exists(),'exclusive NEW audit preparation')
    physical.validate_prepared(root/PREPARED,PREPARE_SHA)
    sources=physical.replay.metric.old.prior.physical.source_closure(root,[Path(__file__).resolve(),root/'tests'/('test_'+NAME+'.py'),root/'innerdocs'/(NAME+'.md')])
    p=dict(schema=NAME+'.protocol',contract=contract(),physical_prepared=str(root/PREPARED),physical_prepare_sha256=PREPARE_SHA,physical_run=str(root/RUN),sources_sha256={str(s):io.sha(s.read_bytes()) for s in sources})
    output.mkdir(exist_ok=False);write(output/'protocol.json',p)
    seals=[create_seal(files=sources[i:i+40],args_json=io.canonical(dict(auditor=NAME)),output_dir=output/f'source_seal_{i//40}') for i in range(0,len(sources),40)]
    receipt=dict(status='prepared_saved_data_only',protocol_sha256=io.sha((output/'protocol.json').read_bytes()),source_seals=seals)
    write(output/'prepare.json',receipt);return receipt


def validate_prepared(prepared,sha):
    prepared=Path(prepared).absolute();require(io.sha((prepared/'prepare.json').read_bytes())==sha,'auditor receipt SHA');r=physical.read_json(prepared/'prepare.json')
    require(io.sha((prepared/'protocol.json').read_bytes())==r['protocol_sha256'],'auditor protocol bytes');p=physical.read_json(prepared/'protocol.json')
    require(p['schema']==NAME+'.protocol' and p['contract']==contract() and p['physical_prepare_sha256']==PREPARE_SHA and p['physical_prepared']==str(PROJECT/PREPARED) and p['physical_run']==str(PROJECT/RUN),'audit contract')
    for path,h in p['sources_sha256'].items():require(io.sha(Path(path).read_bytes())==h,'audit source drift')
    for s in r['source_seals']:verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    return p


def run(prepared,sha,report_sha,output):
    p=validate_prepared(prepared,sha);output=Path(output).absolute();require(not output.exists(),'exclusive NEW audit output')
    reader=io.Evidence(PROJECT);reader.raw(Path(prepared).absolute()/'prepare.json',sha);reader.raw(Path(prepared).absolute()/'protocol.json')
    result=compute(p['physical_run'],report_sha,reader);validate_prepared(prepared,sha);reader.recheck()
    result.update(auditor_prepare_sha256=sha,input_bindings=dict(sorted(reader.hashes.items())))
    output.mkdir(exist_ok=False);write(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group(required=True)
    for n in ('prepare','verify','run'):g.add_argument('--'+n,action='store_true')
    parser.add_argument('--output',type=Path);parser.add_argument('--prepared',type=Path);parser.add_argument('--prepare-sha256');parser.add_argument('--report-sha256');a=parser.parse_args()
    if a.prepare:require(a.output is not None,'new preparation output');r=prepare(PROJECT,a.output)
    else:
        require(a.prepared is not None and a.prepare_sha256 is not None,'reviewed auditor SHA')
        if a.verify:validate_prepared(a.prepared,a.prepare_sha256);r=dict(status='verified')
        else:require(a.output is not None and a.report_sha256 is not None,'actual report SHA and new output');r=run(a.prepared,a.prepare_sha256,a.report_sha256,a.output)
    print(io.canonical({k:v for k,v in r.items() if k in ('status','protocol_sha256','full_same_horizon_pair_count','strata')}))


if __name__=='__main__':main()
