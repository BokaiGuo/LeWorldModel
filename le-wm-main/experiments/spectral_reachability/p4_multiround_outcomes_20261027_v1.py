"""Bounded saved multiround journal audit and original task-label arithmetic.

No model, environment, rollout, physics, action selection or threshold fitting.
Incomplete endpoints remain incomplete, independently of integrity classification.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import sysconfig

import numpy as np
from PIL import Image

from experiments.spectral_reachability import p4_feedback_admission_20261027_v1 as saved
from experiments.spectral_reachability import p4_multiround_feedback_runtime_20261027_v1 as runtime
from experiments.spectral_reachability import audit_pusht_sealed_task_contract as official
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal

TAG = 'p4_multiround_outcomes_20261027_v1'
REPO = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO/'results/p4_multiround_feedback_runtime_20261027_v1_prepare_v1/report.json'
ENV = Path(sysconfig.get_paths()['purelib'])/'stable_worldmodel/envs/pusht/env.py'
ENV_SHA = 'c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2'
POSE_SOURCE = REPO/'experiments/spectral_reachability/run_pusht_d2p_p1_physical_calibration.py'
POSE_SHA = '916d0979838e502720940aca24f8dc717ce256f0db2e3ffdeccb457f3912e1de'
require = saved.require
MAX_EVENTS = 2000
MAX_JOURNAL_BYTES = 32*1024*1024


def label(state, goal):
    state, goal = saved.array(state,(7,)).astype(float), saved.array(goal,(7,)).astype(float)
    xy = float(np.linalg.norm(state[2:4]-goal[2:4]))
    angle = float(abs(np.arctan2(np.sin(state[4]-goal[4]), np.cos(state[4]-goal[4]))))
    return dict(official=official.labels(state,goal),
        pose=dict(xy_distance=xy,angle_distance=angle,pose_cost=xy+20.0*angle))


def metric_sources(inputs):
    """Bind reviewed evaluator/pose contracts BEFORE opening outcome traces."""
    require(inputs.hash(ENV)==ENV_SHA, 'unsupported installed official evaluator source')
    text = inputs.raw(POSE_SOURCE).decode()
    fn = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_pose_label')
    require(hashlib.sha256(ast.get_source_segment(text,fn).encode()).hexdigest()==POSE_SHA,
        'original pose evaluator changed')
    inputs.hash(Path(official.__file__))
    return dict(official_source=str(ENV), official_source_sha256=ENV_SHA,
        official_semantics='unchanged norm(goal[:4]-state[:4])<20 and min(abs(angle_delta),2*pi-abs(angle_delta))<pi/9; state_distance includes all seven coordinates',
        pose_source=str(POSE_SOURCE), pose_function_sha256=POSE_SHA,
        pose_semantics='object_xy_norm + 20*abs(atan2(sin(angle_delta),cos(angle_delta)))',
        retuned=False, simulation_calls=0, neural_calls=0)


def load_prepared(inputs, folder):
    p, r = inputs.json(folder/'protocol.json'), inputs.json(folder/'prepare_receipt.json')
    digest = inputs.hash(folder/'protocol.json')
    require(r['status']=='prepared_not_run' and r['protocol_sha256']==digest and
        r['census_sha256']==inputs.hash(folder/'planned_census.json'), 'runtime prepared binding differs')
    require(p['schema']==runtime.TAG and p['contract']==runtime.contract() and p['slots']==runtime.roster(),
        'fixed runtime contract changed')
    require(inputs.json(folder/'planned_census.json')==dict(slots=runtime.roster(), **runtime.census(runtime.roster())),
        'initial census changed')
    require(r['source_seals'], 'runtime source seals missing')
    for seal in r['source_seals']:
        inputs.seal(seal)
        require(inputs.json(Path(seal['output_dir'])/'args.json')['protocol_sha256']==digest,
            'source seal/protocol differs')
    require(p['bindings'], 'bound original data/witnesses required')
    for path, value in p['bindings'].items():
        require(inputs.hash(path)==value, 'runtime bound input changed: '+path)
    original = inputs.json(Path(p['source'])/'protocol.json')
    case = next(c for c in original['cases'] if c['role']=='selected')
    require(p['root_sha256']==runtime.ROOT and type(p['root_index']) is int and p['root_index']==1 and
        all(p[k]==original[k] for k in ('initial_context','goal','q','root_rgb','policy')) and
        all(p[k]==case[k] for k in ('prefix','between_raw','prefix_rgb','saved_predicted_prefix')),
        'original fixed root/input/prefix/metric binding differs')
    cap = inputs.json(original['capability'])
    require(p['physical_goal']==next(row for row in cap['rows'] if row['method']=='selected_zero')['context_identity']['goal'],
        'original fixed physical goal changed')
    require(inputs.json(folder/'launch_receipt.json')['protocol_sha256']==digest, 'launch/protocol differs')
    return p, r


def image(inputs, event):
    require(inputs.hash(event['rgb_path'])==event['rgb_sha256'], 'journal RGB file SHA differs')
    with Image.open(io.BytesIO(inputs.raw(event['rgb_path']))) as im:
        rgb = np.asarray(im).copy()
    require(rgb.shape==(224,224,3) and rgb.dtype==np.uint8, 'official uint8 RGB required')
    require(saved.rgb_sha(rgb)==event['pixel_sha256'], 'journal RGB pixels differ')
    return rgb


def one(events, kind, **fields):
    values = [e for e in events if e['kind']==kind and all(e.get(k)==v for k,v in fields.items())]
    require(len(values)==1, f'exactly one {kind}/{fields} required')
    return values[0]


def normalized_history(c, between):
    _, history = saved.context(c)
    require(np.array_equal(history,saved.normalize(np.asarray(between))), 'history/action normalization differs')


def classify(row, observed, attempts, returns):
    """Runtime guard abort != scientific negative != corrupted evidence."""
    if row['status']=='blocked':
        return 'blocked'
    full = (row['status']=='completed' and len(attempts)==len(returns)==20 and
        [v['primitive_step'] for v in observed]==list(range(1,21)) and
        row['completed_packets']==4 and row['failure'] is None and not row['guard']['abstained'])
    if full:
        return 'complete20'
    if (row.get('failure') or {}).get('type')=='ObservationUnavailable' and row['guard']['abstained']:
        return 'guard_aborted'
    if observed and (observed[-1]['terminated'] or observed[-1]['truncated']):
        return 'environment_terminated'
    return 'execution_or_integrity_failure'


def branch(inputs, p, row, events, slots, root_frames):
    arm = row['arm']
    require(row['root_sha256']==runtime.ROOT, 'branch root differs')
    ev = [e for e in events if e.get('arm')==arm]
    obs = [e for e in ev if e['kind']=='observation']
    attempts = [e for e in ev if e['kind']=='step_attempt']
    returns = [e for e in ev if e['kind']=='step_returned']
    require(len(attempts)<=20 and len(returns)<=len(attempts) and len(obs)<=len(returns), 'physical work exceeds fixed plan')
    for values in (obs,attempts,returns):
        require([e['primitive_step'] for e in values]==list(range(1,len(values)+1)), 'step sequence gap/duplicate')
    if row['status']=='blocked':
        require(not obs and not attempts and not returns, 'blocked branch has dispatched steps')
        return dict(arm=arm, classification='blocked', last_observed=None, exact20=None, boundaries=[])
    require(row['selected_attempts']==len(attempts) and row['selected_returns']==len(returns), 'branch physical count differs')
    if not any(e['kind']=='initial' for e in ev):
        require(not obs and not attempts and not returns and row['status']=='aborted' and row['failure'] is not None,
            'missing initial observation after dispatch')
        require(one(ev,'arm_complete')['row']==row, 'initial failure receipt differs')
        return dict(arm=arm,classification='initialization_failed',last_observed=None,exact20=None,
            boundaries=[],failure=row['failure'],selected_attempts=0,selected_returns=0)
    init = one(ev,'initial')
    initial_rgb = image(inputs, init)
    require(np.array_equal(initial_rgb,root_frames[-1]), 'initial RGB differs from original root archive')
    reset = one(ev,'reset_identity')
    require(reset['goal']==p['physical_goal'] and
        saved.rgb_sha(root_frames)==reset['identity']['history_rgb_sha256'] and
        np.array_equal(np.repeat(np.asarray(reset['history'])[1:,None,:],5,axis=1),p['between_raw']),
        'reset goal/RGB/raw history differs')
    rgb = {0:initial_rgb}
    for e in obs:
        rgb[e['primitive_step']] = image(inputs,e)
    labels = [dict(primitive_step=0,state=init['state'],**label(init['state'],p['physical_goal']))]
    for e in obs:
        labels.append(dict(primitive_step=e['primitive_step'],state=e['state'],**label(e['state'],p['physical_goal'])))
        require(abs(e['reward']+labels[-1]['official']['official_state_distance'])<=1e-8,
            'saved reward/official state distance differs')
        a, r = attempts[e['primitive_step']-1], returns[e['primitive_step']-1]
        require(a['event_index']<r['event_index']<e['event_index'], 'attempt/return/observation order differs')
    guard = row['guard']
    require(guard['root_id']==runtime.ROOT and guard['primitive_commands_authorized']==len(attempts), 'guard authorization differs')
    pixels = {step:int(np.count_nonzero(np.all(value==[65,105,225],axis=-1))) for step,value in rgb.items()}
    require(guard['observations']==[dict(primitive_step=step,canonical_agent_pixels=pixels[step]) for step in sorted(rgb)],
        'guard RGB counts/observation history differs')
    for e in attempts:
        require(pixels.get(e['primitive_step']-1,0)>0, 'command after absent/missing RGB')
    expected_commands = deepcopy_list(p['prefix'])
    between = np.asarray(p['between_raw'])
    frame_history = [saved.rgb_sha(f) for f in root_frames]
    predicted = None
    boundaries = []
    for boundary in runtime.BOUNDARIES:
        slot = next(s for s in slots if s['arm']==arm and s['boundary']==boundary)
        begins = [e for e in ev if e['kind']=='boundary_begin' and e['boundary']==boundary]
        selections = [e for e in ev if e['kind']=='selection' and e['boundary']==boundary]
        if not begins:
            require(not selections and slot['status']=='blocked_not_reached', 'unreached slot has a selection')
            continue
        require(boundary in rgb and boundary<=len(obs), 'planning at partial/non-observed boundary')
        if boundary==5:
            gate = one(ev,'prefix_identity')
            with Image.open(io.BytesIO(inputs.raw(p['prefix_rgb']))) as im:
                expected = np.asarray(im).copy()
            require(gate['pixels_exact'] is True and np.array_equal(rgb[5],expected), 'step5 original prefix gate differs')
        previous_packet = expected_commands[boundary-5:boundary]
        require(len(previous_packet)==5, 'prior packet missing')
        frame_history = frame_history[1:]+[saved.rgb_sha(rgb[boundary])]
        between = np.stack([between[1],previous_packet])
        begin = one(ev,'boundary_begin',boundary=boundary)
        require(begin['frames_pixel_sha256']==frame_history and np.array_equal(begin['between_raw'],between),
            'rolling observed RGB/raw packet history differs')
        contexts = [e for e in ev if e['kind']=='planning_context' and e['boundary']==boundary]
        if not contexts:
            require(not selections and slot['status']=='aborted', 'missing context not marked aborted')
            continue
        c = one(ev,'planning_context',boundary=boundary)
        for key in ('observed_context','predicted_context','chosen_context'):
            normalized_history(c[key],between)
        require(c['chosen_context']==c['observed_context' if arm=='actual' else 'predicted_context'] and
            c['control_uses_observed_latents'] is False, 'actual/imagined intervention differs')
        if predicted is None:
            predicted = c['predicted_context']
            runtime.validate_transition(p['initial_context'],predicted)
            require(np.allclose(predicted['latent_history'][-1],p['saved_predicted_prefix'],rtol=0,atol=2e-5),
                'prefix prediction differs from original saved witness')
        require(c['predicted_context']==predicted, 'predicted control history not own previous successor')
        if boundary>5:
            older = boundaries[-1]['observed_context']['latent_history'][1:]
            require(np.allclose(c['observed_context']['latent_history'][:2],older,rtol=0,atol=2e-5),
                'observed older latent history differs from own previous observed boundaries')
        diagnostic = dict(boundary=boundary,observed_context=c['observed_context'],
            chosen_equals_observed=c['chosen_context']==c['observed_context'],
            chosen_equals_predicted=c['chosen_context']==c['predicted_context'],
            observed_minus_predicted_last_l2=float(np.linalg.norm(np.asarray(c['observed_context']['latent_history'][-1])-
                np.asarray(c['predicted_context']['latent_history'][-1]))), selected_candidate_id=None)
        boundaries.append(diagnostic)
        if not selections:
            require(slot['status']=='aborted', 'missing selection not marked aborted')
            continue
        s = one(ev,'selection',boundary=boundary)
        envelope = {k:v for k,v in s.items() if k not in ('kind','event_index','monotonic_seconds','selection_sha256')}
        require(saved.content_sha(envelope)==s['selection_sha256']==slot['selection_sha256'], 'selection envelope SHA differs')
        require(s['root_sha256']==runtime.ROOT and s['context_sha256']==saved.content_sha(c['chosen_context']) and
            s['metric_sha256']==saved.content_sha([p['goal'],p['q']]) and np.array_equal(s['incoming_between_raw'],between),
            'selection root/context/original metric/incoming packet differs')
        plan = s['compiler_plan']
        require(plan['context']==c['chosen_context'] and plan['goal']==p['goal'] and plan['q']==p['q'] and
            plan['prefix']==p['prefix'] and plan['arm']==arm and plan['role']=='selected', 'compiler binding differs')
        winner = runtime.compiler.verify_plan(plan)  # Existing saved bank algebra ONLY.
        # Backend work records bind arm through the unique method, not an arm field.
        derivative = one(events,'derivative_receipt',method=f'{arm}_{boundary}')
        require(derivative['nominal_terminal']==plan['nominal_terminal'] and derivative['jacobian']==plan['jacobian'],
            'durable derivative/selection witness differs')
        candidate_events = [e for e in events if e['kind']=='candidate' and e['method']==f'{arm}_{boundary}']
        durable_candidates = [e['candidate'] for e in candidate_events]
        require(durable_candidates==plan['candidates'], 'durable nonlinear candidate/selection bank differs')
        require(derivative['event_index']<s['event_index'] and all(e['event_index']<s['event_index'] for e in candidate_events),
            'derivative/candidate work was not durable before selection')
        require(s['commands']==winner['commands'][5:] and winner['candidate_id']==slot['selected_candidate_id'],
            'selected commands/candidate differ')
        runtime.validate_transition(c['chosen_context'],s['successor'],winner['predicted_terminal'])
        normalized_history(s['successor'],np.stack([between[1],s['commands']]))
        require(begin['event_index']<c['event_index']<s['event_index'], 'boundary/selection order differs')
        expected_commands += deepcopy_list(s['commands'])
        predicted = s['successor']
        diagnostic.update(selected_candidate_id=winner['candidate_id'], model_Q_cost=winner['nonlinear_terminal_cost'])
        next_attempts = [e for e in attempts if boundary<e['primitive_step']<=boundary+5]
        require(all(s['event_index']<e['event_index'] for e in next_attempts), 'physical dispatch preceded durable selection')
        completes = [e for e in ev if e['kind']=='packet_complete' and e['boundary']==boundary]
        require(len(completes)<=1, 'duplicate packet completion')
        if completes:
            done = completes[0]
            require(boundary+5 in rgb and slot['status']=='completed' and done['next_boundary']==boundary+5 and
                done['predicted_context']==predicted and np.array_equal(done['next_between_raw'],np.stack([between[1],s['commands']])),
                'completed packet state shift differs')
    for e in attempts:
        i = e['primitive_step']-1
        require(i<len(expected_commands) and e['command']==expected_commands[i], 'executed command differs from bound prefix/winner')
    classification = classify(row,obs,attempts,returns)
    if row['status']=='completed':
        require(classification=='complete20', 'claimed completed arm is not complete20')
    final_arm = one(ev,'arm_complete')
    require(final_arm['row']==row, 'durable arm completion/report differs')
    last = labels[-1]
    exact20 = last if classification=='complete20' else None
    return dict(arm=arm,classification=classification, full_horizon_completed=classification=='complete20',
        selected_attempts=len(attempts),selected_returns=len(returns),observed_steps=len(obs),
        endpoint_unobserved_after_return=len(returns)>len(obs), last_observed=last, exact20=exact20,
        endpoint_unobserved_after_attempt=len(attempts)>len(obs),
        prefix=next((v for v in labels if v['primitive_step']==5),None), steps=labels,
        boundaries=boundaries, failure=row['failure'], guard_abstained=guard['abstained'],
        observed_at_step20=20 in rgb, no_force_or_causal_inference=True)


def deepcopy_list(value):
    return [list(v) for v in value]


def compare(rows):
    a, b = [next(r for r in rows if r['arm']==arm) for arm in ('actual','imagined')]
    comparable = a['exact20'] is not None and b['exact20'] is not None
    return dict(exact20_comparable=comparable,
        actual_minus_imagined=None if not comparable else dict(
            official_state_distance=a['exact20']['official']['official_state_distance']-b['exact20']['official']['official_state_distance'],
            official_position_error=a['exact20']['official']['official_position_error']-b['exact20']['official']['official_position_error'],
            object_xy_distance=a['exact20']['pose']['xy_distance']-b['exact20']['pose']['xy_distance'],
            wrapped_angle_distance=a['exact20']['pose']['angle_distance']-b['exact20']['pose']['angle_distance'],
            pose_cost=a['exact20']['pose']['pose_cost']-b['exact20']['pose']['pose_cost']),
        partial_endpoints_pooled=False, policy_identity_is_efficacy_test=False,
        significance_claim=False,P1_admission=False,P4_admission=False)


def work_records(report, events):
    expected = []
    for arm in runtime.ARMS:
        expected.append((arm+'_prefix','evaluation',1))
        for b in runtime.BOUNDARIES:
            for suffix,phase,n in (('_compile','planning',2),('_bank','planning',10),('_independent_replay','evaluation',1)):
                expected.append((f'{arm}_{b}'+suffix,phase,n))
    indices=[]
    for r in report['predict']:
        matches=[(i,p,n) for i,(method,p,n) in enumerate(expected) if method==r['method']]
        require(len(matches)==1, 'unknown predictor work scope')
        index,phase,n=matches[0];indices.append(index)
        for field in ('predict_calls','predict_samples','failed_predict_calls'):
            saved.integer(r[field])
        require(r['phase']==phase and r['horizon_tokens']==1 and r['predict_calls']==r['predict_samples']<=n and
            r['failed_predict_calls']<=r['predict_calls'] and
            r['context_lengths']==({} if r['predict_calls']==0 else {'3':r['predict_calls']}),
            'predictor phase/count/context/horizon differs')
        if r['completed']:
            require(r['predict_calls']==n and r['failed_predict_calls']==0, 'completed predictor scope has wrong work')
    require(indices==sorted(set(indices)), 'duplicate/out-of-order predictor scopes')
    previous={k:0 for k in report['cost']}
    for event in events:
        if 'cost' not in event:
            continue
        for key,n in event['cost'].items():
            saved.integer(n)
            require(key in previous and previous[key]<=n<=report['cost'][key], 'nonmonotonic/unknown actual work counter')
            previous[key]=n
    require(previous==report['cost'], 'final cumulative work differs')
    if report['status']=='calibration_recorded':
        require(len(indices)==20, 'full run requires all twenty predictor scopes')
        expected_cost=dict(predict_samples=80,selected_attempts=40,selected_returns=40,
            encode_attempts=6,encode_completed=6,encode_frames_attempted=18,visual_encoder_attempts=6,
            visual_encoder_completed=6,visual_encoder_frames_attempted=18,visual_encoder_frames_completed=18,
            jacobian_attempts=6,jacobian_completed=6,bank_attempts=6,bank_completed=6,
            action_encoder_attempts=80,action_encoder_completed=80,action_encoder_input_tokens=240)
        require(all(report['cost'][k]==n for k,n in expected_cost.items()), 'full run actual-work census differs')


def audit(report_path, inputs=None):
    inputs = saved.Inputs() if inputs is None else inputs
    metric = metric_sources(inputs)
    path = Path(report_path).resolve()
    p, receipt = load_prepared(inputs,path.parent)
    report = inputs.json(path)
    require(report['schema']==runtime.TAG and report['protocol_sha256']==receipt['protocol_sha256'] and
        report['source_seals']==receipt['source_seals'] and report['P1_admission'] is False and report['P4_admission'] is False,
        'runtime report/protocol/scientific boundary differs')
    journal = path.parent/'runtime.events.jsonl'
    raw = inputs.raw(journal,MAX_JOURNAL_BYTES)
    require(hashlib.sha256(raw).hexdigest()==report['journal_sha256'], 'journal/report SHA differs')
    lines = raw.splitlines()
    require(0<len(lines)<=MAX_EVENTS, 'bounded nonempty journal required')
    events = [saved.strict_json(line) for line in lines]
    require([e['event_index'] for e in events]==list(range(len(events))), 'journal event index gap/duplicate')
    require(all(np.isfinite(e['monotonic_seconds']) for e in events) and
        all(a['monotonic_seconds']<=b['monotonic_seconds'] for a,b in zip(events,events[1:])), 'journal time order differs')
    require(events[0]['kind']=='all_slots_before_allocation' and events[0]['slots']==runtime.roster() and
        events[0]['branches']==list(runtime.ARMS), 'preallocation branch/slot census missing')
    final = one(events,'final_census')
    require(final is events[-1] and all(final[k]==report[k] for k in ('rows','slots','cost','predict')), 'final census/report differs')
    work_records(report,events)
    runtime.census(report['slots'])
    require([r['arm'] for r in report['rows']]==list(runtime.ARMS), 'two ordered distinct arms required')
    require(report['cost']['setup_attempts']==sum(e['kind']=='setup_step_attempt' for e in events) and
        report['cost']['setup_returns']==sum(e['kind']=='setup_step_returned' for e in events), 'setup physics accounting differs')
    require(report['actual_predict_samples']==sum(r['predict_samples'] for r in report['predict']) and
        report['scalar_vjp_directions_completed']==192*report['cost']['jacobian_completed'] and
        report['svd_calls_completed']==2*report['cost']['bank_completed'], 'saved derivative/predict accounting differs')
    frames = np.load(io.BytesIO(inputs.raw(p['root_rgb'],2*1024*1024)),allow_pickle=False)
    require(frames.shape==(3,224,224,3) and frames.dtype==np.uint8, 'original RGB archive differs')
    rows = [branch(inputs,p,row,events,report['slots'],frames) for row in report['rows']]
    require(report['cost']['selected_attempts']==sum(r.get('selected_attempts',0) for r in rows) and
        report['cost']['selected_returns']==sum(r.get('selected_returns',0) for r in rows), 'selected physics accounting differs')
    require(report['cost']['encode_attempts']==sum(e['kind']=='encode_begin' for e in events) and
        report['cost']['jacobian_attempts']==sum(e['kind']=='jacobian_begin' for e in events) and
        report['cost']['bank_attempts']==sum(e['kind']=='bank_begin' for e in events), 'durable attempted work differs')
    if report['status']=='calibration_recorded':
        require(all(r['classification']=='complete20' for r in rows) and report['failures']==[] and
            report['model_unchanged'] is True and report['partial_internal_work_unknown'] is False and
            report['actual_predict_samples']==80, 'complete runtime claim differs')
    inputs.stable()
    return dict(schema=TAG,status='saved_arithmetic_audit_passed',runtime_status=report['status'],
        runtime_failures=report['failures'],metric_contract=metric,root_sha256=runtime.ROOT,
        physical_goal=p['physical_goal'],rows=rows,comparison=compare(rows),saved_cost=report['cost'],
        inputs=inputs.hashes, inputs_unchanged=True, neural_calls=0,physics_calls=0,
        P1_admission=False,P4_admission=False,efficacy_verified=False,
        limitation='one reused root, not a significance test or causal mechanism/admission result')


def run(output, report=DEFAULT_REPORT):
    output = Path(output).resolve()
    require(output.name.startswith(TAG) and not output.exists(), 'new exclusive outcomes output required')
    output.mkdir(exist_ok=False)
    files = [Path(__file__),REPO/f'tests/test_{TAG}.py',REPO/f'innerdocs/{TAG}.md',
        Path(saved.__file__),Path(runtime.__file__),Path(official.__file__),ENV,POSE_SOURCE,
        Path(runtime.compiler.__file__),Path(runtime.compiler.packet.__file__)]
    seal = create_seal(files=files,args_json=json.dumps(dict(report=str(Path(report).resolve()),neural_calls=0,physics_calls=0)),
        output_dir=output/'source_seal')
    try:
        value = audit(report)
        value['audit_source_seal'] = seal
        from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal
        value['source_verification'] = verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
        saved.write(output/'report.json',value)
        print(json.dumps(dict(status=value['status'],comparison=value['comparison'],
            classifications=[r['classification'] for r in value['rows']])))
    except (Exception,KeyboardInterrupt) as exc:
        saved.write(output/'failure.json',dict(status='audit_fail_closed',type=type(exc).__name__,message=str(exc),audit_source_seal=seal))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=DEFAULT_REPORT)
    parser.add_argument('--output',type=Path,required=True)
    run(**vars(parser.parse_args()))
