"""Read-only paired-v2 implementation/root-cause audit; no model or simulator calls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p2_composition_paired_physical_20261027_v2 as physical
from experiments.spectral_reachability import p2_composition_admission_20261027_v1 as evidence
from experiments.spectral_reachability.audit_pusht_sealed_task_contract import labels
from experiments.spectral_reachability.d2p_composition_witness_v2 import digest
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal

TAG = 'p2_composition_paired_code_audit_20261027_v1'
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_PHYSICAL = PROJECT/'results/p2_composition_paired_physical_20261027_v2_main01'
planner = physical.sampling
require = evidence.require
BOUNDARY = dict(P1_admission=False, P2_admission=False, equal_total_compute=False,
    new_neural_calls=0, new_physical_steps=0, outcome_tuning=False,
    independent_derivative_validation=False, calibration_only=True,
    counterfactual_nonwinner_physics_available=False)

def array(value, shape):
    a = np.asarray(value)
    require(a.dtype.kind in 'fi' and a.shape == shape and np.isfinite(a).all(),
        'finite exact-shape array required')
    return a.astype(np.float64)

def near(a, b, name, atol=1e-10):
    require(np.shape(a)==np.shape(b) and np.isfinite(a).all() and np.isfinite(b).all()
        and np.allclose(a,b,rtol=0,atol=atol), name+' mismatch')

def keyed(rows, fields):
    return evidence.index(rows, fields)

def metric(vector, goal, q):
    residual = q@(array(vector,(192,))-array(goal,(192,)))
    return float(residual@residual/192)

def normalized(actions, normalization):
    raw = array(actions,(5,2)).astype(np.float32).reshape(10)
    mean,std = (array(normalization[k],(2,)).astype(np.float32) for k in ('mean','std'))
    require(np.all(std>0) and np.all(np.abs(raw)<=1),'positive stats/bounded raw required')
    return ((raw-np.tile(mean,5))/np.tile(std,5)).tolist()

def rolling(before, after, actions, normalization):
    """Check bookkeeping only; do NOT regenerate predicted latent values."""
    array(before['latent_history'],(3,192))
    array(after['latent_history'],(3,192))
    require(after['latent_history'][:2]==before['latent_history'][1:],
        'rolling latent history mismatch')
    require(after['action_history']==[before['action_history'][-1],normalized(actions,normalization)],
        'rolling action history mismatch')

def local_diagnostic(candidate, compiled):
    """Independent residual/ball/KKT arithmetic; no call to suffix_solution."""
    local = candidate['suffix_compilation']
    goal = array(compiled['goal_vectors'][0],(192,))
    q = array(compiled['scoring']['metric']['projection'],(192,192))
    j = array(local['jacobian'],(192,10))
    nominal = array(local['nominal'],(192,))
    prefix = array(candidate['prefix_logits'],(10,))
    x = array(local['suffix_logits'],(10,))
    near(prefix,local['prefix_logits'],'local prefix',0)
    near(array(candidate['logits'],(20,)),np.r_[prefix,x],'joint coordinates',0)
    radius = float(np.sqrt(max(0.,planner.sra.CAP**2-prefix@prefix)))
    near(local['remaining_radius'],radius,'remaining ball radius')
    require(np.linalg.norm(np.r_[prefix,x])<=planner.sra.CAP*(1+1e-12),'joint logit ball violation')
    near(candidate['actions'],np.tanh(np.r_[prefix,x]).astype(np.float32).reshape(10,2),
        'time-major float32 tanh mapping',0)
    k, residual = q@j, q@(goal-nominal)
    solve = local['solve']
    near(solve['params'],x,'saved solution',0)
    linear = float(np.mean((residual-k@x)**2))
    penalty = float(.002*(x@x)/192)
    if radius>0:
        multiplier = evidence.finite(solve['ball_multiplier'])
        require(multiplier>=0,'negative ball multiplier')
        stationarity = k.T@(k@x-residual)+(.002+multiplier)*x
        scale = 1+np.linalg.norm(k.T@residual)
        error = float(np.linalg.norm(stationarity)/scale)
        require(error<1e-10,'ridge KKT stationarity mismatch')
        require(abs(multiplier*(np.linalg.norm(x)-radius))<1e-8,'ball complementarity mismatch')
        near(solve['linear_terminal_cost'],linear,'linear mean square')
        near(solve['ridge_penalty'],penalty,'ridge normalization')
        near(solve['own_reach_value'],linear+penalty,'ridge objective')
        near(solve['norm'],np.linalg.norm(x),'solve norm')
        near(solve['radius'],radius,'solve radius')
    else:
        require(np.array_equal(x,np.zeros(10)) and solve['singleton_zero'] is True,
            'zero-radius singleton mismatch')
        error = 0.
    end = candidate['witness']['receipt']['terminal_context']['latent_history'][-1]
    nonlinear = metric(end,goal,q)
    near(candidate['cost'],nonlinear,'nonlinear original Q selection cost')
    nominal_cost = metric(nominal,goal,q)
    return dict(candidate_id=candidate['candidate_id'],index=candidate['index'],
        nominal_zero_suffix_q_mse=nominal_cost, linear_suffix_q_mse=linear,
        nonlinear_suffix_q_mse=nonlinear, linear_predicted_improvement=nominal_cost-linear,
        nonlinear_model_improvement=nominal_cost-nonlinear,
        nonlinear_minus_linear_q_mse=nonlinear-linear,
        projected_linearization_error_mse=float(np.mean((q@(array(end,(192,))-nominal-j@x))**2)),
        suffix_logit_norm=float(np.linalg.norm(x)),remaining_radius=radius,
        raw_tanh_minus_logit_l2=float(np.linalg.norm(np.tanh(x)-x)),
        scaled_kkt_residual=error)

def gains(joint, composition):
    """Positive means composition better; never compare different metric units."""
    names = ('predicted_q_mse','encoded_q_mse','pose_cost')
    out = {name:joint[name]-composition[name] for name in names}
    out['model_vs_encoding_rank_reversal'] = bool(out['predicted_q_mse']*out['encoded_q_mse']<0)
    out['model_vs_pose_rank_reversal'] = bool(out['predicted_q_mse']*out['pose_cost']<0)
    return out

def artifacts(directory, report, reader):
    durable = reader.load(directory/'physical_report.json')
    require(keyed(durable['rows'],('root_sha256','method'))==keyed(report['rows'],('root_sha256','method')),
        'durable/final physical rows mismatch')
    encodings = keyed(report['encodings'],('root_sha256','method'))
    for row in report['rows']:
        stem = f'root_{row["root_index"]}_{row["method"]}_replay0'
        require(reader.load(directory/(stem+'.physical.json'))==row,'saved physical row mismatch')
        require(reader.load(directory/(stem+'.encoding.json'))==encodings[(row['root_sha256'],row['method'])],
            'saved encoding mismatch')
        journal = Path(row['journal_path']).resolve()
        require(journal==directory/(stem+'.events.jsonl'),'wrong branch journal path')
        reader.bind(journal,row['journal_sha256'])
        events = [json.loads(line,object_pairs_hook=evidence.strict_object) for line in journal.read_text().splitlines()]
        require([e['kind'] for e in events]==['initial']+['attempt','step_returned','observation']*10,
            'durable full H2 event sequence mismatch')
        observations = []
        for e in events:
            if 'rgb_path' in e:
                image = Path(e['rgb_path']).resolve()
                require(image.parent==directory and image.name.startswith(stem+'_rgb_'),'RGB path mismatch')
                reader.bind(image,e['rgb_file_sha256'])
            if e['kind']=='observation':
                observations.append({k:v for k,v in e.items() if k not in ('rgb_path','rgb_file_sha256')})
        require(observations==row['trace'],'durable observations mismatch')
        require(events[-1]['rgb_path']==row['terminal_rgb_path']
            and events[-1]['rgb_file_sha256']==row['terminal_rgb_sha256'],'endpoint RGB mismatch')

def compute(directory, reader):
    directory = Path(directory).resolve()
    execution,report = (reader.load(directory/name) for name in ('protocol.json','report.json'))
    require(report['schema']==physical.TAG+'.report' and report['status']=='calibration_recorded'
        and not report['failures'] and report['gates'] and all(v is True for v in report['gates'].values()),
        'complete physical evidence required')
    _,protocol,prepared,context = physical.load_prepared(execution['prepared'])
    reader.bindings(protocol['bindings'])
    reader.seals(prepared['source_seals'])
    for name in ('protocol.json','prepared.json'):
        reader.load(Path(execution['prepared'])/name)
    require(report['prepared_protocol_sha256']==execution['prepared_protocol_sha256']==digest(protocol)
        and report['source_seals']==prepared['source_seals']
        and execution['plans']==protocol['plans']==context['plans'],'physical source/plan binding mismatch')
    require(all(report[k]==v for k,v in physical.BOUNDARY.items()) and report['new_predict_calls']==0,
        'physical boundary mismatch')
    require(execution['encode_terminals'] is True and report['attempted_terminal_encode_samples']==8
        and report['selected_path_env_step_attempts']==report['selected_path_env_step_returns']==80,
        'complete eight encoded H2 endpoints required')
    planning = Path(protocol['planning'])
    saved = reader.load(planning/'report.json')
    pprotocol = reader.load(planning/'protocol.json')
    pprepared = reader.load(planning/'prepared.json')
    reader.seals(pprepared['source_seals'])
    require(saved['protocol_sha256']==digest(pprotocol),'planning protocol mismatch')
    reader.bindings(pprotocol['bindings'])
    artifacts(directory,report,reader)
    roots = keyed(saved['rows'],('root_sha256',))
    compiled = keyed(context['source']['rows'],('root_sha256',))
    branches = keyed(report['rows'],('root_sha256','method'))
    encodings = keyed(report['encodings'],('root_sha256','method'))
    plans = keyed(protocol['plans'],('root_sha256','method'))
    require(len(roots)==4 and roots.keys()==compiled.keys()
        and set(plans)==set(branches)==set(encodings)=={(r[0],a) for r in roots for a in planner.ARMS},
        'four-root eight-arm stable census mismatch')
    results = []
    normalization = pprotocol['context']['normalization']
    for rootkey,row in sorted(roots.items(),key=lambda pair:pair[1]['root_index']):
        root = rootkey[0]
        c = compiled[rootkey]['compiled']
        q = array(c['scoring']['metric']['projection'],(192,192))
        goal = array(c['goal_vectors'][0],(192,))
        bank = reader.load(planning/f'root_{row["root_index"]}_bank.json')
        arms = keyed(row['results'],('arm',))
        joint,composition = (arms[(a,)] for a in planner.ARMS)
        require(len(joint['attempts'])==len(composition['attempts'])==40,'all candidate slots required')
        ja = {a['candidate_id']:a for a in joint['attempts']}
        diagnostics = []
        initial = composition['attempts'][0]['witness']['edges'][0]['input_context']
        for attempt in composition['attempts']:
            i = attempt['index']
            require(type(i) is int and 0<=i<40,'candidate slot mismatch')
            peer = ja[f'{planner.ARMS[0]}_{i:02d}']
            near(attempt['prefix_logits'],np.asarray(bank['logits'])[i,:10],'shared bank prefix',0)
            require(attempt['actions'][:5]==peer['actions'][:5],'paired prefix command mismatch')
            edges = attempt['witness']['edges']
            require(len(edges)==2 and edges[0]['input_context']==initial
                and edges[0]['output_context']==edges[1]['input_context']
                and digest(edges[1]['input_context'])==attempt['suffix_compilation']['input_context_sha256'],
                'own complete prefix context binding mismatch')
            for edge in edges:
                rolling(edge['input_context'],edge['output_context'],edge['actions'],normalization)
            # Joint receipt only has the whole-H2 edge; its terminal history must
            # contain the same retained root boundary plus two new packets.
            end = peer['witness']['receipt']['terminal_context']
            require(end['latent_history'][0]==initial['latent_history'][-1]
                and end['action_history']==[normalized(peer['actions'][:5],normalization),
                    normalized(peer['actions'][5:],normalization)],'joint H2 history mismatch')
            d = local_diagnostic(attempt,c)
            d['same_prefix_joint_q_mse'] = metric(end['latent_history'][-1],goal,q)
            near(peer['cost'],d['same_prefix_joint_q_mse'],'joint nonlinear score')
            d['composition_model_gain_over_same_prefix_joint'] = d['same_prefix_joint_q_mse']-d['nonlinear_suffix_q_mse']
            diagnostics.append(d)
        selected = {}
        for arm,result in ((planner.ARMS[0],joint),(planner.ARMS[1],composition)):
            winner = min(result['attempts'],key=lambda a:a['cost'])
            require(result['winner']['candidate_id']==winner['candidate_id']
                and result['winner']['independent_full_replay_exact'] is True,'stable replayed selection mismatch')
            key = root,arm
            plan,branch,encoding = plans[key],branches[key],encodings[key]
            physical.validate_plan(plan)
            require(plan['selected_candidate_id']==winner['candidate_id']
                and plan['witness']==winner['witness'] and branch['selection_sha256']==plan['selection_sha256']
                and branch['root_index']==row['root_index'] and branch['status']=='completed'
                and branch['full_horizon_completed'] is True and not branch['failures']
                and branch['physical_env_step_attempts']==branch['physical_env_step_returns']==10
                and len(branch['trace'])==10 and [e['command'] for e in branch['trace']]==plan['commands'],
                'selected witness/actual dispatch mismatch')
            identity = branch['context_identity']
            require(identity['history_rgb_sha256']==plan['root']['root_rgb_sha256']
                and identity['goal_rgb_sha256']==plan['root']['goal_rgb_sha256']
                and identity['normalized_between_history']==plan['root']['context']['action_history'],
                'reset goal/context mismatch')
            physical.validate_reset_history(identity['raw_history'],normalization,identity['normalized_between_history'])
            state,g = array(branch['final_state'],(7,)),array(identity['goal'],(7,))
            require(state.tolist()==branch['trace'][-1]['state'],'final observation mismatch')
            official = labels(state,g)
            require(official==branch['official'],'original official arithmetic mismatch')
            angle = float(abs(np.arctan2(np.sin(state[4]-g[4]),np.cos(state[4]-g[4]))))
            pose = dict(xy_distance=float(np.linalg.norm(state[2:4]-g[2:4])),angle_distance=angle)
            pose['pose_cost'] = pose['xy_distance']+20*angle
            for name,value in pose.items():
                near(branch['pose'][name],value,'saved pose')
            require(encoding['status']=='completed' and encoding['attempted_encode_samples']==1
                and encoding['failure'] is None,'saved real endpoint encoding required')
            predicted = metric(winner['witness']['receipt']['terminal_context']['latent_history'][-1],goal,q)
            actual = metric(encoding['terminal_vector'],goal,q)
            selected[arm] = dict(candidate_id=winner['candidate_id'],index=winner['index'],
                predicted_q_mse=predicted,encoded_q_mse=actual,pose_cost=pose['pose_cost'],
                prediction_cost_error=actual-predicted,
                projected_endpoint_prediction_error_mse=float(np.mean((q@(
                    array(encoding['terminal_vector'],(192,))-array(winner['witness']['receipt']['terminal_context']['latent_history'][-1],(192,))))**2)),
                witness_sha256=digest(winner['witness']),selection_sha256=plan['selection_sha256'])
        left,right = (branches[(root,a)]['context_identity'] for a in planner.ARMS)
        require(left==right,'paired reset/goal identities differ')
        gains_row = gains(*(selected[a] for a in planner.ARMS))
        selected_diagnostic = next(d for d in diagnostics if d['index']==selected[planner.ARMS[1]]['index'])
        # Local solve may lose versus its zero suffix after nonlinear replay;
        # it is NOT guaranteed nonlinear improvement and was never used as selector.
        results.append(dict(root_sha256=root,root_index=row['root_index'],selected=selected,
            composition_improvement=gains_row,selected_local_diagnostic=selected_diagnostic,
            candidate_diagnostics=diagnostics,
            candidates_nonlinear_worse_than_zero_suffix=sum(d['nonlinear_model_improvement']<0 for d in diagnostics),
            candidates_worse_than_same_prefix_joint=sum(d['composition_model_gain_over_same_prefix_joint']<0 for d in diagnostics),
            q_symmetry_max_error=float(np.max(np.abs(q-q.T))),
            q_idempotence_max_error=float(np.max(np.abs(q@q-q))),
            resource_receipt=deepcopy(row['resource_receipt'])))
    reader.unchanged()
    return dict(schema=TAG+'.report',status='saved_contract_audit_completed',roots=results,
        demonstrated_fixable_implementation_error=False,proposed_new_planner_version=None,
        implementation_contract=dict(
            residual='Q@(goal0 - terminal(prefix_context, zero_suffix)); not goal-minus-prefix-latent',
            derivative_coordinates='10 pre-tanh suffix logits at zero; tanh derivative there is one; predicted prefix held fixed',
            solve_objective='(norm(QJ*x-Q(goal-nominal))^2 + .002*norm(x)^2)/192; dividing BOTH terms preserves optimizer',
            selection='stable first minimum of nonlinear Q terminal MSE over all40; not ridge objective or prefix score',
            support='one full20 logit L2 ball, residual suffix radius, then float32 tanh; not raw-box uniform sampling',
            action_execution='guard returns unchanged raw command; relative env target=agent.position+100*raw; window clip is commented out',
            history='3 latent boundaries and2 normalized between-state packets; append new packet then shift; no incoming-history substitution',
            witness='source-bound exporter re-solves all suffixes; audit independently checks KKT and rolling histories; no independent new neural replay'),
        evidence_diagnosis=dict(
            model_to_real_q_reversal_root_indices=[r['root_index'] for r in results
                if r['composition_improvement']['model_vs_encoding_rank_reversal']],
            real_q_better_but_pose_worse_root_indices=[r['root_index'] for r in results
                if r['composition_improvement']['encoded_q_mse']>0 and r['composition_improvement']['pose_cost']<0],
            interpretation='Observed ranking gaps, not identified code defects or causal proof of a particular model failure mechanism'),
        verdict='No demonstrated residual/coordinate/clipping/history/Q/winner contract defect. Saved nonlinear model ranking can disagree with real endpoint Q and pose; no post-outcome method repair justified.',
        improvement_direction='joint minus composition, positive is improvement; compare signs only across different metric units',
        limitations=['Saved derivatives are not independently checked by AD/finite differences.',
            'Nominal suffix value/derivative truth cannot be established from receipt binding alone.',
            'Only the two selected witnesses/root have real endpoint encodings; no candidate-wide physical ranking.',
            'Original Q is a latent objective, not the official pose/state metric.',
            'Conditional suffix is compiled at a predicted, not observed, prefix; model error may propagate.',
            'tanh logits and full logit ball are declared, not a uniform raw-box proposal or a local trust guarantee.'],
        planning_totals=protocol['planning_totals'],
        additional_audit_work='source-bound exporter array rebuild plus 160 independent KKT/linearization checks; no new neural work',
        **BOUNDARY)

def run(physical_run, output):
    output = Path(output).resolve()
    require(TAG in output.name,'exclusive audit namespace required')
    if output.exists():
        raise FileExistsError(output)
    reader = evidence.EvidenceReader()
    result = compute(physical_run,reader)
    files = sorted({Path(__file__).resolve(),PROJECT/'tests'/f'test_{TAG}.py',
        PROJECT/'innerdocs'/f'{TAG}.md',Path(physical.__file__).resolve(),
        Path(planner.__file__).resolve(),Path(evidence.__file__).resolve(),
        Path(labels.__code__.co_filename).resolve(),Path(create_seal.__code__.co_filename).resolve()})
    protocol = dict(schema=TAG+'.protocol',physical_run=str(Path(physical_run).resolve()),
        bindings=deepcopy(reader.hashes),boundary=BOUNDARY,source_paths=list(map(str,files)))
    output.mkdir(exist_ok=False)
    physical.write(output/'protocol.json',protocol)
    seal = create_seal(files=files,args_json=json.dumps(dict(protocol_sha256=digest(protocol))),
        output_dir=output/'source_seal_0')
    result.update(protocol_sha256=digest(protocol),source_seals=[seal],input_bindings=deepcopy(reader.hashes))
    physical.write(output/'report.json',result)
    reader.unchanged()
    verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    print(json.dumps(dict(status=result['status'],output=str(output),P2_admission=False)))
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical-run',type=Path,default=DEFAULT_PHYSICAL)
    parser.add_argument('--output',type=Path,required=True)
    run(**vars(parser.parse_args(argv)))

if __name__=='__main__':
    main()

