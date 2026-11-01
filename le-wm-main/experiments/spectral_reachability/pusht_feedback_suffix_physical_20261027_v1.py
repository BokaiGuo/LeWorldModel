"""Consume six sealed suffix winners; exact step5 RGB gate BEFORE suffix dispatch.

prepare performs no physics/CUDA. run requires a separately prepared directory.
No predictor, action search, training, outcome selection or scientific admission.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p3_matched_physical_20261027_v1 as durable
from experiments.spectral_reachability.pusht_rgb_observability_guard import RgbObservabilityGuard, ObservationUnavailable

TAG = 'pusht_feedback_suffix_physical_20261027_v1'
COMPILER = 'pusht_feedback_suffix_compile_20261027_v1'
ROOT = '23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050'
ROLES = ('selected','greedy','zero')
ARMS = ('imagined','actual')
REPO = Path(__file__).resolve().parents[2]
DEFAULT_FEEDBACK = REPO/'results/pusht_geometry_feedback_response_20261027_v1_prepare_v1'
write, sha = durable.write, durable.digest


def read(path):
    return json.loads(Path(path).read_text())


def content_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def array(value, shape):
    a = np.asarray(value)
    if a.dtype.kind not in 'fi' or a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'finite real {shape} required')
    return a.astype(float)


def census(plans, rows=None):
    expected = {(r,a) for r in ROLES for a in ARMS}
    if len(plans) != 6 or {(p['role'],p['arm']) for p in plans} != expected:
        raise ValueError('six unique role x context-arm winners required')
    for p in plans:
        if (p['root_sha256'] != ROOT or type(p['root_index']) is not int or p['root_index'] != 1
                or p['planned_commands'] != 10 or p['planned_horizon_tokens'] != 2
                or p['selection_sha256'] != content_sha({k:v for k,v in p.items() if k != 'selection_sha256'})):
            raise ValueError('bound plan identity/receipt/horizon differs')
    if rows is not None:
        index = {p['method']:p for p in plans}
        if len(rows) != 6 or {r['method'] for r in rows} != set(index):
            raise ValueError('all six branch receipts required before encoding')
        for r in rows:
            p = index[r['method']]
            if any(r[k] != p[k] for k in ('root_sha256','root_index','role','arm','selection_sha256')):
                raise ValueError('physical branch identity differs')
    return dict(branches=6,planned_selected_path_steps=60,root_sha256=ROOT,root_index=1,
                planned_prefix_steps=30,planned_suffix_steps=30,role_aliases_retained=True)


def plans_from_saved(report, feedback_protocol, feedback_report):
    """Pure validation; model-only nonlinear argmin, never read physical outcomes."""
    from experiments.spectral_reachability import pusht_feedback_suffix_compile_20261027_v1 as compiler
    if (report.get('schema') not in (COMPILER,COMPILER+'.report')
            or report.get('status') != 'plans_sealed'
            or report.get('failure') is not None or report.get('model_unchanged') is not True
            or report.get('partial_internal_work_unknown') is not False
            or report.get('actual_predict_samples') != 78 or report.get('jacobian_attempts') != 6
            or report.get('jacobian_materializations') != 6 or report.get('scalar_vjp_directions_completed') != 1152
            or report.get('svd_calls_completed') != 12 or report.get('new_encode_calls') != 0
            or report.get('new_physics_steps') != 0):
        raise ValueError('complete plans_sealed compiler and explicit 78predict/6Jac accounting required')
    expected_work = [(r+'_'+a+suffix,n) for r in ROLES for a in ARMS
                     for suffix,n in (('_compile',2),('_nonlinear_bank',10),('_independent_replay',1))]
    records = report.get('predict',[])
    if len(records) != 18:
        raise ValueError('18 complete compiler predictor scopes required')
    for record,(method,n) in zip(records,expected_work):
        if (record['method'] != method or record['predict_calls'] != n or record['predict_samples'] != n
                or record['completed'] is not True or record['failed_predict_calls'] != 0
                or record['horizon_tokens'] != 1 or record['context_lengths'] not in ({'3':n},{3:n})
                or record['phase'] != ('evaluation' if method.endswith('_independent_replay') else 'planning')):
            raise ValueError('compiler predictor scope order/work differs')
    saved = report['plans']
    if len(saved) != 6 or {(p['role'],p['arm']) for p in saved} != {(r,a) for r in ROLES for a in ARMS}:
        raise ValueError('six unique saved winners required')
    if feedback_report['status'] != 'diagnostic_recorded' or feedback_report['failure'] is not None:
        raise ValueError('completed original feedback contexts required')
    contexts = {c['role']:c for c in feedback_report['contexts']}
    cases = {c['role']:c for c in feedback_protocol['cases']}
    plans = []
    for role in ROLES:
        for arm in ARMS:
            s = next(p for p in saved if p['role'] == role and p['arm'] == arm)
            compiler.verify_plan(s)  # Pure saved-bank algebra, no neural execution.
            case = cases[role]
            if (s['root_sha256'] != ROOT or type(s['root_index']) is not int or s['root_index'] != 1
                    or s['prefix'] != case['prefix'] or s['context'] != contexts[role][arm]
                    or s['goal'] != feedback_protocol['goal'] or s['q'] != feedback_protocol['q']):
                raise ValueError('original root/prefix/context/goal/Q binding differs')
            goal, q = array(s['goal'],(192,)), array(s['q'],(192,192))
            candidates = s['candidates']
            ids = [c['candidate_id'] for c in candidates]
            if len(candidates) != 10 or len(set(ids)) != 10 or any(not isinstance(i,str) or not i for i in ids):
                raise ValueError('ten distinct spectral candidate receipts required')
            for c in candidates:
                params = array(c['suffix_params'],(10,)); commands = array(c['commands'],(10,2))
                prediction = array(c['predicted_terminal'],(192,))
                cost = float(array(c['nonlinear_terminal_cost'],()))
                if (np.any(np.abs(commands)>1) or np.linalg.norm(params)>np.sqrt(5)*.75+1e-6
                        or not np.array_equal(commands[:5],case['prefix'])
                        or not np.allclose(commands[5:],np.tanh(params).reshape(5,2),rtol=0,atol=1e-7)
                        or abs(cost-float(np.mean((q@(goal-prediction))**2))) > 1e-12):
                    raise ValueError('candidate action/chart/nonlinear Q cost differs')
            winner = min(candidates,key=lambda c:c['nonlinear_terminal_cost'])
            if s['selected_candidate_id'] != winner['candidate_id']:
                raise ValueError('stable nonlinear model-only argmin required')
            p = dict(root_sha256=ROOT,root_index=1,role=role,arm=arm,method=role+'_'+arm,
                commands=deepcopy(winner['commands']),selected_candidate_id=winner['candidate_id'],
                prefix=deepcopy(case['prefix']),predictor_context=deepcopy(s['context']),
                expected_prefix_rgb=case['prefix_rgb'],
                expected_prefix_rgb_sha256=sha(case['prefix_rgb']),
                goal=s['goal'],q=s['q'],saved_candidate=dict(
                    predicted_prefix=contexts[role][arm]['latent_history'][-1],
                    predicted_terminal=winner['predicted_terminal']),
                planned_commands=10,planned_horizon_tokens=2,corrected_search_horizon=1,
                prediction_scope='one remaining suffix token; physical replay is five prefix plus five suffix')
            p['selection_sha256'] = content_sha(p); plans.append(p)
    census(plans)
    return plans


def prepare(planning_report, feedback_dir=DEFAULT_FEEDBACK):
    from experiments.spectral_reachability import pusht_geometry_feedback_response_20261027_v1 as feedback
    from experiments.spectral_reachability import pusht_geometry_capability_20261027_v1 as cap
    from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal
    path = Path(planning_report).resolve(); fd = Path(feedback_dir).resolve()
    report = read(path)
    from experiments.spectral_reachability import pusht_feedback_suffix_compile_20261027_v1 as compiler
    compiler_protocol,compiler_receipt = compiler.verify_prepared(path.parent)
    frozen = feedback.verify_frozen(fd); observed = read(fd/'report.json')
    plans = plans_from_saved(report,frozen,observed)
    protocol_path = path.parent/'protocol.json'; protocol = read(protocol_path)
    if (protocol != compiler_protocol or Path(protocol['source']).resolve() != fd
            or protocol['original_initial_context'] != frozen['initial_context']
            or protocol['original_root_rgb'] != frozen['root_rgb']
            or report['source_seals'] != compiler_receipt['source_seals']):
        raise ValueError('compiler prepared source/initial context/root RGB/seals differ')
    if [{k:s[k] for k in ('root_sha256','root_index','role','arm','prefix','context','goal','q')}
            for s in report['plans']] != protocol['cases']:
        raise ValueError('compiler six plans differ from frozen case order')
    if report.get('protocol_sha256') != sha(protocol_path):
        raise ValueError('compiler protocol hash differs')
    compiler_bindings = protocol.get('bindings')
    if not compiler_bindings or any(sha(p)!=h for p,h in compiler_bindings.items()):
        raise ValueError('compiler input bindings missing/changed')
    for f in (fd/'protocol.json',fd/'report.json',fd/'prepare_receipt.json'):
        if compiler_bindings.get(str(f)) != sha(f):
            raise ValueError('compiler must bind original feedback protocol/report/prepare receipt')
    seals = report.get('source_seals')
    if not seals:
        raise ValueError('compiler prerun source seals required')
    bindings = dict(compiler_bindings)
    for seal in seals:
        verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
        for name in ('manifest.json','args.json'):
            f = Path(seal['output_dir'])/name; bindings[str(f.resolve())] = sha(f)
    # Recover metadata and refs through ORIGINAL capability/physical-v2 prepare.
    capability_protocol = read(Path(frozen['capability']).parent/'protocol.json')
    context = cap.prepare(capability_protocol['planning_report'])
    bindings.update(context[4]); bindings.update(frozen['bindings'])
    for f in (path,protocol_path,fd/'protocol.json',fd/'report.json',fd/'prepare_receipt.json'):
        bindings[str(f)] = sha(f)
    f = path.parent/'prepare_receipt.json'; bindings[str(f)] = sha(f)
    for p in plans:
        bindings[p['expected_prefix_rgb']] = p['expected_prefix_rgb_sha256']
    return plans, bindings, context, seals


def freeze(planning_report, output, feedback_dir=DEFAULT_FEEDBACK, encode_terminals=False):
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal
    from experiments.spectral_reachability import run_pusht_sampling_comparison_20261027_v2 as identity
    from experiments.spectral_reachability import pusht_full_packet_physical_20261027 as endpoints
    plans,bindings,context,compiler_seals = prepare(planning_report,feedback_dir)
    base,checked,source,config = context[:4]
    output = Path(output).resolve()
    if not output.name.startswith(TAG):
        raise ValueError('exclusive tagged physical output required')
    # Conservative disk check, no deletion or storage override.
    from experiments.spectral_reachability.pusht_geometry_capability_20261027_v1 import disk_guard
    disk = disk_guard(output,6)
    files = checked.sources_before_allocate(source,config)+[
        Path(__file__),Path(durable.__file__),Path(identity.__file__),Path(endpoints.__file__),
        Path(inspect.getfile(RgbObservabilityGuard)),REPO/f'tests/test_{TAG}.py',REPO/f'innerdocs/{TAG}.md']
    for seal in compiler_seals:
        files += [Path(e['source']) for e in read(Path(seal['output_dir'])/'manifest.json')['files']]
    files = sorted(set(p.resolve() for p in files))
    output.mkdir(exist_ok=False)
    protocol = dict(schema=TAG,planning_report=str(Path(planning_report).resolve()),
        feedback_dir=str(Path(feedback_dir).resolve()),plans=plans,bindings=bindings,census=census(plans),disk=disk,
        encode_terminals=bool(encode_terminals),max_endpoint_encode_samples=12 if encode_terminals else 0,
        prefix_gate='exact uint8 step5 pixel equality; stop before step6 on mismatch',
        official_pose='unchanged run_pusht_sampling_comparison_v2.official_checked and base.physical._pose_label',
        new_predict_calls=0,P1_admission=False,efficacy_verified=False)
    write(output/'protocol.json',protocol)
    seals = [create_seal(files=files[i:i+32],args_json=json.dumps(dict(protocol_sha256=sha(output/'protocol.json'))),
             output_dir=output/f'source_seal_{i//32}') for i in range(0,len(files),32)]
    receipt = dict(status='prepared_not_run',protocol_sha256=sha(output/'protocol.json'),source_seals=seals,
                   actual_physics_steps=0,actual_encode_samples=0,new_predict_calls=0,P1_admission=False)
    write(output/'prepare_receipt.json',receipt)
    return receipt


def verify_prepared(output):
    from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal
    output = Path(output); p = read(output/'protocol.json'); r = read(output/'prepare_receipt.json')
    if r['status'] != 'prepared_not_run' or sha(output/'protocol.json') != r['protocol_sha256']:
        raise ValueError('physical protocol changed')
    if not r['source_seals']:
        raise ValueError('consumer source seals required')
    for s in r['source_seals']:
        verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    if any(sha(f)!=h for f,h in p['bindings'].items()):
        raise ValueError('physical bound inputs changed')
    census(p['plans'])
    return p


def execute_ten(env, commands, *, root_id, expected_prefix, state_of, event):
    """Injected non-neural tests use real guard and fake env, never fake success."""
    raw = array(commands,(10,2))
    expected_prefix = np.asarray(expected_prefix)
    if np.any(np.abs(raw)>1) or expected_prefix.shape != (224,224,3) or expected_prefix.dtype != np.uint8:
        raise ValueError('bounded ten commands and original uint8 prefix image required')
    guard = RgbObservabilityGuard(root_id=root_id,agent_rgb=(65,105,225))
    attempted = completed = 0
    trace, final_state, prefix_receipt, failure = [], None, dict(status='not_reached'), None
    status = 'completed'
    try:
        rgb = np.asarray(env.render()).copy(); final_state = np.asarray(state_of(env)).tolist()
        event(dict(kind='initial',state=final_state),rgb); guard.observe(rgb,primitive_step=0)
        for i,command in enumerate(raw):
            authorized = guard.dispatch(command)
            event(dict(kind='attempt',primitive_step=i+1,command=command.tolist()),None)
            attempted += 1
            result = env.step(authorized); completed += 1
            event(dict(kind='step_returned',primitive_step=i+1),None)
            if len(result) != 5:
                raise ValueError('Gymnasium five-field transition required')
            rgb = np.asarray(env.render()).copy(); final_state = np.asarray(state_of(env)).tolist()
            obs = dict(kind='observation',primitive_step=i+1,command=command.tolist(),state=final_state,
                       reward=float(result[1]),terminated=bool(result[2]),truncated=bool(result[3]))
            event(obs,rgb); trace.append(obs)
            if i == 4:
                matched = rgb.dtype == np.uint8 and np.array_equal(rgb,expected_prefix)
                prefix_receipt = dict(status='matched' if matched else 'mismatch',primitive_step=5,
                    expected_pixel_sha256=hashlib.sha256(expected_prefix.tobytes()).hexdigest(),
                    observed_pixel_sha256=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
                    pixels_exact=bool(matched))
                event(dict(kind='prefix_identity',**prefix_receipt),None)
                if not matched:
                    status = 'prefix_mismatch'; break
            guard.observe(rgb,primitive_step=i+1)
            if result[2] or result[3]:
                status = 'environment_terminated'; break
    except ObservationUnavailable as exc:
        status, failure = 'observation_unavailable',durable.failure(exc,'guard')
    except (Exception,KeyboardInterrupt) as exc:
        status, failure = 'execution_failed',durable.failure(exc,'execution')
    return dict(status=status,failure=failure,attempted_commands=attempted,completed_commands=completed,
        planned_commands=10,remaining_commands_not_executed=10-completed,trace=trace,final_state=final_state,
        prefix_identity=prefix_receipt,full_horizon_completed=status=='completed' and completed==10,
        physical_env_step_attempts=attempted,physical_env_step_returns=completed,
        actual_endpoint_primitive_steps=completed,guard=guard.receipt())


def physical_branch(plan, output, reset, env, state_of, evaluate):
    from PIL import Image
    stem = f'root_1_{plan["method"]}_replay0'; output = Path(output)
    row = {k:deepcopy(plan[k]) for k in ('root_sha256','root_index','role','arm','method','selection_sha256')}
    row.update(status='blocked',repetition=0,trace=[],completed_commands=0,attempted_commands=0,
        planned_commands=10,planned_horizon_tokens=2,full_horizon_completed=False,failures=[],
        physical_env_step_attempts=0,physical_env_step_returns=0,terminal_rgb_path=None,
        prefix_identity=dict(status='not_reached'),warmup_cost_scope='separate reset/warmup, not selected-path steps')
    journal = output/(stem+'.events.jsonl')
    try:
        with journal.open('x') as stream:
            def event(value,image):
                value = deepcopy(value)
                if image is not None:
                    image_path = output/f'{stem}_rgb_{value.get("primitive_step",0)}.png'
                    with image_path.open('xb') as f:
                        Image.fromarray(image).save(f,format='PNG'); f.flush(); os.fsync(f.fileno())
                    value.update(rgb_path=str(image_path),rgb_file_sha256=sha(image_path))
                    row.update(terminal_rgb_path=str(image_path),terminal_rgb_sha256=sha(image_path))
                stream.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
            row['context_identity'] = reset()
            if sha(plan['expected_prefix_rgb']) != plan['expected_prefix_rgb_sha256']:
                raise ValueError('original prefix image changed before execution')
            with Image.open(plan['expected_prefix_rgb']) as im:
                expected = np.asarray(im).copy()
            row.update(execute_ten(env,plan['commands'],root_id=plan['root_sha256'],expected_prefix=expected,
                                   state_of=state_of,event=event))
            if row['failure'] is not None:
                row['failures'].append(row['failure'])
            if row['final_state'] is not None:
                row.update(evaluate(row['final_state']))
    except (Exception,KeyboardInterrupt) as exc:
        row.update(status='branch_failed'); row['failures'].append(durable.failure(exc,'physical_branch'))
    row['journal_path'] = str(journal)
    if journal.exists():
        row['journal_sha256'] = sha(journal)
    write(output/(stem+'.physical.json'),row)
    return row


def run(output):
    from experiments.spectral_reachability import pusht_full_packet_physical_20261027 as endpoints
    from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import verify_rgb_context, official_checked
    output = Path(output).resolve(); p = verify_prepared(output)
    plans,bindings,context,_ = prepare(p['planning_report'],p['feedback_dir'])
    if plans != p['plans'] or bindings != p['bindings']:
        raise ValueError('prepare changed before physical run')
    base,checked,source,config,_,_,upstream,refs,policy_protocol = context
    write(output/'launch_receipt.json',dict(protocol_sha256=sha(output/'protocol.json'),P1_admission=False))
    rows,failures,encodings = [],[],[]
    env = model = before = host = evaluator_seal = None
    try:
        env = base.gym.make('swm/PushT-v1',render_mode='rgb_array')
        from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal
        evaluator_seal = create_seal(files=[Path(env.unwrapped.eval_state.__code__.co_filename).resolve()],
            args_json=json.dumps(dict(protocol_sha256=sha(output/'protocol.json'))),output_dir=output/'evaluator_source_seal')
        write(output/'evaluator_source_receipt.json',evaluator_seal)
        if tuple(env.unwrapped.variation_space['agent']['color'].value.tolist()) != (65,105,225):
            raise ValueError('original renderer required')
        roots = base.collect_nonterminal_boundary_contact_roots(env,count=4,seed=int(upstream['protocol']['root_seed']),oversample_factor=4)
        keys = [base.root_fingerprint(r) for r in roots]
        if keys != policy_protocol['roots'] or len(keys)!=4 or keys[1]!=ROOT:
            raise ValueError('exact original four root identities/order required; no replacement')
        root = roots[1]
        for plan in plans:
            verify_prepared(output)
            def reset():
                if base.root_fingerprint(root) != ROOT:
                    raise ValueError('root mutated')
                frames,history,goal = base.physical._reset_and_warmup(env,root,history_len=3)
                identity = verify_rgb_context(frames,goal,refs[ROOT])
                raw = np.repeat(np.asarray(history)[1:,None,:],5,axis=1)
                expected = next(c['between_raw'] for c in read(Path(p['feedback_dir'])/'protocol.json')['cases'] if c['role']==plan['role'])
                if not np.array_equal(raw,expected):
                    raise ValueError('reset incoming raw action history differs')
                identity.update(history=np.asarray(history).tolist(),goal=np.asarray(root['goal']).tolist())
                return identity
            def evaluate(state):
                return dict(official=official_checked(env,state,root['goal']),pose=base.physical._pose_label(state,root['goal']))
            rows.append(physical_branch(plan,output,reset,env,base.p0.packets.state_of,evaluate))
            print(f'{plan["method"]}: {rows[-1]["status"]}',flush=True)
    except (Exception,KeyboardInterrupt) as exc:
        failures.append(durable.failure(exc,'physical_setup'))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                failures.append(durable.failure(exc,'env_close'))
    existing = {r['method'] for r in rows}
    for plan in plans:
        if plan['method'] not in existing:
            def blocked():
                raise RuntimeError('physical setup unavailable; retain planned slot')
            rows.append(physical_branch(plan,output,blocked,None,None,None))
    census(plans,rows)
    write(output/'physical_report.json',dict(rows=rows,failures=failures,new_predict_calls=0,P1_admission=False))
    if p['encode_terminals']:
        def setup():
            nonlocal model,before,host
            verify_prepared(output); census(plans,rows)
            if read(output/'physical_report.json')['rows'] != rows:
                raise ValueError('durable physical census differs')
            for r in rows:
                if read(output/f'root_1_{r["method"]}_replay0.physical.json') != r:
                    raise ValueError('physical branch changed before CUDA')
            import torch
            from PIL import Image
            host = base.require_cuda_allocation('cuda'); device = torch.device('cuda:0')
            torch.backends.cuda.matmul.allow_tf32 = False
            model = base.load_compatible_pretrained(Path(policy_protocol['policy']),device=device).eval().requires_grad_(False)
            before = checked.tensor_state_hash(model)
            def forbid(*args,**kwargs):
                raise RuntimeError('physical consumer forbids model.predict')
            model.predict = forbid
            transform = base.p0.image_transform()
            def encode(path):
                with Image.open(path) as im:
                    rgb = np.asarray(im.convert('RGB')).copy()
                with torch.no_grad():
                    v = model.encode({'pixels':base.p0._image_tensor(rgb,transform=transform,device=device)})['emb'][0,0]
                if v.device.type != 'cuda' or not torch.isfinite(v).all():
                    raise ValueError('finite host CUDA endpoint encoding required')
                torch.cuda.synchronize(device)
                return v.cpu().tolist()
            return encode
        encodings = endpoints.encode_observations(rows,plans,output,setup,
            {ROOT:(np.asarray(plans[0]['goal']),np.asarray(plans[0]['q']))})
    checks = dict(six_winners_retained=True,all_full_horizon=all(r['full_horizon_completed'] for r in rows),
        all_prefix_pixels_exact=all(r['prefix_identity']['status']=='matched' for r in rows),
        traces_match=all(len(r['trace'])==r['completed_commands'] and all(e['primitive_step']==i+1 and
            e['command']==plan['commands'][i] for i,e in enumerate(r['trace'])) for plan,r in zip(plans,rows)),
        encodings_complete=not p['encode_terminals'] or len(encodings)==12 and all(e['status']=='completed' for e in encodings))
    try:
        verify_prepared(output); checks['inputs_sources_unchanged'] = True
        if evaluator_seal is not None:
            from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal
            verify_seal(output_dir=evaluator_seal['output_dir'],expected_manifest_sha256=evaluator_seal['manifest_sha256'])
        if model is not None:
            checks['model_frozen_unchanged'] = before==checked.tensor_state_hash(model) and all(not m.training for m in model.modules()) and all(not x.requires_grad for x in model.parameters())
    except Exception as exc:
        checks['inputs_sources_unchanged'] = False; failures.append(durable.failure(exc,'final_integrity'))
    passed = all(checks.values()) and not failures
    write(output/'report.json',dict(schema=TAG,status='calibration_recorded' if passed else 'integrity_fail',
        protocol_sha256=sha(output/'protocol.json'),rows=rows,encodings=encodings,gates=checks,failures=failures,host=host,
        source_seals=read(output/'prepare_receipt.json')['source_seals'],evaluator_source_seal=evaluator_seal,
        physical_env_step_attempts=sum(r['physical_env_step_attempts'] for r in rows),
        physical_env_step_returns=sum(r['physical_env_step_returns'] for r in rows),
        attempted_encode_samples=sum(e['attempted_encode_samples'] for e in encodings),
        new_predict_calls=0,P1_admission=False,efficacy_verified=False,significance_claim=False))
    if not passed:
        raise RuntimeError('physical integrity failed; all six slots retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run'))
    parser.add_argument('--planning-report',type=Path)
    parser.add_argument('--feedback-dir',type=Path,default=DEFAULT_FEEDBACK)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--encode-terminals',action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.planning_report is None:
            parser.error('prepare requires --planning-report')
        print(json.dumps(freeze(args.planning_report,args.output,args.feedback_dir,args.encode_terminals)))
    else:
        run(args.output)
