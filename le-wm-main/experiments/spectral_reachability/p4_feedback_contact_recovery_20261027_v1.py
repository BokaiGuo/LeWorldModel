"""Saved six-branch feedback recovery arithmetic; no simulator/neural execution."""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability import p4_saved_contact_geometry_20261027_v1 as geometry
from experiments.spectral_reachability import pusht_feedback_suffix_physical_20261027_v1 as physical

saved = geometry.saved
TAG = 'p4_feedback_contact_recovery_20261027_v1'
REPO = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO/'results/pusht_feedback_suffix_physical_20261027_v1_prepare_v1/report.json'
POSE_SOURCE = REPO/'experiments/spectral_reachability/run_pusht_d2p_p1_physical_calibration.py'
POSE_FUNCTION_SHA = '916d0979838e502720940aca24f8dc717ce256f0db2e3ffdeccb457f3912e1de'
require = saved.require


def pose(state, goal):
    """Unchanged original evaluator arithmetic, not a newly fitted task metric."""
    state, goal = saved.array(state,(7,)), saved.array(goal,(7,))
    xy = float(np.linalg.norm(state[2:4]-goal[2:4]))
    angle = float(abs(np.arctan2(np.sin(state[4]-goal[4]),np.cos(state[4]-goal[4]))))
    return dict(xy_distance=xy,angle_distance=angle,pose_cost=xy+20.0*angle)


def close(a,b,message):
    require(np.isfinite([a,b]).all() and abs(a-b) <= 1e-10,message)


def prepared(inputs, folder):
    protocol = inputs.json(folder/'protocol.json')
    receipt = inputs.json(folder/'prepare_receipt.json')
    digest = inputs.hash(folder/'protocol.json')
    require(receipt['status'] == 'prepared_not_run' and receipt['protocol_sha256'] == digest,
            'prepared protocol binding differs')
    require(receipt['source_seals'], 'source seals required')
    for seal in receipt['source_seals']:
        inputs.seal(seal)
        require(inputs.json(Path(seal['output_dir'])/'args.json')['protocol_sha256'] == digest,
                'seal/protocol binding differs')
    require(protocol['bindings'], 'input bindings required')
    for path, expected in protocol['bindings'].items():
        require(inputs.hash(path) == expected, 'bound input changed: '+path)
    return protocol,receipt


def verify_inputs(inputs, folder):
    """All source/compiler/context/action bindings, no runtime prepare functions."""
    p, receipt = prepared(inputs,folder)
    require(p['schema'] == physical.TAG, 'physical schema differs')
    compiler_path = Path(p['planning_report'])
    cp, cr = prepared(inputs,compiler_path.parent)
    compiler_report = inputs.json(compiler_path)
    fd = Path(p['feedback_dir'])
    fp, fr = prepared(inputs,fd)
    feedback_report = inputs.json(fd/'report.json')
    require(Path(cp['source']).resolve() == fd.resolve() and
        cp['original_initial_context'] == fp['initial_context'] and cp['original_root_rgb'] == fp['root_rgb'],
        'compiler original source/context/RGB differs')
    require(compiler_report['source_seals'] == cr['source_seals'] and
        compiler_report['protocol_sha256'] == cr['protocol_sha256'], 'compiler source/protocol differs')
    for path in (compiler_path,compiler_path.parent/'protocol.json',compiler_path.parent/'prepare_receipt.json',
                 fd/'protocol.json',fd/'report.json',fd/'prepare_receipt.json'):
        require(p['bindings'].get(str(path)) == inputs.hash(path), 'physical must bind compiler/feedback receipts')
    # Calls ONLY existing pure saved-bank mathematical verification. Never prepare/run.
    plans = physical.plans_from_saved(compiler_report,fp,feedback_report)
    require(plans == p['plans'], 'physical actions differ from sealed compiler winners')
    require([{k:s[k] for k in ('root_sha256','root_index','role','arm','prefix','context','goal','q')}
        for s in compiler_report['plans']] == cp['cases'], 'compiler plans/cases differ')
    for role in physical.ROLES:
        value = inputs.json(fd/f'{role}.contexts.json')
        require(value == next(c for c in feedback_report['contexts'] if c['role'] == role), 'feedback context file differs')
    cap_path = Path(fp['capability'])
    require(fp['bindings'].get(str(cap_path)) == inputs.hash(cap_path), 'original capability binding differs')
    cap = inputs.json(cap_path)
    for seal in cap['source_seals']:
        inputs.seal(seal)
    originals = {role:next(r for r in cap['rows'] if r['method'] == role+'_zero'
        and r['root_sha256'] == physical.ROOT) for role in physical.ROLES}
    for role,row in originals.items():
        require(inputs.json(cap_path.parent/f'root_1_{role}_zero_replay0.physical.json') == row,
            'original capability branch differs')
        require(inputs.hash(row['journal_path']) == row['journal_sha256'], 'original capability journal differs')
    goals = [row['context_identity']['goal'] for row in originals.values()]
    require(all(g == goals[0] for g in goals), 'original physical goals differ')
    return p,receipt,compiler_report,originals


def image(inputs, event):
    require(inputs.hash(event['rgb_path']) == event['rgb_file_sha256'], 'RGB file binding differs')
    with Image.open(io.BytesIO(inputs.raw(event['rgb_path']))) as im:
        value = np.asarray(im).copy()
    require(value.shape == (224,224,3) and value.dtype == np.uint8, 'exact uint8 RGB required')
    return value


def summarize_steps(steps):
    by_step = {s['primitive_step']:s for s in steps}
    prefix = by_step.get(5)
    suffix = [s for s in steps if s['primitive_step'] > 5]
    entries = [s['primitive_step'] for s in suffix if not by_step[s['primitive_step']-1]['near_contact'] and s['near_contact']]
    return dict(prefix_near_contact=None if prefix is None else prefix['near_contact'],
        prefix_clearance=None if prefix is None else prefix['signed_surface_clearance'],
        suffix_observed_steps=len(suffix),suffix_full_step10_observed=10 in by_step,
        suffix_near_contact_steps=sum(s['near_contact'] for s in suffix),
        suffix_min_clearance=min((s['signed_surface_clearance'] for s in suffix),default=None),
        suffix_max_clearance=max((s['signed_surface_clearance'] for s in suffix),default=None),
        suffix_near_contact_entry_steps=entries,
        suffix_geometric_reentry_observed=None if prefix is None else bool(entries),
        suffix_first_near_contact_step=next((s['primitive_step'] for s in suffix if s['near_contact']),None),
        suffix_max_block_vertex_step=max((s['block_max_vertex_displacement'] for s in suffix),default=None),
        pose_at_prefix=None if prefix is None else prefix['pose'],
        pose_at_exact_terminal=None if 10 not in by_step else by_step[10]['pose'],
        exact_terminal_minus_prefix_pose=None if prefix is None or 10 not in by_step else
            by_step[10]['pose']['pose_cost']-prefix['pose']['pose_cost'],
        force_contact=None,causal_push=None,
        interpretation='snapshot geometric entries only; no intervening substep-contact guarantee or force causality')


def analyze_branch(inputs, row, plan, original, g):
    """Retain failed/partial slots; missing step10 is never an early terminal."""
    for key in ('root_sha256','root_index','role','arm','method','selection_sha256'):
        require(row[key] == plan[key], 'branch identity differs: '+key)
    n, attempts = row['completed_commands'],row['attempted_commands']
    saved.integer(n); saved.integer(attempts)
    require(0 <= n <= attempts <= 10 and attempts-n <= 1, 'attempt/return counts differ')
    require(row['physical_env_step_attempts'] == attempts and row['physical_env_step_returns'] == n,
            'physical counts differ')
    if row['status'] == 'completed':
        require(n == attempts == 10 and row['full_horizon_completed'] is True, 'false full completion')
    require(inputs.hash(row['journal_path']) == row['journal_sha256'], 'journal binding differs')
    events = [saved.strict_json(v) for v in inputs.raw(row['journal_path']).splitlines()]
    initial = [e for e in events if e['kind'] == 'initial']
    observations = [e for e in events if e['kind'] == 'observation']
    dispatches = [e for e in events if e['kind'] == 'attempt']
    returns = [e for e in events if e['kind'] == 'step_returned']
    require([e['primitive_step'] for e in dispatches] == list(range(1,attempts+1)) and
        [e['primitive_step'] for e in returns] == list(range(1,n+1)), 'journal dispatch/return census differs')
    require([e['primitive_step'] for e in observations] == list(range(1,len(observations)+1)) and
        len(observations) <= n and len(row['trace']) == len(observations), 'observed step census differs')
    require(not row['status'] == 'completed' or len(observations) == 10, 'completed branch missing observations')
    for e in dispatches:
        require(e['command'] == plan['commands'][e['primitive_step']-1], 'dispatched action differs')
    if not initial:
        require(not observations and not attempts and row['status'] != 'completed', 'missing initial snapshot')
        return dict(method=plan['method'],role=plan['role'],arm=plan['arm'],steps=[],
            observed_status=row['status'],summary=summarize_steps([]),physical_complete=False)
    require(len(initial) == 1 and events[0] == initial[0], 'initial journal identity differs')
    require(initial[0]['state'] == original['context_identity']['state'], 'original initial physical state differs')
    require(row['context_identity'] == {k:v for k,v in original['context_identity'].items() if k != 'state'},
            'root RGB/history/physical goal differs')
    goal = row['context_identity']['goal']
    previous = None
    steps = []
    for e in initial+observations:
        step = e.get('primitive_step',0)
        if step:
            require({k:v for k,v in e.items() if k not in ('rgb_path','rgb_file_sha256')} == row['trace'][step-1],
                'observation/branch trace differs')
            require(e['command'] == plan['commands'][step-1], 'observed action differs')
            require(events.index(dispatches[step-1]) < events.index(returns[step-1]) < events.index(e), 'journal event order differs')
            if step <= 5:
                require(e['state'] == original['trace'][step-1]['state'] and e['command'] == original['trace'][step-1]['command'],
                    'original prefix state/action replay differs')
        rgb = image(inputs,e)
        item = geometry.label(e['state'],previous,g)
        item.update(primitive_step=step,pose=pose(e['state'],goal),rgb_path=e['rgb_path'],
                    rgb_file_sha256=e['rgb_file_sha256'],command=e.get('command'))
        steps.append(item); previous = e['state']
        if step == 5:
            require(inputs.hash(plan['expected_prefix_rgb']) == plan['expected_prefix_rgb_sha256'], 'expected prefix file differs')
            with Image.open(io.BytesIO(inputs.raw(plan['expected_prefix_rgb']))) as im:
                expected = np.asarray(im).copy()
            matched = np.array_equal(rgb,expected)
            identity = dict(status='matched' if matched else 'mismatch',primitive_step=5,
                expected_pixel_sha256=saved.rgb_sha(expected),observed_pixel_sha256=saved.rgb_sha(rgb),pixels_exact=matched)
            require(row['prefix_identity'] == identity, 'pixel gate receipt differs')
            gates = [v for v in events if v['kind'] == 'prefix_identity']
            require(gates == [dict(kind='prefix_identity',**identity)] and events.index(e) < events.index(gates[0]), 'prefix gate journal differs')
            if len(dispatches) > 5:
                require(matched and events.index(gates[0]) < events.index(dispatches[5]), 'suffix dispatched before matched pixel gate')
    require(row['final_state'] == (observations[-1]['state'] if observations else initial[0]['state']), 'final observed state differs')
    if 'pose' in row:
        for k,v in pose(row['final_state'],goal).items():
            close(row['pose'][k],v,'saved original pose arithmetic differs')
    last = (observations or initial)[-1]
    require(row['terminal_rgb_path'] == last['rgb_path'] and row['terminal_rgb_sha256'] == last['rgb_file_sha256'], 'last observed RGB differs')
    summary = summarize_steps(steps)
    baseline = pose(original['final_state'],goal)
    summary.update(original_capability_zero_suffix_pose=baseline,
        exact_terminal_minus_original_zero_suffix_pose=None if 10 not in {s['primitive_step'] for s in steps}
            else steps[-1]['pose']['pose_cost']-baseline['pose_cost'])
    return dict(method=plan['method'],role=plan['role'],arm=plan['arm'],root_sha256=plan['root_sha256'],
        selected_candidate_id=plan['selected_candidate_id'],observed_status=row['status'],
        physical_complete=row['status'] == 'completed' and n == 10,steps=steps,summary=summary)


def encoding_costs(inputs, folder, encodings, row, plan):
    result = {}
    q,goal = saved.array(plan['q'],(192,192)),saved.array(plan['goal'],(192,))
    for kind,step in (('prefix',5),('terminal',10)):
        values = [e for e in encodings if e['method'] == plan['method'] and e['kind'] == kind]
        require(len(values) == 1, 'exact encoding slot required')
        e = values[0]
        require(inputs.json(folder/f'root_1_{plan["method"]}_{kind}.encoding.json') == e, 'encoding file differs')
        require(e['root_sha256'] == physical.ROOT, 'encoding root differs')
        saved.integer(e['attempted_encode_samples'])
        require(e['attempted_encode_samples'] <= 1, 'encoding sample count differs')
        record = dict(status=e['status'],observed_Q_goal_cost=None)
        pred = saved.array(plan['saved_candidate']['predicted_'+kind],(192,))
        record['predicted_Q_goal_cost'] = float(np.mean((q@(goal-pred))**2))
        if e['status'] == 'completed':
            require(e['attempted_encode_samples'] == 1 and row['completed_commands'] >= step,
                'encoding wrong root or shortened endpoint')
            vector = saved.array(e['vector'],(192,))
            obs = next(v for v in row['trace'] if v['primitive_step'] == step)
            # Per-image path is checked against the journal by analyze_branch.
            expected_path = folder/f'root_1_{plan["method"]}_replay0_rgb_{step}.png'
            require(Path(e['rgb_path']) == expected_path and inputs.hash(expected_path) == e['rgb_sha256'], 'encoded endpoint RGB differs')
            require(obs['primitive_step'] == step and e['comparable'] is True, 'non-comparable encoding')
            observed = float(np.mean((q@(goal-vector))**2))
            close(e['observed_Q_goal_cost'],observed,'observed Q cost differs')
            close(e['prediction_mse'],float(np.mean((vector-pred)**2)),'prediction MSE differs')
            close(e['projected_prediction_mse'],float(np.mean((q@(vector-pred))**2)),'projected MSE differs')
            record['observed_Q_goal_cost'] = observed
        result[kind] = record
    return result


def audit(report_path):
    inputs = saved.Inputs()
    g = geometry.contract(inputs)  # Fixed before any new outcomes are read.
    prior = inputs.json(REPO/'results/p4_saved_contact_geometry_20261027_v1_main_v1/report.json')
    inputs.seal(prior['audit_source_seal'])
    require(prior['geometry'] == g, 'old fixed geometry contract changed')
    inputs.hash(Path(geometry.__file__)); inputs.hash(POSE_SOURCE)
    import pymunk._chipmunk as chipmunk
    inputs.hash(chipmunk.__file__)
    # Bind original metric source explicitly without importing gym/model modules.
    source = inputs.raw(POSE_SOURCE).decode()
    fn = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == '_pose_label')
    pose_source_sha = hashlib.sha256(ast.get_source_segment(source,fn).encode()).hexdigest()
    require(pose_source_sha == POSE_FUNCTION_SHA, 'original pose definition changed')
    folder = Path(report_path).resolve().parent
    p,receipt,compiler_report,originals = verify_inputs(inputs,folder)
    report = inputs.json(report_path)
    require(report['schema'] == physical.TAG and report['protocol_sha256'] == receipt['protocol_sha256']
        and report['source_seals'] == receipt['source_seals'], 'physical report provenance differs')
    require(report['new_predict_calls'] == 0 and report['P1_admission'] is False and
        report['efficacy_verified'] is False and report['significance_claim'] is False, 'physical scientific/runtime boundary differs')
    if report['evaluator_source_seal'] is not None:
        inputs.seal(report['evaluator_source_seal'])
    physical.census(p['plans'],report['rows'])
    require(inputs.json(folder/'physical_report.json')['rows'] == report['rows'], 'durable six-branch report differs')
    outputs = []
    for plan in p['plans']:
        row = next(r for r in report['rows'] if r['method'] == plan['method'])
        require(inputs.json(folder/f'root_1_{plan["method"]}_replay0.physical.json') == row, 'durable branch differs')
        value = analyze_branch(inputs,row,plan,originals[plan['role']],g)
        value['saved_neural_arithmetic'] = encoding_costs(inputs,folder,report['encodings'],row,plan) if p['encode_terminals'] else None
        outputs.append(value)
    if p['encode_terminals']:
        require(len(report['encodings']) == 12, 'twelve retained endpoint encoding slots required')
        require(report['attempted_encode_samples'] == sum(e['attempted_encode_samples'] for e in report['encodings']),
            'total historical encode samples differ')
    require(report['physical_env_step_attempts'] == sum(r['attempted_commands'] for r in report['rows']) and
        report['physical_env_step_returns'] == sum(r['completed_commands'] for r in report['rows']), 'total physical counts differ')
    pairs = []
    for role in physical.ROLES:
        a,b = [next(r for r in outputs if r['role'] == role and r['arm'] == arm) for arm in physical.ARMS]
        comparable = a['physical_complete'] and b['physical_complete']
        pairs.append(dict(role=role,full_endpoint_comparable=comparable,
            actual_minus_imagined_pose_cost=None if not comparable else
                b['summary']['pose_at_exact_terminal']['pose_cost']-a['summary']['pose_at_exact_terminal']['pose_cost'],
            imagined_summary=a['summary'],actual_summary=b['summary'],
            interpretation='one known-root descriptive pair; no independence, equivalence or force-causal claim'))
    return dict(schema=TAG,status='saved_contact_recovery_recorded',geometry=g,rows=outputs,paired_roles=pairs,
        pose_contract=dict(source=str(POSE_SOURCE),function='_pose_label',function_source_sha256=pose_source_sha,
            formula='norm(block_xy-goal_xy) + 20*abs(wrapped block_angle-goal_angle)'),
        historical_physical_status=report['status'],reported_source_gates=report['gates'],
        compiler_report_sha256=inputs.hash(p['planning_report']),input_sha256=inputs.hashes,
        all_inputs_unchanged=inputs.stable(),
        limitations=['near-contact is snapshot geometry, never force/dynamic contact',
            'saved RGB/encodings verified, not newly rendered or neurally replayed',
            'no fitted thresholds, no outcome-dependent selection; all six slots retained',
            'changed prediction or contact co-occurrence does not imply task benefit or causal pushing'])


def run(report, output):
    report,output = Path(report).resolve(),Path(output).resolve()
    if not report.is_file():
        return dict(schema=TAG,status='awaiting_physical_report',report=str(report),output_created=False)
    require(output.name.startswith(TAG), 'exclusive tagged output required')
    output.mkdir(exist_ok=False)
    from experiments.spectral_reachability import pusht_feedback_suffix_compile_20261027_v1 as compiler
    files = [Path(__file__).resolve(),REPO/f'tests/test_{TAG}.py',REPO/f'innerdocs/{TAG}.md',
        Path(geometry.__file__),Path(saved.__file__),Path(saved.verify_seal.__code__.co_filename),
        Path(physical.__file__),Path(compiler.__file__),Path(compiler.response.__file__),Path(compiler.packet.__file__),
        geometry.ENV,geometry.SITE/'pymunk/body.py',geometry.SITE/'pymunk/shapes.py',geometry.SITE/'pymunk/space.py',POSE_SOURCE]
    seal = saved.create_seal(files=files,args_json=json.dumps(dict(report=str(report),neural=False,physics=False)),
        output_dir=output/'source_seal')
    try:
        result = audit(report)
    except (OSError,ValueError,TypeError,KeyError,IndexError,StopIteration) as exc:
        result = dict(schema=TAG,status='integrity_fail',error=type(exc).__name__+': '+str(exc))
    result.update(P1_admission=False,P4_admission=False,new_neural_calls=0,new_physics_steps=0,
        audit_source_seal=seal,audit_source_verification=saved.verify_seal(
            output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256']))
    saved.write(output/'report.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=DEFAULT_REPORT)
    parser.add_argument('--output',type=Path,required=True)
    result = run(**vars(parser.parse_args()))
    print(json.dumps({k:v for k,v in result.items() if k in ('schema','status','error','output_created')},sort_keys=True))
    raise SystemExit(0 if result['status'] == 'saved_contact_recovery_recorded' else 3 if result['status'] == 'awaiting_physical_report' else 2)
