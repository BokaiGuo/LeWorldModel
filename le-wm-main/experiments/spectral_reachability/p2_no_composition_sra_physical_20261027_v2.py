"""Independent consumer of four sealed joint-H2 SRA H2 witnesses.

Prepare/validation are model-free and never instantiate a simulator. Run is
explicit main-thread physics, optional terminal encoding is HOST CUDA only.
No planning, predictions, resampling, old whitelist edits or scientific admission.
"""
import argparse
from copy import deepcopy
import importlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

from experiments.spectral_reachability import p2_no_composition_sra_20261027_v2 as sampling
from experiments.spectral_reachability import p2_full20_physical_20261027_v2 as helpers
from experiments.spectral_reachability import p3_matched_physical_20261027_v1 as durable
from experiments.spectral_reachability.d2p_composition_witness_v2 import compose, digest
from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal, verify_seal
from experiments.spectral_reachability.run_pusht_sampling_comparison_20261027_v2 import verify_rgb_context, official_checked

TAG = 'p2_no_composition_sra_physical_20261027_v2'
PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_PLANNING = PROJECT/'results/p2_no_composition_sra_20261027_v2_prepared01'
EXPECTED_PLANNER_SHA256 = '6e1402ac8abe3aabfffefd24aa016c341e6eb4f40d3f60e233152460ca70e45a'
EXPECTED_REPORT_SHA256 = '522afa5a68b23a6ff829c5c6d820d858953dccc0f778548e747ba594ff73937a'
sha = sampling.full20.sha
write = durable.write


def require(condition, message):
    if not condition:
        raise ValueError(message)


def indexed(rows, key):
    require(isinstance(rows,list) and len(rows) == 4, 'all four roots required')
    result = {r[key]:r for r in rows}
    require(len(result) == 4, 'duplicate stable root identity')
    return result


ACCOUNTING = dict(
    per_root=dict(candidate_predict_samples=80, replay_predict_samples=4,
        compilation_predict_samples=4, total_predict_samples=88,
        jacobian_calls=1, svd_calls=2, scalar_output_vjp_directions=192,
        processed_context_token_samples=264),
    four_root=dict(total_predict_samples=352, compilation_predict_samples=16,
        jacobian_calls=4, svd_calls=8, scalar_output_vjp_directions=768,
        processed_context_token_samples=1056),
    reference_per_root_predict_samples=84, additional_per_root_predict_samples=4,
    prediction_budget_matched=False, fair_total_budget_comparison=False,
    verification_work=dict(svd_calls=8,bounded_solve_calls=4,neural_predict_calls=0,
        scope='additional read-only array verification, NOT included in the 352-call calibration ledger'),
    unmeasured=deepcopy(sampling.BOUNDARY['unmeasured']))


def plans_from_saved(report, protocol, prepared, compilation, verified_actions):
    """Additional strict metadata checks after planner.selected_actions verification.

    verified_actions must come from the planner's source-bound read-only exporter.
    No neural callbacks, recomputed costs, winner changes or new proposals here.
    """
    require(report['schema'] == sampling.TAG+'.report' and report['status'] == 'calibration_completed'
        and report['protocol_sha256'] == digest(protocol) and report['source_seals'] == prepared['source_seals']
        and bool(report['gates']) and all(v is True for v in report['gates'].values()),
        'complete protocol/source-bound SRA calibration required')
    require(protocol['schedule'] == sampling.SCHEDULE and protocol['boundary'] == sampling.BOUNDARY,
            'fixed SRA schedule/boundary required')
    for key,value in sampling.BOUNDARY.items():
        require(report[key] == value, 'SRA claim boundary changed: '+key)
    source = indexed(compilation['rows'],'root_sha256')
    rows = indexed(report['rows'],'root_sha256')
    ready = indexed(protocol['context']['cases'],'root_sha256')
    require(source.keys() == rows.keys() == ready.keys(), 'root identity census mismatch')
    require({r['root_index'] for r in ready.values()} == set(range(4)), 'four unique root indices required')
    require(set(verified_actions) == {(key,sampling.ARM) for key in ready}, 'exported action census mismatch')
    plans = []
    for key,reference in sorted(ready.items(),key=lambda item:item[1]['root_index']):
        row,root = rows[key],reference['root']
        compiled = source[key]['compiled']
        initial = compiled['input_context']
        expected = dict(layout='states_with_between_state_actions_v1',context_length=3,
            latent_dim=192,action_dim=10,latent_history=initial['emb'][0],
            action_history=initial['actions'][0][1:])
        require(root['root_id'] == key and root['context'] == expected
            and row['root'] == root and row['root_index'] == reference['root_index']
            and row['action_domain'] == reference['action_domain'], 'full root/context/domain mismatch')
        require(reference['metric_sha256'] == compiled['scoring']['metric']['sha256']
            and reference['goal_vector_sha256'] == digest(compiled['goal_vectors'][0])
            and reference['projection_sha256'] == digest(compiled['scoring']['metric']['projection']),
            'original goal0/Q identity mismatch')
        result = row['result']
        require(row['complete'] is True and row['error'] is None and result['arm'] == sampling.ARM
            and result['fixture_only'] is False and result['candidate_and_replay_complete'] is True
            and result['status'] == 'search_complete' and result['replay_error'] is None
            and result['planning_predict_samples'] == 80 and result['replay_predict_samples'] == 4
            and result['selection_evaluations'] == result['attempted_candidate_evaluations'] == 40,
            'complete real forty-candidate SRA required')
        for name,value in sampling.BOUNDARY.items():
            require(row[name] == result[name] == value, 'row/result claim boundary changed')
        entries,actual = row['actual_model_entries'],row['actual_prediction_budget']
        require(sampling.accounting_ok(entries,actual,result) and len(result['ledger']) == 84
            and [e['phase'] for e in result['ledger']] == ['planning']*80+['replay']*4
            and all(e['kind'] in ('predict','action_encoder') and e['scope']['arm'] == sampling.ARM
                    for e in entries), 'actual88 dispatch/compilation accounting mismatch')
        provenance = sampling.provenance_for(protocol,prepared,reference)
        require(actual['binding'] == dict(protocol_id=provenance['source_sha256'],task_id=sampling.TAG,
            root_id=key,frozen_model_id=provenance['checkpoint_sha256'])
            and actual['method'] == sampling.ARM and actual['predict_sample_limit'] == 88
            and actual['attempted_predict_samples'] == 88 and actual['blocked_calls'] == 0
            and all(p['completed'] is True and p['failed_predict_calls'] == 0 for p in actual['phases']),
            'actual88 budget/root binding mismatch')
        for kind,field,token_field in (
                ('predict','predict_calls','processed_context_token_samples'),
                ('action_encoder','action_encoder_calls','action_encoder_processed_context_token_samples')):
            events = [e for e in entries if e['kind'] == kind]
            require(len(events) == row[field] == 88 and sum(e['samples'] for e in events) == 88
                and sum(e['context_tokens'] for e in events) == row[token_field] == 264,
                'actual encoder/predict work mismatch')
        ops = [e for e in row['compilation_operations'] if e['completed']]
        receipt_compile = row['compile_receipt']
        require([e['kind'] for e in ops] == ['jacobian','bounded_ridge_svd','mode_svd']
            and receipt_compile['completed_jacobian_calls'] == 1
            and receipt_compile['completed_svd_calls'] == 2
            and receipt_compile['completed_scalar_output_vjp_directions'] == 192,
            'Jacobian/SVD overhead missing')
        actions = np.asarray(verified_actions[key,sampling.ARM])
        witness = result['witness']
        receipt = witness['receipt']
        require(actions.shape == (10,2) and np.isfinite(actions).all() and np.all(np.abs(actions)<=1)
            and receipt['actions'] == actions.tolist()
            and witness['independent_full_replay_exact'] is True and len(witness['edges']) == 1,
            'exported bounded H2 witness mismatch')
        rebound = compose(root=root,provenance=receipt['provenance'],action_domain=reference['action_domain'],
            edges=witness['edges'],replay=lambda c,a:deepcopy(receipt['terminal_context']),max_edges=1,max_actions=10)
        require(rebound == receipt, 'SRA receipt changed')
        require(abs(sampling.full20.metric_for(compiled)(receipt['terminal_context'])-
                    result['selected_model_cost']) <= 1e-12, 'original Q terminal cost differs')
        plan = dict(root_index=reference['root_index'],root_sha256=key,method=sampling.ARM,
            commands=actions.tolist(),planned_horizon_tokens=2,planned_commands=10,corrected_search_horizon=2,
            root=deepcopy(root),action_domain=deepcopy(reference['action_domain']),
            selected_candidate_id=result['selected_candidate_id'],receipt_sha256=receipt['receipt_sha256'],
            planning_accounting=deepcopy(ACCOUNTING['per_root']))
        plan['selection_sha256'] = digest(plan)
        plans.append(plan)
    for name,value in dict(actual_predict_calls=352,actual_action_encoder_calls=352,
            actual_action_encoder_processed_context_token_samples=1056,actual_selection_evaluations=160,
            actual_processed_context_token_samples=1056,completed_jacobian_calls=4,
            completed_svd_calls=8,completed_scalar_output_vjp_directions=768).items():
        require(report[name] == value, 'four-root work census mismatch: '+name)
    return plans


def inputs(planning):
    from experiments.spectral_reachability import audit_pusht_d2p_goal_agent_intervention as base
    from experiments.spectral_reachability import run_pusht_d2p_budget_guard_cuda as checked
    planning = Path(planning).resolve()
    require((planning/'report.json').is_file(), 'SRA CUDA report missing; wait for main-thread completion')
    path,protocol,prepared = sampling.load_prepared(planning)
    require(sha(path/'report.json') == EXPECTED_REPORT_SHA256, 'exact main-thread SRA report required')
    verify_archived_exporter(prepared)
    exported = sampling.selected_actions(path)
    source = json.loads(Path(protocol['context']['compilation']).read_text())
    require((path/'report.json').is_file(), 'SRA CUDA report missing; no physical plans may be prepared')
    report = json.loads((path/'report.json').read_text())
    plans = plans_from_saved(report,protocol,prepared,source,exported['actions'])
    bindings = deepcopy(protocol['bindings'])
    for filename in ('protocol.json','prepared.json','report.json'):
        bindings[str(path/filename)] = sha(path/filename)
    require(exported['report_sha256'] == bindings[str(path/'report.json')], 'report changed during export')
    require(exported['verification_work'] == ACCOUNTING['verification_work'], 'verification work scope changed')
    for case in protocol['context']['cases']:
        bank = path/f'root_{case["root_index"]}_bank.json'
        bindings[str(bank)] = sha(bank)
    reports = [json.loads(Path(p).read_text()) for p in protocol['bindings'] if Path(p).name == 'report.json']
    interventions = [r for r in reports if 'bank_path' in r.get('protocol',{})]
    require(len(interventions) == 1, 'unique physical root provenance required')
    upstream_protocol = interventions[0]['protocol']
    bank_path = Path(upstream_protocol['bank_path']).resolve()
    require(sha(bank_path) == upstream_protocol['bank_sha256'], 'physical bank SHA mismatch')
    bank = json.loads(bank_path.read_text())
    upstream_path = Path(bank['upstream_path']).resolve()
    require(sha(upstream_path) == upstream_protocol['upstream_sha256'], 'physical RGB provenance SHA mismatch')
    upstream = base.physical._load_preflight(upstream_path)
    refs = indexed(upstream['rows'],'root_sha256')
    require(set(refs) == {p['root_sha256'] for p in plans}, 'physical reference census mismatch')
    for plan in plans:
        reference = refs[plan['root_sha256']]
        require(plan['root']['root_rgb_sha256'] == reference['history_rgb_sha256']
            and plan['root']['goal_rgb_sha256'] == reference['goal_rgb_sha256'], 'physical root/goal RGB mismatch')
    for p in (bank_path,upstream_path): bindings[str(p)] = sha(p)
    _,_,config = checked.validate_inputs(protocol['context']['compilation'],protocol['context']['policy'])
    return base,checked,source,config,bindings,plans,upstream,refs,protocol


def verify_archived_exporter(prepared):
    """No retroactive exporter: require it in the original verified source seal."""
    source = str(Path(sampling.__file__).resolve())
    matches = []
    for seal in prepared['source_seals']:
        directory = Path(seal['output_dir'])
        manifest = json.loads((directory/'manifest.json').read_text())
        for entry in manifest['files']:
            if entry['source'] == source:
                archive = directory/entry['archive']
                require(entry['sha256'] == sha(archive) == sha(source) == EXPECTED_PLANNER_SHA256,
                        'original sealed SRA planner bytes required; no retrospective reseal')
                require('\ndef selected_actions(output):' in archive.read_text(),
                        'exporter absent from original seal; implement independent consumer verification')
                matches.append(dict(source=source,archive=str(archive),sha256=entry['sha256'],
                                    manifest_sha256=seal['manifest_sha256']))
    require(len(matches) == 1, 'unique pre-execution sealed exporter required')
    return matches[0]


def environment_creator(base):
    """Resolve installed class source without gym.make/reset/step/render."""
    entry = base.gym.spec('swm/PushT-v1').entry_point
    if callable(entry): return entry
    module, name = entry.split(':')
    return getattr(importlib.import_module(module),name)


def prepare(planning,output):
    output = Path(output).resolve()
    require(TAG in output.name, 'exclusive tagged physical prepare directory required')
    if output.exists(): raise FileExistsError(output)
    base,checked,source,config,bindings,plans,upstream,refs,protocol = inputs(planning)
    creator = environment_creator(base)
    files = checked.sources_before_allocate(source,config)
    files += [Path(module.__file__) for name,module in tuple(sys.modules.items())
        if name.startswith('experiments.') and getattr(module,'__file__','').endswith('.py')]
    environment_sources = sorted({str(Path(inspect.getfile(obj)).resolve()) for obj in
        (creator,*[getattr(creator,name) for name in ('eval_state','reset','render','step')])})
    files += [Path(p) for p in environment_sources]
    files += [Path(__file__),PROJECT/'tests'/f'test_{TAG}.py',PROJECT/'innerdocs'/f'{TAG}.md']
    files = sorted({p.resolve() for p in files})
    for path in environment_sources: bindings[path] = sha(path)
    value = dict(schema=TAG+'.protocol',planning=str(Path(planning).resolve()),bindings=bindings,plans=plans,
        sealed_exporter_sha256=EXPECTED_PLANNER_SHA256,planning_report_sha256=EXPECTED_REPORT_SHA256,
        environment_source_paths=environment_sources,source_paths=[str(p) for p in files],
        physical_repetitions=1,planned_commands_per_branch=10,
        total_planned_commands=40,actual_planning_predict_samples_verified=352,
        physical_before_optional_encoding=True,root_selection_changed=False,extra_search_or_training=False,
        P1_admission=False,P2_admission=False,planning_accounting=deepcopy(ACCOUNTING),
        claim_boundary=deepcopy(sampling.BOUNDARY),prediction_budget_matched=False,
        fair_total_budget_comparison=False,scope='four known calibration roots, joint-H2 SRA selected paths only; reset/warmup excluded from selected-path counters')
    output.mkdir(exist_ok=False)
    write(output/'protocol.json',value)
    seals = [create_seal(files=files[i:i+40],args_json=json.dumps(dict(protocol_sha256=digest(value))),
                        output_dir=output/f'source_seal_{i//40}') for i in range(0,len(files),40)]
    write(output/'prepared.json',dict(schema=TAG+'.prepared',protocol_sha256=digest(value),source_seals=seals,
        status='physical_consumer_prepared_not_executed',new_neural_calls=0,new_physical_steps=0))
    load_prepared(output)
    print(json.dumps(dict(status='physical_consumer_prepared_not_executed',output=str(output),
        protocol_sha256=digest(value),branches=4,planned_commands=40,actual_planning_samples_verified=352)))


def load_prepared(output):
    output = Path(output).resolve()
    protocol = json.loads((output/'protocol.json').read_text())
    prepared = json.loads((output/'prepared.json').read_text())
    require(protocol['schema'] == TAG+'.protocol' and prepared['schema'] == TAG+'.prepared'
            and digest(protocol) == prepared['protocol_sha256']
            and protocol['planning_accounting'] == ACCOUNTING
            and protocol['claim_boundary'] == sampling.BOUNDARY
            and protocol['P2_admission'] is False and protocol['prediction_budget_matched'] is False
            and protocol['fair_total_budget_comparison'] is False,
            'physical prepared protocol digest mismatch')
    paths = []
    for seal in prepared['source_seals']:
        verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
        directory = Path(seal['output_dir'])
        require(json.loads((directory/'args.json').read_text()) == dict(protocol_sha256=digest(protocol)),
                'physical source/protocol seal binding mismatch')
        paths += [e['source'] for e in json.loads((directory/'manifest.json').read_text())['files']]
    required = {str(Path(__file__).resolve()),str(PROJECT/'tests'/f'test_{TAG}.py'),
                str(PROJECT/'innerdocs'/f'{TAG}.md'),str(Path(sampling.__file__).resolve()),
                str(Path(helpers.__file__).resolve()),str(Path(durable.__file__).resolve())}
    require(sorted(paths) == protocol['source_paths'] and required <= set(paths), 'physical source closure missing')
    for path,expected in protocol['bindings'].items():
        require(sha(path) == expected, 'physical bound input/source changed: '+path)
    context = inputs(protocol['planning'])
    require(context[5] == protocol['plans'], 'saved physical plans differ from SRA witnesses')
    require(all(protocol['bindings'].get(p) == h for p,h in context[4].items()),
            'physical input binding closure missing')
    return output,protocol,prepared,context


def physical_branch(*,plan,output,reset,env,state_of,evaluate):
    require(plan['method'] == sampling.ARM, 'joint-H2 SRA arm only')
    require(digest({k:v for k,v in plan.items() if k != 'selection_sha256'}) == plan['selection_sha256'],
            'physical plan selection digest mismatch')
    return helpers.physical_branch(plan=plan,output=output,reset=reset,env=env,state_of=state_of,evaluate=evaluate)


def run(prepared,output,encode_terminals=False):
    _,protocol,sealed,context = load_prepared(prepared)
    base,checked,source,config,bindings,plans,upstream,refs,planning_protocol = context
    output = Path(output).resolve()
    require(TAG in output.name, 'new tagged physical execution output required')
    output.mkdir(exist_ok=False)
    write(output/'protocol.json',dict(prepared=str(Path(prepared).resolve()),
        prepared_protocol_sha256=sealed['protocol_sha256'],plans=plans,encode_terminals=encode_terminals))
    rows,failures,encodings = [],[],[]
    env = model = before = host = None
    try:
        env = base.gym.make('swm/PushT-v1',render_mode='rgb_array')
        require(str(Path(inspect.getfile(type(env.unwrapped))).resolve()) in protocol['environment_source_paths'],
                'physical environment class source not prepared')
        require(tuple(env.unwrapped.variation_space['agent']['color'].value.tolist()) == (65,105,225),
                'unsupported renderer color')
        roots = base.collect_nonterminal_boundary_contact_roots(env,count=4,seed=int(upstream['protocol']['root_seed']),oversample_factor=4)
        require(len(roots) == 4, 'all four physical roots required')
        roots = {base.root_fingerprint(root):root for root in roots}
        require(set(roots) == set(refs), 'physical roots differ; no replacement')
        for plan in plans:
            root = roots[plan['root_sha256']]
            def reset():
                require(base.root_fingerprint(root) == plan['root_sha256'], 'physical root changed')
                frames,history,goal = base.physical._reset_and_warmup(env,root,history_len=3)
                identity = verify_rgb_context(frames,goal,refs[plan['root_sha256']])
                identity.update(history=np.asarray(history).tolist(),state=base.p0.packets.state_of(env).tolist(),
                                goal=np.asarray(root['goal']).tolist())
                return identity
            def evaluate(state):
                return dict(official=official_checked(env,state,root['goal']),pose=base.physical._pose_label(state,root['goal']))
            rows.append(physical_branch(plan=plan,output=output,reset=reset,env=env,
                state_of=base.p0.packets.state_of,evaluate=evaluate))
    except (Exception,KeyboardInterrupt) as exc:
        failures.append(durable.failure(exc,'physical_setup_or_branch'))
    finally:
        if env is not None:
            try: env.close()
            except Exception as exc: failures.append(durable.failure(exc,'env_close'))
    helpers.materialize_missing(plans,rows,output)
    write(output/'physical_report.json',dict(schema=TAG+'.physical_before_encoding',rows=rows,
        planning_accounting=deepcopy(ACCOUNTING),
        failures=deepcopy(failures),new_predict_calls=0,P1_admission=False,P2_admission=False))
    if encode_terminals:
        def setup():
            nonlocal model,before,host
            import torch
            from PIL import Image
            host = base.require_cuda_allocation('cuda')
            device = torch.device('cuda:0')
            model = base.load_compatible_pretrained(Path(planning_protocol['context']['policy']),device=device).eval().requires_grad_(False)
            before = checked.tensor_state_hash(model)
            def forbid(*a,**kw): raise RuntimeError('terminal encoding forbids predictions')
            model.predict = forbid
            transform = base.p0.image_transform()
            def encode(path):
                with Image.open(path) as image: rgb = np.asarray(image.convert('RGB')).copy()
                with torch.no_grad():
                    vector = model.encode({'pixels':base.p0._image_tensor(rgb,transform=transform,device=device)})['emb'][0,0]
                require(vector.device.type == 'cuda' and torch.isfinite(vector).all(), 'finite HOST CUDA vector required')
                torch.cuda.synchronize(device)
                return vector.cpu().tolist()
            return encode
        encodings = helpers.optional_encodings(rows,output,setup)
    gates = dict(all_four_retained=len(rows) == 4 and len({r['root_sha256'] for r in rows}) == 4,
        all_fixed_H2=all(r['full_horizon_completed'] and r['completed_commands'] == 10 for r in rows),
        trace_matches_selected=all(len(r['trace']) == r['completed_commands'] and
            [e['command'] for e in r['trace']] == next(p['commands'] for p in plans if p['root_sha256'] == r['root_sha256'])[:r['completed_commands']] for r in rows),
        no_branch_errors=all(not r.get('failures') for r in rows),
        physical_persisted=(output/'physical_report.json').is_file(),
        encodings_complete=not encode_terminals or len(encodings) == 4 and all(e['status'] == 'completed' for e in encodings))
    try:
        load_prepared(prepared)
        gates['sources_and_inputs_unchanged'] = True
    except Exception as exc:
        gates['sources_and_inputs_unchanged'] = False
        failures.append(durable.failure(exc,'final_source_check'))
    if model is not None:
        gates.update(model_unchanged=before == checked.tensor_state_hash(model),
            frozen_eval=all(not m.training for m in model.modules()) and all(not p.requires_grad for p in model.parameters()))
    passed = all(gates.values()) and not failures
    write(output/'report.json',dict(schema=TAG+'.report',status='calibration_recorded' if passed else 'integrity_fail',
        prepared_protocol_sha256=sealed['protocol_sha256'],rows=rows,encodings=encodings,failures=failures,gates=gates,
        source_seals=sealed['source_seals'],host=host,new_predict_calls=0,
        selected_path_env_step_attempts=sum(r['physical_env_step_attempts'] for r in rows),
        selected_path_env_step_returns=sum(r['physical_env_step_returns'] for r in rows),
        attempted_terminal_encode_samples=sum(e['attempted_encode_samples'] for e in encodings),
        P1_admission=False,P2_admission=False,physical_efficacy_verified=False,
        prediction_budget_matched=False,planning_accounting=deepcopy(ACCOUNTING),
        fair_total_budget_comparison=False,actual_guard_endpoints_may_differ=True))
    if not passed: raise RuntimeError('SRA physical incomplete; all branches retained; no retries/reselection')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--planning',type=Path,default=DEFAULT_PLANNING)
    prep.add_argument('--output',type=Path,required=True)
    execute = commands.add_parser('run')
    execute.add_argument('--prepared',type=Path,required=True)
    execute.add_argument('--output',type=Path,required=True)
    execute.add_argument('--encode-terminals',action='store_true')
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    prepare(**args) if command == 'prepare' else run(**args)


if __name__ == '__main__':
    main()

